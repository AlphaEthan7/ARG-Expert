"""
ARG_Expert: training loop and checkpointing.
"""
import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, balanced_accuracy_score, cohen_kappa_score
)
from tqdm import tqdm

try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

from torch.utils.data import DataLoader

from .config import Config, logger
from .architecture import ARGTransformer
from .loss import ARGExpertLoss
from .data import ARGDataset, collate_fn, NegativeSampler


class ARGTrainer:
    """ARG_Expert trainer."""

    def __init__(self, model: ARGTransformer, config: Config):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)

        self.criterion = ARGExpertLoss(config)
        self.criterion.to(self.device)

        self.best_val_f1 = -1.0
        self.patience_counter = 0

        self.use_amp = config.use_amp and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            logger.info("Mixed precision training (AMP) enabled")

        os.makedirs(config.output_dir, exist_ok=True)

    def create_optimizer(self, stage: str) -> torch.optim.Optimizer:
        fused_available = torch.cuda.is_available()

        if stage == "warmup":
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.config.warmup_lr,
                weight_decay=self.config.warmup_weight_decay,
                fused=fused_available
            )
        else:
            param_groups = [
                {'params': self.model.embedding.esm_model.parameters(), 'lr': self.config.finetune_lr_embedding},
                {'params': self.model.embedding.learnable_embed.parameters(), 'lr': self.config.finetune_lr_other},
                {'params': self.model.embedding.projection.parameters(), 'lr': self.config.finetune_lr_other},
                {'params': self.model.transformer_branch.parameters(), 'lr': self.config.finetune_lr_other},
                {'params': self.model.fusion_module.parameters(), 'lr': self.config.finetune_lr_other},
                {'params': self.model.heads.parameters(), 'lr': self.config.finetune_lr_other},
            ]
            optimizer = AdamW(param_groups, weight_decay=self.config.finetune_weight_decay, fused=fused_available)

        if fused_available:
            logger.info("Using fused AdamW optimizer")

        return optimizer

    def create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        stage: str,
        num_epochs: int,
        num_batches: int
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        if stage == "warmup":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10 * num_batches,
                T_mult=2,
                eta_min=1e-6
            )
        else:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.config.finetune_warmup_epochs * num_batches
            )
            cosine_scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10 * num_batches,
                T_mult=2,
                eta_min=1e-6
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.config.finetune_warmup_epochs * num_batches]
            )

        return scheduler

    def train_epoch(
        self,
        dataloader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        stage_name: str
    ) -> Dict[str, float]:
        self.model.train()

        total_loss = 0.0
        total_binary_loss = 0.0
        total_multiclass_loss = 0.0
        total_aecr_loss = 0.0

        all_binary_preds_gpu = []
        all_binary_labels_gpu = []
        all_multiclass_preds_gpu = []
        all_multiclass_labels_gpu = []

        use_aecr = getattr(self.config, "use_aecr", False)

        pbar = tqdm(dataloader, desc=f"Training {stage_name}")

        for batch_idx, batch in enumerate(pbar):
            tokens = batch['tokens'].to(self.device, non_blocking=True)
            binary_labels = batch['labels'].to(self.device, non_blocking=True)
            multiclass_labels = batch['categories'].to(self.device, non_blocking=True)

            with autocast(device_type='cuda', enabled=self.use_amp):
                outputs = self.model(tokens, return_attn_weights=use_aecr)
                losses = self.criterion(outputs, binary_labels, multiclass_labels)

                binary_preds = (outputs['binary_pred'].squeeze(-1) > 0).float()
                all_binary_preds_gpu.append(binary_preds)
                all_binary_labels_gpu.append(binary_labels)

                arg_mask = binary_labels == 1
                # Exclude class 10 ("other") from train metrics
                valid_mask = arg_mask & (multiclass_labels != 10)
                if valid_mask.sum() > 0:
                    multiclass_logits = outputs['multiclass_pred'][valid_mask]
                    multiclass_logits[:, 10] = float('-inf')
                    multiclass_preds = multiclass_logits.argmax(dim=1)
                    all_multiclass_preds_gpu.append(multiclass_preds)
                    all_multiclass_labels_gpu.append(multiclass_labels[valid_mask])

            optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                self.scaler.scale(losses['total']).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            total_loss += losses['total']
            total_binary_loss += losses['binary']
            total_multiclass_loss += losses['multiclass']
            if 'aecr' in losses:
                total_aecr_loss += losses['aecr']

            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                postfix = {
                    'loss': f"{losses['total'].item():.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
                }
                if 'aecr' in losses:
                    postfix['aecr'] = f"{losses['aecr'].item():.4f}"
                pbar.set_postfix(postfix)

        num_batches = len(dataloader)
        metrics = {
            'loss': (total_loss / num_batches).item(),
            'binary_loss': (total_binary_loss / num_batches).item(),
            'multiclass_loss': (total_multiclass_loss / num_batches).item(),
        }
        if use_aecr and isinstance(total_aecr_loss, torch.Tensor):
            metrics['aecr_loss'] = (total_aecr_loss / num_batches).item()

        if all_binary_preds_gpu:
            all_binary_preds = torch.cat(all_binary_preds_gpu).cpu().numpy()
            all_binary_labels = torch.cat(all_binary_labels_gpu).cpu().numpy()
            metrics['binary_acc'] = accuracy_score(all_binary_labels, all_binary_preds)
        if all_multiclass_preds_gpu:
            all_multiclass_preds = torch.cat(all_multiclass_preds_gpu).cpu().numpy()
            all_multiclass_labels = torch.cat(all_multiclass_labels_gpu).cpu().numpy()
            metrics['multiclass_acc'] = accuracy_score(all_multiclass_labels, all_multiclass_preds)

        return metrics

    def evaluate(self, dataloader) -> Dict[str, float]:
        self.model.eval()

        all_binary_preds = []
        all_binary_probs = []
        all_binary_labels = []
        all_multiclass_preds = []
        all_multiclass_probs = []
        all_multiclass_labels = []

        with torch.inference_mode():
            for batch in tqdm(dataloader, desc="Evaluating"):
                tokens = batch['tokens'].to(self.device, non_blocking=True)
                binary_labels = batch['labels'].to(self.device, non_blocking=True)
                multiclass_labels = batch['categories'].to(self.device, non_blocking=True)

                with autocast(device_type='cuda', enabled=self.use_amp):
                    outputs = self.model(tokens)

                binary_logits = outputs['binary_pred'].squeeze(-1)
                binary_probs = torch.sigmoid(binary_logits).cpu().numpy()
                binary_preds = (binary_probs > self.config.binary_threshold).astype(int)

                all_binary_probs.extend(binary_probs)
                all_binary_preds.extend(binary_preds)
                all_binary_labels.extend(binary_labels.cpu().numpy())

                arg_mask = binary_labels == 1
                # Exclude class 10 ("other") from multiclass metrics
                valid_mask = arg_mask & (multiclass_labels != 10)
                if valid_mask.sum() > 0:
                    multiclass_logits = outputs['multiclass_pred'][valid_mask]
                    multiclass_logits[:, 10] = float('-inf')
                    multiclass_probs = torch.nn.functional.softmax(multiclass_logits, dim=1).cpu().numpy()
                    multiclass_preds = multiclass_probs.argmax(axis=1)

                    all_multiclass_probs.extend(multiclass_probs)
                    all_multiclass_preds.extend(multiclass_preds)
                    all_multiclass_labels.extend(multiclass_labels[valid_mask].cpu().numpy())

        metrics = {}

        metrics['binary_acc'] = accuracy_score(all_binary_labels, all_binary_preds)
        metrics['binary_precision'] = precision_score(all_binary_labels, all_binary_preds, zero_division=0)
        metrics['binary_recall'] = recall_score(all_binary_labels, all_binary_preds, zero_division=0)
        metrics['binary_f1'] = f1_score(all_binary_labels, all_binary_preds, zero_division=0)
        try:
            metrics['binary_auroc'] = roc_auc_score(all_binary_labels, all_binary_probs)
        except ValueError:
            metrics['binary_auroc'] = 0.0
        try:
            metrics['binary_auprc'] = average_precision_score(all_binary_labels, all_binary_probs)
        except ValueError:
            metrics['binary_auprc'] = 0.0
        metrics['binary_mcc'] = matthews_corrcoef(all_binary_labels, all_binary_preds)

        metrics['binary_specificity'] = recall_score(all_binary_labels, all_binary_preds, zero_division=0, pos_label=0)
        metrics['binary_balanced_acc'] = balanced_accuracy_score(all_binary_labels, all_binary_preds)
        metrics['binary_cohens_kappa'] = cohen_kappa_score(all_binary_labels, all_binary_preds)

        cm = confusion_matrix(all_binary_labels, all_binary_preds)
        metrics['binary_confusion_matrix'] = cm.tolist()
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['binary_npv'] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        else:
            metrics['binary_npv'] = 0.0

        if len(all_multiclass_preds) > 0:
            metrics['multiclass_acc'] = accuracy_score(all_multiclass_labels, all_multiclass_preds)
            metrics['multiclass_macro_f1'] = f1_score(
                all_multiclass_labels, all_multiclass_preds,
                average='macro', zero_division=0
            )
            metrics['multiclass_micro_f1'] = f1_score(
                all_multiclass_labels, all_multiclass_preds,
                average='micro', zero_division=0
            )

            try:
                probs_array = np.vstack(all_multiclass_probs)
                labels_array = np.array(all_multiclass_labels)
                top3_preds = np.argsort(probs_array, axis=1)[:, -3:]
                top3_correct = np.any(top3_preds == labels_array[:, None], axis=1)
                metrics['multiclass_top3_acc'] = float(top3_correct.mean())
            except Exception:
                metrics['multiclass_top3_acc'] = 0.0

        return metrics

    def train(
        self,
        train_loader,
        val_loader,
        stage_name: str,
        neg_sampler: Optional[NegativeSampler] = None,
        train_arg_seqs: Optional[List[str]] = None,
        train_arg_labels: Optional[List[int]] = None,
        train_arg_cats: Optional[List[int]] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        assert stage_name in ("warmup", "finetune")

        do_resampling = neg_sampler is not None

        # Compute class weights from training ARG distribution
        if train_arg_cats is not None and len(train_arg_cats) > 0:
            class_counts = torch.zeros(self.config.num_classes, device=self.device)
            for cat in train_arg_cats:
                if 0 <= cat < self.config.num_classes:
                    class_counts[cat] += 1
            self.criterion.update_class_weights(class_counts)
            logger.info(f"Class weights updated based on {len(train_arg_cats)} ARGs: {self.criterion.class_weights.tolist()}")

        self.best_val_f1 = -1.0
        self.patience_counter = 0

        num_epochs = {
            "warmup": self.config.warmup_epochs,
            "finetune": self.config.finetune_epochs
        }[stage_name]

        optimizer = self.create_optimizer(stage_name)
        scheduler = self.create_scheduler(optimizer, stage_name, num_epochs, len(train_loader))

        history = {
            'train_loss': [],
            'val_binary_f1': [],
            'val_multiclass_f1': []
        }

        eval_interval = 1
        patience = self.config.finetune_patience if stage_name == "finetune" else num_epochs

        for epoch in range(num_epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Stage {stage_name} - Epoch {epoch+1}/{num_epochs}")
            logger.info(f"{'='*50}")

            if do_resampling and epoch > 0:
                n_pos = len(train_arg_seqs)
                neg_seqs = neg_sampler.sample(n_pos)

                combined_seqs = list(train_arg_seqs) + neg_seqs
                combined_labels = list(train_arg_labels) + [0] * len(neg_seqs)
                combined_cats = list(train_arg_cats) + [-1] * len(neg_seqs)

                indices = list(range(len(combined_seqs)))
                random.shuffle(indices)
                combined_seqs = [combined_seqs[i] for i in indices]
                combined_labels = [combined_labels[i] for i in indices]
                combined_cats = [combined_cats[i] for i in indices]

                epoch_dataset = ARGDataset(
                    combined_seqs, combined_labels, combined_cats,
                    max_length=self.config.max_seq_length
                )
                # persistent_workers=False: each epoch rebuilds fresh workers to
                # avoid the segfault that occurs when stale processes accumulate
                # across repeated DataLoader reconstruction.
                train_loader = DataLoader(
                    epoch_dataset,
                    batch_size=batch_size or self.config.finetune_batch_size,
                    shuffle=True,
                    num_workers=self.config.num_workers,
                    collate_fn=collate_fn,
                    pin_memory=self.config.pin_memory,
                    prefetch_factor=self.config.prefetch_factor,
                    persistent_workers=False,
                )

            train_metrics = self.train_epoch(train_loader, optimizer, scheduler, stage_name)
            logger.info(f"Train loss: {train_metrics['loss']:.4f}")

            should_eval = (epoch + 1) % eval_interval == 0 or epoch == num_epochs - 1

            if should_eval:
                val_metrics = self.evaluate(val_loader)
                logger.info(f"Val metrics: {val_metrics}")
                # Free eval activations before next epoch's AECR allocation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                history['train_loss'].append(train_metrics['loss'])
                history['val_binary_f1'].append(val_metrics['binary_f1'])
                if 'multiclass_macro_f1' in val_metrics:
                    history['val_multiclass_f1'].append(val_metrics['multiclass_macro_f1'])

                current_score = val_metrics['binary_f1']
                if 'multiclass_macro_f1' in val_metrics:
                    current_score = 0.7 * val_metrics['binary_f1'] + 0.3 * val_metrics['multiclass_macro_f1']
                if current_score > self.best_val_f1:
                    self.best_val_f1 = current_score
                    self.save_checkpoint(
                        f"best_model_{stage_name}.pt",
                        save_optimizer=False
                    )
                    self.patience_counter = 0
                    logger.info(f"New best model saved! Score: {current_score:.4f} (binary_f1={val_metrics['binary_f1']:.4f}, macro_f1={val_metrics.get('multiclass_macro_f1', 0):.4f})")
                else:
                    self.patience_counter += 1

                if stage_name == "finetune" and self.patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            else:
                history['train_loss'].append(train_metrics['loss'])
                logger.info(f"Skipping validation (eval every {eval_interval} epochs)")

            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(
                    f"checkpoint_{stage_name}_epoch{epoch+1}.pt",
                    optimizer=optimizer,
                    scheduler=scheduler,
                    save_optimizer=True,
                )

        return history

    def save_checkpoint(self, filename: str, optimizer=None, scheduler=None,
                         save_optimizer: bool = True):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }
        if save_optimizer:
            checkpoint['criterion_state_dict'] = self.criterion.state_dict()
            checkpoint['best_val_f1'] = self.best_val_f1
            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(checkpoint, os.path.join(self.config.output_dir, filename))

    def load_checkpoint(self, filename: str, optimizer=None, scheduler=None):
        checkpoint = torch.load(
            os.path.join(self.config.output_dir, filename),
            map_location=self.device,
            weights_only=False
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'criterion_state_dict' in checkpoint:
            self.criterion.load_state_dict(checkpoint['criterion_state_dict'])
        self.best_val_f1 = checkpoint.get('best_val_f1', -1.0)
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
