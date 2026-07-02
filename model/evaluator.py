"""
ARG_Expert: evaluation and visualization.
"""
import os
import json
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, classification_report,
    balanced_accuracy_score, cohen_kappa_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from .config import Config, AMINO_ACIDS, AA_TO_IDX, logger
from .architecture import ARGTransformer


class ARGEvaluator:
    """ARG model evaluator."""

    NAMED_CLASS_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13]
    OTHER_CLASS_ID = 10

    def __init__(self, model: ARGTransformer, config: Config):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.model.eval()

    def evaluate_comprehensive(
        self,
        dataloader,
        save_results: bool = True,
        use_fallback: bool = True,
    ) -> Dict[str, float]:
        """Evaluate the model on a dataloader.

        Args:
            dataloader: PyTorch DataLoader yielding tokens/labels/categories batches
            save_results: write `evaluation_results.json` to `config.output_dir`
            use_fallback: when True, multiclass evaluation uses the inference-style
                tau fallback: if max softmax across 13 named classes is below
                `config.multiclass_confidence_threshold`, the sample is predicted
                as 'other' (class 10), and class 10 is included in metrics.
                When False, class 10 is excluded entirely (legacy 0.5.3 behavior).
        """
        all_binary_probs = []
        all_binary_preds = []
        all_binary_labels = []
        all_multiclass_probs = []  # 14-dim (with class 10 set to 0 under fallback)
        all_multiclass_preds = []
        all_multiclass_labels = []
        # Full-sample buffers used to compute the union (View C) view, which
        # mixes true-ARG samples with model-predicted-ARG samples.
        all_full_pred_cats = []   # per-sample predicted category id (-1 if binary=0)
        all_full_true_cats = []   # per-sample true category id (-1 if non-ARG)

        tau = getattr(self.config, 'multiclass_confidence_threshold', 0.3)
        named_ids_arr = np.array(self.NAMED_CLASS_IDS)

        with torch.inference_mode():
            for batch in tqdm(dataloader, desc="Evaluating"):
                tokens = batch['tokens'].to(self.device)
                binary_labels = batch['labels'].to(self.device)
                multiclass_labels = batch['categories'].to(self.device)

                outputs = self.model(tokens)

                binary_logits = outputs['binary_pred'].squeeze(-1)
                binary_probs = torch.sigmoid(binary_logits).cpu().numpy()
                binary_preds = (binary_probs > self.config.binary_threshold).astype(int)

                all_binary_probs.extend(binary_probs)
                all_binary_preds.extend(binary_preds)
                all_binary_labels.extend(binary_labels.cpu().numpy())

                # Per-sample multiclass argmax (tau-fallback, all rows)
                full_logits = outputs['multiclass_pred'].clone()
                full_logits[:, self.OTHER_CLASS_ID] = float('-inf')
                full_softmax = F.softmax(full_logits, dim=1).cpu().numpy()
                full_named = full_softmax[:, self.NAMED_CLASS_IDS]
                full_max = full_named.max(axis=1)
                full_argmax = full_named.argmax(axis=1)
                full_pred_cat = np.where(
                    full_max < tau,
                    self.OTHER_CLASS_ID,
                    named_ids_arr[full_argmax],
                )
                # If model says binary=0, the per-sample prediction collapses to -1
                # ("not an ARG"), regardless of the multiclass head's argmax.
                binary_preds_np = binary_preds.astype(int)
                full_pred_cat = np.where(binary_preds_np == 1, full_pred_cat, -1)
                true_cats_np = multiclass_labels.cpu().numpy().astype(int)
                true_cat_filled = np.where(
                    binary_labels.cpu().numpy() == 1, true_cats_np, -1
                )
                all_full_pred_cats.extend(full_pred_cat.tolist())
                all_full_true_cats.extend(true_cat_filled.tolist())

                arg_mask = binary_labels == 1
                if use_fallback:
                    # Include all true ARGs (class 10 included). Predict via 13
                    # named-class softmax + tau fallback to class 10.
                    if arg_mask.sum() > 0:
                        multi_logits = outputs['multiclass_pred'][arg_mask].clone()
                        multi_logits[:, self.OTHER_CLASS_ID] = float('-inf')
                        multi_softmax = F.softmax(multi_logits, dim=1).cpu().numpy()
                        named_probs = multi_softmax[:, self.NAMED_CLASS_IDS]
                        max_named = named_probs.max(axis=1)
                        named_argmax = named_probs.argmax(axis=1)
                        preds = np.where(
                            max_named < tau,
                            self.OTHER_CLASS_ID,
                            named_ids_arr[named_argmax],
                        )
                        all_multiclass_probs.extend(multi_softmax)
                        all_multiclass_preds.extend(preds)
                        all_multiclass_labels.extend(multiclass_labels[arg_mask].cpu().numpy())
                else:
                    # Legacy: exclude class 10 from both ground truth and predictions
                    valid_mask = arg_mask & (multiclass_labels != self.OTHER_CLASS_ID)
                    if valid_mask.sum() > 0:
                        multiclass_logits = outputs['multiclass_pred'][valid_mask]
                        multiclass_logits[:, self.OTHER_CLASS_ID] = float('-inf')
                        multiclass_probs = F.softmax(multiclass_logits, dim=1).cpu().numpy()
                        multiclass_preds = multiclass_probs.argmax(axis=1)

                        all_multiclass_probs.extend(multiclass_probs)
                        all_multiclass_preds.extend(multiclass_preds)
                        all_multiclass_labels.extend(multiclass_labels[valid_mask].cpu().numpy())

        all_binary_probs = np.array(all_binary_probs)
        all_binary_preds = np.array(all_binary_preds)
        all_binary_labels = np.array(all_binary_labels)
        all_multiclass_probs = np.array(all_multiclass_probs)
        all_multiclass_preds = np.array(all_multiclass_preds)
        all_multiclass_labels = np.array(all_multiclass_labels)
        all_full_pred_cats = np.array(all_full_pred_cats, dtype=int)
        all_full_true_cats = np.array(all_full_true_cats, dtype=int)

        results = {}

        cm = confusion_matrix(all_binary_labels, all_binary_preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        else:
            npv = 0.0

        try:
            binary_auroc = roc_auc_score(all_binary_labels, all_binary_probs)
        except ValueError:
            binary_auroc = 0.0
        try:
            binary_auprc = average_precision_score(all_binary_labels, all_binary_probs)
        except ValueError:
            binary_auprc = 0.0

        results['binary'] = {
            'accuracy': accuracy_score(all_binary_labels, all_binary_preds),
            'precision': precision_score(all_binary_labels, all_binary_preds, zero_division=0),
            'recall': recall_score(all_binary_labels, all_binary_preds, zero_division=0),
            'specificity': recall_score(all_binary_labels, all_binary_preds, zero_division=0, pos_label=0),
            'f1': f1_score(all_binary_labels, all_binary_preds, zero_division=0),
            'auroc': binary_auroc,
            'auprc': binary_auprc,
            'mcc': matthews_corrcoef(all_binary_labels, all_binary_preds),
            'balanced_accuracy': balanced_accuracy_score(all_binary_labels, all_binary_preds),
            'cohens_kappa': cohen_kappa_score(all_binary_labels, all_binary_preds),
            'npv': npv,
            'confusion_matrix': cm.tolist()
        }

        if len(all_multiclass_preds) > 0:
            if use_fallback:
                labels_in_report = list(range(14))
                named_only_labels = self.NAMED_CLASS_IDS
                results['multiclass'] = {
                    'mode': f'tau_fallback (tau={tau:.3f})',
                    'accuracy': accuracy_score(all_multiclass_labels, all_multiclass_preds),
                    'macro_f1': f1_score(all_multiclass_labels, all_multiclass_preds,
                                          labels=labels_in_report, average='macro', zero_division=0),
                    'macro_f1_named_only': f1_score(all_multiclass_labels, all_multiclass_preds,
                                                     labels=named_only_labels,
                                                     average='macro', zero_division=0),
                    'micro_f1': f1_score(all_multiclass_labels, all_multiclass_preds,
                                          average='micro', zero_division=0),
                    'weighted_f1': f1_score(all_multiclass_labels, all_multiclass_preds,
                                             average='weighted', zero_division=0),
                }
            else:
                results['multiclass'] = {
                    'mode': 'class10_excluded',
                    'accuracy': accuracy_score(all_multiclass_labels, all_multiclass_preds),
                    'macro_f1': f1_score(all_multiclass_labels, all_multiclass_preds,
                                          average='macro', zero_division=0),
                    'micro_f1': f1_score(all_multiclass_labels, all_multiclass_preds,
                                          average='micro', zero_division=0),
                    'weighted_f1': f1_score(all_multiclass_labels, all_multiclass_preds,
                                             average='weighted', zero_division=0),
                }

            try:
                top3_preds = np.argsort(all_multiclass_probs, axis=1)[:, -3:]
                top3_correct = np.any(top3_preds == all_multiclass_labels[:, None], axis=1)
                results['multiclass']['top3_accuracy'] = float(top3_correct.mean())
            except Exception:
                results['multiclass']['top3_accuracy'] = 0.0

            try:
                n_classes = all_multiclass_probs.shape[1]
                y_true_bin = label_binarize(all_multiclass_labels, classes=range(n_classes))
                per_class_auroc = {}
                for i in range(n_classes):
                    if y_true_bin[:, i].sum() > 0:
                        per_class_auroc[str(i)] = float(roc_auc_score(y_true_bin[:, i], all_multiclass_probs[:, i]))
                results['multiclass']['per_class_auroc'] = per_class_auroc
            except Exception:
                pass

            class_report = classification_report(
                all_multiclass_labels, all_multiclass_preds,
                output_dict=True, zero_division=0
            )
            results['per_class'] = class_report

        # View C (union mask): score on (true_arg==1 | pred_arg==1).
        # Binary FN → pred_cat=-1; binary FP → true_cat=-1.
        union_mask = (all_binary_labels == 1) | (all_binary_preds == 1)
        if union_mask.sum() > 0:
            union_true = all_full_true_cats[union_mask]
            union_pred = all_full_pred_cats[union_mask]
            macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
                union_true, union_pred,
                labels=list(range(14)),
                average='macro', zero_division=0,
            )
            micro_p, micro_r, micro_f, _ = precision_recall_fscore_support(
                union_true, union_pred,
                labels=list(range(14)),
                average='micro', zero_division=0,
            )
            weighted_p, weighted_r, weighted_f, _ = precision_recall_fscore_support(
                union_true, union_pred,
                labels=list(range(14)),
                average='weighted', zero_division=0,
            )
            results['multiclass_union'] = {
                'mode': 'union(true_arg | pred_arg)',
                'support': int(union_mask.sum()),
                'accuracy': accuracy_score(union_true, union_pred),
                'macro_f1': float(macro_f),
                'macro_f1_named_only': f1_score(
                    union_true, union_pred,
                    labels=self.NAMED_CLASS_IDS,
                    average='macro', zero_division=0,
                ),
                'weighted_f1': float(weighted_f),
                'micro_f1': float(micro_f),
                'macro_precision': float(macro_p),
                'macro_recall': float(macro_r),
                'micro_precision': float(micro_p),
                'micro_recall': float(micro_r),
                'weighted_precision': float(weighted_p),
                'weighted_recall': float(weighted_r),
            }

            # Per-class P/R/F1 under View C
            p_arr, r_arr, f_arr, s_arr = precision_recall_fscore_support(
                union_true, union_pred,
                labels=list(range(14)), zero_division=0,
            )
            results['per_class_union'] = {
                str(cid): {
                    'category_id': cid,
                    'support': int(s_arr[cid]),
                    'precision': float(p_arr[cid]),
                    'recall': float(r_arr[cid]),
                    'f1': float(f_arr[cid]),
                }
                for cid in range(14)
            }

        if save_results:
            with open(os.path.join(self.config.output_dir, 'evaluation_results.json'), 'w') as f:
                json.dump(results, f, indent=2, default=str)

        return results

    def visualize_attention(
        self,
        sequence: str,
        category_idx: int = None,
        save_path: str = None
    ):
        from matplotlib import pyplot as plt

        tokens = [AA_TO_IDX.get(aa, 0) for aa in sequence.upper() if aa in AMINO_ACIDS]
        tokens = tokens[:self.config.max_seq_length]
        tokens_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)

        with torch.inference_mode():
            outputs = self.model(tokens_tensor, return_attention=True)
            class_attention = outputs['class_attention'][0].cpu().numpy()

        avg_attention = class_attention.mean(axis=1)

        fig, axes = plt.subplots(1, 2, figsize=(16, 4))

        ax = axes[0]
        im = ax.imshow(class_attention.T, cmap='hot', aspect='auto')
        ax.set_xlabel('Sequence Position')
        ax.set_ylabel('Class')
        ax.set_title('Class Attention Heatmap')
        plt.colorbar(im, ax=ax)

        ax = axes[1]
        ax.plot(avg_attention)
        ax.fill_between(range(len(avg_attention)), avg_attention, alpha=0.3)
        ax.set_xlabel('Sequence Position')
        ax.set_ylabel('Average Attention Weight')
        ax.set_title('Position-wise Average Attention')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        return avg_attention

    def identify_functional_sites(
        self,
        sequence: str,
        threshold: float = 0.8
    ) -> List[Tuple[int, int]]:
        avg_attention = self.visualize_attention(sequence)

        high_attention = avg_attention > threshold * avg_attention.max()

        sites = []
        in_site = False
        start = 0

        for i, is_high in enumerate(high_attention):
            if is_high and not in_site:
                start = i
                in_site = True
            elif not is_high and in_site:
                sites.append((start, i))
                in_site = False

        if in_site:
            sites.append((start, len(high_attention)))

        return sites
