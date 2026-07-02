"""
ARG_Expert: training entry point.
"""
import os
import random

import torch
from torch.utils.data import DataLoader

from .config import Config, set_seed, logger
from .data import ARGDataset, collate_fn, NegativeSampler, load_data
from .architecture import ARGTransformer
from .trainer import ARGTrainer
from .evaluator import ARGEvaluator


def main():
    set_seed(42, deterministic=False)

    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)

    logger.info("="*60)
    logger.info("ARG_Expert Training Pipeline")
    logger.info("="*60)
    logger.info(f"Device: {config.device}")
    logger.info(f"Data path: {config.data_path}")
    logger.info(f"Output directory: {config.output_dir}")

    logger.info("Loading data...")
    train_sequences, train_labels, train_categories = load_data(
        os.path.join(config.data_path, "train.csv")
    )
    val_sequences, val_labels, val_categories = load_data(
        os.path.join(config.data_path, "val.csv")
    )
    test_sequences, test_labels, test_categories = load_data(
        os.path.join(config.data_path, "test.csv")
    )

    logger.info(f"Train: {len(train_sequences)}, Val: {len(val_sequences)}, Test: {len(test_sequences)}")

    train_arg_seqs, train_arg_labels, train_arg_cats = [], [], []
    train_neg_init = []
    for s, l, c in zip(train_sequences, train_labels, train_categories):
        if l == 1:
            train_arg_seqs.append(s)
            train_arg_labels.append(l)
            train_arg_cats.append(c)
        else:
            train_neg_init.append(s)

    logger.info(f"Train ARGs: {len(train_arg_seqs)}, "
                f"Train negs (first epoch): {len(train_neg_init)}")

    neg_pool_hard_path = os.path.join(config.data_path, "train_neg_pool.csv")
    neg_pool_easy_path = os.path.join(config.data_path, "train_neg_pool_easy.csv")
    neg_sampler = NegativeSampler(neg_pool_hard_path, neg_pool_easy_path)

    # Use external val.csv directly — cluster-level validation split
    stage_train_arg_seqs = list(train_arg_seqs)
    stage_train_arg_labels = list(train_arg_labels)
    stage_train_arg_cats = list(train_arg_cats)

    val_dataset = ARGDataset(
        val_sequences, val_labels, val_categories,
        max_length=config.max_seq_length
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.finetune_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers,
    )

    logger.info(f"External validation (cluster-level): {len(val_sequences)} total")

    init_combined_seqs = list(stage_train_arg_seqs) + train_neg_init
    init_combined_labels = list(stage_train_arg_labels) + [0] * len(train_neg_init)
    init_combined_cats = list(stage_train_arg_cats) + [-1] * len(train_neg_init)
    init_indices = list(range(len(init_combined_seqs)))
    random.shuffle(init_indices)
    init_combined_seqs = [init_combined_seqs[i] for i in init_indices]
    init_combined_labels = [init_combined_labels[i] for i in init_indices]
    init_combined_cats = [init_combined_cats[i] for i in init_indices]

    train_dataset_warmup = ARGDataset(
        init_combined_seqs, init_combined_labels, init_combined_cats,
        max_length=config.max_seq_length
    )
    train_loader_warmup = DataLoader(
        train_dataset_warmup,
        batch_size=config.warmup_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers,
    )

    train_dataset_finetune = ARGDataset(
        init_combined_seqs, init_combined_labels, init_combined_cats,
        max_length=config.max_seq_length
    )
    train_loader_finetune = DataLoader(
        train_dataset_finetune,
        batch_size=config.finetune_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers,
    )

    test_dataset = ARGDataset(
        test_sequences, test_labels, test_categories,
        max_length=config.max_seq_length
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.finetune_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers,
    )

    logger.info("Creating model...")
    model = ARGTransformer(config)

    if hasattr(torch, 'compile'):
        try:
            torch._dynamo.config.suppress_errors = True
            model = torch.compile(model, mode="default")
            logger.info("Model compiled with torch.compile (default mode, suppress_errors=True)")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}. Continuing without compilation.")
    else:
        logger.info("torch.compile not available in this PyTorch version")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    logger.info("\n" + "="*60)
    logger.info("Stage A: Warmup (ESM2 frozen)")
    logger.info("="*60)

    trainer = ARGTrainer(model, config)
    history_warmup = trainer.train(
        train_loader_warmup, val_loader, stage_name="warmup",
        neg_sampler=neg_sampler,
        train_arg_seqs=stage_train_arg_seqs,
        train_arg_labels=stage_train_arg_labels,
        train_arg_cats=stage_train_arg_cats,
        batch_size=config.warmup_batch_size,
    )

    trainer.load_checkpoint("best_model_warmup.pt")

    logger.info("\n" + "="*60)
    logger.info("Stage B: Fine-tuning skipped — warmup converged")
    logger.info("="*60)

    # Copy warmup checkpoint as the final model so downstream eval scripts
    # can keep loading best_model_finetune.pt unchanged.
    import shutil
    shutil.copy(
        os.path.join(config.output_dir, "best_model_warmup.pt"),
        os.path.join(config.output_dir, "best_model_finetune.pt")
    )
    logger.info("Copied best_model_warmup.pt -> best_model_finetune.pt")

    logger.info("\n" + "="*60)
    logger.info("Final Evaluation on Test Set")
    logger.info("="*60)

    evaluator = ARGEvaluator(model, config)
    results = evaluator.evaluate_comprehensive(test_loader, save_results=True)

    logger.info("\nBinary Classification Results:")
    for metric, value in results['binary'].items():
        if isinstance(value, (int, float)):
            logger.info(f"  {metric}: {value:.4f}")
        else:
            logger.info(f"  {metric}: {value}")

    if 'multiclass' in results:
        logger.info("\nMulti-Class Classification Results:")
        for metric, value in results['multiclass'].items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")

    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)

    return model, results
