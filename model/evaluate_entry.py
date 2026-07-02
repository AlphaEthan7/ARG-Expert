"""
ARG_Expert: standalone test-set evaluation entry point.
Loads a trained checkpoint and runs comprehensive evaluation on the test split.
"""
import os
import sys

import torch
from torch.utils.data import DataLoader

from .config import Config, set_seed, logger
from .data import ARGDataset, collate_fn, load_data
from .architecture import ARGTransformer
from .evaluator import ARGEvaluator


def main(
    model_path: str = None,
    data_path: str = None,
    output_dir: str = None,
):
    set_seed(42, deterministic=False)

    config = Config()

    if data_path is not None:
        config.data_path = data_path
    if output_dir is not None:
        config.output_dir = output_dir
    else:
        config.output_dir = "./output"

    os.makedirs(config.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("ARG_Expert Test Set Evaluation")
    logger.info("=" * 60)
    logger.info(f"Device: {config.device}")
    logger.info(f"Data path: {config.data_path}")
    logger.info(f"Output directory: {config.output_dir}")

    logger.info("Loading test data...")
    test_sequences, test_labels, test_categories = load_data(
        os.path.join(config.data_path, "test.csv")
    )
    logger.info(f"Test samples: {len(test_sequences)}")

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

    if model_path is None:
        model_path = os.path.join(config.output_dir, "best_model_finetune.pt")
    logger.info(f"Loading checkpoint: {model_path}")

    checkpoint = torch.load(
        model_path,
        map_location=config.device,
        weights_only=False
    )
    state_dict = checkpoint['model_state_dict']

    # Strip torch.compile _orig_mod. prefix if present
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {
            k.replace('_orig_mod.', ''): v for k, v in state_dict.items()
        }
        logger.info("Stripped _orig_mod. prefix from checkpoint state_dict")

    model.load_state_dict(state_dict)
    logger.info("Checkpoint loaded successfully")

    logger.info("\n" + "=" * 60)
    logger.info("Running Comprehensive Evaluation on Test Set")
    logger.info("=" * 60)

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

    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Complete!")
    logger.info(f"Results saved to: {os.path.join(config.output_dir, 'evaluation_results.json')}")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    main()
