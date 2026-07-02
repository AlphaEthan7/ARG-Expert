"""
ARG_Expert: configuration and utilities.
"""
import os
import random
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = False):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


@dataclass
class Config:
    """Model and training configuration."""
    data_path: str = "./data/processed"
    max_seq_length: int = 1000
    min_seq_length: int = 50

    esm_model_name: str = "esm2_t30_150M_UR50D"
    embedding_dim: int = 768
    num_transformer_layers: int = 4
    num_attention_heads: int = 8
    transformer_ff_dim: int = 3072
    dropout: float = 0.1
    num_classes: int = 14

    warmup_epochs: int = 5
    warmup_batch_size: int = 96
    warmup_lr: float = 1e-4
    warmup_weight_decay: float = 0.01

    finetune_epochs: int = 40
    finetune_batch_size: int = 32
    finetune_lr_embedding: float = 3e-6
    finetune_lr_other: float = 1e-4
    finetune_weight_decay: float = 0.01
    finetune_warmup_epochs: int = 5
    finetune_patience: int = 7

    alpha_binary: float = 6.0
    # Per-sample BCE weights: ARG samples get lower weight so the multiclass
    # head dominates backbone gradients on ARG features.
    alpha_binary_arg: float = 1.0
    alpha_binary_non: float = 6.0
    beta_multiclass: float = 1.5
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1
    # Low because asymmetric BCE pushes non-ARG probabilities near zero.
    # Tune with scripts/tune_threshold.py.
    binary_threshold: float = 0.002
    multiclass_confidence_threshold: float = 0.3

    # AECR regularization (attention entropy + local continuity).
    # Only applied during training; attention is captured from the last
    # Transformer layer in train mode.
    use_aecr: bool = True
    aecr_sigma: float = 3.0
    aecr_lambda_ent: float = 0.01
    aecr_lambda_loc: float = 0.005

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 8
    use_amp: bool = True
    pin_memory: bool = True
    prefetch_factor: int = 4
    persistent_workers: bool = False

    val_split: float = 0.1

    output_dir: str = "./output"
    save_interval: int = 10
    eval_interval: int = 10


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i+1 for i, aa in enumerate(AMINO_ACIDS)}
AA_TO_IDX['<pad>'] = 0
