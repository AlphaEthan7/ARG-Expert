"""
ARG_Expert: backward-compatibility shim.

All symbols are re-exported from their new canonical locations.
This file preserves existing import paths and script entry point.
"""
import sys
import os

# Ensure repo root is on sys.path so absolute imports of sibling modules resolve
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from model.config import Config, set_seed, AMINO_ACIDS, AA_TO_IDX
from model.data import ARGDataset, collate_fn, NegativeSampler, load_data
from model.architecture import (
    MultiScaleEmbedding, TransformerEncoder, GatedFusion,
    MultiTaskHeads, ContrastiveLearningModule,
    SEBlock, MultiScaleCNN, ARGTransformer
)
from model.loss import ARGExpertLoss
from model.trainer import ARGTrainer
from model.evaluator import ARGEvaluator
from model.train_entry import main

__all__ = [
    "Config", "set_seed", "AMINO_ACIDS", "AA_TO_IDX",
    "ARGDataset", "collate_fn", "NegativeSampler", "load_data",
    "MultiScaleEmbedding", "TransformerEncoder", "GatedFusion",
    "MultiTaskHeads", "ContrastiveLearningModule",
    "SEBlock", "MultiScaleCNN", "ARGTransformer",
    "ARGExpertLoss",
    "ARGTrainer",
    "ARGEvaluator",
    "main",
]

if __name__ == "__main__":
    main()
