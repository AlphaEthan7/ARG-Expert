"""
ARG_Expert: data loading and sampling.
"""
import os
import random
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .config import Config, AMINO_ACIDS, AA_TO_IDX, logger


class ARGDataset(Dataset):
    """ARG dataset with pre-tokenization to eliminate per-epoch CPU overhead."""

    def __init__(
        self,
        sequences: List[str],
        labels: List[int],
        categories: List[int] = None,
        max_length: int = 1000
    ):
        self.labels = labels
        self.categories = categories if categories is not None else [-1] * len(labels)
        self.max_length = max_length

        self.tokenized_sequences = [
            self._tokenize_and_pad(seq, max_length) for seq in sequences
        ]

    def __len__(self):
        return len(self.tokenized_sequences)

    def __getitem__(self, idx):
        return {
            'tokens': self.tokenized_sequences[idx],
            'label': torch.tensor(self.labels[idx], dtype=torch.float),
            'category': torch.tensor(self.categories[idx], dtype=torch.long),
        }

    @staticmethod
    def _tokenize_and_pad(sequence: str, max_length: int = 1000) -> torch.Tensor:
        sequence = sequence.upper()
        sequence = ''.join([aa for aa in sequence if aa in AMINO_ACIDS])

        if len(sequence) > max_length:
            sequence = sequence[:max_length]

        tokens = [AA_TO_IDX.get(aa, 0) for aa in sequence]

        if len(tokens) < max_length:
            tokens.extend([0] * (max_length - len(tokens)))

        return torch.tensor(tokens, dtype=torch.long)


def collate_fn(batch):
    tokens = torch.stack([item['tokens'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    categories = torch.stack([item['category'] for item in batch])

    return {
        'tokens': tokens,
        'labels': labels,
        'categories': categories,
    }


class NegativeSampler:
    """Per-epoch negative sampler with 50/50 hard + easy mix."""

    def __init__(self, neg_pool_hard_path: str, neg_pool_easy_path: str = None, seed: int = 42):
        self.rng = random.Random(seed)
        self.hard_seqs: List[str] = []
        self.easy_seqs: List[str] = []

        with open(neg_pool_hard_path, 'r') as f:
            header = f.readline()
            for line in f:
                seq = line.strip()
                if seq:
                    self.hard_seqs.append(seq)

        if neg_pool_easy_path and os.path.exists(neg_pool_easy_path):
            with open(neg_pool_easy_path, 'r') as f:
                header = f.readline()
                for line in f:
                    seq = line.strip()
                    if seq:
                        self.easy_seqs.append(seq)
            logger.info(f"NegativeSampler loaded {len(self.hard_seqs):,} hard + {len(self.easy_seqs):,} easy negatives")
        else:
            self.easy_seqs = list(self.hard_seqs)
            logger.info(f"NegativeSampler loaded {len(self.hard_seqs):,} hard negatives (no easy pool)")

    def sample_n(self, n: int) -> List[str]:
        """Sample exactly n negatives, 50% hard + 50% easy."""
        total_available = len(self.hard_seqs) + len(self.easy_seqs)
        if n >= total_available:
            return list(self.hard_seqs) + list(self.easy_seqs)

        n_hard = n // 2
        n_easy = n - n_hard
        # Clamp to available counts
        n_hard = min(n_hard, len(self.hard_seqs))
        n_easy = min(n_easy, len(self.easy_seqs))
        # Rebalance if one pool is exhausted
        deficit = n - (n_hard + n_easy)
        if deficit > 0:
            if n_hard < len(self.hard_seqs):
                n_hard = min(n_hard + deficit, len(self.hard_seqs))
            elif n_easy < len(self.easy_seqs):
                n_easy = min(n_easy + deficit, len(self.easy_seqs))

        sampled = self.rng.sample(self.hard_seqs, n_hard) + self.rng.sample(self.easy_seqs, n_easy)
        return sampled

    def sample(self, n_pos: int) -> List[str]:
        return self.sample_n(n_pos * 3)  # 1:3 ratio


def load_data(data_path: str) -> Tuple[List[str], List[int], List[int]]:
    df = pd.read_csv(data_path)
    sequences = df['sequence'].tolist()
    labels = df['is_arg'].tolist()

    category_mapping = {
        'beta-lactam': 0, 'multidrug': 1, 'MLS': 2, 'aminoglycoside': 3,
        'peptide': 4, 'tetracycline': 5, 'phosphonic': 6, 'glycopeptide': 7,
        'quinolone': 8, 'diaminopyrimidine': 9, 'other': 10, 'phenicol': 11,
        'sulfonamide': 12, 'aminocoumarin': 13, 'non_arg': -1
    }

    raw_categories = df['category'].tolist()
    categories = []
    for cat in raw_categories:
        if isinstance(cat, int) or (isinstance(cat, float) and cat == int(cat)):
            categories.append(int(cat))
        elif cat in category_mapping:
            categories.append(category_mapping[cat])
        else:
            categories.append(-1)

    return sequences, labels, categories
