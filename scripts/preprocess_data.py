#!/usr/bin/env python3
"""
ARG Data Preprocessing Pipeline (v3 — 3-way split, pure random negative sampling)

This script processes raw FASTA files into training-ready datasets:
1. Maps 30 raw ARG categories to 14 classes (threshold < 100 -> "other")
2. Runs CD-HIT 70% clustering on ARGs
3. Performs stratified cluster-level train/val/test split (80/10/10)
4. Selects val/test negatives via random sampling (1:3 ratio)
5. Saves training negative pool for per-epoch random resampling
6. Outputs processed FASTA files, CSV splits, and metadata

Usage:
    conda activate gene_pred
    python scripts/preprocess_data.py
"""

import os
import re
import json
import random
import logging
import subprocess
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Set, Optional

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 30 raw categories -> 14 mapped categories
CATEGORY_MAPPING = {
    'beta-lactam': 'beta-lactam',
    'multidrug': 'multidrug',
    'MLS': 'MLS',
    'aminoglycoside': 'aminoglycoside',
    'peptide': 'peptide',
    'tetracycline': 'tetracycline',
    'phosphonic acid': 'phosphonic',
    'glycopeptide': 'glycopeptide',
    'quinolone': 'quinolone',
    'diaminopyrimidine': 'diaminopyrimidine',
    'phenicol': 'phenicol',
    'sulfonamide': 'sulfonamide',
    'macrolide': 'macrolide',
    'aminocoumarin': 'other',
    'rifamycin': 'other',
    'lincosamide': 'other',
    'efflux': 'other',
    'nitroimidazole': 'other',
    'nucleoside': 'other',
    'streptogramin': 'other',
    'fusidane': 'other',
    'disinfecting agents and antiseptics': 'other',
    'mupirocin': 'other',
    'tetracenomycin': 'other',
    'pleuromutilin': 'other',
    'antibacterial free fatty acids': 'other',
    'ionophore': 'other',
    'elfamycin': 'other',
    'quaternary ammonium': 'other',
    'bicyclomycin': 'other',
}

CLASS_TO_ID = {
    'beta-lactam': 0,
    'multidrug': 1,
    'MLS': 2,
    'aminoglycoside': 3,
    'peptide': 4,
    'tetracycline': 5,
    'phosphonic': 6,
    'glycopeptide': 7,
    'quinolone': 8,
    'diaminopyrimidine': 9,
    'other': 10,
    'phenicol': 11,
    'sulfonamide': 12,
    'macrolide': 13,
}

SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
POS_NEG_RATIO = 3  # 1:3 positive:negative
ESM_MODEL = "esm2_t30_150M_UR50D"
ESM_BATCH_SIZE = 64


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# FASTA I/O
# ---------------------------------------------------------------------------

def parse_fasta(fasta_path: str) -> List[Tuple[str, str, str]]:
    """Parse FASTA, return (header, sequence, raw_category) tuples."""
    records = []
    with open(fasta_path, 'r') as f:
        header = None
        seq_lines = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    sequence = ''.join(seq_lines)
                    raw_cat = header.split('|')[-1].strip() if '|' in header else None
                    records.append((header, sequence, raw_cat))
                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)
        if header is not None:
            sequence = ''.join(seq_lines)
            raw_cat = header.split('|')[-1].strip() if '|' in header else None
            records.append((header, sequence, raw_cat))
    logger.info(f"Parsed {len(records)} sequences from {fasta_path}")
    return records


def write_fasta(records: List[Tuple[str, str, str]], output_path: str):
    """Write records to FASTA. Appends category to header."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for header, sequence, category in records:
            f.write(f">{header}|{category}\n")
            for i in range(0, len(sequence), 60):
                f.write(sequence[i:i+60] + '\n')
    logger.info(f"Wrote {len(records)} sequences to {output_path}")


def write_csv_data(sequences: List[str], labels: List[int],
                   categories: List[int], output_path: str):
    """Write sequences with labels to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("sequence,is_arg,category\n")
        for seq, label, cat in zip(sequences, labels, categories):
            f.write(f"{seq},{label},{cat}\n")
    logger.info(f"Wrote {len(sequences)} records to {output_path}")


# ---------------------------------------------------------------------------
# Category mapping
# ---------------------------------------------------------------------------

def map_category(raw_cat: str) -> str:
    if raw_cat in CATEGORY_MAPPING:
        return CATEGORY_MAPPING[raw_cat]
    logger.warning(f"Unknown category '{raw_cat}', mapping to 'other'")
    return 'other'


# ---------------------------------------------------------------------------
# CD-HIT clustering
# ---------------------------------------------------------------------------

def run_cd_hit(input_fasta: str, output_prefix: str, identity: float = 0.9) -> str:
    """Run CD-HIT on protein sequences. Returns .clstr path."""
    clstr_path = f"{output_prefix}.clstr"
    if os.path.exists(clstr_path):
        logger.info(f"CD-HIT cluster file exists, reusing: {clstr_path}")
        return clstr_path

    # Word size depends on identity threshold (cd-hit recommendation)
    if identity >= 0.9:
        n_word = 5
    elif identity >= 0.8:
        n_word = 4
    elif identity >= 0.7:
        n_word = 3
    else:
        n_word = 2
    cmd = [
        'cd-hit', '-i', input_fasta, '-o', output_prefix,
        '-c', str(identity), '-n', str(n_word), '-d', '0', '-T', '0', '-M', '0',
    ]
    logger.info(f"Running CD-HIT: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"CD-HIT failed: {result.stderr}")
        raise RuntimeError(f"CD-HIT failed: {result.stderr}")
    logger.info(f"CD-HIT completed: {clstr_path}")
    return clstr_path


def parse_cd_hit_clusters(clstr_path: str) -> Dict[int, List[str]]:
    """Parse CD-HIT .clstr -> {cluster_id: [seq_names]}."""
    clusters = defaultdict(list)
    current_cluster = None
    with open(clstr_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>Cluster'):
                current_cluster = int(line.split()[1])
            elif line and current_cluster is not None:
                match = re.search(r'>(.+?)\.\.\.', line)
                if match:
                    clusters[current_cluster].append(match.group(1))
    logger.info(f"Parsed {len(clusters)} clusters from {clstr_path}")
    return dict(clusters)


# ---------------------------------------------------------------------------
# Stratified cluster-level split (train/val/test)
# ---------------------------------------------------------------------------

def stratified_cluster_split(
    clusters: Dict[int, List[str]],
    seq_to_category: Dict[str, str],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Stratified cluster-level train/val/test split.

    Greedy assignment per category: largest clusters assigned first,
    priority to the split furthest below its target. Tie-break: test > val > train.
    """
    cluster_info = []
    for cid, seq_names in clusters.items():
        cats = [seq_to_category.get(s, 'other') for s in seq_names]
        main_cat = Counter(cats).most_common(1)[0][0]
        cluster_info.append((cid, main_cat, len(seq_names)))

    cat_clusters = defaultdict(list)
    for cid, cat, size in cluster_info:
        cat_clusters[cat].append((cid, size))

    train_seqs: Set[str] = set()
    val_seqs: Set[str] = set()
    test_seqs: Set[str] = set()

    for cat, cat_cluster_list in cat_clusters.items():
        cat_cluster_list.sort(key=lambda x: x[1], reverse=True)
        cat_total = sum(size for _, size in cat_cluster_list)
        cat_train_target = int(cat_total * train_ratio)
        cat_val_target = int(cat_total * val_ratio)
        cat_test_target = cat_total - cat_train_target - cat_val_target

        cat_train: Set[str] = set()
        cat_val: Set[str] = set()
        cat_test: Set[str] = set()

        # For small categories, use sequence-level random split to avoid
        # cluster-level allocation bias (one large cluster skews the split)
        if cat_total < 200:
            all_seqs = []
            for cid, _ in cat_cluster_list:
                for sid in clusters[cid]:
                    if seq_to_category.get(sid) == cat:
                        all_seqs.append(sid)
            random.shuffle(all_seqs)
            n_train = int(len(all_seqs) * train_ratio)
            n_val = int(len(all_seqs) * val_ratio)
            cat_train = set(all_seqs[:n_train])
            cat_val = set(all_seqs[n_train:n_train + n_val])
            cat_test = set(all_seqs[n_train + n_val:])
            train_sum = len(cat_train)
            val_sum = len(cat_val)
            test_sum = len(cat_test)
        else:
            train_sum = 0
            val_sum = 0
            test_sum = 0

            for cid, size in cat_cluster_list:
                cluster_seqs = set(clusters[cid])
                train_gap = cat_train_target - train_sum
                val_gap = cat_val_target - val_sum
                test_gap = cat_test_target - test_sum

                gaps = [('train', train_gap), ('val', val_gap), ('test', test_gap)]
                gaps = [(n, g) for n, g in gaps if g > 0]

                if not gaps:
                    cat_train.update(cluster_seqs)
                    train_sum += size
                else:
                    priority = {'test': 0, 'val': 1, 'train': 2}
                    gaps.sort(key=lambda x: (-x[1], priority[x[0]]))
                    best = gaps[0][0]
                    if best == 'train':
                        cat_train.update(cluster_seqs)
                        train_sum += size
                    elif best == 'val':
                        cat_val.update(cluster_seqs)
                        val_sum += size
                    else:
                        cat_test.update(cluster_seqs)
                        test_sum += size

        train_seqs.update(cat_train)
        val_seqs.update(cat_val)
        test_seqs.update(cat_test)
        split_type = "seq-level" if cat_total < 200 else "cluster-level"
        logger.info(
            f"Category '{cat}': train={len(cat_train)} ({train_sum}), "
            f"val={len(cat_val)} ({val_sum}), "
            f"test={len(cat_test)} ({test_sum}) "
            f"[{split_type}]"
        )

    return train_seqs, val_seqs, test_seqs


def sample_negatives(pool_indices: List[int], n_total: int) -> List[int]:
    """Randomly sample n_total indices from pool_indices."""
    if n_total >= len(pool_indices):
        logger.warning(
            f"Requested {n_total} negatives but pool only has {len(pool_indices)}"
        )
        return list(pool_indices)
    selected = random.sample(pool_indices, n_total)
    logger.info(f"Sampled {len(selected)} negatives from pool of {len(pool_indices)}")
    return selected


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    set_seed(SEED)

    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'

    processed_dir = data_dir / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)

    arg_fasta = data_dir / 'ARG_DB.fasta'
    non_arg_fasta = data_dir / 'Non_ARG_DB.fasta'

    arg_out_fasta = processed_dir / 'ARG_DB_for_train.fasta'

    train_csv = processed_dir / 'train.csv'
    val_csv = processed_dir / 'val.csv'
    test_csv = processed_dir / 'test.csv'
    train_neg_pool_csv = processed_dir / 'train_neg_pool.csv'
    report_path = processed_dir / 'preprocessing_report.json'

    temp_dir = processed_dir / 'temp'
    temp_dir.mkdir(exist_ok=True)

    # =========================================================================
    # Step 1: Load and process ARG sequences (NO length filtering)
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 1: Processing ARG sequences")
    logger.info("=" * 60)

    arg_records = parse_fasta(str(arg_fasta))

    mapped_records = []
    raw_category_counts = Counter()
    mapped_category_counts = Counter()

    for header, sequence, raw_cat in arg_records:
        raw_category_counts[raw_cat] += 1
        mapped_cat = map_category(raw_cat)
        mapped_category_counts[mapped_cat] += 1
        mapped_records.append((header, sequence, mapped_cat))

    logger.info(f"Raw category counts: {dict(raw_category_counts)}")
    logger.info(f"Mapped category counts: {dict(mapped_category_counts)}")

    assert len(mapped_category_counts) == 14, (
        f"Expected 14 categories, got {len(mapped_category_counts)}"
    )

    # =========================================================================
    # Step 2: CD-HIT clustering and stratified train/test split
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 2: CD-HIT clustering and stratified train/test split")
    logger.info("=" * 60)

    short_id_to_record = {}
    for idx, (header, sequence, category) in enumerate(mapped_records):
        short_id = f"seq_{idx:06d}"
        short_id_to_record[short_id] = (header, sequence, category)

    cdhit_input = temp_dir / 'arg_for_cdhit.fasta'
    with open(cdhit_input, 'w') as f:
        for short_id, (header, sequence, category) in short_id_to_record.items():
            f.write(f">{short_id}\n")
            for i in range(0, len(sequence), 60):
                f.write(sequence[i:i+60] + '\n')

    seq_to_category = {sid: cat for sid, (_, _, cat) in short_id_to_record.items()}

    cdhit_output_prefix = str(temp_dir / 'arg_cdhit_70')
    for suffix in ['', '.clstr']:
        old = cdhit_output_prefix + suffix
        if os.path.exists(old):
            os.remove(old)

    clstr_path = run_cd_hit(str(cdhit_input), cdhit_output_prefix, identity=0.7)
    clusters = parse_cd_hit_clusters(clstr_path)

    train_seqs, val_seqs, test_seqs = stratified_cluster_split(
        clusters, seq_to_category, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO
    )

    logger.info(f"Split: train={len(train_seqs)}, val={len(val_seqs)}, test={len(test_seqs)}")

    train_arg_records = [short_id_to_record[sid] for sid in train_seqs]
    val_arg_records = [short_id_to_record[sid] for sid in val_seqs]
    test_arg_records = [short_id_to_record[sid] for sid in test_seqs]

    # Verify no leakage
    train_set = set(h for h, s, c in train_arg_records)
    val_set = set(h for h, s, c in val_arg_records)
    test_set = set(h for h, s, c in test_arg_records)
    assert len(train_set & val_set) == 0, "Train/Val overlap!"
    assert len(train_set & test_set) == 0, "Train/Test overlap!"
    assert len(val_set & test_set) == 0, "Val/Test overlap!"

    # Verify all 14 categories in each split
    for name, recs in [('train', train_arg_records), ('val', val_arg_records), ('test', test_arg_records)]:
        split_cats = set(c for h, s, c in recs)
        missing = set(CLASS_TO_ID.keys()) - split_cats
        assert len(missing) == 0, f"Split '{name}' missing categories: {missing}"
        logger.info(f"Split '{name}': {len(recs)} ARG sequences, "
                    f"{len(split_cats)} categories")

    # =========================================================================
    # Step 3: Write ARG output FASTA
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 3: Writing ARG output FASTA")
    logger.info("=" * 60)

    all_arg_records = train_arg_records + test_arg_records
    write_fasta(all_arg_records, str(arg_out_fasta))

    # =========================================================================
    # Step 4: Load non-ARG sequences
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 4: Loading non-ARG sequences")
    logger.info("=" * 60)

    non_arg_records = parse_fasta(str(non_arg_fasta))

    # =========================================================================
    # Step 5: Select val/test negatives (fixed, random sampling)
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 5: Selecting val/test negatives")
    logger.info("=" * 60)

    n_val_pos = len(val_arg_records)
    n_val_neg = n_val_pos * POS_NEG_RATIO
    n_test_pos = len(test_arg_records)
    n_test_neg = n_test_pos * POS_NEG_RATIO
    all_nonarg_indices = list(range(len(non_arg_records)))

    val_neg_indices = set(sample_negatives(all_nonarg_indices, n_val_neg))
    val_neg_records = [non_arg_records[i] for i in val_neg_indices]

    remaining_after_val = [i for i in all_nonarg_indices if i not in val_neg_indices]
    test_neg_indices = set(sample_negatives(remaining_after_val, n_test_neg))
    test_neg_records = [non_arg_records[i] for i in test_neg_indices]

    # =========================================================================
    # Step 6: Build training negative pool (all non-ARGs minus val/test negatives)
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 6: Building training negative pool")
    logger.info("=" * 60)

    reserved_indices = val_neg_indices | test_neg_indices
    train_pool_indices = [i for i in all_nonarg_indices if i not in reserved_indices]

    logger.info(
        f"Training negative pool: {len(train_pool_indices):,} sequences "
        f"(from {len(non_arg_records):,} total, "
        f"{len(val_neg_indices):,} reserved for val, "
        f"{len(test_neg_indices):,} reserved for test)"
    )

    # Save training negative pool (plain sequence list)
    with open(train_neg_pool_csv, 'w') as f:
        f.write("sequence\n")
        for idx in train_pool_indices:
            _, seq, _ = non_arg_records[idx]
            f.write(f"{seq}\n")
    logger.info(f"Saved training negative pool to {train_neg_pool_csv}")

    # =========================================================================
    # Step 7: Pre-sample first-epoch training negatives and write CSVs
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 7: Writing output CSVs")
    logger.info("=" * 60)

    n_train_pos = len(train_arg_records)
    n_train_neg = n_train_pos * POS_NEG_RATIO

    train_neg_idx = set(sample_negatives(train_pool_indices, n_train_neg))
    train_neg_records = [non_arg_records[i] for i in train_neg_idx]

    def build_csv_data(arg_recs, neg_recs):
        sequences, is_arg, categories = [], [], []
        for _, s, c in arg_recs:
            sequences.append(s)
            is_arg.append(1)
            categories.append(CLASS_TO_ID[c])
        for _, s, _ in neg_recs:
            sequences.append(s)
            is_arg.append(0)
            categories.append(-1)
        indices = list(range(len(sequences)))
        random.shuffle(indices)
        return (
            [sequences[i] for i in indices],
            [is_arg[i] for i in indices],
            [categories[i] for i in indices],
        )

    train_seqs_data, train_labels, train_cats = build_csv_data(
        train_arg_records, train_neg_records
    )
    val_seqs_data, val_labels, val_cats = build_csv_data(
        val_arg_records, val_neg_records
    )
    test_seqs_data, test_labels, test_cats = build_csv_data(
        test_arg_records, test_neg_records
    )

    write_csv_data(train_seqs_data, train_labels, train_cats, str(train_csv))
    write_csv_data(val_seqs_data, val_labels, val_cats, str(val_csv))
    write_csv_data(test_seqs_data, test_labels, test_cats, str(test_csv))

    # =========================================================================
    # Step 8: Generate report
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 8: Generating preprocessing report")
    logger.info("=" * 60)

    def split_stats(arg_recs, neg_recs):
        total = len(arg_recs) + len(neg_recs)
        arg_count = len(arg_recs)
        neg_count = len(neg_recs)
        cat_counts = Counter(c for _, _, c in arg_recs)
        return {
            'total': total,
            'arg_count': arg_count,
            'neg_count': neg_count,
            'arg_ratio': round(arg_count / total, 4) if total > 0 else 0,
            'category_distribution': {CLASS_TO_ID[k]: v for k, v in cat_counts.items()},
        }

    report = {
        'seed': SEED,
        'split_type': 'train/val/test (cluster-level 80/10/10)',
        'min_sequence_length': None,
        'category_threshold': 100,
        'num_classes': 14,
        'positive_negative_ratio': '1:3',
        'negative_sampling': 'pure_random',
        'cdhit_identity': 0.7,
        'train_ratio': TRAIN_RATIO,
        'val_ratio': VAL_RATIO,
        'test_ratio': TEST_RATIO,
        'category_mapping': CATEGORY_MAPPING,
        'class_to_id': CLASS_TO_ID,
        'splits': {
            'train': split_stats(train_arg_records, train_neg_records),
            'val': split_stats(val_arg_records, val_neg_records),
            'test': split_stats(test_arg_records, test_neg_records),
        },
        'negative_pool': {
            'train_pool_size': len(train_pool_indices),
            'val_neg_count': len(val_neg_indices),
            'test_neg_count': len(test_neg_indices),
            'per_epoch_resampling': True,
        },
        'output_files': {
            'arg_fasta': str(arg_out_fasta),
            'train_csv': str(train_csv),
            'val_csv': str(val_csv),
            'test_csv': str(test_csv),
            'train_neg_pool_csv': str(train_neg_pool_csv),
        },
    }

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report written to {report_path}")

    # =========================================================================
    # Final Summary
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total ARG sequences: {len(all_arg_records)}")
    logger.info(f"Total non-ARG sequences (full DB): {len(non_arg_records)}")
    logger.info(f"Train: {len(train_seqs_data)} "
                f"({len(train_arg_records)} ARG + {len(train_neg_records)} neg)")
    logger.info(f"Val:   {len(val_seqs_data)} "
                f"({len(val_arg_records)} ARG + {len(val_neg_records)} neg)")
    logger.info(f"Test:  {len(test_seqs_data)} "
                f"({len(test_arg_records)} ARG + {len(test_neg_records)} neg)")
    logger.info(f"Train neg pool: {len(train_pool_indices):,} (per-epoch resampling)")
    logger.info(f"Output directory: {processed_dir}")


if __name__ == '__main__':
    main()
