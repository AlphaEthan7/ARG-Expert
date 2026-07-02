#!/usr/bin/env python3
"""
ARG Data Preprocessing Pipeline.

Builds train/val/test splits from an ARG positive FASTA and up to two
non-ARG negative FASTA files using CD-HIT clustering, stratified splits,
and MMseqs2 hard-negative screening.

Usage:
    conda activate gene_pred
    python scripts/prepare_data.py \\
        --neg_fasta_a data/Non_ARG_DB.fasta \\
        --neg_fasta_b data/GCF_negatives.fasta
"""

import argparse
import os
import re
import json
import random
import logging
import shutil
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict, Set, Iterator

import numpy as np

# Reuse helpers from preprocess_data.py
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from preprocess_data import (
    parse_fasta,
    run_cd_hit,
    parse_cd_hit_clusters,
    stratified_cluster_split,
    sample_negatives,
    write_fasta,
    write_csv_data,
    set_seed,
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 30 raw -> 14 mapped categories (threshold = 90 sequences)
CATEGORY_MAPPING: Dict[str, str] = {
    # >= 90 -> named (13)
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
    'aminocoumarin': 'aminocoumarin',
    # < 90 -> other (17)
    'macrolide': 'other',
    'rifamycin': 'other',
    'efflux': 'other',
    'nitroimidazole': 'other',
    'nucleoside': 'other',
    'fusidane': 'other',
    'lincosamide': 'other',
    'tetracenomycin': 'other',
    'mupirocin': 'other',
    'pleuromutilin': 'other',
    'streptogramin': 'other',
    'disinfecting agents and antiseptics': 'other',
    'ionophore': 'other',
    'antibacterial_fatty_acid': 'other',
    'antibacterial free fatty acids': 'other',  # accept either spelling
    'quaternary ammonium': 'other',
    'elfamycin': 'other',
    'bicyclomycin': 'other',
}

CLASS_TO_ID: Dict[str, int] = {
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
    'aminocoumarin': 13,
}

SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
POS_NEG_RATIO = 3
CATEGORY_THRESHOLD = 90
CDHIT_IDENTITY = 0.7
MMSEQS_SENSITIVITY = 7.5
MMSEQS_EVALUE = 1e-3
MMSEQS_MAX_SEQS = 300
MIN_SEQ_LEN_AFTER_CLEAN = 10

# Paths — defaults read from data/ relative to the repo root.
# Override via --neg_fasta_a / --neg_fasta_b for custom negative sources.
ARG_FASTA_PATH = str(Path(__file__).resolve().parent.parent / 'data' / 'ARG_DB.fasta')
NEG_FASTA_A_DEFAULT = str(Path(__file__).resolve().parent.parent / 'data' / 'Non_ARG_DB.fasta')
NEG_FASTA_B_DEFAULT = str(Path(__file__).resolve().parent.parent / 'data' / 'GCF_negatives.fasta')


# ---------------------------------------------------------------------------
# Environment check
# ---------------------------------------------------------------------------

def check_tools():
    for tool in ('cd-hit', 'mmseqs'):
        if shutil.which(tool) is None:
            raise RuntimeError(
                f"`{tool}` not found on PATH. Run: conda activate gene_pred"
            )
    logger.info(f"cd-hit: {shutil.which('cd-hit')}")
    logger.info(f"mmseqs: {shutil.which('mmseqs')}")


# ---------------------------------------------------------------------------
# Streaming FASTA reader (for the 6 M-sequence negative source)
# ---------------------------------------------------------------------------

def iter_fasta(path: str) -> Iterator[Tuple[str, str]]:
    """Yield (header, sequence) tuples without loading the whole file."""
    header = None
    seq_chunks: List[str] = []
    with open(path, 'r') as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    yield header, ''.join(seq_chunks)
                header = line[1:]
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if header is not None:
            yield header, ''.join(seq_chunks)


def clean_sequence(seq: str) -> str:
    """Uppercase, strip stop codons, drop trailing spaces."""
    return seq.upper().replace('*', '').strip()


# ---------------------------------------------------------------------------
# Step: Merge + dedup negative sources
# ---------------------------------------------------------------------------

def merge_dedup_negatives(
    source_paths: List[str],
    output_fasta: str,
) -> List[Tuple[str, str]]:
    """
    Stream-merge multiple FASTA sources and dedup by sha1(sequence).
    Returns list of (neg_id, sequence) tuples and writes a single FASTA.
    """
    if os.path.exists(output_fasta) and os.path.getsize(output_fasta) > 0:
        logger.info(f"Merged negative FASTA exists, reading back: {output_fasta}")
        records = []
        for h, s in iter_fasta(output_fasta):
            records.append((h, s))
        logger.info(f"Loaded {len(records):,} pre-merged negatives")
        return records

    seen: Set[bytes] = set()
    records: List[Tuple[str, str]] = []
    total_in = 0
    duplicates = 0
    short_dropped = 0

    for src in source_paths:
        logger.info(f"Streaming negatives from: {src}")
        for _hdr, raw_seq in iter_fasta(src):
            total_in += 1
            seq = clean_sequence(raw_seq)
            if len(seq) < MIN_SEQ_LEN_AFTER_CLEAN:
                short_dropped += 1
                continue
            key = hashlib.sha1(seq.encode('ascii', errors='replace')).digest()
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            neg_id = f"neg_{len(records):08d}"
            records.append((neg_id, seq))
            if total_in % 1_000_000 == 0:
                logger.info(
                    f"  ... {total_in:,} processed, "
                    f"unique={len(records):,}, dup={duplicates:,}, "
                    f"short={short_dropped:,}"
                )

    logger.info(
        f"Merge complete: total_in={total_in:,}, unique={len(records):,}, "
        f"duplicates={duplicates:,}, short_dropped={short_dropped:,}"
    )

    os.makedirs(os.path.dirname(output_fasta), exist_ok=True)
    with open(output_fasta, 'w') as f:
        for neg_id, seq in records:
            f.write(f">{neg_id}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i + 60] + '\n')
    logger.info(f"Wrote merged negative FASTA: {output_fasta}")

    # Free the sha1 set; caller only needs records.
    seen.clear()
    return records


# ---------------------------------------------------------------------------
# Step: MMseqs2 hard-negative screening
# ---------------------------------------------------------------------------

def run_mmseqs2_screen(
    query_fasta: str,
    target_fasta: str,
    hits_m8: str,
    tmp_dir: str,
    sensitivity: float = MMSEQS_SENSITIVITY,
    evalue: float = MMSEQS_EVALUE,
    max_seqs: int = MMSEQS_MAX_SEQS,
    split_memory_limit: str = "0",
) -> Tuple[int, int]:
    """
    Run `mmseqs easy-search` and return (total_hit_rows, unique_query_hits).
    If hits_m8 already exists and is non-empty, skip the search.
    """
    os.makedirs(os.path.dirname(hits_m8), exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    if os.path.exists(hits_m8) and os.path.getsize(hits_m8) > 0:
        logger.info(f"MMseqs2 hits file exists, reusing: {hits_m8}")
    else:
        cmd = [
            'mmseqs', 'easy-search',
            query_fasta,
            target_fasta,
            hits_m8,
            tmp_dir,
            '-s', str(sensitivity),
            '--format-output', 'query,target,bits,evalue',
            '-e', str(evalue),
            '--max-seqs', str(max_seqs),
            '--split-memory-limit', split_memory_limit,
            '--remove-tmp-files', '1',
        ]
        logger.info(f"Running MMseqs2: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"MMseqs2 easy-search failed: rc={result.returncode}")

    total_rows = 0
    unique_qids: Set[str] = set()
    with open(hits_m8) as f:
        for line in f:
            total_rows += 1
            unique_qids.add(line.split('\t', 1)[0])
    logger.info(
        f"MMseqs2 parsed: total_rows={total_rows:,}, "
        f"unique_query_hits={len(unique_qids):,}"
    )
    return total_rows, len(unique_qids)


def collect_hit_query_ids(hits_m8: str) -> Set[str]:
    qids: Set[str] = set()
    with open(hits_m8) as f:
        for line in f:
            qids.add(line.split('\t', 1)[0])
    return qids


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='ARG Data Preprocessing Pipeline',
    )
    parser.add_argument(
        '--neg_fasta_a', type=str, default=NEG_FASTA_A_DEFAULT,
        help=f'First negative FASTA source (default: {NEG_FASTA_A_DEFAULT})',
    )
    parser.add_argument(
        '--neg_fasta_b', type=str, default=NEG_FASTA_B_DEFAULT,
        help=f'Second negative FASTA source (default: {NEG_FASTA_B_DEFAULT})',
    )
    args = parser.parse_args()

    neg_fasta_a = args.neg_fasta_a
    neg_fasta_b = args.neg_fasta_b

    # Validate that at least one negative source exists
    missing_a = not os.path.exists(neg_fasta_a)
    missing_b = not os.path.exists(neg_fasta_b)
    if missing_a and missing_b:
        logger.error(f"Missing negative FASTA: {neg_fasta_a}")
        logger.error(f"Missing negative FASTA: {neg_fasta_b}")
        logger.error(
            "Place negative sequences in data/Non_ARG_DB.fasta and/or "
            "data/GCF_negatives.fasta, or provide custom paths via "
            "--neg_fasta_a / --neg_fasta_b."
        )
        sys.exit(1)
    if missing_a:
        logger.warning(f"Negative source A not found, using B only: {neg_fasta_b}")
    if missing_b:
        logger.warning(f"Negative source B not found, using A only: {neg_fasta_a}")

    neg_sources = [p for p in (neg_fasta_a, neg_fasta_b) if os.path.exists(p)]
    check_tools()
    set_seed(SEED)

    base_dir = Path(__file__).resolve().parent.parent
    processed_dir = base_dir / 'data' / 'processed'
    temp_dir = processed_dir / 'temp'
    mmseqs_dir = temp_dir / 'mmseqs'
    processed_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    mmseqs_dir.mkdir(parents=True, exist_ok=True)

    train_csv = processed_dir / 'train.csv'
    val_csv = processed_dir / 'val.csv'
    test_csv = processed_dir / 'test.csv'
    train_neg_pool_csv = processed_dir / 'train_neg_pool.csv'
    report_path = processed_dir / 'preprocessing_report.json'
    arg_out_fasta = processed_dir / 'ARG_DB_for_train.fasta'

    cdhit_input = temp_dir / 'arg_for_cdhit.fasta'
    cdhit_prefix = temp_dir / f'arg_cdhit_{int(CDHIT_IDENTITY * 100)}'
    train_arg_fasta = temp_dir / 'train_arg.fasta'
    neg_merged_fasta = temp_dir / 'neg_merged.fasta'
    hits_m8 = mmseqs_dir / 'hits.m8'

    mmseqs_tmp = Path(tempfile.mkdtemp(prefix='mmseqs_prep_', dir='/tmp'))
    logger.info(f"MMseqs2 tmp dir: {mmseqs_tmp}")

    # =====================================================================
    # Step 1: Parse new ARG FASTA + category mapping
    # =====================================================================
    logger.info("=" * 60)
    logger.info("Step 1: Parsing ARG positives + mapping to 14 classes")
    logger.info("=" * 60)

    arg_records = parse_fasta(ARG_FASTA_PATH)
    mapped_records: List[Tuple[str, str, str]] = []
    raw_counts: Counter = Counter()
    mapped_counts: Counter = Counter()
    unknown_cats: Counter = Counter()

    for header, seq, raw_cat in arg_records:
        raw_counts[raw_cat] += 1
        if raw_cat in CATEGORY_MAPPING:
            mapped = CATEGORY_MAPPING[raw_cat]
        else:
            unknown_cats[raw_cat] += 1
            mapped = 'other'
        mapped_counts[mapped] += 1
        mapped_records.append((header, clean_sequence(seq), mapped))

    if unknown_cats:
        logger.warning(f"Unknown raw categories (mapped to 'other'): {dict(unknown_cats)}")
    logger.info(f"Raw category counts: {dict(raw_counts)}")
    logger.info(f"Mapped category counts: {dict(mapped_counts)}")
    assert len(mapped_counts) == 14, (
        f"Expected 14 classes after mapping; got {len(mapped_counts)}: {list(mapped_counts)}"
    )

    # =====================================================================
    # Step 2: CD-HIT 0.7 + stratified 80/10/10 cluster split
    # =====================================================================
    logger.info("=" * 60)
    logger.info("Step 2: CD-HIT clustering + stratified cluster split")
    logger.info("=" * 60)

    short_id_to_record: Dict[str, Tuple[str, str, str]] = {}
    for idx, (header, seq, cat) in enumerate(mapped_records):
        short_id_to_record[f"seq_{idx:06d}"] = (header, seq, cat)

    with open(cdhit_input, 'w') as f:
        for sid, (_h, seq, _c) in short_id_to_record.items():
            f.write(f">{sid}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i + 60] + '\n')

    seq_to_category = {sid: rec[2] for sid, rec in short_id_to_record.items()}

    clstr_path = run_cd_hit(str(cdhit_input), str(cdhit_prefix), identity=CDHIT_IDENTITY)
    clusters = parse_cd_hit_clusters(clstr_path)

    train_seqs, val_seqs, test_seqs = stratified_cluster_split(
        clusters, seq_to_category, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    )

    train_arg_records = [short_id_to_record[s] for s in train_seqs]
    val_arg_records = [short_id_to_record[s] for s in val_seqs]
    test_arg_records = [short_id_to_record[s] for s in test_seqs]
    logger.info(
        f"Positives — train={len(train_arg_records)}, "
        f"val={len(val_arg_records)}, test={len(test_arg_records)}"
    )

    # Assertions: no positive leakage between splits, all 14 classes present.
    train_set, val_set, test_set = set(train_seqs), set(val_seqs), set(test_seqs)
    assert not (train_set & val_set), "Train/Val overlap (positives)"
    assert not (train_set & test_set), "Train/Test overlap (positives)"
    assert not (val_set & test_set), "Val/Test overlap (positives)"
    for name, recs in [
        ('train', train_arg_records),
        ('val', val_arg_records),
        ('test', test_arg_records),
    ]:
        split_cats = set(r[2] for r in recs)
        missing = set(CLASS_TO_ID) - split_cats
        assert not missing, f"Split '{name}' missing categories: {missing}"

    # =====================================================================
    # Step 3: Write train ARG FASTA (MMseqs2 target) + ARG_DB_for_train.fasta
    # =====================================================================
    logger.info("=" * 60)
    logger.info("Step 3: Writing train ARG FASTA (MMseqs2 target)")
    logger.info("=" * 60)

    train_seq_ids = sorted(train_seqs)
    with open(train_arg_fasta, 'w') as f:
        for sid in train_seq_ids:
            _h, seq, _c = short_id_to_record[sid]
            f.write(f">{sid}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i + 60] + '\n')
    logger.info(f"Wrote {len(train_seq_ids):,} train ARGs to {train_arg_fasta}")

    write_fasta(train_arg_records + test_arg_records, str(arg_out_fasta))

    # =====================================================================
    # Step 4: Merge + dedup negative pool
    # =====================================================================
    logger.info("=" * 60)
    logger.info("Step 4: Merging and deduplicating negative sources")
    logger.info("=" * 60)

    neg_records = merge_dedup_negatives(
        neg_sources,
        str(neg_merged_fasta),
    )
    merged_neg_count = len(neg_records)
    assert merged_neg_count >= 6_000_000, (
        f"Merged negative pool unexpectedly small: {merged_neg_count}"
    )
    neg_id_to_seq: Dict[str, str] = {nid: seq for nid, seq in neg_records}

    # =====================================================================
    # Step 5: MMseqs2 hard-negative screen (target = train ARGs only)
    # =====================================================================
    logger.info("=" * 60)
    logger.info("Step 5: MMseqs2 easy-search (-s 7.5) — query=all negs, target=train ARGs")
    logger.info("=" * 60)

    total_hit_rows, unique_hit_negs = run_mmseqs2_screen(
        query_fasta=str(neg_merged_fasta),
        target_fasta=str(train_arg_fasta),
        hits_m8=str(hits_m8),
        tmp_dir=str(mmseqs_tmp),
    )
    hit_neg_ids = collect_hit_query_ids(str(hits_m8))
    assert len(hit_neg_ids) > 0, "MMseqs2 returned zero hits — investigate"
    logger.info(f"Training negative pool size = {len(hit_neg_ids):,}")

    # =====================================================================
    # Step 6: Sample val/test negatives from the non-hit complement
    # =====================================================================
    logger.info("=" * 60)
    logger.info("Step 6: Sampling val/test negatives (1:3 ratio)")
    logger.info("=" * 60)

    all_neg_ids = [nid for nid, _ in neg_records]
    non_hit_ids = [nid for nid in all_neg_ids if nid not in hit_neg_ids]
    logger.info(
        f"All neg = {len(all_neg_ids):,}, hits = {len(hit_neg_ids):,}, "
        f"non-hit pool = {len(non_hit_ids):,}"
    )

    n_val_neg = len(val_arg_records) * POS_NEG_RATIO
    n_test_neg = len(test_arg_records) * POS_NEG_RATIO

    val_neg_ids = set(sample_negatives(non_hit_ids, n_val_neg))
    remaining_for_test = [nid for nid in non_hit_ids if nid not in val_neg_ids]
    test_neg_ids = set(sample_negatives(remaining_for_test, n_test_neg))

    assert not (val_neg_ids & test_neg_ids), "Val/Test neg overlap"
    assert not (val_neg_ids & hit_neg_ids), "Val neg overlaps training pool"
    assert not (test_neg_ids & hit_neg_ids), "Test neg overlaps training pool"

    val_neg_records = [(nid, neg_id_to_seq[nid], 'non_arg') for nid in val_neg_ids]
    test_neg_records = [(nid, neg_id_to_seq[nid], 'non_arg') for nid in test_neg_ids]

    # =====================================================================
    # Step 7: Save training negative pool CSV (for per-epoch resampling)
    # =====================================================================
    logger.info("=" * 60)
    logger.info("Step 7: Writing training negative pool CSV")
    logger.info("=" * 60)

    with open(train_neg_pool_csv, 'w') as f:
        f.write("sequence\n")
        for nid in hit_neg_ids:
            f.write(f"{neg_id_to_seq[nid]}\n")
    logger.info(
        f"Saved training negative pool ({len(hit_neg_ids):,} seqs) "
        f"to {train_neg_pool_csv}"
    )

    # =====================================================================
    # Step 8: Pre-sample epoch-0 train negatives and write CSVs
    # =====================================================================
    logger.info("=" * 60)
    logger.info("Step 8: Pre-sampling epoch-0 train negatives + writing CSVs")
    logger.info("=" * 60)

    hit_neg_list = list(hit_neg_ids)
    n_train_neg = len(train_arg_records) * POS_NEG_RATIO
    train_neg_picks = set(sample_negatives(hit_neg_list, n_train_neg))
    train_neg_records = [(nid, neg_id_to_seq[nid], 'non_arg') for nid in train_neg_picks]

    def build_csv(arg_recs, neg_recs):
        seqs, labels, cats = [], [], []
        for _h, s, c in arg_recs:
            seqs.append(s); labels.append(1); cats.append(CLASS_TO_ID[c])
        for _id, s, _c in neg_recs:
            seqs.append(s); labels.append(0); cats.append(-1)
        idxs = list(range(len(seqs)))
        random.shuffle(idxs)
        return [seqs[i] for i in idxs], [labels[i] for i in idxs], [cats[i] for i in idxs]

    tr_s, tr_l, tr_c = build_csv(train_arg_records, train_neg_records)
    va_s, va_l, va_c = build_csv(val_arg_records, val_neg_records)
    te_s, te_l, te_c = build_csv(test_arg_records, test_neg_records)

    write_csv_data(tr_s, tr_l, tr_c, str(train_csv))
    write_csv_data(va_s, va_l, va_c, str(val_csv))
    write_csv_data(te_s, te_l, te_c, str(test_csv))

    # =====================================================================
    # Step 9: Generate preprocessing report
    # =====================================================================
    logger.info("=" * 60)
    logger.info("Step 9: Writing preprocessing report")
    logger.info("=" * 60)

    def split_stats(arg_recs, neg_recs):
        total = len(arg_recs) + len(neg_recs)
        cat_counts = Counter(r[2] for r in arg_recs)
        return {
            'total': total,
            'arg_count': len(arg_recs),
            'neg_count': len(neg_recs),
            'arg_ratio': round(len(arg_recs) / total, 4) if total else 0.0,
            'category_distribution': {CLASS_TO_ID[k]: v for k, v in cat_counts.items()},
        }

    report = {
        'seed': SEED,
        'split_type': 'train/val/test (cluster-level 80/10/10) + MMseqs2 all-hits hard negatives',
        'arg_source': ARG_FASTA_PATH,
        'category_threshold': CATEGORY_THRESHOLD,
        'num_classes': 14,
        'cdhit_identity': CDHIT_IDENTITY,
        'positive_negative_ratio': f'1:{POS_NEG_RATIO}',
        'negative_sources': neg_sources,
        'merged_neg_count': merged_neg_count,
        'mmseqs2': {
            'sensitivity': MMSEQS_SENSITIVITY,
            'evalue': MMSEQS_EVALUE,
            'max_seqs': MMSEQS_MAX_SEQS,
            'target': 'train_arg_only',
            'total_hit_rows': total_hit_rows,
            'unique_hit_negs': unique_hit_negs,
        },
        'train_neg_pool_size': len(hit_neg_ids),
        'category_mapping': CATEGORY_MAPPING,
        'class_to_id': CLASS_TO_ID,
        'raw_category_counts': dict(raw_counts),
        'mapped_category_counts': dict(mapped_counts),
        'splits': {
            'train': split_stats(train_arg_records, train_neg_records),
            'val': split_stats(val_arg_records, val_neg_records),
            'test': split_stats(test_arg_records, test_neg_records),
        },
        'negative_pool': {
            'train_pool_size': len(hit_neg_ids),
            'val_neg_count': len(val_neg_ids),
            'test_neg_count': len(test_neg_ids),
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
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote report: {report_path}")

    # =====================================================================
    # Final summary
    # =====================================================================
    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"ARG positives: {len(mapped_records):,}")
    logger.info(f"Merged neg pool: {merged_neg_count:,}")
    logger.info(f"Train neg pool (MMseqs2 hits): {len(hit_neg_ids):,}")
    logger.info(
        f"Train: {len(tr_s):,} ({len(train_arg_records)} ARG + "
        f"{len(train_neg_records)} neg)"
    )
    logger.info(
        f"Val:   {len(va_s):,} ({len(val_arg_records)} ARG + "
        f"{len(val_neg_records)} neg)"
    )
    logger.info(
        f"Test:  {len(te_s):,} ({len(test_arg_records)} ARG + "
        f"{len(test_neg_records)} neg)"
    )
    logger.info(f"Output: {processed_dir}")

    shutil.rmtree(mmseqs_tmp, ignore_errors=True)
    logger.info(f"Removed tmp dir: {mmseqs_tmp}")


if __name__ == '__main__':
    main()
