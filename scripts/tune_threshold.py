"""
Threshold tuning: binary threshold scan (with recall constraint) plus
multiclass confidence tau scan for the "other" fallback (class 10).
"""
import os
import sys
import argparse

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support, matthews_corrcoef,
    confusion_matrix, balanced_accuracy_score, cohen_kappa_score,
    f1_score
)

from model.config import Config, set_seed
from model.data import ARGDataset, collate_fn, load_data
from model.architecture import ARGTransformer

NAMED_CLASS_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13]
OTHER_CLASS_ID = 10


def load_model(config, checkpoint_path):
    model = ARGTransformer(config)
    ckpt = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    state_dict = ckpt['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(config.device)
    model.eval()
    return model


def evaluate_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    return {
        'threshold': threshold,
        'precision': p,
        'recall': r,
        'f1': f1,
        'mcc': mcc,
        'balanced_acc': bal_acc,
        'cohens_kappa': kappa,
        'TP': int(cm[1, 1]),
        'FN': int(cm[1, 0]),
        'FP': int(cm[0, 1]),
        'TN': int(cm[0, 0]),
    }


def evaluate_multiclass_with_fallback(y_cat_true, y_named_probs, tau):
    """Evaluate multiclass with tau-based "other" fallback.

    y_cat_true: shape (N,), true category ids (0-13, including 10)
    y_named_probs: shape (N, 13), softmax probs of named classes
    tau: if max(y_named_probs[i]) < tau, predict class 10 ("other")
    """
    max_named = y_named_probs.max(axis=1)
    named_argmax = y_named_probs.argmax(axis=1)
    y_pred = np.where(
        max_named < tau,
        OTHER_CLASS_ID,
        np.array(NAMED_CLASS_IDS)[named_argmax]
    )

    all_classes = list(range(14))
    macro_f1 = f1_score(y_cat_true, y_pred, labels=all_classes,
                        average='macro', zero_division=0)
    named_macro_f1 = f1_score(y_cat_true, y_pred, labels=NAMED_CLASS_IDS,
                              average='macro', zero_division=0)

    # Per-class metrics for "other"
    other_mask_true = (y_cat_true == OTHER_CLASS_ID)
    other_mask_pred = (y_pred == OTHER_CLASS_ID)
    n_other_true = int(other_mask_true.sum())
    n_other_pred = int(other_mask_pred.sum())
    tp = int((other_mask_true & other_mask_pred).sum())
    fp = int((~other_mask_true & other_mask_pred).sum())
    fn = int((other_mask_true & ~other_mask_pred).sum())
    other_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    other_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    other_f1 = (2 * other_precision * other_recall / (other_precision + other_recall)
                if (other_precision + other_recall) > 0 else 0.0)

    named_mask = (y_cat_true != OTHER_CLASS_ID)
    if named_mask.sum() > 0:
        named_acc = float((y_pred[named_mask] == y_cat_true[named_mask]).mean())
        named_fallback_rate = float(other_mask_pred[named_mask].mean())
    else:
        named_acc = 0.0
        named_fallback_rate = 0.0

    return {
        'tau': tau,
        'macro_f1_14': macro_f1,
        'macro_f1_named': named_macro_f1,
        'other_precision': other_precision,
        'other_recall': other_recall,
        'other_f1': other_f1,
        'other_true': n_other_true,
        'other_pred': n_other_pred,
        'other_tp': tp,
        'other_fp': fp,
        'other_fn': fn,
        'named_acc': named_acc,
        'named_fallback_rate': named_fallback_rate,
    }


def run_inference(model, loader, device):
    """Single pass: collect binary probs and named-class softmax."""
    binary_probs_all = []
    named_probs_all = []
    binary_labels_all = []
    cat_labels_all = []

    with torch.inference_mode():
        for batch in loader:
            tokens = batch['tokens'].to(device)
            out = model(tokens)
            bin_probs = torch.sigmoid(out['binary_pred']).cpu().numpy().flatten()

            # Mask class 10, softmax over 14 dims, keep 13 named classes
            multi_logits = out['multiclass_pred'].clone()
            multi_logits[:, OTHER_CLASS_ID] = float('-inf')
            multi_softmax = F.softmax(multi_logits, dim=1).cpu().numpy()
            named_probs = multi_softmax[:, NAMED_CLASS_IDS]

            binary_probs_all.append(bin_probs)
            named_probs_all.append(named_probs)
            binary_labels_all.append(batch['labels'].numpy().flatten())
            cat_labels_all.append(batch['categories'].numpy().flatten())

    return (
        np.concatenate(binary_probs_all),
        np.concatenate(named_probs_all),
        np.concatenate(binary_labels_all),
        np.concatenate(cat_labels_all),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'],
                        help='Which split to tune on')
    parser.add_argument('--min_recall', type=float, default=0.90,
                        help='Minimum recall constraint (binary)')
    parser.add_argument('--optimize', type=str, default='f1',
                        choices=['f1', 'precision', 'mcc'],
                        help='Metric to optimize under the recall constraint (binary)')
    parser.add_argument('--step', type=float, default=0.001,
                        help='Threshold search granularity (binary)')
    parser.add_argument('--mc_optimize', type=str, default='macro_f1_14',
                        choices=['macro_f1_14', 'macro_f1_named', 'other_f1'],
                        help='Metric to optimize for multiclass tau')
    parser.add_argument('--mc_step', type=float, default=0.01,
                        help='Multiclass tau search granularity')
    args = parser.parse_args()

    set_seed(42, deterministic=False)
    config = Config()

    data_path = os.path.join(config.data_path, f"{args.split}.csv")
    checkpoint_path = os.path.join(config.output_dir, "best_model_finetune.pt")

    print(f"Loading data: {data_path}")
    sequences, labels, categories = load_data(data_path)
    dataset = ARGDataset(sequences, labels, categories, max_length=config.max_seq_length)
    loader = DataLoader(
        dataset, batch_size=64, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=False
    )

    print(f"Loading checkpoint: {checkpoint_path}")
    model = load_model(config, checkpoint_path)

    print("Running inference (collecting binary + named-class probs)...")
    y_prob, named_probs, y_true, cat_true = run_inference(model, loader, config.device)

    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    n_other = int((cat_true == OTHER_CLASS_ID).sum())
    print(f"\nDataset: {args.split}, total={len(y_true)}, pos={n_pos}, neg={n_neg}, true_other={n_other}")
    print(f"Default (th=0.50):", end=" ")
    default = evaluate_at_threshold(y_true, y_prob, 0.5)
    print(f"P={default['precision']:.4f} R={default['recall']:.4f} F1={default['f1']:.4f} MCC={default['mcc']:.4f}")

    # ====== Part 1: Binary threshold sweep ======
    thresholds = np.arange(0.001, 0.999, args.step)
    results = [evaluate_at_threshold(y_true, y_prob, th) for th in thresholds]
    feasible = [r for r in results if r['recall'] >= args.min_recall]

    print(f"\n{'='*70}")
    print(f"[Binary] Threshold scan: {len(thresholds)} points, recall >= {args.min_recall}")
    if not feasible:
        print(f"WARNING: No threshold achieves recall >= {args.min_recall}")
        feasible = sorted(results, key=lambda x: x['recall'], reverse=True)[:1]
    else:
        print(f"Feasible thresholds: {len(feasible)}")

    best = max(feasible, key=lambda x: x[args.optimize])
    print(f"\n[Binary] Best under constraint (optimize {args.optimize}):")
    print(f"  threshold = {best['threshold']:.3f}")
    print(f"  precision = {best['precision']:.4f}")
    print(f"  recall    = {best['recall']:.4f}")
    print(f"  F1        = {best['f1']:.4f}")
    print(f"  MCC       = {best['mcc']:.4f}")
    print(f"  CM: TN={best['TN']} FP={best['FP']} FN={best['FN']} TP={best['TP']}")

    # ====== Part 2: Multiclass tau sweep (ARG samples only) ======
    arg_mask = (y_true == 1)
    cat_arg = cat_true[arg_mask]
    named_probs_arg = named_probs[arg_mask]
    print(f"\n{'='*70}")
    print(f"[Multiclass] tau scan on {arg_mask.sum()} true ARGs ({n_other} are class-10 'other')")

    taus = np.arange(0.0, 1.0 + args.mc_step / 2, args.mc_step)
    mc_results = [evaluate_multiclass_with_fallback(cat_arg, named_probs_arg, tau)
                  for tau in taus]

    print(f"\n[Multiclass] tau=0.00 (never fallback): macro_f1_14={mc_results[0]['macro_f1_14']:.4f}, "
          f"macro_f1_named={mc_results[0]['macro_f1_named']:.4f}, "
          f"other_recall={mc_results[0]['other_recall']:.4f}")

    best_mc = max(mc_results, key=lambda r: r[args.mc_optimize])
    print(f"\n[Multiclass] Best tau (optimize {args.mc_optimize}):")
    print(f"  tau              = {best_mc['tau']:.3f}")
    print(f"  macro_f1_14      = {best_mc['macro_f1_14']:.4f}")
    print(f"  macro_f1_named   = {best_mc['macro_f1_named']:.4f}")
    print(f"  other_precision  = {best_mc['other_precision']:.4f}")
    print(f"  other_recall     = {best_mc['other_recall']:.4f}")
    print(f"  other_f1         = {best_mc['other_f1']:.4f}")
    print(f"  other_pred       = {best_mc['other_pred']} (of {best_mc['other_true']} true)")
    print(f"  named_acc        = {best_mc['named_acc']:.4f}")
    print(f"  named_fallback_rate = {best_mc['named_fallback_rate']:.4f}  "
          f"(named-class ARGs misrouted to 'other')")

    print(f"\n[Multiclass] tau sweep samples:")
    print(f"  {'tau':>5} {'macroF1_14':>11} {'macroF1_named':>14} "
          f"{'other_P':>9} {'other_R':>9} {'other_F1':>9} {'named_acc':>10} {'fall_rate':>10}")
    sample_taus = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7]
    for tau in sample_taus:
        r = evaluate_multiclass_with_fallback(cat_arg, named_probs_arg, tau)
        print(f"  {r['tau']:>5.2f} {r['macro_f1_14']:>11.4f} {r['macro_f1_named']:>14.4f} "
              f"{r['other_precision']:>9.4f} {r['other_recall']:>9.4f} {r['other_f1']:>9.4f} "
              f"{r['named_acc']:>10.4f} {r['named_fallback_rate']:>10.4f}")


if __name__ == "__main__":
    main()
