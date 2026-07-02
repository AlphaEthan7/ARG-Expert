"""
Tier 2 (mini): Fast end-to-end verification with subset of real data.
"""
import sys
import os

# Add repo root to path for imports
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)
os.chdir(repo_root)

import random
import torch
from torch.utils.data import DataLoader

from model.config import Config, set_seed
from model.data import ARGDataset, collate_fn, load_data
from model.architecture import ARGTransformer
from model.loss import ARGExpertLoss
from model.trainer import ARGTrainer


DATA_DIR = 'data/processed'


def test_data_pipeline():
    print("[1/5] Data pipeline...")
    train_seqs, train_labels, train_cats = load_data(
        os.path.join(DATA_DIR, 'train.csv')
    )
    test_seqs, test_labels, test_cats = load_data(
        os.path.join(DATA_DIR, 'test.csv')
    )
    print(f"  Train: {len(train_seqs)}, Test: {len(test_seqs)}")
    assert len(train_seqs) > 0
    assert len(test_seqs) > 0
    print("  PASS")

    # Use first 200 train and 50 test for smoke testing
    return (train_seqs[:200], train_labels[:200], train_cats[:200],
            test_seqs[:50], test_labels[:50], test_cats[:50])


def test_model_and_training(train_seqs, train_labels, train_cats,
                            test_seqs, test_labels, test_cats):
    print("[2/5] Model + 1-batch training...")
    cfg = Config()
    cfg.warmup_epochs = 1
    cfg.finetune_epochs = 1
    cfg.output_dir = 'output_smoke_mini'

    # Internal validation split from train (mimics main training flow)
    n_val = max(1, int(len(train_seqs) * 0.1))
    val_seqs = train_seqs[:n_val]
    val_labels = train_labels[:n_val]
    val_cats = train_cats[:n_val]
    train_s = train_seqs[n_val:]
    train_l = train_labels[n_val:]
    train_c = train_cats[n_val:]

    train_ds = ARGDataset(train_s, train_l, train_c, max_length=cfg.max_seq_length)
    val_ds = ARGDataset(val_seqs, val_labels, val_cats, max_length=cfg.max_seq_length)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)

    model = ARGTransformer(cfg).cuda()

    # Test forward
    batch = next(iter(train_loader))
    tokens = batch['tokens'].cuda()
    out = model(tokens)
    assert out['binary_pred'].shape[0] == min(4, len(train_s))
    print(f"  Forward OK: binary_pred={out['binary_pred'].shape}")

    # Test backward
    criterion = ARGExpertLoss(cfg).cuda()
    losses = criterion(out, batch['labels'].cuda(), batch['categories'].cuda())
    losses['total'].backward()
    print(f"  Backward OK: loss={losses['total'].item():.4f}")

    # Test 1-epoch warmup (without per-epoch resampling for smoke simplicity)
    trainer = ARGTrainer(model, cfg)
    print("  Running 1 epoch warmup...")
    hist = trainer.train(train_loader, val_loader, stage_name="warmup")
    print(f"  Warmup complete: train_loss={hist['train_loss'][-1]:.4f}")

    # Test unfreeze + gradient checkpointing
    model.unfreeze_esm_layers(2)
    model.embedding.use_gradient_checkpointing = True
    trainer2 = ARGTrainer(model, cfg)
    trainer2.load_checkpoint('best_model_warmup.pt')
    print("  Running 1 epoch finetune (with grad checkpointing)...")
    hist2 = trainer2.train(train_loader, val_loader, stage_name="finetune")
    print(f"  Finetune complete: train_loss={hist2['train_loss'][-1]:.4f}")

    # Test evaluate
    trainer2.load_checkpoint('best_model_finetune.pt')
    metrics = trainer2.evaluate(val_loader)
    print(f"  Eval OK: binary_f1={metrics['binary_f1']:.4f}")
    print("  PASS")
    return model, cfg


def test_inference(model_path):
    print("[3/5] Inference compatibility...")
    from model.inference import ARGPredictor
    predictor = ARGPredictor(model_path)
    result = predictor.predict("MKTLLILTLVVVTIVCLDLGYTL")
    assert 'is_arg' in result
    assert 'arg_probability' in result
    print(f"  Prediction: is_arg={result['is_arg']}, prob={result['arg_probability']:.3f}")
    print("  PASS")


def test_checkpoint_format():
    print("[4/5] Checkpoint format...")
    ckpt = torch.load('output_smoke_mini/best_model_finetune.pt',
                       map_location='cpu', weights_only=False)
    assert 'model_state_dict' in ckpt
    assert 'config' in ckpt
    print("  PASS")


def test_padding_mask_effect():
    print("[5/5] Padding mask correctness...")
    cfg = Config()
    model = ARGTransformer(cfg).cuda()
    model.eval()

    # Sequence of length 10, rest padded
    tokens = torch.zeros(1, cfg.max_seq_length, dtype=torch.long).cuda()
    tokens[0, :10] = torch.randint(1, 21, (10,))

    with torch.no_grad():
        out = model(tokens)
    assert out['binary_pred'].shape == (1, 1)
    assert not torch.isnan(out['binary_pred']).any()
    print("  PASS")


if __name__ == "__main__":
    set_seed(42, deterministic=False)
    print("=" * 60)
    print("Tier 2 (mini): Fast End-to-End Verification")
    print("=" * 60)

    try:
        train_s, train_l, train_c, test_s, test_l, test_c = test_data_pipeline()
        model, cfg = test_model_and_training(train_s, train_l, train_c,
                                              test_s, test_l, test_c)
        test_inference('output_smoke_mini/best_model_finetune.pt')
        test_checkpoint_format()
        test_padding_mask_effect()

        print("\n" + "=" * 60)
        print("ALL SMOKE TESTS PASSED")
        print("=" * 60)

    except Exception as e:
        print(f"\nSMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
