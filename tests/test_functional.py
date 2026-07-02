"""
Tier 1: Quick functional tests before full training
"""
import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.config import Config, set_seed
from model.data import ARGDataset, collate_fn
from model.architecture import ARGTransformer
from model.loss import ARGExpertLoss
from model.trainer import ARGTrainer
from torch.utils.data import DataLoader

def test_import():
    print("[1/8] Import test...")
    assert torch.cuda.is_available(), "CUDA not available!"
    print("  PASS: All imports successful, CUDA available")

def test_config():
    print("[2/8] Config validation...")
    cfg = Config()
    assert cfg.max_seq_length == 1000
    assert cfg.warmup_batch_size == 96
    assert cfg.finetune_batch_size == 32
    assert cfg.transformer_ff_dim == 3072
    print(f"  PASS: max_seq={cfg.max_seq_length}, warmup_bs={cfg.warmup_batch_size}, "
          f"finetune_bs={cfg.finetune_batch_size}, ff_dim={cfg.transformer_ff_dim}")

def test_model_build():
    print("[3/8] Model instantiation...")
    cfg = Config()
    model = ARGTransformer(cfg)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total:,}")
    print(f"  Trainable params: {trainable:,}")
    assert total > 0
    assert trainable > 0
    print("  PASS: Model built successfully")

def test_forward():
    print("[4/8] Forward pass...")
    cfg = Config()
    model = ARGTransformer(cfg).cuda()
    batch_size = 4
    tokens = torch.randint(0, 21, (batch_size, cfg.max_seq_length)).cuda()

    # Training mode (no return_attention)
    model.train()
    out = model(tokens)
    assert 'binary_pred' in out
    assert 'multiclass_pred' in out
    assert 'fused_features' in out
    assert out['binary_pred'].shape == (batch_size, 1)
    assert out['multiclass_pred'].shape == (batch_size, cfg.num_classes)
    assert out['fused_features'].shape == (batch_size, cfg.max_seq_length, cfg.embedding_dim)
    print(f"  binary_pred: {out['binary_pred'].shape}")
    print(f"  multiclass_pred: {out['multiclass_pred'].shape}")
    print(f"  fused_features: {out['fused_features'].shape}")

    # Eval mode with return_attention
    model.eval()
    out2 = model(tokens, return_attention=True)
    assert 'class_attention' in out2
    print(f"  class_attention: {out2['class_attention'].shape}")
    print("  PASS: Forward pass OK")

def test_loss():
    print("[5/8] Loss computation...")
    cfg = Config()
    model = ARGTransformer(cfg).cuda()
    criterion = ARGExpertLoss(cfg).cuda()

    tokens = torch.randint(0, 21, (4, cfg.max_seq_length)).cuda()
    binary_labels = torch.tensor([1.0, 0.0, 1.0, 0.0]).cuda()
    multiclass_labels = torch.tensor([0, -1, 5, -1]).cuda()  # -1 for non-ARG

    model.train()
    out = model(tokens)
    losses = criterion(out, binary_labels, multiclass_labels)

    assert 'total' in losses
    assert 'binary' in losses
    assert 'multiclass' in losses
    assert losses['total'].item() > 0
    print(f"  total_loss: {losses['total'].item():.4f}")
    print(f"  binary_loss: {losses['binary'].item():.4f}")
    print(f"  multiclass_loss: {losses['multiclass'].item():.4f}")
    print("  PASS: Loss computation OK")

def test_backward():
    print("[6/8] Backward pass...")
    cfg = Config()
    model = ARGTransformer(cfg).cuda()
    criterion = ARGExpertLoss(cfg).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    tokens = torch.randint(0, 21, (2, cfg.max_seq_length)).cuda()
    binary_labels = torch.tensor([1.0, 0.0]).cuda()
    multiclass_labels = torch.tensor([0, -1]).cuda()

    model.train()
    out = model(tokens)
    losses = criterion(out, binary_labels, multiclass_labels)

    optimizer.zero_grad(set_to_none=True)
    losses['total'].backward()

    # Check gradients exist
    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grad, "No gradients computed!"
    print("  PASS: Backward pass OK, gradients computed")

def test_gradient_checkpointing():
    print("[7/8] Gradient checkpointing...")
    cfg = Config()
    model = ARGTransformer(cfg).cuda()

    # Unfreeze ESM2 and enable checkpointing
    model.unfreeze_esm_layers(2)
    model.embedding.use_gradient_checkpointing = True

    tokens = torch.randint(0, 21, (2, cfg.max_seq_length)).cuda()
    out = model(tokens)
    loss = out['binary_pred'].mean()
    loss.backward()

    print("  PASS: Gradient checkpointing works with unfrozen ESM2")

def test_checkpoint_save_load():
    print("[8/8] Checkpoint save/load...")
    import os
    import tempfile
    cfg = Config()
    model = ARGTransformer(cfg).cuda()

    # Save
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': cfg,
    }
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = f.name
    torch.save(checkpoint, path)

    # Load
    model2 = ARGTransformer(cfg).cuda()
    ckpt = torch.load(path, map_location='cuda')
    model2.load_state_dict(ckpt['model_state_dict'])
    os.remove(path)

    print("  PASS: Checkpoint save/load OK")

if __name__ == "__main__":
    set_seed(42, deterministic=False)
    print("=" * 60)
    print("Tier 1: Quick Functional Tests")
    print("=" * 60)
    test_import()
    test_config()
    test_model_build()
    test_forward()
    test_loss()
    test_backward()
    test_gradient_checkpointing()
    test_checkpoint_save_load()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
