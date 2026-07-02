"""
ARG_Expert: loss functions.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from .config import Config


class AECRLoss(nn.Module):
    """Attention Entropy + Continuity Regularization (adapted from MCT-ARG).

    Entropy term penalizes unfocused attention; local-continuity term
    encourages each query to attend near the diagonal. Padded positions
    are excluded, and the Gaussian kernel is row-normalized.
    """

    def __init__(self, sigma: float = 3.0, lambda_ent: float = 0.01, lambda_loc: float = 0.005):
        super().__init__()
        self.sigma = sigma
        self.lambda_ent = lambda_ent
        self.lambda_loc = lambda_loc
        self._kernels = {}

    def _get_kernel(self, L: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (L, device, dtype)
        if key not in self._kernels:
            i = torch.arange(L, device=device).view(-1, 1).float()
            j = torch.arange(L, device=device).view(1, -1).float()
            kernel = torch.exp(-(i - j).pow(2) / (2 * self.sigma ** 2))
            kernel = kernel / kernel.sum(dim=1, keepdim=True)  # row-normalize
            self._kernels[key] = kernel.to(dtype=dtype)
        return self._kernels[key]

    def _aecr_chunk(
        self,
        attn_chunk: torch.Tensor,
        valid_chunk: torch.Tensor,
        kernel_b: torch.Tensor,
    ):
        """Compute (entropy_sum, local_sum) for one checkpointed chunk."""
        b, H, L, _ = attn_chunk.shape
        v_k = valid_chunk.view(b, 1, 1, L)
        q_mask = valid_chunk.view(b, 1, L)

        attn = attn_chunk * v_k
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-10)

        p = attn.clamp_min(1e-10)
        row_entropy = -(p * torch.log(p)).sum(dim=-1)  # (b, H, L)
        entropy_chunk = (row_entropy * q_mask).sum().to(torch.float32)

        local_similarity = (attn * kernel_b).sum(dim=-1)  # (b, H, L)
        local_chunk = (local_similarity * q_mask).sum().to(torch.float32)

        return entropy_chunk, local_chunk

    def forward(
        self,
        attn_weights: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            attn_weights: (B, H, L, L). Rows are renormalized after zeroing
                padded keys (MultiheadAttention dropout breaks row sums).
            padding_mask: (B, L) bool, True at padded positions.
        Returns:
            scalar regularization loss.
        """
        B, H, L, _ = attn_weights.shape
        device = attn_weights.device
        dtype = attn_weights.dtype

        if padding_mask is not None:
            valid = (~padding_mask).to(dtype=dtype)  # (B, L)
        else:
            valid = torch.ones((B, L), device=device, dtype=dtype)

        kernel = self._get_kernel(L, device, dtype)  # (L, L), row-normalized
        kernel_b = kernel.view(1, 1, L, L)

        denom = valid.sum().to(torch.float32) * H + 1e-8

        chunk_size = min(4, B)  # keep per-chunk memory manageable

        entropy_sum = torch.zeros((), device=device, dtype=torch.float32)
        local_sum = torch.zeros((), device=device, dtype=torch.float32)

        use_checkpoint = attn_weights.requires_grad and torch.is_grad_enabled()

        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            a = attn_weights[start:end]
            v = valid[start:end]
            if use_checkpoint:
                ent_c, loc_c = checkpoint.checkpoint(
                    self._aecr_chunk, a, v, kernel_b, use_reentrant=False
                )
            else:
                ent_c, loc_c = self._aecr_chunk(a, v, kernel_b)
            entropy_sum = entropy_sum + ent_c
            local_sum = local_sum + loc_c

        entropy_loss = entropy_sum / denom
        local_mean = local_sum / denom
        local_loss = 1.0 - local_mean

        return self.lambda_ent * entropy_loss + self.lambda_loc * local_loss


class ARGExpertLoss(nn.Module):
    """ARG_Expert loss (BCE + Focal Loss + optional AECR)."""

    def __init__(self, config: Config, num_classes: int = 14):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.register_buffer("class_weights", torch.ones(num_classes))

        self.use_aecr = getattr(config, "use_aecr", False)
        if self.use_aecr:
            self.aecr = AECRLoss(
                sigma=getattr(config, "aecr_sigma", 3.0),
                lambda_ent=getattr(config, "aecr_lambda_ent", 0.01),
                lambda_loc=getattr(config, "aecr_lambda_loc", 0.005),
            )

    def focal_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        gamma: float = 2.0
    ) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', weight=self.class_weights,
            label_smoothing=self.config.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** gamma * ce_loss).mean()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        binary_target: torch.Tensor,
        multiclass_target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        binary_pred = outputs['binary_pred']
        multiclass_pred = outputs['multiclass_pred']

        # Per-sample BCE with asymmetric ARG/non-ARG weights
        bce_per_sample = F.binary_cross_entropy_with_logits(
            binary_pred.squeeze(-1),
            binary_target.float(),
            reduction='none'
        )

        arg_mask = binary_target == 1
        w_arg = self.config.alpha_binary_arg
        w_non = self.config.alpha_binary_non
        weights = torch.where(arg_mask, w_arg, w_non).to(bce_per_sample.dtype)
        loss_binary_weighted = (bce_per_sample * weights).mean()  # un-normalized mean

        # Unweighted BCE for logging only — keeps existing dashboards comparable.
        loss_binary_raw = bce_per_sample.mean()

        loss_multiclass = torch.tensor(0.0, device=binary_pred.device)

        # Exclude class 10 ("other") — heterogeneous catch-all
        valid_mask = arg_mask & (multiclass_target != 10)

        if valid_mask.sum() > 0:
            loss_multiclass = self.focal_loss(
                multiclass_pred[valid_mask],
                multiclass_target[valid_mask],
                gamma=self.config.focal_gamma
            )

        total_loss = loss_binary_weighted + self.config.beta_multiclass * loss_multiclass

        result = {
            'total': total_loss,
            'binary': loss_binary_raw,
            'multiclass': loss_multiclass,
        }

        if self.use_aecr and 'attn_weights' in outputs:
            loss_aecr = self.aecr(
                outputs['attn_weights'],
                outputs.get('padding_mask', None)
            )
            total_loss = total_loss + loss_aecr
            result['total'] = total_loss
            result['aecr'] = loss_aecr

        return result

    def update_class_weights(self, class_counts: torch.Tensor, max_ratio: float = 10.0):
        weights = 1.0 / (class_counts + 1e-6)
        weights = weights / weights.sum() * len(weights)
        # Cap extreme ratios to prevent gradient explosion for rare classes
        min_weight = weights.max() / max_ratio
        weights = torch.clamp(weights, min=min_weight)
        # Renormalize after clamping
        weights = weights / weights.sum() * len(weights)
        self.class_weights = weights.to(self.class_weights.device)
