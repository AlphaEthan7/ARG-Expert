"""
ARG_Expert: model architecture.
"""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import esm

from .config import Config, AA_TO_IDX, logger


class MultiScaleEmbedding(nn.Module):
    """Embedding layer: ESM2 + learnable embeddings, projected to model dimension."""

    def __init__(self, esm_model_name: str = "esm2_t30_150M_UR50D", dropout: float = 0.1):
        super().__init__()

        logger.info(f"Loading ESM2 model: {esm_model_name}")
        self.esm_model, self.esm_alphabet = esm.pretrained.load_model_and_alphabet(esm_model_name)

        for param in self.esm_model.parameters():
            param.requires_grad = False

        self.esm_dim = self.esm_model.embed_tokens.embedding_dim
        self.esm_layers = len(self.esm_model.layers)

        self.learnable_dim = 256
        self.learnable_embed = nn.Embedding(21, self.learnable_dim, padding_idx=0)

        total_dim = self.esm_dim + self.learnable_dim
        self.projection = nn.Sequential(
            nn.Linear(total_dim, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        esm_tok_to_idx = self.esm_alphabet.tok_to_idx
        unk_idx = esm_tok_to_idx.get('<unk>', 3)
        pad_idx = esm_tok_to_idx.get('<pad>', 1)
        aa_to_esm = {}
        for aa, idx in AA_TO_IDX.items():
            if aa == '<pad>':
                aa_to_esm[idx] = pad_idx
            elif aa in esm_tok_to_idx:
                aa_to_esm[idx] = esm_tok_to_idx[aa]
            else:
                aa_to_esm[idx] = unk_idx
        token_map = [aa_to_esm.get(i, unk_idx) for i in range(21)]
        self.register_buffer('esm_token_map', torch.tensor(token_map, dtype=torch.long))
        self._esm_requires_grad = False
        self.use_gradient_checkpointing = False

        logger.info(f"MultiScaleEmbedding initialized: ESM dim={self.esm_dim}, Total dim={total_dim} -> 768")

    def _esm_forward(self, esm_tokens: torch.Tensor) -> torch.Tensor:
        esm_results = self.esm_model(
            esm_tokens,
            repr_layers=[self.esm_layers],
            return_contacts=False
        )
        return esm_results["representations"][self.esm_layers]

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        esm_tokens = self.esm_token_map[tokens]

        if self._esm_requires_grad:
            if self.use_gradient_checkpointing and self.training:
                esm_embeds = torch.utils.checkpoint.checkpoint(
                    self._esm_forward, esm_tokens, use_reentrant=False
                )
            else:
                esm_embeds = self._esm_forward(esm_tokens)
        else:
            with torch.no_grad():
                esm_embeds = self._esm_forward(esm_tokens)

        learnable_embeds = self.learnable_embed(tokens)

        combined = torch.cat([esm_embeds, learnable_embeds], dim=-1)
        return self.projection(combined)

    def unfreeze_last_layers(self, num_layers: int = 2):
        for param in self.esm_model.parameters():
            param.requires_grad = False

        for i in range(1, num_layers + 1):
            for param in self.esm_model.layers[-i].parameters():
                param.requires_grad = True

        self._esm_requires_grad = any(p.requires_grad for p in self.esm_model.parameters())
        logger.info(f"Unfroze last {num_layers} layers of ESM2")


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class MultiScaleCNN(nn.Module):
    """Multi-scale CNN branch (unused legacy, kept for backward compatibility)."""

    def __init__(
        self,
        in_dim: int = 768,
        out_dim: int = 768,
        kernel_sizes: List[int] = [3, 5, 7],
        num_filters: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.kernel_sizes = kernel_sizes

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.se_blocks = nn.ModuleList()

        for k in kernel_sizes:
            self.convs.append(nn.Conv1d(in_dim, num_filters, kernel_size=k, padding=k//2))
            self.bns.append(nn.BatchNorm1d(num_filters))
            self.se_blocks.append(SEBlock(num_filters))

        total_filters = num_filters * len(kernel_sizes)
        self.fusion = nn.Sequential(
            nn.Conv1d(total_filters, out_dim, kernel_size=1),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.residual_proj = nn.Conv1d(in_dim, out_dim, kernel_size=1) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)

        conv_outputs = []
        for conv, bn, se in zip(self.convs, self.bns, self.se_blocks):
            c = se(bn(F.gelu(conv(x))))
            conv_outputs.append(c)

        multi_scale = torch.cat(conv_outputs, dim=1)
        out = self.fusion(multi_scale)
        out = out + self.residual_proj(x)
        return out.transpose(1, 2)


class TransformerEncoder(nn.Module):
    """Transformer encoder branch.

    First num_layers-1 layers use stacked nn.TransformerEncoderLayer (fast path).
    Last layer is held separately so attention weights can be exposed for AECR
    regularization without disabling fast path for earlier layers.
    """

    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 2000
    ):
        super().__init__()

        self.pos_encoding = nn.Embedding(max_len, d_model)

        def make_layer():
            return nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )

        assert num_layers >= 1
        if num_layers >= 2:
            self.transformer = nn.TransformerEncoder(make_layer(), num_layers=num_layers - 1)
        else:
            self.transformer = None
        self.last_layer = make_layer()

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor = None,
        return_attn_weights: bool = False
    ):
        batch_size, seq_len, _ = x.size()

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_encoding(positions)

        if self.transformer is not None:
            x = self.transformer(x, src_key_padding_mask=padding_mask)

        if not return_attn_weights:
            x = self.last_layer(x, src_key_padding_mask=padding_mask)
            return x, None

        # Manually unfold last TransformerEncoderLayer (norm_first=True) to
        # capture attention weights from MultiheadAttention.
        layer = self.last_layer
        residual = x
        x_norm = layer.norm1(x)
        attn_out, attn_weights = layer.self_attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        x = residual + layer.dropout1(attn_out)
        x = x + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(x))))))
        return x, attn_weights


class GatedFusion(nn.Module):
    """Gated residual fusion — lightweight alternative to cross-attention."""

    def __init__(self, dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        g = self.gate(features)
        out = g * features + (1 - g) * self.ffn(features)
        return self.norm(out)


class ContrastiveLearningModule(nn.Module):
    """Contrastive learning module (legacy, kept for checkpoint compatibility)."""

    def __init__(self, input_dim: int = 768, proj_dim: int = 128, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

        self.projector = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, proj_dim)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = features.mean(dim=1)
        z = F.normalize(self.projector(pooled), dim=1)
        return z

    def nt_xent_loss(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        batch_size = z_i.size(0)

        z = torch.cat([z_i, z_j], dim=0)
        sim_matrix = torch.mm(z, z.t()) / self.temperature

        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix = sim_matrix.masked_fill(mask, -1e4)

        pos_mask = torch.zeros_like(mask)
        pos_mask[:batch_size, batch_size:] = torch.eye(batch_size, device=z.device)
        pos_mask[batch_size:, :batch_size] = torch.eye(batch_size, device=z.device)

        pos_sim = sim_matrix[pos_mask].view(2 * batch_size, 1)
        neg_sim = sim_matrix[~pos_mask].view(2 * batch_size, -1)

        logits = torch.cat([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z.device)

        return F.cross_entropy(logits, labels)


class MultiTaskHeads(nn.Module):
    """Multi-task prediction heads (decoupled: no shared layer)."""

    def __init__(
        self,
        input_dim: int = 768,
        num_classes: int = 14,
        dropout: float = 0.3
    ):
        super().__init__()

        # Independent heads — no shared projection layer
        self.binary_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, 1)
        )

        self.multiclass_head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_classes)
        )

        self.class_attention = nn.Linear(input_dim, num_classes)

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        pooled = features.mean(dim=1)

        binary_out = self.binary_head(pooled)
        multiclass_out = self.multiclass_head(pooled)

        if return_attention:
            attention_weights = torch.softmax(self.class_attention(features), dim=1)
            return binary_out, multiclass_out, attention_weights

        return binary_out, multiclass_out


class ARGTransformer(nn.Module):
    """Complete ARG_Expert model."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.embedding = MultiScaleEmbedding(
            esm_model_name=config.esm_model_name,
            dropout=config.dropout
        )

        self.transformer_branch = TransformerEncoder(
            d_model=config.embedding_dim,
            nhead=config.num_attention_heads,
            num_layers=config.num_transformer_layers,
            dim_feedforward=config.transformer_ff_dim,
            dropout=config.dropout,
            max_len=config.max_seq_length
        )

        self.fusion_module = GatedFusion(
            dim=config.embedding_dim,
            dropout=config.dropout
        )

        self.contrastive_module = ContrastiveLearningModule(
            input_dim=config.embedding_dim,
            proj_dim=128,
            temperature=0.5
        )

        self.heads = MultiTaskHeads(
            input_dim=config.embedding_dim,
            num_classes=config.num_classes,
            dropout=config.dropout * 3
        )

    def forward(
        self,
        tokens: torch.Tensor,
        return_attention: bool = False,
        return_attn_weights: bool = False
    ) -> Dict[str, torch.Tensor]:
        embedded = self.embedding(tokens)
        padding_mask = (tokens == 0)

        trans_features, attn_weights = self.transformer_branch(
            embedded, padding_mask=padding_mask, return_attn_weights=return_attn_weights
        )
        fused_features = self.fusion_module(trans_features)

        if return_attention:
            binary_out, multiclass_out, class_attention = self.heads(fused_features, return_attention=True)
        else:
            binary_out, multiclass_out = self.heads(fused_features, return_attention=False)
            class_attention = None

        outputs = {
            'binary_pred': binary_out,
            'multiclass_pred': multiclass_out,
            'fused_features': fused_features,
        }

        if class_attention is not None:
            outputs['class_attention'] = class_attention
        if attn_weights is not None:
            outputs['attn_weights'] = attn_weights
            outputs['padding_mask'] = padding_mask

        return outputs

    def unfreeze_esm_layers(self, num_layers: int = 2):
        self.embedding.unfreeze_last_layers(num_layers)
