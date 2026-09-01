"""
Conditioners: frozen encoder features -> one L2-normalized vector for AudioLDM.

Terminology (previously inconsistent -- "mapper", "xformer" and "conditioner" all referred
to this): a **conditioner** turns the frozen encoder's frame sequence into the single vector
that tells AudioLDM's U-Net *what* to generate, as distinct from the latent being generated.
`conditioner` is the term used everywhere now; `mapper` is kept only as a deprecated alias so
older checkpoints and configs still load.

Its whole job is (B, T, D_in) -> (B, 512): pool over time, project to width. Everything below
is a different answer to "how much machinery does that pooling and projection need?", which
is an open question -- Whisper has already contextualised the sequence, so extra temporal
modelling here may be redundant, and 32M from-scratch params on 15k pairs is a lot for what
is essentially a projection.

The ladder, simple -> complex. Each rung changes ONE thing so a difference is attributable:

  L0 mean_linear   masked mean-pool + one linear      0.66M   is a linear probe enough?
  L1 attn_mlp      attention-pool + MLP               7.09M   does learned pooling help?
  L2 xformer       transformer + attention-pool      32.28M   does temporal modelling help?
  L3 qformer       k learned queries + MLP           18.63M   is one pooled vector the limit?

Note the ladder is ordered by conceptual complexity, not parameter count: qformer has fewer
params than xformer because xformer's two 4096-wide transformer FFNs dominate.

Shared contract, so the trainer never needs to know which rung is in use:
    forward(features, mask) -> (B, output_dim), L2-normalized
    null_condition(batch, device)      the unconditional vector for CFG
    apply_cfg_dropout(cond, p)         AudioLDM's condition dropout
Subclasses implement `encode()` and inherit normalization, the null embedding and dropout.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# AudioLDM's CLAP projection width. Verified at runtime against the U-Net's
# projection_class_embeddings_input_dim rather than trusted as a constant.
CLAP_EMBED_DIM = 512


class AttentionPool(nn.Module):
    """Pool (B, T, D) -> (B, D) with a single learned query, honouring a padding mask."""

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * dim**-0.5)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x:    (B, T, D)
        mask: (B, T) bool, True = real frame. None means all frames are real.
        """
        x = self.norm(x)
        q = self.query.expand(x.shape[0], -1, -1)

        key_padding_mask = None
        if mask is not None:
            # MultiheadAttention wants True = IGNORE, the inverse of our convention.
            key_padding_mask = ~mask
            # A fully masked row produces NaNs, so keep frame 0 alive.
            all_masked = key_padding_mask.all(dim=1)
            if all_masked.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked, 0] = False

        pooled, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask)
        return pooled.squeeze(1)


class BaseConditioner(nn.Module):
    """Shared normalization, null embedding and CFG dropout for every rung."""

    def __init__(self, output_dim: int = CLAP_EMBED_DIM) -> None:
        super().__init__()
        self.output_dim = output_dim
        # Unconditional embedding, learned. Provides the branch classifier-free guidance
        # needs at sampling time.
        self.null_embedding = nn.Parameter(torch.zeros(output_dim))

    def encode(self, features: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # L2-normalized because AudioLDM's pipeline normalizes its CLAP conditions; an
        # unnormalized vector is out of distribution for the U-Net even if its direction is
        # right.
        return F.normalize(self.encode(features, mask), dim=-1)

    def null_condition(self, batch_size: int, device: torch.device) -> torch.Tensor:
        null = self.null_embedding.unsqueeze(0).expand(batch_size, -1).to(device)
        return F.normalize(null, dim=-1)

    def apply_cfg_dropout(self, cond: torch.Tensor, dropout_prob: float) -> torch.Tensor:
        """Replace whole conditions with the null embedding, as AudioLDM does for CFG."""
        if dropout_prob <= 0.0:
            return cond
        keep = torch.rand(cond.shape[0], device=cond.device) >= dropout_prob
        null = self.null_condition(cond.shape[0], cond.device)
        return torch.where(keep.unsqueeze(1), cond, null)

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _masked_mean(features: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """
    Mean over real frames only.

    Masking matters more than it looks: Whisper emits a fixed 1500 frames for its 30 s
    window while TTAS speech averages 4.9 s (~245 real frames), so an unmasked mean would
    dilute the utterance with ~84% silence.
    """
    if mask is None:
        return features.mean(dim=1)
    m = mask.unsqueeze(-1).to(features.dtype)
    return (features * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


class MeanPoolLinear(BaseConditioner):
    """L0: the minimal baseline -- masked mean-pool, one linear layer."""

    def __init__(self, input_dim: int, output_dim: int = CLAP_EMBED_DIM) -> None:
        super().__init__(output_dim)
        self.input_dim = input_dim
        self.proj = nn.Linear(input_dim, output_dim)

    def encode(self, features: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        return self.proj(_masked_mean(features, mask))


class AttnPoolMLP(BaseConditioner):
    """L1: learned attention pooling + MLP head, no temporal self-attention."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = CLAP_EMBED_DIM,
        hidden_dim: int = 1024,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(output_dim)
        self.input_dim = input_dim
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.pool = AttentionPool(hidden_dim, num_heads=num_heads)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode(self, features: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        return self.head(self.pool(self.in_proj(features), mask=mask))


class XFormerConditioner(BaseConditioner):
    """
    L2: self-attention over the frame sequence, then attention pooling.

    Submodule names (in_proj / blocks / pool / out / null_embedding) are deliberately
    unchanged from the original `SpeechToClapMapper`, so state_dict keys match and
    checkpoints written before the rename still load.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = CLAP_EMBED_DIM,
        hidden_dim: int = 1024,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(output_dim)
        self.input_dim = input_dim

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        )
        self.pool = AttentionPool(hidden_dim, num_heads=num_heads)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode_sequence(self, features: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """
        The PRE-POOL per-frame representation, (B, T, hidden_dim).

        `encode` computes this and then discards it in `self.pool`, purely because
        AudioLDM's class_labels slot wants a single vector. A2S's content channel wants a
        SEQUENCE (see salm/a2s/conditioner_sequence.py), so it is exposed here rather than
        recomputed -- same weights, same forward, one shared computation.
        """
        x = self.in_proj(features)
        src_key_padding_mask = None if mask is None else ~mask
        for block in self.blocks:
            x = block(x, src_key_padding_mask=src_key_padding_mask)
        return x

    def encode(self, features: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        return self.out(self.pool(self.encode_sequence(features, mask), mask=mask))


class QueryPoolMapper(BaseConditioner):
    """
    L3: k learned query tokens cross-attend to the frames, then are concatenated.

    Probes whether single-vector pooling is the limiting factor: multiple queries can each
    specialise (scene, action, intensity) before the final projection, while still fitting
    AudioLDM's single-vector conditioning interface.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = CLAP_EMBED_DIM,
        hidden_dim: int = 1024,
        num_queries: int = 8,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(output_dim)
        self.input_dim = input_dim
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * hidden_dim**-0.5)
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList(
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * num_queries, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode(self, features: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        x = self.norm(self.in_proj(features))
        q = self.queries.expand(x.shape[0], -1, -1)

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask
            all_masked = key_padding_mask.all(dim=1)
            if all_masked.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked, 0] = False

        for attn in self.layers:
            delta, _ = attn(q, x, x, key_padding_mask=key_padding_mask)
            q = q + delta

        return self.head(q.flatten(1))


# Deprecated alias. Kept so checkpoints and configs written before the rename keep working;
# prefer XFormerConditioner in new code.
SpeechToClapMapper = XFormerConditioner
