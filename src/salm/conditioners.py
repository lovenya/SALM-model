"""
Conditioners: frozen encoder features -> one L2-normalized vector for AudioLDM.

These are the rungs of the simple->complex ladder. Each changes exactly ONE thing from
the previous, so a difference in results is attributable:

  L0 mean_linear   mean-pool + single linear        ~0.7M   is a linear probe enough?
  L1 attn_mlp      attention-pool + MLP            ~5M     does learned pooling help?
  L2 xformer       transformer + attention-pool    ~32M    does temporal modelling help?
                   (lives in mapper.py, built first)
  L3 qformer       k learned queries + MLP         ~35M    is one pooled vector the limit?

Why start below L2: L2 was written first out of expedience, and at 15k training pairs it
carries ~2,150 params per example. L0 beating it is a real possibility, and without the
lower rungs we cannot tell whether that capacity earns anything.

All variants share the same contract so the trainer never needs to know which is in use:
    forward(features, mask) -> (B, out_dim), L2-normalized
    null_condition(batch, device)          the unconditional vector for CFG
    apply_cfg_dropout(cond, p)             AudioLDM's condition dropout

Note on duplication: `mapper.py::SpeechToClapMapper` (L2) implements this contract
independently rather than subclassing `BaseConditioner`. That is deliberate -- changing its
module structure would change its state_dict keys and break loading the checkpoints already
on disk from the first run.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from salm.mapper import CLAP_EMBED_DIM, AttentionPool


class BaseConditioner(nn.Module):
    """Shared normalization, null embedding, and CFG dropout."""

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
        # unnormalized vector is out of distribution for the U-Net even if its direction
        # is right.
        return F.normalize(self.encode(features, mask), dim=-1)

    def null_condition(self, batch_size: int, device: torch.device) -> torch.Tensor:
        null = self.null_embedding.unsqueeze(0).expand(batch_size, -1).to(device)
        return F.normalize(null, dim=-1)

    def apply_cfg_dropout(self, cond: torch.Tensor, dropout_prob: float) -> torch.Tensor:
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

    Masking matters more than it looks: Whisper pads every input to 30 s, while TTAS speech
    averages 4.9 s, so an unmasked mean would dilute the utterance with ~80% silence.
    """
    if mask is None:
        return features.mean(dim=1)
    m = mask.unsqueeze(-1).to(features.dtype)
    return (features * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


class MeanPoolLinear(BaseConditioner):
    """L0: the minimal baseline -- masked mean-pool, one linear layer."""

    def __init__(self, input_dim: int, output_dim: int = CLAP_EMBED_DIM) -> None:
        super().__init__(output_dim)
        self.proj = nn.Linear(input_dim, output_dim)

    def encode(self, features: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        return self.proj(_masked_mean(features, mask))


class AttnPoolMLP(BaseConditioner):
    """L1: learned attention pooling + MLP head, but no temporal self-attention."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = CLAP_EMBED_DIM,
        hidden_dim: int = 1024,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(output_dim)
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


class QueryPoolMapper(BaseConditioner):
    """
    L3: k learned query tokens cross-attend to the frames, then are concatenated.

    Motivation: L0-L2 all squeeze everything through a single pooled vector. Multiple
    queries can each specialise (e.g. scene, action, intensity) before the final
    projection, which probes whether single-vector pooling is the limiting factor --
    without leaving AudioLDM's single-vector conditioning interface.
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
            # An all-masked row yields NaNs from attention; keep frame 0 alive.
            all_masked = key_padding_mask.all(dim=1)
            if all_masked.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked, 0] = False

        for attn in self.layers:
            delta, _ = attn(q, x, x, key_padding_mask=key_padding_mask)
            q = q + delta

        return self.head(q.flatten(1))
