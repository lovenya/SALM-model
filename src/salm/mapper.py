"""
The mapper: frozen speech features -> AudioLDM's conditioning vector.

AudioLDM conditions its U-Net on a SINGLE global 512-d CLAP embedding (passed as
`class_labels`, not a cross-attention token sequence). So the mapper's whole job is
to turn a variable-length frame sequence into one 512-d vector that lands in CLAP's
shared audio/text space.

That is a small target, which is the point: with ~15k training pairs, a compact
mapper is what the data can actually support. The pretrained U-Net already knows how
to turn a CLAP vector into plausible ambient audio -- we only need to predict the
right vector.

Design choices worth knowing:
  - Attention pooling, not mean pooling. Whisper pads every input to 30 s, so most
    frames are silence; mean pooling would dilute a 5 s utterance with 25 s of
    nothing. A learned query attends to the frames that matter, and the padding mask
    hard-excludes the rest.
  - Output is L2-normalized, because AudioLDM's pipeline L2-normalizes CLAP
    embeddings before conditioning. Matching that norm matters -- an unnormalized
    vector is out of distribution for the U-Net even if its direction is right.
  - A learned null embedding provides the unconditional branch for classifier-free
    guidance, replacing the condition on ~10% of training steps.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# AudioLDM's CLAP projection width. Verified at runtime against
# unet.config.projection_class_embeddings_input_dim.
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
            # A row that is entirely masked produces NaNs, so keep frame 0 alive.
            all_masked = key_padding_mask.all(dim=1)
            if all_masked.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked, 0] = False

        pooled, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask)
        return pooled.squeeze(1)


class SpeechToClapMapper(nn.Module):
    """
    Frozen speech features (B, T, D_in) -> L2-normalized 512-d CLAP-space vector.

    Optionally runs a few self-attention layers over the frame sequence before
    pooling, so the mapper can do some temporal reasoning rather than pooling raw
    encoder states.
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
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

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

        # Unconditional embedding for classifier-free guidance.
        self.null_embedding = nn.Parameter(torch.zeros(output_dim))

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        features: (B, T, input_dim) from the frozen encoder
        mask:     (B, T) bool, True = real frame
        returns:  (B, output_dim), L2-normalized
        """
        x = self.in_proj(features)

        src_key_padding_mask = None if mask is None else ~mask
        for block in self.blocks:
            x = block(x, src_key_padding_mask=src_key_padding_mask)

        pooled = self.pool(x, mask=mask)
        out = self.out(pooled)
        return F.normalize(out, dim=-1)

    def null_condition(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """The unconditional CLAP vector, L2-normalized to match real conditions."""
        null = self.null_embedding.unsqueeze(0).expand(batch_size, -1)
        return F.normalize(null, dim=-1)

    def apply_cfg_dropout(
        self, cond: torch.Tensor, dropout_prob: float, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """
        Replace whole conditions with the null embedding at `dropout_prob`, which is how
        AudioLDM trains the unconditional branch needed for classifier-free guidance.
        """
        if dropout_prob <= 0.0:
            return cond

        b = cond.shape[0]
        keep = torch.rand(b, device=cond.device, generator=generator) >= dropout_prob
        null = self.null_condition(b, cond.device)
        return torch.where(keep.unsqueeze(1), cond, null)

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
