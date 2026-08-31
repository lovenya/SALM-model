"""
AmbientContentPredictor -- the F1 piece. Predicts a content sequence FROM AMBIENT AUDIO
ALONE, trained to match the (target-speech-derived) content sequence that the renderer was
trained and proven to work with (2026-08-30, job 433314: oracle content -> near-verbatim
speech; retrieval content -> fluent but wrong-topic speech).

WHY THIS EXISTS. Retrieval (nearest-training-scene lookup) is the current honest, deployable
content source, and it works (fluent English), but it can only ever hand back something a
training example already said verbatim -- it cannot generalize. This module replaces the
lookup with a TRAINED regression: ambient CLAP sequence -> predicted content sequence,
learned to approximate the true content sequence as closely as possible.

NOT CHEATING, same reasoning as O2's alignment loss. The true (target-speech-derived) content
sequence is used ONLY as the training-time answer key for a regression loss -- exactly how O2
already uses the target's CLAP-text embedding to train the pooled conditioner. At inference,
this module's forward pass takes ONLY ambient audio features; the target is never touched.

HONEST CEILING. Ambient audio does not determine exact words (the F1 finding, unchanged by
this module) -- the ridge probe measured topic-level signal at R@1@16~0.45, not verbatim. This
predictor should generalize better than nearest-neighbour retrieval (it can blend patterns
across many training examples instead of snapping to the single closest one), but it will not,
and should not be expected to, reach oracle's near-verbatim accuracy -- no amount of capacity
recovers information that was never in the ambient input to begin with.

ARCHITECTURE. Ambient CLAP sequence is 32 frames over the clip; the content sequence the
renderer expects is 256 frames (25 fps, matching the VAE latent's own time axis exactly -- see
content_concat.py). So this is a small encoder-then-upsample model, not a like-for-like
seq2seq: a shallow Transformer lets the 32 ambient frames exchange context, then a strided
ConvTranspose1d stack upsamples x8 (32->64->128->256) to the content sequence's frame rate.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AmbientContentPredictor(nn.Module):
    def __init__(
        self,
        ambient_dim: int = 768,
        content_dim: int = 512,
        ambient_frames: int = 32,
        content_frames: int = 256,
        hidden_dim: int = 512,
        num_encoder_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if content_frames % ambient_frames != 0:
            raise ValueError(
                f"content_frames ({content_frames}) must be an exact multiple of "
                f"ambient_frames ({ambient_frames}) for the ConvTranspose1d stack's stride-2 "
                "layers to land exactly -- got a non-integer or non-power-of-2 upsample factor."
            )
        upsample_factor = content_frames // ambient_frames
        n_up = upsample_factor.bit_length() - 1
        if 2 ** n_up != upsample_factor:
            raise ValueError(
                f"upsample factor {upsample_factor} is not a power of 2 -- the stride-2 "
                "ConvTranspose1d stack below assumes it is (32 -> 256 is 2**3, the case this "
                "was written for)."
            )

        self.in_proj = nn.Linear(ambient_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, ambient_frames, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Upsample 32 -> 256 along time: three stride-2 ConvTranspose1d layers, operating on
        # (B, C, T) as Conv1d expects.
        up_blocks = []
        for _ in range(n_up):
            up_blocks += [
                nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(num_groups=8, num_channels=hidden_dim),
                nn.GELU(),
            ]
        self.upsample = nn.Sequential(*up_blocks)
        self.out_proj = nn.Linear(hidden_dim, content_dim)

    def forward(self, ambient_seq: torch.Tensor, ambient_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        ambient_seq: (B, ambient_frames, ambient_dim)
        ambient_mask: (B, ambient_frames) bool, True = real frame. Optional -- CLAP's fixed-
            length sequence has no padding in practice, but this is honored if given.
        Returns: (B, content_frames, content_dim), the predicted content sequence.
        """
        x = self.in_proj(ambient_seq) + self.pos_embed
        key_padding_mask = None
        if ambient_mask is not None:
            key_padding_mask = ~ambient_mask  # TransformerEncoder wants True = IGNORE
            if key_padding_mask.all(dim=1).any():
                # A fully-masked row would make attention undefined (softmax over -inf).
                # Should never happen for a real clip; degrading to "unmasked" for that row
                # is safer than emitting NaNs into the rest of the batch.
                key_padding_mask = key_padding_mask & ~key_padding_mask.all(dim=1, keepdim=True)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)   # (B, 32, H)
        x = x.transpose(1, 2)                                         # (B, H, 32)
        x = self.upsample(x)                                          # (B, H, 256)
        x = x.transpose(1, 2)                                         # (B, 256, H)
        return self.out_proj(x)                                       # (B, 256, content_dim)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def content_prediction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    MSE + (1 - cosine similarity), both averaged over frames and batch. MSE alone can be
    minimized by shrinking toward zero on frames the model is unsure about; adding cosine
    keeps DIRECTION honest even when magnitude is uncertain -- the standard fix for exactly
    this failure mode (the same reasoning that motivated InfoNCE over plain cosine earlier
    in this project, applied here to a regression rather than a contrastive setting).
    """
    mse = torch.nn.functional.mse_loss(pred, target)
    cos = torch.nn.functional.cosine_similarity(pred, target, dim=-1).mean()
    return mse + (1.0 - cos)
