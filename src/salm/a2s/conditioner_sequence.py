"""
SequentialContentHead -- let the CONDITIONER itself produce the content sequence, end to end,
with no oracle, no retrieval, and no separately-trained predictor.

WHY THIS EXISTS, and why it is the experiment the project actually needs. Three things are
now established:

  1. The frame-aligned concat channel WORKS when fed good content (2026-08-30, job 433314:
     oracle content -> near-verbatim speech; content_scale 1.5 -> dist 0.389 against a
     0.372 VAE-roundtrip ceiling).
  2. A POOLED conditioning vector cannot carry speech content, confirmed three independent
     ways (oracle ablation, AudioLDM 2's own lambda=1-for-speech, our own flat metrics).
  3. Predicting content by REGRESSION fails: MSE/cosine converges on E[content|ambient], the
     blurry mean, which carries no committed phonetic identity (2026-08-31: predicted
     content scored 0.94-0.96 vs plain retrieval's 0.50, at two capacity settings hitting an
     identical ~0.70 val_cos ceiling).

Retrieval sidesteps (3) by always returning a real example's content -- but then the words
are always borrowed verbatim from the training set, which is not a system that generates
speech, only one that re-voices past speech.

THE UNTESTED OPTION. XFormerConditioner already computes a per-frame representation
`(B, 32, hidden_dim)` after its transformer blocks, and then DISCARDS it in `self.pool`
purely because `class_labels` wanted one vector. This head taps that pre-pool sequence,
upsamples it to the content channel's 256-frame / 25 fps rate, and feeds the concat channel
directly. Trained end to end on the diffusion loss alone: the conditioner learns whatever
sequential conditioning actually helps generate the target, rather than matching a proxy.

WHY THIS SHOULD ESCAPE THE REGRESSION BLUR. The predictor was optimized to match a target
vector, and the minimizer of that objective is the mean. Here the objective is the diffusion
loss itself -- conditioning is rewarded for helping produce the RIGHT SPEECH, not for being
close-on-average to a content vector -- and the one-to-many nature of the task is absorbed
by the diffusion sampler, which is what a conditional generative model is supposed to do.
Deterministic conditioning plus stochastic sampling is the standard setup; we simply never
tried it with a sequence.

HONEST RISK. The gradient reaching the conditioner is weaker and more indirect than direct
supervision. The ORIGINAL A2S trained end to end this way and failed -- but through the
pooled bottleneck of (2), which is precisely the variable being changed. `val_content_bind_gap`
is the early-warning instrument: if it stays ~0 through ~epoch 5-10, this head is producing
timing/energy structure without phonetic identity, the same failure the cross-attention branch
had, and the run can be killed cheaply.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SequentialContentHead(nn.Module):
    """
    (B, ambient_frames, hidden_dim) pre-pool conditioner features
        -> (B, content_frames, content_dim) content sequence.

    Deliberately small: this is a resampling/projection head on top of representations the
    conditioner's transformer blocks already computed, not a second encoder. Capacity for
    understanding the scene lives in the conditioner, which is trained jointly.
    """

    def __init__(
        self,
        hidden_dim: int,
        content_dim: int,
        ambient_frames: int = 32,
        content_frames: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if content_frames % ambient_frames != 0:
            raise ValueError(
                f"content_frames ({content_frames}) must be an exact multiple of "
                f"ambient_frames ({ambient_frames})."
            )
        factor = content_frames // ambient_frames
        n_up = factor.bit_length() - 1
        if 2 ** n_up != factor:
            raise ValueError(f"upsample factor {factor} must be a power of 2 (32 -> 256 is 2**3).")

        blocks = []
        for _ in range(n_up):
            blocks += [
                nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(num_groups=8, num_channels=hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        self.upsample = nn.Sequential(*blocks)
        self.out_proj = nn.Linear(hidden_dim, content_dim)
        self.ambient_frames = ambient_frames
        self.content_frames = content_frames

    def forward(self, pre_pool: torch.Tensor) -> torch.Tensor:
        # Fail loudly here rather than silently producing a content sequence of the wrong
        # length: the concat channel needs EXACTLY content_frames to align 1:1 with the
        # latent's 256 time steps, and a mismatch would otherwise surface as an opaque
        # broadcast error deep inside the U-Net.
        if pre_pool.shape[1] != self.ambient_frames:
            raise RuntimeError(
                f"SequentialContentHead built for {self.ambient_frames} ambient frames but "
                f"got {pre_pool.shape[1]}; the upsample stack is a fixed power-of-2 ladder."
            )
        x = pre_pool.transpose(1, 2)      # (B, H, ambient_frames)
        x = self.upsample(x)              # (B, H, content_frames)
        x = x.transpose(1, 2)             # (B, content_frames, H)
        return self.out_proj(x)           # (B, content_frames, content_dim)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
