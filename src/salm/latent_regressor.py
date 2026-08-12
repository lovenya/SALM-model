"""
T0 -- the simplest defensible baseline: a transformer trained from scratch that regresses the
target VAE latent directly. No diffusion, no noise schedule, one forward pass.

WHY THIS EXISTS
---------------
The supervising PhD's point is that reviewers will ask why architectural complexity was
introduced, and the only answer that survives is a measured gap against something simpler. A
transformer generating raw audio from scratch on 42 h would produce noise, so "simple transformer
from scratch" needs a sane reading. This is it: everything the frozen AudioLDM stack already gives
us is kept (the mel VAE and HiFi-GAN vocoder, both frozen and both measured healthy -- VAE
round-trip 0.9993), and the ONLY learned component is a small transformer mapping frozen Whisper
features to the latent.

  frozen Whisper encoder -> [this module] -> latent (8, 256, 16) -> frozen VAE -> frozen vocoder

That makes it a genuine ablation of the diffusion process itself, holding the encoder, the
generator's decoder and the data fixed. It is also nearly free: the latents are precomputed, so an
epoch is a forward pass and an MSE.

WHAT WE EXPECT, AND ON WHAT GROUNDS
-----------------------------------
It should underperform, and the reason to state is NOT "the task is one-to-many". Our dataset
pairing is literally 1:1 -- 15,033 speech clips, 15,033 distinct targets -- so nothing in the
training data forces averaging, and a high-capacity regressor could in principle memorise it.

The measured reason is that interpolation in this latent space is off-distribution. On our own
train latents (see docs/EXPERIMENT_LOG.md):

    single real latent      std 1.032
    blend of just 2         std 0.753   (0.73x)
    blend of 16             std 0.421   (0.41x)

AudioLDM's decoder expects std ~1.0 (its 0.9393 scaling factor is calibrated for it; our latents
measure 0.977 after scaling). A regressor must interpolate on held-out inputs, and an interpolated
latent decodes quieter and less textured. Diffusion avoids this by SAMPLING -- it lands on the
manifold at std ~1 rather than between points at 0.73.

So this module logs `pred_std` as a first-class metric: it converts that prediction into a number.
If predicted latents come out at std ~0.5 while real ones are ~1.0, the collapse is measured rather
than asserted, and that single number motivates the entire diffusion approach.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LatentRegressor(nn.Module):
    """
    Frozen speech features -> the full target latent, in one shot.

    Predicts per latent TIME STEP rather than as one flat vector. A flat head from a pooled
    vector to 8*256*16 = 32,768 outputs would be a single 1280x32768 linear (42 M params) with no
    temporal structure at all, which is both wasteful and unable to vary along time. Instead each
    of the 256 latent frames gets a learned query that cross-attends to the Whisper frames and
    emits its own 8*16 = 128 values, so temporal structure is expressible and the head is ~9 M.

    Args:
        input_dim: frozen encoder feature width (Whisper-large-v3: 1280).
        latent_shape: (channels, time, freq) of the target latent, e.g. (8, 256, 16).
        d_model / n_heads / num_layers / dropout: the transformer.
    """

    def __init__(
        self,
        input_dim: int,
        latent_shape: tuple[int, int, int],
        d_model: int = 512,
        n_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_shape = tuple(latent_shape)
        c, t, f = self.latent_shape
        self.out_per_step = c * f

        self.in_proj = nn.Linear(input_dim, d_model)
        # One query per output latent frame. This is the whole reason the head is small.
        self.queries = nn.Parameter(torch.randn(t, d_model) * 0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,      # pre-norm: trains more stably from scratch
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, self.out_per_step)

        # Zero-init the output head so the model starts by predicting exactly 0 rather than
        # random garbage. At step 0 the loss is then simply E[x0^2] = the latent variance, a known
        # quantity (~1.07), which makes "is it learning at all?" readable from the first epoch
        # instead of being hidden behind a random initial scale.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        features: (B, T_enc, input_dim) frozen encoder output
        mask:     (B, T_enc) True where VALID (our encoders' convention)
        returns:  (B, C, T_lat, F_lat)
        """
        b = features.shape[0]
        mem = self.in_proj(features)
        # nn.Transformer wants True where a position should be IGNORED, which is the inverse of
        # our encoders' valid-mask convention. Getting this backwards would silently attend to
        # padding only -- Whisper pads to 30 s while TTAS speech averages 4.9 s, so the padded
        # region is the MAJORITY of frames and the error would not be subtle.
        mem_key_padding_mask = (~mask.bool()) if mask is not None else None

        q = self.queries.unsqueeze(0).expand(b, -1, -1)
        h = self.decoder(q, mem, memory_key_padding_mask=mem_key_padding_mask)
        out = self.out_proj(self.norm(h))                     # (B, T_lat, C*F)

        c, t, f = self.latent_shape
        # (B, T, C*F) -> (B, C, T, F). Order matters: the head emits C*F per step, so view as
        # (C, F) then move C ahead of T to match the VAE's (B, C, T, F) expectation.
        return out.view(b, t, c, f).permute(0, 2, 1, 3).contiguous()


def build_latent_regressor(
    input_dim: int, latent_shape: tuple[int, int, int], cfg: dict
) -> LatentRegressor:
    def get(key, default):
        return cfg.get(key, default) if isinstance(cfg, dict) else getattr(cfg, key, default)

    return LatentRegressor(
        input_dim=input_dim,
        latent_shape=latent_shape,
        d_model=get("regress_d_model", 512),
        n_heads=get("regress_n_heads", 8),
        num_layers=get("regress_layers", 2),
        dropout=get("regress_dropout", 0.1),
    )
