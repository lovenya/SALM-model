"""
Frame-aligned CHANNEL CONCATENATION of the content sequence into the U-Net's input.

WHY THIS EXISTS -- the measurement that killed cross-attention. Job 432420 (2026-08-30)
generated 4 content sources x 6 checkpoints under matched rows/seeds/settings, with the
CLAP encoder bug fixed and parity-verified, from a properly-trained epoch-99 checkpoint.
Result, by ear: ORACLE content (the target speech's own encoding -- perfect, and exactly
what training saw) and SHUFFLED content (a different utterance's -- deliberately wrong)
produce essentially the SAME audio. Only `zero` (no content at all) differs.

So the IP-Adapter cross-attention branch registers content PRESENCE but does not bind
content IDENTITY. It supplies an utterance-level hint -- energy, timing, where speech sits
in the canvas -- which is why it genuinely improved val_x0_rel_mse (0.371 -> 0.330) while
never making a single word intelligible.

WHY CONCATENATION SHOULD BIND WHERE ATTENTION DID NOT. Modern speech generation does not
cross-attend to content; it concatenates a FRAME-ALIGNED content sequence onto the noisy
input along the channel axis:
  - F5-TTS, verified from local source (F5-TTS/src/f5_tts/model/backbones/dit.py:162):
        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
  - E2-TTS does the same without F5's ConvNeXt text refinement.
  - VoiceBox uses frame-aligned phones; VoiceLDM upsamples text to frame rate with a durator.
Cross-attention makes the model LEARN a soft alignment over 256 keys. Concatenation hands
it the alignment as a hard architectural fact: content frame t is wired to latent frame t,
and cannot be averaged away across the utterance.

WE ARE IN THE BEST CASE FOR THIS, FOR FREE. AudioLDM's mel is 1024 frames over 10.24 s
(100 fps); the VAE downsamples time 4x, so the latent (8, 256, 16) runs at exactly 25 fps.
`SimpleCTCASR.encode()` returns exactly 256 frames. Content frame t and latent frame t are
the same 40 ms of the same waveform -- a 1:1 match, with no durator, no forced aligner, and
no filler-token padding. F5 has to fake this with left-packed filler tokens; we have it.

ZERO-INIT SAFETY, same discipline as the adapter it replaces. `conv_in` is widened from 8
to 8+extra_channels input channels: the pretrained weights are copied into [:, :8] verbatim
and the NEW channels are zero-initialized, so at step 0 the model is bit-identical to the
current A2S baseline and can only diverge via gradient that helps. This is the same trick
Stable Diffusion's inpainting checkpoint used to widen conv_in from 4 to 9 channels.

COST. The projector plus the new conv_in slice is ~10-45k trainable params, against the
~1-3M spent on to_k_ip/to_v_ip for the cross-attention branch that did not work.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ContentProjector(nn.Module):
    """
    (B, T=256, D) content sequence -> (B, extra_channels, T=256, F=16), latent-shaped.

    `mode` controls the frequency axis, and the choice is a real inductive-bias decision:

      "broadcast" (default): project to `extra_channels` per timestep and repeat across all
          16 frequency bins. Says "this content is present at time t, across the spectrum",
          which is what F5's channel-concat actually communicates -- content is a property
          of a TIME, not of a time-frequency cell. ~D*extra_channels params (tiny).

      "full": project to `extra_channels * n_freq` and reshape, letting the content write a
          different value per frequency bin. Strictly more expressive, ~16x the params, and
          free to overfit structure that content does not really carry.
    """

    def __init__(
        self,
        content_dim: int,
        extra_channels: int = 8,
        n_freq: int = 16,
        mode: str = "broadcast",
    ) -> None:
        super().__init__()
        if mode not in ("broadcast", "full"):
            raise ValueError(f"mode must be 'broadcast' or 'full', got {mode!r}")
        self.mode = mode
        self.extra_channels = extra_channels
        self.n_freq = n_freq
        out = extra_channels if mode == "broadcast" else extra_channels * n_freq
        self.proj = nn.Linear(content_dim, out)

    def forward(self, content_seq: torch.Tensor) -> torch.Tensor:
        b, t, _ = content_seq.shape
        x = self.proj(content_seq)                                  # (B, T, out)
        if self.mode == "broadcast":
            x = x.unsqueeze(-1).expand(b, t, self.extra_channels, self.n_freq)
            x = x.permute(0, 2, 1, 3)                               # (B, C, T, F)
        else:
            x = x.view(b, t, self.extra_channels, self.n_freq).permute(0, 2, 1, 3)
        return x.contiguous()


def widen_conv_in(unet, extra_channels: int) -> nn.Conv2d:
    """
    Replace `unet.conv_in` with one accepting `8 + extra_channels` input channels.

    Pretrained weights are copied into the first 8 input channels and the bias is kept
    verbatim; the new channels are ZERO-initialized, so the widened conv is numerically
    identical to the original for any input whose extra channels are ignored at init.
    Mutates `unet` in place and returns the new module.

    Idempotent guard: refuses to widen twice, which would silently discard the pretrained
    slice on a second call (e.g. an accidental double-install before a resume).
    """
    old = unet.conv_in
    if old.in_channels != 8:
        raise RuntimeError(
            f"conv_in already has {old.in_channels} input channels, expected the pretrained 8 "
            "-- refusing to widen twice (that would discard the pretrained weights)."
        )
    new = nn.Conv2d(
        8 + extra_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=old.bias is not None,
    )
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :8].copy_(old.weight)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    new = new.to(device=old.weight.device, dtype=old.weight.dtype)
    unet.conv_in = new
    return new


def concat_content(latent: torch.Tensor, content_map: torch.Tensor) -> torch.Tensor:
    """Concatenate along the CHANNEL axis. Shapes must agree on (B, T, F) exactly."""
    if latent.shape[0] != content_map.shape[0] or latent.shape[2:] != content_map.shape[2:]:
        raise RuntimeError(
            f"content map {tuple(content_map.shape)} does not align with latent "
            f"{tuple(latent.shape)} on (B, T, F) -- the 1:1 frame alignment is the whole "
            "point of this module, so this is a hard error rather than an interpolation."
        )
    return torch.cat([latent, content_map], dim=1)
