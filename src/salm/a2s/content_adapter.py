"""
Rank 2 of the F1/F2 plan (2026-08-26b): install a gated cross-attention CONTENT stream onto
the pretrained AudioLDM U-Net, using diffusers' own IPAdapterAttnProcessor2_0 -- not
hand-rolled attention mechanics.

WHY THIS EXISTS. The oracle-conditioning ablation (2026-08-24) proved a pooled 512-d vector
cannot carry speech content, and AudioLDM 2's own paper independently confirms it: they use
pooling rate lambda=1 (512 tokens/10s, ~51.2 tok/s) for TTS specifically, vs lambda=8 (~6.4
tok/s) for general audio, because pooled features lose the detail speech needs. Our A2S
delivers ~0 tok/s. This module adds a genuine sequence channel.

WHY IPAdapterAttnProcessor2_0 AND NOT A CUSTOM MODULE. Verified against the installed
diffusers source (0.39.0), not assumed:

  - `cross_attention_dim == block_out_channels` exactly in our U-Net's config -- the
    fingerprint of `attn2` currently running as a SECOND SELF-ATTENTION layer (diffusers
    falls back to `encoder_hidden_states = hidden_states` when passed None). Its `to_k`/
    `to_v` are therefore calibrated to accept an input of the SAME width as the block's own
    latent features, not an external sequence -- reusing that pathway for a differently-
    shaped content sequence would require retraining pretrained weights from a mismatched
    starting point, destroying the "starts at current baseline" safety property.
  - `IPAdapterAttnProcessor2_0` is diffusers' own, battle-tested mechanism for EXACTLY this:
    a NEW, separate cross-attention branch (its own `to_k_ip`/`to_v_ip`, freshly initialized)
    added alongside the existing one, without touching pretrained weights at all.
  - Verified call-path: `UNet2DConditionModel.process_encoder_hidden_states()` is a no-op
    for our config (`encoder_hid_dim_type` is None, so every special-cased branch is
    skipped and the input passes through unchanged) -- so passing a raw tuple
    `encoder_hidden_states=(None, [content_seq])` reaches the attention layers untouched.
    Inside the processor, that tuple unpacks to `encoder_hidden_states=None` (restoring
    today's exact self-attention fallback on the PRIMARY branch, unchanged) plus
    `ip_hidden_states=[content_seq]` for the NEW branch.
  - Verified `Attention.set_processor()` does `self.processor = processor`, and
    `IPAdapterAttnProcessor2_0` IS an `nn.Module`, so standard PyTorch submodule
    registration means its parameters show up in `unet.parameters()` automatically -- no
    extra wiring needed for the optimizer to see them.

ZERO-INIT SAFETY. `to_k_ip`/`to_v_ip` are freshly initialized (not zero by default in
diffusers' own constructor) -- explicitly zero the VALUE projection here so the new branch's
attention-weighted output is exactly zero at init regardless of what the (randomly-
initialized) key/query similarity pattern looks like. The model is therefore bit-identical
to the current baseline at step 0, and can only move away from it via gradient signal that
actually helps -- the same zero-init discipline as O2's null embedding and O-F's design.

ONLY attn2 (cross-attention) modules get a new processor; attn1 (self-attention) keeps its
existing one completely untouched, via an explicit dict covering every attention layer
(`set_attn_processor` requires a processor for ALL of them if given a dict, verified from
source -- omitting attn1 would raise, not silently skip it).
"""

from __future__ import annotations

import torch
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0


def install_content_cross_attention(
    unet, content_dim: int, num_content_tokens: int | None = None
) -> dict:
    """
    Mutates `unet` in place: installs a new IP-Adapter-style cross-attention branch on every
    attn2 (cross-attention) module, leaves attn1 (self-attention) untouched.

    content_dim: width of the content sequence that will be passed at forward time (e.g. 512
        for our CTC probe's GRU output).
    num_content_tokens: unused by the mechanism itself (IPAdapterAttnProcessor2_0's Linear
        layers operate on the last dimension only, so sequence length is free at forward
        time) -- accepted for logging/documentation only, so a caller can record what length
        this was actually validated against.

    Returns a dict of {module_name: hidden_size} for the newly-adapted layers, for logging.
    """
    current = unet.attn_processors
    new_processors: dict = {}
    adapted: dict[str, int] = {}

    for name, proc in current.items():
        if ".attn2." not in name:
            # attn1 (self-attention): keep whatever processor is already there, unchanged.
            new_processors[name] = proc
            continue

        # Derive this block's hidden_size from the EXISTING processor's own attn2 module
        # rather than assuming a fixed down/up-block ordering -- safer than hardcoding
        # [128, 256, 384, 640] against a specific block index that could be wrong.
        module_path = name.rsplit(".processor", 1)[0]
        attn2_module = unet.get_submodule(module_path)
        hidden_size = attn2_module.to_q.out_features

        new_proc = IPAdapterAttnProcessor2_0(
            hidden_size=hidden_size,
            cross_attention_dim=content_dim,
            num_tokens=(1,),   # ONE content-sequence source, not a token-length parameter
            scale=1.0,
        )
        # Zero-init the VALUE projection -- see module docstring. Query/key stay randomly
        # initialized (they only shape attention WEIGHTS, not the contributed magnitude;
        # a zero value-projection makes the weighted sum zero regardless of the pattern).
        for v_proj in new_proc.to_v_ip:
            torch.nn.init.zeros_(v_proj.weight)

        new_processors[name] = new_proc
        adapted[module_path] = hidden_size

    unet.set_attn_processor(new_processors)
    return adapted


def content_encoder_hidden_states(content_seq: torch.Tensor):
    """
    Build the `encoder_hidden_states` argument for `unet(...)` that reaches ONLY the new
    content branch, leaving the primary (currently self-attention-fallback) branch exactly
    as it behaves today. See module docstring for why this specific tuple shape is required
    and verified safe against our U-Net's config.
    """
    return (None, [content_seq])
