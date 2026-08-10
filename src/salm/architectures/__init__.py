"""
Architecture registry: name -> (frozen encoder, trainable conditioner).

The point of this indirection is that adding a rung to the ladder means adding an entry
here, not editing the training loop. `train_ldm.py` reads `arch: <name>` from config and
asks for components; it never imports a specific conditioner.

Naming: <direction>_<generator>_<encoder>_<conditioner>, e.g. s2a_audioldm_whisper_xformer.
That keeps run directories self-describing, since runs are grouped as runs/<arch>/<run>/.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from salm.conditioners import AttnPoolMLP, MeanPoolLinear, QueryPoolMapper
from salm.mapper import CLAP_EMBED_DIM, SpeechToClapMapper


@dataclass
class Components:
    encoder: object          # FrozenEncoder
    conditioner: torch.nn.Module


def _build_whisper_encoder(cfg: dict, device: torch.device):
    from salm.encoders.whisper_encoder import WhisperSpeechEncoder
    from configs import paths as project_paths

    return WhisperSpeechEncoder(
        cfg.get("whisper_path", str(project_paths.WHISPER_LARGE_V3_PATH)),
        dtype=torch.float16,
        device=device,
    )


def _build_qwen_encoder(cfg: dict, device: torch.device):
    raise NotImplementedError(
        "Qwen2.5-Omni encoder (ladder rung L4) is not wired yet. Weights are local at "
        "configs.paths.QWEN25_OMNI_3B_PATH; mirror the Audio-Omni pattern in "
        "code/Audio-Omni/audio_omni/models/conditioners.py (thinker hidden states at "
        "layer_idx=-2 + zero-init linear), and watch the float/bfloat16 bias dtype bug."
    )


ENCODERS: dict[str, Callable] = {
    "whisper": _build_whisper_encoder,
    "qwen_omni": _build_qwen_encoder,
}


def _cond_mean_linear(input_dim: int, output_dim: int, cfg: dict):
    return MeanPoolLinear(input_dim, output_dim)


def _cond_attn_mlp(input_dim: int, output_dim: int, cfg: dict):
    return AttnPoolMLP(
        input_dim,
        output_dim,
        hidden_dim=cfg.get("mapper_hidden_dim", 1024),
        dropout=cfg.get("mapper_dropout", 0.1),
    )


def _cond_xformer(input_dim: int, output_dim: int, cfg: dict):
    return SpeechToClapMapper(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=cfg.get("mapper_hidden_dim", 1024),
        num_layers=cfg.get("mapper_layers", 2),
        dropout=cfg.get("mapper_dropout", 0.1),
    )


def _cond_qformer(input_dim: int, output_dim: int, cfg: dict):
    return QueryPoolMapper(
        input_dim,
        output_dim,
        hidden_dim=cfg.get("mapper_hidden_dim", 1024),
        num_queries=cfg.get("mapper_num_queries", 8),
        num_layers=cfg.get("mapper_layers", 2),
        dropout=cfg.get("mapper_dropout", 0.1),
    )


CONDITIONERS: dict[str, Callable] = {
    "mean_linear": _cond_mean_linear,   # L0
    "attn_mlp": _cond_attn_mlp,         # L1
    "xformer": _cond_xformer,           # L2 (the first run)
    "qformer": _cond_qformer,           # L3
}

# Ladder order, for reporting and for sanity-checking comparisons.
LADDER = ["mean_linear", "attn_mlp", "xformer", "qformer"]


def parse_arch(arch: str) -> tuple[str, str]:
    """
    's2a_audioldm_whisper_xformer' -> ('whisper', 'xformer').

    Kept strict on purpose: a typo in an arch name should fail at startup with a list of
    valid options, not silently fall back to a default and quietly invalidate a run.
    """
    parts = arch.split("_")
    encoder = next((p for p in parts if p in ENCODERS), None)
    conditioner = next(
        (c for c in CONDITIONERS if arch.endswith(c) or f"_{c}_" in f"_{arch}_"), None
    )
    if encoder is None or conditioner is None:
        raise ValueError(
            f"cannot parse arch {arch!r}.\n"
            f"  encoders:     {sorted(ENCODERS)}\n"
            f"  conditioners: {sorted(CONDITIONERS)}\n"
            f"  expected form: <direction>_<generator>_<encoder>_<conditioner>"
        )
    return encoder, conditioner


def build(arch: str, cfg: dict, device: torch.device, output_dim: int = CLAP_EMBED_DIM):
    """Build the frozen encoder and trainable conditioner named by `arch`."""
    encoder_name, conditioner_name = parse_arch(arch)
    encoder = ENCODERS[encoder_name](cfg, device)
    conditioner = CONDITIONERS[conditioner_name](
        encoder.feature_dim, output_dim, cfg
    ).to(device)
    return Components(encoder=encoder, conditioner=conditioner)
