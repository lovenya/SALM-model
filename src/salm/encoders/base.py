"""
Frozen conditioning-encoder interface.

Every candidate encoder is frozen, so the only thing training cares about is the shape
and dtype of the features it emits. Keeping that behind one interface is what lets us
swap Whisper <-> CLAP-audio <-> Qwen2.5-Omni without touching the training loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class FrozenEncoder(ABC, torch.nn.Module):
    """
    Maps a batch of 16 kHz mono speech waveforms to frame-level features.

    Implementations must:
      - load weights from a LOCAL path (compute nodes have no internet),
      - set eval mode and disable grads for all parameters,
      - return (B, T_feat, D) float tensors.
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """D in the (B, T_feat, D) output -- the conditioner's input width."""

    @abstractmethod
    def forward(self, wav_16k: torch.Tensor) -> torch.Tensor:
        """
        wav_16k: (B, samples) float32, mono, 16 kHz.
        returns: (B, T_feat, D)
        """

    def freeze(self) -> "FrozenEncoder":
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        return self
