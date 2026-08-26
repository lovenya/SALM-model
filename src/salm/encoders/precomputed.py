"""
Stand-in for a frozen encoder whose features are already on disk.

The A2S conditioning features are precomputed (see scripts/a2s/precompute_a2s_conditioning.py),
so no encoder runs inside the training loop at all. The trainer still needs *something* that
reports `feature_dim`, because the conditioner is built to match the encoder's width -- that is
this class's whole job.

Why precompute for A2S when S2A deliberately did not: S2A kept raw speech in the loop so the
encoder and its pooling stayed swappable. That reasoning does not carry over. CLAP's frame
sequence is (32, 768) per clip, so the full A2S train split is ~0.9 GB -- small enough to cache,
unlike Whisper's 1500 frames -- and the conditioner still owns the pooling, so nothing about the
ladder is frozen into the cache. Only the *encoder choice* is, and swapping it means one cheap
re-run of the precompute script. In exchange the training loop loses a frozen forward pass and
the CPU-bound decode/resample behind it, which is what makes a wide A2S sweep affordable.

This is the same mechanism `docs/OPEN_ITEMS.md` item 7 ("Precompute frozen Whisper features")
will need for S2A, so it is shared infrastructure rather than an A2S special case.
"""

from __future__ import annotations

import torch

from salm.encoders.base import FrozenEncoder


class PrecomputedSequenceEncoder(FrozenEncoder):
    """Reports the cached features' width; refuses to encode."""

    def __init__(self, feature_dim: int, source: str = "precomputed") -> None:
        super().__init__()
        self._feature_dim = int(feature_dim)
        self.source = source
        self.freeze()

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def _refuse(self):
        raise RuntimeError(
            f"{type(self).__name__} holds no weights -- features come from {self.source}. "
            "Reaching here means the training loop tried to encode a waveform on a run "
            "configured for precomputed features, which would silently train on garbage."
        )

    # Both take *args so the guard message survives either call signature. Delegating
    # encode_with_mask to a one-argument forward() raised an arity TypeError instead, which
    # is exactly the confusing failure this class exists to prevent.
    def forward(self, *args, **kwargs) -> torch.Tensor:  # pragma: no cover - guard
        self._refuse()

    def encode_with_mask(self, *args, **kwargs):  # pragma: no cover - guard
        self._refuse()
