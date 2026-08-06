"""
Frozen Whisper encoder as the speech-understanding front-end.

This uses ONLY `WhisperModel.encoder`. The decoder is never constructed and no text is
ever produced -- that distinction is the point. Routing through a transcript is exactly
what the project's cascaded baseline does, so an end-to-end model must not, or the
comparison is meaningless.

Whisper always pads its input to a fixed 30 s mel window, giving 1500 encoder frames
regardless of the true utterance length. TTAS speech averages ~4.9 s, so most of those
frames are padding. We return the whole sequence plus a validity mask so the mapper can
attention-pool over real content instead of averaging in silence.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import WhisperFeatureExtractor, WhisperModel

from salm.encoders.base import FrozenEncoder

# Local weights -- no download needed (see CONTEXT.md asset audit).
DEFAULT_WHISPER_PATH = (
    "/scratch/l/lovenya/salm_migration_clean/models/cascaded/asr/whisper-large-v3"
)

WHISPER_SR = 16000
WHISPER_WINDOW_SECONDS = 30.0
WHISPER_ENCODER_FRAMES = 1500


class WhisperSpeechEncoder(FrozenEncoder):
    def __init__(
        self,
        model_path: str | Path = DEFAULT_WHISPER_PATH,
        dtype: torch.dtype = torch.float16,
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(str(model_path))

        # Load the full model then keep only the encoder, so the decoder weights are
        # released rather than sitting in VRAM unused.
        full = WhisperModel.from_pretrained(str(model_path), torch_dtype=dtype)
        self.encoder = full.get_encoder().to(self.device)
        del full

        self._feature_dim = int(self.encoder.config.d_model)
        self.freeze()

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def frames_for_seconds(self, seconds: float) -> int:
        """How many of the 1500 encoder frames correspond to `seconds` of real audio."""
        frac = min(seconds / WHISPER_WINDOW_SECONDS, 1.0)
        return max(1, int(round(frac * WHISPER_ENCODER_FRAMES)))

    @torch.no_grad()
    def forward(self, wav_16k: torch.Tensor) -> torch.Tensor:
        """wav_16k: (B, samples) float32 @16k -> (B, 1500, d_model)"""
        wavs = [w.detach().float().cpu().numpy() for w in wav_16k]

        inputs = self.feature_extractor(
            wavs,
            sampling_rate=WHISPER_SR,
            return_tensors="pt",
        )
        features = inputs.input_features.to(self.device, dtype=self.dtype)
        out = self.encoder(features).last_hidden_state
        return out.float()

    @torch.no_grad()
    def encode_with_mask(
        self, wav_16k: torch.Tensor, lengths_samples: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Same as forward, plus a (B, 1500) bool mask marking frames backed by real audio
        rather than Whisper's 30 s zero padding.
        """
        feats = self.forward(wav_16k)
        b, t, _ = feats.shape

        valid_frames = torch.tensor(
            [self.frames_for_seconds(float(n) / WHISPER_SR) for n in lengths_samples],
            device=feats.device,
        )
        idx = torch.arange(t, device=feats.device).unsqueeze(0).expand(b, t)
        mask = idx < valid_frames.unsqueeze(1)
        return feats, mask
