"""
Whisper-large-v3 ASR wrapper, shared between the A2S gate (scripts/a2s/check_speech_stack.py)
and the generated-audio WER scorer (scripts/a2s/score_generated_wer.py).

Uses the WITH-decoder Whisper model, unlike salm.encoders.whisper_encoder.WhisperSpeechEncoder
(the frozen ENCODER-only model used for S2A conditioning, which never produces text) -- two
different jobs for the same base model, so kept as two separate wrappers rather than one
class trying to serve both.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from salm.data.frontend import AUDIOLDM_SR


class WhisperASR:
    """Whisper-large-v3 with its decoder -- an independent judge, not part of any training loop."""

    def __init__(self, model_path: str | Path, device: str) -> None:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.device = device
        self.processor = WhisperProcessor.from_pretrained(str(model_path))
        self.model = (
            WhisperForConditionalGeneration.from_pretrained(
                str(model_path), dtype=torch.float16
            )
            .to(device)
            .eval()
        )

    @torch.no_grad()
    def transcribe(self, wavs: list[np.ndarray], batch_size: int = 8) -> list[str]:
        out: list[str] = []
        for i in range(0, len(wavs), batch_size):
            chunk = wavs[i : i + batch_size]
            inputs = self.processor(
                chunk, sampling_rate=AUDIOLDM_SR, return_tensors="pt"
            )
            feats = inputs.input_features.to(self.device, dtype=torch.float16)
            ids = self.model.generate(
                feats, language="en", task="transcribe", max_new_tokens=180
            )
            out.extend(self.processor.batch_decode(ids, skip_special_tokens=True))
        return [s.strip() for s in out]
