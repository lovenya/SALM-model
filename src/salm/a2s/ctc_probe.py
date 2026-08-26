"""
A small, from-scratch CTC ASR model over AudioLDM's own mel frontend.

WHY THIS EXISTS. No off-the-shelf ASR model's input matches AudioLDM's mel config (16 kHz,
64 mel bins, hop 160) -- common pretrained CTC/ASR frontends use 80 bins or raw waveform. A
model trained on a mismatched frontend cannot be trusted to score mel content faithfully, so
this trains a small one directly on the SAME `MelExtractor(norm_mode="none")` output the VAE
itself encodes/decodes. That is also the entire point: once trained, this becomes the frozen
content-loss head for O-F (`train_ldm.py`'s `lambda_ctc` term) -- it scores the DECODED mel
directly, with no vocoder in the loop and no dependence on any conditioning embedding, which
sidesteps the CLAP-text ceiling the 2026-08-24 oracle-conditioning ablation confirmed:
oracle-perfect CLAP-text conditioning still produced incomprehensible speech, by ear, so
supervision has to reach the generation path some other way. See docs/EXPERIMENT_LOG.md.

Architecture: 2 strided conv2d blocks (4x downsample in both time and frequency) -> flatten
channels+freq per timestep -> 2-layer bidirectional GRU -> linear to CTC_VOCAB_SIZE, trained
with `torch.nn.CTCLoss(blank=CTC_BLANK)`. Deliberately small and standard (a DeepSpeech2-style
frontend) -- this only needs to be "good enough to provide a real content gradient", not
state-of-the-art ASR, and a bigger model would cost more to train for no clear benefit here.

At AudioLDM's canvas (1024 mel frames = 10.24 s), the conv stack outputs a FIXED 256 frames
regardless of input content, verified against the real shapes before this was written --
so `input_lengths` for CTCLoss is a constant per batch, not something to infer from padding.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from salm.a2s.wer import CTC_BLANK, CTC_VOCAB_SIZE

MEL_FRAMES_IN = 1024      # AudioLDM's 10.24 s canvas at hop 160 / 16 kHz
MEL_BINS_IN = 64
CTC_DOWNSAMPLE = 4        # two stride-2 conv blocks
CTC_FRAMES_OUT = MEL_FRAMES_IN // CTC_DOWNSAMPLE   # 256, fixed -- see module docstring


class SimpleCTCASR(nn.Module):
    def __init__(
        self,
        mel_bins: int = MEL_BINS_IN,
        conv_channels: int = 32,
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
        vocab_size: int = CTC_VOCAB_SIZE,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(inplace=True),
        )
        freq_out = mel_bins // 4
        rnn_in = conv_channels * freq_out
        self.rnn = nn.GRU(
            rnn_in, rnn_hidden, num_layers=rnn_layers, batch_first=True,
            bidirectional=True, dropout=dropout if rnn_layers > 1 else 0.0,
        )
        self.head = nn.Linear(rnn_hidden * 2, vocab_size)
        self.encoder_dim = rnn_hidden * 2   # bidirectional GRU output width

    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        """
        The conv+GRU trunk's output, BEFORE the final linear-to-vocab head.

        2026-08-26, Rank 2 of the F1/F2 plan: this is the content-sequence source for the
        cross-attention falsification test -- run on the TARGET speech's mel (available
        during training, never at A2S inference time; the falsification test is deliberately
        oracle-content, see configs/exp/content_xattn_oracle.yaml). It is a real,
        already-validated (WER 0.41) representation of AudioLDM's own mel frontend at
        phoneme-adjacent rate (256 frames per 10.24s clip, ~25 fps -- above the ~0 fps a
        pooled vector delivers, though below the 51.2 tok/s AudioLDM 2 uses for its own TTS;
        a cheaper first test of "does ANY real sequence at ANY reasonable rate help" before
        building a dedicated phoneme encoder).

        Returns (B, T_out=256, encoder_dim=512).
        """
        x = mel.unsqueeze(1)
        x = self.conv(x)
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)
        x, _ = self.rnn(x)
        return x

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        mel: (B, T, F) log-mel, AudioLDM's own scale (norm_mode="none", no rescaling).
        returns: (B, T_out, vocab_size) LOG-probabilities, ready for `nn.CTCLoss`
                 (transpose to (T_out, B, vocab_size) at the call site -- CTCLoss wants
                 time first).
        """
        x = mel.unsqueeze(1)                              # (B, 1, T, F)
        x = self.conv(x)                                  # (B, C, T/4, F/4)
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)     # (B, T/4, C*F/4)
        x, _ = self.rnn(x)                                 # (B, T/4, 2*hidden)
        logits = self.head(x)                              # (B, T/4, vocab)
        return torch.log_softmax(logits, dim=-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class CTCProbeTranscriber:
    """
    Wraps a trained SimpleCTCASR as a `.transcribe(wavs) -> list[str]` ASR, matching
    salm.a2s.asr.WhisperASR's interface so both can be used interchangeably by a scorer.

    2026-08-26: this is also the SECOND, architecturally-independent judge for the
    inter-ASR-agreement metric (Rank 0 of the F1/F2 plan) -- CTC vs Whisper's attention-
    based seq2seq decoder are different enough in kind that two of them AGREEING on a
    generated clip is real evidence of actual phonetic content, not an artifact of one
    model's particular failure mode. No new download needed: this is the same checkpoint
    already trained tonight as O-F's frozen content-loss head.
    """

    def __init__(self, checkpoint: str | Path, device: str) -> None:
        from salm.data.frontend import MelConfig, MelExtractor

        self.device = torch.device(device)
        self.model = SimpleCTCASR().to(self.device).eval()
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model"])
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.extractor = MelExtractor(MelConfig(norm_mode="none"), device=self.device)

    @torch.no_grad()
    def transcribe(self, wavs: list, batch_size: int = 16) -> list[str]:
        from salm.a2s.wer import ctc_greedy_decode

        out: list[str] = []
        for i in range(0, len(wavs), batch_size):
            chunk = wavs[i : i + batch_size]
            wav_t = torch.from_numpy(np.stack(chunk)).float().to(self.device)
            mel = self.extractor.batch(wav_t)
            log_probs = self.model(mel)
            for ids in greedy_decode_batch(log_probs):
                out.append(ctc_greedy_decode(ids))
        return out


@torch.no_grad()
def greedy_decode_batch(log_probs: torch.Tensor) -> list[list[int]]:
    """log_probs: (B, T, vocab) -> per-row list of argmax token ids (blanks/repeats intact;
    collapse with salm.a2s.wer.ctc_greedy_decode for a readable string)."""
    return log_probs.argmax(dim=-1).cpu().tolist()
