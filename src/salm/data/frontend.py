"""
Audio front-end: waveform loading and AudioLDM-compatible mel extraction.

Why this file is delicate
-------------------------
`diffusers` ships AudioLDM for INFERENCE only, so it provides no mel-spectrogram
extractor -- its pipeline only ever goes latent -> mel -> waveform. Training needs
the opposite direction (waveform -> mel -> latent), so we must reproduce AudioLDM's
mel exactly. If our mel does not match the distribution the VAE was trained on, the
VAE encodes garbage and every downstream result is meaningless while still "running".

AudioLDM's mel comes from a TacotronSTFT-style transform:
  sr 16000, n_fft 1024, hop 160, win 1024, n_mels 64, fmin 0, fmax 8000,
  then log compression: log(clamp(magnitude, min=1e-5) * C) with C = 1 (natural log).

The post-log normalization is the part we are least certain about (AudioLDM's own
codebase has several variants across repos/versions), so it is exposed as a config
knob rather than hardcoded. Do not trust any setting here until
`scripts/check_audioldm_stack.py` shows a clean mel -> vocoder round-trip.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
import soundfile as sf
import torch


# AudioLDM operates at 16 kHz on 10.24 s clips (1024 mel frames at hop 160).
AUDIOLDM_SR = 16000
AUDIOLDM_N_FFT = 1024
AUDIOLDM_HOP = 160
AUDIOLDM_WIN = 1024
AUDIOLDM_N_MELS = 64
AUDIOLDM_FMIN = 0
AUDIOLDM_FMAX = 8000
AUDIOLDM_CLIP_SECONDS = 10.24
AUDIOLDM_N_FRAMES = 1024


@dataclass
class MelConfig:
    """AudioLDM mel settings. `norm_mode` is the uncertain part -- see module docstring."""

    sample_rate: int = AUDIOLDM_SR
    n_fft: int = AUDIOLDM_N_FFT
    hop_length: int = AUDIOLDM_HOP
    win_length: int = AUDIOLDM_WIN
    n_mels: int = AUDIOLDM_N_MELS
    fmin: int = AUDIOLDM_FMIN
    fmax: int = AUDIOLDM_FMAX
    clip_seconds: float = AUDIOLDM_CLIP_SECONDS
    n_frames: int = AUDIOLDM_N_FRAMES

    # Post-log normalization variant:
    #   "none"      -> raw natural-log mel (AudioLDM TacotronSTFT output)
    #   "audiomae"  -> (x + 4.5) / 5.0, the AudioMAE-style shift/scale some
    #                  AudioLDM code paths apply
    norm_mode: str = "none"


def load_waveform_mono_16k(path: str, target_seconds: float | None = None) -> np.ndarray:
    """
    Load a wav as mono float32 at 16 kHz.

    The TTAS data is 24 kHz, and its ambient `audio_file` clips are ~70% stereo while
    `speech_file` is always mono (see CONTEXT.md audit), so downmixing is mandatory
    rather than defensive.

    If `target_seconds` is given, the waveform is center-cropped or zero-padded to
    exactly that length.
    """
    wav, sr = sf.read(path, dtype="float32", always_2d=True)

    # (frames, channels) -> mono
    wav = wav.mean(axis=1)

    if sr != AUDIOLDM_SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=AUDIOLDM_SR, res_type="soxr_hq")

    if target_seconds is not None:
        wav = _fix_length(wav, int(round(target_seconds * AUDIOLDM_SR)))

    return wav.astype(np.float32, copy=False)


def _fix_length(wav: np.ndarray, target_len: int) -> np.ndarray:
    """Center-crop or right-pad with zeros to `target_len` samples."""
    n = wav.shape[0]
    if n == target_len:
        return wav
    if n > target_len:
        start = (n - target_len) // 2
        return wav[start : start + target_len]
    return np.pad(wav, (0, target_len - n), mode="constant")


def is_near_silent(wav: np.ndarray, rms_threshold: float = 1e-4) -> bool:
    """
    Flag degenerate clips. The data audit found at least one 0.01 s ambient clip, and a
    silent target would contribute a meaningless training signal.
    """
    if wav.size == 0:
        return True
    return float(np.sqrt(np.mean(np.square(wav)))) < rms_threshold


class MelExtractor:
    """
    Waveform -> log-mel, matching AudioLDM's TacotronSTFT.

    Output shape is (n_frames, n_mels) -- mel bins LAST. That ordering is what both
    AudioLDM's VAE and the `SpeechT5HifiGan` vocoder in diffusers expect, and getting
    it transposed is a silent failure mode.
    """

    def __init__(self, config: MelConfig | None = None, device: str | torch.device = "cpu"):
        self.config = config or MelConfig()
        self.device = torch.device(device)

        mel_basis = librosa.filters.mel(
            sr=self.config.sample_rate,
            n_fft=self.config.n_fft,
            n_mels=self.config.n_mels,
            fmin=self.config.fmin,
            fmax=self.config.fmax,
        )
        self.mel_basis = torch.from_numpy(mel_basis).float().to(self.device)
        self.window = torch.hann_window(self.config.win_length).to(self.device)

    def __call__(self, wav: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Returns (n_frames, n_mels) float32 on `self.device`."""
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav)
        wav = wav.float().to(self.device)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        mel = self.batch(wav)
        return mel.squeeze(0)

    def batch(self, wav: torch.Tensor) -> torch.Tensor:
        """Batched variant. Input (B, samples) -> output (B, n_frames, n_mels)."""
        cfg = self.config
        wav = wav.float().to(self.device)

        # AudioLDM's STFT reflect-pads by n_fft//2 and uses center=False, which is
        # equivalent to center=True with reflect padding for these settings.
        spec = torch.stft(
            wav,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        magnitude = torch.abs(spec)                       # (B, freq, frames)
        mel = torch.matmul(self.mel_basis, magnitude)     # (B, n_mels, frames)

        # Dynamic range compression, C = 1 -> natural log.
        mel = torch.log(torch.clamp(mel, min=1e-5))

        if cfg.norm_mode == "audiomae":
            mel = (mel + 4.5) / 5.0
        elif cfg.norm_mode != "none":
            raise ValueError(f"unknown norm_mode: {cfg.norm_mode!r}")

        mel = mel.transpose(1, 2)                         # (B, frames, n_mels)
        return self._fix_frames(mel, cfg.n_frames)

    def denormalize(self, mel: torch.Tensor) -> torch.Tensor:
        """Invert `norm_mode` so a mel can be handed to the vocoder in its native scale."""
        if self.config.norm_mode == "audiomae":
            return mel * 5.0 - 4.5
        return mel

    @staticmethod
    def _fix_frames(mel: torch.Tensor, n_frames: int) -> torch.Tensor:
        """Crop or pad the time axis to exactly `n_frames`."""
        cur = mel.shape[1]
        if cur == n_frames:
            return mel
        if cur > n_frames:
            return mel[:, :n_frames, :]
        pad = mel.new_full((mel.shape[0], n_frames - cur, mel.shape[2]), float(mel.min()))
        return torch.cat([mel, pad], dim=1)
