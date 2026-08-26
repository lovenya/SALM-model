"""
Frozen CLAP audio encoder: ambient waveform -> frame sequence. The A2S mirror of Whisper.

WHY CLAP HERE, WHEN IT WAS THE WRONG CHOICE FOR S2A. One rule picks the encoder for each
direction: use the encoder whose training objective already represents the thing the
conditioner has to read out.

    S2A conditions on SPEECH   and needs *what was said*      -> Whisper (ASR objective)
    A2S conditions on AMBIENCE and needs *what the scene is*  -> CLAP audio tower
                                                                 (audio<->caption objective)

CLAP is right here for exactly the reason it was wrong for S2A. It was trained against
captions of environmental sound, so on speech it reports only that speech is happening --
which is why this repo kept it as a deliberate S2A *control* -- while on ambience it reports
the scene. See docs/OPEN_ITEMS.md, 2026-08-23.

SEQUENCE OUTPUT, NOT THE POOLED 512-d VECTOR. `get_audio_features` would hand the conditioner
one pre-pooled vector and collapse the whole ladder (mean_linear / attn_mlp / xformer /
qformer) into a 512->512 MLP, making the A2S runs incomparable to the S2A ones rung for rung.
`ClapAudioModel.last_hidden_state` gives the pre-pooling tensor instead.

Shape, derived from laion/clap-htsat-unfused's config (hidden_size 768, depths [2,2,6,2],
patch_stride [4,4], spec_size 256, freq_ratio 4) and ASSERTED at runtime rather than trusted:

    last_hidden_state  (B, 768, 2, 32)   -- (batch, channels, freq_bins, time)
    mean over freq  ->  (B, 768, 32)
    transpose       ->  (B, 32, 768)     -- 32 frames over 10 s = 0.31 s per frame

**The time axis really is time-ordered**, which is not obvious and is worth stating because
getting it wrong would be silent. HTSAT folds a long mel into a square image by stacking
`freq_ratio` consecutive TIME chunks along the frequency axis, and the group-2D reshape at
the end of `ClapAudioEncoder.forward` (transformers/models/clap/modeling_clap.py) undoes
exactly that: `(B, C, F, T) -> (B, C, F/c, c, T) -> permute -> (B, C, c, F/c * T)`. The
`F/c` groups are the consecutive time chunks and `T` is position within a chunk, so the
flattened axis runs forward in time. Verified by reading the installed source.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import torch

from salm.encoders.base import FrozenEncoder

# LAION-CLAP runs at 48 kHz over a fixed 10 s window. The FrozenEncoder contract is 16 kHz in,
# so the resample happens here -- the same 16k -> 48k path that
# scripts/precompute_clap_targets.py already uses. That is deliberate, not incidental: the
# S2A CLAP targets on disk were produced that way and the two must stay comparable.
CLAP_SR = 48000
CLAP_WINDOW_SECONDS = 10.0
CLAP_MAX_SAMPLES = int(CLAP_SR * CLAP_WINDOW_SECONDS)   # 480_000


class ClapAudioSequenceEncoder(FrozenEncoder):
    """Frozen CLAP-HTSAT audio tower, returning frame features rather than a pooled vector."""

    def __init__(
        self,
        model_id: str | Path | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cuda",
    ) -> None:
        super().__init__()
        from transformers import ClapAudioModel, ClapFeatureExtractor
        from configs import paths as project_paths

        model_id = str(model_id or project_paths.CLAP_MODEL_ID)
        self.device = torch.device(device)
        self.dtype = dtype

        self.feature_extractor = ClapFeatureExtractor.from_pretrained(model_id)
        self.audio_model = (
            ClapAudioModel.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
        )
        self._feature_dim = int(self.audio_model.config.hidden_size)
        # Filled in on the first forward, then asserted stable. Not hardcoded: it depends on
        # spec_size / patch_stride / depths, and a config change would otherwise pass silently.
        self._num_frames: int | None = None

        self.freeze()

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def num_frames(self) -> int | None:
        """Frames per clip. None until the first forward has run."""
        return self._num_frames

    def frames_for_seconds(self, seconds: float) -> int:
        """How many of the emitted frames are backed by `seconds` of real audio."""
        if self._num_frames is None:
            raise RuntimeError("num_frames is unknown until the first forward pass")
        frac = min(max(seconds, 0.0) / CLAP_WINDOW_SECONDS, 1.0)
        return max(1, int(round(frac * self._num_frames)))

    @torch.no_grad()
    def forward(self, wav_16k: torch.Tensor) -> torch.Tensor:
        """wav_16k: (B, samples) float32 @16k -> (B, T_feat, D)"""
        wavs = []
        for w in wav_16k:
            x = w.detach().float().cpu().numpy()
            x = librosa.resample(x, orig_sr=16000, target_sr=CLAP_SR, res_type="soxr_hq")
            # Hard cap. The extractor's default truncation is "rand_trunc", which takes a
            # RANDOM crop once the waveform exceeds 10 s -- our 10.24 s clips do, so without
            # this the "frozen" encoder would be nondeterministic and a precomputed cache
            # would not match a re-run.
            wavs.append(x[:CLAP_MAX_SAMPLES])

        inputs = self.feature_extractor(
            wavs, sampling_rate=CLAP_SR, return_tensors="pt"
        )
        feats = inputs.input_features.to(self.device, dtype=self.dtype)

        out = self.audio_model(input_features=feats).last_hidden_state
        if out.dim() != 4:
            raise RuntimeError(
                f"expected CLAP last_hidden_state of rank 4 (B, C, F, T), got {tuple(out.shape)}. "
                "The group-2D reshape in ClapAudioEncoder.forward may have changed; re-read it "
                "before trusting the time ordering of these features."
            )
        # (B, C, F, T) -> average the frequency-bin axis -> (B, C, T) -> (B, T, C)
        seq = out.mean(dim=2).transpose(1, 2).contiguous().float()

        n_frames = seq.shape[1]
        if self._num_frames is None:
            self._num_frames = int(n_frames)
        elif n_frames != self._num_frames:
            raise RuntimeError(
                f"CLAP frame count changed between batches ({self._num_frames} -> {n_frames}); "
                "features would not be stackable or cacheable."
            )
        return seq

    @torch.no_grad()
    def encode_with_mask(
        self, wav_16k: torch.Tensor, lengths_samples: torch.Tensor, truncate: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Same as forward, plus a bool mask marking frames backed by real audio.

        IN THIS PIPELINE THE MASK IS ALWAYS ALL-TRUE, and that is deliberate rather than an
        oversight. Both the precompute and generation hand this encoder a clip already fixed
        to the 10.24 s canvas, so `lengths_samples` is the full canvas and every frame counts
        as real. The alternative -- masking by each clip's true duration -- was rejected
        because the mask would then have to be reproduced identically at generation time, and
        a train/inference mismatch there is silent. The cost is that the tail of genuinely
        short ambient clips (the audit found one at 0.01 s) has its zero-padding pooled as if
        it were scene content, which is exactly what S2A already did with its ambient targets.

        The parameter is kept because it is the FrozenEncoder contract and because variable-
        length ambient input is a plausible later experiment; `cond_mask_all_true_frac` in the
        cache's meta.json records that it currently reads 1.0.

        Contrast Whisper, where masking was essential rather than cosmetic: a 30 s window
        against ~4.9 s of speech meant 84.6% of the sequence was padding.

        `truncate` defaults to False here, the opposite of the Whisper encoder: at 32 frames
        there is no O(T^2) attention cost worth reclaiming, and a variable sequence length
        across batches would break the fixed-shape precomputed cache.
        """
        feats = self.forward(wav_16k)
        b, t, _ = feats.shape

        valid = torch.tensor(
            [self.frames_for_seconds(float(n) / 16000.0) for n in lengths_samples],
            device=feats.device,
        )
        idx = torch.arange(t, device=feats.device).unsqueeze(0).expand(b, t)
        mask = idx < valid.unsqueeze(1)

        if truncate:
            keep = int(max(1, int(valid.max().item())))
            feats, mask = feats[:, :keep, :], mask[:, :keep]
        return feats, mask
