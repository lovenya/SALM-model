"""
Training dataset: precomputed ambient latents + on-the-fly speech waveforms.

Latents come from `scripts/precompute_latents.py` and are memory-mapped, so many
DataLoader workers share one copy instead of each loading a multi-GB array.

Speech is loaded per item rather than precomputed. Whisper's encoder is cheap enough to
run in the training loop, and keeping the raw waveform means the encoder and its pooling
stay swappable -- precomputing encoder features would have frozen a pooling choice into
the cache.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from salm.data.frontend import AUDIOLDM_SR, load_waveform_mono_16k

# Speech is padded/cropped to a fixed length so a batch can be stacked. 16 s covers the
# observed TTAS range (max ~16.1 s, mean ~4.9 s) without truncating real content.
DEFAULT_SPEECH_SECONDS = 16.0


class S2ALatentDataset(Dataset):
    """
    One item is (speech waveform, target ambient latent distribution, metadata).

    Returns the posterior mean and std so the training loop can sample from the latent
    distribution, matching how AudioLDM trains, rather than reusing a single fixed draw.
    """

    def __init__(
        self,
        features_dir: str | Path,
        speech_seconds: float = DEFAULT_SPEECH_SECONDS,
    ) -> None:
        self.features_dir = Path(features_dir)
        self.speech_seconds = speech_seconds
        self.speech_samples = int(round(speech_seconds * AUDIOLDM_SR))

        meta_path = self.features_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No meta.json in {self.features_dir}. Run scripts/precompute_latents.py first."
            )
        self.meta = json.loads(meta_path.read_text())

        self.rows: list[dict] = []
        with (self.features_dir / "index.jsonl").open() as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

        # mmap_mode keeps this out of every worker's resident memory.
        self.latents_mean = np.load(self.features_dir / "latents_mean.npy", mmap_mode="r")
        self.latents_std = np.load(self.features_dir / "latents_std.npy", mmap_mode="r")

        if len(self.rows) != self.latents_mean.shape[0]:
            raise ValueError(
                f"index.jsonl has {len(self.rows)} rows but latents have "
                f"{self.latents_mean.shape[0]} -- precompute output is inconsistent."
            )

    @property
    def latent_shape(self) -> tuple[int, ...]:
        return tuple(self.latents_mean.shape[1:])

    @property
    def vae_scaling_factor(self) -> float:
        return float(self.meta["vae_scaling_factor"])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        li = row["latent_index"]

        wav = load_waveform_mono_16k(row["speech_file"])
        true_len = min(wav.shape[0], self.speech_samples)

        fixed = np.zeros(self.speech_samples, dtype=np.float32)
        fixed[:true_len] = wav[:true_len]

        return {
            "speech": torch.from_numpy(fixed),
            "speech_length": torch.tensor(true_len, dtype=torch.long),
            "latent_mean": torch.from_numpy(np.asarray(self.latents_mean[li], dtype=np.float32)),
            "latent_std": torch.from_numpy(np.asarray(self.latents_std[li], dtype=np.float32)),
            "sample_id": row["sample_id"],
            "audio_cap": row["audio_cap"],
            "speech_text": row["speech_text"],
        }


def collate(batch: list[dict]) -> dict:
    """Stack tensors, keep strings as lists."""
    return {
        "speech": torch.stack([b["speech"] for b in batch]),
        "speech_length": torch.stack([b["speech_length"] for b in batch]),
        "latent_mean": torch.stack([b["latent_mean"] for b in batch]),
        "latent_std": torch.stack([b["latent_std"] for b in batch]),
        "sample_id": [b["sample_id"] for b in batch],
        "audio_cap": [b["audio_cap"] for b in batch],
        "speech_text": [b["speech_text"] for b in batch],
    }
