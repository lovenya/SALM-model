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
        clap_dir: str | Path | None = None,
    ) -> None:
        self.features_dir = Path(features_dir)
        self.speech_seconds = speech_seconds
        self.speech_samples = int(round(speech_seconds * AUDIOLDM_SR))
        self.clap_dir = Path(clap_dir) if clap_dir else None

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

        # CLAP embeddings of the target clips, used for the val cosine metric. Optional
        # because training does not need them -- only the diagnostic does.
        self.clap_target = None
        if self.clap_dir is not None:
            path = self.clap_dir / "clap_target.npy"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found. Run scripts/precompute_clap_targets.py, or set "
                    "clap_features_root='' to train without the cosine metric."
                )
            self.clap_target = np.load(path, mmap_mode="r")
            if self.clap_target.shape[0] != len(self.rows):
                raise ValueError(
                    f"CLAP targets have {self.clap_target.shape[0]} rows but index.jsonl "
                    f"has {len(self.rows)}. These must be built from the same index or the "
                    "cosine metric compares mismatched pairs."
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

        item = {
            "speech": torch.from_numpy(fixed),
            "speech_length": torch.tensor(true_len, dtype=torch.long),
            "latent_mean": torch.from_numpy(np.asarray(self.latents_mean[li], dtype=np.float32)),
            "latent_std": torch.from_numpy(np.asarray(self.latents_std[li], dtype=np.float32)),
            "sample_id": row["sample_id"],
            "audio_cap": row["audio_cap"],
            "speech_text": row["speech_text"],
        }
        if self.clap_target is not None:
            item["clap_target"] = torch.from_numpy(
                np.asarray(self.clap_target[li], dtype=np.float32)
            )
        return item


def collate(batch: list[dict]) -> dict:
    """Stack tensors, keep strings as lists."""
    out = {
        "speech": torch.stack([b["speech"] for b in batch]),
        "speech_length": torch.stack([b["speech_length"] for b in batch]),
        "latent_mean": torch.stack([b["latent_mean"] for b in batch]),
        "latent_std": torch.stack([b["latent_std"] for b in batch]),
        "sample_id": [b["sample_id"] for b in batch],
        "audio_cap": [b["audio_cap"] for b in batch],
        "speech_text": [b["speech_text"] for b in batch],
    }
    if "clap_target" in batch[0]:
        out["clap_target"] = torch.stack([b["clap_target"] for b in batch])
    return out
