"""
A2S dataset: precomputed CLAP conditioning + precomputed target-speech latents.

Simpler than the S2A dataset rather than a copy of it. S2A loads a speech waveform per item
because its encoder runs in the training loop; here both sides are already on disk (see
scripts/a2s/precompute_a2s_conditioning.py for why that trade is the right one for CLAP), so
this class is three memmaps and an index.

Row alignment is the one thing that can silently ruin a run: `cond_seq[i]`, `latents_mean[i]`
and the alignment target `[i]` must all describe the same pair. Both precompute scripts write
in `index.jsonl` order and assert `latent_index == i`, and the shape checks below refuse to
build a dataset whose parts disagree on row count.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# The O2 alignment target, chosen by the pairwise-spread measurement the conditioning
# precompute reports. "text" = CLAP-text of the transcript, "speech" = CLAP-audio of the
# target speech. See the script's docstring for why both exist.
ALIGN_FILES = {"text": "align_text.npy", "speech": "align_speech.npy"}


class A2SLatentDataset(Dataset):
    """One item is (CLAP condition frames, target speech latent distribution, metadata)."""

    def __init__(
        self,
        features_dir: str | Path,
        cond_dir: str | Path,
        align_target: str | None = "text",
    ) -> None:
        self.features_dir = Path(features_dir)
        self.cond_dir = Path(cond_dir)

        meta_path = self.features_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No meta.json in {self.features_dir}. Run:\n"
                f"  python scripts/precompute_latents.py --direction a2s --target speech "
                f"--split <split>"
            )
        self.meta = json.loads(meta_path.read_text())

        # Guard against the failure this cache makes easy: a2s latents accidentally built
        # from the ambient clip. They would train without error and generate the wrong
        # modality. Older caches predate the field, so only a positive mismatch is fatal.
        target = self.meta.get("target")
        if target is not None and target != "speech":
            raise ValueError(
                f"{self.features_dir} holds latents of the '{target}' file, but A2S generates "
                f"speech. Re-run precompute_latents.py with --target speech."
            )

        cond_meta_path = self.cond_dir / "meta.json"
        if not cond_meta_path.exists():
            raise FileNotFoundError(
                f"No meta.json in {self.cond_dir}. Run:\n"
                f"  python scripts/a2s/precompute_a2s_conditioning.py --split <split>"
            )
        self.cond_meta = json.loads(cond_meta_path.read_text())

        self.rows: list[dict] = []
        with (self.features_dir / "index.jsonl").open() as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

        # mmap so many DataLoader workers share one copy instead of each resident.
        self.latents_mean = np.load(self.features_dir / "latents_mean.npy", mmap_mode="r")
        self.latents_std = np.load(self.features_dir / "latents_std.npy", mmap_mode="r")
        self.cond_seq = np.load(self.cond_dir / "cond_seq.npy", mmap_mode="r")
        self.cond_mask = np.load(self.cond_dir / "cond_mask.npy", mmap_mode="r")

        n = len(self.rows)
        for name, arr in (
            ("latents_mean", self.latents_mean),
            ("latents_std", self.latents_std),
            ("cond_seq", self.cond_seq),
            ("cond_mask", self.cond_mask),
        ):
            if arr.shape[0] != n:
                raise ValueError(
                    f"{name} has {arr.shape[0]} rows but index.jsonl has {n}. The caches were "
                    "not built from the same index; every condition would be paired with the "
                    "wrong target."
                )

        self.align_target = None
        self.align_name = None
        if align_target:
            if align_target not in ALIGN_FILES:
                raise ValueError(
                    f"align_target must be one of {sorted(ALIGN_FILES)}, got {align_target!r}"
                )
            path = self.cond_dir / ALIGN_FILES[align_target]
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found. Re-run scripts/a2s/precompute_a2s_conditioning.py "
                    f"(without --skip-speech-align if you want align_target='speech'), or set "
                    f"align_target='' to train without the alignment loss and its val metrics."
                )
            arr = np.load(path, mmap_mode="r")
            if arr.shape[0] != n:
                raise ValueError(
                    f"{path.name} has {arr.shape[0]} rows but index.jsonl has {n}."
                )
            self.align_target = arr
            self.align_name = align_target

    # ---- properties mirroring S2ALatentDataset, so the trainer needs no special-casing
    @property
    def latent_shape(self) -> tuple[int, ...]:
        return tuple(self.latents_mean.shape[1:])

    @property
    def vae_scaling_factor(self) -> float:
        return float(self.meta["vae_scaling_factor"])

    @property
    def clap_target(self):
        """
        Alias for `align_target`, so this dataset duck-types as S2ALatentDataset.

        The trainer's O2 guard reads `train_ds.clap_target` to decide whether the alignment
        loss can run. Both directions' targets ARE CLAP embeddings -- ambience for S2A, the
        transcript's text embedding for A2S -- so one name covers both and the guard needs no
        direction branch.
        """
        return self.align_target

    @property
    def cond_feature_dim(self) -> int:
        return int(self.cond_seq.shape[2])

    @property
    def cond_frames(self) -> int:
        return int(self.cond_seq.shape[1])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        li = row["latent_index"]

        item = {
            # fp16 on disk, fp32 in the batch: the conditioner runs under autocast anyway,
            # but keeping the cast here means the stored dtype is an implementation detail.
            "cond_features": torch.from_numpy(
                np.asarray(self.cond_seq[li], dtype=np.float32)
            ),
            # np.array, not asarray: the mask is already bool, so asarray would hand torch a
            # read-only view of the mmap and torch warns about non-writable tensors.
            "cond_mask": torch.from_numpy(np.array(self.cond_mask[li], dtype=bool)),
            "latent_mean": torch.from_numpy(
                np.asarray(self.latents_mean[li], dtype=np.float32)
            ),
            "latent_std": torch.from_numpy(
                np.asarray(self.latents_std[li], dtype=np.float32)
            ),
            "sample_id": row["sample_id"],
            "audio_cap": row["audio_cap"],
            "speech_text": row["speech_text"],
            # 2026-08-26, Rank 2 (F1/F2 plan): the raw path, not audio -- the oracle content
            # cross-attention falsification test loads+mels the target speech on the fly
            # (only a few epochs, throwaway validation of the hypothesis; not worth a new
            # precompute pass for something explicitly temporary).
            "speech_file": row["speech_file"],
        }
        if self.align_target is not None:
            # Named `clap_target` on purpose: it IS a CLAP embedding, and reusing the S2A key
            # means the O2 loss, val_clap_cos and val_clap_r1 all work unchanged.
            #
            # np.array, not asarray: this cache is already fp32, so asarray(dtype=float32) is
            # a no-op view onto the read-only mmap and torch warns on every batch (same fix as
            # cond_mask above). The other three fields cast fp16 -> fp32, which forces a real
            # copy and is already writable.
            item["clap_target"] = torch.from_numpy(
                np.array(self.align_target[li], dtype=np.float32)
            )
        return item


def a2s_collate(batch: list[dict]) -> dict:
    out = {
        "cond_features": torch.stack([b["cond_features"] for b in batch]),
        "cond_mask": torch.stack([b["cond_mask"] for b in batch]),
        "latent_mean": torch.stack([b["latent_mean"] for b in batch]),
        "latent_std": torch.stack([b["latent_std"] for b in batch]),
        "sample_id": [b["sample_id"] for b in batch],
        "audio_cap": [b["audio_cap"] for b in batch],
        "speech_text": [b["speech_text"] for b in batch],
        "speech_file": [b["speech_file"] for b in batch],
    }
    if "clap_target" in batch[0]:
        out["clap_target"] = torch.stack([b["clap_target"] for b in batch])
    return out
