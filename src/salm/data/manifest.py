"""
Manifest reading for the TTAS clean_v2 splits.

The manifests live in the SpeechAudioLM repo and are read-only from here (this repo is
clean-room: paths only, no code imports). Row paths are relative to the dataset root.

Note the direction manifests are NOT row-aligned with each other -- a2s and s2a were
filtered independently against their own G-Eval scores, so a pair can be in one and not
the other. Never zip them positionally. See CONTEXT.md for audited overlap counts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from configs import paths as project_paths


@dataclass
class TTASRow:
    """One (speech, ambient audio) pair plus its metadata."""

    audio_path: Path        # absolute; the ambient scene = S2A target
    speech_path: Path       # absolute; the utterance = S2A input
    speech_text: str
    audio_cap: str
    speaker_gender: str
    speaker_emotion: str
    original_line_index: int

    @property
    def sample_id(self) -> str:
        """Stable per-row id. TTAS names both files after the same source clip."""
        return self.audio_path.stem


def load_manifest(
    direction: str,
    split: str,
    dataset_root: Path | None = None,
) -> list[TTASRow]:
    """
    Read one direction/split manifest into rows with absolute paths.

    direction: "s2a" or "a2s"
    split:     "train" | "val" | "test"
    """
    if direction == "s2a":
        manifests = project_paths.S2A_MANIFESTS
    elif direction == "a2s":
        manifests = project_paths.A2S_MANIFESTS
    else:
        raise ValueError(f"direction must be 's2a' or 'a2s', got {direction!r}")

    if split not in manifests:
        raise ValueError(f"split must be train/val/test, got {split!r}")

    root = Path(dataset_root or project_paths.TTAS_DATASET_ROOT)
    manifest_path = manifests[split]

    rows: list[TTASRow] = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            rows.append(
                TTASRow(
                    audio_path=root / raw["audio_file"],
                    speech_path=root / raw["speech_file"],
                    speech_text=raw.get("speech_text", ""),
                    audio_cap=raw.get("audio_cap", ""),
                    speaker_gender=raw.get("speaker_gender", ""),
                    speaker_emotion=raw.get("speaker_emotion", ""),
                    original_line_index=raw.get("_original_line_index", -1),
                )
            )
    return rows
