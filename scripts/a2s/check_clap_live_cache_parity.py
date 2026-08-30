#!/usr/bin/env python3
"""Fail-fast proof that live generation CLAP features match the training cache.

Run this on a compute GPU before any Rank-2 listening generation. It compares the same
ambient files through the live encoder against ``cond_seq.npy`` (stored as fp16), and also
requires every live query to retrieve its own cached row at rank 1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from configs import paths as project_paths  # noqa: E402
from salm.data.frontend import AUDIOLDM_CLIP_SECONDS, load_waveform_mono_16k  # noqa: E402
from salm.encoders.clap_encoder import ClapAudioSequenceEncoder  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--min-cosine", type=float, default=0.999)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--index-file", default=None)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    cache_dir = Path(
        args.cache_root or project_paths.OUTPUTS_ROOT / "features_a2s_cond"
    ) / args.split
    index_file = Path(
        args.index_file or project_paths.FEATURES_ROOT / "a2s" / args.split / "index.jsonl"
    )
    cache_file = cache_dir / "cond_seq.npy"
    if not cache_file.exists() or not index_file.exists():
        raise FileNotFoundError(f"parity needs {cache_file} and {index_file}")

    rows = [json.loads(line) for line in index_file.read_text().splitlines() if line]
    cached = np.load(cache_file, mmap_mode="r")
    if len(rows) != cached.shape[0]:
        raise RuntimeError(f"index/cache mismatch: {len(rows)} rows vs {cached.shape[0]} tensors")

    selected_indices = list(
        range(args.start_index, min(args.start_index + args.num_samples, len(rows)))
    )
    if not selected_indices:
        raise ValueError("parity selection is empty")

    encoder = ClapAudioSequenceEncoder(device=args.device)
    cached_keys = F.normalize(
        torch.from_numpy(np.asarray(cached.mean(axis=1), dtype=np.float32)), dim=-1
    )
    per_row: list[dict] = []

    for start in range(0, len(selected_indices), args.batch_size):
        indices = selected_indices[start : start + args.batch_size]
        wavs = [
            load_waveform_mono_16k(
                rows[i]["audio_file"], target_seconds=AUDIOLDM_CLIP_SECONDS
            )
            for i in indices
        ]
        live, _ = encoder.encode_with_mask(
            torch.from_numpy(np.stack(wavs)).float().to(args.device),
            torch.full((len(indices),), int(AUDIOLDM_CLIP_SECONDS * 16000)),
        )
        live = live.float().cpu()
        expected = torch.from_numpy(np.asarray(cached[indices], dtype=np.float32))
        if tuple(live.shape) != tuple(expected.shape):
            raise RuntimeError(
                f"live/cache shape mismatch: {tuple(live.shape)} vs {tuple(expected.shape)}"
            )

        flat_cos = F.cosine_similarity(live.flatten(1), expected.flatten(1), dim=-1)
        live_keys = F.normalize(live.mean(dim=1), dim=-1)
        top = (live_keys @ cached_keys.T).argmax(dim=-1).tolist()
        for j, cache_index in enumerate(indices):
            per_row.append(
                {
                    "sample_id": rows[cache_index]["sample_id"],
                    "cache_index": cache_index,
                    "flattened_cosine": float(flat_cos[j].item()),
                    "max_abs_error": float((live[j] - expected[j]).abs().max().item()),
                    "retrieved_cache_index": int(top[j]),
                    "self_retrieval_rank1": int(top[j]) == cache_index,
                }
            )

    min_cosine = min(row["flattened_cosine"] for row in per_row)
    all_self_rank1 = all(row["self_retrieval_rank1"] for row in per_row)
    passed = min_cosine >= args.min_cosine and all_self_rank1
    report = {
        "schema_version": 1,
        "split": args.split,
        "cache_file": str(cache_file),
        "index_file": str(index_file),
        "num_samples": len(per_row),
        "min_cosine_required": args.min_cosine,
        "min_cosine_observed": min_cosine,
        "all_self_retrieval_rank1": all_self_rank1,
        "passed": passed,
        "rows": per_row,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    if not passed:
        print("ERROR: live CLAP/cache parity gate failed; refusing to trust generation.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
