#!/usr/bin/env python
"""Whisper WER and matched-vs-mismatched text binding for bridge generations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from configs import paths as project_paths  # noqa: E402
from salm.a2s.asr import WhisperASR  # noqa: E402
from salm.a2s.wer import corpus_wer  # noqa: E402
from salm.generate import deranged_indices  # noqa: E402


def score_dir(path: Path, whisper: WhisperASR, batch_size: int, seed: int) -> dict:
    rows = [json.loads(line) for line in (path / "manifest.jsonl").read_text().splitlines()]
    wavs = []
    for row in rows:
        wav, sr = sf.read(row["generated_file"], dtype="float32")
        if sr != 16000:
            raise RuntimeError(f"{row['generated_file']}: expected 16 kHz, got {sr}")
        wavs.append(wav)
    hyps = whisper.transcribe(wavs, batch_size=batch_size)
    refs = [str(r["supplied_text"]) for r in rows]
    paired = corpus_wer(refs, hyps)
    perm = deranged_indices(len(rows), seed)
    mismatched = corpus_wer([refs[i] for i in perm], hyps)
    out = {
        "lane": rows[0]["lane"] if rows else path.name,
        "directory": str(path),
        "n": len(rows),
        "wer_to_supplied_text": paired,
        "wer_to_mismatched_text": mismatched,
        # Positive means the rendered audio is closer to the supplied transcript than to
        # another real transcript.  This is the audio-level paired semantic binding check.
        "audio_text_binding_gap": mismatched["wer"] - paired["wer"],
        "examples": [
            {
                "sample_id": r["sample_id"],
                "supplied_text": ref,
                "whisper": hyp,
            }
            for r, ref, hyp in zip(rows, refs, hyps)
        ],
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1701)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing existing score file: {args.out}")
    for d in args.dirs:
        if not (d / "manifest.jsonl").exists():
            raise SystemExit(f"missing manifest: {d / 'manifest.jsonl'}")
    whisper = WhisperASR(project_paths.WHISPER_LARGE_V3_PATH, args.device)
    results = [score_dir(d, whisper, args.batch_size, args.seed) for d in args.dirs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print("lane                       WER(input)  WER(wrong)  binding", flush=True)
    for r in results:
        print(
            f"{r['lane']:<26} {r['wer_to_supplied_text']['wer']:>10.3f} "
            f"{r['wer_to_mismatched_text']['wer']:>11.3f} "
            f"{r['audio_text_binding_gap']:>+8.3f}", flush=True,
        )
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
