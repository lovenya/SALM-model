"""
Translate our generation output into the manifest schema SpeechAudioLM's evaluator expects.

We reuse that repo's validated metric stack (fad, kad, kl_passt,
target_caption_clap_score, inception_score) rather than reimplementing metrics. It expects
its own schema, though:

    schema_version, row_index, sample_id, stage, model, status,
    input {...}, output {type, audio_file (RELATIVE to the manifest dir), ...},
    timing {...}, error

while `salm.generate` writes a flat schema with absolute paths. This script bridges the two.

Audio is exposed via SYMLINKS, not copies: scratch runs ~90% full and the wavs already
exist. Use --copy if a backend refuses to follow links.

Note `prompt_adherence_clap_score` is deliberately not part of the intended metric set:
it scores audio against the exact persisted *text* generation prompt, and direct S2A has no
text prompt, so the number would be meaningless. `target_caption_clap_score` is the right
per-sample metric here -- it compares generated audio against the dataset target caption.

Usage:
    python scripts/export_eval_manifest.py \
        --generation-dir <run>/generations/s2a_best \
        --out-dir <run>/eval_export/s2a_best --stage s2a --model salm_audioldm_whisper
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generation-dir", required=True,
                    help="dir holding manifest.jsonl + audio/ from salm.generate")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stage", default="s2a", choices=["s2a", "t2a"],
                    help="t2a for the rung0 caption ceiling, s2a for our model")
    ap.add_argument("--model", default="salm_audioldm_whisper")
    ap.add_argument("--dataset", default="ttas")
    ap.add_argument("--split", default="val")
    ap.add_argument("--copy", action="store_true",
                    help="copy wavs instead of symlinking (uses real disk)")
    args = ap.parse_args()

    gen_dir = Path(args.generation_dir)
    src_manifest = gen_dir / "manifest.jsonl"
    if not src_manifest.exists():
        log(f"ERROR: {src_manifest} not found")
        return 1

    out_dir = Path(args.out_dir)
    audio_out = out_dir / "outputs"
    audio_out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in src_manifest.read_text().splitlines() if l.strip()]
    log(f"{len(rows)} rows from {src_manifest}")

    summary = {}
    s_path = gen_dir / "summary.json"
    if s_path.exists():
        summary = json.loads(s_path.read_text())

    out_rows = []
    skipped = 0
    for i, r in enumerate(rows):
        if r.get("status") != "success":
            skipped += 1
            continue

        # Resolve robustly. Older manifests stored ABSOLUTE paths, so any run whose
        # directory has since been moved has stale entries. Fall back to the conventional
        # location beside the manifest, which survives relocation.
        # Prefer the portable relative path when present.
        if r.get("generated_file_rel"):
            src = gen_dir / r["generated_file_rel"]
        else:
            src = Path(r["generated_file"])
        if not src.exists():
            fallback = gen_dir / "audio" / f"{r['sample_id']}.wav"
            if fallback.exists():
                src = fallback
            else:
                log(f"  WARNING: no audio for {r.get('sample_id')} "
                    f"(tried {r['generated_file']} and {fallback}), skipping")
                skipped += 1
                continue

        # Deterministic, index-prefixed names, matching the convention seen in the
        # project's existing t2a manifests.
        name = f"{i:06d}_{r['sample_id']}.wav"
        dst = audio_out / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if args.copy:
            import shutil

            shutil.copy2(src, dst)
        else:
            os.symlink(src.resolve(), dst)

        out_rows.append(
            {
                "schema_version": 1,
                "row_index": i,
                "sample_id": r["sample_id"],
                "stage": args.stage,
                "model": args.model,
                "status": "success",
                "input": {
                    "source_sample_id": r["sample_id"],
                    # The dataset target caption. target_caption_clap_score scores the
                    # generated audio against this.
                    "audio_cap": r.get("audio_cap"),
                    "prompt": r.get("audio_cap"),
                    "speech_text": r.get("speech_text"),
                    "speech_file": r.get("speech_file"),
                    # Reference ambient clip: what FAD/KAD compare the distribution against.
                    "reference_audio_file": r.get("reference_audio_file"),
                },
                "output": {
                    "type": "audio",
                    # Relative to this manifest's directory, as the evaluator expects.
                    "audio_file": f"outputs/{name}",
                    "sample_rate": 16000,
                    "rms": r.get("generated_rms"),
                },
                "timing": {"inference_seconds": None},
                "error": None,
            }
        )

    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")

    meta = {
        "source_generation_dir": str(gen_dir),
        "stage": args.stage,
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "rows_in": len(rows),
        "rows_out": len(out_rows),
        "skipped": skipped,
        "linked": not args.copy,
        "generation_summary": summary,
        # Corpus-level metrics need the reference set; record where it came from.
        "reference_audio_from": "input.reference_audio_file per row",
    }
    (out_dir / "export_meta.json").write_text(json.dumps(meta, indent=2))

    log(f"wrote {manifest_path} ({len(out_rows)} rows, {skipped} skipped)")
    log(f"audio {'copied' if args.copy else 'symlinked'} into {audio_out}")
    log("")
    log("Score it with the SpeechAudioLM stack (its own venv), e.g.:")
    log("  cd /scratch/l/lovenya/SpeechAudioLM && python evaluate.py \\")
    log(f"      --model {args.model} --stage {args.stage} --direction s2a \\")
    log(f"      --manifest {manifest_path} \\")
    log("      --metrics fad,kad,kl_passt,target_caption_clap_score,inception_score --dry-run")
    log("")
    log("Run with --dry-run first: FAD/KAD are corpus-level and their backends have their")
    log("own manifest expectations, so confirm resolution before spending GPU time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
