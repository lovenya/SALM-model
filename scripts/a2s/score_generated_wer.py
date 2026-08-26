"""
WER on GENERATED A2S audio, against an INDEPENDENT judge (Whisper-large-v3).

WHY THIS WAS MISSING. Every A2S result so far (A0/A1/A2, the oracle-conditioning ablation)
was judged by ear -- real, but subjective and not a number anyone else can check or that
scales past 16 clips. `check_speech_stack.py` and `train_ctc_probe.py` both compute WER, but
only ever on REAL target speech (the round-trip gate, and the CTC probe's own training
signal) -- nothing has ever measured whether a GENERATED clip is comprehensible.

WHY WHISPER, NOT THE CTC PROBE THAT O-F TRAINS AGAINST. Scoring O-F's output with the same
CTC head it was trained to satisfy would be circular -- a falling training loss proves the
model is fooling that scorer, not that the words are right, exactly the reasoning that
already rules out CLAP-based metrics for O2 (`OPEN_ITEMS.md`: "target_caption_clap_score /
FAD during training... contaminated under O2. Held-out eval only."). Whisper-large-v3 is
never part of any training loop here and is a much stronger ASR model besides, so it is the
honest, independent judge -- the same reasoning `check_speech_stack.py`'s gate already used.

Reads any number of generation directories (each with manifest.jsonl + audio/), so every
existing lane -- A0/A1/A2's a2s_best and b0_uncond, the oracle ablation's oracle_text and
real_conditioner -- can be scored retroactively in one job, and future lanes (A3/O-F) drop
in the same way.

Usage:
    python scripts/a2s/score_generated_wer.py --dirs <gen_dir1> <gen_dir2> ... --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from configs import paths as project_paths  # noqa: E402
from salm.a2s.asr import WhisperASR  # noqa: E402
from salm.a2s.wer import corpus_wer  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


def score_one_dir(gen_dir: Path, asr: WhisperASR, batch_size: int) -> dict:
    manifest_path = gen_dir / "manifest.jsonl"
    if not manifest_path.exists():
        return {"error": f"no manifest.jsonl in {gen_dir}"}
    rows = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
    if not rows:
        return {"error": f"empty manifest in {gen_dir}"}
    if "speech_text" not in rows[0]:
        # An s2a-direction generation dir (no transcript to score against) -- skip cleanly
        # rather than crash, so a mixed batch of dirs doesn't abort the whole job.
        return {"error": "no speech_text in manifest -- not an A2S generation (skipped)"}

    wavs, refs, ids = [], [], []
    for r in rows:
        if r.get("status") != "success":
            continue
        wav, sr = sf.read(gen_dir / r["generated_file_rel"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wavs.append(wav)
        refs.append(r["speech_text"])
        ids.append(r["sample_id"])

    hyps = asr.transcribe(wavs, batch_size=batch_size)
    result = corpus_wer(refs, hyps)
    result["examples"] = [
        {"sample_id": i, "ref": r, "hyp": h} for i, r, h in zip(ids, refs, hyps)
    ]
    result["n_scored"] = len(refs)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="one or more generation dirs, each with manifest.jsonl + audio/")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None,
                    help="combined comparison JSON (default: alongside the first --dirs entry)")
    args = ap.parse_args()

    log("Loading Whisper-large-v3 (independent judge, never part of any training loop) ...")
    asr = WhisperASR(project_paths.WHISPER_LARGE_V3_PATH, args.device)

    results = {}
    t0 = time.time()
    for d in args.dirs:
        gen_dir = Path(d).resolve()
        # BUG, caught after a real run: "{parent.name}/{name}" collides whenever two runs
        # share a lane name under a common "generations" folder (every probe40_* run does,
        # e.g. .../probe40_A0_plain/generations/a2s_best AND .../probe40_A1_o2/generations/
        # a2s_best both reduced to "generations/a2s_best") -- later dirs silently overwrote
        # earlier ones in `results` and in the saved JSON, even though each dir's own WER was
        # printed correctly to the log in the moment. Relative to OUTPUTS_ROOT is always
        # unique regardless of directory depth/structure (run-scoped vs standalone dirs use
        # different depths), so this can't recur.
        try:
            label = str(gen_dir.relative_to(project_paths.OUTPUTS_ROOT))
        except ValueError:
            label = str(gen_dir)
        log(f"\n[{label}] scoring {gen_dir} ...")
        r = score_one_dir(gen_dir, asr, args.batch_size)
        results[label] = r
        if "error" in r:
            log(f"    SKIPPED: {r['error']}")
            continue
        log(f"    WER={r['wer']:.4f}  ({r['edits']}/{r['ref_words']} word edits, "
            f"n={r['n_scored']})")
        for ex in r["examples"][:2]:
            log(f"      REF: {ex['ref'][:70]}")
            log(f"      HYP: {ex['hyp'][:70]}")

    log("\n" + "=" * 78)
    log("WER COMPARISON (independent judge: Whisper-large-v3, NOT the O-F training CTC head)")
    log("=" * 78)
    scored = {k: v for k, v in results.items() if "error" not in v}
    for label, r in sorted(scored.items(), key=lambda kv: kv[1]["wer"]):
        log(f"  {label:45s} WER={r['wer']:.4f}  n={r['n_scored']}")
    log("=" * 78)
    log(f"elapsed {(time.time()-t0)/60:.1f} min")

    out_path = Path(args.out) if args.out else Path(args.dirs[0]).parent / "wer_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    log(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
