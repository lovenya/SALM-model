#!/usr/bin/env python
"""
How many words does the scene actually control? Two positional curves, one job.

WHY. Job 442814 established that the failure is not clarity and not overfitting. Train and val
score identically (ref_wer 1.29 vs 1.32), so memorisation buys nothing; and positional
inter-ASR agreement on the strongest lane tracks the REAL-SPEECH curve out to word 10, so the
phonemes stay clean the whole way. What collapses is DETERMINISM. Six seeds on one fixed
ambient clip agree on the opening words and then diverge completely:

    position     1     2     3     4     5     6     7     8
    agreement  0.20  0.20  0.17  0.12  0.00  0.00  0.02  0.00

Four words, then a cliff. The model stops using the ambient long before it stops speaking, and
that is a different bug from the one this project has been chasing.

TWO CURVES, MEASURING DIFFERENT THINGS. Reporting either alone has already misled us once.

  1. cross-seed agreement -- same ambient, different noise. Fraction of seed PAIRS emitting the
     same word at position p. This is SCENE CONTROL: high means the scene decided the word,
     low means the model chose freely. Undefined for a deterministic decoder (every seed is
     identical), so greedy lanes are reported as n/a rather than a meaningless 1.00.
  2. positional inter-ASR agreement -- Whisper vs the CTC probe on the SAME audio. This is
     CLARITY: two independently trained recognisers agreeing means real phonemes are present.
     Defined for any decoder.

Read together they separate the two failure modes. Curve 2 collapsing means the audio degrades.
Curve 1 collapsing while curve 2 holds means the audio is fine and the CONDITIONING lapsed.

NEITHER USES A REFERENCE, deliberately. The scene-to-utterance mapping is one-to-many -- job
442799 measured exact-word recovery at f1 0.16 even when given the ground-truth caption -- so
scoring against the held-out utterance mostly measures which of many valid utterances was
drawn. Both curves here ask whether the model is self-consistent and articulate, not whether it
guessed the one transcript on file.

Usage:
  python scripts/a2s/score_plan_horizon.py --group NAME DIR [DIR ...] [--group ...] \
      --ctc-checkpoint CKPT --device cuda --out horizon.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from configs import paths as project_paths  # noqa: E402
from salm.a2s.asr import WhisperASR  # noqa: E402
from salm.a2s.ctc_probe import CTCProbeTranscriber  # noqa: E402
from salm.a2s.wer import normalize_text  # noqa: E402

MAXPOS = 14
# Minimum sample size before a position's rate is reported at all. The two curves count
# different units -- cross-seed counts seed PAIRS (n lanes give n*(n-1)/2), inter-ASR counts
# individual CLIPS -- so they are named separately rather than sharing one constant that
# silently means two things.
MIN_PAIRS = 6      # cross-seed: 4 seed lanes give exactly 6 pairs
MIN_OBS = 6        # inter-ASR: 6 clips reaching this position
HORIZON_THRESHOLD = 0.10


def log(msg: str) -> None:
    print(msg, flush=True)


def transcribe_dir(gen_dir: Path, whisper, ctc, batch_size: int):
    """-> {sample_id: (whisper_words, ctc_words)}, or {} if the lane has no manifest."""
    mp = gen_dir / "manifest.jsonl"
    if not mp.exists():
        return {}
    rows = [json.loads(l) for l in mp.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("status") == "success"]
    if not rows:
        return {}
    wavs = []
    for r in rows:
        w, _ = sf.read(gen_dir / r["generated_file_rel"], dtype="float32")
        wavs.append(w.mean(axis=1) if w.ndim > 1 else w)
    wh = whisper.transcribe(wavs, batch_size=batch_size)
    ct = ctc.transcribe(wavs, batch_size=batch_size)
    return {r["sample_id"]: (normalize_text(a).split(), normalize_text(b).split())
            for r, a, b in zip(rows, wh, ct)}


def curve_cross_seed(per_lane: list[dict]) -> dict[int, float]:
    """Agreement between seed lanes at each position, averaged over clips."""
    by_clip = defaultdict(list)
    for lane in per_lane:
        for sid, (wh, _) in lane.items():
            by_clip[sid].append(wh)
    pos = defaultdict(list)
    for sid, variants in by_clip.items():
        if len(variants) < 2:
            continue
        for p in range(MAXPOS):
            hits = tot = 0
            for a, b in itertools.combinations(variants, 2):
                if p < len(a) and p < len(b):
                    tot += 1
                    hits += (a[p] == b[p])
            if tot:
                pos[p].append((hits, tot))
    out = {}
    for p, vals in pos.items():
        tot = sum(t for _, t in vals)
        if tot >= MIN_PAIRS:
            out[p] = sum(h for h, _ in vals) / tot
    return out


def curve_inter_asr(per_lane: list[dict]) -> dict[int, float]:
    pos = defaultdict(list)
    for lane in per_lane:
        for _sid, (wh, ct) in lane.items():
            for p in range(min(len(wh), len(ct), MAXPOS)):
                pos[p].append(1.0 if wh[p] == ct[p] else 0.0)
    return {p: st.mean(v) for p, v in sorted(pos.items()) if len(v) >= MIN_OBS}


def horizon(curve: dict[int, float]) -> int:
    """Last position (1-indexed) at or above threshold, counting from the start unbroken."""
    n = 0
    for p in range(MAXPOS):
        if curve.get(p, 0.0) >= HORIZON_THRESHOLD:
            n = p + 1
        else:
            break
    return n


def fmt(curve: dict[int, float], n: int = 12) -> str:
    return "".join(f"{curve[p]:5.2f}" if p in curve else "    ." for p in range(n))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", nargs="+", action="append", required=True,
                    metavar=("NAME", "DIR"),
                    help="group label followed by its lane dirs (repeatable)")
    ap.add_argument("--ctc-checkpoint", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--deterministic", nargs="*", default=[],
                    help="group names whose decoder is deterministic; cross-seed is n/a there")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    log("loading Whisper and the CTC probe ...")
    whisper = WhisperASR(project_paths.WHISPER_LARGE_V3_PATH, args.device)
    ctc = CTCProbeTranscriber(args.ctc_checkpoint, device=args.device)

    results = {}
    for grp in args.group:
        name, dirs = grp[0], grp[1:]
        lanes = []
        for d in dirs:
            t = transcribe_dir(Path(d), whisper, ctc, args.batch_size)
            if t:
                lanes.append(t)
        if not lanes:
            log(f"  [{name}] no scorable lanes"); continue
        det = name in args.deterministic
        cs = {} if det else curve_cross_seed(lanes)
        ia = curve_inter_asr(lanes)
        results[name] = {
            "n_lanes": len(lanes),
            "n_clips": len({s for l in lanes for s in l}),
            "deterministic": det,
            "cross_seed": {str(k): v for k, v in cs.items()},
            "inter_asr": {str(k): v for k, v in ia.items()},
            "horizon_cross_seed": None if det else horizon(cs),
            "horizon_inter_asr": horizon(ia),
        }
        log(f"  [{name}] {len(lanes)} lanes, {results[name]['n_clips']} clips")

    log("\n" + "=" * 84)
    log("PLAN HORIZON   (word position ->)")
    log(f"  cross-seed = scene control (same ambient, different noise). Threshold "
        f"{HORIZON_THRESHOLD} for the horizon.")
    log("  inter-ASR  = clarity (Whisper vs CTC on the same audio). Real speech ~0.88 falling")
    log("               to ~0.45; the zero-content control is 0.00 everywhere.")
    log("=" * 84)
    head = "".join(f"{p+1:>5}" for p in range(12))
    log(f"  {'group':22s} {'curve':10s} {head}   horizon")
    for name, r in results.items():
        if r["deterministic"]:
            log(f"  {name:22s} {'cross-seed':10s} " + "  n/a (deterministic decoder)")
        else:
            cs = {int(k): v for k, v in r["cross_seed"].items()}
            log(f"  {name:22s} {'cross-seed':10s} {fmt(cs)}   {r['horizon_cross_seed']}")
        ia = {int(k): v for k, v in r["inter_asr"].items()}
        log(f"  {'':22s} {'inter-ASR':10s} {fmt(ia)}   {r['horizon_inter_asr']}")
        log("")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
