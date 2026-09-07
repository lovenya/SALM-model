#!/usr/bin/env python
"""
Paired, length-adjusted comparison of generation lanes.

WHY THIS EXISTS. Job 443751 (the plan vector) was declared null on `horizon_cross_seed`, a
statistic that has since been shown to return 1 or 0 for every arm and every control ever
measured. The same job's audio scores were written to disk and never read; they showed the
treatment beating its own matched control by 0.077 inter_asr_dist -- comparable to the greedy
decoding win that reframed the project. The direction was closed and the checkpoints deleted.

Two lessons are built into this script:

  1. COMPARE PAIRED, ON THE SAME CLIPS. Lane means hide which clips moved. The run-to-run floor
     on unmatched comparisons is ~0.02-0.03 (measured: the same LM and renderer family across
     two jobs differ by that much on identical clips through diffusion noise alone), so an
     unpaired difference smaller than that is not evidence.

  2. CONTROL FOR LENGTH. inter_asr_dist rises about +0.0089 per generated word (r = +0.31
     pooled): a model that simply says less scores better without being better. That is exactly
     what the coarse-to-fine result turned out to be. The length-adjusted effect here is the
     intercept of a paired regression of delta-score on delta-words -- the effect at equal
     length -- and it is reported next to the raw delta, never instead of it.

Also reports lexical diversity, because a decoder can buy a better score by retreating into
stock phrases and that must be visible rather than laundered into a win.

Usage:
  python scripts/a2s/compare_arms.py --scores FILE.json --control NAME --treatment NAME [...]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from salm.a2s.wer import normalize_text  # noqa: E402


def lane(scores: dict, name: str) -> dict:
    """-> {sample_id: (inter_asr_dist, word_count, words)} for the lane whose key ends in name."""
    keys = [k for k in scores if k.split("/")[-1] == name]
    if not keys:
        raise SystemExit(f"lane {name!r} not in the scores file; have: "
                         f"{sorted(k.split('/')[-1] for k in scores)}")
    r = scores[keys[0]]
    if "error" in r:
        raise SystemExit(f"lane {name!r} has an error: {r['error']}")
    out = {}
    for e in r["examples"]:
        w = normalize_text(e["whisper"]).split()
        out[e["sample_id"]] = (float(e["inter_asr_dist"]), len(w), w)
    return out


def boot(x: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def diversity(words_list: list[list[str]]) -> dict:
    toks = [w for ws in words_list for w in ws]
    opens = Counter(" ".join(ws[:3]) for ws in words_list if len(ws) >= 3)
    top4 = sum(v for _, v in opens.most_common(4)) / max(len(words_list), 1)
    return {"type_token": len(set(toks)) / max(len(toks), 1),
            "unique_openings": len(opens), "top4_opening_share": top4}


def compare(ctrl: dict, treat: dict, label: str) -> dict:
    ids = sorted(set(ctrl) & set(treat))
    if len(ids) < 8:
        raise SystemExit(f"{label}: only {len(ids)} shared clips -- refusing to compare")
    dc = np.array([treat[i][0] - ctrl[i][0] for i in ids])
    dw = np.array([treat[i][1] - ctrl[i][1] for i in ids])
    lo, hi = boot(dc)
    # Length-adjusted effect: the intercept of delta-score regressed on delta-words, i.e. the
    # effect at equal utterance length. Degenerate if delta-words has no spread, so guard it.
    if float(np.std(dw)) < 1e-9:
        adj = float(dc.mean())
    else:
        slope, intercept = np.polyfit(dw, dc, 1)
        adj = float(intercept)
    return {
        "label": label, "n": len(ids),
        "raw_delta": float(dc.mean()), "ci_lo": lo, "ci_hi": hi,
        "length_adjusted": adj,
        "clips_better": int((dc < 0).sum()),
        "mean_words_ctrl": float(np.mean([ctrl[i][1] for i in ids])),
        "mean_words_treat": float(np.mean([treat[i][1] for i in ids])),
        "div_ctrl": diversity([ctrl[i][2] for i in ids]),
        "div_treat": diversity([treat[i][2] for i in ids]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--treatment", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    scores = json.loads(Path(args.scores).read_text())
    ctrl = lane(scores, args.control)
    results = [compare(ctrl, lane(scores, t), t) for t in args.treatment]

    print(f"\n  control: {args.control}  (n={len(ctrl)} clips)")
    print(f"  lower inter_asr_dist is better; negative delta = the treatment wins\n")
    print(f"  {'treatment':26s} {'raw':>8s} {'95% CI':>18s} {'len-adj':>9s} "
          f"{'better':>8s} {'words':>13s}")
    for r in results:
        print(f"  {r['label']:26s} {r['raw_delta']:+8.4f} "
              f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]".rjust(19)
              + f" {r['length_adjusted']:+9.4f} {r['clips_better']:3d}/{r['n']:<4d}"
              + f" {r['mean_words_ctrl']:5.1f}->{r['mean_words_treat']:5.1f}")
    print()
    print(f"  DIVERSITY -- a decoder can buy a better score by saying safer things")
    print(f"  {'lane':26s} {'type/token':>11s} {'uniq opens':>11s} {'top4 share':>11s}")
    d = results[0]["div_ctrl"]
    print(f"  {args.control:26s} {d['type_token']:11.3f} {d['unique_openings']:11d} "
          f"{d['top4_opening_share']:11.1%}")
    for r in results:
        d = r["div_treat"]
        print(f"  {r['label']:26s} {d['type_token']:11.3f} {d['unique_openings']:11d} "
              f"{d['top4_opening_share']:11.1%}")
    print()
    print("  A CI that spans 0 is not a result. A raw win with a length-adjusted effect near 0")
    print("  means the treatment only said less. Diversity falling means it said safer things.")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
