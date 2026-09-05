#!/usr/bin/env python
"""
Does the content LM REPRODUCE its training utterances, and does it reproduce LONG spans?

WHY THIS EXISTS. Every dedup/frame-level run overfits severely -- dedup_k512 ends at
train_code_loss 0.21 against val_code_loss 3.07, with the val minimum at epoch 9 of 100. So
"the model has memorised the training set" is established from the loss curves. What the loss
curves cannot say is what memorisation SOUNDS like, and that is the question that matters:

  If a heavily-overfit checkpoint, generating from a TRAIN ambient clip, reproduces long
  fluent spans of that clip's true utterance, then the codebook + renderer CAN carry long
  coherent speech and the only broken thing is generalisation.

  If it still emits 3-5 word islands even on train -- where it has all but memorised the token
  sequence -- then the bottleneck is downstream of the LM, and no amount of better planning or
  more data fixes it.

Those two outcomes point at completely different next experiments, which is why this is worth
measuring rather than arguing about.

TWO METRICS, EACH WITH ITS OWN MISMATCHED CONTROL. An absolute WER against a reference is
uninterpretable here -- these clips are all short conversational English, so even unrelated
utterances share function words and score better than chance. Every number below is therefore
reported as a MATCHED-minus-MISMATCHED pair, where the mismatched arm scores the same
generated audio against a DERANGED assignment of the same lane's references (no clip keeps its
own). The derangement is seeded, and identity is impossible by construction, so the control is
drawn from exactly the same text distribution and differs in one thing only: whether the
reference belongs to the clip.

  1. ref_wer            -- word error rate of Whisper(generated) against the reference text.
                           Measures whether the right words came out at all.
  2. longest_span       -- longest run of consecutive words shared by Whisper(generated) and
                           the reference. This is the granularity question stated numerically:
                           the listening complaint is "3-4 words make sense, then another
                           group starts", so a matched-minus-mismatched span gap of ~3 words
                           is that complaint, and ~8 is it being solved.

Neither metric is meaningful on val/test -- the model has never seen those utterances and has
no reason to reproduce them, so both arms should collapse together. That collapse IS the
control for the train numbers: a train gap with no val gap is memorisation, whereas a gap in
both would mean the metric is picking up something generic and must be discarded.

Usage:
  python scripts/a2s/score_memorisation.py --dirs DIR [DIR ...] --device cuda \
      --out /path/to/memorisation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from salm.a2s.asr import WhisperASR  # noqa: E402
from salm.a2s.wer import normalize_text, word_edit_distance  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


def derangement(n: int, seed: int) -> np.ndarray:
    """
    A permutation with no fixed point, so the mismatched arm never accidentally scores a clip
    against its own reference. Built as a single random n-cycle rather than by rejection
    sampling: a cycle of length n is deranged by construction, so this cannot loop forever and
    cannot silently return the identity on a small n (rejection sampling on n=2 or n=3 is where
    that bites).
    """
    if n < 2:
        return np.arange(n)
    order = np.random.default_rng(seed).permutation(n)
    out = np.empty(n, dtype=int)
    out[order] = np.roll(order, 1)
    return out


def longest_common_span(a: list[str], b: list[str]) -> int:
    """Longest run of CONSECUTIVE words present in both -- classic LCSubstring DP, on words."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def score_one_dir(gen_dir: Path, whisper: WhisperASR, batch_size: int, seed: int) -> dict:
    manifest_path = gen_dir / "manifest.jsonl"
    if not manifest_path.exists():
        return {"error": f"no manifest.jsonl in {gen_dir}"}
    rows = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("status") == "success" and r.get("speech_text")]
    if len(rows) < 2:
        return {"error": f"need >=2 rows with speech_text in {gen_dir}, got {len(rows)}"}

    wavs = []
    for r in rows:
        wav, _sr = sf.read(gen_dir / r["generated_file_rel"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wavs.append(wav)

    hyps = [normalize_text(h).split() for h in whisper.transcribe(wavs, batch_size=batch_size)]
    refs = [normalize_text(r["speech_text"]).split() for r in rows]
    perm = derangement(len(rows), seed)

    def arm(ref_for: list[list[str]]) -> tuple[list[float], list[int]]:
        wers = [word_edit_distance(h, r) / max(len(r), 1) for h, r in zip(hyps, ref_for)]
        spans = [longest_common_span(h, r) for h, r in zip(hyps, ref_for)]
        return wers, spans

    m_wer, m_span = arm(refs)
    x_wer, x_span = arm([refs[j] for j in perm])

    return {
        "n_scored": len(rows),
        "ref_wer_matched": float(np.mean(m_wer)),
        "ref_wer_mismatched": float(np.mean(x_wer)),
        # Negative = matched is BETTER = the clip's own words came out. This is the signal.
        "ref_wer_gap": float(np.mean(m_wer) - np.mean(x_wer)),
        "longest_span_matched": float(np.mean(m_span)),
        "longest_span_mismatched": float(np.mean(x_span)),
        # Positive = the clip's own words came out in a LONGER unbroken run than chance.
        "longest_span_gap": float(np.mean(m_span) - np.mean(x_span)),
        "longest_span_max": int(np.max(m_span)),
        "examples": [
            {"sample_id": r["sample_id"], "whisper": " ".join(h), "reference": " ".join(rf),
             "ref_wer": round(w, 3), "longest_span": s}
            for r, h, rf, w, s in zip(rows, hyps, refs, m_wer, m_span)
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    log("loading Whisper ...")
    whisper = WhisperASR(device=args.device)

    scored: dict[str, dict] = {}
    for d in args.dirs:
        gen_dir = Path(d)
        label = f"{gen_dir.parent.name}/{gen_dir.name}"
        log(f"\n[{label}]")
        r = score_one_dir(gen_dir, whisper, args.batch_size, args.seed)
        scored[label] = r
        if "error" in r:
            log(f"    SKIP: {r['error']}")
            continue
        log(f"    ref_wer      matched {r['ref_wer_matched']:.3f}  "
            f"mismatched {r['ref_wer_mismatched']:.3f}  gap {r['ref_wer_gap']:+.3f}")
        log(f"    longest_span matched {r['longest_span_matched']:.2f}  "
            f"mismatched {r['longest_span_mismatched']:.2f}  "
            f"gap {r['longest_span_gap']:+.2f}  (max {r['longest_span_max']})")

    log("\n" + "=" * 78)
    log("MEMORISATION / SPAN SUMMARY   (gap = matched - mismatched)")
    log("  ref_wer_gap      NEGATIVE = the clip's own words came out")
    log("  longest_span_gap POSITIVE = they came out in an unbroken run, in words")
    log("=" * 78)
    log(f"  {'lane':52s} {'wer_gap':>9s} {'span_gap':>9s} {'span':>6s}")
    ok = {k: v for k, v in scored.items() if "error" not in v}
    for label, r in sorted(ok.items(), key=lambda kv: kv[1]["longest_span_gap"], reverse=True):
        log(f"  {label:52s} {r['ref_wer_gap']:+9.3f} {r['longest_span_gap']:+9.2f} "
            f"{r['longest_span_matched']:6.2f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(scored, indent=2))
        log(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
