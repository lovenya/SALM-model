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
from collections import Counter, defaultdict
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
    caps = {r["sample_id"]: (r.get("audio_cap") or "") for r in rows}
    out = {r["sample_id"]: (normalize_text(a).split(), normalize_text(b).split())
           for r, a, b in zip(rows, wh, ct)}
    out["__captions__"] = caps          # sentinel key, stripped by every consumer
    return out


def curve_cross_seed(per_lane: list[dict]) -> dict[int, float]:
    """Agreement between seed lanes at each position, averaged over clips."""
    by_clip = defaultdict(list)
    for lane in per_lane:
        for sid, (wh, _) in ((k, v) for k, v in lane.items() if k != "__captions__"):
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
        for _sid, (wh, ct) in ((k, v) for k, v in lane.items() if k != "__captions__"):
            for p in range(min(len(wh), len(ct), MAXPOS)):
                pos[p].append(1.0 if wh[p] == ct[p] else 0.0)
    return {p: st.mean(v) for p, v in sorted(pos.items()) if len(v) >= MIN_OBS}


def cross_clip_diversity(per_lane: list[dict]) -> float:
    """
    Mean normalised word-edit distance between DIFFERENT clips' transcripts in the same lane.

    The guard against greedy's specific failure. A deterministic decoder maximises coherence by
    always choosing the most likely continuation -- and the most likely continuation given "a
    vehicle engine" may be the same sentence for every vehicle clip in the set. That would
    sound excellent one clip at a time and be worthless: a model that has learned one utterance
    per scene TYPE, not one per scene.

    1.0 means every clip got its own utterance. Near 0 means the lane collapsed to a single
    sentence. Sampled lanes get this for free from their noise; greedy has to earn it, so this
    number is not optional when reading a greedy lane.
    """
    from salm.a2s.wer import word_edit_distance
    texts = []
    for lane in per_lane:
        for sid, (wh, _) in ((k, v) for k, v in lane.items() if k != "__captions__"):
            texts.append((sid, wh))
    dists = []
    for (sa, a), (sb, b) in itertools.combinations(texts, 2):
        if sa == sb:
            continue           # same clip across seeds is diversity we already measure
        dists.append(word_edit_distance(a, b) / max(len(a), len(b), 1))
    return st.mean(dists) if dists else float("nan")


STOP = {"a","an","the","is","are","was","were","of","in","on","at","and","or","with","by",
        "as","to","for","from","it","its","this","that","while","being","made","sound",
        "sounds","noise","occurring","present","followed","then","some","someone"}


def diversity_by_scene_similarity(per_lane: list[dict], thresh: float = 0.34) -> dict:
    """
    Split cross-clip diversity by how SIMILAR the two scenes were.

    `cross_clip_diversity` averages over all clip pairs, and in a set of 128 scenes almost
    every pair is unrelated -- so a model that emits one stock sentence per scene FAMILY still
    scores near 1.0. That is exactly what happened on job 443475: diversity read 0.931 while
    four openings covered 46% of clips and four different engine scenes all produced variants
    of "hold on tight, we're picking up speed".

    Rather than hand-defining families (which begs the question), this buckets pairs by caption
    word-overlap (Jaccard over content words) and reports transcript diversity in each bucket.
    A healthy model looks the same in both. A model reciting per-family stock phrases shows a
    clear drop in the SIMILAR bucket, and that drop is the number to watch.
    """
    from salm.a2s.wer import word_edit_distance
    caps, texts = {}, []
    for lane in per_lane:
        caps.update(lane.get("__captions__", {}))
        for sid, (wh, _) in ((k, v) for k, v in lane.items() if k != "__captions__"):
            texts.append((sid, wh))
    def stem(w: str) -> str:
        """Crude suffix stripping. Ambient captions are full of morphological variants --
        bird/birds, chirp/chirps/chirping, bark/barking -- and word-level Jaccard scores
        those pairs as completely unrelated, which puts genuinely similar scenes in the
        'different' bucket and hides exactly the collapse this function looks for."""
        for suf in ("ing", "ers", "er", "es", "s"):
            if len(w) > len(suf) + 2 and w.endswith(suf):
                return w[: -len(suf)]
        return w

    words = {sid: {stem(w) for w in normalize_text(c).split() if w not in STOP}
             for sid, c in caps.items()}
    sim_d, dif_d = [], []
    for (sa, a), (sb, b) in itertools.combinations(texts, 2):
        if sa == sb:
            continue
        wa, wb = words.get(sa, set()), words.get(sb, set())
        if not wa or not wb:
            continue
        j = len(wa & wb) / len(wa | wb)
        d = word_edit_distance(a, b) / max(len(a), len(b), 1)
        (sim_d if j >= thresh else dif_d).append(d)
    out = {"threshold": thresh, "n_similar_pairs": len(sim_d), "n_different_pairs": len(dif_d)}
    out["diversity_similar_scenes"] = st.mean(sim_d) if sim_d else float("nan")
    out["diversity_different_scenes"] = st.mean(dif_d) if dif_d else float("nan")
    out["family_collapse_gap"] = (out["diversity_different_scenes"]
                                  - out["diversity_similar_scenes"]) if sim_d and dif_d else float("nan")
    return out


def opening_concentration(per_lane: list[dict], n_words: int = 4, top: int = 4) -> dict:
    """
    What fraction of clips share one of the `top` most common openings?

    The defect in its most direct form. On job 443475's greedy lane, four 4-word openings
    covered 46% of 128 clips and "hey can you hear" alone accounted for 39 -- a model with a
    small stock of safe entry phrases. Deterministic decoding causes this by construction:
    argmax takes the single most likely continuation, and at position 0 with no context that
    is simply the most frequent opening in the training data.

    A varied model should sit low here. This is the number nucleus sampling has to move.
    """
    seen: dict[str, str] = {}
    for lane in per_lane:
        for sid, (wh, _) in ((k, v) for k, v in lane.items() if k != "__captions__"):
            if len(wh) >= n_words:
                seen.setdefault(sid, " ".join(wh[:n_words]))
    if not seen:
        return {"opening_top4_share": float("nan"), "opening_most_common": "", "n_clips": 0}
    c = Counter(seen.values()).most_common(top)
    return {
        "opening_top4_share": sum(v for _, v in c) / len(seen),
        "opening_most_common": c[0][0] if c else "",
        "opening_most_common_count": c[0][1] if c else 0,
        "n_clips": len(seen),
    }


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
            "cross_clip_diversity": cross_clip_diversity(lanes),
            **diversity_by_scene_similarity(lanes),
            **opening_concentration(lanes),
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
        log(f"  {'':22s} cross-clip diversity {r['cross_clip_diversity']:.3f}  "
            f"(1.0 = every clip its own utterance; near 0 = collapsed to one sentence)")
        log(f"  {'':22s}   by scene: similar {r['diversity_similar_scenes']:.3f} "
            f"({r['n_similar_pairs']} pairs) | different {r['diversity_different_scenes']:.3f} "
            f"({r['n_different_pairs']} pairs) | COLLAPSE GAP {r['family_collapse_gap']:+.3f}")
        log(f"  {'':22s}   top-4 openings cover {r['opening_top4_share']:.0%} of "
            f"{r['n_clips']} clips | most common x{r['opening_most_common_count']}: "
            f"\"{r['opening_most_common']}\"")
        log("")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
