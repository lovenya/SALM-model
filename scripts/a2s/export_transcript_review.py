#!/usr/bin/env python
"""
Render scored generation lanes as a readable review document.

WHY. Judging 100+ generations from a JSON blob is impractical, and judging them by ear alone
costs hours. This lays each one out as scene caption -> what the model said -> what the CTC
probe heard -> what the human actually said, so a reviewer can scan for the failure the metrics
keep missing: audio that is acoustically fine and lexically empty.

The CTC line is not decoration. Whisper hallucinates fluent text from marginal audio -- it has
a language model and will invent a plausible sentence from mush. The CTC probe has no language
model, so where the two agree the words are really in the waveform, and where Whisper is fluent
and CTC is gibberish, Whisper is filling in. That contrast is the whole point of printing both.

Usage:
  python scripts/a2s/export_transcript_review.py --scores FILE.json --out REVIEW.md \
      [--lanes k2048__best_loss__greedy ...] [--limit 200]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="score_intelligibility.py output JSON")
    ap.add_argument("--gen-root", required=True, help="generation root holding the lane dirs")
    ap.add_argument("--lanes", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    scores = json.loads(Path(args.scores).read_text())
    root = Path(args.gen_root)
    out: list[str] = ["# Generation transcript review", ""]

    for label, r in scores.items():
        lane = label.split("/")[-1]
        if args.lanes and lane not in args.lanes:
            continue
        if "error" in r or "examples" not in r:
            continue
        mp = root / lane / "manifest.jsonl"
        meta = {}
        if mp.exists():
            for line in mp.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    meta[row["sample_id"]] = row

        ex = r["examples"][: args.limit]
        out += [
            f"## {lane}", "",
            f"`inter_asr_dist` {r['inter_asr_distance_mean']:.3f} "
            f"| `real_word_rate` {r['real_word_rate_mean']:.3f} | {len(ex)} shown "
            f"of {r['n_scored']}", "",
            "Reference points: real speech 0.315, VAE round-trip ceiling 0.372, "
            "zero-content floor 0.967.", "",
        ]
        # Worst first: the failures are what needs reading, and a reviewer who stops early
        # should have stopped on the bad ones, not the cherry-picked good ones.
        for e in sorted(ex, key=lambda x: -x["inter_asr_dist"]):
            m = meta.get(e["sample_id"], {})
            out += [
                f"### `{e['sample_id']}` — {m.get('audio_cap', '(no caption)')}", "",
                f"- **generated (Whisper):** {e['whisper']}",
                f"- **generated (CTC probe):** {e['ctc_probe']}",
                f"- **human reference:** {m.get('speech_text', '(unknown)')}",
                f"- dist `{e['inter_asr_dist']}` · rwr `{e['real_word_rate']}`", "",
            ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(out))
    print(f"wrote {args.out} ({Path(args.out).stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
