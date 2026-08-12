"""
Summarise a multi-lane generation run: what was produced, and what it cost.

Re-runnable at any time against an existing output tree, so results can be re-read without
regenerating. Timing is reported deliberately -- generation cost is one of the numbers we claim
against a multi-stage cascade, so it needs measuring rather than estimating.

Usage:
    python scripts/report_generation_evolution.py --gen-root <dir> [--total-seconds N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from package_generation_evolution import lane_sort_key  # noqa: E402  (same directory)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-root", required=True)
    ap.add_argument("--total-seconds", type=float, default=None,
                    help="wall-clock for the whole job, including parallelism across GPUs")
    args = ap.parse_args()

    gen_root = Path(args.gen_root)
    lanes = sorted(
        (d for d in gen_root.iterdir() if d.is_dir() and (d / "summary.json").exists()),
        key=lambda d: lane_sort_key(d.name),
    )
    if not lanes:
        print(f"ERROR: no lanes with summary.json under {gen_root}")
        return 1

    hdr = (f"{'lane':<30} {'epoch':>6} {'trained':>8} {'n':>4} "
           f"{'rms_mean':>9} {'silent':>7} {'sec':>7} {'s/samp':>7}")
    print(hdr)
    print("-" * len(hdr))

    serial = 0.0
    rows = []
    for lane in lanes:
        s = json.loads((lane / "summary.json").read_text())
        n = s.get("num_samples", 0)
        el = float(s.get("elapsed_seconds", 0.0))
        serial += el
        trained = s.get("unet_weights_loaded")
        trained_s = "-" if trained is None else ("yes" if trained else "NO")
        print(f"{lane.name:<30} {str(s.get('checkpoint_epoch','-')):>6} {trained_s:>8} {n:>4} "
              f"{s.get('rms_mean',0):>9.5f} {s.get('near_silent_count',0):>7} "
              f"{el:>7.1f} {el/max(n,1):>7.2f}")
        rows.append(s)

    print()
    print(f"lanes: {len(lanes)}   summed per-lane sampling time: {serial:.1f}s "
          f"({serial/60:.1f} min)")
    if args.total_seconds:
        print(f"wall-clock across 4 GPUs: {args.total_seconds:.0f}s "
              f"({args.total_seconds/60:.1f} min)  "
              f"speedup {serial/max(args.total_seconds,1e-9):.2f}x")
        per = [r for r in rows if r.get("num_samples")]
        if per:
            n = per[0]["num_samples"]
            print(f"cost per generated clip, one lane: "
                  f"{serial/len(lanes)/max(n,1):.2f}s")

    # The things most likely to indicate something is wrong.
    silent = [(l.name, json.loads((l / 'summary.json').read_text()).get("near_silent_count", 0))
              for l in lanes]
    bad = [(n, c) for n, c in silent if c]
    if bad:
        print("\nWARNING near-silent generations present (broken generation path?):")
        for n, c in bad:
            print(f"  {n}: {c}")

    untrained = [l.name for l in lanes
                 if json.loads((l / 'summary.json').read_text()).get("mode") == "s2a"
                 and json.loads((l / 'summary.json').read_text()).get("unet_weights_loaded") is False
                 and not json.loads((l / 'summary.json').read_text()).get("skip_unet_weights")]
    if untrained:
        print("\nWARNING these s2a lanes did NOT load trained U-Net weights and did not ask to skip:")
        for n in untrained:
            print(f"  {n}  <- checkpoint may lack a 'unet' key")

    listen = gen_root / "listen"
    if listen.exists():
        n_dirs = sum(1 for d in listen.iterdir() if d.is_dir())
        print(f"\nlistening tree: {listen}  ({n_dirs} samples)")
        print(f"pull down with:  rsync -avL <host>:{listen}/ ./listen/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
