#!/usr/bin/env python
"""
How wrong are the content LM's codes, and are the errors NEAR misses?

Calibration for renderer corruption augmentation. `code_corruption.corrupt_codes` substitutes
nearest neighbours in codebook space, which simulates a specific KIND of error. If the LM's
actual mistakes are mostly far misses -- acoustically unrelated codes -- then that augmentation
is modelling the wrong thing and should not be run, whatever the rate is set to. This decides
that before any GPU time goes into fine-tuning.

Reports the raw disagreement rate, and the share of disagreements landing on one of the true
code's k nearest neighbours.

Usage:
  python scripts/a2s/measure_lm_code_error.py --lm-checkpoint CKPT --split val [--num 256]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from configs import paths as project_paths  # noqa: E402
from salm.a2s.code_corruption import measure_code_error_rate, neighbour_table  # noqa: E402
from salm.a2s.content_dedup import expand_sequence  # noqa: E402
from salm.a2s.content_lm_dedup import ContentLMDedup  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm-checkpoint", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--num", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--greedy", action="store_true", default=True)
    ap.add_argument("--neighbours", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pay = torch.load(args.lm_checkpoint, map_location="cpu", weights_only=False)
    K = int(pay["num_codes"])
    model = ContentLMDedup(
        num_codes=K, max_run=int(pay["max_run"]), ambient_dim=int(pay["ambient_dim"]),
        max_tokens=int(pay["max_tokens"]), d_model=int(pay["d_model"]),
        n_layers=int(pay["layers"]), n_heads=int(pay["heads"]),
        use_plan=bool(pay.get("use_plan", False)),
    ).to(args.device)
    model.load_state_dict(pay["model"])
    model.eval()

    true = np.load(project_paths.OUTPUTS_ROOT / "content_codes" /
                   f"codes_codebook_k{K}_{args.split}.npy")
    # Same path the dedup trainer reads (train_content_lm_dedup.load_split), so the ambient
    # rows are the ones the LM was actually trained against. Guessing a cache path here would
    # silently compare against a different row ordering.
    amb = np.load(project_paths.OUTPUTS_ROOT / "features_a2s_cond" / args.split / "cond_seq.npy",
                  mmap_mode="r")
    if amb.shape[0] != true.shape[0]:
        raise SystemExit(
            f"{args.split}: ambient {amb.shape[0]} rows vs codes {true.shape[0]} -- row "
            "alignment is not optional here"
        )
    n = min(args.num, true.shape[0], amb.shape[0])
    print(f"comparing {n} clips, K={K}, greedy={args.greedy}", flush=True)

    preds = []
    for i in range(0, n, args.batch_size):
        a = torch.from_numpy(np.asarray(amb[i : i + args.batch_size], dtype=np.float32))
        toks, durs = model.sample(
            a.to(args.device), total_frames=true.shape[1], temperature=1.0,
            top_k=1 if args.greedy else 0,
        )
        preds.append(np.stack([
            expand_sequence(np.asarray(t), np.asarray(d), total=true.shape[1])
            for t, d in zip(toks, durs)
        ]))
    pred = np.concatenate(preds)[:n]

    nbrs = neighbour_table(pay["codes"], k=args.neighbours).numpy()
    r = measure_code_error_rate(pred, true[:n], nbrs)

    print()
    print(f"  frames compared            {r['frames']}")
    print(f"  code error rate            {r['error_rate']:.4f}")
    print(f"  of those, NEAR misses      {r['share_of_errors_that_are_near']:.4f} "
          f"(within the {args.neighbours} nearest codes)")
    print()
    if r["share_of_errors_that_are_near"] < 0.25:
        print("  *** MOSTLY FAR MISSES. Nearest-neighbour corruption models the wrong error;")
        print("      the corrupt arm will not simulate what the LM actually does. Read the")
        print("      quantization arm on its own and rethink the corruption design. ***")
    else:
        print(f"  -> nearest-neighbour corruption is a reasonable simulation. Set")
        print(f"     content_corrupt_rate near {r['error_rate']:.2f} to match.")
    if args.out:
        Path(args.out).write_text(json.dumps(r, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
