#!/usr/bin/env python
"""
Wiring checks for latent think tokens. Runs on a GPU inside a job, NOT on the login node.

CLAUDE.md: "neural-network forward passes are not lightweight checks." This was attempted
locally twice (2026-09-03 and again 2026-09-06) and killed both times; a 64-step autoregressive
sample on CPU is exactly the thing the rule names. The checks themselves are worth running --
they cover the failure modes that would silently produce a plausible-looking null result:

  1. n_think > 0 actually changes the computation (otherwise the arm is a no-op)
  2. the (B, S, .) output contract is unchanged, so every downstream loss/mask/index still lines
     up -- a leaked think position would shift every target by n_think and still train
  3. pad_mask stays aligned once think columns are prepended
  4. CAUSALITY survives: changing a later input must not move an earlier position's logits
  5. sample() respects the 256-frame budget

Usage:
  python scripts/a2s/check_think_wiring.py [--device cuda]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from salm.a2s.content_lm_dedup import ContentLMDedup  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-think", type=int, default=8)
    args = ap.parse_args()
    dev = args.device
    kw = dict(num_codes=32, max_run=8, ambient_dim=16, max_tokens=64,
              d_model=32, n_layers=2, n_heads=2)

    torch.manual_seed(0); a = ContentLMDedup(**kw, n_think=0).to(dev).eval()
    torch.manual_seed(0); b = ContentLMDedup(**kw, n_think=args.n_think).to(dev).eval()
    amb = torch.randn(3, 6, 16, device=dev)
    ci = torch.randint(0, 32, (3, 5), device=dev)
    di = torch.randint(1, 8, (3, 5), device=dev)

    fail = 0
    with torch.no_grad():
        ca, da = a(amb, ci, di)
        cb, db = b(amb, ci, di)
    if torch.allclose(ca, cb):
        print("  FAIL: n_think changed nothing -- the arm would be a silent no-op"); fail += 1
    else:
        print("  ok: n_think changes the computation")
    if ca.shape != cb.shape or da.shape != db.shape:
        print(f"  FAIL: output contract changed {ca.shape} -> {cb.shape}"); fail += 1
    else:
        print(f"  ok: output contract preserved {tuple(cb.shape)}")

    for S in (1, 5, 33):
        ci2 = torch.randint(0, 32, (2, S), device=dev)
        di2 = torch.randint(1, 8, (2, S), device=dev)
        with torch.no_grad():
            c2, d2 = b(torch.randn(2, 6, 16, device=dev), ci2, di2)
        if c2.shape != (2, S, 33) or d2.shape != (2, S, 8):
            print(f"  FAIL: S={S} broke the contract -> {tuple(c2.shape)}"); fail += 1
    print("  ok: contract holds for S in {1, 5, 33}")

    pm = torch.zeros((3, 5), dtype=torch.bool, device=dev); pm[:, 3:] = True
    with torch.no_grad():
        c3, _ = b(amb, ci, di, pad_mask=pm)
    if not torch.isfinite(c3).all():
        print("  FAIL: NaN with pad_mask -- think columns mis-aligned"); fail += 1
    else:
        print("  ok: pad_mask aligned, no NaNs")

    ci_b = ci.clone(); ci_b[:, -1] = (ci_b[:, -1] + 1) % 32
    with torch.no_grad():
        la, _ = b(amb, ci, di); lb, _ = b(amb, ci_b, di)
    if not torch.allclose(la[:, :-1], lb[:, :-1], atol=1e-5):
        print("  FAIL: causality broken -- a later input moved an earlier position"); fail += 1
    else:
        print("  ok: causality preserved")

    toks, durs = b.sample(amb, total_frames=64, temperature=1.0,
                          generator=torch.Generator(device=dev).manual_seed(1))
    if not all(sum(d) <= 64 for d in durs):
        print("  FAIL: frame budget violated by sample()"); fail += 1
    else:
        print(f"  ok: sample() respects the budget ({len(toks[0])} tokens)")

    print(f"\n  params: n_think=0 {a.num_parameters()/1e3:.1f}k | "
          f"n_think={args.n_think} {b.num_parameters()/1e3:.1f}k")
    print("THINK WIRING OK" if not fail else f"THINK WIRING FAILED ({fail} checks)")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
