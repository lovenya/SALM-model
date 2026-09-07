#!/usr/bin/env python
"""
Train ContentLMHier: coarse-to-fine over run-length content tokens.

Reuses the dedup trainer's data loading so the caches, splits and row alignment are identical
and a comparison against `dedup_k*` is like-for-like.

The pre-registered claim is ORDERED, which is stronger than a single arm: the coarse pass helps
only if it fits inside the model's ~20-token plan horizon, so stride 8 (19.1 coarse tokens) and
stride 12 (12.9) should beat stride 6 (25.4, does not fit). If all three behave alike, the
horizon story is wrong and the mechanism is something else.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "a2s"))

from configs import paths as project_paths  # noqa: E402
from salm.a2s.content_lm_hier import ContentLMHier, build_hier_batch  # noqa: E402
from train_content_lm_dedup import load_split, log  # noqa: E402


@torch.no_grad()
def evaluate(model, amb, tok, dur, lens, batch_size, device, stride):
    """Code/duration loss split by COARSE vs FINE, plus the rolled-ambient scene ablation.

    Splitting the loss is the point: a model that gets its coarse pass right and its fine pass
    wrong is a different failure from one that never plans, and a single averaged number cannot
    tell them apart.
    """
    model.eval()
    n = tok.shape[0]
    tot = {"c_coarse": 0.0, "n_coarse": 0, "c_fine": 0.0, "n_fine": 0,
           "d": 0.0, "n_d": 0, "c_roll": 0.0}
    for i in range(0, n, batch_size):
        sel = slice(i, min(i + batch_size, n))
        if (sel.stop - sel.start) < 2:
            break                      # the scene ablation needs a pair to roll between
        a = torch.from_numpy(np.asarray(amb[sel], dtype=np.float32)).to(device)
        b = build_hier_batch(
            torch.from_numpy(tok[sel]).to(device), torch.from_numpy(dur[sel]).to(device),
            torch.from_numpy(lens[sel]).to(device), stride,
            model.bos_id, model.sep_id, model.eos_id,
        )
        mem = model.encode_ambient(a)
        cl, dl = model(a, b["codes_in"], b["durs_in"], memory=mem, pad_mask=b["pad_mask"])
        ct, dt, fine = b["code_tgt"], b["dur_tgt"], b["is_fine"]

        valid = ct != -100
        coarse_pos = valid & ~fine
        fine_pos = valid & fine
        flat = cl.reshape(-1, cl.shape[-1])
        for mask, kc, kn in ((coarse_pos, "c_coarse", "n_coarse"), (fine_pos, "c_fine", "n_fine")):
            m = mask.reshape(-1)
            if bool(m.any()):
                tot[kc] += float(F.cross_entropy(flat[m], ct.reshape(-1)[m], reduction="sum"))
                tot[kn] += int(m.sum())
        dm = (dt != -100).reshape(-1)
        if bool(dm.any()):
            tot["d"] += float(F.cross_entropy(
                dl.reshape(-1, dl.shape[-1])[dm], dt.reshape(-1)[dm], reduction="sum"))
            tot["n_d"] += int(dm.sum())

        cl_r, _ = model(a, b["codes_in"], b["durs_in"],
                        memory=torch.roll(mem, shifts=1, dims=0), pad_mask=b["pad_mask"])
        vm = valid.reshape(-1)
        tot["c_roll"] += float(F.cross_entropy(
            cl_r.reshape(-1, cl_r.shape[-1])[vm], ct.reshape(-1)[vm], reduction="sum"))

    nall = max(tot["n_coarse"] + tot["n_fine"], 1)
    out = {
        "val_coarse_loss": tot["c_coarse"] / max(tot["n_coarse"], 1),
        "val_fine_loss": tot["c_fine"] / max(tot["n_fine"], 1),
        "val_code_loss": (tot["c_coarse"] + tot["c_fine"]) / nall,
        "val_dur_loss": tot["d"] / max(tot["n_d"], 1),
        "val_code_loss_rolled": tot["c_roll"] / nall,
    }
    out["val_loss"] = out["val_code_loss"] + out["val_dur_loss"]
    out["val_scene_gap"] = out["val_code_loss_rolled"] - out["val_code_loss"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codebook", required=True)
    ap.add_argument("--run-name", default="hier_v1")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--max-run", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--lambda-dur", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--max-minutes", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    cb = torch.load(args.codebook, map_location="cpu", weights_only=False)
    K = int(cb["codes"].shape[0])

    tr = load_split(K, "train", args.max_run)
    va = load_split(K, "val", args.max_run)
    amb_tr, tok_tr, dur_tr, len_tr = tr
    amb_va, tok_va, dur_va, len_va = va
    ambient_dim = int(amb_tr.shape[2])

    # coarse + SEP + fine, worst case. Getting this wrong surfaces as an opaque broadcast
    # error deep in the embedding sum, so it is computed rather than guessed.
    Lmax = int(len_tr.max())
    max_tokens = -(-Lmax // args.stride) + 1 + Lmax + 4
    log(f"  stride {args.stride}: max L={Lmax} -> max_tokens {max_tokens}")

    model = ContentLMHier(
        num_codes=K, max_run=args.max_run, ambient_dim=ambient_dim, max_tokens=max_tokens,
        d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
        dropout=args.dropout, stride=args.stride,
    ).to(args.device)
    log(f"ContentLMHier: {model.num_parameters()/1e6:.2f}M params, K={K}, stride={args.stride}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_train = tok_tr.shape[0]
    steps = max(1, n_train // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=steps * args.epochs, pct_start=0.05,
        anneal_strategy="cos")

    run_dir = project_paths.OUTPUTS_ROOT / "content_lm" / args.run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    history, best_loss, t_start = [], 1e9, time.time()

    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(n_train)
        run_c = run_d = 0.0
        nb = 0
        for bi in range(steps):
            sel = np.sort(order[bi * args.batch_size : (bi + 1) * args.batch_size])
            if len(sel) < 2:
                continue
            a = torch.from_numpy(np.asarray(amb_tr[sel], dtype=np.float32)).to(args.device)
            b = build_hier_batch(
                torch.from_numpy(tok_tr[sel]).to(args.device),
                torch.from_numpy(dur_tr[sel]).to(args.device),
                torch.from_numpy(len_tr[sel]).to(args.device), args.stride,
                model.bos_id, model.sep_id, model.eos_id,
            )
            cl, dl = model(a, b["codes_in"], b["durs_in"], pad_mask=b["pad_mask"])
            lc = F.cross_entropy(cl.reshape(-1, cl.shape[-1]),
                                 b["code_tgt"].reshape(-1), ignore_index=-100)
            ld = F.cross_entropy(dl.reshape(-1, dl.shape[-1]),
                                 b["dur_tgt"].reshape(-1), ignore_index=-100)
            loss = lc + args.lambda_dur * ld
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run_c += float(lc.detach()); run_d += float(ld.detach()); nb += 1

        m = evaluate(model, amb_va, tok_va, dur_va, len_va,
                     args.batch_size, args.device, args.stride)
        m.update({"epoch": epoch, "train_code_loss": run_c / max(nb, 1),
                  "train_dur_loss": run_d / max(nb, 1), "lr": sched.get_last_lr()[0],
                  "cfg_stride": args.stride, "cfg_num_codes": K,
                  "epoch_minutes": (time.time() - t_start) / 60})
        history.append(m)
        log(f"  ep {epoch:03d} train(c {m['train_code_loss']:.3f} d {m['train_dur_loss']:.3f}) "
            f"val(coarse {m['val_coarse_loss']:.3f} fine {m['val_fine_loss']:.3f} "
            f"dur {m['val_dur_loss']:.3f}) gap {m['val_scene_gap']:+.4f}")
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))

        payload = {
            "model": model.state_dict(), "epoch": epoch, "metrics": m,
            "num_codes": K, "max_run": args.max_run, "ambient_dim": ambient_dim,
            "max_tokens": max_tokens, "d_model": args.d_model, "layers": args.layers,
            "heads": args.heads, "codes": cb["codes"], "dedup": True, "hier": True,
            "stride": args.stride,
        }
        torch.save(payload, run_dir / "checkpoints" / "checkpoint_last.pt")
        if m["val_loss"] < best_loss:
            best_loss = m["val_loss"]
            torch.save(payload, run_dir / "checkpoints" / "checkpoint_best_loss.pt")
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save(payload, run_dir / "checkpoints" / f"checkpoint_ep{epoch}.pt")
        if args.max_minutes > 0 and (time.time() - t_start) / 60 > args.max_minutes:
            log(f"  stopping: {args.max_minutes} min budget reached at epoch {epoch}")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
