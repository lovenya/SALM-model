#!/usr/bin/env python
"""Train the oracle transcript -> K=512 content-code bridge.

The frozen Flan-T5 states and target-speech codes are precomputed.  Ambient features are
neither loaded nor accepted.  The decisive metric is paired text binding:
``val_text_gap = CE(codes | wrong transcript) - CE(codes | supplied transcript)``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from configs import paths as project_paths  # noqa: E402
from salm.a2s.text_content_lm import TextContentLM, roll_text_pair  # noqa: E402


def log(msg: str, rank: int = 0) -> None:
    if rank == 0:
        print(msg, flush=True)


def teacher_input(codes: torch.Tensor, bos: int) -> torch.Tensor:
    return torch.cat(
        [torch.full((codes.shape[0], 1), bos, device=codes.device, dtype=torch.long),
         codes[:, :-1]], dim=1,
    )


@torch.inference_mode()
def evaluate(model, hidden, mask, codes, batch_size, device, unigram_logp) -> dict:
    model.eval()
    sums = {"paired": 0.0, "rolled": 0.0, "unigram": 0.0}
    seen = correct = changed = examples = pair_wins = 0
    for start in range(0, len(codes), batch_size):
        sl = slice(start, min(start + batch_size, len(codes)))
        if sl.stop - sl.start < 2:
            break
        h = torch.from_numpy(np.asarray(hidden[sl], dtype=np.float32)).to(device)
        m = torch.from_numpy(np.asarray(mask[sl], dtype=np.bool_)).to(device)
        c = torch.from_numpy(codes[sl].astype(np.int64)).to(device)
        inp = teacher_input(c, model.bos_id)
        mem = model.encode_text(h)
        pad = ~m
        logits = model(h, m, inp, memory=mem, memory_pad_mask=pad)
        rmem, rpad = roll_text_pair(mem, pad)
        rolled = model(h, m, inp, memory=rmem, memory_pad_mask=rpad)
        flat = logits.reshape(-1, model.num_codes)
        rflat = rolled.reshape(-1, model.num_codes)
        tgt = c.reshape(-1)
        sums["paired"] += float(F.cross_entropy(flat, tgt, reduction="sum"))
        sums["rolled"] += float(F.cross_entropy(rflat, tgt, reduction="sum"))
        sums["unigram"] += float(-unigram_logp[tgt].sum())
        pred, rpred = flat.argmax(-1), rflat.argmax(-1)
        correct += int((pred == tgt).sum())
        changed += int((pred != rpred).sum())
        seen += tgt.numel()
        per = F.cross_entropy(flat, tgt, reduction="none").view(c.shape).mean(1)
        rper = F.cross_entropy(rflat, tgt, reduction="none").view(c.shape).mean(1)
        pair_wins += int((per < rper).sum())
        examples += c.shape[0]
    out = {
        "val_loss": sums["paired"] / max(seen, 1),
        "val_loss_rolled_text": sums["rolled"] / max(seen, 1),
        "val_unigram_loss": sums["unigram"] / max(seen, 1),
        "val_token_acc": correct / max(seen, 1),
        "val_logits_argmax_change": changed / max(seen, 1),
        "val_pair_win_rate": pair_wins / max(examples, 1),
    }
    out["val_text_gap"] = out["val_loss_rolled_text"] - out["val_loss"]
    out["val_ppl"] = float(np.exp(min(out["val_loss"], 20.0)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codebook", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64, help="per DDP rank")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--max-minutes", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    run_dir = project_paths.OUTPUTS_ROOT / "text_content_lm" / args.run_name
    if rank == 0:
        # Reject before model construction/loading: a gate must never overwrite evidence.
        if run_dir.exists():
            raise SystemExit(f"run directory already exists: {run_dir}")
        (run_dir / "checkpoints").mkdir(parents=True)
    if world > 1:
        dist.barrier()

    cb = torch.load(args.codebook, map_location="cpu", weights_only=False)
    K = int(cb["num_codes"])
    if K != 512:
        raise SystemExit(f"this pre-registered gate is K=512, got K={K}")
    arrays = {}
    metas = {}
    for split in ("train", "val"):
        hp = args.features / f"{split}_hidden.npy"
        mp = args.features / f"{split}_mask.npy"
        jp = args.features / f"{split}_meta.json"
        cp = project_paths.OUTPUTS_ROOT / "content_codes" / f"codes_codebook_k512_{split}.npy"
        for p in (hp, mp, jp, cp):
            if not p.exists():
                raise SystemExit(f"missing required cache: {p}")
        h = np.load(hp, mmap_mode="r")
        m = np.load(mp, mmap_mode="r")
        c = np.load(cp, mmap_mode="r")
        meta = json.loads(jp.read_text())
        n_index = sum(
            1 for line in (project_paths.FEATURES_ROOT / "a2s" / split / "index.jsonl")
            .read_text().splitlines() if line.strip()
        )
        if not (len(h) == len(m) == len(c) == int(meta["num_rows"]) == n_index):
            raise SystemExit(
                f"{split} row mismatch: hidden={len(h)} mask={len(m)} codes={len(c)} "
                f"meta={meta.get('num_rows')} index={n_index}"
            )
        arrays[split] = (h, m, c)
        metas[split] = meta

    htr, mtr, ctr = arrays["train"]
    hva, mva, cva = arrays["val"]
    if ctr.shape[1] != cva.shape[1]:
        raise SystemExit("train/val content-frame widths differ")
    counts = np.bincount(np.asarray(ctr).reshape(-1), minlength=K).astype(np.float64)
    unigram = torch.from_numpy(counts / counts.sum()).float().to(device)
    unigram_logp = unigram.clamp(min=1e-12).log()

    model = TextContentLM(
        K, text_dim=int(htr.shape[2]), content_frames=int(ctr.shape[1]),
        max_text_tokens=int(htr.shape[1]), d_model=args.d_model,
        n_layers=args.layers, n_heads=args.heads, dropout=args.dropout,
    ).to(device)
    raw = model
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        raw = model.module
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    local_n = len(ctr) // world
    steps_per_epoch = max(1, local_n // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.05, anneal_strategy="cos"
    )
    log(
        f"TextContentLM {raw.num_parameters()/1e6:.2f}M params; K={K}; "
        f"DDP={world}; global batch={args.batch_size*world}; {steps_per_epoch} steps/epoch",
        rank,
    )

    rng = np.random.default_rng(args.seed)
    history = []
    best_loss, best_gap = float("inf"), -float("inf")
    global_step = 0
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(ctr))
        usable = (len(order) // (world * args.batch_size)) * world * args.batch_size
        mine = order[:usable].reshape(-1, world, args.batch_size)[:, rank, :]
        run_loss = 0.0
        for sel in mine:
            sel = np.sort(sel)
            h = torch.from_numpy(np.asarray(htr[sel], dtype=np.float32)).to(device)
            m = torch.from_numpy(np.asarray(mtr[sel], dtype=np.bool_)).to(device)
            c = torch.from_numpy(np.asarray(ctr[sel], dtype=np.int64)).to(device)
            logits = model(h, m, teacher_input(c, raw.bos_id))
            loss = F.cross_entropy(
                logits.reshape(-1, K), c.reshape(-1),
                label_smoothing=args.label_smoothing,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if global_step < total_steps - 1:
                sched.step()
            global_step += 1
            run_loss += float(loss.detach())

        if world > 1:
            dist.barrier()
        do_eval = epoch == 0 or (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1
        if rank == 0 and do_eval:
            metrics = evaluate(raw, hva, mva, cva, args.batch_size, device, unigram_logp)
            metrics.update({
                "epoch": epoch,
                "train_loss_rank0": run_loss / max(len(mine), 1),
                "lr": float(opt.param_groups[0]["lr"]),
                "world_size": world,
            })
            history.append(metrics)
            (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            print(
                f"ep {epoch:03d} train={metrics['train_loss_rank0']:.3f} "
                f"val={metrics['val_loss']:.3f} TEXT_GAP={metrics['val_text_gap']:+.3f} "
                f"pair_win={metrics['val_pair_win_rate']:.3f} "
                f"argmax_change={metrics['val_logits_argmax_change']:.3f}", flush=True,
            )
            payload = {
                "model": {k: v.detach().cpu() for k, v in raw.state_dict().items()},
                "epoch": epoch, "metrics": metrics,
                "num_codes": K, "codes": cb["codes"].cpu(),
                "content_frames": raw.content_frames, "text_dim": raw.text_dim,
                "max_text_tokens": raw.max_text_tokens, "d_model": raw.d_model,
                "layers": args.layers, "heads": args.heads,
                "text_model": metas["train"]["model"],
                "feature_meta": metas,
                "bridge_kind": "oracle_text_content_lm",
            }
            torch.save(payload, run_dir / "checkpoints" / "checkpoint_last.pt")
            if metrics["val_loss"] < best_loss:
                best_loss = metrics["val_loss"]
                torch.save(payload, run_dir / "checkpoints" / "checkpoint_best_loss.pt")
            if metrics["val_text_gap"] > best_gap:
                best_gap = metrics["val_text_gap"]
                torch.save(payload, run_dir / "checkpoints" / "checkpoint_best_binding.pt")
            if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                torch.save(payload, run_dir / "checkpoints" / f"checkpoint_ep{epoch}.pt")
        if world > 1:
            dist.barrier()

        stop = torch.tensor(
            [int(args.max_minutes > 0 and (time.time() - started) / 60 >= args.max_minutes)],
            device=device,
        )
        if world > 1:
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if int(stop.item()):
            log(f"stopping cleanly after epoch {epoch}: max-minutes reached", rank)
            break

    if rank == 0:
        best = min(history, key=lambda x: x["val_loss"])
        print(
            f"SUMMARY best val={best['val_loss']:.3f} @ep{best['epoch']} "
            f"text_gap={best['val_text_gap']:+.3f} pair_win={best['val_pair_win_rate']:.3f}",
            flush=True,
        )
    if world > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
