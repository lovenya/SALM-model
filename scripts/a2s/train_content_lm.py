#!/usr/bin/env python
"""
Train ContentLM: autoregressive discrete content codes from an ambient scene.

Step 2 of the discrete-content plan. Only run this if the gate (job 436244,
run_tamia_a2s_content_codebook_gate.sbatch) showed that quantized ORACLE content still
renders intelligibly through the proven renderer -- otherwise the target space itself is too
lossy and a perfect predictor over it would still be useless.

===================== WHAT THIS RUN IS JUDGED ON, decided up front =====================
NOT train loss, and NOT raw validation perplexity. Content frames are ~53% silence and
English phoneme statistics are highly predictable, so a model that entirely IGNORES the
ambient scene can post a healthy cross-entropy by learning the marginal distribution of
speech. That is the same class of failure `val_content_bind_gap` was built to catch on the
concat mechanism, and it would look like success on every curve.

The metric is `val_scene_gap` = loss(rolled ambient) - loss(true ambient), where "rolled"
pairs each code sequence with a DIFFERENT row's real ambient encoding. Rolling rather than
zeroing on purpose: a zeroed memory is out of distribution, so a model could score worse on
it for reasons unrelated to using the scene, which would read as a false positive.

  scene_gap ~ 0            -> the scene is being ignored. Total failure, regardless of ppl.
  scene_gap clearly > 0     -> the ambient input is carrying real predictive information
                               about which codes come next. THE result this stage exists for.

Two supporting guards, both cheap and both catching a specific way this can lie:
  - `val_unigram_loss`: cross-entropy of the fixed marginal code distribution. If the model
    does not beat this, it has learned nothing at all, scene or otherwise.
  - `val_argmax_top1_share`: fraction of positions where the model's argmax is its single
    most-favoured code. Near 1.0 means it collapsed onto silence and is posting a good loss
    by predicting nothing.
========================================================================================

Usage:
  python scripts/a2s/train_content_lm.py --codebook <...>/codebook_k1024.pt --epochs 200
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from configs import paths as project_paths                          # noqa: E402
from salm.a2s.content_codebook import ContentCodebook               # noqa: E402
from salm.a2s.content_lm import ContentLM, shuffled_memory          # noqa: E402
from salm.a2s.ctc_probe import SimpleCTCASR                         # noqa: E402
from salm.data.frontend import (                                    # noqa: E402
    AUDIOLDM_CLIP_SECONDS,
    MelConfig,
    MelExtractor,
    load_waveform_mono_16k,
)


def log(msg: str) -> None:
    print(msg, flush=True)


def code_cache_path(cache_root: Path, codebook: Path, split: str) -> Path:
    """Codes are only meaningful relative to ONE codebook, so the name carries its stem."""
    return cache_root / f"codes_{codebook.stem}_{split}.npy"


def build_code_cache(
    split: str, codebook: Path, book: ContentCodebook, ctc, mel, device: str,
    batch_size: int, out_path: Path,
) -> np.ndarray:
    """One CTC-probe pass over a split's speech, quantized to code indices, cached to disk."""
    index_path = project_paths.FEATURES_ROOT / "a2s" / split / "index.jsonl"
    rows = [json.loads(l) for l in index_path.read_text().splitlines() if l.strip()]
    log(f"    encoding {len(rows)} {split} utterances -> codes ...")
    all_codes = np.zeros((len(rows), book_frames(book)), dtype=np.int16)
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            wavs = [
                load_waveform_mono_16k(
                    r["speech_file"], target_seconds=AUDIOLDM_CLIP_SECONDS, align="left"
                )
                for r in chunk
            ]
            wav_t = torch.from_numpy(np.stack(wavs)).float().to(device)
            content = ctc.encode(mel.batch(wav_t))
            idx = book.indices(content.float())
            all_codes[start : start + len(chunk)] = idx.cpu().numpy().astype(np.int16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, all_codes)
    log(f"    cached {out_path} {all_codes.shape}")
    return all_codes


def book_frames(book: ContentCodebook) -> int:
    from salm.a2s.ctc_probe import CTC_FRAMES_OUT
    return CTC_FRAMES_OUT


@torch.no_grad()
def evaluate(model, amb, codes, bos, batch_size, device, unigram_logp) -> dict:
    """Conditioned loss, ablated loss, and the two guards -- one pass each, no sampling."""
    model.eval()
    n = amb.shape[0]
    tot = tot_roll = tot_uni = 0.0
    correct = seen = 0
    argmax_counts = torch.zeros(model.num_codes, device=device)

    for s in range(0, n, batch_size):
        a = torch.from_numpy(amb[s : s + batch_size]).float().to(device)
        c = torch.from_numpy(codes[s : s + batch_size].astype(np.int64)).to(device)
        if a.shape[0] < 2:
            break                       # the ablation needs a pair to roll between
        inp = torch.cat([torch.full((a.shape[0], 1), bos, dtype=torch.long, device=device),
                         c[:, :-1]], dim=1)

        mem = model.encode_ambient(a)
        logits = model(a, inp, memory=mem)
        flat, tgt = logits.reshape(-1, model.num_codes), c.reshape(-1)
        tot += float(F.cross_entropy(flat, tgt, reduction="sum"))

        # Identical in every respect except which row's scene each sequence is paired with.
        roll = model(a, inp, memory=shuffled_memory(mem))
        tot_roll += float(
            F.cross_entropy(roll.reshape(-1, model.num_codes), tgt, reduction="sum")
        )

        tot_uni += float(-unigram_logp[tgt].sum())
        pred = flat.argmax(dim=-1)
        correct += int((pred == tgt).sum())
        seen += tgt.numel()
        argmax_counts += torch.bincount(pred, minlength=model.num_codes).float()

    share = float((argmax_counts / argmax_counts.sum().clamp(min=1)).max())
    out = {
        "val_loss": tot / max(seen, 1),
        "val_loss_rolled": tot_roll / max(seen, 1),
        "val_unigram_loss": tot_uni / max(seen, 1),
        "val_token_acc": correct / max(seen, 1),
        "val_argmax_top1_share": share,
    }
    out["val_scene_gap"] = out["val_loss_rolled"] - out["val_loss"]
    out["val_ppl"] = float(np.exp(min(out["val_loss"], 20.0)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codebook", required=True)
    ap.add_argument("--run-name", default="content_lm_v1")
    ap.add_argument("--ctc-checkpoint", default=None)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--encode-batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-every", type=int, default=20,
                    help="also write checkpoint_ep<N>.pt every N epochs. The standing repo\n                         rule is that checkpoint-comparison generation covers the full saved\n                         spread, not one pick -- and here that is not optional: scene_gap and\n                         overfitting climb TOGETHER (K=1024: gap +0.026 at ep10 with val_loss\n                         1.32, gap +0.085 at ep144 with val_loss 3.25), so no single epoch is\n                         obviously the right one to listen to.")
    ap.add_argument("--max-minutes", type=float, default=0.0,
                    help="stop cleanly after this much wall clock (0 = no limit). Epoch\n                         time for this model was not measurable before the first run, so\n                         this exists to truncate gracefully with checkpoints written\n                         rather than be SIGKILLed mid-save at the Slurm limit.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    codebook = Path(args.codebook)
    ctc_ckpt = Path(
        args.ctc_checkpoint
        or project_paths.OUTPUTS_ROOT / "ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt"
    )
    run_dir = project_paths.OUTPUTS_ROOT / "content_lm" / args.run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    cache_root = project_paths.OUTPUTS_ROOT / "content_codes"

    for f in (codebook, ctc_ckpt):
        if not f.exists():
            log(f"ERROR: missing {f}")
            return 1

    # ---------------------------------------------------------------- codebook
    cb = torch.load(codebook, map_location=args.device, weights_only=False)
    book = ContentCodebook(int(cb["num_codes"]), int(cb["content_dim"])).to(args.device)
    book.codes.copy_(cb["codes"].to(args.device))
    st = cb.get("stats", {})
    log(f"codebook {codebook.name}: K={book.num_codes} dim={book.content_dim} "
        f"(held-out cosine={st.get('cosine', float('nan')):.4f})")

    # ---------------------------------------------------------------- codes (cached)
    ctc = mel = None
    codes = {}
    for split in ("train", "val"):
        path = code_cache_path(cache_root, codebook, split)
        if path.exists():
            codes[split] = np.load(path)
            log(f"    reusing {path} {codes[split].shape}")
        else:
            if ctc is None:
                log(f"Loading frozen CTC probe from {ctc_ckpt} ...")
                ctc = SimpleCTCASR().to(args.device).eval()
                p = torch.load(ctc_ckpt, map_location=args.device, weights_only=False)
                ctc.load_state_dict(p["model"])
                for prm in ctc.parameters():
                    prm.requires_grad_(False)
                ctc.rnn.train()
                mel = MelExtractor(MelConfig(norm_mode="none"), device=args.device)
            codes[split] = build_code_cache(
                split, codebook, book, ctc, mel, args.device,
                args.encode_batch_size, path,
            )

    # ---------------------------------------------------------------- ambient
    amb = {}
    for split in ("train", "val"):
        p = project_paths.OUTPUTS_ROOT / "features_a2s_cond" / split / "cond_seq.npy"
        amb[split] = np.load(p, mmap_mode="r")
        if amb[split].shape[0] != codes[split].shape[0]:
            log(f"ERROR: {split} row mismatch -- ambient {amb[split].shape[0]} vs "
                f"codes {codes[split].shape[0]}. Row alignment is not optional here.")
            return 1
        log(f"{split}: ambient {amb[split].shape} codes {codes[split].shape}")

    ambient_dim = int(amb["train"].shape[2])
    n_frames = int(codes["train"].shape[1])

    # The unigram baseline is a FIXED reference computed from the training marginal, so a
    # model that fails to beat it has learned nothing -- not merely "less than we hoped".
    counts = np.bincount(codes["train"].reshape(-1), minlength=book.num_codes).astype(np.float64)
    unigram = torch.from_numpy(counts / counts.sum()).float().to(args.device)
    unigram_logp = unigram.clamp(min=1e-12).log()
    log(f"unigram baseline loss = {float(-(unigram * unigram_logp).sum()):.4f} "
        f"(most common code = {float(unigram.max()):.4f} of frames)")

    # ---------------------------------------------------------------- model
    model = ContentLM(
        num_codes=book.num_codes, ambient_dim=ambient_dim, content_frames=n_frames,
        d_model=args.d_model, n_layers=args.layers, n_heads=args.heads, dropout=args.dropout,
    ).to(args.device)
    log(f"ContentLM: {model.num_parameters()/1e6:.2f}M params, K={book.num_codes}, "
        f"{n_frames} positions")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_train = codes["train"].shape[0]
    steps_per_epoch = max(1, n_train // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps,
        pct_start=min(0.3, args.warmup_steps / max(total_steps, 1)), anneal_strategy="cos",
    )
    log(f"{steps_per_epoch} steps/epoch, {total_steps} total")

    history = []
    best_gap = -1e9
    best_loss = 1e9
    rng = np.random.default_rng(args.seed)
    step = 0
    t_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        order = rng.permutation(n_train)
        run_loss = 0.0
        nb = 0
        for bi in range(steps_per_epoch):
            sel = np.sort(order[bi * args.batch_size : (bi + 1) * args.batch_size])
            if len(sel) < 2:
                continue
            a = torch.from_numpy(np.asarray(amb["train"][sel], dtype=np.float32)).to(args.device)
            c = torch.from_numpy(codes["train"][sel].astype(np.int64)).to(args.device)
            inp = torch.cat(
                [torch.full((len(sel), 1), model.bos_id, dtype=torch.long, device=args.device),
                 c[:, :-1]], dim=1
            )
            logits = model(a, inp)
            loss = F.cross_entropy(logits.reshape(-1, model.num_codes), c.reshape(-1))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step < total_steps - 1:
                sched.step()
            step += 1
            run_loss += float(loss)
            nb += 1

        m = evaluate(model, amb["val"], codes["val"], model.bos_id,
                     args.batch_size, args.device, unigram_logp)
        m["epoch"] = epoch
        m["train_loss"] = run_loss / max(nb, 1)
        m["epoch_minutes"] = (time.time() - t0) / 60.0
        m["lr"] = float(opt.param_groups[0]["lr"])
        history.append(m)

        log(f"  epoch {epoch:03d} train={m['train_loss']:.4f} val={m['val_loss']:.4f} "
            f"ppl={m['val_ppl']:.1f} SCENE_GAP={m['val_scene_gap']:+.4f} "
            f"acc={m['val_token_acc']:.4f} uni={m['val_unigram_loss']:.4f} "
            f"top1={m['val_argmax_top1_share']:.3f} ({m['epoch_minutes']:.2f} min)")

        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        payload = {
            "model": model.state_dict(), "epoch": epoch, "metrics": m,
            "codebook": str(codebook), "num_codes": book.num_codes,
            "ambient_dim": ambient_dim, "content_frames": n_frames,
            "d_model": args.d_model, "layers": args.layers, "heads": args.heads,
            # The codebook rides along so a generation run needs only this one file.
            "codes": book.codes.cpu(),
        }
        torch.save(payload, run_dir / "checkpoints" / "checkpoint_last.pt")
        # Selected on scene_gap, NOT val_loss: a checkpoint with a better loss and no
        # scene_gap is precisely the failure this run is designed to detect, and choosing it
        # would hand the next stage a model that ignores its own input.
        if m["val_scene_gap"] > best_gap:
            best_gap = m["val_scene_gap"]
            torch.save(payload, run_dir / "checkpoints" / "checkpoint_best.pt")
        # ALSO keep the best-val_loss checkpoint. Selecting on scene_gap alone picked the most
        # OVERFIT model in the K=1024 run (ep144: gap +0.085 but val_loss 3.25 against its own
        # 1.32 minimum at ep10), because the two rise together -- an overfit conditional model
        # can post a real scene_gap by soft-retrieving the nearest memorised training row. Both
        # extremes are kept so the comparison is decidable by ear instead of by one metric.
        if m["val_loss"] < best_loss:
            best_loss = m["val_loss"]
            torch.save(payload, run_dir / "checkpoints" / "checkpoint_best_loss.pt")
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save(payload, run_dir / "checkpoints" / f"checkpoint_ep{epoch}.pt")

        if args.max_minutes > 0 and (time.time() - t_start) / 60.0 >= args.max_minutes:
            log(f"  stopping at epoch {epoch}: hit --max-minutes={args.max_minutes:g} "
                f"({(time.time()-t_start)/60.0:.1f} min elapsed). Checkpoints are written; "
                "the LR schedule did NOT complete, so this is a truncated run.")
            break

    log("\n" + "=" * 74)
    log("CONTENT LM SUMMARY")
    log("=" * 74)
    gaps = [(h["epoch"], h["val_scene_gap"]) for h in history]
    best = max(gaps, key=lambda x: x[1])
    log(f"val_scene_gap : first={gaps[0][1]:+.4f} last={gaps[-1][1]:+.4f} "
        f"best={best[1]:+.4f} @ep{best[0]}")
    log("    ~0 the whole run = the scene is IGNORED; generic speech statistics only")
    log("    clearly > 0      = ambient audio genuinely predicts which codes come next")
    log(f"val_loss      : first={history[0]['val_loss']:.4f} last={history[-1]['val_loss']:.4f}"
        f"   (unigram baseline {history[-1]['val_unigram_loss']:.4f} -- must beat this)")
    log(f"top1 share    : last={history[-1]['val_argmax_top1_share']:.3f} "
        "(near 1.0 = collapsed onto one code, probably silence)")
    lo = min(history, key=lambda r: r["val_loss"])
    log(f"best val_loss : {lo['val_loss']:.4f} @ep{lo['epoch']} "
        f"(scene_gap there = {lo['val_scene_gap']:+.4f})")
    log("    checkpoint_best.pt      = max scene_gap  (most overfit)")
    log("    checkpoint_best_loss.pt = min val_loss   (least overfit)")
    log("    checkpoint_ep*.pt       = the spread between them -- listen across it")
    log("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
