#!/usr/bin/env python
"""
Train ContentLMDedup: autoregressive (code, duration) pairs over run-length-encoded content.

Step 3 of the granularity fix. The frame-level ContentLM generates per-scene, syntactically
real speech from ambient audio alone but cannot hold a plan past ~3-5 words, and both
alternative explanations are eliminated by measurement (438059 flat epoch ladder, 438062
scene_gap invariant at +0.021 across seven configs). This trains over ~2x shorter sequences.

===================== WHAT THIS RUN IS JUDGED ON, decided up front =====================
`val_scene_gap` = loss(rolled ambient) - loss(true ambient), computed on the CODE head only
so it stays comparable to the frame-level runs' +0.021 floor. Rolled rather than zeroed: a
zeroed memory is out of distribution, so a model could score worse on it for reasons
unrelated to using the scene, which reads as a false positive.

  scene_gap >= ~+0.021   -> the shorter horizon did not cost the scene signal. Necessary,
      not sufficient: the actual question is coherence, which only listening answers.
  scene_gap ~ 0          -> the representation change broke the conditioning. Would be a
      surprise (the codec is lossless) and would point at a bug, not a finding.

Supporting numbers, each catching a specific way this can lie:
  `val_code_loss` vs `val_code_unigram` -- must beat the marginal, or nothing was learned.
  `val_dur_loss`  vs `val_dur_unigram`  -- duration is low-entropy (median 1, mean ~1.8), so
      beating its marginal is a weak bar; failing it means the duration head is dead and the
      expanded frames will be garbage regardless of how good the codes are.
  `val_dur_acc`   -- exact-match duration accuracy. Directly predicts whether re-expansion
      lands near 256 frames or has to be truncated/padded at generation time.
  `val_pred_tokens` -- mean sampled sequence length vs the true mean. If the model emits far
      too few tokens with long durations it is producing smeared, static content that will
      sound worse than the frame-level model no matter what the losses say.
========================================================================================

Usage:
  python scripts/a2s/train_content_lm_dedup.py --codebook <...>/codebook_k1024.pt --epochs 150
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

from configs import paths as project_paths                              # noqa: E402
from salm.a2s.content_dedup import dedup_batch, stats as dedup_stats    # noqa: E402
from salm.a2s.content_lm_dedup import (                                 # noqa: E402
    ContentLMDedup,
    build_teacher_forcing,
    plan_targets,
)


def log(msg: str) -> None:
    print(msg, flush=True)


def load_split(k: int, split: str, max_run: int) -> tuple[np.ndarray, ...]:
    """Cached frame codes + ambient features -> deduped tokens/durations/lengths.

    The dedup itself is a Python loop over 4.5M frames and measures ~58 s per split, which
    would be paid again on every job start and by every concurrent config. Cached to disk,
    keyed by (K, split, max_run) so a different cap cannot silently reuse the wrong arrays.
    """
    root = project_paths.OUTPUTS_ROOT
    codes = np.load(root / "content_codes" / f"codes_codebook_k{k}_{split}.npy").astype(np.int64)
    amb = np.load(root / "features_a2s_cond" / split / "cond_seq.npy", mmap_mode="r")
    if amb.shape[0] != codes.shape[0]:
        raise SystemExit(
            f"{split}: ambient {amb.shape[0]} rows vs codes {codes.shape[0]} -- row "
            "alignment is not optional here"
        )
    s = dedup_stats(codes)
    cache = root / "content_codes_dedup" / f"dedup_k{k}_{split}_r{max_run}.npz"
    if cache.exists():
        z = np.load(cache)
        tok, dur, lens = z["tokens"], z["durations"], z["lengths"]
        log(f"  {split}: reusing {cache.name}")
    else:
        tok, dur, lens = dedup_batch(codes, max_tokens=codes.shape[1], max_run=max_run)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, tokens=tok, durations=dur, lengths=lens)
        log(f"  {split}: wrote {cache.name}")
    log(f"  {split}: {codes.shape} frames -> tokens {tok.shape}, lengths "
        f"{lens.min()}-{lens.max()} (mean {lens.mean():.1f}, "
        f"{codes.shape[1]/lens.mean():.2f}x compression; uncapped {s['compression']:.2f}x)")
    return amb, tok, dur, lens


@torch.no_grad()
def evaluate(model, amb, tok, dur, lens, batch_size, device,
             code_unigram_logp, dur_unigram_logp, do_sample: bool = True) -> dict:
    model.eval()
    n = tok.shape[0]
    c_sum = c_roll = d_sum = 0.0
    c_uni = d_uni = 0.0
    c_seen = d_seen = 0
    d_correct = 0

    for s in range(0, n, batch_size):
        sl = slice(s, s + batch_size)
        a = torch.from_numpy(np.asarray(amb[sl], dtype=np.float32)).to(device)
        if a.shape[0] < 2:
            break                       # the scene ablation needs a pair to roll between
        t = torch.from_numpy(tok[sl]).to(device)
        d = torch.from_numpy(dur[sl]).to(device)
        L = torch.from_numpy(lens[sl]).to(device)
        tf = build_teacher_forcing(t, d, L, model.bos_id, model.num_codes)

        mem = model.encode_ambient(a)
        cl, dl = model(a, tf["codes_in"], tf["durs_in"], memory=mem,
                       pad_mask=tf["pad_mask"])
        ct = tf["code_tgt"].reshape(-1)
        dt = tf["dur_tgt"].reshape(-1)
        cflat = cl.reshape(-1, model.num_codes + 1)
        dflat = dl.reshape(-1, model.max_run)

        c_sum += float(F.cross_entropy(cflat, ct, ignore_index=-100, reduction="sum"))
        d_sum += float(F.cross_entropy(dflat, dt, ignore_index=-100, reduction="sum"))

        # Identical but for WHICH row's scene each sequence is paired with. Code head only,
        # so the number stays comparable to the frame-level runs' scene_gap.
        cl_r, _ = model(a, tf["codes_in"], tf["durs_in"],
                        memory=torch.roll(mem, shifts=1, dims=0),
                        pad_mask=tf["pad_mask"])
        c_roll += float(F.cross_entropy(
            cl_r.reshape(-1, model.num_codes + 1), ct, ignore_index=-100, reduction="sum"))

        cm, dm = ct != -100, dt != -100
        c_seen += int(cm.sum()); d_seen += int(dm.sum())
        c_uni += float(-code_unigram_logp[ct[cm].clamp(max=model.num_codes - 1)].sum())
        d_uni += float(-dur_unigram_logp[dt[dm]].sum())
        d_correct += int((dflat[dm].argmax(-1) == dt[dm]).sum())

    out = {
        "val_code_loss": c_sum / max(c_seen, 1),
        "val_code_loss_rolled": c_roll / max(c_seen, 1),
        "val_dur_loss": d_sum / max(d_seen, 1),
        "val_code_unigram": c_uni / max(c_seen, 1),
        "val_dur_unigram": d_uni / max(d_seen, 1),
        "val_dur_acc": d_correct / max(d_seen, 1),
    }
    out["val_scene_gap"] = out["val_code_loss_rolled"] - out["val_code_loss"]
    out["val_loss"] = out["val_code_loss"] + out["val_dur_loss"]

    # Sampled length: the cheapest guard against "few tokens, long durations", which would
    # produce smeared static content while every loss still looked healthy. Gated because it
    # costs up to `total_frames` SEQUENTIAL forward passes -- every epoch that would be a
    # meaningful slice of the whole training budget, for a diagnostic that moves slowly.
    if do_sample:
        a = torch.from_numpy(np.asarray(amb[:32], dtype=np.float32)).to(device)
        sc, _ = model.sample(a, total_frames=int(tok.shape[1]), temperature=1.0)
        out["val_pred_tokens"] = float(np.mean([len(x) for x in sc]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codebook", required=True)
    ap.add_argument("--run-name", default="content_lm_dedup_v1")
    ap.add_argument("--max-run", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--n-think", type=int, default=0,
                    help="Learnable latent 'think' tokens prepended to the decoder input. The "
                         "model computes over them and the ambient before emitting any code; "
                         "their outputs are discarded. 0 = the old model exactly.")
    ap.add_argument("--lambda-contrast", type=float, default=0.0,
                    help="Weight on a contrastive scene term: softplus(CE_true - CE_rolled), "
                         "per sample. Turns val_scene_gap from a diagnostic into a training "
                         "objective -- reward the true scene pairing over a mismatched one. "
                         "0 disables it and the step is bit-identical to before. Costs one "
                         "extra forward pass per step.")
    ap.add_argument("--use-plan", action="store_true",
                    help="Predict a bag-of-codes plan from the ambient and add it to every "
                         "token embedding. Targets the measured ~20-token plan horizon.")
    ap.add_argument("--lambda-plan", type=float, default=1.0,
                    help="Weight on the plan's auxiliary BCE loss. Only used with --use-plan.")
    ap.add_argument("--lambda-dur", type=float, default=1.0,
                    help="weight on the duration loss. 1.0 by default: both heads matter, and "
                         "a dead duration head makes the expanded frames garbage no matter "
                         "how good the codes are.")
    ap.add_argument("--sample-every", type=int, default=10,
                    help="compute val_pred_tokens every N epochs. Sampling costs up to\n                         256 sequential forward passes, so doing it every epoch would\n                         spend a real fraction of the training budget on a diagnostic\n                         that changes slowly.")
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--max-minutes", type=float, default=0.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--truncate-speech", action="store_true",
                    help="Cut each target at its clip's speech boundary instead of filling all "
                         "256 frames. Speech occupies a median 104 of 256, so this removes the "
                         "~57%% of the planning budget spent on a wordless tail. No renderer "
                         "change: expand_sequence already holds the last code past EOS.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cb_path = Path(args.codebook)
    if not cb_path.exists():
        log(f"ERROR: missing {cb_path}")
        return 1
    cb = torch.load(cb_path, map_location="cpu", weights_only=False)
    K = int(cb["num_codes"])
    run_dir = project_paths.OUTPUTS_ROOT / "content_lm" / args.run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    log(f"codebook {cb_path.name}: K={K} dim={cb['content_dim']} "
        f"(held-out cosine={cb.get('stats', {}).get('cosine', float('nan')):.4f})")

    log("loading + deduplicating:")
    tr = load_split(K, "train", args.max_run)
    va = load_split(K, "val", args.max_run)
    amb_tr, tok_tr, dur_tr, len_tr = tr
    amb_va, tok_va, dur_va, len_va = va
    if args.truncate_speech:
        from salm.a2s.speech_extent import speech_frames, truncate_to_speech
        for split, (tk, dr, ln) in (("train", (tok_tr, dur_tr, len_tr)),
                                    ("val", (tok_va, dur_va, len_va))):
            ef = speech_frames(split, tk.shape[0])
            t2, d2, l2 = truncate_to_speech(tk, dr, ln, ef)
            if split == "train":
                tok_tr, dur_tr, len_tr = t2, d2, l2
            else:
                tok_va, dur_va, len_va = t2, d2, l2
            log(f"  {split}: truncated at the speech boundary -- mean length "
                f"{ln.mean():.1f} -> {l2.mean():.1f} tokens "
                f"(speech ends at frame {ef.mean():.0f}/256 on average)")
    ambient_dim = int(amb_tr.shape[2])
    max_tokens = int(tok_tr.shape[1])

    # Fixed marginals from the training split -- a model that fails to beat these learned
    # nothing at all, scene or otherwise.
    cc = np.bincount(
        np.concatenate([tok_tr[i, :L] for i, L in enumerate(len_tr)]), minlength=K
    ).astype(np.float64)
    dc = np.bincount(
        np.concatenate([dur_tr[i, :L] for i, L in enumerate(len_tr)]) - 1,
        minlength=args.max_run,
    ).astype(np.float64)
    code_logp = torch.from_numpy(np.log(np.clip(cc / cc.sum(), 1e-12, None))).float().to(args.device)
    dur_logp = torch.from_numpy(np.log(np.clip(dc / dc.sum(), 1e-12, None))).float().to(args.device)
    log(f"unigram baselines: code {float(-(torch.softmax(code_logp,0)*code_logp).sum()):.4f}, "
        f"duration {float(-(torch.softmax(dur_logp,0)*dur_logp).sum()):.4f}  "
        f"(most common duration = {dc.max()/dc.sum():.3f} of tokens)")

    model = ContentLMDedup(
        num_codes=K, max_run=args.max_run, ambient_dim=ambient_dim, max_tokens=max_tokens,
        d_model=args.d_model, n_layers=args.layers, n_heads=args.heads, dropout=args.dropout,
        use_plan=args.use_plan, n_think=args.n_think,
    ).to(args.device)
    log(f"ContentLMDedup: {model.num_parameters()/1e6:.2f}M params, K={K}, "
        f"max_run={args.max_run}, up to {max_tokens} tokens")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_train = tok_tr.shape[0]
    steps = max(1, n_train // args.batch_size)
    total = steps * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total, pct_start=0.05, anneal_strategy="cos")
    log(f"{steps} steps/epoch, {total} total")

    history: list[dict] = []
    best_gap, best_loss = -1e9, 1e9
    rng = np.random.default_rng(args.seed)
    step = 0
    t_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        order = rng.permutation(n_train)
        run_c = run_d = run_p = run_ctr = 0.0
        nb = 0
        for bi in range(steps):
            sel = np.sort(order[bi * args.batch_size : (bi + 1) * args.batch_size])
            if len(sel) < 2:
                continue
            a = torch.from_numpy(np.asarray(amb_tr[sel], dtype=np.float32)).to(args.device)
            tf = build_teacher_forcing(
                torch.from_numpy(tok_tr[sel]).to(args.device),
                torch.from_numpy(dur_tr[sel]).to(args.device),
                torch.from_numpy(len_tr[sel]).to(args.device),
                model.bos_id, model.num_codes,
            )
            cl, dl = model(a, tf["codes_in"], tf["durs_in"], pad_mask=tf["pad_mask"])
            lc = F.cross_entropy(cl.reshape(-1, K + 1), tf["code_tgt"].reshape(-1),
                                 ignore_index=-100)
            ld = F.cross_entropy(dl.reshape(-1, args.max_run), tf["dur_tgt"].reshape(-1),
                                 ignore_index=-100)
            loss = lc + args.lambda_dur * ld
            lctr_val = 0.0
            if args.lambda_contrast > 0.0:
                # Every intervention so far tried to make better USE of the conditioning; none
                # ever trained the model to use it. scene_gap has only ever been a diagnostic.
                #
                # Per-sample softplus rather than a hinge: no margin to tune, smooth gradients,
                # and it is exactly the InfoNCE two-way form -- treating the true ambient as the
                # positive and a different clip's as the negative,
                #     -log( e^{-CE_true} / (e^{-CE_true} + e^{-CE_rolled}) )
                #   = softplus(CE_true - CE_rolled)
                # Rolled, not zeroed: a zeroed memory is out of distribution, so pushing away
                # from it would teach avoidance of something the model never sees at generation.
                mem_c = model.encode_ambient(a)
                cl_r, _ = model(a, tf["codes_in"], tf["durs_in"],
                                memory=torch.roll(mem_c, shifts=1, dims=0),
                                pad_mask=tf["pad_mask"])
                flat_t = tf["code_tgt"].reshape(-1)
                per = lambda logits: F.cross_entropy(               # noqa: E731
                    logits.reshape(-1, logits.shape[-1]), flat_t,
                    ignore_index=-100, reduction="none",
                ).view(tf["code_tgt"].shape).sum(1) / (tf["code_tgt"] != -100).sum(1).clamp(min=1)
                ce_true, ce_roll = per(cl), per(cl_r)
                lctr = F.softplus(ce_true - ce_roll).mean()
                loss = loss + args.lambda_contrast * lctr
                lctr_val = float(lctr)
            lp_val = 0.0
            if args.use_plan:
                # Supervise the plan directly. Without this the plan_proj is zero-initialised
                # and has no gradient path that distinguishes it from any other bias, so it
                # would learn nothing useful and the experiment would silently be a no-op.
                mem = model.encode_ambient(a)
                plan_logits = model.predict_plan(mem)
                plan_tgt = plan_targets(
                    torch.from_numpy(tok_tr[sel]).to(args.device),
                    torch.from_numpy(len_tr[sel]).to(args.device), K,
                )
                lp = F.binary_cross_entropy_with_logits(plan_logits, plan_tgt)
                loss = loss + args.lambda_plan * lp
                lp_val = float(lp)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step < total - 1:
                sched.step()
            step += 1
            run_c += float(lc.detach()); run_d += float(ld.detach())
            run_p += lp_val; run_ctr += lctr_val; nb += 1

        want_sample = (args.sample_every > 0
                       and ((epoch + 1) % args.sample_every == 0 or epoch == args.epochs - 1))
        m = evaluate(model, amb_va, tok_va, dur_va, len_va,
                     args.batch_size, args.device, code_logp, dur_logp,
                     do_sample=want_sample)
        m.update({
            "epoch": epoch,
            "train_code_loss": run_c / max(nb, 1),
            "train_dur_loss": run_d / max(nb, 1),
            # Logged every epoch even when the plan is off (then it is a constant 0.0). A plan
            # arm whose value never moves means the head learned nothing and plan_proj stayed
            # near zero -- the run is a no-op, not a null result, and must not be read as one.
            "train_plan_loss": run_p / max(nb, 1),
            "train_contrast_loss": run_ctr / max(nb, 1),
            "epoch_minutes": (time.time() - t0) / 60.0,
            "lr": float(opt.param_groups[0]["lr"]),
            "cfg_dropout": args.dropout, "cfg_weight_decay": args.weight_decay,
            "cfg_layers": args.layers, "cfg_max_run": args.max_run,
            "cfg_lambda_dur": args.lambda_dur, "cfg_num_codes": K, "cfg_truncate_speech": args.truncate_speech,
            "cfg_use_plan": args.use_plan, "cfg_lambda_plan": args.lambda_plan,
            "cfg_lambda_contrast": args.lambda_contrast,
            "cfg_n_think": args.n_think,
            "true_mean_tokens": float(len_va.mean()),
        })
        history.append(m)
        plan_str = f" p {m['train_plan_loss']:.4f}" if args.use_plan else ""
        log(f"  ep {epoch:03d} train(c {m['train_code_loss']:.3f} d {m['train_dur_loss']:.3f}"
            f"{plan_str}) "
            f"val(c {m['val_code_loss']:.3f} d {m['val_dur_loss']:.3f}) "
            f"SCENE_GAP={m['val_scene_gap']:+.4f} dur_acc={m['val_dur_acc']:.3f} "
            + (f"tok={m['val_pred_tokens']:.0f}/{m['true_mean_tokens']:.0f} "
               if "val_pred_tokens" in m else "")
            + f"({m['epoch_minutes']:.2f}m)")

        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        payload = {
            "model": model.state_dict(), "epoch": epoch, "metrics": m,
            "codebook": str(cb_path), "num_codes": K, "max_run": args.max_run,
            "ambient_dim": ambient_dim, "max_tokens": max_tokens,
            "d_model": args.d_model, "layers": args.layers, "heads": args.heads,
            "codes": cb["codes"],          # rides along so generation needs one file
            "dedup": True, "use_plan": args.use_plan, "n_think": args.n_think,
        }
        torch.save(payload, run_dir / "checkpoints" / "checkpoint_last.pt")
        if m["val_scene_gap"] > best_gap:
            best_gap = m["val_scene_gap"]
            torch.save(payload, run_dir / "checkpoints" / "checkpoint_best.pt")
        if m["val_loss"] < best_loss:
            best_loss = m["val_loss"]
            torch.save(payload, run_dir / "checkpoints" / "checkpoint_best_loss.pt")
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save(payload, run_dir / "checkpoints" / f"checkpoint_ep{epoch}.pt")

        if args.max_minutes > 0 and (time.time() - t_start) / 60.0 >= args.max_minutes:
            log(f"  stopping at ep {epoch}: hit --max-minutes={args.max_minutes:g}. "
                "Checkpoints written; the LR schedule did NOT complete.")
            break

    log("\n" + "=" * 74)
    log("CONTENT LM (DEDUP) SUMMARY")
    log("=" * 74)
    bg = max(history, key=lambda r: r["val_scene_gap"])
    bl = min(history, key=lambda r: r["val_loss"])
    log(f"val_scene_gap : max {bg['val_scene_gap']:+.4f} @ep{bg['epoch']} | "
        f"at min-loss ep{bl['epoch']}: {bl['val_scene_gap']:+.4f}")
    log("    frame-level reference: +0.021 at the loss minimum, invariant across 7 configs")
    log(f"val_code_loss : min {bl['val_code_loss']:.4f} (unigram {bl['val_code_unigram']:.4f})")
    log(f"val_dur_loss  : {bl['val_dur_loss']:.4f} (unigram {bl['val_dur_unigram']:.4f}), "
        f"acc {bl['val_dur_acc']:.3f}")
    sampled = [h for h in history if "val_pred_tokens" in h]
    if sampled:
        last = sampled[-1]
        log(f"sampled tokens: {last['val_pred_tokens']:.1f} @ep{last['epoch']} vs true mean "
            f"{last['true_mean_tokens']:.1f}  <- far below means smeared static content")
    log("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
