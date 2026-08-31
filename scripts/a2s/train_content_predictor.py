"""
Train AmbientContentPredictor: ambient audio -> predicted content sequence, no target
speech touched at inference. See src/salm/a2s/content_predictor.py for the full mechanism
and why this is not cheating (the target is a training-time answer key, exactly like O2's
alignment loss already uses the target's CLAP-text embedding).

WHY THIS IS ITS OWN TRAINING PASS, not folded into train_ldm.py. Same reasoning as the CTC
probe: this is a small, fast regression problem (predict a sequence, not denoise a 185M-param
U-Net), and the renderer it will feed (content_concat) is ALREADY TRAINED AND PROVEN
(2026-08-30, job 433314: oracle content -> near-verbatim speech). Nothing here should touch
the renderer -- this only trains a better GUESS to hand it, replacing nearest-neighbour
retrieval with a real learned predictor.

DATA: reuses A2SLatentDataset for the ambient side (cond_seq.npy / cond_mask.npy, already
precomputed and row-aligned) and computes the TRUE content sequence on the fly, from each
row's own target speech, via the frozen CTC probe -- exactly the same computation
content_concat.py's oracle path does inside train_ldm.py, just here as the SUPERVISION
target for a regression loss instead of an input to the renderer.

Single-GPU by design: the predictor is a few million params against 17.6k rows: no need for
DDP, and single-process avoids sampler/all-reduce complexity for a component that only has to
predict "close enough" content, not match the renderer's own scale.

Usage:
    python scripts/a2s/train_content_predictor.py --run-name content_predictor_v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from configs import paths as project_paths  # noqa: E402
from salm.a2s.content_predictor import AmbientContentPredictor, content_prediction_loss  # noqa: E402
from salm.a2s.ctc_probe import SimpleCTCASR  # noqa: E402
from salm.a2s.dataset import A2SLatentDataset, a2s_collate  # noqa: E402
from salm.data.frontend import AUDIOLDM_CLIP_SECONDS, MelConfig, MelExtractor, load_waveform_mono_16k  # noqa: E402
from salm.train_ldm import build_lr_lambda  # noqa: E402  (reuse, don't reimplement)


def log(msg: str) -> None:
    print(msg, flush=True)


@torch.no_grad()
def true_content_seq(batch: dict, extractor: MelExtractor, ctc_head: SimpleCTCASR, device) -> torch.Tensor:
    """The training-time answer key: run the frozen CTC probe over each row's OWN target
    speech. Never used again after this call in this script -- it exists only to compute a
    loss, exactly like O2's target embedding is only ever used to compute a loss."""
    wavs = [
        load_waveform_mono_16k(f, target_seconds=AUDIOLDM_CLIP_SECONDS, align="left")
        for f in batch["speech_file"]
    ]
    wav_t = torch.from_numpy(np.stack(wavs)).float().to(device)
    return ctc_head.encode(extractor.batch(wav_t))


@torch.no_grad()
def evaluate(model, extractor, ctc_head, loader, device) -> dict:
    model.eval()
    total_loss, total_cos, n = 0.0, 0.0, 0
    for batch in loader:
        ambient = batch["cond_features"].to(device)
        mask = batch["cond_mask"].to(device)
        target = true_content_seq(batch, extractor, ctc_head, device)
        pred = model(ambient, mask)
        loss = content_prediction_loss(pred, target)
        cos = torch.nn.functional.cosine_similarity(pred, target, dim=-1).mean()
        bs = ambient.shape[0]
        total_loss += loss.item() * bs
        total_cos += cos.item() * bs
        n += bs
    model.train()
    return {"val_loss": total_loss / max(n, 1), "val_cos": total_cos / max(n, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="content_predictor_v1")
    ap.add_argument("--runs-root", default=str(project_paths.OUTPUTS_ROOT / "content_predictor"))
    ap.add_argument("--features-root", default=str(project_paths.FEATURES_ROOT / "a2s"))
    ap.add_argument("--cond-root", default=str(project_paths.OUTPUTS_ROOT / "features_a2s_cond"))
    ap.add_argument("--ctc-checkpoint",
                     default=str(project_paths.OUTPUTS_ROOT / "ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=300)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--eval-every-epochs", type=int, default=2)
    ap.add_argument("--save-every-epochs", type=int, default=20)
    ap.add_argument("--resume-from", default="")
    ap.add_argument("--max-train-minutes", type=float, default=0.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    run_dir = Path(args.runs_root) / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_ds = A2SLatentDataset(
        Path(args.features_root) / "train", Path(args.cond_root) / "train", align_target=None
    )
    val_ds = A2SLatentDataset(
        Path(args.features_root) / "val", Path(args.cond_root) / "val", align_target=None
    )
    log(f"train: {len(train_ds)} | val: {len(val_ds)}")
    log(f"ambient: {train_ds.cond_frames} frames x {train_ds.cond_feature_dim} dims")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=a2s_collate, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=max(2, args.num_workers // 2),
        collate_fn=a2s_collate, pin_memory=True,
    )

    extractor = MelExtractor(MelConfig(norm_mode="none"), device=device)
    ctc_head = SimpleCTCASR().to(device).eval()
    ctc_payload = torch.load(args.ctc_checkpoint, map_location=device, weights_only=False)
    ctc_head.load_state_dict(ctc_payload["model"])
    for p in ctc_head.parameters():
        p.requires_grad_(False)
    ctc_head.rnn.train()  # cuDNN backward-mode quirk; see train_ldm.py's O-F block for why
    log(f"    CTC probe (target key): epoch={ctc_payload.get('epoch')} best_wer={ctc_payload.get('best_wer')}")

    model = AmbientContentPredictor(
        ambient_dim=train_ds.cond_feature_dim,
        content_dim=ctc_head.encoder_dim,
        ambient_frames=train_ds.cond_frames,
        content_frames=256,
    ).to(device)
    log(f"predictor params: {model.num_parameters()/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, build_lr_lambda(args.warmup_steps, total_steps)
    )
    log(f"steps/epoch={steps_per_epoch} total_steps={total_steps}")

    start_epoch, global_step = 0, 0
    best_val = float("inf")
    history: list[dict] = []
    history_path = run_dir / "history.json"
    if history_path.exists() and not args.resume_from:
        raise SystemExit(
            f"{history_path} already exists and --resume-from was not given -- refusing to "
            "silently mix an old run's epochs into this run_name's history. Pass "
            "--resume-from <ckpt> to continue it, or use a different --run-name."
        )

    if args.resume_from:
        payload = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch = payload["epoch"] + 1
        global_step = payload["step"]
        best_val = payload.get("best_val", float("inf"))
        if history_path.exists():
            history = json.loads(history_path.read_text())
        log(f"resumed from {args.resume_from}: epoch={start_epoch} step={global_step}")

    def save_checkpoint(tag: str, epoch: int) -> None:
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "step": global_step,
            "best_val": best_val,
            "config": vars(args),
        }, ckpt_dir / f"checkpoint_{tag}.pt")

    t_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        t_epoch = time.time()
        running, seen = 0.0, 0
        for batch in train_loader:
            ambient = batch["cond_features"].to(device)
            mask = batch["cond_mask"].to(device)
            target = true_content_seq(batch, extractor, ctc_head, device)

            pred = model(ambient, mask)
            loss = content_prediction_loss(pred, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            running += loss.item() * ambient.shape[0]
            seen += ambient.shape[0]

        train_loss = running / max(seen, 1)
        metrics = {}
        if (epoch % args.eval_every_epochs == 0) or (epoch == args.epochs - 1):
            metrics = evaluate(model, extractor, ctc_head, val_loader, device)

        record = {"epoch": epoch, "step": global_step, "train_loss": train_loss,
                   "epoch_minutes": (time.time() - t_epoch) / 60, **metrics}
        history.append(record)
        history_path.write_text(json.dumps(history, indent=2))

        log(f"  epoch {epoch:03d} train={train_loss:.4f}"
            + (f" val={metrics['val_loss']:.4f} val_cos={metrics['val_cos']:.4f}" if metrics else "")
            + f"  ({record['epoch_minutes']:.2f} min)")

        save_checkpoint("last", epoch)
        if metrics and metrics["val_loss"] < best_val:
            best_val = metrics["val_loss"]
            save_checkpoint("best", epoch)
        if epoch % args.save_every_epochs == 0:
            save_checkpoint(f"ep{epoch:02d}", epoch)

        if args.max_train_minutes > 0 and (time.time() - t_start) / 60 > args.max_train_minutes:
            log(f"hit --max-train-minutes={args.max_train_minutes}, stopping at epoch {epoch}")
            break

    log(f"done. best val_loss={best_val:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
