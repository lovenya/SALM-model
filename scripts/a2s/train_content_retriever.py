"""
Train ContentRetriever: a contrastively-trained ambient<->content matching function, meant
to replace raw CLAP cosine similarity as the retrieval key. See
src/salm/a2s/content_retriever.py for the full mechanism and why this fixes the
AmbientContentPredictor's failure mode (blurry continuous regression) while a plain regressor
could not: this NEVER outputs content directly, only a similarity score used to pick a real
training example's real content, so it cannot produce a blurry average.

Single-GPU: two small MLPs, well under DDP territory.

Usage:
    python scripts/a2s/train_content_retriever.py --run-name content_retriever_v1
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
from salm.a2s.content_retriever import ContentRetriever, retrieval_contrastive_loss  # noqa: E402
from salm.a2s.ctc_probe import SimpleCTCASR  # noqa: E402
from salm.a2s.dataset import A2SLatentDataset, a2s_collate  # noqa: E402
from salm.data.frontend import AUDIOLDM_CLIP_SECONDS, MelConfig, MelExtractor, load_waveform_mono_16k  # noqa: E402
from salm.train_ldm import build_lr_lambda  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


@torch.no_grad()
def true_content_seq(batch: dict, extractor: MelExtractor, ctc_head: SimpleCTCASR, device) -> torch.Tensor:
    wavs = [
        load_waveform_mono_16k(f, target_seconds=AUDIOLDM_CLIP_SECONDS, align="left")
        for f in batch["speech_file"]
    ]
    wav_t = torch.from_numpy(np.stack(wavs)).float().to(device)
    return ctc_head.encode(extractor.batch(wav_t))


@torch.no_grad()
def evaluate(model, extractor, ctc_head, loader, device) -> dict:
    """
    val_r1: within each eval BATCH, does the correct content rank #1 for its own ambient
    query? This is the metric that actually matters -- it is the same operation retrieval
    performs at inference (find the best match), just measured against known-correct answers
    instead of the full training set.
    """
    model.eval()
    total_loss, r1_sum, n = 0.0, 0.0, 0
    for batch in loader:
        ambient = batch["cond_features"].to(device)
        mask = batch["cond_mask"].to(device)
        target = true_content_seq(batch, extractor, ctc_head, device)
        a_emb, c_emb = model(ambient, mask, target)
        loss = retrieval_contrastive_loss(a_emb, c_emb, model.log_temperature)
        sims = a_emb @ c_emb.T
        truth = torch.arange(sims.shape[0], device=device)
        rank = (sims > sims.gather(1, truth[:, None])).sum(1)
        bs = ambient.shape[0]
        total_loss += loss.item() * bs
        r1_sum += (rank == 0).sum().item()
        n += bs
    model.train()
    return {"val_loss": total_loss / max(n, 1), "val_r1": r1_sum / max(n, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="content_retriever_v1")
    ap.add_argument("--runs-root", default=str(project_paths.OUTPUTS_ROOT / "content_retriever"))
    ap.add_argument("--features-root", default=str(project_paths.FEATURES_ROOT / "a2s"))
    ap.add_argument("--cond-root", default=str(project_paths.OUTPUTS_ROOT / "features_a2s_cond"))
    ap.add_argument("--ctc-checkpoint",
                     default=str(project_paths.OUTPUTS_ROOT / "ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256,
                     help="Larger than the predictor's: more in-batch negatives makes the "
                          "contrastive loss a harder, more informative task per step.")
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--eval-every-epochs", type=int, default=1)
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

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=a2s_collate, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=max(2, args.num_workers // 2),
        collate_fn=a2s_collate, pin_memory=True, drop_last=True,  # r1 needs >=2 rows/batch
    )

    extractor = MelExtractor(MelConfig(norm_mode="none"), device=device)
    ctc_head = SimpleCTCASR().to(device).eval()
    ctc_payload = torch.load(args.ctc_checkpoint, map_location=device, weights_only=False)
    ctc_head.load_state_dict(ctc_payload["model"])
    for p in ctc_head.parameters():
        p.requires_grad_(False)
    ctc_head.rnn.train()  # cuDNN backward-mode quirk; see train_ldm.py's O-F block for why

    model = ContentRetriever(
        ambient_dim=train_ds.cond_feature_dim, content_dim=ctc_head.encoder_dim,
        embed_dim=args.embed_dim,
    ).to(device)
    log(f"retriever params: {model.num_parameters()/1e3:.1f}k")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, build_lr_lambda(args.warmup_steps, total_steps))
    log(f"steps/epoch={steps_per_epoch} total_steps={total_steps}")
    chance_r1 = 1.0 / args.batch_size
    log(f"chance val_r1 (batch_size={args.batch_size}): {chance_r1:.4f}")

    start_epoch, global_step = 0, 0
    best_r1 = -1.0
    history: list[dict] = []
    history_path = run_dir / "history.json"
    if history_path.exists() and not args.resume_from:
        raise SystemExit(
            f"{history_path} already exists and --resume-from was not given -- refusing to "
            "silently mix an old run's epochs into this run_name's history."
        )

    if args.resume_from:
        payload = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch = payload["epoch"] + 1
        global_step = payload["step"]
        best_r1 = payload.get("best_r1", -1.0)
        if history_path.exists():
            history = json.loads(history_path.read_text())
        log(f"resumed from {args.resume_from}: epoch={start_epoch} step={global_step}")

    def save_checkpoint(tag: str, epoch: int) -> None:
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch, "step": global_step,
            "best_r1": best_r1, "config": vars(args),
        }, ckpt_dir / f"checkpoint_{tag}.pt")

    t_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        t_epoch = time.time()
        running, seen = 0.0, 0
        for batch in train_loader:
            ambient = batch["cond_features"].to(device)
            mask = batch["cond_mask"].to(device)
            target = true_content_seq(batch, extractor, ctc_head, device)

            a_emb, c_emb = model(ambient, mask, target)
            loss = retrieval_contrastive_loss(a_emb, c_emb, model.log_temperature)

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
                   "epoch_minutes": (time.time() - t_epoch) / 60, "chance_r1": chance_r1, **metrics}
        history.append(record)
        history_path.write_text(json.dumps(history, indent=2))

        log(f"  epoch {epoch:03d} train={train_loss:.4f}"
            + (f" val={metrics['val_loss']:.4f} val_r1={metrics['val_r1']:.4f}"
               f"({metrics['val_r1']/chance_r1:.1f}x chance)" if metrics else "")
            + f"  ({record['epoch_minutes']:.2f} min)")

        save_checkpoint("last", epoch)
        if metrics and metrics["val_r1"] > best_r1:
            best_r1 = metrics["val_r1"]
            save_checkpoint("best", epoch)
        if epoch % args.save_every_epochs == 0:
            save_checkpoint(f"ep{epoch:02d}", epoch)

        if args.max_train_minutes > 0 and (time.time() - t_start) / 60 > args.max_train_minutes:
            log(f"hit --max-train-minutes={args.max_train_minutes}, stopping at epoch {epoch}")
            break

    log(f"done. best val_r1={best_r1:.4f} ({best_r1/chance_r1:.1f}x chance)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
