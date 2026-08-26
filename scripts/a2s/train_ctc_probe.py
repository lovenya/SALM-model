"""
Train the frozen CTC content-loss head used by O-F (`train_ldm.py`'s `lambda_ctc` term).

WHY THIS IS ITS OWN TRAINING PASS. O-F needs a scorer that can tell "these mel frames spell
these words" -- but no off-the-shelf ASR model's input matches AudioLDM's mel frontend (16
kHz, 64 bins; most pretrained CTC/ASR models use 80 bins or raw waveform), so a mismatched
model would score content unreliably. This trains `salm.a2s.ctc_probe.SimpleCTCASR` directly
against `MelExtractor(norm_mode="none")` -- exactly what the VAE encodes/decodes -- on real
(target speech, transcript) pairs. Once trained, its checkpoint is loaded FROZEN inside
train_ldm.py; nothing here runs during diffusion training.

Single-GPU by design: this model is 2.4M params against 17.6k rows, nowhere near needing DDP,
and keeping it single-process avoids sampler/all-reduce complexity for a component that only
has to be "good enough to provide a real content gradient", not state-of-the-art ASR.

Usage:
    python scripts/a2s/train_ctc_probe.py --run-name ctc_probe_v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from configs import paths as project_paths  # noqa: E402
from salm.a2s.ctc_probe import CTC_FRAMES_OUT, SimpleCTCASR, greedy_decode_batch  # noqa: E402
from salm.a2s.wer import CTC_BLANK, ctc_encode, ctc_greedy_decode, corpus_wer  # noqa: E402
from salm.data.frontend import (  # noqa: E402
    AUDIOLDM_CLIP_SECONDS,
    MelConfig,
    MelExtractor,
    load_waveform_mono_16k,
)
from salm.data.manifest import load_manifest  # noqa: E402
from salm.train_ldm import build_lr_lambda  # noqa: E402  (reuse, don't reimplement)


def log(msg: str) -> None:
    print(msg, flush=True)


class SpeechTextDataset(Dataset):
    """One item: (fixed-length target-speech waveform, tokenized transcript, raw transcript)."""

    def __init__(self, split: str) -> None:
        self.rows = load_manifest("a2s", split)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        # LEFT-aligned, matching precompute_latents.py's A2S speech convention: the transcript
        # describes the START of the clip, so an over-long utterance must keep its start.
        wav = load_waveform_mono_16k(
            str(row.speech_path), target_seconds=AUDIOLDM_CLIP_SECONDS, align="left"
        )
        ids = ctc_encode(row.speech_text)
        return {
            "wav": torch.from_numpy(wav),
            "ids": torch.tensor(ids, dtype=torch.long),
            "text": row.speech_text,
            "sample_id": row.sample_id,
        }


def collate(batch: list[dict]) -> dict:
    wavs = torch.stack([b["wav"] for b in batch])
    lengths = torch.tensor([len(b["ids"]) for b in batch], dtype=torch.long)
    # Pad with 0 (== CTC_BLANK). Safe: CTCLoss slices each row to its OWN target_lengths[i],
    # so padded positions past the real length are never read, regardless of pad value.
    max_len = max(1, int(lengths.max().item()))
    ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, : len(b["ids"])] = b["ids"]
    return {
        "wav": wavs,
        "ids": ids,
        "target_lengths": lengths,
        "text": [b["text"] for b in batch],
        "sample_id": [b["sample_id"] for b in batch],
    }


@torch.no_grad()
def evaluate(model, extractor, loader, device) -> dict:
    model.eval()
    refs, hyps = [], []
    for batch in loader:
        mel = extractor.batch(batch["wav"].to(device))
        log_probs = model(mel)
        ids_batch = greedy_decode_batch(log_probs)
        refs.extend(batch["text"])
        hyps.extend(ctc_greedy_decode(ids) for ids in ids_batch)
    model.train()
    result = corpus_wer(refs, hyps)
    result["examples"] = list(zip(refs[:4], hyps[:4]))
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="ctc_probe_v1")
    ap.add_argument("--runs-root", default=str(project_paths.OUTPUTS_ROOT / "ctc_probe"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=300)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--eval-every-epochs", type=int, default=2)
    ap.add_argument("--save-every-epochs", type=int, default=10)
    ap.add_argument("--resume-from", default="")
    ap.add_argument("--max-train-minutes", type=float, default=0.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    run_dir = Path(args.runs_root) / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_ds = SpeechTextDataset("train")
    val_ds = SpeechTextDataset("val")
    log(f"train: {len(train_ds)} | val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=collate, pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=max(2, args.num_workers // 2),
        collate_fn=collate, pin_memory=True,
    )

    extractor = MelExtractor(MelConfig(norm_mode="none"), device=device)
    model = SimpleCTCASR().to(device)
    log(f"model params: {model.num_parameters()/1e6:.2f}M")
    ctc_loss_fn = nn.CTCLoss(blank=CTC_BLANK, zero_infinity=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, build_lr_lambda(args.warmup_steps, total_steps)
    )
    log(f"steps/epoch={steps_per_epoch} total_steps={total_steps}")

    start_epoch, global_step = 0, 0
    best_wer = float("inf")
    history: list[dict] = []
    history_path = run_dir / "history.json"
    # Collision guard: an existing history.json with no --resume-from means this run_name was
    # used before and a fresh start here would silently mix an old run's epochs into this
    # one's history.json rather than starting clean. train_ldm.py's own missing version of
    # this guard is a known, documented gap (OPEN_ITEMS) -- not repeating it here.
    if history_path.exists() and not args.resume_from:
        raise SystemExit(
            f"{history_path} already exists but --resume-from was not given. Either pass "
            f"--resume-from {ckpt_dir}/checkpoint_last.pt to continue this run, or use a "
            f"different --run-name for a fresh one."
        )
    if history_path.exists():
        history = json.loads(history_path.read_text())

    if args.resume_from:
        ckpt_path = Path(args.resume_from)
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
            scheduler.load_state_dict(payload["scheduler"])
            global_step = payload["step"]
        start_epoch = payload["epoch"] + 1
        best_wer = payload.get("best_wer", float("inf"))
        log(f"resumed from {ckpt_path}: epoch={payload['epoch']} step={global_step}")

    def save(epoch: int, tag: str, wer: float | None = None) -> None:
        payload = {
            "epoch": epoch, "step": global_step, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "best_wer": best_wer, "args": vars(args),
        }
        torch.save(payload, ckpt_dir / f"checkpoint_{tag}.pt")

    t_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        t_epoch = time.time()
        running, seen = 0.0, 0
        for batch in train_loader:
            mel = extractor.batch(batch["wav"].to(device))
            log_probs = model(mel)                                # (B, T_out, vocab)
            lp = log_probs.transpose(0, 1)                          # (T_out, B, vocab)
            input_lengths = torch.full(
                (lp.shape[1],), CTC_FRAMES_OUT, dtype=torch.long, device="cpu"
            )
            loss = ctc_loss_fn(
                lp, batch["ids"].to(device), input_lengths, batch["target_lengths"]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            running += loss.item()
            seen += 1
            global_step += 1

        train_loss = running / max(seen, 1)
        epoch_minutes = (time.time() - t_epoch) / 60
        record = {"epoch": epoch, "step": global_step, "train_ctc_loss": train_loss,
                  "epoch_minutes": epoch_minutes}

        if (epoch + 1) % args.eval_every_epochs == 0 or epoch == args.epochs - 1:
            val_result = evaluate(model, extractor, val_loader, device)
            record["val_wer"] = val_result["wer"]
            log(f"epoch {epoch:03d} train_ctc_loss={train_loss:.4f} "
                f"val_wer={val_result['wer']:.4f} ({epoch_minutes:.2f} min)")
            for ref, hyp in val_result["examples"][:2]:
                log(f"    REF: {ref[:70]}")
                log(f"    HYP: {hyp[:70]}")
            if val_result["wer"] < best_wer:
                best_wer = val_result["wer"]
                save(epoch, "best", val_result["wer"])
                log(f"    new best val_wer={best_wer:.4f} -> checkpoint_best")
        else:
            log(f"epoch {epoch:03d} train_ctc_loss={train_loss:.4f} ({epoch_minutes:.2f} min)")

        history.append(record)
        history_path.write_text(json.dumps(history, indent=2))

        if (epoch + 1) % args.save_every_epochs == 0:
            save(epoch, f"ep{epoch:03d}")
        save(epoch, "last")

        if args.max_train_minutes > 0 and (time.time() - t_start) / 60 > args.max_train_minutes:
            log(f"hit max_train_minutes={args.max_train_minutes}; stopping at epoch {epoch}")
            break

    log(f"\ndone. best_wer={best_wer:.4f}  checkpoints: {ckpt_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
