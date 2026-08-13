"""
T0 trainer -- direct latent regression, no diffusion.

    frozen Whisper -> LatentRegressor -> latent (8,256,16) -> [frozen VAE -> vocoder at gen time]

Loss is plain MSE against the precomputed target latent. That is the whole objective; there is no
noise schedule, no timestep sampling and no U-Net. This exists so the ladder has a rung BELOW the
diffusion models, which is what licenses the diffusion complexity in the first place.

THE METRIC THAT MATTERS HERE is `val_pred_std`. Measured on our train latents: a single real latent
has std 1.032, a blend of two has 0.753, a blend of 16 has 0.421. AudioLDM's decoder expects ~1.0.
So if this model's predictions come out at std ~0.5, the variance collapse behind "regression sounds
blurry" is measured rather than asserted, and that one number motivates diffusion directly. It is
logged every epoch and is the recommended `select_metric` alternative if MSE proves uninformative.

Deliberately shares configs/base/s2a_audioldm.yaml and configs/exp/long100.yaml with the diffusion
rungs, so `epochs` and `lr_total_steps` match and the comparison is legitimate.

Launch:
    torchrun --nproc_per_node=4 -m salm.train_regress \
        --config configs/base/s2a_audioldm.yaml configs/arch/s2a_t0_whisper_regress.yaml \
                 configs/exp/long100.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from configs import paths as project_paths
from salm.data.dataset import S2ALatentDataset, collate
from salm.latent_regressor import build_latent_regressor
from salm.train_ldm import (
    TrainConfig,
    build_lr_lambda,
    init_wandb,
    is_main,
    log,
    setup_distributed,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="*", default=[])
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    cfg = TrainConfig()
    known = {f.name for f in dataclasses.fields(TrainConfig)}
    for path in args.config:
        for k, v in (yaml.safe_load(Path(path).read_text()) or {}).items():
            if k in known:
                setattr(cfg, k, v)
    for item in args.override:
        k, _, v = item.partition("=")
        if k not in known:
            continue
        cur = getattr(cfg, k)
        setattr(cfg, k, type(cur)(v) if not isinstance(cur, bool) else v.lower() == "true")

    # Order matters and bit me: setup_distributed returns (rank, world_size, local_rank), as its
    # own docstring says. Unpacking it as (rank, local_rank, world_size) made rank 0 see
    # local_rank=4 and world_size=0, so `cuda:4` raised "invalid device ordinal" and job 406978
    # died in 32 s. Keep this identical to train_ldm.py's call site.
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed + rank)

    run_dir = Path(cfg.runs_root) / cfg.arch / cfg.run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log(f"T0 -- direct latent REGRESSION (no diffusion) | world_size={world_size}")
    log(f"arch:    {cfg.arch}")
    log(f"run dir: {run_dir}")
    log("=" * 70)

    # ------------------------------------------------------------------- data
    train_dir = Path(cfg.features_root) / cfg.direction / cfg.train_split
    val_dir = Path(cfg.features_root) / cfg.direction / cfg.val_split
    train_ds = S2ALatentDataset(train_dir, speech_seconds=cfg.speech_seconds)
    val_ds = S2ALatentDataset(val_dir, speech_seconds=cfg.speech_seconds)
    log(f"train {len(train_ds)} | val {len(val_ds)} | latent {train_ds.latent_shape} "
        f"| vae_scaling {train_ds.vae_scaling_factor:.4f}")

    train_sampler = DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False) \
        if world_size > 1 else None
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=train_sampler,
        shuffle=train_sampler is None, num_workers=cfg.num_workers,
        pin_memory=True, drop_last=True, collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, sampler=val_sampler, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate,
    )

    # ------------------------------------------------------------------ model
    from salm.architectures import ENCODERS, parse_encoder

    encoder_name = parse_encoder(cfg.arch)
    encoder = ENCODERS[encoder_name](dataclasses.asdict(cfg), device)
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()

    model = build_latent_regressor(
        encoder.feature_dim, train_ds.latent_shape, dataclasses.asdict(cfg)
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"regressor params: {n_params:,} ({n_params/1e6:.2f} M)  frozen encoder: "
        f"{sum(p.numel() for p in encoder.parameters())/1e6:.0f} M")

    module = model
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank])
        module = model.module

    optimizer = torch.optim.AdamW(
        module.parameters(), lr=cfg.lr_conditioner, weight_decay=cfg.weight_decay
    )
    steps_per_epoch = max(1, len(train_loader))
    total_steps = cfg.lr_total_steps or steps_per_epoch * cfg.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, build_lr_lambda(cfg.warmup_steps, total_steps)
    )
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.amp_dtype, torch.float32)

    wandb_run = init_wandb(cfg, run_dir, world_size) if is_main() else None

    def target_latents(batch) -> torch.Tensor:
        """
        The same target the diffusion rungs see: a draw from the VAE posterior, scaled by
        AudioLDM's factor. Using the posterior MEAN instead would make this an easier problem
        than the one the diffusion models solve, and the comparison would be invalid.
        """
        mean = batch["latent_mean"].to(device, non_blocking=True)
        std = batch["latent_std"].to(device, non_blocking=True)
        return (mean + std * torch.randn_like(std)) * train_ds.vae_scaling_factor

    @torch.no_grad()
    def validate() -> dict:
        model.eval()
        mse_sum, n_seen = 0.0, 0
        pred_sq, tgt_sq, n_el = 0.0, 0.0, 0
        r1_sum, r_count = 0.0, 0
        for batch in val_loader:
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                feats, mask = encoder.encode_with_mask(
                    batch["speech"].to(device), batch["speech_length"]
                )
                pred = model(feats, mask=mask)
            pred = pred.float()
            tgt = target_latents(batch).float()
            n = pred.shape[0]

            mse_sum += F.mse_loss(pred, tgt).item() * n
            n_seen += n
            # Second moments, accumulated raw so the DDP reduce stays exact. These are what
            # make the variance-collapse claim measurable instead of rhetorical.
            pred_sq += pred.pow(2).sum().item()
            tgt_sq += tgt.pow(2).sum().item()
            n_el += pred.numel()

            # Does a predicted latent match its OWN target better than the other targets in the
            # batch? Chance is 1/n. The analogue of val_clap_r1 for this model, and the only
            # metric here with an unambiguous baseline.
            if n >= 2:
                p = pred.flatten(1)
                t = tgt.flatten(1)
                d = (p.pow(2).sum(1, keepdim=True) - 2 * p @ t.t()
                     + t.pow(2).sum(1)[None, :])          # squared distances (n, n)
                truth = torch.arange(n, device=device)
                r1_sum += (d.argmin(1) == truth).sum().item()
                r_count += n

        if dist.is_initialized():
            packed = torch.tensor(
                [mse_sum, float(n_seen), pred_sq, tgt_sq, float(n_el), r1_sum, float(r_count)],
                device=device, dtype=torch.float64,
            )
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            (mse_sum, n_seen, pred_sq, tgt_sq, n_el, r1_sum, r_count) = (
                packed[0].item(), int(packed[1]), packed[2].item(), packed[3].item(),
                int(packed[4]), packed[5].item(), int(packed[6]),
            )
        model.train()

        out = {
            "val_loss": mse_sum / max(n_seen, 1),
            # RMS about zero, which is what the decoder's input scale actually depends on.
            "val_pred_std": math.sqrt(pred_sq / max(n_el, 1)),
            "val_target_std": math.sqrt(tgt_sq / max(n_el, 1)),
            "val_samples": n_seen,
        }
        # The headline number: 1.0 means the prediction has the same scale as real latents;
        # well below 1.0 is the interpolation collapse that makes regression sound washed out.
        out["val_std_ratio"] = out["val_pred_std"] / max(out["val_target_std"], 1e-8)
        if r_count:
            out["val_latent_r1"] = r1_sum / r_count
            out["val_latent_r1_chance"] = 1.0 / cfg.batch_size
        return out

    def save(tag: str, epoch: int, step: int, val_loss: float | None) -> None:
        if not is_main():
            return
        payload = {
            "model": module.state_dict(),
            "config": dataclasses.asdict(cfg),
            "epoch": epoch,
            "step": step,
            "val_loss": val_loss,
            "latent_shape": list(train_ds.latent_shape),
            "vae_scaling_factor": float(train_ds.vae_scaling_factor),
            "model_kind": "latent_regressor",   # so generate.py cannot mistake this for an LDM
        }
        dest = run_dir / "checkpoints" / f"checkpoint_{tag}.pt"
        tmp = dest.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.replace(dest)            # atomic: a killed job never leaves a truncated checkpoint

    history: list[dict] = []
    best = float("inf")
    deadline = time.time() + cfg.max_train_minutes * 60 if cfg.max_train_minutes else None
    global_step, stop = 0, False

    for epoch in range(cfg.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        run_sum, run_n = 0.0, 0
        t0 = time.time()

        for batch in train_loader:
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                feats, mask = encoder.encode_with_mask(
                    batch["speech"].to(device), batch["speech_length"]
                )
                pred = model(feats, mask=mask)
            loss = F.mse_loss(pred.float(), target_latents(batch).float())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            run_sum += loss.item()
            run_n += 1
            global_step += 1

            if global_step % cfg.log_every == 0:
                log(f"  ep{epoch:02d} step{global_step:06d} loss={run_sum/max(run_n,1):.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} "
                    f"{run_n/max(time.time()-t0,1e-6):.2f} it/s")
                # Collective stop: every rank must agree, or the ranks still in the loop hang on
                # the next gradient all-reduce until the NCCL watchdog kills the job. That exact
                # bug destroyed job 406301 after 91 epochs.
                if deadline is not None:
                    flag = torch.tensor(
                        [1.0 if time.time() > deadline else 0.0], device=device
                    )
                    if dist.is_initialized():
                        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                    if flag.item() > 0:
                        stop = True
                        break

        metrics = validate()
        log(f"  epoch {epoch:02d} train={run_sum/max(run_n,1):.4f} val={metrics['val_loss']:.4f} "
            f"pred_std={metrics['val_pred_std']:.4f} target_std={metrics['val_target_std']:.4f} "
            f"ratio={metrics['val_std_ratio']:.3f}"
            + (f" r1={metrics['val_latent_r1']:.3f}/{metrics['val_latent_r1_chance']:.3f}"
               if "val_latent_r1" in metrics else "")
            + f" ({(time.time()-t0)/60:.1f} min)")

        record = {"epoch": epoch, "step": global_step,
                  "train_loss": run_sum / max(run_n, 1),
                  "epoch_minutes": (time.time() - t0) / 60, **metrics}
        history.append(record)
        if is_main():
            (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        if wandb_run is not None:
            wandb_run.log(
                {"train/epoch_loss": record["train_loss"],
                 **{f"val/{k.replace('val_','')}": v for k, v in metrics.items()},
                 "epoch": epoch},
                step=global_step,
            )

        if metrics["val_loss"] < best:
            best = metrics["val_loss"]
            save("best", epoch, global_step, metrics["val_loss"])
        save("last", epoch, global_step, metrics["val_loss"])
        if stop:
            log("  stopping: internal deadline reached (all ranks agreed)")
            break

    if is_main() and history:
        h = history[-1]
        log("=" * 70)
        log(f"done. {len(history)} epochs | best val MSE {best:.4f}")
        log(f"final pred_std {h['val_pred_std']:.4f} vs target {h['val_target_std']:.4f} "
            f"(ratio {h['val_std_ratio']:.3f})")
        log("  ratio well below 1.0 = the variance collapse that makes regression sound washed")
        log("  out, measured. Reference: blending 2 real latents gives 0.73, blending 16 gives 0.41.")
        log("=" * 70)
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
