"""
Rung 3: AudioLDM's training recipe applied to S2A (speech -> ambient audio).

This is AudioLDM's own objective, unchanged: epsilon-prediction MSE in the VAE latent
space, with classifier-free guidance trained by dropping the condition on a fraction of
steps. The only substitution is WHERE the conditioning vector comes from -- instead of a
CLAP text embedding, a trainable mapper produces it from frozen speech-encoder features.

Frozen: speech encoder, mel VAE, HiFi-GAN vocoder.
Trainable: mapper + U-Net.

Run with torchrun for DDP across the node's 4 H100s:
    torchrun --nproc_per_node=4 -m salm.train_ldm --config configs/s2a_rung3.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from configs import paths as project_paths  # noqa: E402
from salm.data.dataset import S2ALatentDataset, collate  # noqa: E402
from salm.encoders.whisper_encoder import WhisperSpeechEncoder  # noqa: E402
from salm.mapper import SpeechToClapMapper  # noqa: E402


@dataclass
class TrainConfig:
    # data
    features_root: str = str(project_paths.FEATURES_ROOT)
    direction: str = "s2a"
    train_split: str = "train"
    val_split: str = "val"
    speech_seconds: float = 16.0

    # model
    audioldm_model_id: str = project_paths.AUDIOLDM_MODEL_ID
    encoder: str = "whisper"
    whisper_path: str = str(project_paths.WHISPER_LARGE_V3_PATH)
    mapper_hidden_dim: int = 1024
    mapper_layers: int = 2
    mapper_dropout: float = 0.1

    # optimization
    batch_size: int = 16          # per GPU
    num_workers: int = 6
    epochs: int = 20
    lr_mapper: float = 1e-4
    lr_unet: float = 1e-5         # much lower: the U-Net is pretrained, the mapper is not
    weight_decay: float = 1e-2
    warmup_steps: int = 500
    grad_clip: float = 1.0
    cfg_dropout_prob: float = 0.1
    freeze_unet: bool = False     # True reduces this to a mapper-only run
    seed: int = 1337

    # runtime
    amp_dtype: str = "bf16"       # H100: bf16 needs no loss scaling
    log_every: int = 25
    val_every_epochs: int = 1
    save_every_epochs: int = 5
    run_name: str = "s2a_rung3"
    output_root: str = str(project_paths.CHECKPOINTS_ROOT)
    max_train_minutes: float = 0.0   # >0 sets an internal deadline below the Slurm limit

    extra: dict = field(default_factory=dict)


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def log(msg: str) -> None:
    if is_main():
        print(msg, flush=True)


def setup_distributed() -> tuple[int, int, int]:
    """Returns (rank, world_size, local_rank). Works single-process too."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank, world_size, local_rank = 0, 1, 0
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def build_lr_lambda(warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay."""

    def fn(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="YAML overriding TrainConfig fields")
    ap.add_argument("--override", nargs="*", default=[], help="key=value overrides")
    args = ap.parse_args()

    cfg = TrainConfig()
    if args.config:
        raw = yaml.safe_load(Path(args.config).read_text()) or {}
        for k, v in raw.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            else:
                cfg.extra[k] = v
    for item in args.override:
        k, _, v = item.partition("=")
        if not hasattr(cfg, k):
            raise SystemExit(f"unknown config key: {k}")
        cur = getattr(cfg, k)
        setattr(cfg, k, type(cur)(v) if not isinstance(cur, bool) else v.lower() == "true")

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(cfg.seed + rank)

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg.amp_dtype]

    run_dir = Path(cfg.output_root) / cfg.run_name
    if is_main():
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    log("=" * 70)
    log(f"S2A Rung 3 -- AudioLDM diffusion recipe | world_size={world_size}")
    log(f"run dir: {run_dir}")
    log("=" * 70)

    # ------------------------------------------------------------------- data
    train_dir = Path(cfg.features_root) / cfg.direction / cfg.train_split
    val_dir = Path(cfg.features_root) / cfg.direction / cfg.val_split

    train_ds = S2ALatentDataset(train_dir, speech_seconds=cfg.speech_seconds)
    val_ds = S2ALatentDataset(val_dir, speech_seconds=cfg.speech_seconds)
    log(f"train: {len(train_ds)} | val: {len(val_ds)} | latent shape {train_ds.latent_shape}")

    train_sampler = DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=max(2, cfg.num_workers // 2),
        collate_fn=collate,
        pin_memory=True,
    )

    # ----------------------------------------------------------------- models
    log("Loading frozen speech encoder ...")
    if cfg.encoder != "whisper":
        raise NotImplementedError(
            f"encoder {cfg.encoder!r} not wired yet; only 'whisper' is available in this run"
        )
    encoder = WhisperSpeechEncoder(cfg.whisper_path, dtype=torch.float16, device=device)
    log(f"    whisper encoder feature_dim={encoder.feature_dim}")

    log("Loading AudioLDM U-Net + scheduler ...")
    from diffusers import DDPMScheduler, UNet2DConditionModel

    unet = UNet2DConditionModel.from_pretrained(
        cfg.audioldm_model_id, subfolder="unet", torch_dtype=torch.float32
    ).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(
        cfg.audioldm_model_id, subfolder="scheduler"
    )

    cond_dim = unet.config.projection_class_embeddings_input_dim
    log(f"    unet expects class-label dim: {cond_dim}")
    log(f"    unet cross_attention_dim: {unet.config.cross_attention_dim}")
    log(f"    scheduler: {type(noise_scheduler).__name__}, "
        f"train_timesteps={noise_scheduler.config.num_train_timesteps}, "
        f"prediction_type={noise_scheduler.config.prediction_type}")

    mapper = SpeechToClapMapper(
        input_dim=encoder.feature_dim,
        output_dim=cond_dim,
        hidden_dim=cfg.mapper_hidden_dim,
        num_layers=cfg.mapper_layers,
        dropout=cfg.mapper_dropout,
    ).to(device)
    log(f"    mapper trainable params: {mapper.num_trainable_parameters()/1e6:.1f}M")

    if cfg.freeze_unet:
        unet.eval()
        for p in unet.parameters():
            p.requires_grad_(False)
        log("    U-Net FROZEN (mapper-only run)")
    else:
        unet.train()
        unet.enable_gradient_checkpointing()
        log(f"    U-Net trainable params: {sum(p.numel() for p in unet.parameters())/1e6:.1f}M")

    if world_size > 1:
        mapper = DistributedDataParallel(mapper, device_ids=[local_rank])
        if not cfg.freeze_unet:
            unet = DistributedDataParallel(unet, device_ids=[local_rank])

    mapper_module = mapper.module if isinstance(mapper, DistributedDataParallel) else mapper

    # -------------------------------------------------------------- optimizer
    param_groups = [
        {"params": [p for p in mapper.parameters() if p.requires_grad], "lr": cfg.lr_mapper}
    ]
    if not cfg.freeze_unet:
        param_groups.append(
            {"params": [p for p in unet.parameters() if p.requires_grad], "lr": cfg.lr_unet}
        )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, build_lr_lambda(cfg.warmup_steps, total_steps)
    )
    log(f"steps/epoch={steps_per_epoch} total_steps={total_steps}")

    def compute_condition(batch) -> torch.Tensor:
        """Frozen encoder -> mapper -> (B, cond_dim) L2-normalized CLAP-space vector."""
        with torch.no_grad():
            feats, mask = encoder.encode_with_mask(
                batch["speech"].to(device), batch["speech_length"]
            )
        return mapper(feats, mask=mask)

    def diffusion_loss(batch, train: bool) -> torch.Tensor:
        """AudioLDM's objective: MSE between predicted and true noise in latent space."""
        mean = batch["latent_mean"].to(device, non_blocking=True)
        std = batch["latent_std"].to(device, non_blocking=True)

        # Sample from the VAE posterior, then apply the scaling factor, as AudioLDM does.
        latents = mean + std * torch.randn_like(std)
        latents = latents * train_ds.vae_scaling_factor

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
        ).long()
        noisy = noise_scheduler.add_noise(latents, noise, timesteps)

        cond = compute_condition(batch)
        if train:
            cond = mapper_module.apply_cfg_dropout(cond, cfg.cfg_dropout_prob)

        # AudioLDM conditions via class_labels (a single global vector), NOT via
        # cross-attention token sequences -- hence encoder_hidden_states=None.
        pred = unet(
            noisy,
            timesteps,
            encoder_hidden_states=None,
            class_labels=cond,
        ).sample

        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"unsupported prediction_type: {noise_scheduler.config.prediction_type}")

        return F.mse_loss(pred.float(), target.float())

    @torch.no_grad()
    def validate() -> float:
        mapper.eval()
        if not cfg.freeze_unet:
            unet.eval()
        total, count = 0.0, 0
        for batch in val_loader:
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                loss = diffusion_loss(batch, train=False)
            total += loss.item() * batch["speech"].shape[0]
            count += batch["speech"].shape[0]
        mapper.train()
        if not cfg.freeze_unet:
            unet.train()
        return total / max(count, 1)

    def save_checkpoint(tag: str, epoch: int, step: int, val_loss: float | None) -> None:
        if not is_main():
            return
        unet_module = unet.module if isinstance(unet, DistributedDataParallel) else unet
        payload = {
            "epoch": epoch,
            "step": step,
            "val_loss": val_loss,
            "config": asdict(cfg),
            "mapper": mapper_module.state_dict(),
        }
        if not cfg.freeze_unet:
            payload["unet"] = unet_module.state_dict()

        # Write then rename. `last` is rewritten every epoch and the job can be killed by
        # the internal deadline or by Slurm at any moment; saving in place would leave a
        # truncated, unloadable file exactly when the partial checkpoint matters most.
        path = run_dir / f"checkpoint_{tag}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        log(f"    saved {path} (epoch {epoch}, step {step})")

    # ------------------------------------------------------------------ train
    deadline = time.time() + cfg.max_train_minutes * 60 if cfg.max_train_minutes > 0 else None
    history: list[dict] = []
    global_step = 0
    best_val = float("inf")
    t_start = time.time()
    stop = False

    for epoch in range(cfg.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        running, seen = 0.0, 0
        t_epoch = time.time()

        for batch in train_loader:
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                loss = diffusion_loss(batch, train=True)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            trainable = [p for g in param_groups for p in g["params"]]
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)

            optimizer.step()
            scheduler.step()

            running += loss.item()
            seen += 1
            global_step += 1

            if global_step % cfg.log_every == 0:
                lrs = [g["lr"] for g in optimizer.param_groups]
                log(f"  ep{epoch:02d} step{global_step:06d} "
                    f"loss={running/max(seen,1):.4f} "
                    f"lr={'/'.join(f'{x:.2e}' for x in lrs)} "
                    f"{seen/max(time.time()-t_epoch,1e-6):.2f} it/s")

            if deadline is not None and time.time() > deadline:
                log(f"  internal deadline reached at step {global_step}; stopping cleanly")
                stop = True
                break

        train_loss = running / max(seen, 1)

        val_loss = None
        if (epoch + 1) % cfg.val_every_epochs == 0 or stop:
            val_loss = validate()
            log(f"  epoch {epoch:02d} train={train_loss:.4f} val={val_loss:.4f} "
                f"({(time.time()-t_epoch)/60:.1f} min)")
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint("best", epoch, global_step, val_loss)
        else:
            log(f"  epoch {epoch:02d} train={train_loss:.4f} ({(time.time()-t_epoch)/60:.1f} min)")

        history.append(
            {"epoch": epoch, "step": global_step, "train_loss": train_loss, "val_loss": val_loss}
        )
        if is_main():
            (run_dir / "history.json").write_text(json.dumps(history, indent=2))

        # Always refresh `last` so a current checkpoint exists on disk at all times.
        # `best` only appears when val improves, so on a plateau it can go stale, and
        # waiting for the loop to end would leave nothing recent if the job is killed.
        # This is what makes "generate from whatever we have right now" reliable.
        save_checkpoint("last", epoch, global_step, val_loss)

        # Periodic immutable snapshots, kept separately from `last` so samples can be
        # compared across training time.
        if (epoch + 1) % cfg.save_every_epochs == 0:
            save_checkpoint(f"ep{epoch:02d}", epoch, global_step, val_loss)

        if stop:
            break

    save_checkpoint("last", epoch, global_step, history[-1]["val_loss"] if history else None)
    log(f"\ndone in {(time.time()-t_start)/60:.1f} min | best val {best_val:.4f}")

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
