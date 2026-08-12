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
# Encoder and conditioner are resolved through salm.architectures at runtime, so this
# module deliberately imports neither directly.


@dataclass
class TrainConfig:
    # data
    features_root: str = str(project_paths.FEATURES_ROOT)
    # CLAP target embeddings for the val cosine metric. Empty string disables the metric.
    clap_features_root: str = str(project_paths.FEATURES_CLAP_ROOT)
    direction: str = "s2a"
    train_split: str = "train"
    val_split: str = "val"
    speech_seconds: float = 16.0

    # model
    # `arch` names an entry in salm.architectures and also groups runs on disk as
    # runs/<arch>/<run_name>/, so experiments stay self-describing.
    arch: str = "s2a_audioldm_whisper_xformer"
    audioldm_model_id: str = project_paths.AUDIOLDM_MODEL_ID
    whisper_path: str = str(project_paths.WHISPER_LARGE_V3_PATH)
    mapper_hidden_dim: int = 1024
    mapper_layers: int = 2
    mapper_dropout: float = 0.1
    mapper_num_queries: int = 8    # qformer only

    # optimization
    batch_size: int = 16          # per GPU
    num_workers: int = 6
    epochs: int = 20
    lr_mapper: float = 1e-4
    lr_unet: float = 1e-5         # much lower: the U-Net is pretrained, the mapper is not
    weight_decay: float = 1e-2
    warmup_steps: int = 500       # base config sets 300; at 234 steps/epoch that is ~1.28 epochs
    grad_clip: float = 1.0
    # Length of the cosine curve, in optimizer steps. 0 = derive from epochs
    # (steps_per_epoch * epochs), the historical behaviour, so queued runs are unaffected.
    #
    # Set it explicitly to make runs comparable even if one is truncated. With a derived
    # length a run configured for 5 epochs is fully annealed at epoch 5 (LR factor 0.000)
    # while a 100-epoch run is barely decayed there (0.997), so the two are not comparable
    # at any epoch past warmup. With a fixed budget both follow the SAME curve and differ
    # only in how far along they stopped. Use for ladder comparisons; leave 0 for a final
    # model, where annealing all the way to zero gives the best quality.
    lr_total_steps: int = 0
    cfg_dropout_prob: float = 0.1
    freeze_unet: bool = False     # True reduces this to a mapper-only run
    seed: int = 1337

    # runtime
    amp_dtype: str = "bf16"       # H100: bf16 needs no loss scaling
    log_every: int = 25
    val_every_epochs: int = 1
    # Timestep-bucket losses cost 4 extra U-Net passes over the val set. Measured: with them
    # every epoch, validation was 2.87 of a 4.5 min epoch (64%) while training was only
    # 1.63 min. They change slowly, so every 5th epoch is plenty and buys back most of that.
    bucket_every_epochs: int = 5
    # 20, not 5: each checkpoint is ~869 MB, scratch is ~90% full, and 100 epochs at
    # every-5 would write 18.7 GB per run. See CONTEXT.md disk budget.
    save_every_epochs: int = 20
    # Periodic ep## snapshots in fp16 to halve their size. `best`/`last` stay fp32
    # because those are what resume and generation load.
    snapshot_fp16: bool = True
    run_name: str = "s2a_rung3"
    runs_root: str = str(project_paths.RUNS_ROOT)
    max_train_minutes: float = 0.0   # >0 sets an internal deadline below the Slurm limit

    # resume: path to a checkpoint_*.pt. Restores weights, optimizer, scheduler, RNG and
    # continues from the next epoch. Checkpoints without optimizer state warm-start the
    # weights only, and say so loudly.
    resume_from: str = ""

    # W&B. Offline because compute nodes have no internet; `wandb sync` later.
    wandb_enabled: bool = True
    wandb_project: str = "salm-s2a"
    # Set explicitly. The account's default_entity is 'gaspardbtt-' (another project's
    # team), so leaving this empty would silently log our runs there. Available entities:
    # 'lovenya' (personal), 'gaspardbtt-', 'lovenya-org'.
    wandb_entity: str = "lovenya-org"
    # Persisted in the run dir so resumed links append to ONE run rather than creating a
    # new disjoint curve per job.
    wandb_mode: str = "offline"

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
    """
    Linear warmup over `warmup_steps`, then a single cosine decay to zero at `total_steps`.

    One curve, not per-epoch cycling (that would be cosine annealing with warm restarts,
    which we deliberately do not use).

    `total_steps` is what makes runs comparable or not. Derived from `epochs` it couples the
    curve to the stopping point, so a run configured for 5 epochs and a run configured for
    100 epochs are at LR factor 0.000 vs 0.997 at the *same* epoch 5 -- the first has fully
    annealed, the second has barely started decaying. Comparing them is meaningless.
    Passing an explicit budget (cfg.lr_total_steps) decouples the two: every run walks the
    same curve and simply stops at a different point along it.
    """

    def fn(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn


def init_wandb(cfg: "TrainConfig", run_dir: Path, world_size: int):
    """
    Start (or rejoin) an offline W&B run.

    Compute nodes have no internet, so this is always offline; `scripts/wandb_sync.sh`
    uploads afterwards. The run id is persisted to the run dir so a resumed job rejoins the
    SAME W&B run -- otherwise a chained or restarted run appears as several disjoint curves
    and the whole point of the plot is lost.
    """
    if not cfg.wandb_enabled or not is_main():
        return None
    try:
        import wandb
    except ImportError:
        log("    W&B not installed; continuing without it "
            "(uv pip install wandb on a login node)")
        return None

    id_file = run_dir / "wandb_run_id.txt"
    resuming = id_file.exists()
    run_id = id_file.read_text().strip() if resuming else None

    os.environ.setdefault("WANDB_MODE", cfg.wandb_mode)
    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity or None,
        id=run_id,
        resume="allow" if resuming else None,
        name=cfg.run_name,
        group=cfg.arch,
        dir=str(run_dir / "wandb"),
        mode=cfg.wandb_mode,
        config={**asdict(cfg), "world_size": world_size},
    )
    if not resuming:
        id_file.write_text(run.id)
    log(f"    W&B {'rejoined' if resuming else 'started'} run {run.id} (mode={cfg.wandb_mode})")
    return run


def timestep_buckets(num_train_timesteps: int, n_buckets: int = 4) -> list[tuple[int, int]]:
    """
    Edges for per-timestep-bucket validation loss.

    The global diffusion loss is flat because it averages over all noise levels: high-noise
    steps are near-unpredictable (loss ~1) and low-noise steps are easy (loss ~0).
    Conditioning mostly matters at lower noise, so bucketing can expose learning that the
    average hides.
    """
    edges = [round(i * num_train_timesteps / n_buckets) for i in range(n_buckets + 1)]
    return list(zip(edges[:-1], edges[1:]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        nargs="*",
        default=[],
        help="YAML files merged left to right; later files win. Layered as "
             "base/ -> arch/ -> exp/ so an experiment only states its differences.",
    )
    ap.add_argument("--override", nargs="*", default=[], help="key=value overrides")
    args = ap.parse_args()

    cfg = TrainConfig()
    for path in args.config:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        for k, v in raw.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            else:
                # Unknown keys are kept rather than rejected so a config can carry notes
                # for downstream scripts, but they are surfaced in config.json.
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

    # runs/<arch>/<run_name>/ with everything for this run underneath, so an experiment is
    # one self-contained directory.
    run_dir = Path(cfg.runs_root) / cfg.arch / cfg.run_name
    ckpt_dir = run_dir / "checkpoints"
    if is_main():
        for d in (ckpt_dir, run_dir / "logs", run_dir / "generations", run_dir / "wandb"):
            d.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    log("=" * 70)
    log(f"S2A Rung 3 -- AudioLDM diffusion recipe | world_size={world_size}")
    log(f"arch:    {cfg.arch}")
    log(f"run dir: {run_dir}")
    log("=" * 70)

    # ------------------------------------------------------------------- data
    train_dir = Path(cfg.features_root) / cfg.direction / cfg.train_split
    val_dir = Path(cfg.features_root) / cfg.direction / cfg.val_split

    # CLAP targets are only needed for validation, so only the val split loads them.
    val_clap_dir = None
    if cfg.clap_features_root:
        candidate = Path(cfg.clap_features_root) / cfg.direction / cfg.val_split
        if (candidate / "clap_target.npy").exists():
            val_clap_dir = candidate
        else:
            log(f"    NOTE: no CLAP targets at {candidate}; val cosine metric disabled.")
            log("          Run scripts/precompute_clap_targets.py to enable it -- without it")
            log("          the flat diffusion loss is the only signal, which is not enough.")

    train_ds = S2ALatentDataset(train_dir, speech_seconds=cfg.speech_seconds)
    val_ds = S2ALatentDataset(
        val_dir, speech_seconds=cfg.speech_seconds, clap_dir=val_clap_dir
    )
    log(f"train: {len(train_ds)} | val: {len(val_ds)} | latent shape {train_ds.latent_shape}")
    log(f"val CLAP cosine metric: {'ON' if val_clap_dir else 'OFF'}")

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
    # U-Net first: its config tells us the conditioning width, and the conditioner is then
    # built to match rather than to a hardcoded 512.
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

    log(f"Building architecture {cfg.arch!r} from the registry ...")
    from salm.architectures import build as build_arch

    components = build_arch(cfg.arch, asdict(cfg), device, output_dim=cond_dim)
    encoder = components.encoder
    mapper = components.conditioner
    log(f"    frozen encoder feature_dim={encoder.feature_dim}")
    log(f"    conditioner trainable params: {mapper.num_trainable_parameters()/1e6:.2f}M")

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
    if cfg.lr_total_steps > 0:
        total_steps = cfg.lr_total_steps
        log(f"LR curve length FIXED at {total_steps} steps "
            f"({total_steps/steps_per_epoch:.1f} epochs' worth), independent of epochs="
            f"{cfg.epochs}. Runs sharing this value are comparable at equal step counts "
            "even if truncated.")
    else:
        total_steps = steps_per_epoch * cfg.epochs
        log(f"LR curve length derived from epochs: {total_steps} steps. NOTE this couples "
            "the curve to the stopping point, so this run is only comparable to others "
            f"with the same epochs={cfg.epochs} that also ran to completion.")
    log(f"warmup {cfg.warmup_steps} steps = epoch "
        f"{cfg.warmup_steps/max(steps_per_epoch,1):.2f}; cosine decay after that")
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

    def diffusion_loss(
        batch,
        train: bool,
        timesteps: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        AudioLDM's objective: MSE between predicted and true noise in latent space.

        `timesteps` may be supplied to evaluate a specific noise level, which is how the
        per-bucket validation losses are computed.

        `cond` may be supplied to reuse an already-computed conditioning vector. That
        matters a lot: the frozen Whisper encoder is 635M params and always processes a
        padded 30 s window, so it dominates runtime. Validation evaluates the main loss,
        the CLAP cosine and four timestep buckets over the same batch, and recomputing the
        condition for each doubled epoch time until this was threaded through.
        """
        mean = batch["latent_mean"].to(device, non_blocking=True)
        std = batch["latent_std"].to(device, non_blocking=True)

        # Sample from the VAE posterior, then apply the scaling factor, as AudioLDM does.
        latents = mean + std * torch.randn_like(std)
        latents = latents * train_ds.vae_scaling_factor

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        if timesteps is None:
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
            ).long()
        noisy = noise_scheduler.add_noise(latents, noise, timesteps)

        if cond is None:
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

    def diffusion_loss_at_timestep(
        batch, timesteps: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        return diffusion_loss(batch, train=False, timesteps=timesteps, cond=cond)

    buckets = timestep_buckets(noise_scheduler.config.num_train_timesteps, 4)

    @torch.no_grad()
    def validate(with_buckets: bool = True) -> dict:
        """
        Validation with metrics that actually move.

        The global diffusion loss is flat by construction, so it is reported but not relied
        on. The informative numbers are:
          - clap_cos: cosine between the conditioner output and the CLAP embedding of the
            TRUE target clip. Directly measures whether the model is learning to point at
            the right scene.
          - loss_t<lo>_<hi>: loss restricted to a timestep band. Conditioning matters most
            at low noise, where the global average drowns it out.
        """
        mapper.eval()
        if not cfg.freeze_unet:
            unet.eval()

        total, count = 0.0, 0
        cos_sum, cos_count = 0.0, 0
        bucket_sum = [0.0] * len(buckets)
        bucket_count = [0] * len(buckets)

        for batch in val_loader:
            n = batch["speech"].shape[0]

            # Compute the conditioning ONCE and reuse it for every metric below. The
            # frozen Whisper encoder dominates runtime, so recomputing it per metric made
            # validation cost 6 encoder passes per batch and doubled epoch time.
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                cond = compute_condition(batch)

                loss = diffusion_loss(batch, train=False, cond=cond)
                total += loss.item() * n
                count += n

                if "clap_target" in batch:
                    tgt = batch["clap_target"].to(device)
                    # Both sides are L2-normalized, so this is a plain dot product.
                    cos = (cond.float() * F.normalize(tgt.float(), dim=-1)).sum(-1)
                    cos_sum += cos.sum().item()
                    cos_count += n

                # Fixed timestep per bucket, so epoch-to-epoch comparisons are not
                # dominated by timestep resampling noise. Skipped on most epochs because
                # each bucket is another full U-Net pass over the val set.
                if with_buckets:
                    for bi, (lo, hi) in enumerate(buckets):
                        t = torch.full((n,), (lo + hi) // 2, device=device, dtype=torch.long)
                        bl = diffusion_loss_at_timestep(batch, t, cond=cond)
                        bucket_sum[bi] += bl.item() * n
                        bucket_count[bi] += n

        mapper.train()
        if not cfg.freeze_unet:
            unet.train()

        out = {"val_loss": total / max(count, 1)}
        if cos_count:
            out["val_clap_cos"] = cos_sum / cos_count
        if with_buckets:
            for bi, (lo, hi) in enumerate(buckets):
                out[f"val_loss_t{lo}_{hi}"] = bucket_sum[bi] / max(bucket_count[bi], 1)
        return out

    def save_checkpoint(
        tag: str, epoch: int, step: int, val_loss: float | None, resumable: bool = True
    ) -> None:
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

        if resumable:
            # Without these a "resume" silently restarts Adam moments and the LR schedule,
            # which is not a resume at all -- it is a warm start with a different optimizer
            # trajectory. Required for a long job to survive a walltime stop.
            payload["optimizer"] = optimizer.state_dict()
            payload["scheduler"] = scheduler.state_dict()
            payload["torch_rng"] = torch.get_rng_state()
            payload["cuda_rng"] = torch.cuda.get_rng_state_all()
        else:
            # Periodic snapshots exist for listening/comparison, not resuming, so they drop
            # optimizer state and go to fp16. Each fp32 resumable checkpoint is ~869 MB and
            # scratch is ~90% full.
            if cfg.snapshot_fp16:
                for key in ("mapper", "unet"):
                    if key in payload:
                        payload[key] = {
                            k: (v.half() if v.is_floating_point() else v)
                            for k, v in payload[key].items()
                        }

        # Write then rename. `last` is rewritten every epoch and the job can be killed by
        # the internal deadline or by Slurm at any moment; saving in place would leave a
        # truncated, unloadable file exactly when the partial checkpoint matters most.
        path = ckpt_dir / f"checkpoint_{tag}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        size_mb = path.stat().st_size / 1e6
        log(f"    saved {path.name} (epoch {epoch}, step {step}, {size_mb:.0f} MB"
            f"{'' if resumable else ', snapshot'})")

    # ----------------------------------------------------------------- resume
    start_epoch = 0
    global_step = 0
    history: list[dict] = []
    best_val = float("inf")

    if cfg.resume_from:
        ckpt_path = Path(cfg.resume_from)
        if not ckpt_path.exists():
            raise SystemExit(f"resume_from does not exist: {ckpt_path}")
        log(f"Resuming from {ckpt_path}")
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        mapper_module.load_state_dict(payload["mapper"])
        if "unet" in payload and not cfg.freeze_unet:
            unet_mod = unet.module if isinstance(unet, DistributedDataParallel) else unet
            unet_mod.load_state_dict(payload["unet"])

        if "optimizer" in payload and "scheduler" in payload:
            optimizer.load_state_dict(payload["optimizer"])
            scheduler.load_state_dict(payload["scheduler"])
            if "torch_rng" in payload:
                torch.set_rng_state(payload["torch_rng"].cpu().to(torch.uint8))
            start_epoch = int(payload["epoch"]) + 1
            global_step = int(payload["step"])
            log(f"    exact resume: continuing at epoch {start_epoch}, step {global_step}")
        else:
            # Be loud. A silent warm start looks like a resume in the logs but produces a
            # different optimization trajectory, and the LR schedule restarts from warmup.
            log("    WARNING: checkpoint has no optimizer/scheduler state.")
            log("             This is a WARM START (weights only), not an exact resume:")
            log("             Adam moments reset and the LR schedule restarts from step 0.")
            log("             Checkpoints written before resume support was added lack it.")

        hist_path = run_dir / "history.json"
        if hist_path.exists():
            history = json.loads(hist_path.read_text())
            vals = [h["val_loss"] for h in history if h.get("val_loss") is not None]
            if vals:
                best_val = min(vals)
            log(f"    loaded {len(history)} prior epochs from history.json "
                f"(best val {best_val:.4f})")

    wandb_run = init_wandb(cfg, run_dir, world_size)

    # ------------------------------------------------------------------ train
    # NOTE: history/global_step/best_val are initialised in the resume block above, so they
    # must NOT be reset here or a resumed run would silently discard its past.
    deadline = time.time() + cfg.max_train_minutes * 60 if cfg.max_train_minutes > 0 else None
    t_start = time.time()
    stop = False
    epoch = start_epoch - 1   # defined even if the loop body never runs

    if start_epoch >= cfg.epochs:
        log(f"nothing to do: start_epoch={start_epoch} >= epochs={cfg.epochs}. "
            "Raise `epochs` to continue training.")

    for epoch in range(start_epoch, cfg.epochs):
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
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/step_loss": loss.item(),
                            "train/lr_mapper": lrs[0],
                            "train/lr_unet": lrs[1] if len(lrs) > 1 else 0.0,
                            "epoch": epoch,
                        },
                        step=global_step,
                    )

            if deadline is not None and time.time() > deadline:
                log(f"  internal deadline reached at step {global_step}; stopping cleanly")
                stop = True
                break

        train_loss = running / max(seen, 1)

        metrics = {}
        val_loss = None
        if (epoch + 1) % cfg.val_every_epochs == 0 or stop:
            want_buckets = (
                cfg.bucket_every_epochs > 0
                and ((epoch + 1) % cfg.bucket_every_epochs == 0 or epoch == start_epoch or stop)
            )
            metrics = validate(with_buckets=want_buckets)
            val_loss = metrics["val_loss"]
            cos = metrics.get("val_clap_cos")
            log(f"  epoch {epoch:02d} train={train_loss:.4f} val={val_loss:.4f}"
                + (f" clap_cos={cos:+.4f}" if cos is not None else "")
                + f" ({(time.time()-t_epoch)/60:.1f} min)")
            log("      buckets: " + "  ".join(
                f"{k.replace('val_loss_t','t')}={v:.4f}"
                for k, v in metrics.items() if k.startswith("val_loss_t")
            ))
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint("best", epoch, global_step, val_loss)
        else:
            log(f"  epoch {epoch:02d} train={train_loss:.4f} ({(time.time()-t_epoch)/60:.1f} min)")

        record = {
            "epoch": epoch,
            "step": global_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_minutes": (time.time() - t_epoch) / 60,
            **{k: v for k, v in metrics.items() if k != "val_loss"},
        }
        history.append(record)
        if is_main():
            (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/epoch_loss": train_loss,
                    **{f"val/{k.replace('val_', '')}": v for k, v in metrics.items()},
                    "epoch": epoch,
                    "epoch_minutes": record["epoch_minutes"],
                    "gpu_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
                },
                step=global_step,
            )

        # Always refresh `last` so a current checkpoint exists on disk at all times.
        # `best` only appears when val improves, so on a plateau it can go stale, and
        # waiting for the loop to end would leave nothing recent if the job is killed.
        # This is what makes "generate from whatever we have right now" reliable.
        save_checkpoint("last", epoch, global_step, val_loss)

        # Periodic immutable snapshots for comparing samples across training time. Not
        # resumable and fp16, because at ~869 MB each the resumable form would fill scratch.
        if (epoch + 1) % cfg.save_every_epochs == 0:
            save_checkpoint(f"ep{epoch:02d}", epoch, global_step, val_loss, resumable=False)

        if stop:
            break

    save_checkpoint("last", epoch, global_step, history[-1]["val_loss"] if history else None)
    log(f"\ndone in {(time.time()-t_start)/60:.1f} min | best val {best_val:.4f}")

    if is_main() and history:
        cos = [h.get("val_clap_cos") for h in history if h.get("val_clap_cos") is not None]
        if len(cos) >= 2:
            log(f"val CLAP cosine: first={cos[0]:+.4f} last={cos[-1]:+.4f} "
                f"best={max(cos):+.4f} (range {max(cos)-min(cos):.4f})")
            if max(cos) - min(cos) < 0.01:
                log("WARNING: cosine is essentially flat across training. Combined with the")
                log("         flat diffusion loss, that indicates the conditioner is NOT")
                log("         learning the speech->scene mapping. More epochs will not help;")
                log("         change the architecture or the conditioning signal instead.")
        else:
            log("val CLAP cosine unavailable (no precomputed CLAP targets) -- the only")
            log("signal was the diffusion loss, which is flat by construction.")

    if wandb_run is not None:
        wandb_run.finish()

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
