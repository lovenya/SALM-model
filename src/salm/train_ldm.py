"""
Rung 3: AudioLDM's training recipe applied to S2A (speech -> ambient audio).

This is AudioLDM's own objective, unchanged: epsilon-prediction MSE in the VAE latent
space, with classifier-free guidance trained by dropping the condition on a fraction of
steps. The only substitution is WHERE the conditioning vector comes from -- instead of a
CLAP text embedding, a trainable conditioner produces it from frozen speech-encoder features.

Frozen: speech encoder, mel VAE, HiFi-GAN vocoder.
Trainable: conditioner + U-Net.

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

import numpy as np
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
from salm.a2s.wer import CTC_BLANK, ctc_encode  # noqa: E402  (O-F only, cost-free otherwise)
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

    # ---- A2S only (direction: a2s). Ignored by S2A runs. ----
    # Where scripts/a2s/precompute_a2s_conditioning.py wrote the CLAP condition frames.
    # A2S runs no encoder in the loop at all; see salm/encoders/precomputed.py for why.
    cond_features_root: str = str(project_paths.OUTPUTS_ROOT / "features_a2s_cond")
    # Which precomputed embedding the O2 alignment loss pulls the conditioner toward:
    # "text"   = CLAP-text of the transcript      (expected winner; already in CLAP space)
    # "speech" = CLAP-audio of the target speech  (the direct S2A mirror; expected to be
    #            near-collinear and therefore useless -- the precompute script measures it)
    # ""       = no alignment target, which also disables the val CLAP metrics.
    # Choose by the pairwise-spread numbers in the cond cache's meta.json, not by taste.
    align_target: str = "text"

    # model
    # `arch` names an entry in salm.architectures and also groups runs on disk as
    # runs/<arch>/<run_name>/, so experiments stay self-describing.
    arch: str = "s2a_audioldm_whisper_xformer"
    audioldm_model_id: str = project_paths.AUDIOLDM_MODEL_ID
    whisper_path: str = str(project_paths.WHISPER_LARGE_V3_PATH)
    cond_hidden_dim: int = 1024
    cond_layers: int = 2
    cond_dropout: float = 0.1
    cond_num_queries: int = 8    # qformer only

    # T0 latent regressor (salm.train_regress only; ignored by the diffusion trainer). One
    # learned query per output latent frame, each cross-attending to the encoder frames.
    regress_d_model: int = 512
    regress_n_heads: int = 8
    regress_layers: int = 2
    regress_dropout: float = 0.1

    # optimization
    batch_size: int = 16          # per GPU
    num_workers: int = 6
    epochs: int = 20
    lr_conditioner: float = 1e-4
    lr_unet: float = 1e-5         # much lower: the U-Net is pretrained, the conditioner is not
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
    freeze_unet: bool = False     # True reduces this to a conditioner-only run
    seed: int = 1337

    # ---- O2: auxiliary alignment loss (conditioner output -> CLAP of the true target) ----
    #
    # L = L_DDPM + lambda_align * (1 - cos(cond, clap_target))
    #
    # Motivation: the conditioner's only gradient is through the diffusion loss, which is one
    # random timestep per sample averaged over a difficulty gradient that dominates the number.
    # At 1.1 pairs per unique scene (measured) that signal is very likely too weak. Latent CLAP
    # Loss (arXiv 2403.12182) is the published precedent: FAD 8.74 -> 7.10 on nearly our setup.
    #
    # WHY NOT lambda=2000, the published value: that paper's term is a squared error on
    # UNNORMALIZED 512-d embeddings, damped by w(n)=10^(-n/200) so most timesteps contribute
    # ~0. Our conditioner L2-normalizes its output (conditioners.py) and CLAP targets are
    # already normalized, so our term is bounded [0, 2] and sits at ~1.0 at init against a
    # diffusion loss of ~0.198. lambda=2000 would make the auxiliary term ~10,000x the
    # objective and reduce the diffusion loss to decoration. The number does not transfer.
    #
    # Scale reference at init: diffusion ~0.198, (1 - cos) ~1.0. So lambda_align=0.2 makes the
    # two terms comparable, and 1.0 makes alignment ~5x the diffusion loss (aggressive, but
    # defensible when the diffusion signal is the thing we suspect is too weak). Sweep it.
    #
    # 0.0 = off, which keeps every existing config bit-identical in behaviour.
    lambda_align: float = 0.0
    # ---- 2026-08-24: which O2 loss to use. Motivated by a measured gap on A2S -- a
    # closed-form ridge regression on the SAME (input, target) pairs beat the trained
    # conditioner's own val_clap_r1 (0.45 vs 0.35 at batch-16), despite using less input
    # information (mean-pooled, not the full sequence). The plain cosine term has no
    # repulsion: it only pulls cond_i toward target_i and never pushes it away from
    # target_j!=i, so nothing stops every row's prediction drifting toward a shared
    # centroid while still lowering the loss. InfoNCE adds that repulsion directly, against
    # the SAME quantity the project already reports (retrieval accuracy) rather than a proxy
    # for it.
    #   "cosine"  -- 1 - cos(cond_i, target_i), the original O2 term. Unchanged default, so
    #                every existing config and checkpoint reproduces exactly as before.
    #   "infonce" -- symmetric batch contrastive loss (see infonce_loss()). Negatives are the
    #                OTHER rows in the same per-GPU batch, matching val_clap_r1's own
    #                per-batch chance convention. NOT gathered across ranks -- that would
    #                need an all-gather with gradient flow through DDP, a bigger change than
    #                this experiment calls for; deferred.
    align_loss: str = "cosine"
    # InfoNCE temperature. Divides the cosine-similarity logits before the softmax; smaller
    # sharpens the distribution. 0.07 is the standard CLIP/SimCLR starting point, unrelated
    # to anything in this codebase -- worth sweeping if this is adopted rather than trusted.
    align_temp: float = 0.07
    # Cosine ramp for lambda_align over this many optimizer steps (0 = constant from step 0).
    # "REPA Works Until It Doesn't" (arXiv 2505.16792) argues for decaying an alignment term
    # rather than holding it; this is the knob for that experiment, unused by default.
    align_warmup_steps: int = 0

    # ---- O-F, 2026-08-24 (a2s only): content loss on the DECODED mel, not an embedding.
    #
    # Motivated by the oracle-conditioning ablation (job 426005): feeding the row's TRUE
    # CLAP-text embedding straight to class_labels -- removing 100% of conditioner error --
    # still produced incomprehensible speech, by ear. That means the bottleneck is CLAP-
    # text's resolution on word content, not the conditioner's ability to reach it, and no
    # amount of further O2 tuning (InfoNCE, more epochs, bigger conditioner) can fix it,
    # because the PERFECT version of that target already fails to. O-F sidesteps the
    # embedding-target question entirely: reconstruct x0-hat at low-noise timesteps (where
    # the estimate is trustworthy), decode it through the VAE ONLY (no vocoder needed -- a
    # frozen CTC head can score a mel directly), and score the decoded mel against the real
    # transcript with a small CTC model trained specifically on AudioLDM's own mel frontend
    # (scripts/a2s/train_ctc_probe.py -- no off-the-shelf ASR model matches its 64-mel/16kHz
    # config). Additive to O2, not a replacement: they supervise different things (which
    # scene-level embedding to steer toward, vs whether the decoded content matches words).
    #
    # 0.0 = off, bit-identical to every existing config.
    lambda_ctc: float = 0.0
    # Frozen checkpoint from scripts/a2s/train_ctc_probe.py (checkpoint_best.pt). Required
    # when lambda_ctc > 0; there is no sensible default path since it names a specific run.
    ctc_checkpoint: str = ""
    # Only apply the CTC term to rows whose (already-drawn, for the main diffusion loss)
    # timestep falls under this fraction of num_train_timesteps. x0-hat is only a trustworthy
    # estimate of the clean latent at low noise; at high t it is far too noisy for "does this
    # decode to the right words" to mean anything. Reuses the SAME per-example timesteps the
    # main loss already drew -- no extra U-Net forward pass -- so on a given step this is a
    # PARTIAL-BATCH term (expect ~ctc_max_t_frac of the batch to qualify); it contributes 0
    # on a step where no row happens to draw a low t, which is fine averaged over many steps.
    ctc_max_t_frac: float = 0.25

    # ---- Rank 2, 2026-08-26 (a2s only): genuine SEQUENCE content conditioning, via a new
    # gated cross-attention branch on the pretrained U-Net (diffusers' own
    # IPAdapterAttnProcessor2_0 -- see salm.a2s.content_adapter for the full mechanism and
    # why it, not a hand-rolled module, is used). Motivated directly by AudioLDM 2's own
    # paper: they use pooling rate lambda=1 (512 tokens/10s, ~51.2 tok/s) for TTS
    # specifically, vs lambda=8 (~6.4 tok/s) for general audio, because pooled features lose
    # the detail speech needs. Our O2/O-F pathway delivers ~0 tok/s (one pooled vector).
    #
    # This is a FALSIFICATION TEST, not a production design: the content sequence is the
    # frozen CTC probe's encoder output on the TARGET SPEECH's own mel (oracle content --
    # available during training, never at A2S inference time). The question this answers is
    # narrow and cheap to answer: can the U-Net learn to USE a real content sequence at all,
    # via a genuinely new cross-attention channel, within a handful of epochs? If not, either
    # the wiring is wrong or the hypothesis needs rethinking -- either way, cheap to find out
    # before building a real (non-oracle, inference-available) content source.
    #
    # Reuses ctc_checkpoint (the same probe O-F trains against, if lambda_ctc > 0 too) -- one
    # checkpoint, two independent uses (a frozen scorer for O-F; a frozen encoder here).
    use_content_xattn: bool = False

    # runtime
    amp_dtype: str = "bf16"       # H100: bf16 needs no loss scaling
    log_every: int = 25
    val_every_epochs: int = 1
    # Timestep-bucket losses cost 4 extra U-Net passes over the val set. Measured: with them
    # every epoch, validation was 2.87 of a 4.5 min epoch (64%) while training was only
    # 1.63 min. They change slowly, so every 5th epoch is plenty and buys back most of that.
    bucket_every_epochs: int = 5
    # D1/D2 during validation cost 2 extra U-Net passes per val batch, at one fixed low
    # timestep (conditioning matters most at low noise). Run them on the same cadence as the
    # buckets so "is the model using the speech?" becomes a CURVE instead of a post-hoc number.
    # This is the metric the flat eps-MSE cannot give us. 0 disables.
    diag_every_epochs: int = 5
    # Which validation metric picks `checkpoint_best`. "val_loss" is the historical default and
    # is KNOWN BROKEN for this task -- it selected the minimum of a stationary series, i.e. a
    # random epoch (see the comment at the selection site). "val_clap_r1" is the recommended
    # replacement: it has an unambiguous chance baseline of 1/batch_size. Left at the old
    # default so queued and completed runs remain reproducible; new configs should override it.
    select_metric: str = "val_loss"
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


# Validation metrics where a LARGER value is better. Everything else is treated as
# lower-is-better, which is correct for every loss-shaped quantity we log.
_HIGHER_IS_BETTER = {"val_clap_r1", "val_clap_r5", "val_clap_cos",
                     "val_d1_gap", "val_d2_gap", "val_d2_gap_zero"}


def infonce_loss(pred: torch.Tensor, target: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    Symmetric batch contrastive loss (CLIP/SimCLR-style), as an alternative O2 term.

    Unlike `1 - cos(pred_i, target_i)`, this has an explicit REPULSION term: row i's
    prediction is pushed away from every OTHER row's target in the batch, not just pulled
    toward its own. That directly targets the failure mode plain cosine cannot see --
    every row's output drifting toward a shared centroid while the loss keeps falling,
    because cosine-to-own-target never penalizes looking like row j's target too.

    Negatives are the other B-1 rows of THIS batch only (no cross-rank gather), so the
    effective negative count is `batch_size`, matching val_clap_r1's own chance convention
    (1/batch_size) -- the two are then directly comparable.

    pred, target: (B, D), need not be pre-normalized. Returns a scalar in roughly [0, ~1] at
    random init for typical batch sizes (softmax over near-uniform logits gives
    cross_entropy ~= log(B), and dividing by log(B) below keeps this at the same rough scale
    as the cosine term's ~1.0-at-init, so an existing lambda_align is at least in the right
    neighbourhood if this is swapped in -- NOT a derived equivalence, just close enough to
    start a sweep from rather than guessing blind.
    """
    p = F.normalize(pred.float(), dim=-1)
    t = F.normalize(target.float(), dim=-1)
    logits = p @ t.T / temperature                      # (B, B); logits[i, j] = sim(pred_i, target_j)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_pred_to_target = F.cross_entropy(logits, labels)      # row i should point at target_i
    loss_target_to_pred = F.cross_entropy(logits.T, labels)    # target_i should be found by pred_i
    raw = 0.5 * (loss_pred_to_target + loss_target_to_pred)
    return raw / math.log(logits.shape[0])


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

    # Pre-rename aliases. "mapper" was the old name for the conditioner; old YAML and the
    # configs embedded in existing checkpoints still use these keys.
    LEGACY_KEYS = {
        "lr_mapper": "lr_conditioner",
        "mapper_hidden_dim": "cond_hidden_dim",
        "mapper_layers": "cond_layers",
        "mapper_dropout": "cond_dropout",
        "mapper_num_queries": "cond_num_queries",
        "output_root": "runs_root",
        "encoder": None,          # superseded by `arch`; ignored rather than an error
    }

    cfg = TrainConfig()
    for path in args.config:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        for k, v in raw.items():
            if k in LEGACY_KEYS:
                new_k = LEGACY_KEYS[k]
                if new_k is None:
                    continue
                log(f"    config key {k!r} is deprecated; using {new_k!r}")
                k = new_k
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

    # CLAP targets: val needs them for the cosine/retrieval metrics, and TRAIN needs them too
    # once lambda_align > 0 (O2), where they are the auxiliary loss target. They were previously
    # loaded for val only, which would have made the O2 guard below fire unconditionally.
    def _clap_dir(split: str, required: bool) -> Path | None:
        if not cfg.clap_features_root:
            return None
        candidate = Path(cfg.clap_features_root) / cfg.direction / split
        if (candidate / "clap_target.npy").exists():
            return candidate
        msg = f"no CLAP targets at {candidate}"
        if required:
            raise SystemExit(
                f"lambda_align={cfg.lambda_align} needs CLAP targets for the '{split}' split "
                f"but {msg}.\n"
                f"  Precompute them:  python scripts/precompute_clap_targets.py "
                f"--direction {cfg.direction} --split {split}\n"
                f"  Or set lambda_align: 0.0 to train the plain diffusion objective."
            )
        log(f"    NOTE: {msg}; val cosine/retrieval metrics disabled.")
        log("          Run scripts/precompute_clap_targets.py to enable them -- without them")
        log("          the flat diffusion loss is the only signal, which is not enough.")
        return None

    # A2S reads a different cache shape: the condition is precomputed CLAP frames rather
    # than a waveform, and the alignment target is chosen by name. Everything downstream --
    # the loss, CFG, the val metrics, checkpointing -- is identical, which is why this is a
    # branch here rather than a second trainer.
    is_a2s = cfg.direction == "a2s"
    collate_fn = collate
    cond_feature_dim = 0

    if is_a2s:
        from salm.a2s.dataset import A2SLatentDataset, a2s_collate

        collate_fn = a2s_collate
        cond_train = Path(cfg.cond_features_root) / cfg.train_split
        cond_val = Path(cfg.cond_features_root) / cfg.val_split
        align = cfg.align_target or None
        if cfg.lambda_align > 0.0 and not align:
            raise SystemExit(
                f"lambda_align={cfg.lambda_align} needs an alignment target, but "
                "align_target is empty. Set align_target: text (or speech), or "
                "lambda_align: 0.0 for the plain diffusion objective."
            )
        train_ds = A2SLatentDataset(train_dir, cond_train, align_target=align)
        val_ds = A2SLatentDataset(val_dir, cond_val, align_target=align)
        # The select_metric guard further down reads this to decide whether a val_clap*
        # metric can ever be computed. A2S keeps its alignment embeddings in the condition
        # cache rather than a separate CLAP dir, so point it there.
        val_clap_dir = cond_val if val_ds.align_target is not None else None
        cond_feature_dim = train_ds.cond_feature_dim
        if val_ds.cond_feature_dim != cond_feature_dim:
            raise SystemExit(
                f"train/val condition caches disagree on width "
                f"({cond_feature_dim} vs {val_ds.cond_feature_dim})."
            )
        log(f"train: {len(train_ds)} | val: {len(val_ds)} | "
            f"latent shape {train_ds.latent_shape}")
        log(f"A2S condition: {train_ds.cond_frames} frames x {cond_feature_dim} "
            f"(precomputed, no encoder in the loop)")
        log(f"alignment target: {train_ds.align_name or 'OFF'}  "
            f"(lambda_align={cfg.lambda_align})")
    else:
        val_clap_dir = _clap_dir(cfg.val_split, required=False)
        train_clap_dir = _clap_dir(cfg.train_split, required=cfg.lambda_align > 0.0)

        train_ds = S2ALatentDataset(
            train_dir, speech_seconds=cfg.speech_seconds, clap_dir=train_clap_dir
        )
        val_ds = S2ALatentDataset(
            val_dir, speech_seconds=cfg.speech_seconds, clap_dir=val_clap_dir
        )
        log(f"train: {len(train_ds)} | val: {len(val_ds)} | "
            f"latent shape {train_ds.latent_shape}")
        log(f"val CLAP metrics (cos + retrieval r1/r5): {'ON' if val_clap_dir else 'OFF'}")
        log(f"train CLAP targets (for O2 alignment):    {'ON' if train_clap_dir else 'OFF'}")

    train_sampler = DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    # Shard validation across ranks. Without a sampler every rank iterated the FULL val set
    # (118 batches each instead of 30) and only rank 0's numbers were kept -- three quarters
    # of the work discarded. With sharding each rank scores a disjoint slice and the totals
    # are all_reduce'd, so the metric is identical but ~4x cheaper.
    #
    # drop_last=False so no val row is silently skipped; sums are weighted by sample count,
    # which keeps the mean exact even with unequal final batches.
    val_sampler = (
        DistributedSampler(val_ds, shuffle=False, drop_last=False) if world_size > 1 else None
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=max(2, cfg.num_workers // 2),
        collate_fn=collate_fn,
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

    # O-F: the VAE decoder (decode only -- never trained) and the frozen CTC content-loss
    # head. Loaded here, not at import time, so every run without lambda_ctc pays zero cost
    # -- no extra weights in memory, no extra startup time.
    vae = ctc_head = None
    if cfg.lambda_ctc > 0.0:
        # O-F is wired only into the O2 training branch (it needs the x0_hat/timesteps
        # extras that branch already requests), and it is designed as additive to O2, not a
        # replacement -- so requiring lambda_align > 0 here is the actual design, not a
        # shortcut. A config with lambda_ctc > 0 and lambda_align == 0 would otherwise route
        # into the plain-diffusion branch and silently never apply the CTC term at all.
        if cfg.lambda_align <= 0.0:
            raise SystemExit(
                "lambda_ctc > 0 requires lambda_align > 0 too -- O-F is additive to O2, "
                "wired into the same training branch. Set lambda_align (e.g. 1.0, matching "
                "o2_align.yaml) alongside it."
            )
        if not cfg.ctc_checkpoint:
            raise SystemExit(
                "lambda_ctc > 0 but ctc_checkpoint is empty. Point it at a checkpoint from "
                "scripts/a2s/train_ctc_probe.py (checkpoint_best.pt), or set lambda_ctc: 0.0."
            )
        ctc_ckpt_path = Path(cfg.ctc_checkpoint)
        if not ctc_ckpt_path.exists():
            raise SystemExit(f"ctc_checkpoint does not exist: {ctc_ckpt_path}")

        log(f"Loading VAE decoder for O-F (lambda_ctc={cfg.lambda_ctc}) ...")
        from diffusers import AutoencoderKL

        vae = AutoencoderKL.from_pretrained(
            cfg.audioldm_model_id, subfolder="vae", torch_dtype=torch.float32
        ).to(device).eval()
        for p in vae.parameters():
            p.requires_grad_(False)

        log(f"Loading frozen CTC head from {ctc_ckpt_path} ...")
        from salm.a2s.ctc_probe import SimpleCTCASR, CTC_FRAMES_OUT

        ctc_head = SimpleCTCASR().to(device).eval()
        ctc_payload = torch.load(ctc_ckpt_path, map_location=device, weights_only=False)
        ctc_head.load_state_dict(ctc_payload["model"])
        for p in ctc_head.parameters():
            p.requires_grad_(False)
        # cuDNN's fused RNN kernel only allocates the reserve-space its OWN backward pass
        # needs when the module is in train() mode -- eval() skips it entirely, so
        # backward() through a frozen-but-eval GRU fails outright with "cudnn RNN backward
        # can only be called in training mode", even though nothing about it gets updated
        # (requires_grad_(False) above already guarantees that). BatchNorm2d in `conv` stays
        # on eval() (frozen running stats, the actual reason for freezing at all); only the
        # GRU submodule needs train() for cuDNN's backward path, which as a side effect
        # re-enables its inter-layer dropout (0.1, between 2 layers) -- an acceptable, minor
        # source of stochasticity in an already-converged, frozen scorer, not a correctness
        # issue, since none of its weights ever move.
        ctc_head.rnn.train()
        log(f"    CTC probe: epoch={ctc_payload.get('epoch')} "
            f"best_wer={ctc_payload.get('best_wer')}")
        ctc_loss_fn = torch.nn.CTCLoss(blank=CTC_BLANK, zero_infinity=True)

    # Rank 2: content cross-attention. Independent of O-F (lambda_ctc) -- either can be on
    # without the other -- but shares the SAME ctc_head/checkpoint if lambda_ctc already
    # loaded one, rather than loading the model twice.
    content_extractor = None
    if cfg.use_content_xattn:
        if cfg.direction != "a2s":
            raise SystemExit("use_content_xattn is A2S-only (the oracle content source is "
                              "the TARGET SPEECH, which only exists for a2s).")
        if not cfg.ctc_checkpoint:
            raise SystemExit(
                "use_content_xattn requires ctc_checkpoint (same checkpoint O-F uses, or "
                "any scripts/a2s/train_ctc_probe.py checkpoint) -- it is the frozen encoder "
                "that produces the oracle content sequence."
            )
        ctc_ckpt_path = Path(cfg.ctc_checkpoint)
        if not ctc_ckpt_path.exists():
            raise SystemExit(f"ctc_checkpoint does not exist: {ctc_ckpt_path}")

        if ctc_head is None:
            log(f"Loading frozen CTC head from {ctc_ckpt_path} (content-xattn source) ...")
            from salm.a2s.ctc_probe import SimpleCTCASR

            ctc_head = SimpleCTCASR().to(device).eval()
            ctc_payload = torch.load(ctc_ckpt_path, map_location=device, weights_only=False)
            ctc_head.load_state_dict(ctc_payload["model"])
            for p in ctc_head.parameters():
                p.requires_grad_(False)
            ctc_head.rnn.train()   # see the O-F block above for why (cuDNN backward quirk)
            log(f"    CTC probe: epoch={ctc_payload.get('epoch')} "
                f"best_wer={ctc_payload.get('best_wer')}")

        from salm.a2s.content_adapter import install_content_cross_attention
        from salm.data.frontend import MelConfig, MelExtractor

        content_extractor = MelExtractor(MelConfig(norm_mode="none"), device=device)
        adapted = install_content_cross_attention(unet, content_dim=ctc_head.encoder_dim)
        # install_content_cross_attention constructs the new IPAdapterAttnProcessor2_0
        # layers on CPU (no device kwarg) and attaches them via set_attn_processor -- unet
        # itself was already .to(device)'d at load time, so without this the new attn2
        # processors are stranded on CPU while the rest of the U-Net is on CUDA, and DDP
        # wrapping fails with "input module parameters locate in {'cuda', 'cpu'}" (confirmed
        # by job 429416, all 4 ranks, ~22s in -- before any real training step).
        unet = unet.to(device)
        log(f"    installed content cross-attention on {len(adapted)} attn2 layers "
            f"(content_dim={ctc_head.encoder_dim}): {adapted}")

    log(f"Building architecture {cfg.arch!r} from the registry ...")
    from salm.architectures import build as build_arch

    # cond_feature_dim comes from the cache's own shape, so the conditioner's input width
    # can never drift away from what is actually on disk. Zero for S2A, which reads it from
    # the live encoder instead.
    arch_cfg = asdict(cfg)
    arch_cfg["cond_feature_dim"] = cond_feature_dim
    if is_a2s:
        arch_cfg["cond_features_dir"] = str(Path(cfg.cond_features_root) / cfg.train_split)
    components = build_arch(cfg.arch, arch_cfg, device, output_dim=cond_dim)
    encoder = components.encoder
    conditioner = components.conditioner
    log(f"    frozen encoder feature_dim={encoder.feature_dim}")
    log(f"    conditioner trainable params: {conditioner.num_trainable_parameters()/1e6:.2f}M")

    if cfg.freeze_unet:
        unet.eval()
        for p in unet.parameters():
            p.requires_grad_(False)
        log("    U-Net FROZEN (conditioner-only run)")
    else:
        unet.train()
        unet.enable_gradient_checkpointing()
        log(f"    U-Net trainable params: {sum(p.numel() for p in unet.parameters())/1e6:.1f}M")

    if world_size > 1:
        conditioner = DistributedDataParallel(conditioner, device_ids=[local_rank])
        if not cfg.freeze_unet:
            unet = DistributedDataParallel(unet, device_ids=[local_rank])

    conditioner_module = conditioner.module if isinstance(conditioner, DistributedDataParallel) else conditioner

    # -------------------------------------------------------------- optimizer
    param_groups = [
        {"params": [p for p in conditioner.parameters() if p.requires_grad], "lr": cfg.lr_conditioner}
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
        """Frozen features -> conditioner -> (B, cond_dim) L2-normalized CLAP-space vector.

        A2S reads the frozen encoder's output from the batch, because it was precomputed;
        S2A runs the encoder here. Either way the conditioner sees the same (B, T, D) + mask
        contract, so nothing downstream can tell the difference.
        """
        if "cond_features" in batch:
            feats = batch["cond_features"].to(device, non_blocking=True)
            mask = batch["cond_mask"].to(device, non_blocking=True)
        else:
            with torch.no_grad():
                feats, mask = encoder.encode_with_mask(
                    batch["speech"].to(device), batch["speech_length"]
                )
        return conditioner(feats, mask=mask)

    def diffusion_loss(
        batch,
        train: bool,
        timesteps: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
        return_extras: bool = False,
    ):
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
            cond = conditioner_module.apply_cfg_dropout(cond, cfg.cfg_dropout_prob)

        # AudioLDM conditions via class_labels (a single global vector); Rank 2
        # (use_content_xattn) ADDS a genuine sequence channel through a new cross-attention
        # branch (see salm.a2s.content_adapter) without touching this pooled pathway at all
        # -- both are active simultaneously when the flag is on.
        content_kwargs = {}
        if cfg.use_content_xattn:
            from salm.a2s.content_adapter import content_encoder_hidden_states
            from salm.data.frontend import AUDIOLDM_CLIP_SECONDS, load_waveform_mono_16k

            # Oracle content source: the TARGET SPEECH's own mel, run through the frozen CTC
            # probe's encoder. On the fly (no precompute) -- this is a falsification test,
            # not a production path, and never available at real A2S inference time anyway.
            with torch.no_grad():
                wavs = [
                    load_waveform_mono_16k(
                        f, target_seconds=AUDIOLDM_CLIP_SECONDS, align="left"
                    )
                    for f in batch["speech_file"]
                ]
                wav_t = torch.from_numpy(np.stack(wavs)).float().to(device)
                target_mel = content_extractor.batch(wav_t)
                content_seq = ctc_head.encode(target_mel)   # (B, 256, encoder_dim)
            content_kwargs["encoder_hidden_states"] = content_encoder_hidden_states(content_seq)
        else:
            content_kwargs["encoder_hidden_states"] = None

        pred = unet(
            noisy,
            timesteps,
            class_labels=cond,
            **content_kwargs,
        ).sample

        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"unsupported prediction_type: {noise_scheduler.config.prediction_type}")

        loss = F.mse_loss(pred.float(), target.float())
        if not return_extras:
            return loss

        # ------------------------------------------------------------------ x0-space error
        # WHY: eps-MSE is flat by construction. Measured bucket losses 0.559/0.169/0.024/0.001
        # at t=125/375/625/875 track the schedule's SNR (1.84/0.61/0.19/0.036) almost perfectly
        # inversely -- the number is dominated by "how much noise did we add", not by model
        # quality. Recovering the implied clean latent x0-hat and scoring THAT undoes the
        # scaling, so the same model error is comparable across timesteps and the metric can
        # actually move. Free: pure arithmetic on tensors we already have.
        #
        #   x_t = sqrt(abar) * x0 + sqrt(1-abar) * eps
        #   =>  x0_hat = (x_t - sqrt(1-abar) * eps_hat) / sqrt(abar)
        extras = {}
        ac = noise_scheduler.alphas_cumprod.to(device=latents.device, dtype=torch.float32)
        ab = ac[timesteps].view(-1, *([1] * (latents.dim() - 1)))
        eps_hat = pred.float() if noise_scheduler.config.prediction_type == "epsilon" else None
        if eps_hat is not None:
            # NOT wrapped in torch.no_grad(): O-F (lambda_ctc > 0) needs x0_hat to carry
            # gradient back through `pred` into the U-Net/conditioner. This changes nothing
            # for validation, which already runs this whole function inside the caller's
            # @torch.no_grad() decorator, and nothing for the plain diagnostic use below,
            # whose own tensors are explicitly detached before being stored.
            x0_hat = (noisy.float() - (1 - ab).sqrt() * eps_hat) / ab.sqrt().clamp(min=1e-8)
            extras["x0_hat"] = x0_hat
            extras["x0_hat_timesteps"] = timesteps
            with torch.no_grad():
                # Relative, so it is comparable across timesteps and across runs whose latent
                # scale differs. Clamped because at t->T, x0_hat explodes and the raw number
                # would be meaningless rather than informative.
                # CAVEAT, measured: for a FIXED model error this metric GROWS with t (0.63 at
                # t=125 rising to 1587 at t=875), because dividing by sqrt(abar) amplifies the
                # error as abar -> 0. That is the exact mirror of eps-MSE's problem rather than
                # a cure for it: averaged over random t, x0 error is dominated by high-noise
                # samples just as eps-MSE is dominated by low-noise ones. Clamped so it cannot
                # explode, but read it as a complementary view, NOT as a fixed loss. The metrics
                # to actually trust here are val_clap_r1 (real chance baseline) and D1/D2.
                num = (x0_hat.detach() - latents.float()).flatten(1).pow(2).sum(1)
                den = latents.float().flatten(1).pow(2).sum(1).clamp(min=1e-8)
                extras["x0_rel_mse"] = (num / den).clamp(max=1e4).mean()
        return loss, extras

    def diffusion_loss_at_timestep(
        batch, timesteps: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        return diffusion_loss(batch, train=False, timesteps=timesteps, cond=cond)

    @torch.no_grad()
    def diagnose(batch, timesteps: torch.Tensor, cond: torch.Tensor) -> tuple[float, float]:
        """
        D1 and D2 for one batch at one timestep, with the noise held fixed.

        CRITICAL CONTROL: all three variants (correct / shuffled / null) must see the SAME
        latent draw and the SAME noise. Calling the ordinary loss three times would resample
        both, and the differences would be resampling noise rather than the effect of
        conditioning -- which is precisely the mistake that would make this metric useless.
        So the noising is done once here instead of being delegated.

        Returns (d1_gap, d2_gap, d2_gap_zero), all RELATIVE PREDICTION gaps on one scale:
          d1_gap      = ||eps(c) - eps(c_shuffled)|| / ||eps(c)||   sensitivity to WHICH utterance
          d2_gap      = ||eps(c) - eps(null)||      / ||eps(c)||   sensitivity to conditioning
          d2_gap_zero = ||eps(c) - eps(0)||         / ||eps(c)||   same, bypassing the learned null
        """
        n = cond.shape[0]
        mean = batch["latent_mean"].to(device, non_blocking=True)
        std = batch["latent_std"].to(device, non_blocking=True)
        latents = (mean + std * torch.randn_like(std)) * train_ds.vae_scaling_factor
        noise = torch.randn_like(latents)            # ONE draw, shared by all three variants
        noisy = noise_scheduler.add_noise(latents, noise, timesteps)

        # Random n-cycle: a guaranteed derangement, so no sample keeps its own condition.
        # A plain randperm leaves fixed points, which dilutes D1 toward zero. Patching fixed
        # points in place is NOT valid -- it can duplicate indices (a bug caught earlier).
        order = torch.randperm(n, device=device)
        perm = torch.empty_like(order)
        perm[order] = order.roll(-1)

        def eps(c):
            # Deliberately encoder_hidden_states=None even when use_content_xattn is active:
            # D1/D2 isolate sensitivity to the POOLED class_labels conditioning specifically
            # (shuffled/null/zero variants), a different axis from Rank 2's content channel.
            # Threading a real content sequence through here would need four content
            # sequences too and muddy exactly what's being isolated -- out of scope for what
            # this diagnostic answers. Known consequence: D1/D2 numbers do not reflect the
            # content-xattn pathway's contribution when it is on.
            return unet(noisy, timesteps, encoder_hidden_states=None, class_labels=c).sample

        e_c = eps(cond)
        e_s = eps(cond[perm])
        e_n = eps(conditioner_module.null_condition(n, device))
        e_z = eps(torch.zeros_like(cond))

        denom = e_c.float().flatten(1).norm(dim=1).clamp(min=1e-8)
        rel = lambda a, b: ((a - b).float().flatten(1).norm(dim=1) / denom).mean().item()

        # REBUILT 2026-08-12. The original D1 was
        #   loss(shuffled) - loss(correct)
        # which is a difference of two numbers sitting at ~0.19 with sigma 0.006. The deltas it
        # produced were ~0.0002 -- two orders of magnitude below its own resolution -- so it could
        # not distinguish "ignores the condition" from "uses it a little". It reported +0.00010 on a
        # 91-epoch run and +0.00017 on O2, whose conditioner demonstrably reached 9x chance; the
        # metric was blind, not the model.
        #
        # The fix is to compare the PREDICTIONS directly rather than their losses. Swapping in
        # another sample's condition changes eps by a measurable relative amount even when the
        # effect on the averaged loss is invisible, because we are no longer subtracting two large
        # noise-dominated scalars.
        d1 = rel(e_c, e_s)                     # sensitivity to WHICH utterance
        d2 = rel(e_c, e_n)                     # sensitivity to being conditioned at all
        # d2 above uses the LEARNED null embedding, which receives gradient only through the
        # diffusion loss and can drift toward the mean condition -- in which case it reads low even
        # when conditioning works. The zero-vector comparison has no such failure mode, so
        # reporting both makes that confound visible instead of silent.
        d2z = rel(e_c, e_z)
        return d1, d2, d2z

    buckets = timestep_buckets(noise_scheduler.config.num_train_timesteps, 4)

    @torch.no_grad()
    def validate(with_buckets: bool = True, with_diag: bool = False) -> dict:
        """
        Validation with metrics that actually move.

        The global diffusion loss is flat by construction, so it is reported but not relied
        on. The informative numbers are:
          - clap_r1 / clap_r5: RETRIEVAL accuracy in CLAP space. Given a batch of conditioner
            outputs and the CLAP embeddings of their true targets, is the correct target the
            nearest neighbour? This is the single best cheap metric we have, and it is free
            (pure arithmetic, no extra U-Net pass). Unlike raw cosine it is scale-free and has
            an unambiguous chance baseline of 1/batch_size, so a number above chance is
            evidence rather than something to compare against a hand-computed band. Raw cosine
            sat at -0.02 inside a +/-0.044 chance band for 91 epochs and told us nothing.
          - d1_gap: relative change in the PREDICTION when the conditioning is shuffled within
            the batch, at a fixed low timestep. Rebuilt 2026-08-12: the previous version was a
            difference of two losses sitting at 0.19 +/- 0.006 and its deltas came out ~0.0002,
            two orders of magnitude below its own resolution -- so it read ~0 even for O2, whose
            conditioner demonstrably reached 9x chance. Comparing predictions instead of losses
            removes that floor.
          - d2_gap / d2_gap_zero: same form, against the learned null and against a zero vector.
            Both are reported because the learned null receives gradient only through the
            diffusion loss and can drift toward the mean condition, which would make d2_gap read
            low even when conditioning works. Divergence between the two exposes that confound.
            d2_gap is also exactly what classifier-free guidance amplifies, so it sets the
            guidance scale actually needed: a 2% gap at s=3 is still only ~6%.
          - x0_rel_mse: error in CLEAN-LATENT space rather than noise space, which removes the
            SNR scaling that makes eps-MSE flat.
          - loss_t<lo>_<hi>: loss restricted to a timestep band.

        Deliberately NOT here: target_caption_clap_score and FAD. Both need full sampling
        (50 DDIM steps + VAE decode + vocoder), which would dwarf a 2.16 min epoch. Worse,
        under O2 a CLAP-based score partly measures our own training objective, so it must
        never become the selection criterion. They belong in held-out eval, not in the loop.
        """
        conditioner.eval()
        if not cfg.freeze_unet:
            unet.eval()

        total, count = 0.0, 0
        cos_sum, cos_count = 0.0, 0
        r1_sum, r5_sum, r_count = 0.0, 0.0, 0
        x0_sum, x0_count = 0.0, 0
        d1_sum, d2_sum, d2z_sum, d_count = 0.0, 0.0, 0.0, 0
        bucket_sum = [0.0] * len(buckets)
        bucket_count = [0] * len(buckets)
        # Conditioning matters most where signal survives, so diagnose at a low timestep
        # rather than at the average over all t (where it is drowned out).
        diag_t = (buckets[0][0] + buckets[0][1]) // 2

        for batch in val_loader:
            # Batch size from the target, not the condition: "speech" is the S2A condition
            # key and does not exist in an A2S batch, whereas every direction has a latent.
            n = batch["latent_mean"].shape[0]

            # Compute the conditioning ONCE and reuse it for every metric below. The
            # frozen Whisper encoder dominates runtime, so recomputing it per metric made
            # validation cost 6 encoder passes per batch and doubled epoch time.
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                cond = compute_condition(batch)

                loss, extras = diffusion_loss(
                    batch, train=False, cond=cond, return_extras=True
                )
                total += loss.item() * n
                count += n
                if "x0_rel_mse" in extras:
                    x0_sum += extras["x0_rel_mse"].item() * n
                    x0_count += n

                if "clap_target" in batch:
                    tgt = F.normalize(batch["clap_target"].to(device).float(), dim=-1)
                    # Both sides are L2-normalized, so this is a plain dot product.
                    cos = (cond.float() * tgt).sum(-1)
                    cos_sum += cos.sum().item()
                    cos_count += n

                    # Retrieval within the batch: does cond point at its OWN target more than
                    # at the other targets present? Needs n >= 2 to mean anything.
                    if n >= 2:
                        sim = cond.float() @ tgt.t()              # (n, n)
                        truth = torch.arange(n, device=device)
                        rank = (sim > sim.gather(1, truth[:, None])).sum(1)  # 0 = best
                        r1_sum += (rank == 0).sum().item()
                        r5_sum += (rank < min(5, n)).sum().item()
                        r_count += n

                # ------------------------------------------------- D1 / D2 (extra passes)
                if with_diag and n >= 2:
                    ts = torch.full((n,), diag_t, device=device, dtype=torch.long)
                    d1, d2, d2z = diagnose(batch, ts, cond)
                    d1_sum += d1 * n
                    d2_sum += d2 * n
                    d2z_sum += d2z * n
                    d_count += n

                # Fixed timestep per bucket, so epoch-to-epoch comparisons are not
                # dominated by timestep resampling noise. Skipped on most epochs because
                # each bucket is another full U-Net pass over the val set.
                if with_buckets:
                    for bi, (lo, hi) in enumerate(buckets):
                        t = torch.full((n,), (lo + hi) // 2, device=device, dtype=torch.long)
                        bl = diffusion_loss_at_timestep(batch, t, cond=cond)
                        bucket_sum[bi] += bl.item() * n
                        bucket_count[bi] += n

        conditioner.train()
        if not cfg.freeze_unet:
            unet.train()

        # Each rank scored a disjoint slice, so sum the raw totals and counts across ranks
        # before dividing. Reducing the sums (not the per-rank means) keeps the result exact
        # when ranks get unequal numbers of samples.
        if dist.is_initialized():
            packed = torch.tensor(
                [total, float(count), cos_sum, float(cos_count),
                 r1_sum, r5_sum, float(r_count), x0_sum, float(x0_count),
                 d1_sum, d2_sum, d2z_sum, float(d_count)]
                + bucket_sum + [float(c) for c in bucket_count],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            vals = packed.tolist()
            nb = len(buckets)
            total, count, cos_sum, cos_count = vals[0], int(vals[1]), vals[2], int(vals[3])
            r1_sum, r5_sum, r_count = vals[4], vals[5], int(vals[6])
            x0_sum, x0_count = vals[7], int(vals[8])
            d1_sum, d2_sum, d2z_sum, d_count = vals[9], vals[10], vals[11], int(vals[12])
            bucket_sum = vals[13 : 13 + nb]
            bucket_count = [int(c) for c in vals[13 + nb : 13 + 2 * nb]]

        out = {"val_loss": total / max(count, 1)}
        if cos_count:
            out["val_clap_cos"] = cos_sum / cos_count
        if r_count:
            out["val_clap_r1"] = r1_sum / r_count
            out["val_clap_r5"] = r5_sum / r_count
            # Chance depends on batch size, so record it -- otherwise the number is
            # uninterpretable in isolation and in a W&B chart months from now.
            out["val_clap_r1_chance"] = 1.0 / cfg.batch_size
        if x0_count:
            out["val_x0_rel_mse"] = x0_sum / x0_count
        if d_count:
            # Renamed from val_d1_delta: this is now a relative PREDICTION gap, not a loss delta,
            # so the old name would silently mean something different in the same history.json.
            out["val_d1_gap"] = d1_sum / d_count
            out["val_d2_gap"] = d2_sum / d_count
            out["val_d2_gap_zero"] = d2z_sum / d_count
        if with_buckets:
            for bi, (lo, hi) in enumerate(buckets):
                out[f"val_loss_t{lo}_{hi}"] = bucket_sum[bi] / max(bucket_count[bi], 1)
        out["val_samples"] = count
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
            "mapper": conditioner_module.state_dict(),
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
    best_sel = -float("inf") if cfg.select_metric in _HIGHER_IS_BETTER else float("inf")
    # Separate running best for epochs where select_metric was not computed. See the selection site.
    best_fallback = float("inf")

    # Fail fast rather than 3 hours in. A select_metric that is only produced on diag/bucket epochs
    # makes selection alternate between two incomparable scales; one that needs CLAP targets we do
    # not have means checkpoint_best is never written at all.
    _DIAG_ONLY = {"val_d1_gap", "val_d2_gap", "val_d2_gap_zero"}
    if cfg.select_metric in _DIAG_ONLY or cfg.select_metric.startswith("val_loss_t"):
        raise SystemExit(
            f"select_metric='{cfg.select_metric}' is only computed on diagnostic/bucket epochs, so "
            f"selection would alternate between two different scales. Use val_clap_r1 or val_loss."
        )
    if cfg.select_metric.startswith("val_clap") and val_clap_dir is None:
        fix = (
            f"scripts/a2s/precompute_a2s_conditioning.py --split {cfg.val_split} "
            "(and set align_target)"
            if cfg.direction == "a2s"
            else f"scripts/precompute_clap_targets.py --split {cfg.val_split}"
        )
        raise SystemExit(
            f"select_metric='{cfg.select_metric}' needs CLAP targets for the '{cfg.val_split}' "
            f"split, which are missing, so checkpoint_best would never be written. Precompute "
            f"them ({fix}) or set select_metric: val_loss."
        )

    if cfg.resume_from:
        ckpt_path = Path(cfg.resume_from)
        if not ckpt_path.exists():
            raise SystemExit(f"resume_from does not exist: {ckpt_path}")
        log(f"Resuming from {ckpt_path}")
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        conditioner_module.load_state_dict(payload["mapper"])
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
            # Restore the SELECTION metric's best too. Without this a resumed run starts from
            # +/-inf and overwrites checkpoint_best on its very first validation, silently
            # discarding a genuinely better earlier epoch.
            sels = [h[cfg.select_metric] for h in history
                    if h.get(cfg.select_metric) is not None]
            if sels:
                best_sel = (max(sels) if cfg.select_metric in _HIGHER_IS_BETTER else min(sels))
                log(f"    restored best {cfg.select_metric}={best_sel:.5f} from history")
            log(f"    loaded {len(history)} prior epochs from history.json "
                f"(best val {best_val:.4f})")

            # W&B silently DISCARDS any row whose step is not greater than the maximum step it has
            # already recorded for this run id. Since we log with an explicit step=global_step and
            # a resumed job rejoins the same run id, resuming from a checkpoint that is BEHIND the
            # furthest-logged step means every metric until we catch up vanishes with no error.
            # `checkpoint_last` is always the furthest, so it is safe; `checkpoint_best` may not be.
            max_logged = max((h.get("step", 0) for h in history), default=0)
            if global_step < max_logged:
                log(f"    WARNING: resuming at step {global_step} but history reaches "
                    f"{max_logged}.")
                log(f"             W&B will DROP logged metrics until step {max_logged} is passed "
                    "(it rejects non-increasing steps for a resumed run id).")
                log("             history.json will still be correct. To avoid the gap, resume from "
                    "checkpoint_last, or delete wandb_run_id.txt to start a fresh W&B run.")

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

    # O2. Guarded so a config that asks for alignment but has no CLAP targets precomputed fails
    # loudly here instead of silently training the plain objective for hours and looking fine.
    lam_align = float(cfg.lambda_align)
    if cfg.align_loss not in ("cosine", "infonce"):
        raise SystemExit(f"align_loss must be 'cosine' or 'infonce', got {cfg.align_loss!r}")
    if lam_align > 0.0:
        if train_ds.clap_target is None:
            script = (
                "scripts/a2s/precompute_a2s_conditioning.py"
                if cfg.direction == "a2s"
                else "scripts/precompute_clap_targets.py"
            )
            raise SystemExit(
                "lambda_align > 0 but the training set has no alignment target. "
                f"Run {script} for the train split, or set lambda_align: 0.0."
            )
        log(f"O2 alignment loss ACTIVE: lambda={lam_align} on {cfg.align_loss!r} "
            f"(temp={cfg.align_temp} if infonce), warmup={cfg.align_warmup_steps} steps")
        log(f"    scale reference: diffusion loss ~0.198, align term ~1.0 at init "
            f"(cosine: 1-cos bounded [0,2]; infonce: cross-entropy / log(batch_size)), so "
            f"the alignment term starts ~{lam_align/0.198:.1f}x the diffusion term")
    last_align, last_dm = None, None

    for epoch in range(start_epoch, cfg.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        running, seen = 0.0, 0
        ctc_running, ctc_seen = 0.0, 0   # true epoch mean, unlike train_align's last-batch snapshot
        t_epoch = time.time()

        for batch in train_loader:
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                if lam_align > 0.0 and "clap_target" in batch:
                    # Compute the condition ONCE and share it. The alignment term must see the
                    # PRE-dropout condition: CFG dropout replaces some rows with the null
                    # embedding, and pulling a deliberately-nulled row toward a real target
                    # would train the null embedding to encode that sample's scene, which
                    # corrupts both the unconditional branch and CFG at inference.
                    cond_raw = compute_condition(batch)
                    # O-F needs x0_hat + the per-row timesteps diffusion_loss already drew,
                    # so extras are requested here rather than via a second forward pass.
                    if cfg.lambda_ctc > 0.0:
                        loss_dm, dm_extras = diffusion_loss(
                            batch, train=True, cond=cond_raw, return_extras=True
                        )
                    else:
                        loss_dm, dm_extras = diffusion_loss(batch, train=True, cond=cond_raw), None
                    tgt_raw = batch["clap_target"].to(device).float()
                    if cfg.align_loss == "infonce":
                        if cond_raw.shape[0] < 4:
                            raise SystemExit(
                                f"align_loss='infonce' needs a real batch of negatives; "
                                f"got batch_size={cond_raw.shape[0]}. Raise batch_size or "
                                f"use align_loss='cosine'."
                            )
                        align = infonce_loss(cond_raw, tgt_raw, temperature=cfg.align_temp)
                    else:
                        tgt = F.normalize(tgt_raw, dim=-1)
                        # Both sides are unit vectors, so this is 1 - cosine, bounded [0, 2].
                        align = (1.0 - (cond_raw.float() * tgt).sum(-1)).mean()
                    lam = lam_align
                    if cfg.align_warmup_steps > 0:
                        lam = lam_align * min(1.0, global_step / cfg.align_warmup_steps)
                    loss = loss_dm + lam * align
                    last_align = align.item()
                    last_dm = loss_dm.item()

                    # ---- O-F: content loss on the decoded mel, additive to the O2 term
                    # above. See TrainConfig.lambda_ctc for the full motivation. Only rows
                    # whose ALREADY-DRAWN timestep (for the main diffusion loss -- no extra
                    # U-Net forward pass) falls under ctc_max_t_frac get scored this step;
                    # x0-hat is only a trustworthy clean-latent estimate at low noise.
                    if cfg.lambda_ctc > 0.0 and dm_extras is not None:
                        t_sel = dm_extras["x0_hat_timesteps"]
                        low_thresh = int(
                            cfg.ctc_max_t_frac * noise_scheduler.config.num_train_timesteps
                        )
                        low_idx = (t_sel < low_thresh).nonzero(as_tuple=True)[0].tolist()
                        # ctc_encode can, in principle, return an empty list (a transcript that
                        # normalizes to nothing under the fixed vocab) -- CTCLoss's targets must
                        # NEVER contain the blank symbol, so a fake length-1 all-blank target
                        # would be invalid input, not a harmless placeholder. Excluded here,
                        # not padded around.
                        ids_low = [ctc_encode(batch["speech_text"][i]) for i in low_idx]
                        sel_idx = [i for i, ids in zip(low_idx, ids_low) if len(ids) > 0]
                        ids_list = [
                            torch.tensor(ids, dtype=torch.long) for ids in ids_low if len(ids) > 0
                        ]
                        if sel_idx:
                            x0_sel = dm_extras["x0_hat"][sel_idx].float()
                            mel_hat = vae.decode(
                                x0_sel / train_ds.vae_scaling_factor
                            ).sample.squeeze(1)                          # (Bsel, 1024, 64)
                            ctc_log_probs = ctc_head(mel_hat)             # (Bsel, T_out, vocab)
                            lp = ctc_log_probs.transpose(0, 1).float()    # (T_out, Bsel, vocab)

                            target_lengths = torch.tensor(
                                [len(ids) for ids in ids_list], dtype=torch.long
                            )
                            max_len = int(target_lengths.max().item())
                            ids_padded = torch.zeros(
                                len(ids_list), max_len, dtype=torch.long, device=device
                            )
                            for i, ids in enumerate(ids_list):
                                ids_padded[i, : len(ids)] = ids
                            # CTCLoss requires input_lengths/target_lengths on CPU even when
                            # log_probs/targets are on GPU -- a well-known torch quirk, not
                            # something to "optimize away" by moving them.
                            input_lengths = torch.full(
                                (lp.shape[1],), CTC_FRAMES_OUT, dtype=torch.long
                            )
                            ctc_loss_val = ctc_loss_fn(
                                lp, ids_padded, input_lengths, target_lengths
                            )
                            loss = loss + cfg.lambda_ctc * ctc_loss_val
                            ctc_running += ctc_loss_val.item()
                            ctc_seen += 1
                else:
                    loss = diffusion_loss(batch, train=True)
                    last_align, last_dm = None, loss.item()

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
                            "train/lr_conditioner": lrs[0],
                            "train/lr_unet": lrs[1] if len(lrs) > 1 else 0.0,
                            "epoch": epoch,
                        },
                        step=global_step,
                    )

            # The deadline decision MUST be collective. Evaluated per-rank against each rank's own
            # clock, one rank leaves this loop while the others are still waiting on the gradient
            # all-reduce -- the collective never completes and NCCL's watchdog kills the job 600 s
            # later. That is exactly how job 406301 died after 91 good epochs: deadline 07:38:49,
            # last checkpoint 07:39:00, timeout fired 07:49:32 on an ALLREDUCE of 321288 elements
            # (a DDP gradient bucket, not our metrics reduce).
            #
            # Checked only every `log_every` steps so the extra 1-element collective is negligible.
            # DistributedSampler(drop_last=True) guarantees every rank runs the same number of
            # iterations, so this reduce is always reached by all of them.
            if deadline is not None and global_step % cfg.log_every == 0:
                local_stop = 1.0 if time.time() > deadline else 0.0
                if dist.is_initialized():
                    flag = torch.tensor([local_stop], device=device)
                    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                    local_stop = flag.item()
                if local_stop > 0:
                    log(f"  internal deadline reached at step {global_step}; "
                        "all ranks stopping together")
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
            want_diag = (
                cfg.diag_every_epochs > 0
                and ((epoch + 1) % cfg.diag_every_epochs == 0 or epoch == start_epoch or stop)
            )
            metrics = validate(with_buckets=want_buckets, with_diag=want_diag)
            val_loss = metrics["val_loss"]
            cos = metrics.get("val_clap_cos")
            log(f"  epoch {epoch:02d} train={train_loss:.4f} val={val_loss:.4f}"
                + (f" clap_cos={cos:+.4f}" if cos is not None else "")
                + (f" r1={metrics['val_clap_r1']:.3f}/{metrics['val_clap_r1_chance']:.3f}"
                   if "val_clap_r1" in metrics else "")
                + (f" x0={metrics['val_x0_rel_mse']:.4f}" if "val_x0_rel_mse" in metrics else "")
                + (f" align={last_align:.4f}" if last_align is not None else "")
                + (f" ctc={ctc_running/ctc_seen:.4f}({ctc_seen}b)" if ctc_seen else "")
                + f" ({(time.time()-t_epoch)/60:.1f} min)")
            if "val_d1_gap" in metrics:
                # The numbers that decide whether any of this is working. Both are relative
                # prediction gaps now, so they share a scale and are directly comparable.
                log(f"      D1 gap={metrics['val_d1_gap']:.5f}  "
                    f"D2 gap={metrics['val_d2_gap']:.5f}  "
                    f"D2(zero)={metrics['val_d2_gap_zero']:.5f}")
                log(f"        D1 = sensitivity to WHICH utterance; D2 = to being conditioned at "
                    f"all. D2(zero) bypasses the learned null embedding.")
            if want_buckets:
                log("      buckets: " + "  ".join(
                    f"{k.replace('val_loss_t','t')}={v:.4f}"
                    for k, v in metrics.items() if k.startswith("val_loss_t")
                ))

            # Model selection. Selecting on val_loss is KNOWN BROKEN for this task: over 91
            # epochs val loss was 0.1979 +/- 0.0070 with no trend, and the "best" 0.1833 matched
            # the expected minimum of 91 draws from that distribution (0.1805). It was selecting
            # noise, so `checkpoint_best` was effectively a random epoch. Prefer a metric with a
            # real chance baseline; `select_metric` names it and higher-is-better is inferred.
            # The fallback keeps its OWN running best. Sharing one `best_sel` across two metrics
            # on different scales is silently wrong in both directions:
            #   - select on a gap-valued metric (~0.30) and the fallback compares val_loss (0.198)
            #     against it, "improves" trivially, overwrites checkpoint_best AND resets best_sel
            #     to 0.198 -- after which every later diag epoch trivially "improves" again;
            #   - select on val_clap_r1 with no CLAP targets and best_sel stays -inf forever, so
            #     `val_loss < -inf` is never true and NO checkpoint_best is ever written for the
            #     whole run, with generation silently falling back to checkpoint_last.
            sel_key = cfg.select_metric
            sel = metrics.get(sel_key)
            use_fallback = sel is None
            if use_fallback:
                if sel_key != "val_loss":
                    log(f"      note: '{sel_key}' not computed this epoch; "
                        f"selecting on val_loss for this epoch only")
                sel_key, sel = "val_loss", val_loss
                improved = sel < best_fallback
                if improved:
                    best_fallback = sel
            else:
                higher_better = sel_key in _HIGHER_IS_BETTER
                improved = (sel > best_sel) if higher_better else (sel < best_sel)
                if improved:
                    best_sel = sel
            if improved:
                best_val = min(best_val, val_loss)
                save_checkpoint("best", epoch, global_step, val_loss)
                log(f"      new best by {sel_key}={sel:.5f} -> checkpoint_best")
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
        if last_align is not None:
            # Logged separately from train_loss, which is now their SUM. Without the split it is
            # impossible to tell from a curve whether the diffusion term improved or the
            # alignment term simply dominated the total.
            record["train_align"] = last_align
            record["train_diffusion"] = last_dm
        if cfg.lambda_ctc > 0.0:
            # A TRUE epoch mean (accumulated every qualifying step), unlike train_align's
            # last-batch snapshot -- deliberately not repeating that known defect in new code.
            # None (not 0.0) on an epoch where no batch happened to draw a low-t row, so a
            # flat 0.0 in the curve isn't confused with "the term was never active".
            record["train_ctc_loss"] = (ctc_running / ctc_seen) if ctc_seen else None
            record["ctc_qualifying_batches"] = ctc_seen
        history.append(record)
        if is_main():
            (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/epoch_loss": train_loss,
                    **({"train/align": last_align, "train/diffusion": last_dm}
                       if last_align is not None else {}),
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
