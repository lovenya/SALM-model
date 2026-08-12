#!/usr/bin/env bash
# T0 -- direct latent regression, the rung BELOW every diffusion model.
#
#   frozen Whisper -> 9.26 M transformer -> latent -> frozen VAE -> frozen vocoder
#
# Trains AND generates in one job, because generation here is a single forward pass per sample
# rather than a 50-step denoising loop, so it costs seconds rather than minutes.
#
# Deliberately shares configs/base + configs/exp/long100.yaml with the diffusion rungs, so
# `epochs` and `lr_total_steps` match and the comparison against L0/L2 is legitimate.
#
# THE NUMBER THIS JOB EXISTS TO PRODUCE is val_std_ratio (and std_ratio in the generation
# summary). Reference points measured on our own train latents: a single real latent has std
# 1.032, a blend of 2 gives 0.73, a blend of 16 gives 0.41, and AudioLDM's decoder expects ~1.0.
# A ratio near 0.5 here against ~1.0 for diffusion is the measured justification for diffusion.
#
# Usage:
#   sbatch submission_scripts/run_tamia_s2a_t0_regress.sbatch
#   EPOCHS=100 NUM_GEN=32 sbatch ...
#SBATCH --job-name=salm-s2a-t0-regress
#SBATCH --account=aip-ravanelm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus=h100:4
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

ARCH="${ARCH:-s2a_t0_whisper_regress}"
RUN_NAME="${RUN_NAME:-t0_regress}"
EPOCHS="${EPOCHS:-100}"
TRAIN_MINUTES="${TRAIN_MINUTES:-95}"
NUM_GEN="${NUM_GEN:-16}"

RUN_DIR="$RUNS_ROOT/$ARCH/$RUN_NAME"
LOG_DIR="$RUN_DIR/logs"
GEN_DIR="$RUN_DIR/generations"
mkdir -p "$LOG_DIR" "$GEN_DIR"

echo "==================================================================="
echo "T0 direct latent regression (NO diffusion)"
echo "job:    ${SLURM_JOB_ID:-none}   host: $(hostname)"
echo "commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "run:    $ARCH/$RUN_NAME   epochs=$EPOCHS"
echo "==================================================================="

salm_start_monitor "$LOG_DIR" "$RUN_NAME" 60

# ============================================================ 1. train
echo
echo "### [1/2] Training ###"
torchrun --nproc_per_node=4 --master_port=29543 -m salm.train_regress \
    --config configs/base/s2a_audioldm.yaml "configs/arch/${ARCH}.yaml" \
             configs/exp/long100.yaml \
    --override \
        "run_name=$RUN_NAME" \
        "arch=$ARCH" \
        "epochs=$EPOCHS" \
        "max_train_minutes=$TRAIN_MINUTES" \
    2>&1 | tee "$LOG_DIR/01_train.log"

# ============================================================ 2. generate
BEST="$RUN_DIR/checkpoints/checkpoint_best.pt"
LAST="$RUN_DIR/checkpoints/checkpoint_last.pt"
CKPT="$BEST"; [[ -f "$CKPT" ]] || CKPT="$LAST"
[[ -f "$CKPT" ]] || { echo "ERROR: no checkpoint at $BEST or $LAST" >&2; exit 1; }

echo
echo "### [2/2] Generating from $(basename "$CKPT") ###"
CUDA_VISIBLE_DEVICES=0 python scripts/generate_regress.py \
    --checkpoint "$CKPT" --out-dir "$GEN_DIR/t0_best" \
    --split val --num-samples "$NUM_GEN" --copy-reference \
    2>&1 | tee "$LOG_DIR/02_generate.log"

echo
echo "### Result ###"
python - "$RUN_DIR" <<'PY'
import json, pathlib, sys
run = pathlib.Path(sys.argv[1])
h = run / "history.json"
if h.exists():
    hist = json.loads(h.read_text())
    if hist:
        a, b = hist[0], hist[-1]
        print(f"  epochs run:     {len(hist)}")
        print(f"  val MSE:        {a['val_loss']:.4f} -> {b['val_loss']:.4f}")
        print(f"  pred std:       {a['val_pred_std']:.4f} -> {b['val_pred_std']:.4f}")
        print(f"  target std:     {b['val_target_std']:.4f}")
        print(f"  STD RATIO:      {a['val_std_ratio']:.3f} -> {b['val_std_ratio']:.3f}")
        if "val_latent_r1" in b:
            print(f"  latent r1:      {b['val_latent_r1']:.3f} "
                  f"(chance {b['val_latent_r1_chance']:.3f})")
s = run / "generations" / "t0_best" / "summary.json"
if s.exists():
    d = json.loads(s.read_text())
    print(f"  generation:     {d['num_samples']} samples, "
          f"{d['seconds_per_sample']:.2f} s/sample, 0 denoising steps")
    print(f"  gen std_ratio:  {d['std_ratio']:.3f}")
print()
print("  Reference (measured on train latents): single real 1.032, blend of 2 -> 0.73,")
print("  blend of 16 -> 0.41. AudioLDM's decoder expects ~1.0.")
print("  A ratio well below 1.0 is the variance collapse that motivates diffusion.")
PY

echo "OK"
