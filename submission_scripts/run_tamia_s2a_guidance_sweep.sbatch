#!/usr/bin/env bash
# Guidance-scale sweep, then CLAP scoring. Inference only -- no training.
#
# WHY THIS IS THE HIGHEST-VALUE CHEAP EXPERIMENT RIGHT NOW
# --------------------------------------------------------
# O2's conditioner demonstrably learned: val_clap_r1 0.564 against chance 0.0625 (9x) and val CLAP
# cosine +0.613 against a +/-0.0442 chance band (14x). But its measured conditional-vs-null gap is
# only D2 = 0.0234. Classifier-free guidance computes
#
#     eps = eps(null) + s * (eps(c) - eps(null))
#
# and that bracket IS D2. So at the s=3.0 we have always used, conditioning shifts the prediction by
# roughly 3 * 2.3% = 7% away from unconditional -- which is a plausible reason generation could look
# condition-blind even with a conditioner that works. Reaching ~15% deviation needs s ~ 6.5.
#
# s=3.0 was never tuned; it was my arbitrary pick (diffusers' AudioLDM default is 2.5, AudioGen uses
# 3.0). This sweep decides between the two live hypotheses:
#
#   audio_cos RISES with s  -> the conditioner and interface both work, we were under-guiding.
#   audio_cos FLAT across s -> the U-Net is not responding to its class_labels input, so the
#                              conditioning INTERFACE is the bottleneck and Tier 3 (adding a
#                              cross-attention path) is the next move, not more training.
#
# s=1.0 is included deliberately: that is pure conditional prediction with no guidance at all, the
# true bottom of the curve.
#
# Every lane uses identical --split/--num-samples/--start-index/--seed, so the ONLY difference
# between them is s. Then one scoring pass turns six audio directories into one number each --
# without it this job produces sound and no measurement.
#
# Usage:
#   sbatch submission_scripts/run_tamia_s2a_guidance_sweep.sbatch
#   SCALES="1.0 3.0 5.0 7.5 10.0 15.0" NUM_GEN=24 sbatch ...
#   ARCH=... RUN_NAME=... CKPT_TAG=last sbatch ...
#SBATCH --job-name=salm-s2a-guidance-sweep
#SBATCH --account=aip-csubakan
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=00:50:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

ARCH="${ARCH:-s2a_audioldm_whisper_xformer}"
RUN_NAME="${RUN_NAME:-long100_L2_o2}"
CKPT_TAG="${CKPT_TAG:-best}"
NUM_GEN="${NUM_GEN:-24}"
START_INDEX="${START_INDEX:-0}"
SEED="${SEED:-1234}"
SPLIT="${SPLIT:-val}"
SCALES="${SCALES:-1.0 3.0 5.0 7.5 10.0 15.0}"
STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"

RUN_DIR="$RUNS_ROOT/$ARCH/$RUN_NAME"
CKPT="$RUN_DIR/checkpoints/checkpoint_${CKPT_TAG}.pt"
OUT="$RUN_DIR/guidance_sweep_${STAMP}"
LOG_DIR="$RUN_DIR/logs/guidance_sweep_${STAMP}"
mkdir -p "$OUT" "$LOG_DIR"

[[ -f "$CKPT" ]] || { echo "ERROR: no checkpoint at $CKPT" >&2; exit 1; }

echo "==================================================================="
echo "Guidance-scale sweep + CLAP scoring"
echo "job:    ${SLURM_JOB_ID:-none}   host: $(hostname)"
echo "commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "run:    $ARCH/$RUN_NAME  ckpt=$CKPT_TAG"
echo "scales: $SCALES"
echo "shared: --split $SPLIT --num-samples $NUM_GEN --start-index $START_INDEX --seed $SEED"
echo "==================================================================="

salm_start_monitor "$LOG_DIR" "guidance_sweep" 60

# ---------------------------------------------------------------- generate lanes
# Round-robin across the 4 GPUs, waiting between waves so we never oversubscribe a device.
read -r -a SCALE_ARR <<< "$SCALES"
PIDS=()
GPU=0
T0=$(date +%s)
for s in "${SCALE_ARR[@]}"; do
    tag="s${s}"
    echo "  lane $tag on GPU $GPU"
    ( CUDA_VISIBLE_DEVICES=$GPU python -m salm.generate \
        --mode s2a --checkpoint "$CKPT" \
        --split "$SPLIT" --num-samples "$NUM_GEN" --start-index "$START_INDEX" --seed "$SEED" \
        --guidance-scale "$s" \
        --out-dir "$OUT/$tag" --copy-reference \
        > "$LOG_DIR/gen_${tag}.log" 2>&1 ) &
    PIDS+=($!)
    GPU=$(( (GPU + 1) % 4 ))
    # Wave barrier: after filling all 4 GPUs, drain before starting the next four.
    if [[ "$GPU" == "0" ]]; then
        for p in "${PIDS[@]}"; do wait "$p" || echo "  WARN: a lane failed" >&2; done
        PIDS=()
    fi
done
for p in ${PIDS[@]+"${PIDS[@]}"}; do wait "$p" || echo "  WARN: a lane failed" >&2; done
GEN_SECONDS=$(( $(date +%s) - T0 ))
echo "generation wall-clock: ${GEN_SECONDS}s"

# ------------------------------------------------------------------ score lanes
# Without this the job produces six directories of audio and zero measurement.
echo
echo "### CLAP scoring ###"
DIRS=()
for s in "${SCALE_ARR[@]}"; do
    [[ -d "$OUT/s${s}" ]] && DIRS+=("$OUT/s${s}")
done
if [[ ${#DIRS[@]} -eq 0 ]]; then
    echo "ERROR: no lane produced output; see $LOG_DIR" >&2
    exit 1
fi
CUDA_VISIBLE_DEVICES=0 python scripts/score_generations_clap.py \
    --dirs "${DIRS[@]}" --split "$SPLIT" --out "$OUT/sweep_scores.json" \
    2>&1 | tee "$LOG_DIR/score.log"

echo
echo "lanes:   ${#DIRS[@]}"
echo "results: $OUT"
echo "scores:  $OUT/sweep_scores.json"
echo "OK"
