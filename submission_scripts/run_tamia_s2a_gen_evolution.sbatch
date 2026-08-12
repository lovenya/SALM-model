#!/usr/bin/env bash
# Generate audio from EVERY training snapshot of a run, so we can hear how the model evolved.
#
# Also settles two questions at once:
#   - what the U-Net finetuning actually bought: the same snapshot is generated twice, once with
#     the trained U-Net and once with --skip-unet-weights (pristine AudioLDM). Until 2026-08-12
#     generate.py silently discarded the trained U-Net, so every earlier sample was the pristine
#     variant; this A/B makes that measurable instead of just fixed.
#   - where the model sits between the two reference points: rung0 (ground-truth caption -> audio,
#     the ceiling) and rung1 (CLAP-of-speech -> audio, the control expected to be uninformative).
#
# Every lane uses IDENTICAL --split/--num-samples/--start-index/--seed/--batch-size/--steps/
# --guidance-scale, so the outputs are directly comparable sample by sample.
#
# Submit:
#   sbatch submission_scripts/run_tamia_s2a_gen_evolution.sbatch
#   NUM_GEN=32 sbatch ...                 # more samples
#   RUN_NAME=long100 sbatch ...           # a different training run
#
# Resources: whole node is mandatory on TamIA -- partial h100 requests are rejected with
# "The h100 GPUs are only allocated by node." 10 lanes over 4 GPUs is ~3 rounds of ~3 min.
#SBATCH --job-name=salm-s2a-gen-evolution
#SBATCH --account=aip-csubakan
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# Absolute path: sbatch copies this script to the Slurm spool, so $BASH_SOURCE is not the repo.
# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

ARCH="${ARCH:-s2a_audioldm_whisper_xformer}"
RUN_NAME="${RUN_NAME:-long100}"
NUM_GEN="${NUM_GEN:-16}"
STEPS="${STEPS:-100}"
GUIDANCE="${GUIDANCE:-3.0}"
SEED="${SEED:-1337}"
BATCH="${BATCH:-4}"
SPLIT="${SPLIT:-val}"
STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"

RUN_DIR="$RUNS_ROOT/$ARCH/$RUN_NAME"
CKPT_DIR="$RUN_DIR/checkpoints"
GEN_ROOT="$RUN_DIR/generations/evolution_${STAMP}"
LOG_DIR="$RUN_DIR/logs/evolution_${STAMP}"
mkdir -p "$GEN_ROOT" "$LOG_DIR"

[[ -d "$CKPT_DIR" ]] || { echo "ERROR: no checkpoints dir at $CKPT_DIR" >&2; exit 1; }

echo "==================================================================="
echo "S2A generation across training snapshots"
echo "job:    ${SLURM_JOB_ID:-none}   host: $(hostname)"
echo "commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "dirty:  $(if [[ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]]; then echo true; else echo false; fi)"
echo "run:    $ARCH/$RUN_NAME"
echo "shared: --split $SPLIT --num-samples $NUM_GEN --seed $SEED --steps $STEPS --guidance-scale $GUIDANCE"
echo "==================================================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ------------------------------------------------------------------ build lane list
# Each lane: NAME|MODE|CKPT_PATH|EXTRA_ARGS
LANES=()

# Epoch snapshots, numerically sorted. fp16 and non-resumable, but they carry both the
# conditioner and the U-Net, which is all generation needs.
while IFS= read -r ck; do
    [[ -e "$ck" ]] || continue
    tag="$(basename "$ck" .pt)"; tag="${tag#checkpoint_}"
    LANES+=("trained_${tag}|s2a|${ck}|")
done < <(ls -1 "$CKPT_DIR"/checkpoint_ep*.pt 2>/dev/null | sort -V)

for tag in best last; do
    ck="$CKPT_DIR/checkpoint_${tag}.pt"
    [[ -f "$ck" ]] && LANES+=("trained_${tag}|s2a|${ck}|")
done

# Pristine-U-Net A/B on the earliest and latest snapshots only -- enough to size the effect
# without doubling every lane.
FIRST_EP="$(ls -1 "$CKPT_DIR"/checkpoint_ep*.pt 2>/dev/null | sort -V | head -1 || true)"
[[ -n "$FIRST_EP" ]] && LANES+=("pristineunet_$(basename "$FIRST_EP" .pt | sed 's/^checkpoint_//')|s2a|${FIRST_EP}|--skip-unet-weights")
[[ -f "$CKPT_DIR/checkpoint_last.pt" ]] && LANES+=("pristineunet_last|s2a|$CKPT_DIR/checkpoint_last.pt|--skip-unet-weights")

# Reference points, neither needs a checkpoint.
LANES+=("rung0_caption_ceiling|rung0||")
LANES+=("rung1_clap_control|rung1||")

echo
echo "### ${#LANES[@]} lanes ###"
for l in "${LANES[@]}"; do echo "  ${l%%|*}"; done

# ------------------------------------------------------------------ run, 4 GPUs round-robin
run_lane() {
    local spec="$1" gpu="$2"
    local name mode ckpt extra
    IFS='|' read -r name mode ckpt extra <<< "$spec"

    local out="$GEN_ROOT/$name"
    local args=(--mode "$mode" --split "$SPLIT" --num-samples "$NUM_GEN"
                --start-index 0 --seed "$SEED" --batch-size "$BATCH"
                --steps "$STEPS" --guidance-scale "$GUIDANCE" --out-dir "$out")
    [[ -n "$ckpt" ]] && args+=(--checkpoint "$ckpt")
    # Only the s2a lanes copy the reference ambient; rung0/rung1 share the same rows so one copy
    # is enough and scratch is at 91% capacity.
    [[ "$mode" == "s2a" ]] && args+=(--copy-reference)
    # shellcheck disable=SC2086
    [[ -n "$extra" ]] && args+=($extra)

    local t0 t1
    t0=$(date +%s)
    if CUDA_VISIBLE_DEVICES="$gpu" python -m salm.generate "${args[@]}" \
            > "$LOG_DIR/${name}.log" 2>&1; then
        t1=$(date +%s)
        echo "  OK   $name  gpu$gpu  $((t1-t0))s"
    else
        t1=$(date +%s)
        echo "  FAIL $name  gpu$gpu  $((t1-t0))s  (see $LOG_DIR/${name}.log)" >&2
        return 1
    fi
}
export -f run_lane

worker() {
    local gpu="$1" rc=0
    local i=$gpu
    while [[ $i -lt ${#LANES[@]} ]]; do
        run_lane "${LANES[$i]}" "$gpu" || rc=1
        i=$(( i + 4 ))
    done
    return $rc
}

echo
echo "### generating ###"
JOB_T0=$(date +%s)
PIDS=()
for gpu in 0 1 2 3; do
    worker "$gpu" &
    PIDS+=("$!")
done
FAIL=0
for p in "${PIDS[@]}"; do wait "$p" || FAIL=1; done
JOB_T1=$(date +%s)
TOTAL=$(( JOB_T1 - JOB_T0 ))

echo
echo "total generation wall-clock: ${TOTAL}s ($(python -c "print(f'{$TOTAL/60:.1f}')") min)"

# ------------------------------------------------------------------ package + report
echo
echo "### packaging for listening ###"
python scripts/package_generation_evolution.py --gen-root "$GEN_ROOT" \
    ${COPY:+--copy} 2>&1 | tee "$LOG_DIR/package.log"

echo
echo "### report ###"
python scripts/report_generation_evolution.py --gen-root "$GEN_ROOT" \
    --total-seconds "$TOTAL" 2>&1 | tee "$LOG_DIR/report.log"

echo
echo "generations: $GEN_ROOT"
echo "listen here: $GEN_ROOT/listen/"
echo "logs:        $LOG_DIR"
echo
echo "To pull down for listening (-L follows the symlinks):"
echo "  rsync -avL <host>:$GEN_ROOT/listen/ ./listen/"

[[ "$FAIL" == "0" ]] || { echo "ERROR: at least one lane failed" >&2; exit 1; }
echo "OK"
