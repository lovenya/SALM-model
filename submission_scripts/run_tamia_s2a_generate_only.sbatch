#!/usr/bin/env bash
# Generate audio from an EXISTING checkpoint, independently of training.
#
# Safety net: training checkpoints `checkpoint_best.pt` every epoch that improves val,
# `checkpoint_ep<N>.pt` every 5 epochs, and `checkpoint_last.pt` at the end. So a partly
# trained model can produce audio at any time -- useful when a deadline arrives before
# training finishes, or to hear how samples evolve across epochs.
#
# Short by design (~5 min) so it backfills quickly.
#
# Usage:
#   # newest checkpoint of the newest run, 16 samples:
#   sbatch submission_scripts/run_tamia_s2a_generate_only.sbatch
#
#   # a specific checkpoint / count / mode:
#   CKPT=/path/to/checkpoint_ep05.pt NUM_GEN=8 \
#     sbatch submission_scripts/run_tamia_s2a_generate_only.sbatch
#
#   # controls, no checkpoint needed:
#   MODE=rung0 sbatch submission_scripts/run_tamia_s2a_generate_only.sbatch   # caption ceiling
#   MODE=rung1 sbatch submission_scripts/run_tamia_s2a_generate_only.sbatch   # CLAP-on-speech control
# Requests a WHOLE node even though generation uses one GPU. TamIA rejects partial h100
# requests outright: "The h100 GPUs are only allocated by node. Please request a whole
# node using --gpus-per-node=h100:4". Since the node is reserved regardless, CPU/RAM are
# asked for in full too.
#SBATCH --job-name=salm-s2a-generate
#SBATCH --account=aip-ravanelm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/l/lovenya/salm_migration_clean/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

MODE="${MODE:-s2a}"
NUM_GEN="${NUM_GEN:-16}"
SPLIT="${SPLIT:-val}"
STEPS="${STEPS:-100}"
GUIDANCE="${GUIDANCE:-3.0}"
RUN_ID="$(TZ=America/Toronto date +%Y%m%d_%H%M)"

CKPT_ARGS=()
LABEL="$MODE"

if [[ "$MODE" == "s2a" ]]; then
    if [[ -z "${CKPT:-}" ]]; then
        # Newest checkpoint across all runs, so this works without arguments.
        CKPT=$(find "$OUT_ROOT/checkpoints" -name "checkpoint_*.pt" -printf '%T@ %p\n' 2>/dev/null \
               | sort -rn | head -1 | cut -d' ' -f2-)
    fi
    if [[ -z "${CKPT:-}" || ! -f "$CKPT" ]]; then
        echo "ERROR: no checkpoint found under $OUT_ROOT/checkpoints" >&2
        echo "       Pass CKPT=/path/to/checkpoint.pt, or use MODE=rung0 which needs none." >&2
        exit 1
    fi
    echo "using checkpoint: $CKPT"
    ls -la "$CKPT"
    CKPT_ARGS=(--checkpoint "$CKPT")
    LABEL="s2a_$(basename "$CKPT" .pt)"
fi

GEN_DIR="$OUT_ROOT/generations/standalone_${LABEL}_${RUN_ID}"
LOG_DIR="$SALM_SCRATCH/logs/salm_model/standalone_${LABEL}_${RUN_ID}"
mkdir -p "$GEN_DIR" "$LOG_DIR"

salm_print_header "S2A generate-only: mode=$MODE n=$NUM_GEN split=$SPLIT"
salm_start_monitor "$LOG_DIR" "gen_${LABEL}_${RUN_ID}" 30

python -m salm.generate \
    --mode "$MODE" "${CKPT_ARGS[@]}" \
    --split "$SPLIT" --num-samples "$NUM_GEN" \
    --steps "$STEPS" --guidance-scale "$GUIDANCE" \
    --out-dir "$GEN_DIR" --copy-reference \
    2>&1 | tee "$LOG_DIR/generate.log"

echo
echo "=============================================="
python - <<PY
import json, pathlib
s = pathlib.Path("$GEN_DIR/summary.json")
if s.exists():
    d = json.loads(s.read_text())
    print(f"mode={d['mode']} n={d['num_samples']} steps={d['steps']} cfg={d['guidance_scale']}")
    print(f"rms mean={d['rms_mean']:.5f} min={d['rms_min']:.5f} "
          f"near_silent={d['near_silent_count']}")
    if d['near_silent_count']:
        print("WARNING: near-silent outputs -- generation path may be broken.")
else:
    print("MISSING summary.json")
PY
echo
echo "AUDIO: $GEN_DIR/audio/"
echo "=============================================="
