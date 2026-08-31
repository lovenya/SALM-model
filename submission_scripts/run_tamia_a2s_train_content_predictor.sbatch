#!/usr/bin/env bash
# Train AmbientContentPredictor: ambient audio -> predicted content sequence, replacing
# nearest-neighbour retrieval with a real trained regressor. See
# src/salm/a2s/content_predictor.py for the full mechanism, and docs/RESEARCH_DIRECTIONS.md
# (D3) for where this sits in the plan.
#
# WHY NOW. The renderer is proven (2026-08-30, job 433314): oracle content -> near-verbatim
# speech via the frame-aligned concat mechanism. Retrieval content (the honest, deployable
# path) already produces fluent, real English -- just topically wrong, since a lookup can
# only ever hand back something a training example already said. This predictor is meant to
# generalize past that lookup limit.
#
# EXPECTED RESULTS, pre-registered:
#   val_cos (cosine similarity, predicted vs true content sequence) is THE metric -- val_loss
#   includes an MSE term whose scale is less interpretable alone.
#   - val_cos climbs well above ~0 (chance for high-dim random vectors is ~0) and keeps
#     improving for most of training -> the predictor is learning real structure. Plug its
#     checkpoint into `generate.py --content-source predicted` next.
#   - val_cos plateaus low and early -> ambient audio does not carry enough signal for even a
#     trained regressor to beat simple statistics; retrieval remains the best available F1
#     baseline and effort should go to a better SOURCE representation (e.g. LLM captioning)
#     rather than a bigger predictor architecture.
#   THE REAL TEST is downstream, not this number alone: does `predicted` content, scored via
#   scripts/a2s/score_intelligibility.py, beat `retrieval`'s inter_asr_dist on the SAME rows?
#   A predictor with a good val_cos that doesn't move that number would mean the regression
#   target itself (matching the CTC-probe's content space) isn't the right thing to optimize.
#
# Single-GPU by design, same reasoning as the CTC probe: a few million params against 17.6k
# rows, nowhere near needing DDP.
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_train_content_predictor.sbatch
#SBATCH --job-name=salm-a2s-content-predictor
#SBATCH --account=aip-ravanelm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

RUN_NAME="${RUN_NAME:-content_predictor_v1}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
TRAIN_MINUTES="${TRAIN_MINUTES:-35}"
CTC_CKPT="${CTC_CKPT:-$OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt}"
# v1 overfit past ~epoch 4 (val_cos 0.70 -> 0.65 by ep62). Defaults below match v1's own
# (equally-default) values so a plain invocation is unchanged; override for a v2 attempt.
WEIGHT_DECAY="${WEIGHT_DECAY:-1.0e-2}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
DROPOUT="${DROPOUT:-0.1}"
EVAL_EVERY="${EVAL_EVERY:-2}"

RUN_DIR="$OUTPUTS_ROOT/content_predictor/$RUN_NAME"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

for F in "$FEATURES_ROOT/a2s/train/latents_mean.npy" "$OUTPUTS_ROOT/features_a2s_cond/train/cond_seq.npy" "$CTC_CKPT"; do
    [[ -f "$F" ]] || { echo "MISSING: $F" >&2; exit 1; }
done

CKPT="$RUN_DIR/checkpoints/checkpoint_last.pt"
RESUME_ARGS=()
if [[ -f "$CKPT" ]]; then
    echo "resuming from existing checkpoint: $CKPT"
    RESUME_ARGS=("--resume-from" "$CKPT")
fi

salm_print_header "Content predictor: ambient audio -> content sequence, $EPOCHS epochs"
salm_start_monitor "$LOG_DIR" "$RUN_NAME" 60

CUDA_VISIBLE_DEVICES=0 python scripts/a2s/train_content_predictor.py \
    --run-name "$RUN_NAME" --runs-root "$OUTPUTS_ROOT/content_predictor" \
    --ctc-checkpoint "$CTC_CKPT" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --max-train-minutes "$TRAIN_MINUTES" \
    --weight-decay "$WEIGHT_DECAY" --hidden-dim "$HIDDEN_DIM" --dropout "$DROPOUT" \
    --eval-every-epochs "$EVAL_EVERY" \
    "${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}" \
    2>&1 | tee -a "$LOG_DIR/train.log"

echo
echo "==================== CONTENT PREDICTOR SUMMARY ===================="
python - <<PY
import json, pathlib
h = json.loads((pathlib.Path("$RUN_DIR") / "history.json").read_text())
print(f"epochs completed: {len(h)}")
cos = [(r["epoch"], r["val_cos"]) for r in h if r.get("val_cos") is not None]
if cos:
    best = max(cos, key=lambda x: x[1])
    print(f"val_cos: first={cos[0][1]:.4f} last={cos[-1][1]:.4f} best={best[1]:.4f} @ep{best[0]}")
et = [r.get("epoch_minutes", 0) for r in h]
print(f"mean epoch time: {sum(et)/max(len(et),1):.2f} min")
PY
echo
echo "checkpoint for generate.py --content-source predicted:"
echo "  $RUN_DIR/checkpoints/checkpoint_best.pt"
echo "======================================================================="
