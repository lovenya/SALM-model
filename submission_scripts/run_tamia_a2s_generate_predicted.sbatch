#!/usr/bin/env bash
# Generate speech from the AmbientContentPredictor, the first fully audio-only content path
# -- no target speech touched anywhere in this job. See src/salm/a2s/content_predictor.py.
#
# Uses the IMPROVED v4 renderer (200 epochs, better bind/x0 than v3) and BOTH predictor
# checkpoints -- best (epoch 4, best val_cos, before overfitting set in) and last (epoch 62,
# most-trained, val_cos declined but the user's own point stands: diffusion loss curves in
# this project have repeatedly NOT tracked audio quality, so don't prejudge by the number.
# Listen to both.
#
# Also generates `retrieval` on the SAME rows for a direct side-by-side: does a trained
# predictor beat nearest-neighbour lookup, or not.
#
# EXPECTED RESULTS: no strong prior here -- this is genuinely the first look at the fully
# honest inference-time path. What each outcome means:
#   - predicted/best sounds coherent, on-topic-ish, comparable to or better than retrieval
#     -> the regression generalizes past lookup. Worth investing in a better/bigger predictor.
#   - predicted/best sounds like retrieval or worse -> the predictor isn't adding value over
#     a lookup yet; check whether MORE epochs (predicted/last) or regularization changes help,
#     or whether a richer input representation than mean CLAP is needed.
#   - predicted/last (overfit by val_cos) sounds WORSE than predicted/best by ear -> the val_cos
#     curve was telling the truth this time; keep the early-stopped checkpoint.
#   - predicted/last sounds AS GOOD OR BETTER than predicted/best despite worse val_cos -> the
#     regression loss (cosine similarity of a 512-d vector) doesn't track what actually reaches
#     the ear through the renderer, the same disconnect this project has hit before with
#     diffusion losses. Worth building a downstream metric (score_intelligibility.py on the
#     generations, not val_cos on the predictor) as the real model-selection signal.
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_generate_predicted.sbatch
#SBATCH --job-name=salm-a2s-gen-predicted
#SBATCH --account=aip-ravanelm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail
# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

R="$RUNS_ROOT/a2s_audioldm_clapseq_xformer"
CKPT="${CKPT:-$R/content_concat_v4/checkpoints/checkpoint_ep199.pt}"
CTC_CKPT="${CTC_CKPT:-$OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt}"
PRED_ROOT="$OUTPUTS_ROOT/content_predictor/content_predictor_v1/checkpoints"
NUM_GEN="${NUM_GEN:-16}"
STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"
OUT_ROOT="$OUTPUTS_ROOT/generations/predicted_vs_retrieval_${STAMP}"

for F in "$CKPT" "$CTC_CKPT" "$PRED_ROOT/checkpoint_best.pt" "$PRED_ROOT/checkpoint_last.pt"; do
    [[ -f "$F" ]] || { echo "MISSING: $F" >&2; exit 1; }
done

salm_print_header "predicted (best+last) vs retrieval, n=$NUM_GEN, renderer=$CKPT"

for TAG_SRC in "predicted_best:predicted:$PRED_ROOT/checkpoint_best.pt" \
               "predicted_last:predicted:$PRED_ROOT/checkpoint_last.pt" \
               "retrieval:retrieval:"; do
    TAG="${TAG_SRC%%:*}"; REST="${TAG_SRC#*:}"
    SRC="${REST%%:*}"; PRED_CKPT="${REST#*:}"
    OUT="$OUT_ROOT/$TAG"
    mkdir -p "$OUT"
    echo
    echo "=== $TAG (content-source=$SRC) ==="
    EXTRA_ARGS=()
    [[ -n "$PRED_CKPT" ]] && EXTRA_ARGS=(--content-predictor-checkpoint "$PRED_CKPT")
    python -m salm.generate --mode a2s --direction a2s \
        --checkpoint "$CKPT" \
        --content-source "$SRC" --ctc-checkpoint "$CTC_CKPT" \
        --content-retrieval-root "$OUTPUTS_ROOT/features_a2s_cond" \
        "${EXTRA_ARGS[@]}" \
        --split val --num-samples "$NUM_GEN" \
        --out-dir "$OUT" --copy-reference \
        2>&1 | tee "$OUT.log"
done

echo
echo "=========================================================="
echo "listen (input / generated / reference all present, a2s):"
for TAG in predicted_best predicted_last retrieval; do
    echo "  $OUT_ROOT/$TAG/audio/   <id>.wav=generated  <id>__reference.wav=target speech  <id>__input_ambient.wav=input"
done
echo "=========================================================="
