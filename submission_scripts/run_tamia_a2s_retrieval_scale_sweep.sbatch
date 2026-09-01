#!/usr/bin/env bash
# Does the content-scale win (job 433675: oracle dist 0.454->0.389 at scale=1.5, near the
# 0.372 VAE-roundtrip ceiling) generalize to RETRIEVAL -- the actual deployable content
# source? Retrieval alone already beats the predictor badly (0.50 vs 0.94-0.96, 2026-08-31),
# so this checks whether amplifying it closes more of the remaining gap to real speech,
# with zero retraining.
#
# EXPECTED RESULTS:
#   - retrieval's dist improves with scale similarly to oracle's -> content strength was
#     under-weighted for ANY real content, not just oracle-quality content. Ship scale>1 as
#     the new default for real generation immediately.
#   - retrieval does NOT improve, or gets worse, with scale -> the earlier win was specific to
#     oracle's already-correct content; amplifying WRONG-topic content just makes the wrong
#     topic more confidently wrong. Keep scale=1.0 for retrieval; the scale knob stays
#     oracle/diagnostic-only.
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_retrieval_scale_sweep.sbatch
#SBATCH --job-name=salm-a2s-retrieval-scale
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
NUM_GEN="${NUM_GEN:-16}"
STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"
OUT_ROOT="$OUTPUTS_ROOT/generations/retrieval_scale_${STAMP}"

for F in "$CKPT" "$CTC_CKPT"; do
    [[ -f "$F" ]] || { echo "MISSING: $F" >&2; exit 1; }
done

salm_print_header "Retrieval content-scale sweep, renderer=$CKPT"

for SCALE in 1.0 1.5 2.0; do
    OUT="$OUT_ROOT/scale_${SCALE}"
    mkdir -p "$OUT"
    echo
    echo "=== retrieval, content_scale=$SCALE ==="
    python -m salm.generate --mode a2s --direction a2s \
        --checkpoint "$CKPT" \
        --content-source retrieval --ctc-checkpoint "$CTC_CKPT" --content-scale "$SCALE" \
        --content-retrieval-root "$OUTPUTS_ROOT/features_a2s_cond" \
        --split val --num-samples "$NUM_GEN" \
        --out-dir "$OUT" --copy-reference \
        2>&1 | tee "$OUT.log"
done

echo
echo "=== scoring against the calibrated floor ==="
CTRL=$(ls -d "$OUTPUTS_ROOT"/generations/control_lanes_* 2>/dev/null | tail -1)
DIRS=("$CTRL/real_speech" "$CTRL/vae_roundtrip")
for SCALE in 1.0 1.5 2.0; do DIRS+=("$OUT_ROOT/scale_${SCALE}"); done
python scripts/a2s/score_intelligibility.py \
    --dirs "${DIRS[@]}" --ctc-checkpoint "$CTC_CKPT" --device cuda \
    --out "$OUTPUTS_ROOT/intelligibility_retrieval_scale_${STAMP}.json"

echo "=========================================================="
echo "listen: $OUT_ROOT/scale_<value>/audio/"
echo "=========================================================="
