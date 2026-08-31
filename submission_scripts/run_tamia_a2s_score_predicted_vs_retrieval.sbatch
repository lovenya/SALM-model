#!/usr/bin/env bash
# Score the predicted_vs_retrieval generation (run_tamia_a2s_generate_predicted.sbatch)
# against the calibrated intelligibility floor. Meant to run CHAINED after that job via
# --dependency=afterok, so results are ready without anyone watching the queue overnight.
#
# EXPECTED RESULTS: retrieval's own inter_asr_dist is the number to beat (already measured,
# ~0.55-0.56 on job 433314/433428). predicted_best or predicted_last landing meaningfully
# below that -> the trained regressor beats nearest-neighbour lookup, worth investing further.
# Landing at or above retrieval's number -> the lookup remains the better F1 source for now;
# don't retrain the predictor bigger without a different input representation first.
#
# Usage:
#   GEN_ROOT=<predicted_vs_retrieval_TIMESTAMP dir> sbatch --dependency=afterok:<jobid> \
#     submission_scripts/run_tamia_a2s_score_predicted_vs_retrieval.sbatch
#SBATCH --job-name=salm-a2s-score-predicted
#SBATCH --account=aip-ravanelm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail
# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

CTC_CKPT="${CTC_CKPT:-$OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt}"

# Auto-find the most recent predicted_vs_retrieval dir if GEN_ROOT wasn't passed explicitly.
GEN_ROOT="${GEN_ROOT:-$(ls -d "$OUTPUTS_ROOT"/generations/predicted_vs_retrieval_* 2>/dev/null | tail -1)}"
[[ -n "$GEN_ROOT" && -d "$GEN_ROOT" ]] || { echo "ERROR: no predicted_vs_retrieval_* dir found" >&2; exit 1; }

CTRL=$(ls -d "$OUTPUTS_ROOT"/generations/control_lanes_* 2>/dev/null | tail -1)
[[ -n "$CTRL" ]] || { echo "ERROR: no control_lanes_* dir found -- run make_control_lanes.py first" >&2; exit 1; }

DIRS=("$CTRL/real_speech" "$CTRL/vae_roundtrip" "$GEN_ROOT/predicted_best" "$GEN_ROOT/predicted_last" "$GEN_ROOT/retrieval")

salm_print_header "Scoring predicted (best+last) vs retrieval: $GEN_ROOT"

python scripts/a2s/score_intelligibility.py \
    --dirs "${DIRS[@]}" --ctc-checkpoint "$CTC_CKPT" --device cuda \
    --out "$OUTPUTS_ROOT/intelligibility_predicted_vs_retrieval_$(TZ=America/Toronto date +%Y%m%d_%H%M).json"

echo "READ retrieval's number as the bar predicted must beat."
