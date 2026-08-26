#!/usr/bin/env bash
# Rank 0 of the F1/F2 plan: content-independent intelligibility scoring on GENERATED A2S audio.
# See scripts/a2s/score_intelligibility.py for the full mechanism and why WER alone can't be
# trusted to judge O-F (or anything else) -- it conflates "wrong words" with "no real words".
#
# Single-GPU, short (two ASR passes over ~8-10 dirs x 16 clips). Requests the whole node anyway
# (TamIA's h100 policy).
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_score_intelligibility.sbatch
#   DIRS="path1 path2 ..." sbatch submission_scripts/run_tamia_a2s_score_intelligibility.sbatch
#SBATCH --job-name=salm-a2s-score-intel
#SBATCH --account=aip-ravanelm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

CTC_CKPT="${CTC_CKPT:-$OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt}"
R="$RUNS_ROOT/a2s_audioldm_clapseq_xformer"
O="$OUTPUTS_ROOT/generations"

DEFAULT_DIRS="
$R/probe40_A0_plain/generations/a2s_best
$R/probe40_A0_plain/generations/b0_uncond
$R/probe40_A1_o2/generations/a2s_best
$R/probe40_A1_o2/generations/b0_uncond
$R/probe40_A2_o2_infonce/generations/a2s_best
$R/probe40_A2_o2_infonce/generations/b0_uncond
$O/a2s_oracle_ablation_20260824_1116/oracle_text
$O/a2s_oracle_ablation_20260824_1116/real_conditioner
$R/probe40_A3_o2_ctc/generations/a2s_best
$R/probe40_A3_o2_ctc/generations/b0_uncond
"
DIRS="${DIRS:-$DEFAULT_DIRS}"

LOG_DIR="$OUTPUTS_ROOT/logs/salm_model/a2s_score_intel_$(TZ=America/Toronto date +%Y%m%d_%H%M)"
mkdir -p "$LOG_DIR"

salm_print_header "A2S intelligibility scoring (Rank 0, F1/F2 plan): $CTC_CKPT"
salm_start_monitor "$LOG_DIR" "score_intel" 30

if [[ ! -f "$CTC_CKPT" ]]; then
    echo "ERROR: CTC checkpoint not found: $CTC_CKPT" >&2
    exit 1
fi

EXISTING=()
for d in $DIRS; do
    if [[ -f "$d/manifest.jsonl" ]]; then
        EXISTING+=("$d")
    else
        echo "skipping (no manifest yet): $d"
    fi
done

CUDA_VISIBLE_DEVICES=0 python scripts/a2s/score_intelligibility.py \
    --dirs "${EXISTING[@]}" --ctc-checkpoint "$CTC_CKPT" --device cuda \
    --out "$OUTPUTS_ROOT/intelligibility_comparison_$(TZ=America/Toronto date +%Y%m%d_%H%M).json" \
    2>&1 | tee "$LOG_DIR/score.log"
