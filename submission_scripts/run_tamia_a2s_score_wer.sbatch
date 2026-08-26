#!/usr/bin/env bash
# WER on GENERATED A2S audio, scored by an INDEPENDENT judge (Whisper-large-v3) -- see
# scripts/a2s/score_generated_wer.py for why it must be Whisper, not the O-F training CTC head.
#
# Every A2S generation lane so far was judged by ear only. This scores them all retroactively
# in one job -- the real audio is already on disk, nothing needs regenerating.
#
# Single-GPU, short (Whisper inference over ~8 x 16 clips). Requests the whole node anyway
# (TamIA's h100 policy); GPUs 1-3 idle unless DIRS is extended to run more lanes concurrently.
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_score_wer.sbatch
#   DIRS="path1 path2 ..." sbatch submission_scripts/run_tamia_a2s_score_wer.sbatch   # custom set
#SBATCH --job-name=salm-a2s-score-wer
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

LOG_DIR="$OUTPUTS_ROOT/logs/salm_model/a2s_score_wer_$(TZ=America/Toronto date +%Y%m%d_%H%M)"
mkdir -p "$LOG_DIR"

salm_print_header "A2S generated-audio WER scoring (independent judge: Whisper-large-v3)"
salm_start_monitor "$LOG_DIR" "score_wer" 30

# Filter to dirs that actually exist -- A3/O-F's generations may not be written yet if this
# runs before that training job finishes; skip cleanly rather than fail the whole job over it.
EXISTING=()
for d in $DIRS; do
    if [[ -f "$d/manifest.jsonl" ]]; then
        EXISTING+=("$d")
    else
        echo "skipping (no manifest yet): $d"
    fi
done

CUDA_VISIBLE_DEVICES=0 python scripts/a2s/score_generated_wer.py \
    --dirs "${EXISTING[@]}" --device cuda \
    --out "$OUTPUTS_ROOT/wer_comparison_$(TZ=America/Toronto date +%Y%m%d_%H%M).json" \
    2>&1 | tee "$LOG_DIR/score.log"
