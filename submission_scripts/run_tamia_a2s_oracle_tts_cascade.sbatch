#!/usr/bin/env bash
# Rank 1 of the F1/F2 plan: render REAL target transcripts with off-the-shelf TTS, to
# establish the F2 (phonetic realizability) ceiling independent of our own pipeline.
# See scripts/a2s/run_oracle_tts_cascade.py for the full mechanism and the max_new_tokens
# trap (verified against the actual downloaded checkpoint: ships with 8, TTS needs 512).
#
# Two independent lanes on separate GPUs (Kokoro-82M, audioldm2-gigaspeech) -- both read-only
# inference against real transcripts, no training. Requests the whole node anyway (TamIA's
# h100 policy); GPUs 2-3 idle this job, nothing else queued to share them with right now.
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_oracle_tts_cascade.sbatch
#SBATCH --job-name=salm-a2s-oracle-tts
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

SPLIT="${SPLIT:-val}"
NUM_SAMPLES="${NUM_SAMPLES:-16}"
RUN_ID="$(TZ=America/Toronto date +%Y%m%d_%H%M)"
OUT_DIR="$OUTPUTS_ROOT/generations/oracle_tts_cascade_${RUN_ID}"
mkdir -p "$OUT_DIR"

LOG_DIR="$OUTPUTS_ROOT/logs/salm_model/oracle_tts_cascade_${RUN_ID}"
mkdir -p "$LOG_DIR"

salm_print_header "A2S Rank 1: oracle-transcript TTS cascade (F2 ceiling)  n=$NUM_SAMPLES split=$SPLIT"
salm_start_monitor "$LOG_DIR" "oracle_tts_${RUN_ID}" 30

# audioldm2-gigaspeech and kokoro are entirely independent -- separate GPUs.
CUDA_VISIBLE_DEVICES=0 python scripts/a2s/run_oracle_tts_cascade.py \
    --split "$SPLIT" --num-samples "$NUM_SAMPLES" --out-dir "$OUT_DIR" \
    --skip-kokoro \
    > "$LOG_DIR/01_audioldm2.log" 2>&1 &
PID_A=$!
CUDA_VISIBLE_DEVICES=1 python scripts/a2s/run_oracle_tts_cascade.py \
    --split "$SPLIT" --num-samples "$NUM_SAMPLES" --out-dir "$OUT_DIR" \
    --skip-audioldm2 \
    > "$LOG_DIR/02_kokoro.log" 2>&1 &
PID_B=$!
FAIL=0
wait "$PID_A" || FAIL=1
wait "$PID_B" || FAIL=1
tail -n 15 "$LOG_DIR/01_audioldm2.log" "$LOG_DIR/02_kokoro.log"

echo
echo "==================== NEXT: SCORE BOTH LANES ===================="
echo "python scripts/a2s/score_intelligibility.py \\"
echo "    --dirs $OUT_DIR/audioldm2_gigaspeech $OUT_DIR/kokoro_82m \\"
echo "    --ctc-checkpoint $OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt"
echo "==================================================================="
[[ "$FAIL" == "0" ]] || { echo "ERROR: a lane failed -- check logs above" >&2; exit 1; }
echo "OK"
