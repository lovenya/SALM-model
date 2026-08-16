#!/usr/bin/env bash
# Large-scale generation at the MEASURED-optimal guidance, with the floor and ceiling baselines,
# then CLAP scoring and an export ready for the SpeechAudioLM metric suite.
#
# WHY s=7.5. Every sample generated before 2026-08-16 used s=3.0, which was my arbitrary pick. The
# guidance sweep, once scored with the mismatched-target control, showed MARGIN peaking at 7.5:
#
#     s        1.0    3.0    5.0    7.5    10.0   15.0
#     MARGIN  0.189  0.258  0.267  0.280  0.274  0.252
#
# `matched` alone was flat across that range, which is why the first reading of the sweep was wrong.
# s=3.0 is generated here too, so the improvement is an A/B on identical rows rather than a claim.
#
# WHY 512 SAMPLES. FAD and KAD are distributional and unstable on the 24 rows we have been using.
# 512 is enough for them to settle while still fitting the walltime (~0.58 s/clip measured).
#
# THE BASELINES MATTER AS MUCH AS THE MODEL. Measured on 24 rows, and each is now a known quantity:
#   b0     unconditional floor      MARGIN ~0 by construction -- anything real must beat it
#   rung0  ground-truth caption     turned out to be a FLOOR, not a ceiling: MARGIN 0.006, r1 at
#          via CLAP TEXT            chance, matched 0.199 BELOW the 0.293 random-clip baseline
#   rung1  CLAP audio on speech     MARGIN 0.004, r1 at chance, self_sim 0.713 -- confirms CLAP
#                                   encodes "someone is speaking", not what was said
# Regenerating them at scale keeps every comparison on identical rows and the same sample count.
#
# Usage:
#   sbatch submission_scripts/run_tamia_s2a_final_eval.sbatch
#   NUM_GEN=1024 RUN_NAME=long100_L2_o2 sbatch ...
#SBATCH --job-name=salm-s2a-final-eval
#SBATCH --account=aip-ravanelm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=01:15:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

ARCH="${ARCH:-s2a_audioldm_whisper_xformer}"
RUN_NAME="${RUN_NAME:-long100_L2_o2}"
CKPT_TAG="${CKPT_TAG:-best}"
NUM_GEN="${NUM_GEN:-512}"
SPLIT="${SPLIT:-val}"
SEED="${SEED:-1234}"
BEST_S="${BEST_S:-7.5}"
OLD_S="${OLD_S:-3.0}"
EVAL_MODEL="${EVAL_MODEL:-salm_e2e}"     # the name registered in model_capabilities.json
STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"

RUN_DIR="$RUNS_ROOT/$ARCH/$RUN_NAME"
CKPT="$RUN_DIR/checkpoints/checkpoint_${CKPT_TAG}.pt"
OUT="$RUN_DIR/final_eval_${STAMP}"
LOG_DIR="$RUN_DIR/logs/final_eval_${STAMP}"
mkdir -p "$OUT" "$LOG_DIR"
[[ -f "$CKPT" ]] || { echo "ERROR: no checkpoint at $CKPT" >&2; exit 1; }

echo "==================================================================="
echo "Final eval: generation at s=$BEST_S vs s=$OLD_S, plus floor/ceiling baselines"
echo "job:    ${SLURM_JOB_ID:-none}   host: $(hostname)"
echo "commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "run:    $ARCH/$RUN_NAME  ckpt=$CKPT_TAG  n=$NUM_GEN  split=$SPLIT"
echo "==================================================================="

salm_start_monitor "$LOG_DIR" "final_eval" 60

COMMON=(--split "$SPLIT" --num-samples "$NUM_GEN" --start-index 0 --seed "$SEED")

# GPU 0/1: the model at the new optimum and at the old default, for a clean A/B.
( CUDA_VISIBLE_DEVICES=0 python -m salm.generate --mode s2a --checkpoint "$CKPT" \
    "${COMMON[@]}" --guidance-scale "$BEST_S" --out-dir "$OUT/s2a_s${BEST_S}" --copy-reference \
    > "$LOG_DIR/gen_best.log" 2>&1 ) & P0=$!
( CUDA_VISIBLE_DEVICES=1 python -m salm.generate --mode s2a --checkpoint "$CKPT" \
    "${COMMON[@]}" --guidance-scale "$OLD_S" --out-dir "$OUT/s2a_s${OLD_S}" \
    > "$LOG_DIR/gen_old.log" 2>&1 ) & P1=$!
# GPU 2: the ground-truth-caption lane (measured to be a floor, not a ceiling).
( CUDA_VISIBLE_DEVICES=2 python -m salm.generate --mode rung0 \
    "${COMMON[@]}" --guidance-scale "$BEST_S" --out-dir "$OUT/rung0_caption" \
    > "$LOG_DIR/gen_rung0.log" 2>&1 ) & P2=$!
# GPU 3: both zero-information controls, run back to back.
( CUDA_VISIBLE_DEVICES=3 bash -c "
    python -m salm.generate --mode b0 --checkpoint '$CKPT' ${COMMON[*]} \
        --guidance-scale $BEST_S --out-dir '$OUT/b0_unconditional' > '$LOG_DIR/gen_b0.log' 2>&1
    python -m salm.generate --mode rung1 ${COMMON[*]} \
        --guidance-scale $BEST_S --out-dir '$OUT/rung1_clap' > '$LOG_DIR/gen_rung1.log' 2>&1
  " ) & P3=$!

FAIL=0
for p in $P0 $P1 $P2 $P3; do wait "$p" || FAIL=1; done
[[ "$FAIL" == "0" ]] || echo "WARN: at least one generation lane failed; see $LOG_DIR" >&2

# ------------------------------------------------------------------ CLAP scoring
DIRS=()
for d in "$OUT"/*/; do [[ -f "${d}manifest.jsonl" ]] && DIRS+=("${d%/}"); done
if [[ ${#DIRS[@]} -eq 0 ]]; then echo "ERROR: nothing generated" >&2; exit 1; fi
echo
echo "### CLAP scoring (MARGIN + audio_r1 are the numbers that matter) ###"
CUDA_VISIBLE_DEVICES=0 python scripts/score_generations_clap.py \
    --dirs "${DIRS[@]}" --split "$SPLIT" --out "$OUT/final_scores.json" \
    2>&1 | tee "$LOG_DIR/score.log"

# ------------------------------------------------- export for the metric suite
# Bridges our flat schema to SpeechAudioLM's evaluator. --model must be the name registered in
# that repo's configs/model_capabilities.json; the script's own default is the pre-registration
# name and would be rejected by validate_model_task.
echo
echo "### Export for fad / kad / kl_passt / target_caption_clap_score ###"
for d in "${DIRS[@]}"; do
    name="$(basename "$d")"
    python scripts/export_eval_manifest.py \
        --generation-dir "$d" --out-dir "$OUT/eval_export/$name" \
        --stage s2a --model "$EVAL_MODEL" --split "$SPLIT" \
        2>&1 | tee -a "$LOG_DIR/export.log" || echo "  WARN: export failed for $name" >&2
done

echo
echo "results: $OUT"
echo "scores:  $OUT/final_scores.json"
echo "export:  $OUT/eval_export/   (feed to SpeechAudioLM evaluate.py --model $EVAL_MODEL)"
echo "OK"
