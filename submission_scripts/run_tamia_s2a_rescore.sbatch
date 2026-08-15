#!/usr/bin/env bash
# Re-score existing generated audio with the CONTROL that the first scoring run lacked.
#
# No regeneration: every lane's wavs are already on disk. This is a CLAP embedding pass only,
# a few minutes.
#
# WHY THIS EXISTS. The first guidance sweep reported clap_audio_cos ~0.55 across every scale, which
# looks like a strong result and is in fact uninterpretable. Ambient clips resemble one another, so
# a generic output scores well against ANY target. Verified on synthetic data: a lane emitting ONE
# identical output for every input still scores matched = 0.1997, while its margin over mismatched
# targets is exactly 0.0000 and its retrieval accuracy sits precisely at chance.
#
# So the scorer now reports, per lane:
#   matched     cos(generated_i, target_i)          -- the old, misleading number
#   mismatched  cos(generated_i, target_j), i != j  -- THE CONTROL
#   MARGIN      matched - mismatched                -- ~0 means no per-example information
#   audio_r1    is the TRUE target the nearest?     -- chance is 1/n
#
# Scores both the guidance sweep (does MARGIN rise with s?) and the gen-evolution lanes, which
# include the rung0 caption ceiling and the pristine-vs-trained U-Net A/B -- i.e. the upper bound
# and the "did finetuning the U-Net do anything" control, on identical rows.
#
# Usage:
#   sbatch submission_scripts/run_tamia_s2a_rescore.sbatch
#   SWEEP_DIR=/path/to/guidance_sweep_XXXX GEN_DIR=/path/to/generations sbatch ...
#SBATCH --job-name=salm-s2a-rescore
#SBATCH --account=aip-csubakan
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

O2_DIR="$RUNS_ROOT/s2a_audioldm_whisper_xformer/long100_L2_o2"
SWEEP_DIR="${SWEEP_DIR:-$(ls -1d "$O2_DIR"/guidance_sweep_* 2>/dev/null | sort | tail -1)}"
GEN_DIR="${GEN_DIR:-$O2_DIR/generations}"
STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"
OUT="$O2_DIR/rescore_${STAMP}"
mkdir -p "$OUT"

echo "==================================================================="
echo "Re-score with the mismatched-target control"
echo "job:    ${SLURM_JOB_ID:-none}   host: $(hostname)"
echo "commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "sweep:  ${SWEEP_DIR:-none}"
echo "gen:    ${GEN_DIR:-none}"
echo "==================================================================="

# ---------------------------------------------------------------- guidance sweep
if [[ -n "$SWEEP_DIR" && -d "$SWEEP_DIR" ]]; then
    mapfile -t SDIRS < <(ls -1d "$SWEEP_DIR"/s*/ 2>/dev/null | sed 's#/$##')
    if [[ ${#SDIRS[@]} -gt 0 ]]; then
        echo
        echo "### Guidance sweep: does MARGIN rise with s? ###"
        CUDA_VISIBLE_DEVICES=0 python scripts/score_generations_clap.py \
            --dirs "${SDIRS[@]}" --split val --out "$OUT/sweep_scores_v2.json"
    else
        echo "WARN: no s* lanes under $SWEEP_DIR"
    fi
else
    echo "WARN: no guidance sweep directory found"
fi

# ------------------------------------------------------------- gen-evolution lanes
if [[ -d "$GEN_DIR" ]]; then
    mapfile -t GDIRS < <(find "$GEN_DIR" -maxdepth 1 -mindepth 1 -type d \
                         -exec test -f '{}/manifest.jsonl' \; -print | sort)
    if [[ ${#GDIRS[@]} -gt 0 ]]; then
        echo
        echo "### Snapshots, rung0 ceiling, and the pristine-vs-trained U-Net A/B ###"
        CUDA_VISIBLE_DEVICES=1 python scripts/score_generations_clap.py \
            --dirs "${GDIRS[@]}" --split val --out "$OUT/generation_scores_v2.json"
    else
        echo "WARN: no lanes with manifest.jsonl under $GEN_DIR"
    fi
else
    echo "WARN: no generations directory at $GEN_DIR"
fi

echo
echo "results: $OUT"
echo "OK"
