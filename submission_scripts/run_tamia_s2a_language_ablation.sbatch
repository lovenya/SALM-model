#!/usr/bin/env bash
# Does the model actually use LANGUAGE, or just the acoustics of the speech?
#
# WHY THIS IS THE MOST IMPORTANT ABLATION WE HAVE NOT RUN. The paper's central claim is that the
# model infers an *implied* scene from what was said -- "you almost hit me, slow down" contains no
# sound-event noun, so the mapping requires a linguistic inference step. **Nothing we currently
# measure supports that claim.** Our MARGIN of 0.286 and audio_r1 of 35x chance would look identical
# if the model were reading speaker identity, recording channel, background leakage in the speech
# track, or nothing but prosodic energy. A reviewer will ask, and today we cannot answer.
#
# Each lane destroys ONE property and preserves the rest. Verified on a real clip before submitting
# (log-spectrum correlation to the original / waveform correlation / RMS):
#
#   lane      spec_corr  wave_corr   rms      what is destroyed
#   none        1.000      1.000    0.146     -- reference
#   reverse     1.000     -0.020    0.146     ALL linguistic content; spectrum + timbre + energy kept
#   shuffle     0.719     -0.004    0.146     word order and syntax (250 ms segments permuted)
#   lowpass     0.581      0.823    0.101     phonetic detail; F0, rhythm and envelope kept
#   noise       0.001     -0.006    0.146     everything but the energy level
#
# `reverse` is the decisive one: identical long-term spectrum and identical RMS, zero waveform
# correlation. All lanes preserve LENGTH exactly, so the encoder's valid-frame mask is unchanged and
# cannot itself explain any difference.
#
# HOW TO READ THE RESULT (write this down before looking):
#   reverse collapses toward the unconditional floor (MARGIN ~0.00)  => the model IS using language.
#       This is the result the paper needs, and it is a figure.
#   reverse survives near 0.286                                      => the model is NOT using
#       language; it is reading acoustic surface properties, and the central claim must be rewritten.
#   lowpass survives but reverse collapses                           => prosody alone carries it.
#   shuffle survives but reverse collapses                           => a bag of words is enough;
#       syntax is not needed. (A plausible and publishable middle outcome.)
#   noise should land at the floor in every scenario; if it does not, the pipeline is not actually
#       responding to its input and every other lane is uninterpretable.
#
# Reference points, all measured at n=512: ours 0.286 / unconditional (b0) 0.002 / CLAP-on-speech
# 0.000. So the dynamic range this ablation has to resolve is 0.00 to 0.29, and at n=512 the
# detectable difference is ~0.016 -- comfortably enough.
#
# Usage:
#   sbatch submission_scripts/run_tamia_s2a_language_ablation.sbatch
#   NUM_GEN=256 sbatch ...
#SBATCH --job-name=salm-s2a-lang-ablation
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
GS="${GS:-7.5}"
MODES="${MODES:-none reverse shuffle lowpass noise}"
STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"

RUN_DIR="$RUNS_ROOT/$ARCH/$RUN_NAME"
CKPT="$RUN_DIR/checkpoints/checkpoint_${CKPT_TAG}.pt"
OUT="$RUN_DIR/language_ablation_${STAMP}"
LOG_DIR="$RUN_DIR/logs/language_ablation_${STAMP}"
mkdir -p "$OUT" "$LOG_DIR"
[[ -f "$CKPT" ]] || { echo "ERROR: no checkpoint at $CKPT" >&2; exit 1; }

echo "==================================================================="
echo "Language-degradation ablation: does the model use LANGUAGE?"
echo "job:    ${SLURM_JOB_ID:-none}   host: $(hostname)"
echo "commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "run:    $ARCH/$RUN_NAME  ckpt=$CKPT_TAG  n=$NUM_GEN  s=$GS"
echo "modes:  $MODES"
echo "==================================================================="

salm_start_monitor "$LOG_DIR" "lang_ablation" 60

read -r -a MODE_ARR <<< "$MODES"
PIDS=(); GPU=0
for m in "${MODE_ARR[@]}"; do
    echo "  lane $m on GPU $GPU"
    ( CUDA_VISIBLE_DEVICES=$GPU python -m salm.generate \
        --mode s2a --checkpoint "$CKPT" \
        --split "$SPLIT" --num-samples "$NUM_GEN" --start-index 0 --seed "$SEED" \
        --guidance-scale "$GS" --degrade "$m" \
        --out-dir "$OUT/$m" \
        > "$LOG_DIR/gen_${m}.log" 2>&1 ) &
    PIDS+=($!)
    GPU=$(( (GPU + 1) % 4 ))
    # Wave barrier so no GPU is ever oversubscribed (5 modes over 4 devices).
    if [[ "$GPU" == "0" ]]; then
        for p in "${PIDS[@]}"; do wait "$p" || echo "  WARN: a lane failed" >&2; done
        PIDS=()
    fi
done
for p in ${PIDS[@]+"${PIDS[@]}"}; do wait "$p" || echo "  WARN: a lane failed" >&2; done

DIRS=()
for m in "${MODE_ARR[@]}"; do [[ -f "$OUT/$m/manifest.jsonl" ]] && DIRS+=("$OUT/$m"); done
[[ ${#DIRS[@]} -gt 0 ]] || { echo "ERROR: nothing generated; see $LOG_DIR" >&2; exit 1; }

echo
echo "### Scoring -- read MARGIN and audio_r1, not matched ###"
CUDA_VISIBLE_DEVICES=0 python scripts/score_generations_clap.py \
    --dirs "${DIRS[@]}" --split "$SPLIT" --out "$OUT/ablation_scores.json" \
    2>&1 | tee "$LOG_DIR/score.log"

echo
echo "### Verdict ###"
python - "$OUT/ablation_scores.json" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
if not p.exists():
    print("  no scores written"); raise SystemExit(0)
d = json.loads(p.read_text())
base = d.get("none", {}).get("margin")
FLOOR = 0.002          # measured unconditional (b0) MARGIN at n=512
print(f"  {'lane':10s} {'MARGIN':>8} {'audio_r1':>9} {'retained':>9}")
for m in ("none", "reverse", "shuffle", "lowpass", "noise"):
    if m not in d: continue
    v = d[m]
    ret = "" if base in (None, 0) else f"{(v['margin']-FLOOR)/(base-FLOOR)*100:>8.0f}%"
    print(f"  {m:10s} {v['margin']:>8.4f} {v['audio_r1']:>9.4f} {ret:>9}")
print()
print("  'retained' = fraction of the clean model's above-floor MARGIN that survives.")
print("  reverse near 0% => the model IS using language (the claim the paper rests on).")
print("  reverse near 100% => it is reading acoustic surface, and the claim must be rewritten.")
PY

echo
echo "results: $OUT"
echo "OK"
