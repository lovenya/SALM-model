#!/usr/bin/env bash
# LEARNED STOPPING: train with targets trimmed at the speech boundary so the model emits EOS
# itself, instead of being cut off. 8 arms, both waves filling all 4 GPUs.
#
# WHY THIS AND NOT JOB 448455. That job clamped the DECODE budget -- it cut the model off at
# frame N whether or not it had finished, so the tail was severed rather than completed. It made
# things worse (flat 0.3404 -> 0.4216 at 111 frames, all CIs excluding zero) which is
# unsurprising and answers nothing: chopping a sentence hurts. Lovenya caught that forceful
# truncation and variable-length output are different mechanisms and that the two had been
# conflated.
#
# `--truncate-speech` trims the TRAINING TARGETS at each clip's speech boundary. The model never
# sees the wordless tail as something to fill, so it learns that utterances end and where. At
# generation it stops on its own, having finished. No renderer change: `expand_sequence` already
# holds the last code past EOS, exactly what the renderer sees today when a sequence ends early.
#
# Measured on the real k2048 cache: speech ends at frame 116.8/256 on average, sequences shorten
# 151.4 -> 100.6 tokens, and the plan-horizon shortfall falls 7.6x -> 5.0x.
#
# ======================= EXPECTED RESULTS (decided before running) =======================
# 8 arms = 2 models x {off, on} x 2 seeds, one frozen renderer (rladder ep119), greedy decoding,
# 128 val clips each. Two questions in one node:
#
#   FLAT arms  -- the clean test. One variable changes; does learned stopping help at all?
#   HIER arms  -- the practical question. hier stride-12 is BEST_RESULTS.md #1 and the reason it
#                 wins is currently OPEN. If truncation helps it too, hier+truncation is the new
#                 candidate; if truncation helps FLAT up to hier's level, the hierarchy is
#                 redundant and #1's advantage was learned stopping all along.
#
# PRIMARY: paired, length-adjusted inter_asr_dist, on vs off at the SAME seed and model, via
# compare_arms.py (validated by reproducing an audit's numbers to four decimals). Two seed
# pairings per model, because the plan vector showed a single seed proves nothing -- there, one
# seed read -0.077 and three seeds averaged -0.028 with sd 0.055.
#
#   SUCCESS -- truncate-on beats off in both seed pairings for a model, mean raw <= -0.03 and the
#       length-adjusted delta also negative. Then learned stopping is real, 448455's failure was
#       purely an artifact of cutting mid-utterance, and "coherent completion" rather than
#       "length" is the mechanism.
#   FAILURE -- deltas inside +/-0.02, or positive. Then stopping is not the mechanism either, and
#       the reason #1 wins stays genuinely unexplained -- which matters, because it would mean
#       the best result on record is not understood at all.
#   AMBIGUOUS -- raw win, length-adjusted near zero. Then it only shortened the output and
#       inter_asr_dist's ~0.009-per-word bias is doing the work, not the model.
#
# READ FIRST, BEFORE ANY DELTA -- did the model actually learn to stop?
#   1. The trainer prints "truncated at the speech boundary -- mean length X -> Y tokens". If Y
#      is not near 100 for the ON arms, truncation did not apply and the arm is void.
#   2. Generated word count. The ON arms should fall toward the human 13.3 BY CHOICE. If they
#      still emit ~17 words, the model did not learn the EOS position and there is nothing to
#      measure.
#   3. val_code_loss is NOT comparable between ON and OFF: they score different-length targets
#      over the same head. Read the direction of the audio delta, never the loss difference.
# =========================================================================================
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_truncate.sbatch
#SBATCH --job-name=salm-a2s-trunc
#SBATCH --account=aip-ravanelm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

K="${K:-2048}"
EPOCHS="${EPOCHS:-100}"
SEEDS="${SEEDS:-0 1}"
MAX_MINUTES="${MAX_MINUTES:-75}"
NGEN="${NGEN:-128}"
CB="$OUTPUTS_ROOT/content_codebook/codebook_k${K}.pt"
CTC_CKPT="$OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt"
RENDERER="$RUNS_ROOT/a2s_audioldm_clapseq_xformer/rladder_20260906_1654/checkpoints/checkpoint_ep119.pt"
STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"
OUT_ROOT="$OUTPUTS_ROOT/generations/trunc_${STAMP}"

for F in "$CB" "$CTC_CKPT" "$RENDERER"; do
    [[ -f "$F" ]] || { echo "MISSING: $F" >&2; exit 1; }
done
FREE=$(df -Pm /scratch/l/lovenya | awk 'NR==2{print $4}')
echo "free space: ${FREE} MB (need >= 4000: ~1.7 GB checkpoints + ~1 GB wavs)"
[[ "$FREE" -ge 4000 ]] || { echo "ABORT: only ${FREE} MB free" >&2; exit 1; }

# name|model|trunc|seed  -- 8 arms, two waves of 4, so the node is never idle
ARMS=()
for S in $SEEDS; do
  for M in flat hier; do
    for T in off on; do
      ARMS+=("trunc_${M}_${T}_s${S}|${M}|${T}|${S}")
    done
  done
done
for a in "${ARMS[@]}"; do
    d="$OUTPUTS_ROOT/content_lm/${a%%|*}"
    [[ -e "$d" ]] && { echo "ABORT: $d exists -- refusing to overwrite" >&2; exit 1; }
done

salm_print_header "LEARNED STOPPING: ${#ARMS[@]} arms (flat+hier x off/on x ${SEEDS// /,}), all 4 GPUs"
echo "target: hier stride-12 uncapped = 0.2949 (BEST_RESULTS.md #1, reason currently OPEN)"
echo "job 448455 already showed FORCEFUL truncation hurts; this tests LEARNED stopping"

echo
echo "==================== 1. TRAIN (8 arms, two waves of 4) ===================="
i=0
for a in "${ARMS[@]}"; do
    IFS='|' read -r NAME MODEL TRUNC SEED <<< "$a"
    SCRIPT="scripts/a2s/train_content_lm_dedup.py"
    EXTRA=""
    [[ "$MODEL" == "hier" ]] && { SCRIPT="scripts/a2s/train_content_lm_hier.py"; EXTRA="--stride 12"; }
    [[ "$TRUNC" == "on" ]] && EXTRA="$EXTRA --truncate-speech"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$((i % 4)) python -u "$SCRIPT" \
        --codebook "$CB" --run-name "$NAME" --seed "$SEED" \
        --epochs "$EPOCHS" --batch-size 64 \
        --d-model 512 --layers 6 --heads 8 --dropout 0.1 --weight-decay 0.01 \
        --max-run 32 --save-every 0 --max-minutes "$MAX_MINUTES" \
        $EXTRA > "$OUTPUTS_ROOT/logs/slurm/${NAME}_${STAMP}.log" 2>&1 &
    i=$((i + 1))
    [[ $((i % 4)) -eq 0 ]] && wait
done
wait
echo "training done"

echo
echo "==================== 2. DID TRUNCATION ACTUALLY APPLY? ===================="
# Void-arm check, read before any delta. If the ON arms did not shorten, there is nothing here.
for a in "${ARMS[@]}"; do
    NAME="${a%%|*}"
    line=$(grep -h "truncated at the speech boundary" \
           "$OUTPUTS_ROOT/logs/slurm/${NAME}_${STAMP}.log" 2>/dev/null | head -1)
    printf "  %-24s %s\n" "$NAME" "${line:-<no truncation applied>}"
done

echo
echo "==================== 3. GENERATE (greedy, 128 clips, frozen renderer) ===================="
i=0
for a in "${ARMS[@]}"; do
    NAME="${a%%|*}"
    LM="$OUTPUTS_ROOT/content_lm/$NAME/checkpoints/checkpoint_best_loss.pt"
    [[ -f "$LM" ]] || { echo "  skip (absent): $NAME" >&2; continue; }
    d="$OUT_ROOT/$NAME"; mkdir -p "$d"
    CUDA_VISIBLE_DEVICES=$((i % 4)) python -m salm.generate --mode a2s --direction a2s \
        --checkpoint "$RENDERER" --content-source codebook_lm --content-lm-checkpoint "$LM" \
        --content-lm-temperature 1.0 --content-lm-top-k 1 --ctc-checkpoint "$CTC_CKPT" \
        --split val --num-samples "$NGEN" --start-index 0 --seed 1337 \
        --out-dir "$d" --copy-reference > "$d.log" 2>&1 &
    i=$((i + 1))
    [[ $((i % 4)) -eq 0 ]] && wait
done
wait

echo
echo "==================== 4. SCORE ===================="
CTRL="$OUTPUTS_ROOT/generations/control_lanes_20260831_0735"
DIRS="$CTRL/real_speech $CTRL/vae_roundtrip"
for d in "$OUT_ROOT"/*/; do [[ -f "$d/manifest.jsonl" ]] && DIRS="$DIRS ${d%/}"; done
SCORES="$OUTPUTS_ROOT/trunc_intel_${STAMP}.json"
# shellcheck disable=SC2086
python scripts/a2s/score_intelligibility.py --dirs $DIRS --ctc-checkpoint "$CTC_CKPT" \
    --device cuda --out "$SCORES" \
    || { echo "SCORING FAILED -- generations intact under $OUT_ROOT" >&2; exit 1; }

echo
echo "==================== 5. PAIRED: ON vs OFF, same model and seed ===================="
for S in $SEEDS; do
  for M in flat hier; do
    echo; echo "  --- $M, seed $S ---"
    python scripts/a2s/compare_arms.py --scores "$SCORES" \
        --control "trunc_${M}_off_s${S}" --treatment "trunc_${M}_on_s${S}" \
        --out "$OUTPUTS_ROOT/trunc_cmp_${M}_s${S}_${STAMP}.json" || true
  done
done

echo
echo "==================== 6. DID IT CHOOSE TO STOP? (words vs the human 13.3) ===================="
python - "$SCORES" <<'PY' || true
import json, sys, numpy as np
sys.path.insert(0, "src")
from salm.a2s.wer import normalize_text
d = json.loads(open(sys.argv[1]).read())
print(f"  {'lane':24s} {'dist':>7s} {'words':>7s} {'vs 13.3':>8s}")
for k in sorted(d):
    v = d[k]
    if "error" in v: continue
    w = np.mean([len(normalize_text(e["whisper"]).split()) for e in v["examples"]])
    print(f"  {k.split('/')[-1]:24s} {v['inter_asr_distance_mean']:7.4f} {w:7.1f} {w/13.3:8.2f}")
print()
print("  ON arms should fall toward 13.3 BY CHOICE. Still ~17 words = the EOS position was")
print("  not learned and the arm measures nothing. Job 448455 forced 13.7 and got WORSE audio.")
PY

python scripts/a2s/build_listening_bundle.py --gen-root "$OUT_ROOT" \
    --out "$OUTPUTS_ROOT/listening_bundles/trunc_${STAMP}" --max-clips 10 --require-all-lanes || true

echo
echo "=========================================================="
echo "READ IN THIS ORDER: section 2 (did truncation apply), section 6 (did it choose to stop),"
echo "then section 5's LENGTH-ADJUSTED column. A raw win with len-adj near zero is the metric's"
echo "own ~0.009-per-word bias, not the model."
echo "listen: $OUTPUTS_ROOT/listening_bundles/trunc_${STAMP}"
echo "=========================================================="
echo "OK"
