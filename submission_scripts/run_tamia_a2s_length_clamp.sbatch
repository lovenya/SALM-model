#!/usr/bin/env bash
# LENGTH CLAMP: does stopping earlier reproduce the best result, with no hierarchy and no
# retraining? Generation only, ~25 min.
#
# THE MEASURED CLAIM THIS TESTS. The best audio on record (BEST_RESULTS.md #1, hier stride-12)
# wins for exactly one reason, and it is not planning:
#
#     lane                    words   vs human 13.3   utterances > 1.5x reference
#     hier stride-12 (#1)      11.1        0.83                11%
#     flat dedup LM  (#2)      17.6        1.32                42%
#
# Once length is controlled, hier stride-12 is actually +0.046 WORSE than its renderer-matched
# control. All of its advantage is that it stops talking sooner. The hierarchy contributed
# nothing, and its own ordered stride prediction failed (horizons flat at 1 across 6/8/12).
#
# If that is right, clamping the FLAT model's generation length should reproduce the gain
# directly -- no hierarchy, no retraining, one flag. Speech occupies a median 111 of 256
# frames, so the corpus prior is the obvious clamp.
#
# WHY CLAMP TO A CONSTANT RATHER THAN PREDICT THE LENGTH. The independent audit fitted a ridge
# from the pooled CLAP ambient to speech extent over all 17,596 train clips: r = +0.22,
# R^2 = 0.046, MAE 1.38 s against a 1.42 s median baseline. The scene barely predicts how long
# someone talks. Clamp to the prior; do not try to be clever.
#
# ======================= EXPECTED RESULTS (decided before running) =======================
# Two models x four budgets, everything else identical, one frozen renderer.
#
#   PRIMARY: paired, length-adjusted inter_asr_dist against each model's own 256-frame lane,
#   via compare_arms.py (validated by reproducing the audit's numbers to four decimals).
#
#   SUCCESS -- the flat model at ~111 frames reaches or beats hier stride-12's 0.2970, and its
#       word count drops toward the human 13.3. Then length control IS the mechanism, it is
#       available for free on any model, and the hierarchy can be retired.
#   PARTIAL -- clamping helps but does not close the gap to 0.2970. Then length is part of the
#       story and something in the hier model contributes the rest, which would be the first
#       evidence that the hierarchy does anything at all.
#   FAILURE -- clamping does not help or hurts. Then the length correlation was not causal, and
#       the reason hier stride-12 wins is still unexplained -- which would matter, because
#       BEST_RESULTS.md currently states length as the reason.
#
#   A RAW WIN IS NOT ENOUGH HERE, and this job is unusually exposed to it: inter_asr_dist rises
#   ~0.009 per generated word, so ANY clamp mechanically improves the raw score by shortening
#   the output. The length-adjusted column is the real result. If raw improves and
#   length-adjusted does not, the clamp bought nothing beyond the metric's own bias -- and that
#   is exactly how the coarse-to-fine result came to be overstated threefold.
#
#   ALSO WATCH: clipped-off speech. At 90 frames (3.6 s) the clamp may cut mid-utterance, which
#   would sound abrupt rather than better. Word count falling well below the human 13.3 is the
#   warning sign, and the ear is the check.
# =========================================================================================
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_length_clamp.sbatch
#SBATCH --job-name=salm-a2s-clamp
#SBATCH --account=aip-csubakan
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

K="${K:-2048}"
NGEN="${NGEN:-128}"
BUDGETS="${BUDGETS:-0 150 111 90}"     # 0 = uncapped control; 111 = corpus median extent
CTC_CKPT="$OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt"
RENDERER="$RUNS_ROOT/a2s_audioldm_clapseq_xformer/rladder_20260906_1654/checkpoints/checkpoint_ep119.pt"
FLAT="$OUTPUTS_ROOT/content_lm/dedup_k${K}/checkpoints/checkpoint_best_loss.pt"
HIER="$OUTPUTS_ROOT/content_lm/hier_k${K}_s12/checkpoints/checkpoint_best_loss.pt"
STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"
OUT_ROOT="$OUTPUTS_ROOT/generations/clamp_${STAMP}"

for F in "$CTC_CKPT" "$RENDERER" "$FLAT" "$HIER"; do
    [[ -f "$F" ]] || { echo "MISSING: $F" >&2; exit 1; }
done
FREE=$(df -Pm /scratch/l/lovenya | awk 'NR==2{print $4}')
echo "free space: ${FREE} MB (need >= 2500)"
[[ "$FREE" -ge 2500 ]] || { echo "ABORT: only ${FREE} MB free" >&2; exit 1; }

salm_print_header "LENGTH CLAMP: flat vs hier, budgets {$BUDGETS} frames of 256"
echo "target to beat: hier stride-12 uncapped = 0.2970 (BEST_RESULTS.md #1)"
echo "speech occupies a median 111/256 frames; the human reference averages 13.3 words"

LANES="$OUT_ROOT/lanes.txt"; mkdir -p "$OUT_ROOT"; : > "$LANES"
for M in flat hier; do
  for B in $BUDGETS; do
    echo "$M|$B|${M}_b${B}" >> "$LANES"
  done
done
echo "planned lanes: $(wc -l < "$LANES")"

run_lane() {
    local gpu="$1" spec="$2"
    IFS='|' read -r model budget name <<< "$spec"
    local lm="$FLAT"; [[ "$model" == "hier" ]] && lm="$HIER"
    local d="$OUT_ROOT/$name"; mkdir -p "$d"
    echo "[gpu$gpu] $name (budget $budget)"
    CUDA_VISIBLE_DEVICES="$gpu" python -m salm.generate --mode a2s --direction a2s \
        --checkpoint "$RENDERER" --content-source codebook_lm --content-lm-checkpoint "$lm" \
        --content-lm-temperature 1.0 --content-lm-top-k 1 \
        --content-lm-max-frames "$budget" --ctc-checkpoint "$CTC_CKPT" \
        --split val --num-samples "$NGEN" --start-index 0 --seed 1337 \
        --out-dir "$d" --copy-reference > "$d.log" 2>&1 || echo "  FAILED: $name" >&2
}

NGPU=4
for g in $(seq 0 $((NGPU - 1))); do
  ( i=0
    while IFS= read -r spec; do
      [[ $((i % NGPU)) -eq "$g" ]] && run_lane "$g" "$spec"
      i=$((i + 1))
    done < "$LANES" ) &
done
wait
echo "all lanes done"

echo
echo "==================== SCORE ===================="
CTRL="$OUTPUTS_ROOT/generations/control_lanes_20260831_0735"
DIRS="$CTRL/real_speech $CTRL/vae_roundtrip"
for d in "$OUT_ROOT"/*/; do [[ -f "$d/manifest.jsonl" ]] && DIRS="$DIRS ${d%/}"; done
SCORES="$OUTPUTS_ROOT/clamp_intel_${STAMP}.json"
# shellcheck disable=SC2086
python scripts/a2s/score_intelligibility.py --dirs $DIRS --ctc-checkpoint "$CTC_CKPT" \
    --device cuda --out "$SCORES" \
    || { echo "SCORING FAILED -- generations intact under $OUT_ROOT" >&2; exit 1; }

echo
echo "==================== PAIRED vs EACH MODEL'S OWN UNCAPPED LANE ===================="
for M in flat hier; do
    T=""
    for B in $BUDGETS; do [[ "$B" != "0" ]] && T="$T ${M}_b${B}"; done
    echo; echo "  --- $M ---"
    # shellcheck disable=SC2086
    python scripts/a2s/compare_arms.py --scores "$SCORES" --control "${M}_b0" \
        --treatment $T --out "$OUTPUTS_ROOT/clamp_compare_${M}_${STAMP}.json" || true
done

echo
echo "==================== WORD COUNT vs THE HUMAN 13.3 ===================="
python - "$SCORES" <<'PY' || true
import json, sys, numpy as np
sys.path.insert(0, "src")
from salm.a2s.wer import normalize_text
d = json.loads(open(sys.argv[1]).read())
print(f"  {'lane':14s} {'dist':>7s} {'words':>7s} {'vs 13.3':>8s}")
for k in sorted(d):
    v = d[k]
    if "error" in v: continue
    w = np.mean([len(normalize_text(e["whisper"]).split()) for e in v["examples"]])
    print(f"  {k.split('/')[-1]:14s} {v['inter_asr_distance_mean']:7.4f} {w:7.1f} {w/13.3:8.2f}")
PY

python scripts/a2s/build_listening_bundle.py --gen-root "$OUT_ROOT" \
    --out "$OUTPUTS_ROOT/listening_bundles/clamp_${STAMP}" --max-clips 10 --require-all-lanes || true

echo
echo "=========================================================="
echo "READ THE LENGTH-ADJUSTED COLUMN, not the raw one. inter_asr_dist rises ~0.009 per word,"
echo "so ANY clamp improves the raw score just by shortening the output -- that is exactly how"
echo "the coarse-to-fine result came to be overstated threefold."
echo "listen: $OUTPUTS_ROOT/listening_bundles/clamp_${STAMP}"
echo "=========================================================="
echo "OK"
