#!/usr/bin/env bash
# TWO OPERATING POINTS THAT WERE NEVER ACTUALLY TESTED. Generation only, no training, ~30 min.
#
# Both were closed on reasoning that the independent audit (2026-09-07) showed does not hold.
#
# 1. CFG x GREEDY WAS NEVER RUN. Job 442847 swept w in {1.0, 1.5, 2.5, 4.0} but every guided
#    lane used top_k=0 -- sampling. Its only greedy lane was fixed at w=1.0. The project moved
#    to greedy hours later and never revisited it. At w=1.5 under SAMPLING, guidance roughly
#    doubled scene control at the first two words with clarity intact; whether that survives a
#    deterministic decoder is unknown and costs one generation lane to find out.
#
# 2. top_p=0.7 WAS CLOSED WITHOUT ANYONE LISTENING TO IT. Job 443737 concluded clarity and
#    variety are one axis with no free point, which is true on the numbers:
#
#        lane        inter_asr_dist   top-4 opening share   family collapse gap
#        greedy          0.3688             46.1%                 +0.116
#        top_p 0.7       0.4720             11.7%                 +0.010
#        reference          --              10.9%                    --
#
#    But top_p 0.7 is the ONLY lane that reproduces human opening diversity, and greedy's stock
#    phrases are exactly what the user described on 2026-09-07 as "every speech seems
#    repetitive". The trade was decided on clarity alone, by a metric that has twice disagreed
#    with the ear. Nobody has heard it.
#
# ======================= EXPECTED RESULTS (decided before running) =======================
# THE DELIVERABLE IS THE LISTENING BUNDLE. The numbers here are already known for two of the
# five lanes; what is not known is which one a listener prefers.
#
#   CFG x greedy (w=1.5, w=2.5 against w=1.0):
#     SUCCESS -- w=1.5 improves on greedy's 0.3688 paired and length-adjusted, without losing
#         lexical diversity. Free: one flag, no retraining, and it becomes the operating point.
#     FAILURE -- flat or worse. Guidance needs the stochasticity it was tuned under, which is
#         itself worth knowing since it explains why the original sweep looked promising.
#     Watch w=2.5: under sampling it destroyed the audio (0.46 -> 0.15 -> 0.06 by word three).
#         If it does NOT collapse under greedy, guidance and sampling interact and the earlier
#         ceiling on w was an artifact of the decoder.
#
#   top_p 0.7 vs greedy: a pure listening question. If the varied-but-muddier lane is preferred,
#     inter_asr_dist has been mis-ranking the trade all along and the operating point should
#     change. If greedy is preferred, the stock phrases are the price of intelligibility and
#     that is settled rather than assumed.
#
#   PAIRED AND LENGTH-ADJUSTED, via compare_arms.py, which was validated by reproducing the
#   audit's numbers on job 443751's data to four decimals. inter_asr_dist rises ~0.009 per
#   generated word, so any raw win must be checked against the length-adjusted column before
#   it counts.
#
#   NOT AN OUTCOME on its own: a raw inter_asr_dist win. That is how the coarse-to-fine result
#   was overstated 3x -- the model was simply saying less.
# =========================================================================================
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_operating_point.sbatch
#SBATCH --job-name=salm-a2s-oppoint
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
CTC_CKPT="$OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt"
RENDERER="$RUNS_ROOT/a2s_audioldm_clapseq_xformer/rladder_20260906_1654/checkpoints/checkpoint_ep119.pt"
LM="$OUTPUTS_ROOT/content_lm/dedup_k${K}/checkpoints/checkpoint_best_loss.pt"
STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"
OUT_ROOT="$OUTPUTS_ROOT/generations/oppoint_${STAMP}"

for F in "$CTC_CKPT" "$RENDERER" "$LM"; do
    [[ -f "$F" ]] || { echo "MISSING: $F" >&2; exit 1; }
done
FREE=$(df -Pm /scratch/l/lovenya | awk 'NR==2{print $4}')
echo "free space: ${FREE} MB (need >= 2000 for ~613 MB of wavs)"
[[ "$FREE" -ge 2000 ]] || { echo "ABORT: only ${FREE} MB free" >&2; exit 1; }

salm_print_header "OPERATING POINT: CFG x greedy, and top_p 0.7 -- neither ever tested"

# spec: topk|topp|cfg|name.  CFG needs a batch of >= 2 (its unconditional branch is a rolled
# batch), which generation satisfies; NGEN=128 is far above that.
LANES="$OUT_ROOT/lanes.txt"; mkdir -p "$OUT_ROOT"; : > "$LANES"
echo "1|0.0|1.0|greedy_baseline"  >> "$LANES"
echo "1|0.0|1.5|greedy_cfg1.5"    >> "$LANES"
echo "1|0.0|2.5|greedy_cfg2.5"    >> "$LANES"
echo "0|0.7|1.0|top_p0.7"         >> "$LANES"
echo "0|0.85|1.0|top_p0.85"       >> "$LANES"
echo "planned lanes: $(wc -l < "$LANES")"

run_lane() {
    local gpu="$1" spec="$2"
    IFS='|' read -r topk topp cfg name <<< "$spec"
    local d="$OUT_ROOT/$name"; mkdir -p "$d"
    echo "[gpu$gpu] $name (top_k=$topk top_p=$topp cfg=$cfg)"
    CUDA_VISIBLE_DEVICES="$gpu" python -m salm.generate --mode a2s --direction a2s \
        --checkpoint "$RENDERER" --content-source codebook_lm --content-lm-checkpoint "$LM" \
        --content-lm-temperature 1.0 --content-lm-top-k "$topk" --content-lm-top-p "$topp" \
        --content-lm-cfg "$cfg" --ctc-checkpoint "$CTC_CKPT" \
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
SCORES="$OUTPUTS_ROOT/oppoint_intel_${STAMP}.json"
# shellcheck disable=SC2086
python scripts/a2s/score_intelligibility.py --dirs $DIRS --ctc-checkpoint "$CTC_CKPT" \
    --device cuda --out "$SCORES" \
    || { echo "SCORING FAILED -- generations intact under $OUT_ROOT" >&2; exit 1; }

echo
echo "==================== PAIRED vs GREEDY, LENGTH-ADJUSTED ===================="
python scripts/a2s/compare_arms.py --scores "$SCORES" --control greedy_baseline \
    --treatment greedy_cfg1.5 greedy_cfg2.5 top_p0.7 top_p0.85 \
    --out "$OUTPUTS_ROOT/oppoint_compare_${STAMP}.json" || true

echo
echo "==================== VARIETY ===================="
GROUP_ARGS=(); DET=()
while IFS='|' read -r _tk _tp _cfg name; do
    [[ -f "$OUT_ROOT/$name/manifest.jsonl" ]] || continue
    GROUP_ARGS+=(--group "$name" "$OUT_ROOT/$name"); DET+=("$name")
done < "$LANES"
python scripts/a2s/score_plan_horizon.py "${GROUP_ARGS[@]}" --deterministic "${DET[@]}" \
    --ctc-checkpoint "$CTC_CKPT" --device cuda \
    --out "$OUTPUTS_ROOT/oppoint_variety_${STAMP}.json" \
    || echo "VARIETY SCORING FAILED -- generations intact" >&2

python scripts/a2s/build_listening_bundle.py --gen-root "$OUT_ROOT" \
    --out "$OUTPUTS_ROOT/listening_bundles/oppoint_${STAMP}" --max-clips 10 --require-all-lanes || true

echo
echo "=========================================================="
echo "THE DELIVERABLE IS THE BUNDLE -- two of these five lanes already have known numbers;"
echo "what is unknown is which one a listener prefers."
echo "listen: $OUTPUTS_ROOT/listening_bundles/oppoint_${STAMP}"
echo "  greedy_baseline  46% stock openings, clearest"
echo "  top_p0.7         12% stock openings (human is 11%), muddier"
echo "Check the length-adjusted column before treating any raw win as real."
echo "=========================================================="
echo "OK"
