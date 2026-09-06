#!/usr/bin/env bash
# NUCLEUS SAMPLING SWEEP. Generation only, no retraining.
#
# WHY. Greedy decoding produced the best result this project has recorded (job 443475: k2048 at
# best_loss, inter_asr_dist 0.3688 over 128 val clips against a VAE round-trip ceiling of
# 0.3722 and real speech at 0.3146). Reading those 128 transcripts showed what it costs:
#
#     four 4-word openings cover 46% of the clips
#     "hey can you hear" alone accounts for 39 of 128
#     four different engine scenes all produced variants of
#         "hold on tight, we're picking up speed"
#
# That is greedy by construction. Argmax takes the single most likely continuation, and at
# position 0 with no context that is just the most frequent opening in the training data. The
# model has a small stock of safe entry phrases and reaches for one every time.
#
# Nucleus sampling is the middle ground neither greedy nor temperature covers: keep the
# smallest set of codes reaching probability mass p and drop the rest. Temperature rescales the
# WHOLE distribution and leaves the long tail reachable -- which is why the 0.3/0.5/0.7/1.0
# ladder in 443379 was flat (0.484 / 0.445 / 0.471 / 0.452 for k2048). top_p removes the tail
# without flattening the head.
#
# IT HAD NEVER BEEN TESTED. `--content-lm-top-p` has existed as a flag the whole time but was
# wired only into the FRAME-LEVEL sampler; the dedup sample() had no top_p parameter, so
# passing it with a dedup checkpoint silently generated unguided audio. Fixed in this commit,
# along with a guard that refuses top_k and top_p together.
#
# ======================= EXPECTED RESULTS (decided before running) =======================
# This is a TRADE, not a search for a winner, and both sides are now measured.
#
#   CLARITY  -- inter_asr_dist. Anchors: real speech 0.3146, VAE ceiling 0.3722, zero floor
#               0.967. Greedy baseline 0.3688 at 128 clips.
#   VARIETY  -- opening_top4_share (new). Greedy baseline 46%. Validated against the manual
#               count on 443475's data: the metric reports 46% and "hey can you hear" x39,
#               matching what was found by hand.
#   FAMILY   -- family_collapse_gap (new): diversity among caption-similar pairs vs
#               caption-different pairs. Greedy baseline +0.116 for k2048 (0.794 vs 0.909).
#               `cross_clip_diversity` alone reads 0.909 and hides this, which is why it is
#               no longer reported on its own.
#
#   SUCCESS -- some p holds inter_asr_dist at or under ~0.40 while cutting opening_top4_share
#       to roughly 20% or below and shrinking the collapse gap. That p becomes the shipping
#       configuration, and it costs one flag.
#   FAILURE, MODE 1 -- clarity degrades in lockstep with variety at every p, tracing a straight
#       line between greedy and t=1.0. Then the two are a single axis with no free lunch, the
#       stock phrases are the price of intelligibility in this model, and the fix has to come
#       from the model rather than the decoder.
#   FAILURE, MODE 2 -- variety does not move at all even at p=0.7. That would mean the stock
#       openings are not a decoding artifact but genuinely dominate the posterior at position
#       0 -- the model really believes "hey can you hear" is the answer to most scenes. That is
#       a training-distribution finding, not a sampling one, and points at how the content
#       codes were built rather than at the decoder.
#
#   WATCH FOR: p too low collapses toward greedy (identical numbers), p too high toward t=1.0.
#   If p=0.7 and greedy agree to three decimals the sweep did not span anything and the range
#   needs widening, not interpreting.
#
#   NOT AN OUTCOME: inter_asr_dist alone. It rewards clear articulation, and the whole point
#   here is that greedy already maximises it by saying safe things. Read all three together.
# =========================================================================================
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_nucleus_sweep.sbatch
#SBATCH --job-name=salm-a2s-nucleus
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
TAG="${TAG:-checkpoint_best_loss}"
PS="${PS:-0.7 0.85 0.95}"
NGEN="${NGEN:-128}"
SPLIT="${SPLIT:-val}"
CTC_CKPT="$OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt"
RENDERER="$RUNS_ROOT/a2s_audioldm_clapseq_xformer/content_concat_v4/checkpoints/checkpoint_best.pt"
LM="$OUTPUTS_ROOT/content_lm/dedup_k${K}/checkpoints/${TAG}.pt"
STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"
OUT_ROOT="$OUTPUTS_ROOT/generations/nucleus_${STAMP}"

for F in "$CTC_CKPT" "$RENDERER" "$LM"; do
    [[ -f "$F" ]] || { echo "MISSING: $F" >&2; exit 1; }
done
FREE=$(df -Pm /scratch/l/lovenya | awk 'NR==2{print $4}')
echo "free space: ${FREE} MB (need >= 2500 for ~613 MB of wavs plus headroom)"
[[ "$FREE" -ge 2500 ]] || { echo "ABORT: only ${FREE} MB free" >&2; exit 1; }

salm_print_header "NUCLEUS SWEEP: K=$K $TAG, p in {$PS} vs greedy and t=1.0"

# spec: topk|topp|temp|name   -- greedy and t=1.0 are the two endpoints the sweep sits between
LANES="$OUT_ROOT/lanes.txt"; mkdir -p "$OUT_ROOT"; : > "$LANES"
echo "1|0.0|1.0|greedy"   >> "$LANES"
for P in $PS; do echo "0|$P|1.0|top_p${P}" >> "$LANES"; done
echo "0|0.0|1.0|temp1.0"  >> "$LANES"
echo "planned lanes: $(wc -l < "$LANES")"

run_lane() {
    local gpu="$1" spec="$2"
    IFS='|' read -r topk topp temp name <<< "$spec"
    local out_dir="$OUT_ROOT/$name"
    mkdir -p "$out_dir"
    echo "[gpu$gpu] $name (top_k=$topk top_p=$topp temp=$temp)"
    CUDA_VISIBLE_DEVICES="$gpu" python -m salm.generate --mode a2s --direction a2s \
        --checkpoint "$RENDERER" --content-source codebook_lm \
        --content-lm-checkpoint "$LM" --content-lm-temperature "$temp" \
        --content-lm-top-k "$topk" --content-lm-top-p "$topp" \
        --ctc-checkpoint "$CTC_CKPT" --split "$SPLIT" --num-samples "$NGEN" \
        --start-index 0 --seed 1337 --out-dir "$out_dir" --copy-reference \
        > "$out_dir.log" 2>&1 || echo "  FAILED: $name" >&2
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

DIRS=""
GROUP_ARGS=(); DET=()
while IFS='|' read -r _tk _tp _t name; do
    [[ -f "$OUT_ROOT/$name/manifest.jsonl" ]] || continue
    DIRS="$DIRS $OUT_ROOT/$name"
    GROUP_ARGS+=(--group "$name" "$OUT_ROOT/$name"); DET+=("$name")
done < "$LANES"
[[ -n "$DIRS" ]] || { echo "no lanes produced a manifest" >&2; exit 1; }

echo
echo "==================== CLARITY ===================="
CTRL="$OUTPUTS_ROOT/generations/control_lanes_20260831_0735"
SCORES="$OUTPUTS_ROOT/nucleus_intel_${STAMP}.json"
# shellcheck disable=SC2086
python scripts/a2s/score_intelligibility.py --dirs $CTRL/real_speech $CTRL/vae_roundtrip $DIRS \
    --ctc-checkpoint "$CTC_CKPT" --device cuda --out "$SCORES" \
    || echo "SCORING FAILED (clarity) -- generations intact" >&2

echo
echo "==================== VARIETY + FAMILY COLLAPSE ===================="
python scripts/a2s/score_plan_horizon.py "${GROUP_ARGS[@]}" --deterministic "${DET[@]}" \
    --ctc-checkpoint "$CTC_CKPT" --device cuda \
    --out "$OUTPUTS_ROOT/nucleus_variety_${STAMP}.json" \
    || echo "SCORING FAILED (variety) -- generations intact" >&2

python scripts/a2s/export_transcript_review.py --scores "$SCORES" --gen-root "$OUT_ROOT" \
    --limit 200 --out "$OUTPUTS_ROOT/transcript_review_nucleus_${STAMP}.md" || true
python scripts/a2s/build_listening_bundle.py --gen-root "$OUT_ROOT" \
    --out "$OUTPUTS_ROOT/listening_bundles/nucleus_${STAMP}" --max-clips 8 || true

echo
echo "=========================================================="
echo "READ ALL THREE TOGETHER -- clarity alone is what greedy already maximises by playing safe:"
echo "  inter_asr_dist       greedy 0.3688 | real speech 0.3146 | VAE ceiling 0.3722"
echo "  opening_top4_share   greedy 46%    | lower is more varied"
echo "  family_collapse_gap  greedy +0.116 | lower means similar scenes still differ"
echo "If p=0.7 and greedy agree to three decimals, the sweep spanned nothing -- widen it."
echo "listen: $OUTPUTS_ROOT/listening_bundles/nucleus_${STAMP}"
echo "=========================================================="
echo "OK"
