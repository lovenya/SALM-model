#!/usr/bin/env bash
# GREEDY AND THE TEMPERATURE LADDER, all three codebooks. Generation only, no retraining.
#
# WHY. The user listened to job 442847's lanes and reported greedy as clearly the most coherent
# of anything heard so far. That is a listening result, not a metric result -- no number in
# 442847 said greedy was better, because none of the three metrics built for this project
# measures within-utterance coherence, which is the thing being described.
#
# It also contradicts a decision baked into this codebase since the codebook LM was written.
# ContentLM.sample carried the comment "Sampling, never argmax: argmax walks straight back into
# the conditional mean that the discrete formulation exists to escape." That reasoning is sound
# for CONTINUOUS content -- the mean of several plausible mel trajectories is off-manifold, and
# that is exactly why A2S regression failed. It does NOT transfer to discrete tokens: argmax
# over a categorical distribution returns the most likely SEQUENCE, which is a real utterance,
# not an average of utterances. The argument was carried across a boundary where it stops
# holding, and every generation this project has produced has been sampled as a result.
#
# Greedy costs nothing to test: one flag, no retraining, ~6 minutes.
#
# ======================= EXPECTED RESULTS (decided before running) =======================
# THE VERDICT IS BY EAR. That is not a cop-out, it is the honest position after 442814 and
# 442847: inter_asr_dist rewards saying something clearly over saying the right thing,
# cross-seed agreement measures determinism (which greedy trivially maximises and which a
# one-to-many task should not have anyway), and reference WER is bounded at f1 0.16 even from
# a ground-truth caption. The numbers below bound the claim; they cannot make it.
#
#   THE ONE NUMBER THAT CAN VETO GREEDY: cross_clip_diversity, new in score_plan_horizon.py.
#   A deterministic decoder maximises coherence by always taking the most likely continuation,
#   and the most likely continuation given "a vehicle engine" may be the SAME SENTENCE for
#   every vehicle clip. That sounds excellent one clip at a time and is worthless -- one
#   utterance per scene TYPE, not per scene.
#     Today's sampled lanes measure 0.950-0.969 (near the 1.0 ceiling), so:
#     >= ~0.85  greedy keeps per-clip specificity. Coherence is a real gain.
#     ~0.5-0.85 partial collapse. Families of scenes share a sentence. Low temperature
#               (0.3/0.5) is then the right answer rather than pure greedy.
#     <= ~0.4   collapsed. Greedy is a mirage; discard it however good one clip sounds.
#   The metric was validated on CPU: identical transcripts score 0.000, disjoint score 1.000,
#   and same-clip-different-seed pairs are excluded so they cannot inflate it.
#
#   SUPPORTING, not deciding: inter_asr_dist and real_word_rate place each lane on the existing
#   scale (real speech 0.3146, VAE ceiling 0.3722, zero-content floor ~0.967). Expect greedy to
#   improve inter_asr_dist -- argmax picks high-confidence codes, which is exactly what makes
#   two ASRs agree. Do NOT read that as confirming coherence; it is close to tautological.
#
#   THE TEMPERATURE LADDER exists because greedy and sampling are not the only options. If
#   greedy wins on coherence but loses on diversity, the answer is somewhere at 0.3-0.7 and
#   this job locates it rather than requiring another queue wait.
#
#   NOT MEASURED HERE: cross-seed agreement. Each group is a single lane, so there are no seed
#   pairs; the scorer is told every group is deterministic so it prints n/a instead of a
#   misleading zero. 442847 already measured that axis.
# =========================================================================================
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_greedy_temp_sweep.sbatch
#SBATCH --job-name=salm-a2s-greedy-temp
#SBATCH --account=aip-csubakan
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

KS="${KS:-512 1024 2048}"
TEMPS="${TEMPS:-0.3 0.5 0.7 1.0}"
NGEN="${NGEN:-16}"
SPLIT="${SPLIT:-val}"
SEED="${SEED:-1337}"
CTC_CKPT="$OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt"
RENDERER="$RUNS_ROOT/a2s_audioldm_clapseq_xformer/content_concat_v4/checkpoints/checkpoint_best.pt"
STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"
OUT_ROOT="$OUTPUTS_ROOT/generations/greedy_temp_${STAMP}"

for F in "$CTC_CKPT" "$RENDERER"; do
    [[ -f "$F" ]] || { echo "MISSING: $F" >&2; exit 1; }
done
FREE=$(df -Pm /scratch/l/lovenya | awk 'NR==2{print $4}')
echo "free space: ${FREE} MB (need >= 1500)"
[[ "$FREE" -ge 1500 ]] || { echo "ABORT: only ${FREE} MB free" >&2; exit 1; }

salm_print_header "GREEDY + TEMPERATURE LADDER: K in {$KS}, greedy at two checkpoints"

LANES="$OUT_ROOT/lanes.txt"; mkdir -p "$OUT_ROOT"; : > "$LANES"
# spec: K|tag|topk|temp|name
for K in $KS; do
    # Greedy at BOTH checkpoints: best_loss is the val optimum, best (~ep90) was the strongest
    # lane on inter_asr_dist. 442814 showed overfitting costs nothing audible, so both are live.
    for TAG in checkpoint_best_loss checkpoint_best; do
        echo "$K|$TAG|1|1.0|k${K}__${TAG#checkpoint_}__greedy" >> "$LANES"
    done
    for T in $TEMPS; do
        echo "$K|checkpoint_best|0|$T|k${K}__best__t${T}" >> "$LANES"
    done
done
echo "planned lanes: $(wc -l < "$LANES")"

run_lane() {
    local gpu="$1" spec="$2"
    IFS='|' read -r k tag topk temp name <<< "$spec"
    local lm="$OUTPUTS_ROOT/content_lm/dedup_k${k}/checkpoints/${tag}.pt"
    [[ -f "$lm" ]] || { echo "  skip (absent): $name" >&2; return 0; }
    local out_dir="$OUT_ROOT/$name"
    mkdir -p "$out_dir"
    echo "[gpu$gpu] $name"
    CUDA_VISIBLE_DEVICES="$gpu" python -m salm.generate --mode a2s --direction a2s \
        --checkpoint "$RENDERER" --content-source codebook_lm \
        --content-lm-checkpoint "$lm" --content-lm-temperature "$temp" \
        --content-lm-top-k "$topk" \
        --ctc-checkpoint "$CTC_CKPT" --split "$SPLIT" --num-samples "$NGEN" \
        --start-index 0 --seed "$SEED" --out-dir "$out_dir" --copy-reference \
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

echo
echo "==================== DIVERSITY + CLARITY ===================="
# NOTE: GROUP_ARGS, never GROUPS -- GROUPS is a bash built-in array of the caller's Unix group
# ids, and assigning to it silently does nothing (cost job 442847 a near-miss).
GROUP_ARGS=(); DET=(); DIRS=""
while IFS='|' read -r _k _tag _topk _temp name; do
    [[ -f "$OUT_ROOT/$name/manifest.jsonl" ]] || continue
    GROUP_ARGS+=(--group "$name" "$OUT_ROOT/$name")
    DET+=("$name")
    DIRS="$DIRS $OUT_ROOT/$name"
done < "$LANES"
[[ ${#GROUP_ARGS[@]} -gt 0 ]] || { echo "no scorable lanes" >&2; exit 1; }

python scripts/a2s/score_plan_horizon.py "${GROUP_ARGS[@]}" \
    --deterministic "${DET[@]}" \
    --ctc-checkpoint "$CTC_CKPT" --device cuda \
    --out "$OUTPUTS_ROOT/greedy_temp_horizon_${STAMP}.json" \
    || echo "SCORING FAILED (horizon) -- generations intact, rescore separately" >&2

CTRL="$OUTPUTS_ROOT/generations/control_lanes_20260831_0735"
# shellcheck disable=SC2086
python scripts/a2s/score_intelligibility.py \
    --dirs $CTRL/real_speech $CTRL/vae_roundtrip $DIRS \
    --ctc-checkpoint "$CTC_CKPT" --device cuda \
    --out "$OUTPUTS_ROOT/greedy_temp_intel_${STAMP}.json" \
    || echo "SCORING FAILED (intelligibility) -- generations intact" >&2

echo
echo "==================== LISTENING BUNDLE ===================="
BUNDLE="$OUTPUTS_ROOT/listening_bundles/greedy_temp_${STAMP}"
python scripts/a2s/build_listening_bundle.py \
    --gen-root "$OUT_ROOT" --out "$BUNDLE" --max-clips 8 \
    || echo "BUNDLE FAILED -- lanes are intact under $OUT_ROOT" >&2

echo
echo "=========================================================="
echo "READ cross_clip_diversity FIRST. Sampled lanes measure 0.950-0.969 today."
echo "  A greedy lane below ~0.4 has collapsed to one sentence per scene TYPE and must be"
echo "  discarded no matter how good a single clip sounds. 0.5-0.85 means use temp 0.3-0.5."
echo "then listen: $BUNDLE"
echo "  rsync -avzP -e \"ssh -J rorqual\" \"tamia:$BUNDLE/\" ~/salm_listen/greedy_temp/"
echo "=========================================================="
echo "OK"
