#!/usr/bin/env bash
# CLASSIFIER-FREE GUIDANCE on the content LM. Generation only -- nothing is retrained.
#
# WHY. Job 442814 moved the target. Three things it settled:
#   - overfitting is NOT the cause. Train and val score identically (ref_wer 1.29 vs 1.32,
#     span 1.06 vs 0.83 words), even at ~ep90 where train_code_loss is 0.21. Memorisation
#     buys nothing audible. val and test agree, so there is no split problem either.
#   - clarity is NOT the cause. Positional inter-ASR agreement on the strongest lane tracks
#     the REAL-SPEECH curve out to word 10 (0.62 0.75 0.38 0.44 0.50 0.50 0.40 0.53 0.47 0.62
#     against real speech's 0.88 0.81 0.75 0.44 0.56 0.56 0.47 0.53 0.43 0.54). The later
#     words are clearly articulated.
#   - DETERMINISM is. Six seeds on one fixed ambient clip agree 0.20 0.20 0.17 0.12 and then
#     0.00. A four-word horizon, then a cliff.
#
# So the model stops USING the ambient long before it stops speaking. Listening agrees: the
# user reports 3-4 sensible words followed by a group that "looks random" -- random being
# exactly what a free-running decoder produces once conditioning lapses.
#
# THE INTERVENTION. Classifier-free guidance, at generation time, no retraining:
#
#     log p  =  log p(mismatched scene)  +  w * [ log p(true scene) - log p(mismatched) ]
#
# Whatever the true ambient contributes is amplified at EVERY step, not only where the model
# was already inclined to use it. w = 1 reproduces the current model bit-for-bit.
#
# The unconditional branch is a ROLLED batch, not a zeroed memory -- the same reasoning
# `shuffled_memory` documents for the scene ablation: a zeroed memory is out of distribution,
# so extrapolating away from it would push the sequence off-manifold for reasons unrelated to
# the scene. Only the code head is guided; guiding duration risks run lengths that no longer
# fill the 256-frame canvas.
#
# ======================= EXPECTED RESULTS (decided before running) =======================
# Metric: scripts/a2s/score_plan_horizon.py, two curves per group.
#   cross-seed agreement -- same ambient, six noises. SCENE CONTROL. The headline.
#   inter-ASR agreement  -- Whisper vs CTC on the same audio. CLARITY. The guard rail.
# The scorer was validated against 442814's data on CPU and reproduces its hand-computed
# curve exactly (horizon 4 at `best`, 3 at `best_loss`), so a change here is the intervention.
#
#   SUCCESS -- cross-seed horizon extends past 4 (say 6-8) at some w, with the inter-ASR curve
#       still near real speech. Guidance is the fix, it costs one extra forward per step and
#       no retraining, and the next move is tuning w and porting it to the frame-level sampler.
#   FAILURE, MODE 1 -- cross-seed horizon is flat in w. The ambient's contribution is not
#       merely under-weighted, it is absent from the later positions: the model has learned a
#       short-range mapping and amplifying nothing yields nothing. That rules out a whole class
#       of decode-time fixes and points at the training objective (the LM is never asked to
#       hold the scene across the sequence) or at the unit (BPE over content codes -- already
#       written in src/salm/a2s/content_bpe.py and never run).
#   FAILURE, MODE 2 -- cross-seed rises but inter-ASR collapses. Guidance is buying determinism
#       by driving codes off-manifold; the audio degrades into consistent noise. Read as a
#       ceiling on w, not as success, and check the audio before believing any horizon number.
#
#   THE GREEDY CONTROL IS NOT OPTIONAL. top_k=1 fixes the content codes, so its six seeds
#   differ ONLY in diffusion and vocoder noise. Its cross-seed curve is therefore the CEILING
#   any guided lane could reach -- if the renderer itself only reproduces the same word 60% of
#   the time from identical codes, then 0.20 at position 1 is a third of the achievable
#   maximum, not a fifth of 1.0. Every horizon below must be read against it. Without this
#   lane the whole table is uncalibrated.
#
#   WHAT WOULD INVALIDATE THE JOB: if the w=1.0 group does not reproduce 442814's curve
#   (0.20 0.20 0.17 0.12 0.00) within noise, something changed in the sampler and no
#   comparison across w is admissible. Check that row first.
# =========================================================================================
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_cfg_horizon.sbatch
#SBATCH --job-name=salm-a2s-cfg-horizon
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

K="${K:-2048}"                      # the strongest LM lane on record (inter_asr_dist 0.452)
TAG="${TAG:-checkpoint_best}"
WS="${WS:-1.0 1.5 2.5 4.0}"
SEEDS="${SEEDS:-11 22 33 44 55 66}"  # 6 seeds -> 15 pairs per position, well over MIN_PAIRS
NGEN="${NGEN:-8}"
SPLIT="${SPLIT:-val}"
CTC_CKPT="$OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt"
RENDERER="$RUNS_ROOT/a2s_audioldm_clapseq_xformer/content_concat_v4/checkpoints/checkpoint_best.pt"
LM_CKPT="$OUTPUTS_ROOT/content_lm/dedup_k${K}/checkpoints/${TAG}.pt"
STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"
OUT_ROOT="$OUTPUTS_ROOT/generations/cfg_horizon_${STAMP}"

for F in "$CTC_CKPT" "$RENDERER" "$LM_CKPT"; do
    [[ -f "$F" ]] || { echo "MISSING: $F" >&2; exit 1; }
done
FREE=$(df -Pm /scratch/l/lovenya | awk 'NR==2{print $4}')
echo "free space: ${FREE} MB (need >= 1500)"
[[ "$FREE" -ge 1500 ]] || { echo "ABORT: only ${FREE} MB free" >&2; exit 1; }

salm_print_header "CFG PLAN HORIZON: K=$K $TAG, w in {$WS} + greedy control"

LANES="$OUT_ROOT/lanes.txt"; mkdir -p "$OUT_ROOT"; : > "$LANES"
for W in $WS; do
  for S in $SEEDS; do echo "$W|0|$S|w${W}__s${S}" >> "$LANES"; done
done
# Greedy: top_k=1 fixes the codes, so these six seeds differ only in diffusion noise. This is
# the CEILING for cross-seed agreement and every other row is read against it.
for S in $SEEDS; do echo "1.0|1|$S|greedy__s${S}" >> "$LANES"; done
echo "planned lanes: $(wc -l < "$LANES")"

run_lane() {
    local gpu="$1" spec="$2"
    IFS='|' read -r w topk seed name <<< "$spec"
    local out_dir="$OUT_ROOT/$name"
    mkdir -p "$out_dir"
    echo "[gpu$gpu] $name"
    CUDA_VISIBLE_DEVICES="$gpu" python -m salm.generate --mode a2s --direction a2s \
        --checkpoint "$RENDERER" --content-source codebook_lm \
        --content-lm-checkpoint "$LM_CKPT" --content-lm-temperature 1.0 \
        --content-lm-cfg "$w" --content-lm-top-k "$topk" \
        --ctc-checkpoint "$CTC_CKPT" --split "$SPLIT" --num-samples "$NGEN" \
        --start-index 0 --seed "$seed" --out-dir "$out_dir" --copy-reference \
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
echo "==================== PLAN HORIZON ===================="
GROUP_ARGS=()
for W in $WS; do
    dirs=(); for S in $SEEDS; do
        [[ -f "$OUT_ROOT/w${W}__s${S}/manifest.jsonl" ]] && dirs+=("$OUT_ROOT/w${W}__s${S}")
    done
    [[ ${#dirs[@]} -gt 0 ]] && GROUP_ARGS+=(--group "cfg_w${W}" "${dirs[@]}")
done
gdirs=(); for S in $SEEDS; do
    [[ -f "$OUT_ROOT/greedy__s${S}/manifest.jsonl" ]] && gdirs+=("$OUT_ROOT/greedy__s${S}")
done
[[ ${#gdirs[@]} -gt 0 ]] && GROUP_ARGS+=(--group "greedy_CEILING" "${gdirs[@]}")
[[ ${#GROUP_ARGS[@]} -gt 0 ]] || { echo "no scorable lanes" >&2; exit 1; }

python scripts/a2s/score_plan_horizon.py "${GROUP_ARGS[@]}" \
    --ctc-checkpoint "$CTC_CKPT" --device cuda \
    --out "$OUTPUTS_ROOT/plan_horizon_${STAMP}.json" \
    || echo "SCORING FAILED -- generations are intact, rescore separately" >&2

echo
echo "=========================================================="
echo "READ IN THIS ORDER:"
echo "  1. does cfg_w1.0 reproduce 0.20 0.20 0.17 0.12 0.00? if not, STOP -- sampler changed"
echo "  2. greedy_CEILING -- the maximum cross-seed agreement the renderer allows at all"
echo "  3. horizon vs w    -- does scene control extend past 4 words?"
echo "  4. inter-ASR row   -- if it collapses as w rises, the gain is off-manifold, not real"
echo "listen: $OUT_ROOT/*/audio/"
echo "=========================================================="
echo "OK"
