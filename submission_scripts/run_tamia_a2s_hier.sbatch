#!/usr/bin/env bash
# COARSE-TO-FINE CONTENT LM: plan at a resolution that fits the measured horizon.
#
# THE ARITHMETIC. Cross-seed agreement on a fixed ambient clip (442814) runs 0.20 0.20 0.17
# 0.12 then 0.00, so the model holds ~20 run-length tokens of plan. The full sequence it must
# emit is 151.4 tokens (measured, k2048 dedup cache). A 7.6x shortfall.
#
#     full sequence    151 tokens vs a 20-token horizon  ->  7.6x short
#     coarse @ stride 8  19 tokens vs a 20-token horizon  ->  FITS
#
# Everything tried so far attacked this by making the model USE its 20 tokens better -- CFG
# (442847) bought two words, the plan vector (443751) was null, think tokens (443989) were
# null. None changed what 20 tokens SPAN. A coarse pass does: at 1/8 resolution the whole
# utterance fits inside the horizon the model actually has, so it can commit to a trajectory
# across the entire utterance before emitting a single fine token. The fine pass then reads
# that skeleton through ordinary self-attention over its own context.
#
# NOT THE PLAN VECTOR AGAIN. That predicted a bag of codes -- one summary, no time structure --
# and beat a SCENE-BLIND marginal by 0.0040 nats, so the model was asked to commit to something
# carrying almost nothing. A coarse SEQUENCE has time structure, spans the whole utterance, and
# every one of its tokens is supervised by real data rather than by a target the agent invented.
#
# ======================= EXPECTED RESULTS (decided before running) =======================
# THE PREDICTION IS ORDERED, which is far stronger than a single arm. Measured on real data:
#   stride  6 -> 25.4 coarse tokens  DOES NOT fit the horizon
#   stride  8 -> 19.1 coarse tokens  fits
#   stride 12 -> 12.9 coarse tokens  fits with margin
# If the horizon story is right, 8 and 12 beat 6. If all three behave alike, the story is WRONG
# and the mechanism is something else -- record that rather than salvaging it.
#
#   PRIMARY: cross-seed agreement horizon (score_plan_horizon.py), against dedup_k2048's
#   matched control which scored horizon 1 at K=2048 on 8 clips (job 443751's no-plan arm).
#     SUCCESS  horizon rises for stride 8/12 and not for 6. The hierarchy is the mechanism, and
#              the follow-up is a coarser-still level or a learned rather than strided sketch.
#     FAILURE  flat everywhere. Then planning resolution is not the constraint either. With
#              decoding (443737), commitment (443751), computation (443989) and now hierarchy
#              eliminated, what remains is the representation -- BPE at the corrected bar, and
#              variable-length output -- reached by elimination rather than preference.
#
#   READ FIRST, BEFORE THE HORIZON -- three ways this can be a no-op wearing a result:
#     1. `coarse plan: N tokens ... M rows planned NOTHING` in the generation logs. A row whose
#        coarse pass collapsed emitted fine tokens with no plan behind them. If M is large the
#        arm never tested the hypothesis.
#     2. val_coarse_loss vs val_fine_loss in the trainer. If the coarse loss never falls, the
#        sketch is noise and the fine pass is conditioning on nothing.
#     3. mean coarse length at generation should match training (19.1 at stride 8). A large
#        mismatch means the sampler's SEP handling is wrong, not that the idea failed.
#
#   SECONDARY: val_fine_loss against dedup_k2048's val_code_loss (1.7588 at its ep9 minimum).
#   Not directly comparable -- the hier model scores a longer sequence over a K+3 head -- so
#   read the DIRECTION, not the difference.
#
#   NOT AN OUTCOME: inter_asr_dist. It measures ASR agreement, and it disagreed with the ear
#   twice on 2026-09-06. Scored for the record; the horizon is the deliverable.
# =========================================================================================
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_hier.sbatch
#SBATCH --job-name=salm-a2s-hier
#SBATCH --account=aip-csubakan
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

K="${K:-2048}"
EPOCHS="${EPOCHS:-100}"
SEED="${SEED:-0}"
MAX_MINUTES="${MAX_MINUTES:-95}"
STRIDES="${STRIDES:-6 8 12}"
SEEDS_GEN="${SEEDS_GEN:-11 22 33 44 55 66}"
CB="$OUTPUTS_ROOT/content_codebook/codebook_k${K}.pt"
CTC_CKPT="$OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt"
# The FROZEN renderer, settled 2026-09-07 by ear from the 150-epoch ladder.
RENDERER="$RUNS_ROOT/a2s_audioldm_clapseq_xformer/rladder_20260906_1654/checkpoints/checkpoint_ep119.pt"
STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"
OUT_ROOT="$OUTPUTS_ROOT/generations/hier_${STAMP}"

for F in "$CB" "$CTC_CKPT" "$RENDERER"; do
    [[ -f "$F" ]] || { echo "MISSING: $F" >&2; exit 1; }
done
for ST in $STRIDES; do
    d="$OUTPUTS_ROOT/content_lm/hier_k${K}_s${ST}"
    [[ -e "$d" ]] && { echo "ABORT: $d exists -- refusing to overwrite" >&2; exit 1; }
done
FREE=$(df -Pm /scratch/l/lovenya | awk 'NR==2{print $4}')
echo "free space: ${FREE} MB (need >= 5000)"
[[ "$FREE" -ge 5000 ]] || { echo "ABORT: only ${FREE} MB free" >&2; exit 1; }

salm_print_header "COARSE-TO-FINE: K=$K, strides {$STRIDES}, $EPOCHS epochs, seed $SEED"
echo "renderer FROZEN at rladder ep119; only the content LM varies"

echo
echo "==================== 1. TRAIN (one stride per GPU) ===================="
i=0
for ST in $STRIDES; do
    CUDA_VISIBLE_DEVICES=$i python -u scripts/a2s/train_content_lm_hier.py \
        --codebook "$CB" --run-name "hier_k${K}_s${ST}" --stride "$ST" --seed "$SEED" \
        --epochs "$EPOCHS" --batch-size 64 \
        --d-model 512 --layers 6 --heads 8 --dropout 0.1 --weight-decay 0.01 \
        --max-run 32 --save-every 20 --max-minutes "$MAX_MINUTES" \
        > "$OUTPUTS_ROOT/logs/slurm/hier_s${ST}_${STAMP}.log" 2>&1 &
    i=$((i + 1))
done
wait
echo "training done"

echo
echo "==================== 2. DID THE COARSE PASS LEARN ANYTHING? ===================="
# Read before the horizon. A coarse loss that never falls means the sketch is noise and the
# fine pass conditioned on nothing -- a no-op, not a null result.
python - "$OUTPUTS_ROOT" $STRIDES <<'PY' || true
import json, sys
from pathlib import Path
R = Path(sys.argv[1]) / "content_lm"
print(f"  {'run':18s} {'ep':>4s} {'coarse':>8s} {'fine':>8s} {'coarse@0':>9s} {'scene_gap':>10s}")
for st in sys.argv[2:]:
    h = R / f"hier_k2048_s{st}" / "history.json"
    if not h.exists():
        print(f"  hier_s{st}: no history"); continue
    hist = json.loads(h.read_text())
    lo = min(hist, key=lambda e: e["val_loss"])
    c0 = hist[0]["val_coarse_loss"]
    flat = "  <-- FLAT, sketch is noise" if abs(c0 - lo["val_coarse_loss"]) < 0.05 else ""
    print(f"  hier_k2048_s{st:<4s} {lo['epoch']:4d} {lo['val_coarse_loss']:8.4f} "
          f"{lo['val_fine_loss']:8.4f} {c0:9.4f} {lo['val_scene_gap']:+10.4f}{flat}")
print()
print("  baseline for direction only (different head width, different sequence):")
print("    dedup_k2048 val_code_loss 1.7588 @ep9, scene_gap +0.0322")
PY

echo
echo "==================== 3. GENERATE (6-seed sweep, the horizon) ===================="
for ST in $STRIDES; do
    LM="$OUTPUTS_ROOT/content_lm/hier_k${K}_s${ST}/checkpoints/checkpoint_best_loss.pt"
    [[ -f "$LM" ]] || { echo "  skip (absent): stride $ST" >&2; continue; }
    for S in $SEEDS_GEN; do
        d="$OUT_ROOT/s${ST}__seed${S}"; mkdir -p "$d"
        CUDA_VISIBLE_DEVICES=$(( (S / 11) % 4 )) python -m salm.generate --mode a2s --direction a2s \
            --checkpoint "$RENDERER" --content-source codebook_lm --content-lm-checkpoint "$LM" \
            --content-lm-temperature 1.0 --ctc-checkpoint "$CTC_CKPT" \
            --split val --num-samples 8 --start-index 0 --seed "$S" \
            --out-dir "$d" --copy-reference > "$d.log" 2>&1 || echo "  FAILED: s$ST seed$S" >&2 &
    done
    wait
    d="$OUT_ROOT/s${ST}__greedy"; mkdir -p "$d"
    CUDA_VISIBLE_DEVICES=0 python -m salm.generate --mode a2s --direction a2s \
        --checkpoint "$RENDERER" --content-source codebook_lm --content-lm-checkpoint "$LM" \
        --content-lm-temperature 1.0 --content-lm-top-k 1 --ctc-checkpoint "$CTC_CKPT" \
        --split val --num-samples 64 --start-index 0 --seed 1337 \
        --out-dir "$d" --copy-reference > "$d.log" 2>&1 || echo "  FAILED: s$ST greedy" >&2
done

echo
echo "==================== 4. DID THE PLAN SURVIVE SAMPLING? ===================="
# The generation logs report the coarse length actually emitted. Training says 19.1 at stride
# 8; a large mismatch means the sampler's SEP handling is broken, not that the idea failed.
grep -h "coarse plan:" "$OUT_ROOT"/*.log 2>/dev/null | sort | uniq -c | head -12 \
    || echo "  (no coarse-plan lines -- the hier branch may not have been taken)"

echo
echo "==================== 5. HORIZON (the deliverable) ===================="
GROUP_ARGS=()
for ST in $STRIDES; do
    dirs=()
    for S in $SEEDS_GEN; do
        [[ -f "$OUT_ROOT/s${ST}__seed${S}/manifest.jsonl" ]] && dirs+=("$OUT_ROOT/s${ST}__seed${S}")
    done
    [[ ${#dirs[@]} -gt 0 ]] && GROUP_ARGS+=(--group "stride${ST}" "${dirs[@]}")
done
[[ ${#GROUP_ARGS[@]} -gt 0 ]] || { echo "no seed lanes to score" >&2; exit 1; }
python scripts/a2s/score_plan_horizon.py "${GROUP_ARGS[@]}" \
    --ctc-checkpoint "$CTC_CKPT" --device cuda \
    --out "$OUTPUTS_ROOT/hier_horizon_${STAMP}.json" \
    || echo "HORIZON SCORING FAILED -- generations intact" >&2

echo
echo "==================== 6. clarity, for the record only ===================="
CTRL="$OUTPUTS_ROOT/generations/control_lanes_20260831_0735"
DIRS="$CTRL/real_speech $CTRL/vae_roundtrip"
for ST in $STRIDES; do
    d="$OUT_ROOT/s${ST}__greedy"; [[ -f "$d/manifest.jsonl" ]] && DIRS="$DIRS $d"
done
# shellcheck disable=SC2086
python scripts/a2s/score_intelligibility.py --dirs $DIRS --ctc-checkpoint "$CTC_CKPT" \
    --device cuda --out "$OUTPUTS_ROOT/hier_intel_${STAMP}.json" \
    || echo "SCORING FAILED -- generations intact" >&2
python scripts/a2s/build_listening_bundle.py --gen-root "$OUT_ROOT" \
    --out "$OUTPUTS_ROOT/listening_bundles/hier_${STAMP}" --max-clips 8 --require-all-lanes || true

echo
echo "=========================================================="
echo "READ IN THIS ORDER:"
echo "  1. section 2 -- did the coarse loss FALL? if flat, the sketch is noise and this is a"
echo "     no-op, not a null result"
echo "  2. section 4 -- coarse length at generation should match training (19.1 at stride 8),"
echo "     and 'rows planned NOTHING' should be near zero"
echo "  3. the HORIZON table. The prediction is ORDERED: strides 8 and 12 beat stride 6."
echo "     Flat everywhere means the horizon story is wrong -- say so."
echo "listen: $OUTPUTS_ROOT/listening_bundles/hier_${STAMP}"
echo "=========================================================="
echo "OK"
