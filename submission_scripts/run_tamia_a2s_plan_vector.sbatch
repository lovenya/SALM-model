#!/usr/bin/env bash
# PLAN VECTOR: does a global, once-per-utterance commitment extend the plan horizon?
# Matched ablation -- plan vs no-plan, same seed, same everything else. Then greedy generation.
#
# THE MEASUREMENT THIS ANSWERS. Cross-seed agreement on a fixed ambient clip (job 442814):
#
#     position     1     2     3     4     5     6     7     8
#     agreement  0.20  0.20  0.17  0.12  0.00  0.00  0.02  0.00
#
# The scene controls about four words and then stops mattering, while positional inter-ASR
# agreement shows the audio stays clearly articulated to word 10 -- so the model keeps speaking
# clearly, untethered. Converted to tokens at the measured 2.85 words/s: the model holds ~20
# run-length tokens of plan and one utterance needs ~56.
#
#     representation      plan horizon    one utterance    shortfall
#     frame-level            39 tokens       111 tokens       2.8x
#     run-length (2.0x)      20 tokens        56 tokens       2.8x
#
# COMPRESSION CANNOT CLOSE THIS. It shrinks both sides equally, so the ratio is invariant --
# which is exactly why run-length's 2.0x bought so little and why BPE alone would not either.
# Everything tried so far attacks the denominator. This attacks the numerator, and it is the
# only untried idea that does.
#
# THE INTERVENTION. The ambient is already cross-attended at every step, so the information is
# available throughout; what is missing is COMMITMENT. Nothing forces the model to decide once
# what this utterance will be, so it re-derives its intent from the last few tokens at every
# step and drifts. The plan head predicts, from the pooled ambient, a distribution over WHICH
# CODES appear anywhere in the utterance -- a bag of codes, order-free on purpose -- and that
# is projected and ADDED to every token embedding, biasing all 256 steps at once.
#
# WHY A DISTRIBUTION AND NOT A REGRESSED EMBEDDING. This project exists because
# `argmin E[(f(x)-y)^2] = E[y|x]` and the conditional mean of symbolic targets is off-manifold:
# the average of five utterances is not an utterance. A regressed plan EMBEDDING walks straight
# back into that. A distribution over codes does not -- the average of valid distributions is a
# valid distribution, so an uncertain plan is meaningful rather than degenerate. Same reasoning
# that moved the content targets to discrete codes, applied one level up.
#
# ======================= EXPECTED RESULTS (decided before running) =======================
# The ablation is clean by construction and this was verified, not assumed: `plan_proj` is
# zero-initialised and the plan modules are built AFTER the positional embeddings, so at the
# same seed a use_plan=True model is BIT-IDENTICAL to its use_plan=False twin (checked: max
# logit difference 0.00e+00). Any divergence is the plan, not a different initialisation.
#
#   PRIMARY, and the reason for the job -- cross-seed agreement horizon, from
#   score_plan_horizon.py. Baseline horizon is 4 words at `best_loss`.
#     SUCCESS   horizon 6+ with inter-ASR clarity holding near the real-speech curve. The plan
#               is the missing mechanism, and the follow-up is a richer plan (ordered segments,
#               or a coarse-to-fine cascade) rather than a bigger model.
#     FAILURE   horizon unchanged at 4. Then a global commitment is not what is missing, and
#               the constraint is the decoder's capacity to USE a plan over 56 steps rather
#               than its access to one. That would leave shrinking the requirement (BPE at the
#               corrected ~20-25 token bar) as the remaining option, reached by elimination.
#
#   SECONDARY -- val_loss. The plan adds a real auxiliary task, so a small val_loss regression
#   is acceptable if the horizon moves; the horizon is the deliverable, not the loss. But a
#   LARGE regression (> ~0.15) means lambda_plan is fighting the main objective and the sweep
#   should be re-run lower rather than the idea discarded.
#
#   THE PLAN MUST ACTUALLY BE LEARNED, and this is checked before anything else is read. The
#   trainer logs plan BCE each epoch. If it sits flat at its initial value the head learned
#   nothing, `plan_proj` stayed near zero, and the run is a no-op wearing the label of an
#   experiment -- an outcome that would otherwise be misreported as "the plan does not help".
#
#   NOT AN OUTCOME: inter_asr_dist. It rewards clear articulation, greedy already maximises it
#   by saying safe things, and it is reported here only to confirm the audio did not collapse.
#
#   CONFOUND, stated: two lambda_plan values are run, so a null result covers a range rather
#   than one arbitrary weight. It is still only two points on that axis.
# =========================================================================================
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_plan_vector.sbatch
#SBATCH --job-name=salm-a2s-plan-vector
#SBATCH --account=aip-csubakan
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

K="${K:-2048}"                       # the codebook that wins under greedy (0.3688 at 128 clips)
EPOCHS="${EPOCHS:-100}"
SEED="${SEED:-0}"
MAX_MINUTES="${MAX_MINUTES:-85}"
NGEN="${NGEN:-64}"
SEEDS_GEN="${SEEDS_GEN:-11 22 33 44 55 66}"
CTC_CKPT="$OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt"
RENDERER="$RUNS_ROOT/a2s_audioldm_clapseq_xformer/content_concat_v4/checkpoints/checkpoint_best.pt"
CB="$OUTPUTS_ROOT/content_codebook/codebook_k${K}.pt"
STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"
OUT_ROOT="$OUTPUTS_ROOT/generations/plan_${STAMP}"

for F in "$CTC_CKPT" "$RENDERER" "$CB"; do
    [[ -f "$F" ]] || { echo "MISSING: $F" >&2; exit 1; }
done
FREE=$(df -Pm /scratch/l/lovenya | awk 'NR==2{print $4}')
echo "free space: ${FREE} MB (need >= 5000)"
[[ "$FREE" -ge 5000 ]] || { echo "ABORT: only ${FREE} MB free" >&2; exit 1; }

# name|extra-args   -- the no-plan arm is the matched control, not a historical comparison
CONFIGS=(
  "plan_k${K}_noplan|"
  "plan_k${K}_p1.0|--use-plan --lambda-plan 1.0"
  "plan_k${K}_p0.3|--use-plan --lambda-plan 0.3"
)
for c in "${CONFIGS[@]}"; do
    d="$OUTPUTS_ROOT/content_lm/${c%%|*}"
    [[ -e "$d" ]] && { echo "ABORT: $d exists -- refusing to overwrite" >&2; exit 1; }
done

salm_print_header "PLAN VECTOR: K=$K, ${#CONFIGS[@]} arms (no-plan control + 2 lambda), seed $SEED"

echo
echo "==================== 1. TRAIN (one arm per GPU) ===================="
i=0
for c in "${CONFIGS[@]}"; do
    NAME="${c%%|*}"; EXTRA="${c#*|}"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$i python -u scripts/a2s/train_content_lm_dedup.py \
        --codebook "$CB" --run-name "$NAME" --seed "$SEED" \
        --epochs "$EPOCHS" --batch-size 64 \
        --d-model 512 --layers 6 --heads 8 --dropout 0.1 --weight-decay 0.01 \
        --max-run 32 --sample-every 10 --save-every 20 --max-minutes "$MAX_MINUTES" \
        $EXTRA > "$OUTPUTS_ROOT/logs/slurm/plan_${NAME}_${STAMP}.log" 2>&1 &
    i=$((i + 1))
done
wait
echo "training done"

echo
echo "==================== 2. DID THE PLAN ACTUALLY LEARN? ===================="
# Read this before anything else. A flat plan loss means the run is a no-op, not a null result.
python - "$OUTPUTS_ROOT" "${CONFIGS[@]}" <<'PY' || true
import json, sys
from pathlib import Path
R = Path(sys.argv[1]) / "content_lm"
for spec in sys.argv[2:]:
    name = spec.split("|")[0]
    h = R / name / "history.json"
    if not h.exists():
        print(f"  {name}: no history"); continue
    hist = json.loads(h.read_text())
    lo = min(hist, key=lambda e: e["val_loss"])
    p0 = hist[0].get("train_plan_loss", 0.0)
    pN = lo.get("train_plan_loss", 0.0)
    flat = " *** PLAN LOSS FLAT -- head learned nothing, this arm is a NO-OP ***" \
           if (p0 > 0 and abs(p0 - pN) < 0.01) else ""
    print(f"  {name:22s} epochs {len(hist):3d} | best val_loss {lo['val_loss']:.4f} @ep{lo['epoch']}"
          f" | scene_gap {lo.get('val_scene_gap', float('nan')):+.4f}"
          f" | plan BCE {p0:.4f} -> {pN:.4f}{flat}")
PY

echo
echo "==================== 3. GENERATE (greedy + a 6-seed sweep) ===================="
# Greedy for the headline clarity number; the seed sweep is what measures the horizon, and it
# needs sampling because cross-seed agreement is undefined for a deterministic decoder.
for c in "${CONFIGS[@]}"; do
    NAME="${c%%|*}"
    LM="$OUTPUTS_ROOT/content_lm/$NAME/checkpoints/checkpoint_best_loss.pt"
    [[ -f "$LM" ]] || { echo "  skip (absent): $NAME" >&2; continue; }
    d="$OUT_ROOT/${NAME}__greedy"; mkdir -p "$d"
    CUDA_VISIBLE_DEVICES=0 python -m salm.generate --mode a2s --direction a2s \
        --checkpoint "$RENDERER" --content-source codebook_lm --content-lm-checkpoint "$LM" \
        --content-lm-temperature 1.0 --content-lm-top-k 1 --ctc-checkpoint "$CTC_CKPT" \
        --split val --num-samples "$NGEN" --start-index 0 --seed 1337 \
        --out-dir "$d" --copy-reference > "$d.log" 2>&1 || echo "  FAILED: $NAME greedy" >&2
    for S in $SEEDS_GEN; do
        d="$OUT_ROOT/${NAME}__s${S}"; mkdir -p "$d"
        CUDA_VISIBLE_DEVICES=$(( (S / 11) % 4 )) python -m salm.generate --mode a2s --direction a2s \
            --checkpoint "$RENDERER" --content-source codebook_lm --content-lm-checkpoint "$LM" \
            --content-lm-temperature 1.0 --ctc-checkpoint "$CTC_CKPT" \
            --split val --num-samples 8 --start-index 0 --seed "$S" \
            --out-dir "$d" --copy-reference > "$d.log" 2>&1 || echo "  FAILED: $NAME s$S" >&2 &
    done
    wait
done

echo
echo "==================== 4. THE HORIZON (the deliverable) ===================="
GROUP_ARGS=()
for c in "${CONFIGS[@]}"; do
    NAME="${c%%|*}"; dirs=()
    for S in $SEEDS_GEN; do
        [[ -f "$OUT_ROOT/${NAME}__s${S}/manifest.jsonl" ]] && dirs+=("$OUT_ROOT/${NAME}__s${S}")
    done
    [[ ${#dirs[@]} -gt 0 ]] && GROUP_ARGS+=(--group "$NAME" "${dirs[@]}")
done
[[ ${#GROUP_ARGS[@]} -gt 0 ]] || { echo "no seed lanes to score" >&2; exit 1; }
python scripts/a2s/score_plan_horizon.py "${GROUP_ARGS[@]}" \
    --ctc-checkpoint "$CTC_CKPT" --device cuda \
    --out "$OUTPUTS_ROOT/plan_horizon_${STAMP}.json" \
    || echo "HORIZON SCORING FAILED -- generations intact" >&2

echo
echo "==================== 5. CLARITY (guard rail only) ===================="
CTRL="$OUTPUTS_ROOT/generations/control_lanes_20260831_0735"
DIRS="$CTRL/real_speech $CTRL/vae_roundtrip"
for c in "${CONFIGS[@]}"; do
    d="$OUT_ROOT/${c%%|*}__greedy"; [[ -f "$d/manifest.jsonl" ]] && DIRS="$DIRS $d"
done
# shellcheck disable=SC2086
python scripts/a2s/score_intelligibility.py --dirs $DIRS --ctc-checkpoint "$CTC_CKPT" \
    --device cuda --out "$OUTPUTS_ROOT/plan_intel_${STAMP}.json" \
    || echo "CLARITY SCORING FAILED -- generations intact" >&2

python scripts/a2s/build_listening_bundle.py --gen-root "$OUT_ROOT" \
    --out "$OUTPUTS_ROOT/listening_bundles/plan_${STAMP}" --max-clips 8 || true

echo
echo "=========================================================="
echo "READ IN THIS ORDER:"
echo "  1. section 2 -- did the plan arms even train, and how far did val_loss move?"
echo "  2. cross-seed HORIZON, no-plan vs plan. Baseline is 4 words. This is the deliverable."
echo "  3. inter-ASR row -- clarity must not have collapsed to buy the horizon"
echo "  4. inter_asr_dist -- guard rail only; greedy maximises it by playing safe"
echo "listen: $OUTPUTS_ROOT/listening_bundles/plan_${STAMP}"
echo "=========================================================="
echo "OK"
