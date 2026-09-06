#!/usr/bin/env bash
# LATENT THINK TOKENS: give the model computation before it speaks, with no target imposed.
#
# THE DIAGNOSIS THIS FOLLOWS FROM. A transformer keeps no persistent state. An RNN carries a
# hidden vector forward and UPDATES it, so an intent accumulates; a transformer recomputes
# everything from the surface tokens every step. This model is already autoregressive -- the
# causal mask is real and each step sees every token it emitted -- but it must re-infer WHY it
# said those things from the words alone, and nothing forces the answer at step 40 to match
# step 5. Measured (job 442814): cross-seed agreement 0.20 0.20 0.17 0.12 then 0.00. The intent
# has quietly become something else by word four, while the audio stays clearly articulated.
#
# WHY NOT ANOTHER PLAN VECTOR. Job 443751 came back null, and the post-mortem is the reason
# this job exists in this shape. The plan head reached BCE 0.2033 against a SCENE-BLIND
# per-code marginal of 0.2073 -- it beat "ignore the ambient entirely" by 0.0040 nats. The
# model was asked to commit to something that carried essentially no scene information, so a
# null horizon says nothing about whether commitment helps. That was a target-choice failure,
# and the target was mine.
#
# Think tokens impose NO target. `n_think` learnable vectors are prepended to the decoder
# input; the model runs self-attention over them and cross-attention to the ambient before
# emitting any code, then their outputs are discarded. Extra computation in latent space, no
# token ever produced from it, trained end to end on the unchanged objective. Whatever is worth
# computing, the model decides -- so this cannot inherit the same mistake.
#
# ======================= EXPECTED RESULTS (decided before running) =======================
# STEP 1 IS A WIRING GATE on GPU (check_think_wiring.py), and it is not ceremony. Four of its
# five checks cover failures that would produce a plausible-looking null:
#   - n_think changing nothing (a silent no-op arm)
#   - a leaked think position shifting every downstream target by n_think, which would still
#     train and still converge, just on the wrong alignment
#   - pad_mask mis-alignment
#   - broken causality, which would let the model peek ahead and look better for a bad reason
# If the gate fails, nothing below is interpretable.
#
# THREE ARMS, same seed, K=2048, 100 epochs, matched to job 443751's no-plan control:
#   n_think=0  -- CONTROL. Bit-identical to the current model.
#   n_think=8  -- modest budget.
#   n_think=32 -- 32 positions of latent computation before the first code.
#
#   PRIMARY: cross-seed horizon (score_plan_horizon.py). The matched control in 443751 scored
#   horizon 1 at K=2048 on 8 clips; that, not the older 4-word figure from a different K and
#   clip count, is the number to beat.
#     SUCCESS  horizon rises with n_think, clarity holding. Latent computation is the missing
#              mechanism; the follow-up is PERIODIC think tokens (a block every ~16 codes) so
#              the opinion is refined during generation rather than only formed before it.
#     FAILURE  horizon flat across all three. Then extra computation before speaking is not
#              what is missing either, and with commitment (443751) and decoding (443737) both
#              eliminated, the constraint sits in the representation -- promoting the codebook
#              and BPE work by elimination rather than preference.
#
#   SECONDARY: val_loss. Think tokens add capacity, so a small improvement is expected and is
#   NOT evidence for the mechanism -- job 438062 already swept capacity (two depths, 3x
#   dropout, 10x weight decay) and found scene_gap invariant at +0.0203..+0.0222. If val_loss
#   improves while the horizon does not move, that is capacity, not thinking, and should be
#   written up as such.
#
#   HONEST LIMIT: these are PREFIX think tokens -- one block of computation before the first
#   code, not an opinion refined as generation proceeds. That is the simpler and standard form,
#   and it is what this job tests. Periodic insertion is the follow-up, gated on this showing
#   signal, because it complicates the loss alignment and the sampler and should not be built
#   speculatively.
# =========================================================================================
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_think_tokens.sbatch
#SBATCH --job-name=salm-a2s-think
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

K="${K:-2048}"
EPOCHS="${EPOCHS:-100}"
SEED="${SEED:-0}"
MAX_MINUTES="${MAX_MINUTES:-85}"
SEEDS_GEN="${SEEDS_GEN:-11 22 33 44 55 66}"
NGEN="${NGEN:-64}"
CTC_CKPT="$OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt"
RENDERER="$RUNS_ROOT/a2s_audioldm_clapseq_xformer/content_concat_v4/checkpoints/checkpoint_best.pt"
CB="$OUTPUTS_ROOT/content_codebook/codebook_k${K}.pt"
STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"
OUT_ROOT="$OUTPUTS_ROOT/generations/think_${STAMP}"

for F in "$CTC_CKPT" "$RENDERER" "$CB"; do
    [[ -f "$F" ]] || { echo "MISSING: $F" >&2; exit 1; }
done
FREE=$(df -Pm /scratch/l/lovenya | awk 'NR==2{print $4}')
echo "free space: ${FREE} MB (need >= 5000)"
[[ "$FREE" -ge 5000 ]] || { echo "ABORT: only ${FREE} MB free" >&2; exit 1; }

CONFIGS=("think_k${K}_n0|0" "think_k${K}_n8|8" "think_k${K}_n32|32")
for c in "${CONFIGS[@]}"; do
    d="$OUTPUTS_ROOT/content_lm/${c%%|*}"
    [[ -e "$d" ]] && { echo "ABORT: $d exists -- refusing to overwrite" >&2; exit 1; }
done

salm_print_header "THINK TOKENS: K=$K, n_think in {0, 8, 32}, seed $SEED"

echo
echo "==================== 1. WIRING GATE (GPU) ===================="
# On GPU, not the login node: CLAUDE.md forbids NN forward passes there, and this was
# attempted locally twice and killed both times. Nothing below is interpretable if this fails.
CUDA_VISIBLE_DEVICES=0 python scripts/a2s/check_think_wiring.py --device cuda --n-think 8 \
    || { echo "WIRING GATE FAILED -- not training on a broken implementation" >&2; exit 1; }

echo
echo "==================== 2. TRAIN (one arm per GPU) ===================="
i=0
for c in "${CONFIGS[@]}"; do
    NAME="${c%%|*}"; NT="${c#*|}"
    CUDA_VISIBLE_DEVICES=$i python -u scripts/a2s/train_content_lm_dedup.py \
        --codebook "$CB" --run-name "$NAME" --seed "$SEED" \
        --epochs "$EPOCHS" --batch-size 64 \
        --d-model 512 --layers 6 --heads 8 --dropout 0.1 --weight-decay 0.01 \
        --max-run 32 --sample-every 10 --save-every 20 --max-minutes "$MAX_MINUTES" \
        --n-think "$NT" > "$OUTPUTS_ROOT/logs/slurm/think_${NAME}_${STAMP}.log" 2>&1 &
    i=$((i + 1))
done
wait
echo "training done"

echo
echo "==================== 3. LOSS SUMMARY ===================="
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
    print(f"  {name:20s} epochs {len(hist):3d} | best val_loss {lo['val_loss']:.4f} @ep{lo['epoch']}"
          f" | scene_gap {lo.get('val_scene_gap', float('nan')):+.4f}"
          f" | dur_acc {lo.get('val_dur_acc', float('nan')):.3f}")
print()
print("  A val_loss gain with a FLAT horizon is capacity, not thinking -- 438062 already")
print("  swept capacity and found scene_gap invariant. Say so if that is what happened.")
PY

echo
echo "==================== 4. GENERATE (6-seed sweep, the horizon) ===================="
for c in "${CONFIGS[@]}"; do
    NAME="${c%%|*}"
    LM="$OUTPUTS_ROOT/content_lm/$NAME/checkpoints/checkpoint_best_loss.pt"
    [[ -f "$LM" ]] || { echo "  skip (absent): $NAME" >&2; continue; }
    for S in $SEEDS_GEN; do
        d="$OUT_ROOT/${NAME}__s${S}"; mkdir -p "$d"
        CUDA_VISIBLE_DEVICES=$(( (S / 11) % 4 )) python -m salm.generate --mode a2s --direction a2s \
            --checkpoint "$RENDERER" --content-source codebook_lm --content-lm-checkpoint "$LM" \
            --content-lm-temperature 1.0 --ctc-checkpoint "$CTC_CKPT" \
            --split val --num-samples 8 --start-index 0 --seed "$S" \
            --out-dir "$d" --copy-reference > "$d.log" 2>&1 || echo "  FAILED: $NAME s$S" >&2 &
    done
    wait
    d="$OUT_ROOT/${NAME}__greedy"; mkdir -p "$d"
    CUDA_VISIBLE_DEVICES=0 python -m salm.generate --mode a2s --direction a2s \
        --checkpoint "$RENDERER" --content-source codebook_lm --content-lm-checkpoint "$LM" \
        --content-lm-temperature 1.0 --content-lm-top-k 1 --ctc-checkpoint "$CTC_CKPT" \
        --split val --num-samples "$NGEN" --start-index 0 --seed 1337 \
        --out-dir "$d" --copy-reference > "$d.log" 2>&1 || echo "  FAILED: $NAME greedy" >&2
done

echo
echo "==================== 5. HORIZON (the deliverable) ===================="
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
    --out "$OUTPUTS_ROOT/think_horizon_${STAMP}.json" \
    || echo "HORIZON SCORING FAILED -- generations intact" >&2

echo
echo "==================== 6. CLARITY (guard rail) ===================="
CTRL="$OUTPUTS_ROOT/generations/control_lanes_20260831_0735"
DIRS="$CTRL/real_speech $CTRL/vae_roundtrip"
for c in "${CONFIGS[@]}"; do
    d="$OUT_ROOT/${c%%|*}__greedy"; [[ -f "$d/manifest.jsonl" ]] && DIRS="$DIRS $d"
done
# shellcheck disable=SC2086
python scripts/a2s/score_intelligibility.py --dirs $DIRS --ctc-checkpoint "$CTC_CKPT" \
    --device cuda --out "$OUTPUTS_ROOT/think_intel_${STAMP}.json" \
    || echo "CLARITY SCORING FAILED -- generations intact" >&2
python scripts/a2s/build_listening_bundle.py --gen-root "$OUT_ROOT" \
    --out "$OUTPUTS_ROOT/listening_bundles/think_${STAMP}" --max-clips 8 || true

echo
echo "=========================================================="
echo "READ IN THIS ORDER:"
echo "  1. the wiring gate -- if it failed, nothing else means anything"
echo "  2. HORIZON vs n_think. 443751's matched control scored 1 at K=2048/8 clips."
echo "  3. val_loss -- an improvement with a flat horizon is CAPACITY, not thinking"
echo "  4. clarity -- must not have collapsed to buy the horizon"
echo "listen: $OUTPUTS_ROOT/listening_bundles/think_${STAMP}"
echo "=========================================================="
echo "OK"
