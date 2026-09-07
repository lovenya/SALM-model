#!/usr/bin/env bash
# CONTRASTIVE SCENE OBJECTIVE: is 5.28 nats the DATA's ceiling, or just what unpressured
# training finds? Lovenya's idea, and the only experiment that separates the two.
#
# THE INVARIANCE THIS TESTS. Converted to per-utterance information (gap x tokens/clip), every
# content LM ever trained here extracts the same amount of scene information:
#
#     dedup_k512   5.34    frame_k512   5.63
#     dedup_k1024  5.26    frame_k1024  5.19       mean 5.28 nats, sd 0.22
#     dedup_k2048  4.90    frame_k2048  5.37       (~7.6 bits per utterance)
#
# Two representations, three codebook sizes, and they agree to +/-4%. This invariance is the
# strongest evidence in the project, and it has been used to argue that conditioning
# interventions are futile -- CFG, the plan vector and think tokens were all read against it.
#
# BUT EVERY ONE OF THOSE SIX MODELS WAS TRAINED WITHOUT A SCENE OBJECTIVE. The invariance says
# models never ASKED to use the scene all extract the same amount. It does not say more cannot
# be extracted. Nobody has ever put pressure on that term. `val_scene_gap` has been a
# diagnostic for the entire project and never once an objective.
#
#     (a) 5.28 nats is what the ambient CONTAINS -> the ceiling is known, everything is capped
#     (b) 5.28 nats is what unpressured training FINDS -> there is untouched headroom
#
# THE MECHANISM. Per-sample softplus(CE_true - CE_rolled): the InfoNCE two-way form with the
# true ambient as positive and a rolled batch (a different clip's ambient) as negative. Rolled
# rather than zeroed, for the reason `shuffled_memory` documents -- a zeroed memory is out of
# distribution, so pushing away from it teaches avoidance of something never seen at
# generation. Costs one extra forward pass per step, so ~2x training time.
#
# ======================= EXPECTED RESULTS (decided before running) =======================
# THE TERM IS GAMEABLE AND THIS IS NOT HYPOTHETICAL. It depends only on the DIFFERENCE:
#     CE_true 1.50 / CE_roll 4.00  ->  loss 0.0789
#     CE_true 1.20 / CE_roll 3.70  ->  loss 0.0789   IDENTICAL
# A model can minimise it by becoming BRITTLE to mismatch rather than better at the true
# pairing. That is exactly the pattern where val_scene_gap rises with overfitting (LEARNINGS,
# 2026-09-05), so a rising gap on its own proves nothing.
#
#   READ IN THIS ORDER, and the first two decide it:
#     1. val_code_loss ON THE TRUE PAIRING. Baseline dedup_k2048 is 1.7588 at its ep9 minimum.
#        If this WORSENS while the gap grows, the term is gaming and the arm is discarded
#        however good the gap looks.
#     2. val_scene_gap x tokens, in nats/utterance, against the 5.28 +/- 0.22 invariance.
#     3. paired length-adjusted greedy inter_asr_dist vs the lambda=0 control.
#
#   SUCCESS -- nats/utterance rises materially above 5.28 (say > 6.5) WITH val_code_loss held
#       or improved. Then the invariance was a property of the OBJECTIVE, not the data, there
#       is headroom nobody has touched, and this becomes the main line.
#   GAMING -- gap rises, val_code_loss worsens. The term found the cheap solution. Report it as
#       such; do not report the gap.
#   NULL -- nats/utterance stays near 5.28 across both lambdas. Then the invariance IS the data
#       and the project's conditioning ceiling is established rather than assumed. That closes
#       the most consequential open question here, and it is worth the job even so.
#
#   DOSE-RESPONSE, which is why there are two lambdas rather than one: if 2.0 moves the gap
#   more than 0.5 and both beat the control, that is a mechanism responding to pressure. A
#   non-monotone result is noise, and the plan-vector post-mortem is the reason for insisting.
# =========================================================================================
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_contrast.sbatch
#SBATCH --job-name=salm-a2s-contrast
#SBATCH --account=aip-csubakan
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
SEED="${SEED:-0}"
LAMBDAS="${LAMBDAS:-0.0 0.5 2.0}"
MAX_MINUTES="${MAX_MINUTES:-120}"   # contrast doubles the step; 100 ep at ~0.9 min = ~90 min
NGEN="${NGEN:-128}"
CB="$OUTPUTS_ROOT/content_codebook/codebook_k${K}.pt"
CTC_CKPT="$OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt"
RENDERER="$RUNS_ROOT/a2s_audioldm_clapseq_xformer/rladder_20260906_1654/checkpoints/checkpoint_ep119.pt"
STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"
OUT_ROOT="$OUTPUTS_ROOT/generations/contrast_${STAMP}"

for F in "$CB" "$CTC_CKPT" "$RENDERER"; do
    [[ -f "$F" ]] || { echo "MISSING: $F" >&2; exit 1; }
done
for L in $LAMBDAS; do
    d="$OUTPUTS_ROOT/content_lm/contrast_k${K}_l${L}"
    [[ -e "$d" ]] && { echo "ABORT: $d exists -- refusing to overwrite" >&2; exit 1; }
done
FREE=$(df -Pm /scratch/l/lovenya | awk 'NR==2{print $4}')
echo "free space: ${FREE} MB (need >= 3000)"
[[ "$FREE" -ge 3000 ]] || { echo "ABORT: only ${FREE} MB free" >&2; exit 1; }

salm_print_header "CONTRASTIVE SCENE OBJECTIVE: lambda in {$LAMBDAS}, K=$K, seed $SEED"
echo "the bar: 5.28 +/- 0.22 nats/utterance, invariant across 6 prior models"
echo "baseline val_code_loss to hold: 1.7588 (dedup_k2048 @ep9)"

echo
echo "==================== 1. TRAIN (one lambda per GPU) ===================="
i=0
for L in $LAMBDAS; do
    CUDA_VISIBLE_DEVICES=$i python -u scripts/a2s/train_content_lm_dedup.py \
        --codebook "$CB" --run-name "contrast_k${K}_l${L}" --seed "$SEED" \
        --epochs "$EPOCHS" --batch-size 64 \
        --d-model 512 --layers 6 --heads 8 --dropout 0.1 --weight-decay 0.01 \
        --max-run 32 --save-every 0 --max-minutes "$MAX_MINUTES" \
        --lambda-contrast "$L" \
        > "$OUTPUTS_ROOT/logs/slurm/contrast_l${L}_${STAMP}.log" 2>&1 &
    i=$((i + 1))
done
wait
echo "training done"

echo
echo "==================== 2. DID IT EXTRACT MORE, OR JUST GAME THE TERM? ===================="
python - "$OUTPUTS_ROOT" "$K" $LAMBDAS <<'PY' || true
import json, sys
from pathlib import Path
R = Path(sys.argv[1]) / "content_lm"; K = sys.argv[2]
print(f"  {'lambda':>7s} {'ep':>4s} {'val_code_loss':>14s} {'gap/token':>10s} {'tokens':>8s} "
      f"{'nats/utt':>9s} {'contrast':>9s}")
base = None
for L in sys.argv[3:]:
    h = R / f"contrast_k{K}_l{L}" / "history.json"
    if not h.exists():
        print(f"  lambda {L}: no history"); continue
    hist = json.loads(h.read_text())
    lo = min(hist, key=lambda e: e["val_loss"])
    tk = lo.get("true_mean_tokens", 152.4)
    nats = lo["val_scene_gap"] * tk
    if base is None:
        base = lo["val_code_loss"]
    flag = ""
    if L != "0.0":
        if lo["val_code_loss"] > base + 0.05:
            flag = "  *** GAMING: true-pairing loss WORSE than control ***"
        elif nats > 6.5:
            flag = "  <-- above the 5.28 invariance"
    print(f"  {L:>7s} {lo['epoch']:4d} {lo['val_code_loss']:14.4f} {lo['val_scene_gap']:10.4f} "
          f"{tk:8.1f} {nats:9.2f} {lo.get('train_contrast_loss', 0.0):9.4f}{flag}")
print()
print("  val_code_loss on the TRUE pairing decides this, not the gap. The term depends only on")
print("  the difference, so a brittle model scores identically to a better one.")
PY

echo
echo "==================== 3. GENERATE (greedy, 128 clips, frozen renderer) ===================="
i=0
for L in $LAMBDAS; do
    LM="$OUTPUTS_ROOT/content_lm/contrast_k${K}_l${L}/checkpoints/checkpoint_best_loss.pt"
    [[ -f "$LM" ]] || { echo "  skip (absent): lambda $L" >&2; continue; }
    d="$OUT_ROOT/lambda${L}"; mkdir -p "$d"
    CUDA_VISIBLE_DEVICES=$i python -m salm.generate --mode a2s --direction a2s \
        --checkpoint "$RENDERER" --content-source codebook_lm --content-lm-checkpoint "$LM" \
        --content-lm-temperature 1.0 --content-lm-top-k 1 --ctc-checkpoint "$CTC_CKPT" \
        --split val --num-samples "$NGEN" --start-index 0 --seed 1337 \
        --out-dir "$d" --copy-reference > "$d.log" 2>&1 &
    i=$((i + 1))
done
wait

echo
echo "==================== 4. SCORE + PAIRED COMPARISON ===================="
CTRL="$OUTPUTS_ROOT/generations/control_lanes_20260831_0735"
DIRS="$CTRL/real_speech $CTRL/vae_roundtrip"
for d in "$OUT_ROOT"/*/; do [[ -f "$d/manifest.jsonl" ]] && DIRS="$DIRS ${d%/}"; done
SCORES="$OUTPUTS_ROOT/contrast_intel_${STAMP}.json"
# shellcheck disable=SC2086
python scripts/a2s/score_intelligibility.py --dirs $DIRS --ctc-checkpoint "$CTC_CKPT" \
    --device cuda --out "$SCORES" \
    || { echo "SCORING FAILED -- generations intact under $OUT_ROOT" >&2; exit 1; }

TREAT=""
for L in $LAMBDAS; do [[ "$L" != "0.0" ]] && TREAT="$TREAT lambda${L}"; done
# shellcheck disable=SC2086
python scripts/a2s/compare_arms.py --scores "$SCORES" --control lambda0.0 \
    --treatment $TREAT --out "$OUTPUTS_ROOT/contrast_compare_${STAMP}.json" || true

python scripts/a2s/build_listening_bundle.py --gen-root "$OUT_ROOT" \
    --out "$OUTPUTS_ROOT/listening_bundles/contrast_${STAMP}" --max-clips 10 --require-all-lanes || true

echo
echo "=========================================================="
echo "READ section 2 FIRST: val_code_loss on the true pairing, then nats/utterance vs 5.28."
echo "  A gap that grew while val_code_loss worsened is the term gaming itself -- discard it."
echo "  Both lambdas near 5.28 means the invariance IS the data, and the conditioning ceiling"
echo "  for this project is established rather than assumed. That is a real answer either way."
echo "listen: $OUTPUTS_ROOT/listening_bundles/contrast_${STAMP}"
echo "=========================================================="
echo "OK"
