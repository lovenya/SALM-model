#!/usr/bin/env bash
# Hear the LATE checkpoints of the coarse-to-fine run. Generation only, ~15 min.
#
# WHY, when the loss says not to. hier_k2048_s12's val_loss bottoms at ep10 (2.7165) and climbs
# monotonically to 5.0062 by ep99 -- textbook overfitting, and every earlier content LM behaved
# the same way. The audio Lovenya rated the most intelligible of the project came from that
# ep10 checkpoint.
#
# But asking anyway is right, for a reason this project has learned twice the hard way:
# `inter_asr_dist` disagreed with the ear on 2026-09-06 (two lanes 0.27 apart were heard as
# "90/95% same"), and the renderer ladder's late rungs sounded CLEARER by ear while the metric
# was flat. A loss curve is not a perceptual curve, and val_loss on a token objective has never
# been shown to track intelligibility here. ~15 minutes settles it.
#
# ======================= EXPECTED RESULTS (decided before running) =======================
#   PRIOR: ep79/ep99 sound WORSE. val_loss nearly doubles from ep10, and the same pattern held
#   for the dedup ladder (443555), where ep99 was the worst lane by inter_asr_dist in all three
#   codebooks. Two independent precedents point the same way.
#
#   THE INTERESTING OUTCOME IS THE OTHER ONE. If a late checkpoint sounds BETTER despite
#   val_loss at 5.0, then token-level val_loss is not merely an imperfect proxy for
#   intelligibility -- it is anti-correlated with it, and every checkpoint this project has
#   selected on `best_loss` was chosen by the wrong criterion. That would be a bigger finding
#   than anything the coarse-to-fine job produced, so it is worth 15 minutes to rule out.
#
#   WATCH: utterance LENGTH. The measured reason ep10 sounds good is that it stops early --
#   11.1 words against the flat model's 17.6, human reference 13.3. If the late checkpoints
#   ramble again, that is the mechanism reasserting itself and confirms length control, not
#   planning, is what the coarse plan bought.
#
#   NOT AN OUTCOME: inter_asr_dist alone. Scored for the record; the ear decides.
# =========================================================================================
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_hier_late.sbatch
#SBATCH --job-name=salm-a2s-hier-late
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

RUN="${RUN:-hier_k2048_s12}"
TAGS="${TAGS:-checkpoint_best_loss checkpoint_ep79 checkpoint_ep99}"
NGEN="${NGEN:-64}"
CTC_CKPT="$OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt"
RENDERER="$RUNS_ROOT/a2s_audioldm_clapseq_xformer/rladder_20260906_1654/checkpoints/checkpoint_ep119.pt"
STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"
OUT_ROOT="$OUTPUTS_ROOT/generations/hierlate_${STAMP}"

for F in "$CTC_CKPT" "$RENDERER"; do
    [[ -f "$F" ]] || { echo "MISSING: $F" >&2; exit 1; }
done
FREE=$(df -Pm /scratch/l/lovenya | awk 'NR==2{print $4}')
echo "free space: ${FREE} MB (need >= 1500)"
[[ "$FREE" -ge 1500 ]] || { echo "ABORT: only ${FREE} MB free" >&2; exit 1; }

salm_print_header "HIER LATE CHECKPOINTS: $RUN, tags {$TAGS}, frozen renderer ep119"

for TAG in $TAGS; do
    LM="$OUTPUTS_ROOT/content_lm/$RUN/checkpoints/${TAG}.pt"
    [[ -f "$LM" ]] || { echo "  skip (absent): $TAG" >&2; continue; }
    d="$OUT_ROOT/${TAG#checkpoint_}"; mkdir -p "$d"
    echo; echo "=== $TAG ==="
    CUDA_VISIBLE_DEVICES=0 python -m salm.generate --mode a2s --direction a2s \
        --checkpoint "$RENDERER" --content-source codebook_lm --content-lm-checkpoint "$LM" \
        --content-lm-temperature 1.0 --content-lm-top-k 1 --ctc-checkpoint "$CTC_CKPT" \
        --split val --num-samples "$NGEN" --start-index 0 --seed 1337 \
        --out-dir "$d" --copy-reference 2>&1 | tee "$d.log" \
        | grep -E "coarse plan:|sampled" | head -3 || echo "  FAILED: $TAG" >&2
done

echo
echo "==================== LENGTH: does it start rambling again? ===================="
python - "$OUT_ROOT" <<'PY' || true
import json, sys, numpy as np
from pathlib import Path
root = Path(sys.argv[1])
print(f"  {'checkpoint':14s} {'coarse plan':>12s}")
for d in sorted(root.glob("*/")):
    log = root / f"{d.name}.log"
    if not log.exists(): continue
    cp = [l for l in log.read_text().splitlines() if "coarse plan:" in l]
    val = cp[0].split("coarse plan:")[1].split("tokens")[0].strip() if cp else "--"
    print(f"  {d.name:14s} {val:>12s}")
print("\n  (word counts come from the scoring step below; ep10 measured 11.1 words,")
print("   the flat dedup LM 17.6, the human reference 13.3)")
PY

echo
echo "==================== SCORE (for the record) ===================="
CTRL="$OUTPUTS_ROOT/generations/control_lanes_20260831_0735"
DIRS="$CTRL/real_speech $CTRL/vae_roundtrip"
for d in "$OUT_ROOT"/*/; do [[ -f "$d/manifest.jsonl" ]] && DIRS="$DIRS ${d%/}"; done
# shellcheck disable=SC2086
python scripts/a2s/score_intelligibility.py --dirs $DIRS --ctc-checkpoint "$CTC_CKPT" \
    --device cuda --out "$OUTPUTS_ROOT/hierlate_intel_${STAMP}.json" \
    || echo "SCORING FAILED -- generations intact" >&2
python scripts/a2s/build_listening_bundle.py --gen-root "$OUT_ROOT" \
    --out "$OUTPUTS_ROOT/listening_bundles/hierlate_${STAMP}" --max-clips 10 --require-all-lanes || true

echo
echo "=========================================================="
echo "listen: $OUTPUTS_ROOT/listening_bundles/hierlate_${STAMP}"
echo "  best_loss is ep10 (val_loss 2.72) -- the lane already rated best by ear"
echo "  ep79 (4.76) and ep99 (5.01) are deep in overfitting by the loss curve"
echo "If a late one sounds BETTER, token val_loss is anti-correlated with intelligibility and"
echo "every best_loss selection in this project used the wrong criterion. Say so if it happens."
echo "=========================================================="
echo "OK"
