#!/usr/bin/env bash
# RETRAIN all three dedup content LMs and KEEP EVERY CHECKPOINT, then generate greedily from
# the full epoch spread. Requested by Lovenya 2026-09-06.
#
# WHY THIS IS NEEDED AT ALL. Job 439442 trained these exact three configs for 100 epochs and
# saved ep19/39/59/79/99 plus last/best/best_loss. During the 2026-09-05 disk crisis the agent
# recommended deleting ep59, ep79, ep99 and checkpoint_last from all four dedup runs, and that
# recommendation was taken. The literal final epoch has been unavailable since, across repeated
# requests for it. This run exists to undo that, and the deletion should not have been proposed.
#
# NEW RUN NAMES, deliberately: dedup_k*_full, NOT dedup_k*. The surviving dedup_k*/best_loss
# checkpoints hold the best result this project has recorded (k2048 greedy, inter_asr_dist
# 0.338 against a 0.372 VAE ceiling). Nothing here may overwrite them.
#
# Same hyperparameters and same seed as 439442 -- d_model 512, 6 layers, 8 heads, dropout 0.1,
# weight decay 0.01, max_run 32, batch 64, seed 0 -- so the trajectory should reproduce.
#
# ======================= EXPECTED RESULTS (decided before running) =======================
# FIRST, A FREE CORRECTNESS CHECK. Same seed and hyperparameters means the new best_loss should
# land at the same epoch (9 for K=512/2048, 10 for K=1024) with val_loss within noise of
# 2.3684 / 2.4177 / 2.4927. If it does NOT, training is not deterministic here and the new
# epoch spread cannot be compared against any earlier dedup result -- check that before reading
# anything else below.
#
# THE ACTUAL QUESTION -- what do the late checkpoints sound like? The loss curve makes a firm
# prediction, and the job is worth running because a prediction is not a measurement:
#
#   val_loss(k512):  ep9 2.37 | ep19 2.63 | ep39 3.62 | ep87 5.01 | ep99 (deleted, ~5.1)
#
#   EXPECTED -- ep99 is the WORST lane of the spread, continuing the monotone degradation
#       already visible from ep9 to ep87. Under greedy, best_loss beat best by 0.13-0.17
#       inter_asr_dist in all three codebooks; ep99 should be at or below `best`.
#       This is the outcome the evidence predicts, and it closes the question rather than
#       opening one: the last epoch is documented, not merely asserted to be bad.
#   SURPRISE -- ep99 matches or beats best_loss. That would mean val_loss on the run-length
#       token objective does not track generated speech quality at all, which would invalidate
#       checkpoint selection for every content LM in this project and demand a rethink of what
#       we optimise. Low prior, high consequence, and cheap to rule out.
#   AMBIGUOUS -- the late lanes cluster with `best` and differ little from each other. Then
#       the ladder flattens after ~ep40 and the extra checkpoints add nothing; record that so
#       nobody regenerates them again.
#
# Note the metric caveat that applies to every lane here: inter_asr_dist rewards clear
# articulation, and greedy decoding picks high-confidence codes, so absolute values are
# flattering. The COMPARISON across checkpoints is matched (same decoder, same clips, same
# seed) and is what this job is for.
#
# DISK. 8 checkpoints x 117 MB x 3 runs = ~2.8 GB of checkpoints, plus ~600 MB of wavs.
# Checked before starting; /scratch hit 100% mid-write during 439442.
# =========================================================================================
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_dedup_retrain_full.sbatch
#SBATCH --job-name=salm-a2s-dedup-retrain
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

EPOCHS="${EPOCHS:-100}"
SEED="${SEED:-0}"
MAX_MINUTES="${MAX_MINUTES:-85}"
NUM_GEN="${NUM_GEN:-16}"
CTC_CKPT="$OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt"
RENDERER="$RUNS_ROOT/a2s_audioldm_clapseq_xformer/content_concat_v4/checkpoints/checkpoint_best.pt"
STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"
OUT_ROOT="$OUTPUTS_ROOT/generations/dedup_full_${STAMP}"

for F in "$CTC_CKPT" "$RENDERER"; do
    [[ -f "$F" ]] || { echo "MISSING: $F" >&2; exit 1; }
done
FREE=$(df -Pm /scratch/l/lovenya | awk 'NR==2{print $4}')
echo "free space: ${FREE} MB (need >= 6000 for ~2.8 GB checkpoints + ~600 MB wavs + headroom)"
[[ "$FREE" -ge 6000 ]] || { echo "ABORT: only ${FREE} MB free" >&2; exit 1; }

# Refuse to clobber. The surviving dedup_k*/best_loss checkpoints are the best result on
# record; a name collision here would be unrecoverable.
for K in 512 1024 2048; do
    d="$OUTPUTS_ROOT/content_lm/dedup_k${K}_full"
    [[ -e "$d" ]] && { echo "ABORT: $d already exists -- refusing to overwrite" >&2; exit 1; }
done

salm_print_header "DEDUP RETRAIN (full spread kept): K=512/1024/2048, $EPOCHS epochs, seed $SEED"

echo
echo "==================== 1. TRAIN (3 configs, one per GPU) ===================="
i=0
for K in 512 1024 2048; do
    CB="$OUTPUTS_ROOT/content_codebook/codebook_k${K}.pt"
    [[ -f "$CB" ]] || { echo "MISSING codebook $CB -- skipping k$K" >&2; continue; }
    CUDA_VISIBLE_DEVICES=$i python -u scripts/a2s/train_content_lm_dedup.py \
        --codebook "$CB" --run-name "dedup_k${K}_full" --seed "$SEED" \
        --epochs "$EPOCHS" --batch-size 64 \
        --d-model 512 --layers 6 --heads 8 --dropout 0.1 --weight-decay 0.01 \
        --max-run 32 --sample-every 10 --save-every 20 --max-minutes "$MAX_MINUTES" \
        > "$OUTPUTS_ROOT/logs/slurm/dedup_full_k${K}_${STAMP}.log" 2>&1 &
    i=$((i + 1))
done
wait
echo "training done"

echo
echo "==================== 2. REPRODUCIBILITY CHECK ===================="
# Read this before any generation lane. If the retrain did not reproduce, the epoch spread is
# not comparable to anything measured before.
python - "$OUTPUTS_ROOT" <<'PY' || true
import json, sys
from pathlib import Path
R = Path(sys.argv[1]) / "content_lm"
ref = {512: (9, 2.3684), 1024: (10, 2.4177), 2048: (9, 2.4927)}
for K, (ep0, vl0) in ref.items():
    h = R / f"dedup_k{K}_full" / "history.json"
    if not h.exists():
        print(f"  k{K}: no history"); continue
    hist = json.loads(h.read_text())
    lo = min(hist, key=lambda e: e["val_loss"])
    d = abs(lo["val_loss"] - vl0)
    flag = "OK" if (lo["epoch"] == ep0 and d < 0.05) else "*** DIVERGED ***"
    print(f"  k{K}: best_loss ep{lo['epoch']} val {lo['val_loss']:.4f} "
          f"(439442: ep{ep0} {vl0:.4f}, delta {d:.4f})  {flag}")
    print(f"        epochs {len(hist)}, final val_loss {hist[-1]['val_loss']:.4f}")
PY

echo
echo "==================== 3. GENERATE, GREEDY, EVERY CHECKPOINT ===================="
# Greedy because 443379 showed it beats sampling by a wide margin and removes seed noise from
# a checkpoint comparison -- exactly what is wanted when the epoch is the variable.
for K in 512 1024 2048; do
    CKDIR="$OUTPUTS_ROOT/content_lm/dedup_k${K}_full/checkpoints"
    [[ -d "$CKDIR" ]] || continue
    for CK in "$CKDIR"/*.pt; do
        tag="$(basename "${CK%.pt}")"; tag="${tag#checkpoint_}"
        out_dir="$OUT_ROOT/k${K}__${tag}__greedy"
        mkdir -p "$out_dir"
        if [[ "$(df -Pm /scratch/l/lovenya | awk 'NR==2{print $4}')" -lt 500 ]]; then
            echo "  STOPPING: free space under 500 MB. Completed lanes are intact." >&2; break 2
        fi
        echo "=== k$K $tag (greedy) ==="
        CUDA_VISIBLE_DEVICES=0 python -m salm.generate --mode a2s --direction a2s \
            --checkpoint "$RENDERER" --content-source codebook_lm \
            --content-lm-checkpoint "$CK" --content-lm-temperature 1.0 --content-lm-top-k 1 \
            --ctc-checkpoint "$CTC_CKPT" --split val --num-samples "$NUM_GEN" \
            --start-index 0 --seed 1337 --out-dir "$out_dir" --copy-reference \
            > "$out_dir.log" 2>&1 || echo "  FAILED: k$K $tag" >&2
    done
done

echo
echo "==================== 4. SCORE ===================="
CTRL="$OUTPUTS_ROOT/generations/control_lanes_20260831_0735"
DIRS="$CTRL/real_speech $CTRL/vae_roundtrip"
for d in "$OUT_ROOT"/*/; do [[ -f "$d/manifest.jsonl" ]] && DIRS="$DIRS ${d%/}"; done
SCORES="$OUTPUTS_ROOT/dedup_full_intel_${STAMP}.json"
# shellcheck disable=SC2086
python scripts/a2s/score_intelligibility.py --dirs $DIRS \
    --ctc-checkpoint "$CTC_CKPT" --device cuda --out "$SCORES" \
    || echo "SCORING FAILED -- generations intact under $OUT_ROOT" >&2

python scripts/a2s/export_transcript_review.py --scores "$SCORES" --gen-root "$OUT_ROOT" \
    --limit 40 --out "$OUTPUTS_ROOT/transcript_review_dedup_full_${STAMP}.md" \
    || echo "REVIEW EXPORT FAILED" >&2

python scripts/a2s/build_listening_bundle.py --gen-root "$OUT_ROOT" \
    --out "$OUTPUTS_ROOT/listening_bundles/dedup_full_${STAMP}" --max-clips 8 \
    || echo "BUNDLE FAILED -- lanes intact" >&2

echo
echo "=========================================================="
echo "READ THE REPRODUCIBILITY CHECK IN SECTION 2 FIRST."
echo "If best_loss did not land at ep9/ep10 with matching val_loss, the epoch spread is not"
echo "comparable to job 439442 and nothing below should be read against earlier numbers."
echo "listen: $OUTPUTS_ROOT/listening_bundles/dedup_full_${STAMP}"
echo "=========================================================="
echo "OK"
