#!/usr/bin/env bash
# Train ContentRetriever: a learned ambient<->content matching function, meant to replace
# raw CLAP cosine similarity as retrieval's key. See src/salm/a2s/content_retriever.py for
# the full mechanism and why this targets AmbientContentPredictor's actual failure mode
# (blurry continuous regression) rather than repeating it -- this NEVER outputs content
# directly, only a score used to pick a real training example.
#
# EXPECTED RESULTS, pre-registered. val_r1 (does the correct content rank #1 for its own
# ambient query, within an eval batch) is THE metric -- val_loss alone is not informative in
# isolation.
#   - val_r1 climbs to several x chance (batch_size=256 -> chance=0.0039) and keeps improving
#     for most of training -> the learned embedding space genuinely beats raw CLAP cosine
#     similarity at this specific matching task. Wire it into generate.py next (needs a
#     one-time precompute of train-side content embeddings) and compare its retrieved-content
#     intelligibility against plain retrieval's 0.4999-0.454 baseline (jobs 433677/433675).
#   - val_r1 stays near chance -> the contrastive objective isn't finding structure raw CLAP
#     cosine similarity doesn't already have; the bottleneck is the AMBIENT feature space
#     itself (CLAP), not the matching function on top of it. Don't build the generate.py
#     integration; revisit CLAP alternatives instead (docs/RESEARCH_DIRECTIONS.md D3).
#
# Single-GPU by design: two small MLPs, well under DDP territory.
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_train_content_retriever.sbatch
#SBATCH --job-name=salm-a2s-content-retriever
#SBATCH --account=aip-ravanelm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail
# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

RUN_NAME="${RUN_NAME:-content_retriever_v1}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TRAIN_MINUTES="${TRAIN_MINUTES:-30}"
CTC_CKPT="${CTC_CKPT:-$OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt}"

RUN_DIR="$OUTPUTS_ROOT/content_retriever/$RUN_NAME"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

for F in "$FEATURES_ROOT/a2s/train/latents_mean.npy" "$OUTPUTS_ROOT/features_a2s_cond/train/cond_seq.npy" "$CTC_CKPT"; do
    [[ -f "$F" ]] || { echo "MISSING: $F" >&2; exit 1; }
done

CKPT="$RUN_DIR/checkpoints/checkpoint_last.pt"
RESUME_ARGS=()
if [[ -f "$CKPT" ]]; then
    echo "resuming from existing checkpoint: $CKPT"
    RESUME_ARGS=("--resume-from" "$CKPT")
fi

salm_print_header "Content retriever: learned ambient<->content matching, $EPOCHS epochs"
salm_start_monitor "$LOG_DIR" "$RUN_NAME" 60

CUDA_VISIBLE_DEVICES=0 python scripts/a2s/train_content_retriever.py \
    --run-name "$RUN_NAME" --runs-root "$OUTPUTS_ROOT/content_retriever" \
    --ctc-checkpoint "$CTC_CKPT" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --max-train-minutes "$TRAIN_MINUTES" \
    "${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}" \
    2>&1 | tee -a "$LOG_DIR/train.log"

echo
echo "==================== CONTENT RETRIEVER SUMMARY ===================="
python - <<PY
import json, pathlib
h = json.loads((pathlib.Path("$RUN_DIR") / "history.json").read_text())
print(f"epochs completed: {len(h)}")
r1 = [(r["epoch"], r["val_r1"]) for r in h if r.get("val_r1") is not None]
if r1:
    best = max(r1, key=lambda x: x[1])
    chance = h[0].get("chance_r1")
    print(f"val_r1: first={r1[0][1]:.4f} last={r1[-1][1]:.4f} best={best[1]:.4f} @ep{best[0]}"
          + (f"  (chance={chance:.4f}, best is {best[1]/chance:.1f}x)" if chance else ""))
et = [r.get("epoch_minutes", 0) for r in h]
print(f"mean epoch time: {sum(et)/max(len(et),1):.2f} min")
PY
echo "======================================================================="
