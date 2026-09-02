#!/usr/bin/env bash
# STEP 2 of the discrete-content plan: train the autoregressive predictor over content codes.
#
# THE GATE PASSED (job 436244, 2026-09-02). Quantized ORACLE content through the proven
# content_concat_v4 renderer, all lanes at content_scale=1.0:
#     real speech        0.3146      oracle (unquantized)  0.4231
#     vae ceiling        0.3722      quantized K=2048      0.4843
#                                    quantized K=1024      0.5480
#                                    quantized K=512       0.6014
#                                    quantized K=256       0.6926
# Aggregate degrades monotonically with K, but the WORDS clearly survive -- K=1024 returned
# "Come on, old girl, wake up" (more accurate than unquantized oracle's "Come on, Olko, wake
# up"), and K=2048 returned "This trick works every time. They always bring the food."
# verbatim. real_word_rate held at 0.90-0.96 everywhere. So phoneme identity survives
# frame-wise quantization and a predictor over these codes is worth building.
#
# WHY K=1024 IS THE PRIMARY. K=2048 is 0.06 more faithful but doubles the vocabulary the LM
# must learn from only 17,596 utterances, and PREDICTION is the binding constraint here --
# four consecutive content sources have failed at exactly this step, none of them for lack of
# reconstruction fidelity. Both are trained here since the LM is cheap; K=1024 first so the
# more likely-to-work configuration gets the wall clock if the budget runs short.
#
# ======================= EXPECTED RESULTS (decided before running) =======================
# The metric is val_scene_gap = loss(rolled ambient) - loss(true ambient), where "rolled"
# pairs each code sequence with a DIFFERENT row's real ambient encoding. NOT train loss and
# NOT raw perplexity: content is ~53% silence and English phoneme statistics are highly
# predictable, so a model that entirely ignores the scene can post a healthy cross-entropy.
# That is the same class of lie val_content_bind_gap was built to catch on the concat channel.
#
#   SUCCESS -- val_scene_gap clearly and durably > 0, while val_loss beats the unigram
#     baseline the script prints. Meaning: ambient audio genuinely predicts which content
#     codes come next. That is the first honest evidence in this project that scene -> content
#     is learnable at all, and step 3 (sample codes -> dequantize -> the frozen renderer)
#     becomes worth running.
#
#   FAILURE -- val_scene_gap ~ 0 while val_loss still falls. Meaning: the model learned
#     generic speech statistics and ignores the scene. This is the SAME verdict as the four
#     previous content sources, now reached with the mean-collapse escape route removed, and
#     it would say the ~24h of paired data simply does not contain a learnable
#     ambience -> content signal. That points squarely at VoiceLDM-scale pretraining rather
#     than a sixth mechanism, and it is worth knowing cheaply.
#
#   AMBIGUOUS -- scene_gap small but consistently positive across both K. Real but weak
#     signal; next step is conditioning capacity (more layers / cross-attn heads) or a richer
#     ambient representation (RESEARCH_DIRECTIONS.md D3) before spending a generation job.
#
#   COLLAPSE GUARD -- val_argmax_top1_share near 1.0 means the model predicts one code
#     (almost certainly silence) everywhere. Loss will look fine. Treat as failure.
#
#   NOTE: checkpoint_best is selected on val_scene_gap, NOT val_loss. A checkpoint with the
#   better loss and no scene_gap is exactly the failure this run exists to detect, and
#   selecting it would hand step 3 a model that ignores its own input.
# =========================================================================================
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_content_lm.sbatch
#SBATCH --job-name=salm-a2s-content-lm
#SBATCH --account=aip-ravanelm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

CB_DIR="${CB_DIR:-$OUTPUTS_ROOT/content_codebook}"
CTC_CKPT="${CTC_CKPT:-$OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-64}"
# Epoch time for this model could not be measured before its first run (no GPU forward passes
# on login nodes), so each K gets a wall-clock budget and truncates GRACEFULLY with its
# checkpoints written rather than being SIGKILLed at the Slurm limit mid-save. 55 min each
# leaves ~20 min of the 2:30 for the two code-cache passes and slack.
MAX_MINUTES="${MAX_MINUTES:-55}"
# K order matters: the primary configuration runs first so it gets the wall clock if the
# budget turns out tighter than estimated.
KS="${KS:-1024 2048}"

for F in "$CTC_CKPT" "$FEATURES_ROOT/a2s/train/index.jsonl" \
         "$OUTPUTS_ROOT/features_a2s_cond/train/cond_seq.npy" \
         "$OUTPUTS_ROOT/features_a2s_cond/val/cond_seq.npy"; do
    [[ -e "$F" ]] || { echo "MISSING: $F" >&2; exit 1; }
done

salm_print_header "ContentLM: autoregressive content codes from ambient, K=[$KS]"

for K in $KS; do
    CB="$CB_DIR/codebook_k${K}.pt"
    if [[ ! -f "$CB" ]]; then
        echo "MISSING codebook: $CB (skipping K=$K)" >&2
        continue
    fi
    RUN_NAME="content_lm_k${K}_v1"
    echo
    echo "==================== K=$K -> $RUN_NAME ===================="
    # Single GPU by design: a ~20M-parameter decoder over 256-position sequences with 17,596
    # training rows is nowhere near DDP territory, and keeping it on one device means the
    # scene ablation is computed on exactly the batch it is reported for.
    CUDA_VISIBLE_DEVICES=0 python scripts/a2s/train_content_lm.py \
        --codebook "$CB" \
        --run-name "$RUN_NAME" \
        --ctc-checkpoint "$CTC_CKPT" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --max-minutes "$MAX_MINUTES" \
        2>&1 | tee -a "$OUTPUTS_ROOT/content_lm/${RUN_NAME}_train.log" \
        || echo "  FAILED: K=$K (continuing to the next K)" >&2
done

echo
echo "==================== COMPARISON ACROSS K ===================="
python - <<'PY'
import json, os, pathlib
root = pathlib.Path(os.environ["OUTPUTS_ROOT"]) / "content_lm"
rows = []
for run in sorted(root.glob("content_lm_k*_v1")):
    h = run / "history.json"
    if not h.exists():
        print(f"{run.name}: no history.json (crashed before first epoch?)")
        continue
    hist = json.loads(h.read_text())
    if not hist:
        continue
    best = max(hist, key=lambda r: r["val_scene_gap"])
    last = hist[-1]
    rows.append((run.name, len(hist), best, last))

if not rows:
    print("no runs produced history -- check the logs above")
else:
    print(f"{'run':>22} {'eps':>4} {'best gap':>10} {'@ep':>5} {'last gap':>10} "
          f"{'val_loss':>9} {'unigram':>9} {'top1':>6}")
    for name, n, best, last in rows:
        print(f"{name:>22} {n:>4} {best['val_scene_gap']:>+10.4f} {best['epoch']:>5} "
              f"{last['val_scene_gap']:>+10.4f} {last['val_loss']:>9.4f} "
              f"{last['val_unigram_loss']:>9.4f} {last['val_argmax_top1_share']:>6.3f}")
    print()
    print("READ IT LIKE THIS:")
    print("  best gap clearly > 0 AND val_loss < unigram -> the scene predicts content;")
    print("      step 3 (sample -> dequantize -> frozen renderer) is worth running")
    print("  best gap ~ 0                                -> scene ignored; same verdict as")
    print("      the four previous content sources, now with mean-collapse ruled out")
    print("  top1 near 1.0                               -> collapsed onto one code; the")
    print("      loss is meaningless regardless of what the gap says")
PY
echo "=========================================================="
echo "OK"
