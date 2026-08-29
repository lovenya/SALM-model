#!/usr/bin/env bash
# First REAL (non-oracle) generation from Rank 2's content cross-attention: retrieval-based
# content (closest-matching TRAINING scene's own speech content), not the oracle content the
# checkpoint was trained on. See src/salm/generate.py's --use-content-xattn and
# src/salm/a2s/content_adapter.py for the mechanism.
#
# This is genuinely untested code -- no forward pass could be run on the login node to debug
# it first (per repo policy). Real risks, both fail loudly rather than silently: (1) the
# checkpoint-load order (adapter must be installed on the U-Net BEFORE loading its weights,
# or load_trained_unet's strict=False silently drops them) and (2) the classifier-free-
# guidance batch doubling inside the unet.forward monkeypatch. A crash here is real,
# actionable information about which of those is wrong -- not a wasted job.
#
# What success/failure/ambiguous each mean, decided BEFORE running, not after:
#   - crash                -> the wiring has a bug; the traceback says which of the two
#                              risks above it is. Fix and retry, cheap (single GPU, minutes).
#   - runs, but sounds identical to plain-O2 baseline (probe40_A1_o2) -> retrieval content
#     isn't specific/accurate enough to move the U-Net's output at all.
#   - runs, and sounds DIFFERENT from plain-O2 (even if not yet comprehensible) -> the
#     mechanism is doing something with imperfect content; worth a training run exposed to
#     retrieval-quality content specifically, rather than only oracle content.
#   Listening is the actual test -- no metric here, per the "skip WER for now" decision.
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_content_xattn_retrieval_gen.sbatch
#SBATCH --job-name=salm-a2s-xattn-gen
#SBATCH --account=aip-ravanelm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

RUN_NAME="${RUN_NAME:-content_xattn_falsify_100ep}"
CKPT_DIR="$RUNS_ROOT/a2s_audioldm_clapseq_xformer/$RUN_NAME/checkpoints"
CTC_CKPT="${CTC_CKPT:-$OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt}"
NUM_GEN="${NUM_GEN:-16}"
STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"
OUT_ROOT="$OUTPUTS_ROOT/generations/content_xattn_retrieval_${STAMP}"

# Every saved checkpoint, not just "best" -- 2026-08-29 correction: "best" is picked by
# val_clap_r1, which peaks EARLY (epoch 8 here) and has nothing to do with content-xattn's
# own signal (val_x0_rel_mse), which kept improving past epoch 90. Generating from only one
# epoch answers "does this ONE snapshot sound different", not "does the mechanism improve
# with training" -- the actual question. STANDING POLICY GOING FORWARD: every checkpoint-
# comparison generation job covers the full saved spread (first, every ~20 epochs, last,
# best), not a single pick, in ONE job -- this is cheap (14s/16 samples measured) compared
# to the queue-wait cost of finding out later that one epoch wasn't enough to judge by.
[[ -f "$CTC_CKPT" ]] || { echo "MISSING: $CTC_CKPT" >&2; exit 1; }
CKPTS=()
for name in checkpoint_ep19.pt checkpoint_ep39.pt checkpoint_ep59.pt checkpoint_ep79.pt \
            checkpoint_ep99.pt checkpoint_last.pt checkpoint_best.pt; do
    [[ -f "$CKPT_DIR/$name" ]] && CKPTS+=("$name")
done
[[ ${#CKPTS[@]} -gt 0 ]] || { echo "MISSING: no checkpoints found in $CKPT_DIR" >&2; exit 1; }
echo "checkpoints to generate from: ${CKPTS[*]}"

salm_print_header "Content-xattn retrieval generation across ${#CKPTS[@]} checkpoints, n=$NUM_GEN each"

for name in "${CKPTS[@]}"; do
    tag="${name%.pt}"
    out_dir="$OUT_ROOT/$tag"
    echo
    echo "=== $tag ==="
    python -m salm.generate --mode a2s --direction a2s \
        --checkpoint "$CKPT_DIR/$name" \
        --use-content-xattn --ctc-checkpoint "$CTC_CKPT" \
        --split val --num-samples "$NUM_GEN" \
        --out-dir "$out_dir" --copy-reference \
        2>&1 | tee "$out_dir.log"
done

echo
echo "=========================================================="
echo "listen: $OUT_ROOT/<checkpoint_tag>/audio/   (<id>.wav = generated, <id>__reference.wav = real target)"
echo "compare against: $RUNS_ROOT/a2s_audioldm_clapseq_xformer/probe40_A1_o2/generations/a2s_best/audio/"
echo "=========================================================="
