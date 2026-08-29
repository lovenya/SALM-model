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
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

CHECKPOINT="${CHECKPOINT:-$RUNS_ROOT/a2s_audioldm_clapseq_xformer/content_xattn_falsify_100ep/checkpoints/checkpoint_best.pt}"
CTC_CKPT="${CTC_CKPT:-$OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt}"
NUM_GEN="${NUM_GEN:-16}"
OUT_DIR="${OUT_DIR:-$OUTPUTS_ROOT/generations/content_xattn_retrieval_$(TZ=America/Toronto date +%Y%m%d_%H%M)}"

for f in "$CHECKPOINT" "$CTC_CKPT"; do
    [[ -f "$f" ]] || { echo "MISSING: $f" >&2; exit 1; }
done

salm_print_header "First real (retrieval-content) generation from Rank 2, n=$NUM_GEN"

python -m salm.generate --mode a2s --direction a2s \
    --checkpoint "$CHECKPOINT" \
    --use-content-xattn --ctc-checkpoint "$CTC_CKPT" \
    --split val --num-samples "$NUM_GEN" \
    --out-dir "$OUT_DIR" --copy-reference \
    2>&1 | tee "$OUT_DIR.log"

echo
echo "=========================================================="
echo "listen: $OUT_DIR/audio/   (<id>.wav = generated, <id>__reference.wav = real target)"
echo "compare against: $RUNS_ROOT/a2s_audioldm_clapseq_xformer/probe40_A1_o2/generations/a2s_best/audio/"
echo "=========================================================="
