#!/usr/bin/env bash
# STEP 1 of the discrete-content plan, and a GATE: build content codebooks, then prove
# quantization does not destroy intelligibility -- before any predictor is written.
#
# WHY DISCRETE CONTENT. Four content sources have been tried. The two honest ones failed
# identically for a mathematical reason: argmin_f E[(f(x)-y)^2] = E[y|x]. A deterministic
# map trained against specific targets converges on the conditional MEAN -- not because the
# data is one-to-many (it is a strict bijection) but because NEIGHBOURING scenes have
# unrelated transcripts and a smooth function cannot separate them, and because at validation
# the input is unseen, where interpolating between known neighbours IS averaging. In content
# space that mean is off-manifold: the average of two utterances' encodings is not an
# utterance. See src/salm/a2s/content_codebook.py for the full argument.
#
# A codebook makes every possible output a real content frame, and makes the predictor's
# objective cross-entropy over categories (optimum = the true conditional distribution, whose
# mean is not itself a category) so that SAMPLING yields one committed utterance, not blur.
# Precedent for exactly this failure and fix: lip-to-speech, 65-78% WER with mel regression
# -> 50.2/44.8% with discrete units (Choi et al., arXiv 2305.19603).
#
# ======================= EXPECTED RESULTS (decided before running) =======================
# This job answers ONE question: does quantizing content to K real codes preserve enough for
# the ALREADY-PROVEN renderer (content_concat_v4) to still produce intelligible speech?
#
# It is measured the only way that has ever been trustworthy here -- generate audio and score
# it with the content-independent ASR instrument. Reconstruction cosine is NOT the test; this
# project has repeatedly seen a metric move while the audio did not. Calibration:
#     real speech            0.315
#     VAE round-trip ceiling 0.372
#     UNQUANTIZED oracle     0.389   <-- the number to stay near
#     noise floor            ~0.95
#
#   PASS -- quantized oracle lands close to 0.389 (say <= ~0.45) at some K. Then the discrete
#     representation is faithful enough, K is chosen from this measurement, and step 2 (the
#     autoregressive predictor over codes) is worth building. This is the ONLY outcome that
#     justifies writing the predictor.
#
#   FAIL -- quantized oracle collapses toward the ~0.95 noise floor at every K, INCLUDING
#     K=2048. Then phoneme identity does not survive frame-wise vector quantization in this
#     space, and the whole discrete plan is dead as specified -- cheap to learn now rather
#     than after building a predictor on top of it. The fallback is a coarser-grained unit
#     (residual/product quantization, or units from a self-supervised speech model such as
#     HuBERT rather than our CTC probe's GRU output).
#
#   PARTIAL -- degrades smoothly with K (e.g. 0.42 at K=2048, 0.55 at K=512). Informative and
#     usable: it prices the representation, and step 2 proceeds at the largest K that holds,
#     accepting that a bigger codebook is a harder prediction problem.
#
#   The `oracle` lane is regenerated in this same job at the same seed and rows, so the
#   comparison is against a freshly measured baseline rather than a number from a past run
#   with different sampling conditions.
# =========================================================================================
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_content_codebook_gate.sbatch
#SBATCH --job-name=salm-a2s-cb-gate
#SBATCH --account=aip-ravanelm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

CODES="${CODES:-256 512 1024 2048}"
CTC_CKPT="${CTC_CKPT:-$OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt}"
RENDERER="${RENDERER:-$RUNS_ROOT/a2s_audioldm_clapseq_xformer/content_concat_v4/checkpoints/checkpoint_best.pt}"
CB_DIR="${CB_DIR:-$OUTPUTS_ROOT/content_codebook}"
NUM_GEN="${NUM_GEN:-16}"
STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"
OUT_ROOT="$OUTPUTS_ROOT/generations/codebook_gate_${STAMP}"

for F in "$CTC_CKPT" "$RENDERER" "$FEATURES_ROOT/a2s/train/index.jsonl"; do
    [[ -e "$F" ]] || { echo "MISSING: $F" >&2; exit 1; }
done

salm_print_header "Content codebook gate: fit K=[$CODES], then quantized-oracle generation"

# ---------------------------------------------------------------- 1. fit codebooks
# Single GPU: k-means over a 1.5M x 512 frame pool, plus one CTC-probe pass over the 17,596
# training speech files. Minutes, not hours.
echo
echo "==================== BUILDING CODEBOOKS ===================="
CUDA_VISIBLE_DEVICES=0 python scripts/a2s/build_content_codebook.py \
    --split train \
    --ctc-checkpoint "$CTC_CKPT" \
    --out-dir "$CB_DIR" \
    --codes $CODES \
    --overwrite \
    2>&1 | tee "$OUTPUTS_ROOT/logs/slurm/codebook_build_${STAMP}.log"

# ---------------------------------------------------------------- 2. the gate
# Baseline first, at the same rows/seed, so the comparison is measured not remembered.
echo
echo "==================== GENERATION: BASELINE + EACH K ===================="
mkdir -p "$OUT_ROOT/oracle"
echo
echo "=== oracle (unquantized baseline) ==="
CUDA_VISIBLE_DEVICES=0 python -m salm.generate --mode a2s --direction a2s \
    --checkpoint "$RENDERER" \
    --content-source oracle --ctc-checkpoint "$CTC_CKPT" \
    --split val --num-samples "$NUM_GEN" \
    --out-dir "$OUT_ROOT/oracle" --copy-reference \
    2>&1 | tee "$OUT_ROOT/oracle.log" || echo "  FAILED: oracle baseline" >&2

for K in $CODES; do
    CB="$CB_DIR/codebook_k${K}.pt"
    [[ -f "$CB" ]] || { echo "  MISSING codebook: $CB (skipping)" >&2; continue; }
    out_dir="$OUT_ROOT/quantized_k${K}"
    mkdir -p "$out_dir"
    echo
    echo "=== oracle_quantized, K=$K ==="
    CUDA_VISIBLE_DEVICES=0 python -m salm.generate --mode a2s --direction a2s \
        --checkpoint "$RENDERER" \
        --content-source oracle_quantized --content-codebook "$CB" \
        --ctc-checkpoint "$CTC_CKPT" \
        --split val --num-samples "$NUM_GEN" \
        --out-dir "$out_dir" --copy-reference \
        2>&1 | tee "$out_dir.log" || echo "  FAILED: K=$K (continuing)" >&2
done

# ---------------------------------------------------------------- 3. score them
echo
echo "==================== INTELLIGIBILITY SCORING ===================="
CTRL="$OUTPUTS_ROOT/generations/control_lanes_20260831_0735"
DIRS="$CTRL/real_speech $CTRL/vae_roundtrip $OUT_ROOT/oracle"
for K in $CODES; do
    [[ -f "$OUT_ROOT/quantized_k${K}/manifest.jsonl" ]] && DIRS="$DIRS $OUT_ROOT/quantized_k${K}"
done

CUDA_VISIBLE_DEVICES=0 python scripts/a2s/score_intelligibility.py \
    --dirs $DIRS --ctc-checkpoint "$CTC_CKPT" --device cuda \
    --out "$OUTPUTS_ROOT/intelligibility_codebook_gate_${STAMP}.json" \
    2>&1 | tee "$OUTPUTS_ROOT/logs/slurm/codebook_gate_score_${STAMP}.log"

echo
echo "=========================================================="
echo "THE GATE: compare quantized_k* against oracle in the table above."
echo "  close to oracle          -> representation is faithful; build the predictor at that K"
echo "  collapsed toward ~0.95   -> frame-wise VQ loses phoneme identity; discrete plan needs"
echo "                              a different unit (residual VQ, or HuBERT-style units)"
echo
echo "listen: $OUT_ROOT/{oracle,quantized_k*}/audio/"
echo "codebooks: $CB_DIR/codebook_k*.pt  (+ codebook_manifest.json for the fit stats)"
echo "=========================================================="
echo "OK"
