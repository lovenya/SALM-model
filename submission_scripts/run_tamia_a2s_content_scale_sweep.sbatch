#!/usr/bin/env bash
# Content-strength sweep: scale the concat content map before it reaches the U-Net,
# scale in {0.5, 1.0, 1.5, 2.0, 3.0}, on the ORACLE content source (the one already proven
# near-verbatim at scale=1.0, job 433314) so any effect is isolated from content-source
# quality. NOT classifier-free guidance -- see --content-scale's help in generate.py for why
# a full dual-CFG loop was not attempted unattended overnight.
#
# EXPECTED RESULTS, pre-registered:
#   - intelligibility roughly flat across scale -> the model already uses content near-
#     maximally at its trained weight; amplifying the map further doesn't help, and a real
#     dual-CFG implementation (separate w_desc/w_cont) is unlikely to be worth building.
#   - intelligibility improves with scale > 1, up to a point -> the content channel IS
#     under-weighted relative to what it could contribute; dual CFG becomes worth building
#     next, since it would let scene and content be balanced independently rather than both
#     scaled together as guidance_scale already does for scene alone.
#   - very high scale (2.0, 3.0) breaks down (artifacts, instability) -> expected and fine;
#     the useful range is whatever comes before that.
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_content_scale_sweep.sbatch
#SBATCH --job-name=salm-a2s-content-scale
#SBATCH --account=aip-ravanelm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail
# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

R="$RUNS_ROOT/a2s_audioldm_clapseq_xformer"
CKPT="${CKPT:-$R/content_concat_v4/checkpoints/checkpoint_ep199.pt}"
CTC_CKPT="${CTC_CKPT:-$OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt}"
NUM_GEN="${NUM_GEN:-16}"
STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"
OUT_ROOT="$OUTPUTS_ROOT/generations/content_scale_sweep_${STAMP}"

for F in "$CKPT" "$CTC_CKPT"; do
    [[ -f "$F" ]] || { echo "MISSING: $F" >&2; exit 1; }
done

salm_print_header "Content-scale sweep on oracle content, renderer=$CKPT"

for SCALE in 0.5 1.0 1.5 2.0 3.0; do
    OUT="$OUT_ROOT/scale_${SCALE}"
    mkdir -p "$OUT"
    echo
    echo "=== content_scale=$SCALE ==="
    python -m salm.generate --mode a2s --direction a2s \
        --checkpoint "$CKPT" \
        --content-source oracle --ctc-checkpoint "$CTC_CKPT" --content-scale "$SCALE" \
        --split val --num-samples "$NUM_GEN" \
        --out-dir "$OUT" --copy-reference \
        2>&1 | tee "$OUT.log"
done

echo
echo "=== scoring all scales against the calibrated floor ==="
CTRL=$(ls -d "$OUTPUTS_ROOT"/generations/control_lanes_* 2>/dev/null | tail -1)
if [[ -n "$CTRL" ]]; then
    DIRS=("$CTRL/real_speech" "$CTRL/vae_roundtrip")
    for SCALE in 0.5 1.0 1.5 2.0 3.0; do DIRS+=("$OUT_ROOT/scale_${SCALE}"); done
    python scripts/a2s/score_intelligibility.py \
        --dirs "${DIRS[@]}" --ctc-checkpoint "$CTC_CKPT" --device cuda \
        --out "$OUTPUTS_ROOT/intelligibility_scale_sweep_${STAMP}.json"
else
    echo "no control_lanes_* dir found -- skipping scoring, generations still saved"
fi

echo "=========================================================="
echo "listen: $OUT_ROOT/scale_<value>/audio/"
echo "=========================================================="
