#!/usr/bin/env bash
# Score the matched content-source lanes AGAINST REAL POSITIVE CONTROLS.
#
# Why the controls matter: every lane ever scored landed in 0.955-0.993 inter-ASR
# distance, a 3.8% range against a ceiling of 1.0, and score_intelligibility.py's own
# pre-registered control ("on REAL target speech, inter-ASR distance ~0.2-0.3") was never
# actually built or run. Without a measured floor, "flat across all variants" cannot be
# distinguished from "all variants below the metric's resolution".
#
# EXPECTED RESULTS, pre-registered:
#   real_speech lane should read FAR below the generated lanes (expect ~0.42-0.50 -- the
#     CTC probe's own val WER on real speech is 0.4145, so 0.2-0.3 in the docstring is
#     optimistic). vae_roundtrip should be close behind it: that pair brackets the best any
#     latent-diffusion model on this stack could sound.
#   - Controls low (~0.45) AND all generated lanes ~0.97  -> the metric is sound and the
#     generations genuinely contain no word-level content. Believe the flat result.
#   - Controls ALSO ~0.9                                  -> the metric is saturated and has
#     never been able to see what we were asking it. Every "flat" conclusion is void and the
#     metric must be replaced before any further steering on it.
#   - oracle separating from shuffled at ep99             -> the content channel binds
#     phonetic identity, not just presence. That is the whole question.
#SBATCH --job-name=salm-a2s-score-matched
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

GEN_ROOT="${GEN_ROOT:-$RUNS_ROOT/a2s_audioldm_clapseq_xformer/content_xattn_falsify_100ep/generations/rank2_matched_full_432420}"
CTC_CKPT="${CTC_CKPT:-$OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt}"
NUM="${NUM:-16}"
STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"
CTRL_ROOT="$OUTPUTS_ROOT/generations/control_lanes_$STAMP"

[[ -d "$GEN_ROOT" ]] || { echo "MISSING: $GEN_ROOT" >&2; exit 1; }
[[ -f "$CTC_CKPT" ]] || { echo "MISSING: $CTC_CKPT" >&2; exit 1; }

salm_print_header "Building positive-control lanes, then scoring matched content sources"

python scripts/a2s/make_control_lanes.py \
    --out-root "$CTRL_ROOT" --split val --num-samples "$NUM" --device cuda

DIRS=("$CTRL_ROOT/real_speech" "$CTRL_ROOT/vae_roundtrip")
for s in oracle retrieval shuffled zero; do
    for c in ep19 ep99 best; do
        [[ -f "$GEN_ROOT/$s/$c/manifest.jsonl" ]] && DIRS+=("$GEN_ROOT/$s/$c")
    done
done
echo "scoring ${#DIRS[@]} lanes"

python scripts/a2s/score_intelligibility.py \
    --dirs "${DIRS[@]}" --ctc-checkpoint "$CTC_CKPT" --device cuda \
    --out "$OUTPUTS_ROOT/intelligibility_matched_$STAMP.json"

echo "=========================================================="
echo "READ THE real_speech AND vae_roundtrip ROWS FIRST -- they calibrate everything else."
echo "=========================================================="
