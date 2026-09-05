#!/usr/bin/env bash
# Score the 442808 generations. Scoring only -- nothing is generated, nothing is trained.
#
# WHY THIS EXISTS. Job 442808 generated all 36 lanes and 576 wavs successfully in ~3 minutes,
# then died in its scoring phase on `WhisperASR.__init__() missing 1 required positional
# argument: 'model_path'`. The signature was guessed rather than read -- the exact failure mode
# CLAUDE.md names ("verify APIs against installed source"), and the second time in this repo.
# The audio is intact and needs no regeneration; only the two scoring passes must be redone.
#
# FIXED AND VERIFIED SINCE: score_memorisation.py now calls
# `WhisperASR(project_paths.WHISPER_LARGE_V3_PATH, args.device)`, matching src/salm/a2s/asr.py
# and score_intelligibility.py:150, and the path was confirmed to exist with a config.json.
# The scorer was then run end to end against these real manifests with stub ASRs on CPU:
#   perfect-ASR stub -> ref_wer_gap -1.27, longest_span_gap +12.12   (correctly signed)
#   empty-ASR stub   -> both gaps exactly 0.00                       (control not leaking)
# So the only thing unexercised below is Whisper itself.
#
# ======================= EXPECTED RESULTS (decided before running) =======================
# Calibration from the stub run, needed to read the numbers: these utterances average ~13
# words, the mismatched-arm span baseline is ~1.1 words, and a verbatim transcript scores
# longest_span_gap +12. So the scale runs 0 (nothing reproduced) to +12 (verbatim).
#
#   THE DECISIVE LANE: split__best__train__* -- the overfit checkpoint (~ep90, train_code_loss
#   0.21) generating from TRAIN ambience. It is a free upper bound on everything downstream of
#   the LM.
#
#   A. train span_gap >= ~6 while val/test sit near 0
#        -> codebook + renderer CAN carry long coherent speech; the islands are a
#           GENERALISATION failure. Next move is data and LM regularisation, and the two-level
#           hierarchy is unnecessary.
#   B. train span_gap ~ 0 too, still 3-5 word islands where the LM has all but memorised
#        -> bottleneck is DOWNSTREAM of the LM. Since the oracle lane renders near-verbatim
#           from TRUE codes, this localises the failure to the gap between nearly-right and
#           exactly-right codes: the representation is not robust to small code errors.
#   C. val and test disagree -> split problem; every val number in the project needs re-reading
#           before anything else proceeds.
#
#   CHECK BEFORE A/B/C: if longest_span_gap is ~0 in EVERY lane including train, the metric did
#   not discriminate on real audio and no conclusion may be drawn from it. The stub run rules
#   out a coding error, but not Whisper being too weak on this audio to recover shared spans.
#
#   ALSO EXPECTED, from the 442807 result: `best` should NOT be uniformly worse than
#   `best_loss` on val. inter_asr_dist across the 442807 ladder was flat-to-improving in 4 of 6
#   cells while val loss doubled. If the memorisation metric ALSO fails to separate ep9 from
#   ep90 on train -- where the loss difference is 1.34 vs 0.21 -- then these audio metrics are
#   collectively insensitive to model quality and the ear is the only instrument left.
#
#   SEED LANES, read separately: per-clip agreement across the 6 seeds. Same topic with
#   different wording = the scene genuinely constrains content. Six unrelated topics = it does
#   not, and scene_gap needs replacing rather than reinterpreting.
# =========================================================================================
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_rescore_split_seed.sbatch
#   GEN_ROOT=/path/to/other_root sbatch submission_scripts/run_tamia_a2s_rescore_split_seed.sbatch
#SBATCH --job-name=salm-a2s-rescore
#SBATCH --account=aip-csubakan
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

# Default to the newest split_seed_* root rather than a hardcoded stamp: 442808's own log
# reported a stamp one minute off from the directory that was actually created, and a
# hardcoded guess would silently score nothing.
GEN_ROOT="${GEN_ROOT:-$(ls -dt "$OUTPUTS_ROOT"/generations/split_seed_* 2>/dev/null | head -1)}"
[[ -n "$GEN_ROOT" && -d "$GEN_ROOT" ]] || { echo "no split_seed_* generation root found" >&2; exit 1; }
CTC_CKPT="$OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt"
[[ -f "$CTC_CKPT" ]] || { echo "MISSING: $CTC_CKPT" >&2; exit 1; }
STAMP="$(basename "$GEN_ROOT" | sed 's/^split_seed_//')"

DIRS=""
N=0
for d in "$GEN_ROOT"/*/; do
    [[ -f "$d/manifest.jsonl" ]] || continue
    DIRS="$DIRS ${d%/}"; N=$((N + 1))
done
[[ "$N" -ge 1 ]] || { echo "no lanes with a manifest under $GEN_ROOT" >&2; exit 1; }

salm_print_header "RESCORE 442808: $N lanes under $(basename "$GEN_ROOT")"
echo "wavs on disk: $(find "$GEN_ROOT" -name '*.wav' | wc -l)"

echo
echo "==================== MEMORISATION / SPAN (the decisive one) ===================="
# shellcheck disable=SC2086
python scripts/a2s/score_memorisation.py \
    --dirs $DIRS --device cuda \
    --out "$OUTPUTS_ROOT/memorisation_${STAMP}.json"

echo
echo "==================== INTELLIGIBILITY (scale only) ===================="
# shellcheck disable=SC2086
python scripts/a2s/score_intelligibility.py \
    --dirs $DIRS --ctc-checkpoint "$CTC_CKPT" --device cuda \
    --out "$OUTPUTS_ROOT/intelligibility_split_seed_${STAMP}.json"

echo
echo "=========================================================="
echo "READ IN THIS ORDER:"
echo "  1. is longest_span_gap ~0 in EVERY lane, train included? -> metric failed, stop here"
echo "  2. train vs val at split__best__*  -> outcome A (generalisation) or B (representation)"
echo "  3. val vs test                     -> must agree, or outcome C (split problem)"
echo "  4. seed__* lanes                   -> does the scene pin the topic across 6 seeds?"
echo "listen: $GEN_ROOT/*/audio/"
echo "=========================================================="
echo "OK"
