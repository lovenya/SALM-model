#!/usr/bin/env bash
# 128 GREEDY GENERATIONS PER CODEBOOK AT best_loss, ALL TRANSCRIBED. No retraining.
#
# WHY. Job 443379 produced the best result this project has recorded: k2048 at best_loss (epoch
# 9) with greedy decoding scores inter_asr_dist 0.338 against a VAE round-trip CEILING of 0.372
# and real speech at 0.315, with real_word_rate 0.971 and no mode collapse (cross-clip
# diversity 0.927). The user listened and called it "super better", with the honest
# qualification that it is "not always coherent -- sometimes some seconds are not that
# intelligible", and asked for ~100 transcribed generations to read rather than 16.
#
# Sixteen clips cannot separate "this model is good" from "these sixteen clips were kind".
# 128 per codebook can, and transcripts are far faster to review than audio.
#
# TWO CORRECTIONS THIS JOB BAKES IN.
#   1. best_loss, not best. `checkpoint_best` is selected on val_scene_gap, and that metric
#      rises monotonically WITH overfitting (+0.041 at ep9 to +0.134 at ep87 for k512). So
#      `best` is effectively the most overfit surviving checkpoint, chosen by a criterion that
#      rewards memorisation. Its val_loss is 5.01 against best_loss's 2.37. Under sampling the
#      two looked alike; under greedy best_loss wins by 0.13-0.17 inter_asr_dist in all three
#      codebooks. Every lane here is best_loss (epoch 9 for K=512/2048, epoch 10 for K=1024).
#   2. Greedy, not sampling. The "never argmax" reasoning in this codebase was an argument
#      about CONTINUOUS means imported into discrete token space, where argmax returns the most
#      likely sequence -- a real utterance, not an average of utterances.
#
# ======================= EXPECTED RESULTS (decided before running) =======================
# The point is a document to READ, not a number to quote. Scoring runs so the lanes sit on the
# established scale, but the deliverable is transcript_review_*.md.
#
#   THE CTC LINE IS THE GUARD. Whisper has a language model and will hallucinate a fluent
#   sentence out of mush; the CTC probe has none. Where both agree, the words are really in the
#   waveform. Where Whisper is fluent and CTC is gibberish, Whisper is filling in and the audio
#   is NOT intelligible. Any read of these transcripts that ignores the CTC line will
#   overestimate the model, which is precisely the error to avoid here.
#   The review doc sorts WORST FIRST for the same reason -- a reviewer who stops early should
#   have stopped on the failures.
#
#   SUCCESS -- inter_asr_dist at 128 clips stays near the 16-clip value (0.338 for k2048).
#       The result generalises beyond a lucky sample, and the next question is which specific
#       clips fail and whether they share a property (scene type, reference length, ambient
#       energy) that suggests the mechanism.
#   REGRESSION -- dist rises materially, say past ~0.45. Then 0.338 was small-sample luck; the
#       honest headline becomes the 128-clip number and the greedy claim needs softening in
#       every doc that now records it.
#   EITHER WAY the user's specific complaint -- some seconds are unintelligible -- should show
#       up as a subset of clips where Whisper and CTC diverge sharply. Counting those is what
#       turns "sometimes" into a rate, and the rate is what the next experiment has to move.
#
#   NOT AN OUTCOME: a low inter_asr_dist on its own. Greedy picks high-confidence codes, which
#   is close to tautologically what makes two recognisers agree. real_word_rate (0.971 at 16
#   clips) and the transcripts themselves are the load-bearing evidence.
# =========================================================================================
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_greedy_transcribe.sbatch
#SBATCH --job-name=salm-a2s-greedy-transcribe
#SBATCH --account=aip-csubakan
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

KS="${KS:-2048 1024 512}"
NGEN="${NGEN:-128}"
SPLIT="${SPLIT:-val}"
TAG="${TAG:-checkpoint_best_loss}"     # epoch 9/10 -- the val optimum, NOT checkpoint_best
CTC_CKPT="$OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt"
RENDERER="$RUNS_ROOT/a2s_audioldm_clapseq_xformer/content_concat_v4/checkpoints/checkpoint_best.pt"
STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"
OUT_ROOT="$OUTPUTS_ROOT/generations/greedy128_${STAMP}"

for F in "$CTC_CKPT" "$RENDERER"; do
    [[ -f "$F" ]] || { echo "MISSING: $F" >&2; exit 1; }
done
FREE=$(df -Pm /scratch/l/lovenya | awk 'NR==2{print $4}')
echo "free space: ${FREE} MB (need >= 2000 for ~380 MB of wavs plus headroom)"
[[ "$FREE" -ge 2000 ]] || { echo "ABORT: only ${FREE} MB free" >&2; exit 1; }

salm_print_header "GREEDY x $NGEN at $TAG, K in {$KS}"

# One codebook per GPU: three independent lanes, no sharding logic needed.
gpu=0
for K in $KS; do
    LM="$OUTPUTS_ROOT/content_lm/dedup_k${K}/checkpoints/${TAG}.pt"
    [[ -f "$LM" ]] || { echo "  skip (absent): dedup_k${K}/${TAG}" >&2; continue; }
    out_dir="$OUT_ROOT/k${K}__${TAG#checkpoint_}__greedy"
    mkdir -p "$out_dir"
    echo "[gpu$gpu] k${K}"
    CUDA_VISIBLE_DEVICES="$gpu" python -m salm.generate --mode a2s --direction a2s \
        --checkpoint "$RENDERER" --content-source codebook_lm \
        --content-lm-checkpoint "$LM" --content-lm-temperature 1.0 --content-lm-top-k 1 \
        --ctc-checkpoint "$CTC_CKPT" --split "$SPLIT" --num-samples "$NGEN" \
        --start-index 0 --seed 1337 --out-dir "$out_dir" --copy-reference \
        > "$out_dir.log" 2>&1 || echo "  FAILED: k${K}" >&2 &
    gpu=$((gpu + 1))
done
wait
echo "all lanes done"

DIRS=""
for d in "$OUT_ROOT"/*/; do [[ -f "$d/manifest.jsonl" ]] && DIRS="$DIRS ${d%/}"; done
[[ -n "$DIRS" ]] || { echo "no lanes produced a manifest" >&2; exit 1; }

echo
echo "==================== SCORING ===================="
CTRL="$OUTPUTS_ROOT/generations/control_lanes_20260831_0735"
SCORES="$OUTPUTS_ROOT/greedy128_intel_${STAMP}.json"
# shellcheck disable=SC2086
python scripts/a2s/score_intelligibility.py \
    --dirs $CTRL/real_speech $CTRL/vae_roundtrip $DIRS \
    --ctc-checkpoint "$CTC_CKPT" --device cuda --out "$SCORES" \
    || { echo "SCORING FAILED -- generations intact under $OUT_ROOT" >&2; exit 1; }

echo
echo "==================== REVIEW DOCUMENT ===================="
REVIEW="$OUTPUTS_ROOT/transcript_review_${STAMP}.md"
python scripts/a2s/export_transcript_review.py \
    --scores "$SCORES" --gen-root "$OUT_ROOT" --limit 200 --out "$REVIEW" \
    || echo "REVIEW EXPORT FAILED -- scores are in $SCORES" >&2

echo
echo "==================== DIVERSITY (mode-collapse guard) ===================="
GROUP_ARGS=(); DET=()
for d in "$OUT_ROOT"/*/; do
    n="$(basename "${d%/}")"
    [[ -f "$d/manifest.jsonl" ]] && { GROUP_ARGS+=(--group "$n" "${d%/}"); DET+=("$n"); }
done
python scripts/a2s/score_plan_horizon.py "${GROUP_ARGS[@]}" --deterministic "${DET[@]}" \
    --ctc-checkpoint "$CTC_CKPT" --device cuda \
    --out "$OUTPUTS_ROOT/greedy128_horizon_${STAMP}.json" \
    || echo "DIVERSITY SCORING FAILED -- generations intact" >&2

echo
echo "=========================================================="
echo "READ:  $REVIEW   (sorted WORST FIRST, on purpose)"
echo "  rsync -avzP -e \"ssh -J rorqual\" \"tamia:$REVIEW\" ~/salm_listen/"
echo "Where Whisper is fluent and the CTC probe is gibberish, Whisper is hallucinating and the"
echo "audio is NOT intelligible. Agreement between the two is the only real evidence."
echo "=========================================================="
echo "OK"
