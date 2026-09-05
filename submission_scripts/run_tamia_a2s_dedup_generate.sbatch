#!/usr/bin/env bash
# Generation for the run-length (dedup) models. Standalone -- reads the checkpoints job 439442
# already trained, so nothing is retrained.
#
# WHY THIS EXISTS SEPARATELY. Job 439442 trained all four configs successfully (100 epochs
# each) and then died in its own generation phase with `soundfile ... System error` on write:
# /scratch hit 100%, 0 bytes free. Two lanes had completed. The models are fine; only the
# plumbing broke.
#
# DELIBERATELY TRIMMED, 13 lanes not 43. The original sweep was greedy, and the late
# checkpoints were never the ones worth hearing: val_loss bottoms at epoch 9 and nearly
# DOUBLES by epoch 64 (2.37 -> 4.58 for K=512), so ep39 and `best` (~ep90) are substantially
# memorisation. The frame-level run already established that the overfit end sounds no better
# than mid-training. So this covers the loss minimum and just past it, and nothing else.
#
#   best_loss (~ep9) x 3 K x 2 temps  = 6   the honest checkpoint, at each run's own optimum
#   ep19             x 3 K x 2 temps  = 6   just past it, not yet badly overfit
#   zero control                      = 1   without it "sounds like speech" is unfalsifiable
#                                      13   ~204 MB of wavs
#
# WHAT THE TRAINING ALREADY ESTABLISHED (439442), measured at each run's own loss minimum:
#
#   run              min loss  @ep   scene_gap   dur acc   tokens/true
#   dedup_k512        2.3684    9     +0.0414     0.709      135/129
#   dedup_k1024       2.4177   10     +0.0370     0.738      144/142
#   dedup_k2048       2.4927    9     +0.0322     0.763      159/152
#   dedup_k2048_s1    2.4985    9     +0.0310     0.763      153/152
#
# The frame-level floor was +0.021, INVARIANT across seven configs (3 K, 3x dropout, 10x
# weight decay, two depths, label smoothing). Run-length gives +0.031 to +0.041, and the seed
# pair (+0.0322 vs +0.0310) puts run-to-run variance at ~0.001 -- so the roughly 2x increase
# is on the order of 12 standard deviations. That is the first properly error-barred result in
# this project. Token counts tracking the truth rules out the degenerate "few tokens, long
# durations" failure, and duration accuracy 0.71-0.76 means re-expansion is honest rather than
# padded into shape.
#
# ======================= EXPECTED RESULTS (decided before running) =======================
# NONE OF THE ABOVE MEASURES COHERENCE, which is the entire question. The frame-level model
# also looked healthy on its metrics and still produced 3-5 word islands; `inter_asr_dist`
# rewards saying something CLEARLY over saying the right thing, and has already ranked LM
# lanes above their own oracle ceiling. So the scores here only place the audio on a scale.
#
#   THE TEST IS BY EAR. Walk one sample id across these lanes and the FRAME-LEVEL lanes in
#   generations/content_lm_gen_20260903_0328 (same renderer, same sample ids, same seeds --
#   only the representation differs). Do the 3-5 word islands become 6-8 word phrases?
#
#   SUCCESS -- longer coherent spans. Granularity was the binding constraint, and the next
#       move is pushing further in the same direction (a coarser linguistic unit).
#   FAILURE -- islands unchanged. Then 2x compression was not enough. Since more epochs
#       (438059) and regularization (438062) are already eliminated by measurement, the
#       answer becomes a two-level semantic/acoustic hierarchy: a coarse linguistic layer for
#       planning plus these codes for fidelity. Reached by elimination, not by guessing.
#   The doubled scene_gap is consistent with BOTH -- better conditioning of the same short
#   islands is exactly as plausible as longer ones. That is why it is not the verdict.
#
# DISK. /scratch hit 100% during job 439442 and our own checkpoints were only 3.4 GB of the
# ~32 GB that disappeared, so something else on this shared filesystem is growing. The job
# checks free space before starting and aborts early rather than dying mid-write, and it
# re-checks between lanes so a partial sweep still leaves complete lanes behind.
# =========================================================================================
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_dedup_generate.sbatch
#SBATCH --job-name=salm-a2s-dedup-gen
#SBATCH --account=aip-csubakan
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

CTC_CKPT="$OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt"
RENDERER="$RUNS_ROOT/a2s_audioldm_clapseq_xformer/content_concat_v4/checkpoints/checkpoint_best.pt"
NUM_GEN="${NUM_GEN:-16}"
TEMPS="${TEMPS:-0.7 1.0}"
# EVERY checkpoint on disk, and the LAST epoch is never optional. The user has asked for
# last-epoch generations repeatedly; this sweep trimmed to best_loss+ep19 under disk pressure
# on 2026-09-05 and, worse, the cleanup that preceded it deleted ep99/last outright -- so the
# literal final epoch is unrecoverable for the dedup runs without retraining. Defaulting to
# the full surviving spread so a trim has to be a deliberate override, never a default.
CKPT_TAGS="${CKPT_TAGS:-checkpoint_best_loss checkpoint_ep19 checkpoint_ep39 checkpoint_best}"
KS="${KS:-512 1024 2048}"
MIN_FREE_MB="${MIN_FREE_MB:-800}"
STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"
OUT_ROOT="$OUTPUTS_ROOT/generations/dedup_gen2_${STAMP}"

for F in "$CTC_CKPT" "$RENDERER"; do
    [[ -f "$F" ]] || { echo "MISSING: $F" >&2; exit 1; }
done

free_mb() { df -Pm /scratch/l/lovenya | awk 'NR==2{print $4}'; }

# Fail fast rather than mid-write. Job 439442 died with soundfile "System error" partway
# through a lane, leaving truncated output; an up-front check turns that into a clean refusal.
FREE=$(free_mb)
echo "free space: ${FREE} MB (need >= ${MIN_FREE_MB} MB)"
if [[ "$FREE" -lt "$MIN_FREE_MB" ]]; then
    echo "ABORT: only ${FREE} MB free on /scratch. Generation needs ~204 MB of wavs plus" >&2
    echo "       headroom; running now would repeat 439442's mid-write failure." >&2
    exit 1
fi

salm_print_header "DEDUP GENERATION: 13 lanes from job 439442's checkpoints (no retraining)"
echo "compare against: $OUTPUTS_ROOT/generations/content_lm_gen_20260903_0328 (frame-level)"

DONE=0
for K in $KS; do
  for TAG in $CKPT_TAGS; do
    LM_CKPT="$OUTPUTS_ROOT/content_lm/dedup_k${K}/checkpoints/${TAG}.pt"
    [[ -f "$LM_CKPT" ]] || { echo "  skip (absent): dedup_k${K}/${TAG}" >&2; continue; }
    for T in $TEMPS; do
        if [[ "$(free_mb)" -lt 300 ]]; then
            echo "  STOPPING: free space fell below 300 MB after $DONE lanes." >&2
            echo "  Completed lanes are intact; re-run when there is room." >&2
            break 3
        fi
        out_dir="$OUT_ROOT/k${K}__${TAG#checkpoint_}__t${T}"
        mkdir -p "$out_dir"
        echo; echo "=== K=$K $TAG t=$T (run-length, AMBIENT ONLY) ==="
        CUDA_VISIBLE_DEVICES=0 python -m salm.generate --mode a2s --direction a2s \
            --checkpoint "$RENDERER" --content-source codebook_lm \
            --content-lm-checkpoint "$LM_CKPT" --content-lm-temperature "$T" \
            --ctc-checkpoint "$CTC_CKPT" --split val --num-samples "$NUM_GEN" \
            --out-dir "$out_dir" --copy-reference \
            2>&1 | tee "$out_dir.log" || echo "  FAILED: K=$K $TAG t=$T" >&2
        DONE=$((DONE + 1))
    done
  done
done

mkdir -p "$OUT_ROOT/zero"
echo; echo "=== zero content (null control) ==="
CUDA_VISIBLE_DEVICES=0 python -m salm.generate --mode a2s --direction a2s \
    --checkpoint "$RENDERER" --content-source zero --ctc-checkpoint "$CTC_CKPT" \
    --split val --num-samples "$NUM_GEN" --out-dir "$OUT_ROOT/zero" --copy-reference \
    2>&1 | tee "$OUT_ROOT/zero.log" || echo "  FAILED: zero" >&2

echo
echo "==================== SCORING ===================="
CTRL="$OUTPUTS_ROOT/generations/control_lanes_20260831_0735"
DIRS="$CTRL/real_speech $CTRL/vae_roundtrip"
# The frame-level lanes at the SAME checkpoints, so the comparison is in one table rather
# than across two logs from different days.
FRAME="$OUTPUTS_ROOT/generations/content_lm_gen_20260903_0328"
for d in "$FRAME/k512__best_loss__t0.7" "$FRAME/k1024__best_loss__t0.7" \
         "$FRAME/k2048__best_loss__t0.7" "$FRAME/k2048__ep19__t0.7" "$FRAME/zero"; do
    [[ -f "$d/manifest.jsonl" ]] && DIRS="$DIRS $d"
done
for d in "$OUT_ROOT"/*/; do
    [[ -f "$d/manifest.jsonl" ]] && DIRS="$DIRS ${d%/}"
done
CUDA_VISIBLE_DEVICES=0 python scripts/a2s/score_intelligibility.py \
    --dirs $DIRS --ctc-checkpoint "$CTC_CKPT" --device cuda \
    --out "$OUTPUTS_ROOT/intelligibility_dedup2_${STAMP}.json" \
    2>&1 | tee "$OUTPUTS_ROOT/logs/slurm/dedup_gen2_score_${STAMP}.log"

echo
echo "=========================================================="
echo "listen: $OUT_ROOT/k*__*__t*/audio/"
echo "  dedup_gen2_* lanes are RUN-LENGTH; content_lm_gen_20260903_0328 lanes are FRAME-LEVEL"
echo
echo "THE QUESTION no metric here answers: same sample id, dedup vs frame-level --"
echo "do the 3-5 word islands become 6-8 word phrases?"
echo "free space remaining: $(free_mb) MB"
echo "=========================================================="
echo "OK"
