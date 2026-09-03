#!/usr/bin/env bash
# THE GRANULARITY EXPERIMENT: run-length content tokens vs frame-level, controlled.
#
# Four configs train in PARALLEL (one per GPU), then generation sweeps the full checkpoint
# spread. Hyperparameters are matched EXACTLY to the frame-level baseline (job 436344: 100
# epochs, batch 64, d_model 512, 6 layers, dropout 0.1, weight_decay 0.01) so the only
# difference between these runs and those is the REPRESENTATION. Anything else would make the
# comparison unreadable.
#
# WHY THIS AND NOTHING ELSE. The frame-level ContentLM already generates per-scene speech from
# ambient audio alone -- across three separately-trained K, roughly two of three lanes per clip
# are clearly scene-appropriate when checked against the ambient captions ("the birds are
# getting me crazy" for birds chirping, "Turn the key" for an engine). What it cannot do is
# hold a plan: 3-5 words, then the sentence restarts. Both alternative explanations are
# eliminated by measurement, not argument:
#     more epochs      -> job 438059, flat epoch ladder; ep99 no better than ep39, and the
#                         user's own listening across the full spread found them "all the same"
#     regularization   -> job 438062, scene_gap at the loss minimum invariant at +0.021 across
#                         seven configs (3 K, 3x dropout, 10x weight decay, 2 depths, label
#                         smoothing). That invariance measures the DATA, not the model.
# What remains is that the model plans over 256 tokens at 25 fps, one 40 ms acoustic state per
# step, and holds a plan for ~30 of them.
#
# HONEST EXPECTATION, stated so the result is not over-read either way. Measured compression is
# 2.0x, NOT the 3-5x initially assumed: 45% of adjacent frames are already identical and the
# differing ones sit at cosine ~0.79. So this halves the planning horizon; it does not dissolve
# it. The smoke test (438114) already confirmed the pipeline runs, the duration head learns
# fast (70.6% exact-match accuracy at epoch 8), and scene_gap tracks the frame-level trajectory
# (+0.0177 at ep8 vs the frame-level +0.021 at ~ep10).
#
# ======================= EXPECTED RESULTS (decided before running) =======================
# THE TEST IS BY EAR, and it is specific: do the 3-5 word islands become 6-8 word phrases?
# No metric here measures coherence -- inter_asr_dist scores whether words are present and
# clear, and demonstrably rewards saying something clearly over saying the right thing (LM
# lanes have scored BETTER than their own oracle ceiling). So the numbers below gate whether
# the generations are worth listening to; they do not decide the question.
#
#   scene_gap >= ~+0.021 at the loss minimum -> the shorter horizon did not cost the scene
#       signal. NECESSARY, not sufficient. Below that means the representation change hurt
#       conditioning, and the generations are not worth much attention.
#   val_dur_acc well above chance -> the duration head works, so re-expansion lands near 256
#       frames honestly rather than being padded/truncated into shape.
#   val_pred_tokens near the true mean (~128 for K=512) -> the model is not emitting a few
#       tokens with long durations, which would be smeared static content scoring fine on
#       every loss while sounding worse than the frame-level model.
#
#   SUCCESS  -- longer coherent spans by ear. Then granularity was the binding constraint and
#       the next step is pushing it further (a coarser linguistic unit).
#   FAILURE  -- islands unchanged at ~3-5 words. Then 2x was not enough, and since more
#       epochs and regularization are already eliminated, the answer is the two-level
#       semantic/acoustic hierarchy: a coarse linguistic layer for planning plus these codes
#       for fidelity. That is a real conclusion, not a dead end -- it is what AudioLM-family
#       systems do, and it would be reached by elimination rather than by guessing.
#
# The 4th GPU runs K=2048 at a SECOND SEED. This project has repeatedly read conclusions off
# single runs; two seeds of one config price the run-to-run variance, which tells us whether
# differences between K are real or noise. Cheap, since the GPU would otherwise idle.
# =========================================================================================
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_content_lm_dedup.sbatch
#SBATCH --job-name=salm-a2s-dedup
#SBATCH --account=aip-ravanelm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

CB_DIR="$OUTPUTS_ROOT/content_codebook"
CTC_CKPT="$OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt"
RENDERER="$RUNS_ROOT/a2s_audioldm_clapseq_xformer/content_concat_v4/checkpoints/checkpoint_best.pt"
EPOCHS="${EPOCHS:-100}"
# 0.215 min/epoch measured on the smoke (d=256, L=4); the real model is ~2x wider and deeper,
# so ~0.43 min/epoch -> 100 epochs ~= 43 min. 55 is the graceful-truncation valve.
MAX_MINUTES="${MAX_MINUTES:-55}"
NUM_GEN="${NUM_GEN:-16}"
TEMPS="${TEMPS:-0.7 1.0}"
CKPT_TAGS="${CKPT_TAGS:-checkpoint_best_loss checkpoint_ep19 checkpoint_ep39 checkpoint_ep59 checkpoint_ep79 checkpoint_ep99 checkpoint_best}"
STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"
OUT_ROOT="$OUTPUTS_ROOT/generations/dedup_gen_${STAMP}"

for F in "$CTC_CKPT" "$RENDERER"; do
    [[ -f "$F" ]] || { echo "MISSING: $F" >&2; exit 1; }
done

# name|K|seed
CONFIGS=(
    "dedup_k512|512|0"
    "dedup_k1024|1024|0"
    "dedup_k2048|2048|0"
    "dedup_k2048_s1|2048|1"
)

salm_print_header "DEDUP: ${#CONFIGS[@]} configs in PARALLEL, $EPOCHS epochs, matched to job 436344"
echo "frame-level baselines to compare against (job 436344, same hyperparameters):"
echo "  k512  val_loss 1.1428 @ep7  scene_gap there +0.0220   max +0.0874 @ep90"
echo "  k1024 val_loss 1.2930 @ep7  scene_gap there +0.0203   max +0.0807 @ep86"
echo "  k2048 val_loss 1.4397 @ep7  scene_gap there +0.0210   max +0.0590 @ep87"
echo

mkdir -p "$OUTPUTS_ROOT/content_lm"
PIDS=()
for i in "${!CONFIGS[@]}"; do
    IFS='|' read -r NAME K SEED <<< "${CONFIGS[$i]}"
    CB="$CB_DIR/codebook_k${K}.pt"
    [[ -f "$CB" ]] || { echo "MISSING codebook $CB -- skipping $NAME" >&2; continue; }
    echo "GPU $i <- $NAME (K=$K seed=$SEED)"
    CUDA_VISIBLE_DEVICES=$i python -u scripts/a2s/train_content_lm_dedup.py \
        --codebook "$CB" --run-name "$NAME" --seed "$SEED" \
        --epochs "$EPOCHS" --batch-size 64 \
        --d-model 512 --layers 6 --heads 8 --dropout 0.1 --weight-decay 0.01 \
        --max-run 32 --sample-every 10 --save-every 20 --max-minutes "$MAX_MINUTES" \
        > "$OUTPUTS_ROOT/content_lm/${NAME}_train.log" 2>&1 &
    PIDS+=("$!")
done

echo
echo "launched ${#PIDS[@]} concurrent runs; waiting ..."
# Collect every exit status rather than let `wait` short-circuit: one config failing must not
# discard the other three, and `set -e` would abort the job on the first non-zero.
FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then echo "  config $i OK"
    else echo "  config $i FAILED (see its own log)" >&2; FAILED=$((FAILED+1)); fi
done
echo "training done: $(( ${#PIDS[@]} - FAILED ))/${#PIDS[@]} succeeded"

echo
echo "==================== TRAINING SUMMARY ===================="
# argv not os.environ (_salm_env.sh assigns without exporting), and `|| true` because a
# cosmetic summary must never fail the job -- that combination cost job 436256 its dependent
# generation run on 2026-09-02.
python - "$OUTPUTS_ROOT" <<'PY' || true
import json, pathlib, sys
root = pathlib.Path(sys.argv[1]) / "content_lm"
print(f"{'run':>18} {'eps':>4} {'min loss':>9} {'@ep':>4} {'gap there':>10} "
      f"{'max gap':>8} {'dur acc':>8} {'tok/true':>12}")
for run in sorted(root.glob("dedup_k*")):
    h = run / "history.json"
    if not h.exists():
        print(f"{run.name:>18}  no history.json"); continue
    hist = json.loads(h.read_text())
    if not hist: continue
    lo = min(hist, key=lambda r: r["val_loss"])
    bg = max(hist, key=lambda r: r["val_scene_gap"])
    sm = [r for r in hist if "val_pred_tokens" in r]
    tok = f"{sm[-1]['val_pred_tokens']:.0f}/{sm[-1]['true_mean_tokens']:.0f}" if sm else "--"
    print(f"{run.name:>18} {len(hist):>4} {lo['val_loss']:>9.4f} {lo['epoch']:>4} "
          f"{lo['val_scene_gap']:>+10.4f} {bg['val_scene_gap']:>+8.4f} "
          f"{lo['val_dur_acc']:>8.3f} {tok:>12}")
print()
print("scene_gap at the loss minimum vs the frame-level +0.021 floor is the gate;")
print("tok/true far below 1 means smeared static content regardless of the losses.")
PY
echo "=========================================================="

# ---------------------------------------------------------------- generation
for K in 512 1024 2048; do
  RUN="dedup_k${K}"
  for TAG in $CKPT_TAGS; do
    LM_CKPT="$OUTPUTS_ROOT/content_lm/$RUN/checkpoints/${TAG}.pt"
    [[ -f "$LM_CKPT" ]] || { echo "  skip (absent): $RUN/$TAG" >&2; continue; }
    for T in $TEMPS; do
        out_dir="$OUT_ROOT/k${K}__${TAG#checkpoint_}__t${T}"
        mkdir -p "$out_dir"
        echo; echo "=== K=$K $TAG t=$T (run-length, ambient only) ==="
        CUDA_VISIBLE_DEVICES=0 python -m salm.generate --mode a2s --direction a2s \
            --checkpoint "$RENDERER" --content-source codebook_lm \
            --content-lm-checkpoint "$LM_CKPT" --content-lm-temperature "$T" \
            --ctc-checkpoint "$CTC_CKPT" --split val --num-samples "$NUM_GEN" \
            --out-dir "$out_dir" --copy-reference \
            2>&1 | tee "$out_dir.log" || echo "  FAILED: K=$K $TAG t=$T" >&2
    done
  done
done

# The null control, at matched noise. Without it, "sounds like speech" is unfalsifiable.
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
for d in "$OUT_ROOT"/*/; do
    [[ -f "$d/manifest.jsonl" ]] && DIRS="$DIRS ${d%/}"
done
CUDA_VISIBLE_DEVICES=0 python scripts/a2s/score_intelligibility.py \
    --dirs $DIRS --ctc-checkpoint "$CTC_CKPT" --device cuda \
    --out "$OUTPUTS_ROOT/intelligibility_dedup_${STAMP}.json" \
    2>&1 | tee "$OUTPUTS_ROOT/logs/slurm/dedup_gen_score_${STAMP}.log"

echo
echo "=========================================================="
echo "listen: $OUT_ROOT/k*__*__t*/audio/"
echo
echo "THE QUESTION, and no metric here answers it: walk one sample id along the epoch ladder"
echo "and compare against the FRAME-LEVEL generations in content_lm_gen_20260903_0328."
echo "Do the 3-5 word islands become 6-8 word phrases? If yes, granularity was the constraint."
echo "If they are unchanged, 2x was not enough and the answer is a two-level semantic/acoustic"
echo "hierarchy -- reached by elimination, since epochs and regularization are already ruled out."
echo "=========================================================="
echo "OK"
