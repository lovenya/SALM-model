#!/usr/bin/env bash
# STEP 3 of the discrete-content plan: sample content codes from ambient audio alone, and
# find out whether they produce speech.
#
# THIS IS THE DEPLOYABLE PATH. Nothing here reads the target speech and nothing looks up a
# training row. For each unseen validation clip: its own ambient audio -> CLAP -> ContentLM
# -> sampled code sequence -> dequantize to real content frames -> the FROZEN, proven
# content_concat_v4 renderer. If this produces words, it is the first time in this project
# that speech has been generated from a scene without cheating.
#
# WHAT THE PREVIOUS STEPS ESTABLISHED
#   gate (436244): quantized ORACLE content still renders -- K=1024 returned "Come on, old
#     girl, wake up" and K=2048 "This trick works every time. They always bring the food."
#     verbatim. So the discrete representation preserves phoneme identity.
#   LM (436256):  val_scene_gap climbed monotonically (+0.0025 @ep0 -> +0.0259 @ep10), the
#     first non-flat scene-conditioning signal in this project. But it is SMALL: a ~2.6%
#     perplexity reduction from the scene, with the autoregressive context doing most of the
#     work.
#
# ======================= EXPECTED RESULTS (decided before running) =======================
# The open question this job answers, and it does not follow from scene_gap: a positive
# scene_gap proves the scene is USED, not that it is used ENOUGH. Sampling has to tilt far
# enough to commit to different WORDS, and a 2.6% distribution shift may not do that.
# Exactly the same distinction as "bind_gap proves binding, not intelligibility".
#
# Scored on the calibrated content-independent instrument, all lanes at content_scale 1.0:
#     real speech           0.3146      oracle (unquantized) 0.4231
#     vae ceiling           0.3722      oracle_quantized K=1024 0.5480  <- the ceiling FOR
#                                                                          THIS PIPELINE
#     noise floor           ~0.95
#
#   SUCCESS -- codebook_lm scores meaningfully below the `zero` control and produces real
#     words. Anything approaching 0.5480 would mean sampled content is nearly as good as
#     perfectly-quantized oracle content, which would be a genuine result. Even landing
#     clearly between `zero` and 0.5480 is real progress: honest ambient-only speech.
#
#   FAILURE -- codebook_lm is indistinguishable from `zero`. Meaning: the scene signal is
#     real (scene_gap says so) but too weak to change which words get sampled. That is a
#     DIFFERENT and more actionable verdict than the previous four failures -- it says make
#     the conditioning stronger, not that conditioning is impossible. The designated next
#     move would be a contrastive/negative-conditioning term that turns scene_gap itself into
#     a training objective (rewarding the true pairing, penalising mismatched ones) rather
#     than leaving it a passive diagnostic.
#
#   TEMPERATURE IS THE KNOB, so it is swept rather than guessed. Low temperature commits
#     harder but walks back toward argmax -- i.e. toward the conditional mean this whole
#     design exists to escape. High temperature is diverse but incoherent. 0.7 / 1.0 are both
#     run; the sampled-distinct-codes count printed per lane shows where each sits.
#
#   CONTROLS that make the above readable, all at matched diffusion noise (the LM sampler
#     uses a SEPARATE RNG stream so drawing codes cannot shift the denoising noise):
#     `zero`             -- the learned null. If codebook_lm matches this, sampling did nothing.
#     `oracle_quantized` -- the ceiling for this exact pipeline at this exact K.
# =========================================================================================
#
# Usage (as a dependency on the training job):
#   sbatch --dependency=afterok:<train_jobid> submission_scripts/run_tamia_a2s_content_lm_generate.sbatch
#SBATCH --job-name=salm-a2s-lm-gen
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

# All three K, and for each a SPREAD of checkpoints -- not one pick. Two reasons:
#   1. Standing repo rule for checkpoint-comparison generation jobs.
#   2. Specific to this model: scene_gap and overfitting rise TOGETHER, so max-scene_gap and
#      min-val_loss are different epochs with different failure characters, and no metric
#      settles which produces better speech. The ear does.
KS="${KS:-512 1024 2048}"
CKPT_TAGS="${CKPT_TAGS:-checkpoint_best_loss checkpoint_ep39 checkpoint_best}"
# checkpoint_best is selected on val_scene_gap, NOT val_loss -- a checkpoint with the better
# loss and no scene_gap is precisely the failure this pipeline is built to detect.
CTC_CKPT="${CTC_CKPT:-$OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt}"
RENDERER="${RENDERER:-$RUNS_ROOT/a2s_audioldm_clapseq_xformer/content_concat_v4/checkpoints/checkpoint_best.pt}"
NUM_GEN="${NUM_GEN:-16}"
TEMPS="${TEMPS:-0.7 1.0}"
STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"
OUT_ROOT="$OUTPUTS_ROOT/generations/content_lm_gen_${STAMP}"

for F in "$CTC_CKPT" "$RENDERER"; do
    [[ -f "$F" ]] || { echo "MISSING: $F" >&2; exit 1; }
done

salm_print_header "ContentLM generation: ambient -> sampled codes -> frozen renderer"
echo "K values:        $KS"
echo "checkpoint tags: $CKPT_TAGS"
echo "temperatures:    $TEMPS"

python - "$OUTPUTS_ROOT" "$KS" <<'PY' || true
import json, pathlib, sys
root, ks = pathlib.Path(sys.argv[1]) / "content_lm", sys.argv[2].split()
for k in ks:
    h = root / f"content_lm_k{k}_v1" / "history.json"
    if not h.exists():
        print(f"  K={k}: no history.json"); continue
    hist = json.loads(h.read_text())
    bg = max(hist, key=lambda r: r["val_scene_gap"])
    bl = min(hist, key=lambda r: r["val_loss"])
    print(f"  K={k}: {len(hist)} eps | max gap {bg['val_scene_gap']:+.4f} @ep{bg['epoch']} "
          f"(loss {bg['val_loss']:.3f}) | min loss {bl['val_loss']:.3f} @ep{bl['epoch']} "
          f"(gap {bl['val_scene_gap']:+.4f})")
PY

# ---------------------------------------------------------------- the real lanes
for K in $KS; do
  for TAG in $CKPT_TAGS; do
    LM_CKPT="$OUTPUTS_ROOT/content_lm/content_lm_k${K}_v1/checkpoints/${TAG}.pt"
    if [[ ! -f "$LM_CKPT" ]]; then
        echo "  skip (absent): $LM_CKPT" >&2
        continue
    fi
    for T in $TEMPS; do
        out_dir="$OUT_ROOT/k${K}__${TAG#checkpoint_}__t${T}"
        mkdir -p "$out_dir"
        echo
        echo "=== K=$K $TAG temperature=$T (AMBIENT AUDIO ONLY) ==="
        CUDA_VISIBLE_DEVICES=0 python -m salm.generate --mode a2s --direction a2s \
            --checkpoint "$RENDERER" \
            --content-source codebook_lm \
            --content-lm-checkpoint "$LM_CKPT" \
            --content-lm-temperature "$T" \
            --ctc-checkpoint "$CTC_CKPT" \
            --split val --num-samples "$NUM_GEN" \
            --out-dir "$out_dir" --copy-reference \
            2>&1 | tee "$out_dir.log" || echo "  FAILED: K=$K $TAG t=$T (continuing)" >&2
    done
  done
done

# ---------------------------------------------------------------- controls
mkdir -p "$OUT_ROOT/zero"
echo
echo "=== zero content (the null control) ==="
CUDA_VISIBLE_DEVICES=0 python -m salm.generate --mode a2s --direction a2s \
    --checkpoint "$RENDERER" --content-source zero --ctc-checkpoint "$CTC_CKPT" \
    --split val --num-samples "$NUM_GEN" \
    --out-dir "$OUT_ROOT/zero" --copy-reference \
    2>&1 | tee "$OUT_ROOT/zero.log" || echo "  FAILED: zero control" >&2

# One ceiling per K: the best this pipeline could do at that codebook coarseness if the
# predictor were perfect. Without it, a mediocre lm_* score is unattributable between
# "quantization is lossy at this K" and "the predictor is weak".
for K in $KS; do
    CB="$OUTPUTS_ROOT/content_codebook/codebook_k${K}.pt"
    [[ -f "$CB" ]] || continue
    out_dir="$OUT_ROOT/oracle_quantized_k${K}"
    mkdir -p "$out_dir"
    echo
    echo "=== oracle_quantized K=$K (the ceiling for THIS pipeline at this K) ==="
    CUDA_VISIBLE_DEVICES=0 python -m salm.generate --mode a2s --direction a2s \
        --checkpoint "$RENDERER" --content-source oracle_quantized --content-codebook "$CB" \
        --ctc-checkpoint "$CTC_CKPT" \
        --split val --num-samples "$NUM_GEN" \
        --out-dir "$out_dir" --copy-reference \
        2>&1 | tee "$out_dir.log" || echo "  FAILED: oracle_quantized K=$K" >&2
done

# ---------------------------------------------------------------- score
echo
echo "==================== INTELLIGIBILITY SCORING ===================="
CTRL="$OUTPUTS_ROOT/generations/control_lanes_20260831_0735"
DIRS="$CTRL/real_speech $CTRL/vae_roundtrip"
for d in "$OUT_ROOT"/*/; do
    [[ -f "$d/manifest.jsonl" ]] && DIRS="$DIRS ${d%/}"
done

CUDA_VISIBLE_DEVICES=0 python scripts/a2s/score_intelligibility.py \
    --dirs $DIRS --ctc-checkpoint "$CTC_CKPT" --device cuda \
    --out "$OUTPUTS_ROOT/intelligibility_content_lm_${STAMP}.json" \
    2>&1 | tee "$OUTPUTS_ROOT/logs/slurm/content_lm_gen_score_${STAMP}.log"

echo
echo "=========================================================="
echo "THE QUESTION: does lm_t* beat the zero control?"
echo "  clearly below zero, words present -> ambient-only speech generation WORKS"
echo "  indistinguishable from zero        -> scene signal real but too weak to change words;"
echo "                                        next move is a contrastive conditioning term"
echo "listen: $OUT_ROOT/{lm_t*,zero,oracle_quantized}/audio/"
echo "  <id>.wav  generated   <id>__input_ambient.wav  the scene   <id>__reference.wav  truth"
echo "=========================================================="
echo "OK"
