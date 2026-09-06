#!/usr/bin/env bash
# RENDERER FINE-TUNE: close the train/test gap on the content channel.
#
# THE DEFECT. `train_ldm.py` feeds the U-Net continuous `ctc_head.encode()` features. Every
# generation path feeds it codebook CENTROIDS (`generate.py`: content_codebook.dequantize).
# The renderer has never once seen quantized content during training, and nobody noticed.
#
# THE COST IS ALREADY MEASURED, matched clips, PERFECT codes, one job
# (codebook_gate_20260902_0656):
#
#     oracle            (continuous, true content)   0.4231
#     quantized_k2048   (true codes, quantized)      0.4843     <-- +0.061 for quantization
#     quantized_k1024                                0.5480
#     quantized_k512                                 0.6014
#
# +0.061 is the mismatch alone, with nothing else varying. For scale, the ENTIRE remaining gap
# between the best LM lane (0.3688) and real speech (0.3146) is 0.054.
#
# WHY NOW. Job 443737 showed clarity and variety are a single axis: greedy scores 0.3688 with
# 46% stock openings, and every step toward variety costs clarity monotonically (top_p 0.7 ->
# 0.4720 / 12%, 0.85 -> 0.5445 / 7%, 1.0 -> 0.6380 / 5%). Greedy is clear because argmax picks
# HIGH-CONFIDENCE codes; sampling picks lower-confidence ones and clarity collapses. That is
# the signature of a renderer that cannot tolerate imperfect codes. If robustness decouples the
# axis, nucleus sampling becomes affordable and the stock phrases stop being the price of
# intelligibility.
#
# WHY NOT TRAIN ON LM-SAMPLED CODES DIRECTLY -- the obvious version, and it is ill-posed.
# Pairing LM codes with the ORIGINAL audio teaches the renderer to produce the true utterance
# from codes that say something else, i.e. to ignore its content channel. That is
# `argmin E[(f(x)-y)^2] = E[y|x]` again, the conditional-mean collapse this project exists to
# escape, and every training curve would look healthy while it happened. Arm 3 instead corrupts
# TRUE codes with NEAREST-NEIGHBOUR substitutions, so the target audio stays a valid answer.
#
# ======================= EXPECTED RESULTS (decided before running) =======================
# STEP 1 IS A GATE, and it can cancel arm 3. `measure_lm_code_error.py` reports what share of
# the LM's code errors land on a near neighbour. Nearest-neighbour corruption simulates only
# that kind of error. If the share is under ~0.25 the LM mostly makes FAR misses, arm 3 is
# modelling the wrong thing, and its result must be read as uninformative rather than negative.
# Arm 2 is unaffected either way.
#
# THREE ARMS, all warm-started from content_concat_v4 and fine-tuned 60 epochs:
#   A. continuous  -- CONTROL. Identical fine-tune, content channel unchanged. Without this,
#      any gain is attributable to "we trained 60 more epochs" rather than to quantization.
#   B. quantized   -- content quantized through codebook_k2048, matching generation exactly.
#   C. quant+corrupt -- B plus nearest-neighbour substitution at 0.10 (provisional; step 1
#      says whether that matches the LM's real rate, and the log states the discrepancy).
#
#   SUCCESS -- B beats A on generated audio by something approaching the +0.061 the mismatch
#       costs. The renderer should be trained on quantized content from here on, and every
#       past oracle-vs-quantized comparison was reading a fixable artifact as a property of
#       the codebook.
#   PARTIAL -- B beats A but by much less than 0.061. Then most of that gap is the information
#       genuinely lost to quantization rather than the mismatch, and the codebook itself (D3,
#       different construction) is the lever, not the renderer.
#   NULL -- B == A. The renderer already generalises across the gap and the +0.061 is real
#       quantization loss. Closes this direction cleanly and promotes the codebook work.
#   C vs B isolates error-robustness on top of matching. If C wins, re-run the nucleus sweep
#       against C -- the clarity/variety trade is what it is meant to loosen, and that would
#       be measured, not assumed.
#
#   WATCH: val_content_bind_gap. If it collapses in B or C, the renderer stopped using the
#   content channel at all -- the conditional-mean failure -- and a better inter_asr_dist would
#   then be meaningless. Check it before believing any improvement.
# =========================================================================================
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_renderer_quant.sbatch
#SBATCH --job-name=salm-a2s-renderer-quant
#SBATCH --account=aip-csubakan
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

# The arch config FILENAME, not the short run-dir suffix. Job 443788 lost all three
# arms in 84 seconds to "clapseq_xformer", which is the run-directory name, while the
# file is a2s_audioldm_clapseq_xformer.yaml.
ARCH="${ARCH:-a2s_audioldm_clapseq_xformer}"
K="${K:-2048}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR_TOTAL_STEPS="${LR_TOTAL_STEPS:-4080}"
TRAIN_MINUTES="${TRAIN_MINUTES:-45}"
NGEN="${NGEN:-64}"
STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"
RUNS_SUBDIR="${RUNS_SUBDIR:-a2s_audioldm_clapseq_xformer}"
RENDERER_CKPT="$RUNS_ROOT/$RUNS_SUBDIR/content_concat_v4/checkpoints/checkpoint_best.pt"
CTC_CKPT="$OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt"
LM_CKPT="$OUTPUTS_ROOT/content_lm/dedup_k${K}/checkpoints/checkpoint_best_loss.pt"
OUT_ROOT="$OUTPUTS_ROOT/generations/rquant_${STAMP}"

for F in "configs/arch/${ARCH}.yaml" configs/exp/content_concat.yaml \
         configs/exp/content_concat_quant.yaml \
         configs/exp/content_concat_quantcorrupt.yaml \
         "$RENDERER_CKPT" "$CTC_CKPT" "$LM_CKPT" \
         "$OUTPUTS_ROOT/content_codebook/codebook_k${K}.pt"; do
    [[ -f "$F" ]] || { echo "MISSING: $F" >&2; exit 1; }
done
FREE=$(df -Pm /scratch/l/lovenya | awk 'NR==2{print $4}')
echo "free space: ${FREE} MB (need >= 8000: 3 renderer runs plus wavs)"
[[ "$FREE" -ge 8000 ]] || { echo "ABORT: only ${FREE} MB free" >&2; exit 1; }

salm_print_header "RENDERER QUANT FINE-TUNE: 2 arms x $EPOCHS epochs from content_concat_v4"

echo
echo "==================== 1. GATE: are the LM's errors NEAR misses? ===================="
CUDA_VISIBLE_DEVICES=0 python scripts/a2s/measure_lm_code_error.py \
    --lm-checkpoint "$LM_CKPT" --split val --num 256 --device cuda \
    --out "$OUTPUTS_ROOT/lm_code_error_${STAMP}.json" \
    2>&1 | tee "$OUTPUTS_ROOT/logs/slurm/lm_code_error_${STAMP}.log" || true

echo
echo "==================== 2. TRAIN (3 arms, sequential, 4 GPUs each) ===================="
# Sequential rather than one-arm-per-GPU: the renderer is a 185M U-Net at batch 16 and the
# original v4 run used all four GPUs. Three 45-minute arms fit the 2.5 h wall comfortably.
# Run names are computed here, NOT returned from the function. Capturing them with
# $(run_arm ...) | tail -1 would silently pick up any stray line the function prints, and a
# wrong run name means step 4 generates from the wrong checkpoint while looking fine.
A_CONT="rquant_continuous_${STAMP}"
A_QUANT="rquant_quantized_${STAMP}"
A_CORR="rquant_quantcorrupt_${STAMP}"

run_arm() {
    local name="$1" exp="$2" run_name="$3"
    echo; echo "=== arm: $name -> $run_name ==="
    torchrun --nproc_per_node=4 --master_port=29561 -m salm.train_ldm \
        --config configs/base/a2s_audioldm.yaml "configs/arch/${ARCH}.yaml" \
            configs/exp/o2_align.yaml "configs/exp/${exp}.yaml" \
        --override \
            "run_name=$run_name" "arch=$ARCH" "epochs=$EPOCHS" \
            "max_train_minutes=$TRAIN_MINUTES" "batch_size=$BATCH_SIZE" \
            "lr_total_steps=$LR_TOTAL_STEPS" \
            "renderer_checkpoint=$RENDERER_CKPT" "freeze_renderer=false" \
        > "$OUTPUTS_ROOT/logs/slurm/rquant_${name}_${STAMP}.log" 2>&1 \
        || echo "  ARM FAILED: $name" >&2
}
run_arm continuous   content_concat               "$A_CONT"
run_arm quantized    content_concat_quant         "$A_QUANT"
# ARM C (nearest-neighbour corruption) DROPPED after job 443788's gate.
# Measured: code "error rate" vs the reference is 0.9931 with only 2% near misses --
# but that lane scores inter_asr_dist 0.3688, near real speech. Both cannot hold if
# these were errors. They are not: the LM generates a DIFFERENT valid utterance, which
# is correct for a one-to-many task. The renderer is never handed corrupted codes, so
# there is nothing for corruption augmentation to simulate. Dropped on the measurement,
# not retuned. The surviving question -- does the LM draw from a different code
# DISTRIBUTION than training showed -- is now reported by step 1.

echo
echo "==================== 3. BIND GAP -- did the content channel survive? ===================="
# Read before any audio number. A renderer that stopped using content would post a BETTER
# inter_asr_dist by generating generic clean speech, which would be a regression dressed as a win.
python - "$RUNS_ROOT" "$A_CONT" "$A_QUANT" <<'PY' || true
import json, sys
from pathlib import Path
root = Path(sys.argv[1]) / "a2s_audioldm_clapseq_xformer"
for name in sys.argv[2:]:
    h = root / name / "history.json"
    if not h.exists():
        print(f"  {name}: no history"); continue
    hist = json.loads(h.read_text())
    last = hist[-1]
    bg = [e.get("val_content_bind_gap") for e in hist if e.get("val_content_bind_gap") is not None]
    print(f"  {name:34s} epochs {len(hist):3d} | val_loss {last.get('val_loss', float('nan')):.4f}"
          f" | bind_gap first {bg[0]:+.4f} last {bg[-1]:+.4f}" if bg else
          f"  {name:34s} epochs {len(hist):3d} (no bind gap recorded)")
PY

echo
echo "==================== 4. GENERATE (greedy, matched) ===================="
for pair in "continuous:$A_CONT" "quantized:$A_QUANT"; do
    label="${pair%%:*}"; run="${pair#*:}"
    ck="$RUNS_ROOT/$RUNS_SUBDIR/$run/checkpoints/checkpoint_best.pt"
    [[ -f "$ck" ]] || { echo "  skip (no checkpoint): $label" >&2; continue; }
    d="$OUT_ROOT/$label"; mkdir -p "$d"
    CUDA_VISIBLE_DEVICES=0 python -m salm.generate --mode a2s --direction a2s \
        --checkpoint "$ck" --content-source codebook_lm --content-lm-checkpoint "$LM_CKPT" \
        --content-lm-temperature 1.0 --content-lm-top-k 1 --ctc-checkpoint "$CTC_CKPT" \
        --split val --num-samples "$NGEN" --start-index 0 --seed 1337 \
        --out-dir "$d" --copy-reference > "$d.log" 2>&1 || echo "  FAILED: $label" >&2
    # The oracle-quantized lane on the SAME renderer: this is where the +0.061 lives, so it is
    # the arm that most directly shows whether the mismatch was closed.
    d2="$OUT_ROOT/${label}__oracle_quant"; mkdir -p "$d2"
    CUDA_VISIBLE_DEVICES=0 python -m salm.generate --mode a2s --direction a2s \
        --checkpoint "$ck" --content-source oracle_quantized \
        --content-codebook "$OUTPUTS_ROOT/content_codebook/codebook_k${K}.pt" \
        --content-lm-checkpoint "$LM_CKPT" --ctc-checkpoint "$CTC_CKPT" \
        --split val --num-samples "$NGEN" --start-index 0 --seed 1337 \
        --out-dir "$d2" --copy-reference > "$d2.log" 2>&1 || echo "  FAILED: ${label} oracle" >&2
done

echo
echo "==================== 5. SCORE ===================="
CTRL="$OUTPUTS_ROOT/generations/control_lanes_20260831_0735"
DIRS="$CTRL/real_speech $CTRL/vae_roundtrip"
for d in "$OUT_ROOT"/*/; do [[ -f "$d/manifest.jsonl" ]] && DIRS="$DIRS ${d%/}"; done
# shellcheck disable=SC2086
python scripts/a2s/score_intelligibility.py --dirs $DIRS --ctc-checkpoint "$CTC_CKPT" \
    --device cuda --out "$OUTPUTS_ROOT/rquant_intel_${STAMP}.json" \
    || echo "SCORING FAILED -- generations intact" >&2
python scripts/a2s/build_listening_bundle.py --gen-root "$OUT_ROOT" \
    --out "$OUTPUTS_ROOT/listening_bundles/rquant_${STAMP}" --max-clips 8 || true

echo
echo "=========================================================="
echo "READ IN THIS ORDER:"
echo "  1. step 1 gate -- if near-miss share < 0.25, arm C is uninformative, not negative"
echo "  2. step 3 bind gap -- a collapsed gap means the content channel died; stop there"
echo "  3. *__oracle_quant lanes: continuous vs quantized. The +0.061 mismatch lives here."
echo "  4. the LM lanes: does closing the mismatch help the deployed path?"
echo "listen: $OUTPUTS_ROOT/listening_bundles/rquant_${STAMP}"
echo "=========================================================="
echo "OK"
