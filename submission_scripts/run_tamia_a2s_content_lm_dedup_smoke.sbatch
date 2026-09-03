#!/usr/bin/env bash
# SMOKE TEST for the run-length (dedup) content pipeline. Deliberately small and short: the
# question is "does the wiring run end to end on a GPU", not "is it good".
#
# WHY A JOB AND NOT A LOCAL RUN. This was attempted on the login node twice on 2026-09-03 and
# killed both times -- CLAUDE.md forbids it outright ("neural-network forward passes are not
# lightweight checks"), and a 2-epoch CPU run of a transformer over ~130-token sequences is
# not a lightweight check. Everything cheap has already been verified without a GPU: 33 unit
# tests, exact codec round-trip on 12,000 real utterances, and the generation branch's
# expansion logic simulated without a model.
#
# WHAT IS STILL UNVERIFIED, and therefore what this job is for:
#   1. the trainer completes an epoch on GPU and writes a usable checkpoint
#   2. val_scene_gap is computed without error on the two-head model
#   3. generate.py detects the "dedup" checkpoint flag and takes the run-length branch
#   4. sampled (code, duration) pairs expand to 256 frames and render through the FROZEN
#      content_concat_v4 renderer without a shape or dtype error
#
# ======================= EXPECTED RESULTS (decided before running) =======================
# This job cannot tell us whether the idea works -- 8 epochs on a small model will not
# produce coherent speech, and reading its audio as evidence either way would be a mistake.
#
#   PASS -- it completes, writes checkpoints, and the generation lanes produce 16 wavs each.
#     Then the real 3-K training run is worth submitting.
#   FAIL -- a traceback. That is the entire point of running it small: the cost of finding a
#     wiring bug here is ~12 minutes instead of a wasted multi-hour run, and the traceback
#     names which of the four unverified things above is wrong.
#
# The ONE number worth a glance: val_scene_gap should be near zero at these epoch counts
# (the frame-level model needed ~10 epochs to reach +0.02), so a large value here would
# indicate a leak in the ablation rather than success.
# =========================================================================================
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_content_lm_dedup_smoke.sbatch
#SBATCH --job-name=salm-a2s-dedup-smoke
#SBATCH --account=aip-ravanelm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

K="${K:-512}"
RUN_NAME="${RUN_NAME:-dedup_smoke_k${K}}"
EPOCHS="${EPOCHS:-8}"
CB="$OUTPUTS_ROOT/content_codebook/codebook_k${K}.pt"
CTC_CKPT="$OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt"
RENDERER="$RUNS_ROOT/a2s_audioldm_clapseq_xformer/content_concat_v4/checkpoints/checkpoint_best.pt"
STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"
OUT_ROOT="$OUTPUTS_ROOT/generations/dedup_smoke_${STAMP}"

for F in "$CB" "$CTC_CKPT" "$RENDERER" \
         "$OUTPUTS_ROOT/content_codes/codes_codebook_k${K}_train.npy"; do
    [[ -f "$F" ]] || { echo "MISSING: $F" >&2; exit 1; }
done

salm_print_header "DEDUP SMOKE: $EPOCHS epochs, K=$K, small model -- wiring only"

echo
echo "==================== 1. TRAIN ===================="
# Small on purpose: 4 layers / d_model 256 keeps this fast, and the point is the code path.
CUDA_VISIBLE_DEVICES=0 python -u scripts/a2s/train_content_lm_dedup.py \
    --codebook "$CB" \
    --run-name "$RUN_NAME" \
    --epochs "$EPOCHS" \
    --batch-size 64 \
    --d-model 256 --layers 4 --heads 4 \
    --sample-every 4 \
    --save-every 4 \
    --max-minutes 12

LM_CKPT="$OUTPUTS_ROOT/content_lm/$RUN_NAME/checkpoints/checkpoint_last.pt"
[[ -f "$LM_CKPT" ]] || { echo "FAIL: training wrote no checkpoint" >&2; exit 1; }
echo "trained checkpoint: $LM_CKPT"

echo
echo "==================== 2. GENERATE ===================="
# Two temperatures only, 8 samples each -- enough to prove the branch runs and the frames
# render, not enough to judge anything.
for T in 0.8 1.0; do
    out_dir="$OUT_ROOT/dedup_t${T}"
    mkdir -p "$out_dir"
    echo
    echo "=== run-length sampling, temperature=$T ==="
    CUDA_VISIBLE_DEVICES=0 python -m salm.generate --mode a2s --direction a2s \
        --checkpoint "$RENDERER" \
        --content-source codebook_lm \
        --content-lm-checkpoint "$LM_CKPT" \
        --content-lm-temperature "$T" \
        --ctc-checkpoint "$CTC_CKPT" \
        --split val --num-samples 8 \
        --out-dir "$out_dir" --copy-reference \
        2>&1 | tee "$out_dir.log" || echo "  FAILED at temperature=$T" >&2
done

echo
echo "==================== 3. VERDICT ===================="
FAIL=0
for T in 0.8 1.0; do
    n=$(ls "$OUT_ROOT/dedup_t${T}/audio"/*.wav 2>/dev/null | wc -l)
    echo "  temperature $T: $n wavs (expect 24 = 8 samples x 3 files)"
    [[ "$n" -ge 24 ]] || FAIL=1
done
# Confirm the run-length branch was actually taken, rather than silently falling through to
# the frame-level path -- which would produce plausible-looking output from the wrong code.
if grep -q "RUN-LENGTH" "$OUT_ROOT"/dedup_t1.0.log 2>/dev/null; then
    echo "  run-length branch confirmed taken"
else
    echo "  WARNING: 'RUN-LENGTH' not found in the log -- the dedup branch may NOT have been"
    echo "           taken, which would mean the checkpoint's dedup flag was not detected" >&2
    FAIL=1
fi
grep -h "run-length tokens ->" "$OUT_ROOT"/dedup_t1.0.log 2>/dev/null | head -3 || true

python - "$OUTPUTS_ROOT/content_lm/$RUN_NAME" <<'PY' || true
import json, pathlib, sys
h = pathlib.Path(sys.argv[1]) / "history.json"
if h.exists():
    hist = json.loads(h.read_text())
    last = hist[-1]
    print(f"  {len(hist)} epochs | code {last['val_code_loss']:.3f} "
          f"(unigram {last['val_code_unigram']:.3f}) | dur {last['val_dur_loss']:.3f} "
          f"(unigram {last['val_dur_unigram']:.3f}) acc {last['val_dur_acc']:.3f} | "
          f"scene_gap {last['val_scene_gap']:+.4f}")
    print("  (scene_gap near 0 is EXPECTED this early -- the frame-level model needed ~10")
    print("   epochs to reach +0.02. A large value here would suggest an ablation leak.)")
PY

echo
if [[ "$FAIL" == "0" ]]; then
    echo "SMOKE PASSED -- the run-length pipeline runs end to end. The real 3-K run is next."
else
    echo "SMOKE FAILED -- see above. Cheap to have found here." >&2
fi
echo "listen (not meaningful yet): $OUT_ROOT/dedup_t*/audio/"
echo "OK"
