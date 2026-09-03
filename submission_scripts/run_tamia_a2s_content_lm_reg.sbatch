#!/usr/bin/env bash
# ContentLM REGULARIZATION sweep -- four configs in parallel, one per GPU.
#
# WHY. The first ContentLM sweep (job 436344) worked -- every LM lane crushed the zero-content
# control (0.44-0.67 vs 0.9667) and produced varied, per-scene, syntactically real speech from
# ambient audio alone. But it is badly overfit, and the numbers are not subtle:
#
#     k512   val_loss 1.143 @ep7 -> 2.518 @ep99  (+120%)   train 0.285   train/val 0.11
#     k1024  val_loss 1.293 @ep7 -> 2.732        (+111%)   train 0.331   train/val 0.12
#     k2048  val_loss 1.440 @ep7 -> 2.949        (+105%)   train 0.355   train/val 0.12
#
# Train loss ~9x below val loss, on 17,596 sentences with a 27M model at dropout=0.1 and
# weight_decay=0.01. That is under-regularized by a wide margin, and it explains two things
# we measured: generation quality PEAKS mid-training (ep39 beat ep87 in job 436345) rather
# than at the end, and val_scene_gap's late growth is confounded with memorisation -- an
# overfit conditional model can post a real scene_gap by soft-retrieving the nearest
# memorised training row, which is the retrieval approach already rejected, arrived at
# implicitly through weights.
#
# LISTENING EVIDENCE THIS IS AIMED AT (2026-09-03, user): the generated speech is clearly
# intelligible and per-scene varied, but "often a group of 3-4 words will make sense, then
# randomly another group would start". 3-4 words is ~25-40 frames of 256 -- local structure
# holds, sentence-level structure does not. Two candidate causes, and they are separable:
#   (a) the model memorises instead of generalising  <- THIS JOB
#   (b) frame-level codes at 25 fps are too fine a unit for sentence-level planning, which
#       needs deduplication + a duration predictor. Deliberately NOT built yet: job 438059
#       (full checkpoint-spread generation) is what tells us whether coherence LENGTHENS with
#       training. If it does, (a) is binding and this job is the fix. If the islands are
#       pinned at ~30 frames at every epoch, (b) is binding and dedup is the move.
#
# FOUR GPUs, FOUR CONFIGS, IN PARALLEL. Job 436344 ran three K sequentially on ONE GPU
# (1:56:30 wall, 91% util on that device, 6.3GB of 81GB VRAM, three GPUs idle). These runs are
# fully independent, so one per device costs the same wall clock as a single run.
#
# ======================= EXPECTED RESULTS (decided before running) =======================
# The baselines to beat are job 436344's, above. Two metrics, and they answer different things:
#
#   val_loss  -- does regularization actually fix the overfit? SUCCESS is the minimum moving
#     LATER than ep7 and LOWER than the k-matched baseline (1.440 for k2048, 1.143 for k512).
#     If the minimum stays at ep7 and just rises, the regularization is too strong and is
#     costing fit without buying generalization.
#
#   val_scene_gap AT that minimum -- currently only +0.021 at ep7 while reaching +0.059/+0.087
#     late, i.e. all the scene signal lives in the overfit regime. SUCCESS is a run where a
#     healthy val_loss and a meaningful scene_gap coincide at the SAME epoch. That would make
#     the scene signal trustworthy for the first time, rather than inseparable from
#     memorisation. This is the single most important outcome here.
#
#   FAILURE looks like: val_loss minimum unchanged at ep7 across all four configs, or
#     scene_gap collapsing to ~0 everywhere. The first says regularization is the wrong lever
#     and capacity/data is binding; the second says the scene signal WAS memorisation all
#     along, which would be a genuinely important negative and would redirect to (b) and to
#     VoiceLDM-scale pretraining.
#
#   AMBIGUOUS: modest improvement in both without the peak moving. Then the answer is that
#     regularization is not the binding constraint, and 438059's listening verdict decides
#     between more data and dedup.
#
# The four configs, chosen to answer four independent questions in one job rather than to
# grid-search:
#   A  k2048 drop0.3 wd0.1              does plain stronger regularization help?
#   B  k2048 drop0.3 wd0.1 layers4      is 6 layers simply too much capacity for 17.6k rows?
#   C  k512  drop0.3 wd0.1              is the regularization effect K-dependent? k512 had the
#                                       lowest val_loss and highest scene_gap, so if the
#                                       overfit is fixed it may become the best overall
#   D  k2048 drop0.3 wd0.1 ls0.1        do distance-blind soft targets help, given the codes
#                                       are k-means centroids where a neighbour is nearly
#                                       right? Held as its own config, not folded into A,
#                                       because softer targets broaden the sampled
#                                       distribution and could WORSEN coherence
#
# 150 epochs rather than 100: if regularization works, the peak should move later, and a
# 100-epoch budget could not show it. 150 x 0.37 = ~56 min, all four concurrent.
# =========================================================================================
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_content_lm_reg.sbatch
#SBATCH --job-name=salm-a2s-lm-reg
#SBATCH --account=aip-ravanelm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

CB_DIR="${CB_DIR:-$OUTPUTS_ROOT/content_codebook}"
CTC_CKPT="${CTC_CKPT:-$OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt}"
EPOCHS="${EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-64}"
# Generous per-run cap: four concurrent runs on separate devices should each hold ~0.37
# min/epoch, so 150 epochs is ~56 min. 70 leaves room if sharing the node slows them, and
# truncates gracefully with checkpoints written rather than being killed mid-save.
MAX_MINUTES="${MAX_MINUTES:-70}"
SAVE_EVERY="${SAVE_EVERY:-20}"

for F in "$CTC_CKPT" "$OUTPUTS_ROOT/features_a2s_cond/train/cond_seq.npy" \
         "$OUTPUTS_ROOT/features_a2s_cond/val/cond_seq.npy"; do
    [[ -e "$F" ]] || { echo "MISSING: $F" >&2; exit 1; }
done

# name|K|dropout|weight_decay|layers|label_smoothing
CONFIGS=(
    "reg_a_k2048_d30_wd10|2048|0.3|0.1|6|0.0"
    "reg_b_k2048_d30_wd10_L4|2048|0.3|0.1|4|0.0"
    "reg_c_k512_d30_wd10|512|0.3|0.1|6|0.0"
    "reg_d_k2048_d30_wd10_ls10|2048|0.3|0.1|6|0.1"
)

salm_print_header "ContentLM regularization sweep: ${#CONFIGS[@]} configs in PARALLEL, one per GPU"
echo "baselines to beat (job 436344): k512 val_loss 1.143@ep7 gap +0.0874@ep90"
echo "                               k2048 val_loss 1.440@ep7 gap +0.0590@ep87"
echo

mkdir -p "$OUTPUTS_ROOT/content_lm"
PIDS=()
for i in "${!CONFIGS[@]}"; do
    IFS='|' read -r NAME K DROP WD LAYERS LS <<< "${CONFIGS[$i]}"
    CB="$CB_DIR/codebook_k${K}.pt"
    if [[ ! -f "$CB" ]]; then
        echo "MISSING codebook $CB -- skipping $NAME" >&2
        continue
    fi
    echo "GPU $i <- $NAME  (K=$K dropout=$DROP wd=$WD layers=$LAYERS label_smoothing=$LS)"
    CUDA_VISIBLE_DEVICES=$i python scripts/a2s/train_content_lm.py \
        --codebook "$CB" \
        --run-name "$NAME" \
        --ctc-checkpoint "$CTC_CKPT" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --dropout "$DROP" \
        --weight-decay "$WD" \
        --layers "$LAYERS" \
        --label-smoothing "$LS" \
        --save-every "$SAVE_EVERY" \
        --max-minutes "$MAX_MINUTES" \
        > "$OUTPUTS_ROOT/content_lm/${NAME}_train.log" 2>&1 &
    PIDS+=("$!")
done

echo
echo "launched ${#PIDS[@]} concurrent runs; waiting ..."
# Collect every exit status rather than let `wait` short-circuit: one config failing must not
# discard the other three, and `set -e` would otherwise abort the whole job on the first
# non-zero. Logs are per-run, so a failure is attributable.
FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "  config $i OK"
    else
        echo "  config $i FAILED (see its own log)" >&2
        FAILED=$((FAILED + 1))
    fi
done
echo "done: $(( ${#PIDS[@]} - FAILED ))/${#PIDS[@]} succeeded"

echo
echo "==================== REGULARIZATION SWEEP SUMMARY ===================="
# argv, not os.environ: _salm_env.sh assigns OUTPUTS_ROOT without exporting it. `|| true`
# because a cosmetic summary must never fail the job -- that exact combination cost job
# 436256 its dependent generation run on 2026-09-02.
python - "$OUTPUTS_ROOT" <<'PY' || true
import json, pathlib, sys
root = pathlib.Path(sys.argv[1]) / "content_lm"
runs = sorted(root.glob("reg_*")) + sorted(root.glob("content_lm_k*_v1"))
print(f"{'run':>28} {'eps':>4} {'drop':>5} {'wd':>5} {'ls':>4} {'L':>2} "
      f"{'min loss':>9} {'@ep':>4} {'gap there':>10} {'max gap':>8} {'@ep':>4}")
for run in runs:
    h = run / "history.json"
    if not h.exists():
        print(f"{run.name:>28}  no history.json")
        continue
    hist = json.loads(h.read_text())
    if not hist:
        continue
    lo = min(hist, key=lambda r: r["val_loss"])
    bg = max(hist, key=lambda r: r["val_scene_gap"])
    last = hist[-1]
    print(f"{run.name:>28} {len(hist):>4} {last.get('cfg_dropout', 0.1):>5.2f} "
          f"{last.get('cfg_weight_decay', 0.01):>5.2f} {last.get('cfg_label_smoothing', 0.0):>4.1f} "
          f"{last.get('cfg_layers', 6):>2} {lo['val_loss']:>9.4f} {lo['epoch']:>4} "
          f"{lo['val_scene_gap']:>+10.4f} {bg['val_scene_gap']:>+8.4f} {bg['epoch']:>4}")
print()
print("READ IT LIKE THIS:")
print("  min-loss epoch moves LATER than ep7 and lower than baseline -> overfit is fixed")
print("  a healthy min loss AND a meaningful gap at the SAME epoch    -> the scene signal is")
print("      finally trustworthy rather than inseparable from memorisation (the key outcome)")
print("  min-loss still at ep7 everywhere -> regularization is the wrong lever; capacity or")
print("      data is binding, and 438059's listening verdict decides dedup vs more data")
print("  gap collapsing to ~0 -> the scene signal WAS memorisation; important negative")
PY
echo "======================================================================"
echo
echo "NEXT: generate from the winner's full checkpoint spread --"
echo "  KS=<K> LM_RUN=<run name> sbatch submission_scripts/run_tamia_a2s_content_lm_generate.sbatch"
echo "OK"
