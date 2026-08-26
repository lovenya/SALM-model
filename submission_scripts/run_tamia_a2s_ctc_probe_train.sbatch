#!/usr/bin/env bash
# Train the frozen CTC content-loss head O-F needs (see scripts/a2s/train_ctc_probe.py).
#
# WHY THIS JOB EXISTS. The 2026-08-24 oracle-conditioning ablation showed even a PERFECT
# CLAP-text target produces incomprehensible speech, by ear -- so the bottleneck is the
# target/interface, not the conditioner. O-F sidesteps it by supervising the DECODED mel
# directly with a real content loss during A2S training, which needs a scorer that
# understands AudioLDM's own mel frontend. No off-the-shelf ASR model matches it (64 mel
# bins, most use 80), so this trains one from scratch on real (target speech, transcript)
# pairs -- this job produces the checkpoint that gets loaded FROZEN by the real A2S run.
#
# Single-GPU (2.4M params, 17.6k rows -- nowhere near needing DDP). Requests the whole node
# anyway (TamIA's h100 policy), CUDA_VISIBLE_DEVICES=0 only; GPUs 1-3 sit idle this job --
# nothing else is queued to share them with right now.
#
# Sizing is a first estimate, not a measurement: 60 epochs, batch 64, ~274 steps/epoch (same
# row count as the diffusion runs). Unknown per-epoch time since this model has never run;
# TRAIN_MINUTES leaves headroom inside the walltime and the run checkpoints on its own
# schedule regardless, so a conservative first guess costs nothing but wall-clock.
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_ctc_probe_train.sbatch
#SBATCH --job-name=salm-a2s-ctc-probe
#SBATCH --account=aip-csubakan
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

RUN_NAME="${RUN_NAME:-ctc_probe_v1}"     # stable on purpose -- enables auto-resume below
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TRAIN_MINUTES="${TRAIN_MINUTES:-165}"    # inside the 3 h ask, leaves buffer for the last save

RUN_DIR="$OUTPUTS_ROOT/ctc_probe/$RUN_NAME"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

salm_print_header "A2S CTC probe training: $EPOCHS epochs  bs=$BATCH_SIZE  run=$RUN_NAME"
salm_start_monitor "$LOG_DIR" "$RUN_NAME" 60

CKPT="$RUN_DIR/checkpoints/checkpoint_last.pt"
RESUME_ARGS=()
if [[ -f "$CKPT" ]]; then
    echo "resuming from existing checkpoint: $CKPT"
    python -c "
import torch
p=torch.load('$CKPT',map_location='cpu',weights_only=False)
print(f\"  epoch={p['epoch']} step={p['step']} best_wer={p.get('best_wer')}\")"
    RESUME_ARGS=(--resume-from "$CKPT")
else
    echo "fresh start (no checkpoint_last.pt in $RUN_DIR/checkpoints)"
fi

CUDA_VISIBLE_DEVICES=0 python scripts/a2s/train_ctc_probe.py \
    --run-name "$RUN_NAME" --runs-root "$OUTPUTS_ROOT/ctc_probe" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --max-train-minutes "$TRAIN_MINUTES" \
    "${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}" \
    2>&1 | tee -a "$LOG_DIR/train.log"

echo
echo "==================== CTC PROBE SUMMARY ===================="
python - <<PY
import json, pathlib
h = json.loads((pathlib.Path("$RUN_DIR") / "history.json").read_text())
print(f"epochs completed: {len(h)}")
wers = [(r["epoch"], r["val_wer"]) for r in h if r.get("val_wer") is not None]
if wers:
    best = min(wers, key=lambda x: x[1])
    print(f"val_wer: first={wers[0][1]:.4f} last={wers[-1][1]:.4f} best={best[1]:.4f} @ep{best[0]}")
    if best[1] > 0.6:
        print("  >>> WER still high. This probe may not be a reliable content-loss head yet --")
        print("      check examples in the log before trusting O-F runs built on this checkpoint.")
et = [r.get("epoch_minutes", 0) for r in h]
print(f"mean epoch time: {sum(et)/max(len(et),1):.2f} min")
PY
echo
echo "checkpoint for O-F: $RUN_DIR/checkpoints/checkpoint_best.pt"
echo "=============================================================="
