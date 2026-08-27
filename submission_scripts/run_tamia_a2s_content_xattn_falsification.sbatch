#!/usr/bin/env bash
# Rank 2 falsification test: can the U-Net learn to use a REAL content sequence at all, via
# a genuine cross-attention channel (diffusers' own IPAdapterAttnProcessor2_0, zero-init so
# training starts bit-identical to today), given ORACLE content (the target speech's own
# CTC-probe encoding -- never available at real inference, deliberately, since the question
# here is narrowly "does the mechanism work at all"). See src/salm/a2s/content_adapter.py
# and configs/exp/content_xattn_oracle.yaml for the full mechanism and motivation.
#
# TRAINING ONLY -- no generation step. A checkpoint trained on oracle content cannot be used
# to generate real A2S output (the oracle source isn't available at inference either way),
# and the existing generate.py does not know how to reinstall this adapter onto a vanilla
# U-Net, so a generation step here would either fail or be meaningless. If this falsification
# test succeeds, THAT is what justifies building a real (non-oracle) content source and the
# generation-side wiring -- not this run.
#
# Short by design: 12 epochs. O2 and InfoNCE both showed real movement within 2-10 epochs
# tonight -- if this works at all, expect SOME signal in this budget.
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_content_xattn_falsification.sbatch
#SBATCH --job-name=salm-a2s-xattn-falsify
#SBATCH --account=aip-ravanelm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus=h100:4
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

ARCH="${ARCH:-a2s_audioldm_clapseq_xformer}"
RUN_NAME="${RUN_NAME:-content_xattn_falsify_v1}"
EPOCHS="${EPOCHS:-12}"
TRAIN_MINUTES="${TRAIN_MINUTES:-75}"
# Both default to what every other A2S probe used (batch_size=16 -> 274 steps/epoch), so a
# plain invocation is bit-identical to before. Pass both together if overriding either --
# lr_total_steps must match steps_per_epoch * EPOCHS for THIS batch_size, or the LR curve
# lands at the wrong position for the epoch budget actually run.
BATCH_SIZE="${BATCH_SIZE:-16}"
LR_TOTAL_STEPS="${LR_TOTAL_STEPS:-27400}"

RUN_DIR="$RUNS_ROOT/$ARCH/$RUN_NAME"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

# Same prerequisite checks as the real training script -- fail fast, not at minute 30.
MISSING=0
for SPLIT in train val; do
    for F in "$FEATURES_ROOT/a2s/$SPLIT/latents_mean.npy" "$OUTPUTS_ROOT/features_a2s_cond/$SPLIT/cond_seq.npy"; do
        [[ -f "$F" ]] || { echo "MISSING: $F" >&2; MISSING=1; }
    done
done
CTC_CKPT="$OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt"
[[ -f "$CTC_CKPT" ]] || { echo "MISSING: $CTC_CKPT" >&2; MISSING=1; }
[[ "$MISSING" == "0" ]] || { echo "ERROR: missing prerequisites, see above" >&2; exit 1; }

cat > "$RUN_DIR/run_metadata.json" <<EOF
{
  "run_name": "$RUN_NAME", "arch": "$ARCH", "purpose": "Rank 2 falsification test (F1/F2 plan)",
  "training_only": true, "content_source": "oracle (target speech via frozen CTC probe encoder)",
  "slurm_job_id": "${SLURM_JOB_ID:-none}", "account": "${SLURM_JOB_ACCOUNT:-unknown}",
  "hostname": "$(hostname)", "repo_path": "$REPO",
  "git_commit": "$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)",
  "git_dirty": $(if [[ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]]; then echo true; else echo false; fi),
  "epochs": $EPOCHS, "train_minutes": $TRAIN_MINUTES,
  "started_at": "$(TZ=America/Toronto date -Iseconds)"
}
EOF

salm_print_header "Rank 2 falsification test: $EPOCHS epochs, oracle content xattn, run=$RUN_NAME"
salm_start_monitor "$LOG_DIR" "$RUN_NAME" 60

CKPT="$RUN_DIR/checkpoints/checkpoint_last.pt"
RESUME_ARGS=()
if [[ -f "$CKPT" ]]; then
    echo "resuming from existing checkpoint: $CKPT"
    RESUME_ARGS=("resume_from=$CKPT")
fi

torchrun --nproc_per_node=4 --master_port=29545 -m salm.train_ldm \
    --config configs/base/a2s_audioldm.yaml "configs/arch/${ARCH}.yaml" \
        configs/exp/o2_align.yaml configs/exp/content_xattn_oracle.yaml \
    --override \
        "run_name=$RUN_NAME" "arch=$ARCH" "epochs=$EPOCHS" \
        "max_train_minutes=$TRAIN_MINUTES" \
        "batch_size=$BATCH_SIZE" "lr_total_steps=$LR_TOTAL_STEPS" \
        ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
    2>&1 | tee -a "$LOG_DIR/train.log"

echo
echo "==================== FALSIFICATION TEST SUMMARY ===================="
python - <<PY
import json, pathlib
run = pathlib.Path("$RUN_DIR")
h = json.loads((run / "history.json").read_text()) if (run / "history.json").exists() else []
print(f"epochs completed: {len(h)}")
if h:
    def series(k):
        return [(r["epoch"], r[k]) for r in h if r.get(k) is not None]
    for key in ("val_loss", "val_x0_rel_mse"):
        values = series(key)
        if values:
            print(f"{key}: first={values[0][1]:.4f} last={values[-1][1]:.4f}")
    r1 = series("val_clap_r1")
    chance = next((r["val_clap_r1_chance"] for r in h if r.get("val_clap_r1_chance") is not None), None)
    if r1:
        first, last = r1[0], r1[-1]
        best = max(r1, key=lambda x: x[1])
        print(f"val_clap_r1: first={first[1]:.4f} last={last[1]:.4f} best={best[1]:.4f} @ep{best[0]}"
              + (f"  (chance={chance:.4f})" if chance else ""))
    align = series("train_align")
    if align:
        print(f"train_align: first={align[0][1]:.4f} last={align[-1][1]:.4f}")
    d2 = series("val_d2_gap")
    if d2:
        print(f"val_d2_gap : first={d2[0][1]:.4f} last={d2[-1][1]:.4f}  "
              "(pooled-condition sensitivity with oracle content held fixed)")
else:
    print("NO epochs completed -- check the log above for a crash.")
PY
echo "======================================================================="
echo "Next, if this shows real movement: build a non-oracle content source and the"
echo "generation-side wiring. If it does not: check the wiring, or the falsification"
echo "held and this specific mechanism needs a different content source / rate."
echo "OK"
