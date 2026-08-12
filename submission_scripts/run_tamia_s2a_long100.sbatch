#!/usr/bin/env bash
# The long run: 100 epochs of Rung 3 (AudioLDM eps-MSE + CFG) to find where this saturates.
#
# AUTO-RESUME: RUN_NAME is stable (not a timestamp), and the script resumes from
# checkpoint_last.pt if one exists. So if this hits the walltime, simply resubmitting the
# same command continues from where it stopped rather than starting over. That is the
# safety net for asking a single 4 h allocation.
#
# Sizing from measured 2.08 min/epoch: 100 epochs ~= 208 min, plus CLAP precompute and
# generation. TRAIN_MINUTES=205 leaves ~10 min inside the 4 h limit for validation, the
# final checkpoint, generation and the report, instead of being hard-killed mid-write.
#
# Queue caveat: large walltime asks fall outside backfill windows. Earlier today a 2 h ask
# was predicted to start 1 h 40 m later than a <=1:30 ask. Check the real predicted start
# with `squeue -j <id> -o %S` after submitting -- do NOT trust `sbatch --test-only`, which
# simulates a job queued behind everything else.
#
# Usage:
#   sbatch submission_scripts/run_tamia_s2a_long100.sbatch
#   ARCH=s2a_audioldm_whisper_mean_linear RUN_NAME=long100_L0 sbatch ...   # another rung
#
#   # extra exp overlays, appended AFTER long100.yaml so they win:
#   ARCH=s2a_audioldm_whisper_mean_linear RUN_NAME=long100_L0f \
#     EXTRA=configs/exp/frozen_unet.yaml sbatch ...                       # frozen U-Net
#   ARCH=s2a_audioldm_whisper_xformer RUN_NAME=long100_L2_o2 \
#     EXTRA=configs/exp/o2_align.yaml sbatch ...                          # O2 alignment loss
#
# RUN_NAME must be unique per experiment: the run dir is $RUNS_ROOT/$ARCH/$RUN_NAME and
# auto-resume keys off checkpoint_last.pt inside it. Reusing a name across two DIFFERENT
# configs would silently resume one experiment into the other.
#SBATCH --job-name=salm-s2a-long100
#SBATCH --account=aip-ravanelm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus=h100:4
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

ARCH="${ARCH:-s2a_audioldm_whisper_xformer}"
RUN_NAME="${RUN_NAME:-long100}"        # stable on purpose -- see AUTO-RESUME above
EPOCHS="${EPOCHS:-100}"
TRAIN_MINUTES="${TRAIN_MINUTES:-205}"
NUM_GEN="${NUM_GEN:-16}"

RUN_DIR="$RUNS_ROOT/$ARCH/$RUN_NAME"
LOG_DIR="$RUN_DIR/logs"
GEN_DIR="$RUN_DIR/generations"
mkdir -p "$LOG_DIR" "$GEN_DIR"

cat > "$RUN_DIR/run_metadata.json" <<EOF
{
  "run_name": "$RUN_NAME",
  "arch": "$ARCH",
  "slurm_job_id": "${SLURM_JOB_ID:-none}",
  "account": "${SLURM_JOB_ACCOUNT:-unknown}",
  "hostname": "$(hostname)",
  "repo_path": "$REPO",
  "git_commit": "$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)",
  "git_dirty": $(if [[ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]]; then echo true; else echo false; fi),
  "venv": "$VENV",
  "epochs": $EPOCHS,
  "train_minutes": $TRAIN_MINUTES,
  "started_at": "$(TZ=America/Toronto date -Iseconds)"
}
EOF

salm_print_header "S2A long run: $EPOCHS epochs  arch=$ARCH  run=$RUN_NAME"
salm_start_monitor "$LOG_DIR" "$RUN_NAME" 60

# ================================================ 1. CLAP targets (once, cached)
# The val cosine metric is the only interpretable progress signal we have, because the
# diffusion loss is flat by construction. Without this the long run tells us nothing.
echo
echo "### [1/3] CLAP target embeddings (val) ###"
if [[ -f "$FEATURES_CLAP_ROOT/s2a/val/clap_target.npy" ]]; then
    echo "already present, skipping"
    python -c "
import json,pathlib
d=json.loads(pathlib.Path('$FEATURES_CLAP_ROOT/s2a/val/meta.json').read_text())
print(f\"  rows={d['rows']} dim={d['dim']} pairwise_cos_mean={d['pairwise_cosine_mean']:.4f}\")"
else
    CUDA_VISIBLE_DEVICES=0 python scripts/precompute_clap_targets.py \
        --direction s2a --split val 2>&1 | tee "$LOG_DIR/01_clap_val.log"
fi

# O2 needs CLAP targets for TRAIN as well, since they are the auxiliary loss target there --
# not just a metric. ~15k rows against val's 1.9k, so budget roughly 12 min the first time;
# cached afterwards, so every later O2 run skips it. Only run when actually needed, to avoid
# spending that on every plain run. Force with PRECOMPUTE_TRAIN_CLAP=1.
NEED_TRAIN_CLAP="${PRECOMPUTE_TRAIN_CLAP:-0}"
if [[ "${EXTRA:-}" == *o2_align* ]]; then NEED_TRAIN_CLAP=1; fi
if [[ "$NEED_TRAIN_CLAP" == "1" ]]; then
    echo
    echo "### [1b/3] CLAP target embeddings (train, needed by O2) ###"
    if [[ -f "$FEATURES_CLAP_ROOT/s2a/train/clap_target.npy" ]]; then
        echo "already present, skipping"
    else
        CUDA_VISIBLE_DEVICES=0 python scripts/precompute_clap_targets.py \
            --direction s2a --split train 2>&1 | tee "$LOG_DIR/01b_clap_train.log"
    fi
fi

# ======================================================== 2. train (auto-resume)
echo
echo "### [2/3] Training ###"
CKPT="$RUN_DIR/checkpoints/checkpoint_last.pt"
RESUME_ARGS=()
if [[ -f "$CKPT" ]]; then
    echo "resuming from existing checkpoint: $CKPT"
    python -c "
import torch,sys
p=torch.load('$CKPT',map_location='cpu',weights_only=False)
print(f\"  epoch={p['epoch']} step={p['step']} resumable={'optimizer' in p}\")"
    # Just the key=value; --override is already on the torchrun command below.
    RESUME_ARGS=("resume_from=$CKPT")
else
    echo "fresh start (no checkpoint_last.pt in $RUN_DIR/checkpoints)"
fi

# Extra exp overlays, space-separated, appended last so they override long100.yaml.
# Validated here rather than letting torchrun fail 30 s in with a confusing traceback.
read -r -a EXTRA_CONFIGS <<< "${EXTRA:-}"
for f in ${EXTRA_CONFIGS[@]+"${EXTRA_CONFIGS[@]}"}; do
    [[ -f "$REPO/$f" ]] || { echo "ERROR: EXTRA config not found: $REPO/$f" >&2; exit 1; }
    echo "extra config: $f"
done

torchrun --nproc_per_node=4 --master_port=29541 -m salm.train_ldm \
    --config configs/base/s2a_audioldm.yaml "configs/arch/${ARCH}.yaml" configs/exp/long100.yaml \
        ${EXTRA_CONFIGS[@]+"${EXTRA_CONFIGS[@]}"} \
    --override \
        "run_name=$RUN_NAME" \
        "arch=$ARCH" \
        "epochs=$EPOCHS" \
        "max_train_minutes=$TRAIN_MINUTES" \
        ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
    2>&1 | tee -a "$LOG_DIR/02_train.log"

# =============================================================== 3. generation
echo
echo "### [3/3] Generation ###"
BEST="$RUN_DIR/checkpoints/checkpoint_best.pt"
[[ -f "$BEST" ]] || BEST="$RUN_DIR/checkpoints/checkpoint_last.pt"

CUDA_VISIBLE_DEVICES=0 python -m salm.generate \
    --mode s2a --checkpoint "$BEST" --split val --num-samples "$NUM_GEN" \
    --out-dir "$GEN_DIR/s2a_best" --copy-reference \
    > "$LOG_DIR/03_gen_s2a.log" 2>&1 &
GEN_A=$!
CUDA_VISIBLE_DEVICES=1 python -m salm.generate \
    --mode rung0 --split val --num-samples "$NUM_GEN" \
    --out-dir "$GEN_DIR/rung0_text_ceiling" \
    > "$LOG_DIR/03_gen_rung0.log" 2>&1 &
GEN_B=$!
GEN_FAIL=0
wait "$GEN_A" || GEN_FAIL=1
wait "$GEN_B" || GEN_FAIL=1
tail -n 5 "$LOG_DIR/03_gen_s2a.log" "$LOG_DIR/03_gen_rung0.log"

# ===================================================================== report
echo
echo "==================== RUN SUMMARY ===================="
python - <<PY
import json, pathlib
run = pathlib.Path("$RUN_DIR")
h = json.loads((run / "history.json").read_text()) if (run / "history.json").exists() else []
print(f"epochs completed: {len(h)}")
if h:
    vals = [(r["epoch"], r["val_loss"]) for r in h if r.get("val_loss") is not None]
    cos  = [(r["epoch"], r["val_clap_cos"]) for r in h if r.get("val_clap_cos") is not None]
    if vals:
        best = min(vals, key=lambda x: x[1])
        print(f"val loss:  first={vals[0][1]:.4f} last={vals[-1][1]:.4f} best={best[1]:.4f} @ep{best[0]}")
    if cos:
        top = max(cos, key=lambda x: x[1])
        print(f"CLAP cos:  first={cos[0][1]:+.4f} last={cos[-1][1]:+.4f} best={top[1]:+.4f} @ep{top[0]}")
        print(f"           range over training = {max(c for _,c in cos)-min(c for _,c in cos):.4f}")
        if max(c for _,c in cos) - min(c for _,c in cos) < 0.01:
            print("  >>> FLAT. The conditioner is not learning the speech->scene mapping.")
            print("      More epochs will not help; change architecture or conditioning.")
    else:
        print("CLAP cos:  MISSING -- only the flat diffusion loss was available.")
    print(f"mean epoch time: {sum(r.get('epoch_minutes',0) for r in h)/len(h):.2f} min")
PY
echo
echo "run dir:     $RUN_DIR"
echo "audio:       $GEN_DIR/s2a_best/audio/  vs  $GEN_DIR/rung0_text_ceiling/audio/"
echo "W&B: sync with  bash scripts/wandb_sync.sh $RUN_DIR"
echo "===================================================="
[[ "$GEN_FAIL" == "0" ]] || { echo "ERROR: a generation lane failed" >&2; exit 1; }
echo "OK"
