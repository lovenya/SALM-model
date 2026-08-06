#!/usr/bin/env bash
# Stage 2: Rung 3 AudioLDM training (DDP x4) + generation.
#
# Requires stage 1's precomputed latents. Reads the mel norm_mode from the features
# meta.json, so there is no run-ID coupling between the two jobs.
#
# 2 h ask: 90 min internal training deadline + generation + reporting, which still
# backfills far more easily than a single 3 h job.
#
# Resources: TamIA h100 node = 48 CPUs / 500000M / 4x H100-80GB, whole node reserved.
#SBATCH --job-name=salm-s2a-stage2
#SBATCH --account=aip-ravanelm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus=h100:4
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/l/lovenya/salm_migration_clean/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/_salm_env.sh"

SMOKE="${SMOKE:-0}"
RUN_ID="${RUN_ID:-$(TZ=America/Toronto date +%Y%m%d_%H%M)}"
RUN_NAME="s2a_rung3_${RUN_ID}"
if [[ "$SMOKE" == "1" ]]; then
    RUN_NAME="s2a_rung3_smoke_${RUN_ID}"
fi

RUN_DIR="$OUT_ROOT/checkpoints/$RUN_NAME"
LOG_DIR="$SALM_SCRATCH/logs/salm_model/$RUN_NAME"
GEN_DIR="$OUT_ROOT/generations/$RUN_NAME"

# Reject a colliding run dir BEFORE model loading, not after.
if [[ -d "$RUN_DIR" ]]; then
    echo "ERROR: run dir already exists: $RUN_DIR" >&2
    exit 1
fi
mkdir -p "$RUN_DIR" "$LOG_DIR" "$GEN_DIR"

# Fail early if stage 1 did not run.
for split in train val; do
    if [[ ! -f "$FEATURES_ROOT/s2a/$split/meta.json" ]]; then
        echo "ERROR: missing precomputed features for '$split'." >&2
        echo "       Run run_tamia_s2a_stage1_gate_precompute.sbatch first." >&2
        exit 1
    fi
done

cat > "$RUN_DIR/run_metadata.json" <<EOF
{
  "run_name": "$RUN_NAME",
  "slurm_job_id": "${SLURM_JOB_ID:-none}",
  "account": "${SLURM_JOB_ACCOUNT:-unknown}",
  "hostname": "$(hostname)",
  "repo_path": "$REPO",
  "git_commit": "$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)",
  "git_dirty": $(if [[ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]]; then echo true; else echo false; fi),
  "venv": "$VENV",
  "features_root": "$FEATURES_ROOT/s2a",
  "smoke": "$SMOKE",
  "started_at": "$(TZ=America/Toronto date -Iseconds)"
}
EOF

salm_print_header "S2A stage 2: Rung 3 training + generation  (smoke=$SMOKE)"
salm_start_monitor "$LOG_DIR" "$RUN_NAME" 30

python - <<PY
import json, pathlib
for split in ("train", "val"):
    d = json.loads((pathlib.Path("$FEATURES_ROOT/s2a") / split / "meta.json").read_text())
    print(f"features {split}: n={d['kept']} latent={d['latent_shape']} "
          f"norm_mode={d['mel_config']['norm_mode']}")
PY

# ================================================================= 1. Training
echo
echo "### [1/2] Rung 3 training (AudioLDM eps-MSE + CFG dropout, DDP x4) ###"
TRAIN_MINUTES="${TRAIN_MINUTES:-90}"
EPOCHS="${EPOCHS:-20}"
if [[ "$SMOKE" == "1" ]]; then
    TRAIN_MINUTES=5
    EPOCHS=2
fi
echo "internal deadline: ${TRAIN_MINUTES} min (under the 2 h Slurm limit, leaving room"
echo "for validation, checkpointing, generation, and the report)"

torchrun --nproc_per_node=4 --master_port=29517 -m salm.train_ldm \
    --config configs/s2a_rung3.yaml \
    --override \
        "run_name=$RUN_NAME" \
        "output_root=$OUT_ROOT/checkpoints" \
        "max_train_minutes=$TRAIN_MINUTES" \
        "epochs=$EPOCHS" \
    2>&1 | tee "$LOG_DIR/03_train.log"

CKPT="$RUN_DIR/checkpoint_best.pt"
if [[ ! -f "$CKPT" ]]; then
    echo "WARNING: no checkpoint_best.pt; falling back to checkpoint_last.pt" >&2
    CKPT="$RUN_DIR/checkpoint_last.pt"
fi
if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: training produced no checkpoint." >&2
    exit 1
fi

# =============================================================== 2. Generation
# Our model and the rung0 text ceiling on separate GPUs, over the SAME val rows, so
# the two are directly A/B comparable.
echo
echo "### [2/2] Generation ###"
NUM_GEN="${NUM_GEN:-16}"
if [[ "$SMOKE" == "1" ]]; then
    NUM_GEN=2
fi

CUDA_VISIBLE_DEVICES=0 python -m salm.generate \
    --mode s2a --checkpoint "$CKPT" \
    --split val --num-samples "$NUM_GEN" \
    --out-dir "$GEN_DIR/s2a" --copy-reference \
    > "$LOG_DIR/04_gen_s2a.log" 2>&1 &
GEN_S2A=$!

CUDA_VISIBLE_DEVICES=1 python -m salm.generate \
    --mode rung0 \
    --split val --num-samples "$NUM_GEN" \
    --out-dir "$GEN_DIR/rung0_text_ceiling" \
    > "$LOG_DIR/04_gen_rung0.log" 2>&1 &
GEN_R0=$!

GEN_FAIL=0
wait "$GEN_S2A" || GEN_FAIL=1
wait "$GEN_R0" || GEN_FAIL=1
tail -6 "$LOG_DIR/04_gen_s2a.log" "$LOG_DIR/04_gen_rung0.log"

# ===================================================================== report
echo
echo "### Summary ###"
python - <<PY
import json, pathlib
run_dir = pathlib.Path("$RUN_DIR")
gen_dir = pathlib.Path("$GEN_DIR")

h = run_dir / "history.json"
if h.exists():
    hist = json.loads(h.read_text())
    print(f"epochs completed: {len(hist)}")
    for r in hist:
        v = r["val_loss"]
        print(f"  ep{r['epoch']:02d} train={r['train_loss']:.4f} "
              f"val={'n/a' if v is None else f'{v:.4f}'}")
    vals = [r["val_loss"] for r in hist if r["val_loss"] is not None]
    if vals:
        print(f"best val loss: {min(vals):.4f}")
else:
    print("no history.json")

for name in ("s2a", "rung0_text_ceiling"):
    s = gen_dir / name / "summary.json"
    if s.exists():
        d = json.loads(s.read_text())
        print(f"{name}: n={d['num_samples']} rms_mean={d['rms_mean']:.5f} "
              f"near_silent={d['near_silent_count']}")
    else:
        print(f"{name}: MISSING summary.json")
PY

echo
echo "checkpoints: $RUN_DIR"
echo "generations: $GEN_DIR"
echo "logs:        $LOG_DIR"
echo
echo "LISTEN to compare our model against the text ceiling:"
echo "  $GEN_DIR/s2a/audio/"
echo "  $GEN_DIR/rung0_text_ceiling/audio/"

if [[ "$GEN_FAIL" != "0" ]]; then
    echo "ERROR: a generation lane failed." >&2
    exit 1
fi
echo "OK"
