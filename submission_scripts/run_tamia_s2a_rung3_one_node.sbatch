#!/usr/bin/env bash
# S2A Rung 0 gate -> precompute -> Rung 3 AudioLDM training -> generation.
#
# Packed into ONE job because a TamIA GPU allocation reserves the whole 4-H100 node
# anyway, and because splitting would mean queueing twice.
#
# Stages fail fast: if the gate or precompute fails there is no point training, so the
# job exits rather than burning the allocation.
#
# Account note: submitted to aip-ravanelm because the user's FairShare there (0.123) is
# ~3x better than on aip-csubakan (0.037), so it schedules sooner. The standing 60/40
# csubakan/ravanelm split should be rebalanced on later, less time-critical runs.
#
# Submit:
#   sbatch submission_scripts/run_tamia_s2a_rung3_one_node.sbatch
# Smoke first (tiny, ~15 min, proves every stage boundary):
#   SMOKE=1 sbatch submission_scripts/run_tamia_s2a_rung3_one_node.sbatch

# Resources: a TamIA h100 node is 48 CPUs / 500000M / 4x H100-80GB (NVLink). The GPU
# allocation reserves the whole node, so CPU and RAM requests are only cgroup limits
# inside it -- under-asking starves our own dataloader workers without freeing anything
# for anyone else. So take effectively the whole node. 480G stays safely under the
# 500000M physical limit.
#SBATCH --job-name=salm-s2a-rung3
#SBATCH --account=aip-ravanelm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus=h100:4
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/l/lovenya/salm_migration_clean/logs/slurm/%x-%j.out

set -euo pipefail

REPO=/scratch/l/lovenya/SALM-model
cd "$REPO"

module --force purge
module load StdEnv/2023 gcc cuda/12.6 python/3.12.4 scipy-stack

VENV=/scratch/l/lovenya/salm_migration_clean/.venvs/salm-model/.venv-salm-train-audioldm-py312
# shellcheck disable=SC1091
source "$VENV/bin/activate"

export SALM_SCRATCH="${SALM_SCRATCH:-/scratch/l/lovenya/salm_migration_clean}"
export HF_HOME="$SALM_SCRATCH/models/hf_cache"
export HF_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$REPO:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
# 48 CPUs / 4 DDP ranks = 12 each; leave headroom for the 8 dataloader workers per rank.
export OMP_NUM_THREADS=6

SMOKE="${SMOKE:-0}"
RUN_ID="${RUN_ID:-$(TZ=America/Toronto date +%Y%m%d_%H%M)}"
RUN_NAME="s2a_rung3_${RUN_ID}"
if [[ "$SMOKE" == "1" ]]; then
    RUN_NAME="s2a_rung3_smoke_${RUN_ID}"
fi

OUT_ROOT="$SALM_SCRATCH/salm_model"
RUN_DIR="$OUT_ROOT/checkpoints/$RUN_NAME"
LOG_DIR="$SALM_SCRATCH/logs/salm_model/$RUN_NAME"
GEN_DIR="$OUT_ROOT/generations/$RUN_NAME"
GATE_DIR="$OUT_ROOT/rung0_gate/$RUN_NAME"

# Reject an existing run directory BEFORE any expensive work, rather than discovering
# the collision after model loading.
if [[ -d "$RUN_DIR" ]]; then
    echo "ERROR: run dir already exists: $RUN_DIR" >&2
    exit 1
fi
mkdir -p "$RUN_DIR" "$LOG_DIR" "$GEN_DIR" "$GATE_DIR"

# ---------------------------------------------------------------- run metadata
# The job-start commit is the authoritative code identity for this run.
GIT_COMMIT=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo "unknown")
GIT_DIRTY=$(if [[ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]]; then echo true; else echo false; fi)

cat > "$RUN_DIR/run_metadata.json" <<EOF
{
  "run_name": "$RUN_NAME",
  "slurm_job_id": "${SLURM_JOB_ID:-none}",
  "account": "${SLURM_JOB_ACCOUNT:-unknown}",
  "hostname": "$(hostname)",
  "repo_path": "$REPO",
  "git_commit": "$GIT_COMMIT",
  "git_dirty": $GIT_DIRTY,
  "venv": "$VENV",
  "smoke": "$SMOKE",
  "started_at": "$(TZ=America/Toronto date -Iseconds)"
}
EOF

echo "==================================================================="
echo "run:      $RUN_NAME"
echo "job:      ${SLURM_JOB_ID:-none}  host: $(hostname)"
echo "commit:   $GIT_COMMIT (dirty=$GIT_DIRTY)"
echo "smoke:    $SMOKE"
echo "==================================================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -c "import torch;print('torch',torch.__version__,'gpus',torch.cuda.device_count())"

# --------------------------------------------------------- resource monitoring
# 30 s interval so model-load spikes and short stages are actually captured.
python scripts/resource_monitor.py \
    --out-dir "$LOG_DIR" --tag "$RUN_NAME" --interval 30 &
MONITOR_PID=$!
cleanup() {
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT

# =============================================================== 1. Rung 0 gate
# Validates weights load offline, mel round-trips through the vocoder, the VAE
# round-trips, and sampling produces audible audio. Everything downstream is
# meaningless if this fails, so it runs first and hard-gates the job.
echo
echo "### [1/4] Rung 0 gate ###"
CUDA_VISIBLE_DEVICES=0 python scripts/check_audioldm_stack.py \
    --out-dir "$GATE_DIR" \
    --device cuda \
    2>&1 | tee "$LOG_DIR/01_gate.log"

BEST_NORM=$(python -c "
import json,pathlib
p=pathlib.Path('$GATE_DIR/report.json')
print(json.loads(p.read_text()).get('best_norm_mode','none'))
")
echo "gate selected mel norm_mode: $BEST_NORM"

# ============================================================ 2. Precompute latents
# Train and val splits run concurrently on separate GPUs.
echo
echo "### [2/4] Precompute VAE latents (norm_mode=$BEST_NORM) ###"
PRE_ARGS=(--direction s2a --norm-mode "$BEST_NORM" --overwrite)
if [[ "$SMOKE" == "1" ]]; then
    PRE_ARGS+=(--limit 64)
fi

CUDA_VISIBLE_DEVICES=0 python scripts/precompute_latents.py \
    "${PRE_ARGS[@]}" --split train > "$LOG_DIR/02_precompute_train.log" 2>&1 &
PRE_TRAIN=$!
CUDA_VISIBLE_DEVICES=1 python scripts/precompute_latents.py \
    "${PRE_ARGS[@]}" --split val > "$LOG_DIR/02_precompute_val.log" 2>&1 &
PRE_VAL=$!

PRE_FAIL=0
wait "$PRE_TRAIN" || PRE_FAIL=1
wait "$PRE_VAL" || PRE_FAIL=1
tail -5 "$LOG_DIR/02_precompute_train.log" "$LOG_DIR/02_precompute_val.log"
if [[ "$PRE_FAIL" != "0" ]]; then
    echo "ERROR: precompute failed; not starting training." >&2
    exit 1
fi

# ================================================================ 3. Rung 3 train
echo
echo "### [3/4] Rung 3 training (AudioLDM diffusion recipe, DDP x4) ###"
TRAIN_MINUTES="${TRAIN_MINUTES:-100}"
EPOCHS="${EPOCHS:-20}"
if [[ "$SMOKE" == "1" ]]; then
    TRAIN_MINUTES=5
    EPOCHS=2
fi
echo "internal deadline: ${TRAIN_MINUTES} min (below the 3 h Slurm limit)"

torchrun --nproc_per_node=4 --master_port=29517 -m salm.train_ldm \
    --config configs/s2a_rung3.yaml \
    --override \
        "run_name=$RUN_NAME" \
        "output_root=$OUT_ROOT/checkpoints" \
        "max_train_minutes=$TRAIN_MINUTES" \
        "epochs=$EPOCHS" \
    2>&1 | tee "$LOG_DIR/03_train.log"

BEST_CKPT="$RUN_DIR/checkpoint_best.pt"
if [[ ! -f "$BEST_CKPT" ]]; then
    echo "WARNING: no checkpoint_best.pt; falling back to checkpoint_last.pt" >&2
    BEST_CKPT="$RUN_DIR/checkpoint_last.pt"
fi
if [[ ! -f "$BEST_CKPT" ]]; then
    echo "ERROR: training produced no checkpoint." >&2
    exit 1
fi

# =================================================================== 4. Generate
# rung0 (text ceiling) and s2a (our model) on separate GPUs, so the two are
# directly comparable on the same val rows.
echo
echo "### [4/4] Generation ###"
NUM_GEN="${NUM_GEN:-16}"
if [[ "$SMOKE" == "1" ]]; then
    NUM_GEN=2
fi

CUDA_VISIBLE_DEVICES=0 python -m salm.generate \
    --mode s2a --checkpoint "$BEST_CKPT" \
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

# ====================================================================== report
echo
echo "### Summary ###"
python - <<PY
import json, pathlib
run_dir = pathlib.Path("$RUN_DIR")
gen_dir = pathlib.Path("$GEN_DIR")

hist_path = run_dir / "history.json"
if hist_path.exists():
    hist = json.loads(hist_path.read_text())
    print(f"epochs completed: {len(hist)}")
    for h in hist:
        v = h["val_loss"]
        print(f"  ep{h['epoch']:02d} train={h['train_loss']:.4f} "
              f"val={'n/a' if v is None else f'{v:.4f}'}")
    vals = [h["val_loss"] for h in hist if h["val_loss"] is not None]
    if vals:
        print(f"best val loss: {min(vals):.4f}")
else:
    print("no history.json found")

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
echo "artifacts:"
echo "  checkpoints: $RUN_DIR"
echo "  generations: $GEN_DIR"
echo "  gate:        $GATE_DIR"
echo "  logs:        $LOG_DIR"
echo
echo "LISTEN to compare our model vs the text ceiling:"
echo "  $GEN_DIR/s2a/audio/"
echo "  $GEN_DIR/rung0_text_ceiling/audio/"

if [[ "$GEN_FAIL" != "0" ]]; then
    echo "ERROR: a generation lane failed (see logs above)." >&2
    exit 1
fi
echo "OK"
