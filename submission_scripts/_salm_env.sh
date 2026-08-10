# Shared environment preamble for SALM-model Slurm jobs. Source this, do not execute it.
#
# All four HF cache variables are exported deliberately: the project previously hit
# offline-resolution failures when only HF_HOME/TRANSFORMERS_CACHE were set, because
# huggingface_hub reads HF_HUB_CACHE / HUGGINGFACE_HUB_CACHE separately.

REPO=/scratch/l/lovenya/SALM-model
cd "$REPO"

module --force purge
module load StdEnv/2023 gcc cuda/12.6 python/3.12.4 scipy-stack

VENV=/scratch/l/lovenya/salm_migration_clean/.venvs/salm-model/.venv-salm-train-audioldm-py312
# shellcheck disable=SC1091
source "$VENV/bin/activate"

export SALM_SCRATCH="${SALM_SCRATCH:-/scratch/l/lovenya/salm_migration_clean}"
export HF_HOME="$SALM_SCRATCH/models/hf_cache"
# NOTE the "/hub" suffix. HF_HUB_CACHE defaults to $HF_HOME/hub, which is where
# `hf download` actually writes. Setting it to $HF_HOME (as some older SpeechAudioLM
# scripts do) points the resolver one level too high, and every repo-id load then fails
# with a misleading "does not appear to have a file named config.json" even though the
# cache is complete. Those older scripts get away with it because their downloads
# predate this layout and sit directly in hf_cache/.
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$REPO:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
# 48 CPUs / 4 ranks = 12 each; leave headroom for dataloader workers.
export OMP_NUM_THREADS=6

# Dedicated output root for this repo. Deliberately NOT under salm_migration_clean, which
# belongs to the SpeechAudioLM repo (its datasets, metric caches, venvs).
OUTPUTS_ROOT="${OUTPUTS_ROOT:-/scratch/l/lovenya/salm_model_outputs}"
FEATURES_ROOT="$OUTPUTS_ROOT/features"
FEATURES_CLAP_ROOT="$OUTPUTS_ROOT/features_clap"
RUNS_ROOT="$OUTPUTS_ROOT/runs"
# Back-compat alias: older scripts in this repo referred to OUT_ROOT.
OUT_ROOT="$OUTPUTS_ROOT"
mkdir -p "$OUTPUTS_ROOT/logs/slurm"

# W&B must be offline: compute nodes have no internet. `scripts/wandb_sync.sh` uploads
# afterwards from a login node. Without this, wandb.init blocks on a network timeout.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_ENTITY="${WANDB_ENTITY:-lovenya-org}"
export WANDB_PROJECT="${WANDB_PROJECT:-salm-s2a}"
# Keep W&B's own cache/artifacts off the home quota.
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$OUTPUTS_ROOT/.wandb_cache}"

salm_print_header() {
    local label="$1"
    echo "==================================================================="
    echo "$label"
    echo "job:    ${SLURM_JOB_ID:-none}   host: $(hostname)"
    echo "commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "dirty:  $(if [[ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]]; then echo true; else echo false; fi)"
    echo "==================================================================="
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    python -c "import torch;print('torch',torch.__version__,'gpus',torch.cuda.device_count())"
}

# Start the resource sampler and register cleanup. Callers pass a log dir and tag.
salm_start_monitor() {
    local log_dir="$1" tag="$2" interval="${3:-30}"
    mkdir -p "$log_dir"
    python scripts/resource_monitor.py --out-dir "$log_dir" --tag "$tag" --interval "$interval" &
    SALM_MONITOR_PID=$!
    trap 'kill "$SALM_MONITOR_PID" 2>/dev/null || true; wait "$SALM_MONITOR_PID" 2>/dev/null || true' EXIT
}
