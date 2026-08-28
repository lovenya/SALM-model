#!/bin/bash
# Upload offline W&B runs from a LOGIN NODE (compute nodes have no internet).
#
# Training always runs with WANDB_MODE=offline, which writes a local run directory instead
# of streaming. This pushes those directories to wandb.ai afterwards.
#
# Usage:
#   bash scripts/wandb_sync.sh                    # sync every unsynced run
#   bash scripts/wandb_sync.sh <run_dir>          # sync one run's offline dirs
#   bash scripts/wandb_sync.sh --list             # show what would be synced

set -euo pipefail

OUTPUTS_ROOT="${OUTPUTS_ROOT:-/scratch/l/lovenya/salm_model_outputs}"
VENV=/scratch/l/lovenya/salm_migration_clean/.venvs/salm-model/.venv-salm-train-audioldm-py312

module load StdEnv/2023 gcc cuda/12.6 python/3.12.4 scipy-stack 2>/dev/null || true
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Entity matters: the account's default_entity is another project's team ('gaspardbtt-'),
# so it is set explicitly here too, not just in the training config.
export WANDB_ENTITY="${WANDB_ENTITY:-lovenya-org}"
export WANDB_PROJECT="${WANDB_PROJECT:-salm-s2a}"

if ! python -c "import wandb" 2>/dev/null; then
    echo "ERROR: wandb is not installed in $VENV" >&2
    echo "       uv pip install wandb" >&2
    exit 1
fi

SEARCH_ROOT="${1:-$OUTPUTS_ROOT/runs}"
if [[ "${1:-}" == "--list" ]]; then
    SEARCH_ROOT="$OUTPUTS_ROOT/runs"
    echo "offline runs under $SEARCH_ROOT:"
    find "$SEARCH_ROOT" -type d -name "offline-run-*" 2>/dev/null | sort
    exit 0
fi

mapfile -t OFFLINE < <(find "$SEARCH_ROOT" -type d -name "offline-run-*" 2>/dev/null | sort)

if [[ ${#OFFLINE[@]} -eq 0 ]]; then
    echo "no offline W&B runs found under $SEARCH_ROOT"
    exit 0
fi

echo "found ${#OFFLINE[@]} offline run(s); entity=$WANDB_ENTITY project=$WANDB_PROJECT"
for d in "${OFFLINE[@]}"; do echo "  $d"; done

# Never sync a run that is still being written: a premature `.wandb.synced` marker makes every
# later sync skip it, silently losing the rest of the run.
if squeue -u "$USER" -h -t R 2>/dev/null | grep -q .; then
    echo
    echo "WARNING: you have RUNNING jobs. If any is still writing one of the dirs above, syncing"
    echo "         it now will mark it synced and later data will be skipped. Ctrl-C to abort;"
    echo "         continuing in 10s."
    sleep 10
fi

# ONE PER DIRECTORY, not batched. 2026-08-27: a single corrupted transaction log
# (`transactionlog: error getting next record: EOF`, from a run that died mid-write) made a
# batched `wandb sync dir1 dir2 dir3 ...` call silently stop processing the REST of the
# batch too -- the command still exited 0 and this script printed "synced", but only the
# runs that happened to be ordered before the corrupted one ever got a `.wandb.synced`
# marker. Confirmed by re-running the exact same runs individually: every one of them
# synced cleanly on its own. Slower (one process launch per run) but a bad run can only
# ever cost that one run, never the rest of the batch.
echo
FAILED=()
for d in "${OFFLINE[@]}"; do
    echo "--- $d"
    if ! wandb sync --yes --project "$WANDB_PROJECT" --entity "$WANDB_ENTITY" "$d"; then
        echo "  FAILED (see error above) -- continuing with the rest" >&2
        FAILED+=("$d")
    fi
done
echo
echo "markers written:"
find "$SEARCH_ROOT" -name "*.wandb.synced" 2>/dev/null | wc -l
echo "offline dirs are safe to delete AFTER syncing, but keep wandb_run_id.txt in each run dir"
echo "or a future resume will start a new W&B run instead of continuing the existing one."
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo
    echo "${#FAILED[@]} run(s) failed and were skipped (not blocking the rest):" >&2
    printf '  %s\n' "${FAILED[@]}" >&2
    exit 1
fi
