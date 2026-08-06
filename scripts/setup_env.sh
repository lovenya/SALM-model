#!/bin/bash
# Create the SALM-model training environment.
#
# Run this on a LOGIN NODE (needs internet). Compute nodes have none.
#
# Naming follows the existing project convention -- purpose, backbone, python version:
#   .venv-salm-train-audioldm-py312
# and lives on scratch next to the other project envs so a multi-GB torch install does
# not eat the home quota.
#
# Usage:
#   bash scripts/setup_env.sh            # create (fails if it already exists)
#   RECREATE=1 bash scripts/setup_env.sh # delete and rebuild

set -euo pipefail

ENV_ROOT="/scratch/l/lovenya/salm_migration_clean/.venvs/salm-model"
ENV_NAME=".venv-salm-train-audioldm-py312"
ENV_PATH="${ENV_ROOT}/${ENV_NAME}"

export UV_CACHE_DIR="/scratch/l/lovenya/salm_migration_clean/cache/uv"
# Keep model weights on scratch, not in home.
export HF_HOME="/scratch/l/lovenya/salm_migration_clean/models/hf_cache"

echo "=== modules ==="
module load StdEnv/2023 gcc cuda/12.6 python/3.12.4 scipy-stack
python --version

if [[ -d "${ENV_PATH}" ]]; then
    if [[ "${RECREATE:-0}" == "1" ]]; then
        echo "=== removing existing env ${ENV_PATH} ==="
        rm -rf "${ENV_PATH}"
    else
        echo "ERROR: ${ENV_PATH} already exists. Use RECREATE=1 to rebuild." >&2
        exit 1
    fi
fi

mkdir -p "${ENV_ROOT}"

echo "=== creating ${ENV_PATH} ==="
uv venv --python "$(which python)" "${ENV_PATH}"

# shellcheck disable=SC1091
source "${ENV_PATH}/bin/activate"

echo "=== installing packages ==="
uv pip install \
    torch torchaudio \
    diffusers transformers accelerate \
    soundfile librosa numpy pyyaml

echo "=== verification ==="
python - <<'PY'
import torch, diffusers, transformers, soundfile, librosa, numpy
print("torch       ", torch.__version__)
print("torchaudio  ", __import__("torchaudio").__version__)
print("diffusers   ", diffusers.__version__)
print("transformers", transformers.__version__)
print("soundfile   ", soundfile.__version__)
print("librosa     ", librosa.__version__)
print("numpy       ", numpy.__version__)
print("cuda avail  ", torch.cuda.is_available())
print("device count", torch.cuda.device_count())
from diffusers import AudioLDMPipeline
print("AudioLDMPipeline importable: OK")
PY

echo
echo "=== done ==="
echo "Activate with:"
echo "  module load StdEnv/2023 gcc cuda/12.6 python/3.12.4 scipy-stack"
echo "  source ${ENV_PATH}/bin/activate"
