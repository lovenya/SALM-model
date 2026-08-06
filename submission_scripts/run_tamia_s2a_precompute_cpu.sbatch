#!/usr/bin/env bash
# Precompute VAE latents on a CPU node.
#
# Why this is CPU-viable: precompute is resample + mel STFT + VAE *encode*. The encode
# path is a downsampling conv over a 1024x64 mel, which is cheap. The expensive thing in
# this stack is the HiFi-GAN vocoder (it upsamples to 163840 samples) and diffusion
# sampling -- neither is used here.
#
# Why bother: CPU nodes on TamIA are idle and start immediately, while a whole-node
# 4-GPU allocation waits ~45 min. Doing this here keeps precompute off the GPU critical
# path entirely, so the GPU job can go straight to training.
#
# Both mel norm_mode variants are computed, because the choice between them is decided
# by the GPU gate's vocoder round-trip, which cannot run cheaply here. Computing both
# (~2 GB extra) removes the dependency instead of guessing. Training then reads whichever
# variant the gate blesses.
#
# Submit:
#   sbatch submission_scripts/run_tamia_s2a_precompute_cpu.sbatch
#SBATCH --job-name=salm-s2a-precompute-cpu
#SBATCH --account=aip-ravanelm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=240G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/l/lovenya/salm_migration_clean/logs/slurm/%x-%j.out

set -euo pipefail

# Absolute path on purpose: Slurm copies the submitted script to
# /var/spool/slurmd/job<N>/slurm_script, so $BASH_SOURCE points there, not at the repo,
# and a relative source would fail at runtime.
# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

SMOKE="${SMOKE:-0}"
RUN_ID="${RUN_ID:-$(TZ=America/Toronto date +%Y%m%d_%H%M)}"
TAG="s2a_precompute_cpu_${RUN_ID}"
LOG_DIR="$SALM_SCRATCH/logs/salm_model/$TAG"
mkdir -p "$LOG_DIR"

echo "==================================================================="
echo "S2A precompute on CPU  (smoke=$SMOKE)"
echo "job:    ${SLURM_JOB_ID:-none}   host: $(hostname)"
echo "commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "cpus:   ${SLURM_CPUS_PER_TASK:-?}"
echo "==================================================================="
python -c "import torch;print('torch',torch.__version__,'threads',torch.get_num_threads())"

salm_start_monitor "$LOG_DIR" "$TAG" 30

# 4 concurrent processes (2 norm modes x 2 splits) on 48 cores.
export OMP_NUM_THREADS=11
export MKL_NUM_THREADS=11

PRE_COMMON=(--direction s2a --device cpu --batch-size 8 --overwrite)
if [[ "$SMOKE" == "1" ]]; then
    PRE_COMMON+=(--limit 32)
fi

declare -a PIDS=()
declare -a LABELS=()

for norm in none audiomae; do
    for split in train val; do
        label="${norm}_${split}"
        python scripts/precompute_latents.py \
            "${PRE_COMMON[@]}" \
            --norm-mode "$norm" \
            --split "$split" \
            --out-dir "$OUT_ROOT/features_cpu/norm_${norm}" \
            > "$LOG_DIR/precompute_${label}.log" 2>&1 &
        PIDS+=("$!")
        LABELS+=("$label")
        echo "launched $label (pid $!)"
    done
done

FAIL=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "OK    ${LABELS[$i]}"
    else
        echo "FAIL  ${LABELS[$i]}" >&2
        FAIL=1
    fi
done

echo
echo "### tails ###"
for label in "${LABELS[@]}"; do
    echo "--- $label ---"
    tail -4 "$LOG_DIR/precompute_${label}.log"
done

echo
echo "### Summary ###"
python - <<PY
import json, pathlib
root = pathlib.Path("$OUT_ROOT/features_cpu")
for norm in ("none", "audiomae"):
    for split in ("train", "val"):
        p = root / f"norm_{norm}" / "s2a" / split / "meta.json"
        if p.exists():
            d = json.loads(p.read_text())
            print(f"norm={norm:9s} {split:5s}: kept={d['kept']}/{d['manifest_rows']} "
                  f"dropped={d['dropped']} latent={d['latent_shape']} "
                  f"({d['elapsed_seconds']/60:.1f} min)")
        else:
            print(f"norm={norm:9s} {split:5s}: MISSING")
PY

echo
echo "features: $OUT_ROOT/features_cpu/norm_{none,audiomae}/s2a/{train,val}"
echo "logs:     $LOG_DIR"
if [[ "$FAIL" != "0" ]]; then
    echo "ERROR: at least one precompute lane failed." >&2
    exit 1
fi
echo "OK"
