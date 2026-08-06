#!/usr/bin/env bash
# Stage 1: Rung 0 gate + VAE latent precompute.
#
# Split out from training as a SHORT job on purpose. A whole 4-GPU node is needed, and
# Slurm's backfill scheduler slots a 1 h job into far more gaps than a 3 h one, so this
# starts sooner. It also means a gate failure costs a 1 h allocation, not 3 h.
#
# Actual runtime is ~20 min; 1 h is the ask, which keeps a comfortable buffer while
# staying in the short-job backfill window.
#
# Fully decoupled from stage 2: this writes latents plus a meta.json recording the mel
# norm_mode it selected, and stage 2 reads that. No run-ID coordination needed.
#
# Resources: TamIA h100 node = 48 CPUs / 500000M / 4x H100-80GB. The GPU allocation
# reserves the whole node, so CPU/RAM asks are cgroup limits inside it -- under-asking
# would only starve our own workers.
#SBATCH --job-name=salm-s2a-stage1
#SBATCH --account=aip-ravanelm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus=h100:4
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/l/lovenya/salm_migration_clean/logs/slurm/%x-%j.out

set -euo pipefail

# Absolute path on purpose: Slurm copies the submitted script to
# /var/spool/slurmd/job<N>/slurm_script, so $BASH_SOURCE points there, not at the repo,
# and a relative source would fail at runtime.
# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

SMOKE="${SMOKE:-0}"
RUN_ID="${RUN_ID:-$(TZ=America/Toronto date +%Y%m%d_%H%M)}"
TAG="s2a_stage1_${RUN_ID}"

LOG_DIR="$SALM_SCRATCH/logs/salm_model/$TAG"
GATE_DIR="$OUT_ROOT/rung0_gate/$TAG"
mkdir -p "$LOG_DIR" "$GATE_DIR"

salm_print_header "S2A stage 1: Rung 0 gate + precompute  (smoke=$SMOKE)"
salm_start_monitor "$LOG_DIR" "$TAG" 30

# ============================================================== 1. Rung 0 gate
# Hard gate. Verifies weights load offline, the mel round-trips through the vocoder,
# the VAE round-trips, and sampling produces audible audio. It also scores both mel
# normalization variants and picks the winner -- that is the one genuinely uncertain
# parameter in the front-end, so it is settled by measurement, not assumption.
echo
echo "### [1/2] Rung 0 gate ###"
CUDA_VISIBLE_DEVICES=0 python scripts/check_audioldm_stack.py \
    --out-dir "$GATE_DIR" --device cuda \
    2>&1 | tee "$LOG_DIR/01_gate.log"

read -r BEST_NORM MEL_CORR VAE_CORR <<< "$(python - <<PY
import json, pathlib
d = json.loads(pathlib.Path("$GATE_DIR/report.json").read_text())
best = d.get("best_norm_mode", "none")
mel = d.get("mel_roundtrip", {}).get(best, {}).get("mel_corr", float("nan"))
vae = d.get("vae_roundtrip", {}).get("mel_corr", float("nan"))
print(best, f"{mel:.4f}", f"{vae:.4f}")
PY
)"

echo
echo "gate results: norm_mode=$BEST_NORM mel_corr=$MEL_CORR vae_corr=$VAE_CORR"

# Refuse to precompute on a mel we cannot trust. A wrong mel scale makes the VAE encode
# out-of-distribution input, and training would still "work" -- loss would decrease
# toward a useless target -- so this must fail loudly here instead.
python - <<PY
import sys
mel = float("$MEL_CORR")
if not (mel >= 0.80):
    sys.exit(
        f"ABORT: mel round-trip correlation {mel:.4f} < 0.80. The mel representation "
        "does not match AudioLDM's vocoder, so latents would be garbage. "
        "Listen to $GATE_DIR/01_melroundtrip_*.wav and fix frontend.py before training."
    )
print(f"mel round-trip OK ({mel:.4f})")
PY

# ========================================================= 2. Precompute latents
# Train and val on separate GPUs, concurrently.
echo
echo "### [2/2] Precompute VAE latents (norm_mode=$BEST_NORM) ###"
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
tail -6 "$LOG_DIR/02_precompute_train.log" "$LOG_DIR/02_precompute_val.log"
if [[ "$PRE_FAIL" != "0" ]]; then
    echo "ERROR: precompute failed." >&2
    exit 1
fi

echo
echo "### Summary ###"
python - <<PY
import json, pathlib
for split in ("train", "val"):
    p = pathlib.Path("$FEATURES_ROOT/s2a") / split / "meta.json"
    if p.exists():
        d = json.loads(p.read_text())
        print(f"{split}: kept={d['kept']}/{d['manifest_rows']} dropped={d['dropped']} "
              f"latent={d['latent_shape']} norm={d['mel_config']['norm_mode']} "
              f"({d['elapsed_seconds']/60:.1f} min)")
    else:
        print(f"{split}: MISSING meta.json")
PY

echo
echo "features: $FEATURES_ROOT/s2a"
echo "gate:     $GATE_DIR"
echo "logs:     $LOG_DIR"
echo "OK -- stage 2 (training) can now run."
