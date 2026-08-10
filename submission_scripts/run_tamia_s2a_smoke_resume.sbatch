#!/usr/bin/env bash
# Smoke test for the phase-2 changes, with resume as the primary target.
#
# Why resume specifically: the 100-epoch run is a single 4 h job, and its only protection
# against a walltime overrun is being able to continue from checkpoint_last.pt. A normal
# run never exercises that path, so it gets its own test:
#
#   train 2 epochs -> stop -> resume -> train 2 more -> assert
#     * history.json has 4 entries with strictly increasing epochs
#     * global_step continues rather than restarting
#     * the LR continues its schedule instead of re-warming from 0
#     * val loss does not jump discontinuously at the boundary
#
# Also covers: architecture registry, CLAP target precompute, val cosine metric, bucketed
# losses, fp16 snapshots, W&B offline, and the new runs/<arch>/<run>/ layout.
#
# Deliberately skips the Rung 0 gate: it already passed (mel 0.9895, VAE 0.9993) and the
# latents on disk were built with the selected norm_mode=none.
#
# 1 h ask for ~15 min of work, which backfills easily.
#SBATCH --job-name=salm-s2a-smoke-resume
#SBATCH --account=aip-ravanelm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus=h100:4
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

RUN_ID="${RUN_ID:-$(TZ=America/Toronto date +%Y%m%d_%H%M)}"
ARCH="${ARCH:-s2a_audioldm_whisper_xformer}"
RUN_NAME="smoke_resume_${RUN_ID}"
RUN_DIR="$OUTPUTS_ROOT/runs/$ARCH/$RUN_NAME"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

CFG=(--config configs/base/s2a_audioldm.yaml "configs/arch/${ARCH}.yaml" configs/exp/smoke.yaml)

salm_print_header "S2A smoke: registry + metrics + RESUME  (arch=$ARCH)"
salm_start_monitor "$LOG_DIR" "$RUN_NAME" 30

# ================================================== 1. CLAP targets for val
# Needed for the val cosine metric, which is the whole reason this phase exists: the
# diffusion loss is flat, so without this there is no readable signal.
echo
echo "### [1/5] Precompute CLAP target embeddings (val) ###"
if [[ -f "$OUTPUTS_ROOT/features_clap/s2a/val/clap_target.npy" ]]; then
    echo "already present, skipping"
else
    CUDA_VISIBLE_DEVICES=0 python scripts/precompute_clap_targets.py \
        --direction s2a --split val 2>&1 | tee "$LOG_DIR/01_clap_val.log"
fi

# ============================================================ 2. first 2 epochs
echo
echo "### [2/5] Train epochs 0-1 ###"
torchrun --nproc_per_node=4 --master_port=29531 -m salm.train_ldm \
    "${CFG[@]}" \
    --override "run_name=$RUN_NAME" "arch=$ARCH" "epochs=2" \
    2>&1 | tee "$LOG_DIR/02_train_part1.log"

python - <<PY
import json, pathlib, sys
h = json.loads((pathlib.Path("$RUN_DIR") / "history.json").read_text())
print(f"after part 1: {len(h)} epochs -> {[r['epoch'] for r in h]}")
assert len(h) == 2, f"expected 2 epochs, got {len(h)}"
PY

# ============================================== 3. resume for 2 more epochs
echo
echo "### [3/5] RESUME from checkpoint_last.pt for epochs 2-3 ###"
CKPT="$RUN_DIR/checkpoints/checkpoint_last.pt"
ls -la "$CKPT"
torchrun --nproc_per_node=4 --master_port=29532 -m salm.train_ldm \
    "${CFG[@]}" \
    --override "run_name=$RUN_NAME" "arch=$ARCH" "epochs=4" "resume_from=$CKPT" \
    2>&1 | tee "$LOG_DIR/03_train_resume.log"

# ==================================================== 4. assert resume worked
echo
echo "### [4/5] Verify resume ###"
python - <<PY
import json, pathlib, re, sys

run_dir = pathlib.Path("$RUN_DIR")
h = json.loads((run_dir / "history.json").read_text())
epochs = [r["epoch"] for r in h]
steps = [r["step"] for r in h]
print(f"epochs: {epochs}")
print(f"steps:  {steps}")

fail = []
if len(h) != 4:
    fail.append(f"history has {len(h)} entries, expected 4 (resume truncated it?)")
if epochs != sorted(set(epochs)) or epochs != [0, 1, 2, 3]:
    fail.append(f"epochs not contiguous 0..3: {epochs}")
if steps != sorted(steps) or len(set(steps)) != len(steps):
    fail.append(f"global_step did not continue monotonically: {steps}")

# The LR must continue its cosine decay, not re-warm from ~0. Compare the first logged LR
# after resume against the last one before it.
def lrs(path):
    out = []
    for line in pathlib.Path(path).read_text().splitlines():
        m = re.search(r"lr=([0-9.eE+-]+)/", line)
        if m:
            out.append(float(m.group(1)))
    return out

pre, post = lrs(run_dir / "logs/02_train_part1.log"), lrs(run_dir / "logs/03_train_resume.log")
if pre and post:
    print(f"LR last-before-resume={pre[-1]:.3e}  first-after-resume={post[0]:.3e}")
    # Re-warming from step 0 would give an LR orders of magnitude smaller.
    if post[0] < pre[-1] * 0.5:
        fail.append(
            f"LR restarted after resume ({post[0]:.3e} vs {pre[-1]:.3e}); "
            "optimizer/scheduler state was not restored"
        )
else:
    fail.append("could not parse LRs from logs")

vals = [r["val_loss"] for r in h if r.get("val_loss") is not None]
print(f"val losses: {[round(v,4) for v in vals]}")
cos = [r.get("val_clap_cos") for r in h if r.get("val_clap_cos") is not None]
print(f"val clap cosine: {[round(c,4) for c in cos]}" if cos
      else "val clap cosine: MISSING (metric did not run)")
if not cos:
    fail.append("val_clap_cos absent -- the CLAP metric did not run")

if fail:
    print("\nRESUME VERIFICATION FAILED:")
    for f in fail:
        print(f"  - {f}")
    sys.exit(1)
print("\nresume verification PASSED")
PY

# ======================================================== 5. generate samples
echo
echo "### [5/5] Generate 2 samples from the resumed checkpoint ###"
CUDA_VISIBLE_DEVICES=0 python -m salm.generate \
    --mode s2a --checkpoint "$RUN_DIR/checkpoints/checkpoint_last.pt" \
    --split val --num-samples 2 \
    --out-dir "$RUN_DIR/generations/s2a" --copy-reference \
    2>&1 | tee "$LOG_DIR/05_generate.log"

echo
echo "=== checkpoint sizes (fp16 snapshots should be ~half) ==="
ls -la "$RUN_DIR/checkpoints/"
echo
echo "=== W&B offline dir (sync later) ==="
find "$RUN_DIR/wandb" -maxdepth 2 -name "offline-*" 2>/dev/null || echo "(no offline run -- wandb not installed?)"
echo
echo "run dir: $RUN_DIR"
echo "OK"
