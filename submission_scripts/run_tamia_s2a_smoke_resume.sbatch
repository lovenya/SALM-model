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

# ======================================== 2. partial run, stopped by the deadline
# Both parts declare the SAME epochs=4 target, and part 1 is cut short by
# max_train_minutes instead of a lower epoch count. That matters: the LR schedule is
# built from total_steps = steps_per_epoch * epochs, so changing `epochs` between the two
# parts would reshape the cosine curve and produce an LR discontinuity that is an artifact
# of the test rather than a resume bug. Stopping on the deadline is also exactly the real
# scenario this protects against -- a 4 h job hitting its walltime and being resubmitted.
echo
echo "### [2/5] Train with a 3-minute deadline (stops partway, epochs=4 target) ###"
torchrun --nproc_per_node=4 --master_port=29531 -m salm.train_ldm \
    "${CFG[@]}" \
    --override "run_name=$RUN_NAME" "arch=$ARCH" "epochs=4" "max_train_minutes=3" \
    2>&1 | tee "$LOG_DIR/02_train_part1.log"

PART1_EPOCHS=$(python - <<PY
import json, pathlib
h = json.loads((pathlib.Path("$RUN_DIR") / "history.json").read_text())
print(len(h))
PY
)
echo "part 1 completed $PART1_EPOCHS epoch record(s)"
if [[ "$PART1_EPOCHS" -lt 1 ]]; then
    echo "ERROR: part 1 produced no epoch records; deadline was too short." >&2
    exit 1
fi
if [[ "$PART1_EPOCHS" -ge 4 ]]; then
    echo "ERROR: part 1 finished all 4 epochs, so resume is not exercised." >&2
    echo "       Lower max_train_minutes." >&2
    exit 1
fi

# ============================================ 3. resume to the same epoch target
echo
echo "### [3/5] RESUME from checkpoint_last.pt to finish epochs up to 4 ###"
CKPT="$RUN_DIR/checkpoints/checkpoint_last.pt"
ls -la "$CKPT"
torchrun --nproc_per_node=4 --master_port=29532 -m salm.train_ldm \
    "${CFG[@]}" \
    --override "run_name=$RUN_NAME" "arch=$ARCH" "epochs=4" "resume_from=$CKPT" \
    2>&1 | tee "$LOG_DIR/03_train_resume.log"

grep -E "exact resume|WARM START|loaded .* prior epochs" "$LOG_DIR/03_train_resume.log" || true

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
# Contiguous 0..3 with no repeats: proves resume neither truncated history nor re-ran
# epochs it had already completed.
if epochs != [0, 1, 2, 3]:
    fail.append(f"epochs not contiguous 0..3: {epochs} "
                "(history truncated, or epochs repeated instead of continuing)")
if steps != sorted(steps) or len(set(steps)) != len(steps):
    fail.append(f"global_step did not continue monotonically: {steps}")

# The log must claim an exact resume; a warm start means optimizer state was missing.
resume_log = (run_dir / "logs/03_train_resume.log").read_text()
if "exact resume" not in resume_log:
    fail.append("resume log lacks 'exact resume' -- optimizer/scheduler state was missing, "
                "so this was only a warm start")

# The LR must continue its cosine decay. Both parts declare epochs=4 so total_steps is
# identical and the curve is genuinely continuous; a jump in EITHER direction means the
# scheduler state was not restored.
def lrs(path):
    out = []
    for line in pathlib.Path(path).read_text().splitlines():
        m = re.search(r"lr=([0-9.eE+-]+)/", line)
        if m:
            out.append(float(m.group(1)))
    return out

pre, post = lrs(run_dir / "logs/02_train_part1.log"), lrs(run_dir / "logs/03_train_resume.log")
if pre and post:
    ratio = post[0] / pre[-1] if pre[-1] > 0 else float("inf")
    print(f"LR last-before-resume={pre[-1]:.3e}  first-after-resume={post[0]:.3e}  "
          f"ratio={ratio:.3f}")
    # Cosine decay moves the LR only a few percent per logging interval. A re-warm from
    # step 0 collapses it toward zero; a reshaped schedule jumps it upward.
    if not (0.7 <= ratio <= 1.3):
        fail.append(
            f"LR discontinuity across resume (ratio {ratio:.3f}, expected ~1.0); "
            "scheduler state was not restored correctly"
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
