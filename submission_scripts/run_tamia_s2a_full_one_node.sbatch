#!/usr/bin/env bash
# Full S2A chain in ONE allocation: Rung 0 gate -> precompute -> Rung 3 training -> generation.
#
# Why combined rather than the stage1/stage2 split: the split isolates failures nicely,
# but each job releases the node and re-queues, and a second whole-node wait is
# unpredictable. Against a hard deadline, one queue wait beats better isolation. Use the
# split scripts when there is no deadline pressure.
#
# Stages still fail fast, so a broken gate does not burn the rest of the allocation.
#
# Resources: TamIA h100 node = 48 CPUs / 500000M / 4x H100-80GB. A GPU allocation
# reserves the whole node, so CPU/RAM asks are cgroup limits inside it.
#SBATCH --job-name=salm-s2a-full
#SBATCH --account=aip-ravanelm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus=h100:4
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/l/lovenya/salm_migration_clean/logs/slurm/%x-%j.out

set -euo pipefail

# Absolute path on purpose: Slurm copies the submitted script to
# /var/spool/slurmd/job<N>/slurm_script, so $BASH_SOURCE points there, not at the repo.
# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

SMOKE="${SMOKE:-0}"
RUN_ID="${RUN_ID:-$(TZ=America/Toronto date +%Y%m%d_%H%M)}"
RUN_NAME="s2a_rung3_${RUN_ID}"
[[ "$SMOKE" == "1" ]] && RUN_NAME="s2a_rung3_smoke_${RUN_ID}"

RUN_DIR="$OUT_ROOT/checkpoints/$RUN_NAME"
LOG_DIR="$SALM_SCRATCH/logs/salm_model/$RUN_NAME"
GEN_DIR="$OUT_ROOT/generations/$RUN_NAME"
GATE_DIR="$OUT_ROOT/rung0_gate/$RUN_NAME"

if [[ -d "$RUN_DIR" ]]; then
    echo "ERROR: run dir already exists: $RUN_DIR" >&2
    exit 1
fi
mkdir -p "$RUN_DIR" "$LOG_DIR" "$GEN_DIR" "$GATE_DIR"

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
  "smoke": "$SMOKE",
  "started_at": "$(TZ=America/Toronto date -Iseconds)"
}
EOF

salm_print_header "S2A full chain: gate -> precompute -> train -> generate  (smoke=$SMOKE)"
salm_start_monitor "$LOG_DIR" "$RUN_NAME" 30

STAGE_T0=$(date +%s)
stage_elapsed() { echo "[+$((($(date +%s) - STAGE_T0) / 60)) min]"; }

# ============================================================== 1. Rung 0 gate
echo
echo "### [1/4] Rung 0 gate $(stage_elapsed) ###"
CUDA_VISIBLE_DEVICES=0 python scripts/check_audioldm_stack.py \
    --out-dir "$GATE_DIR" --device cuda 2>&1 | tee "$LOG_DIR/01_gate.log"

read -r BEST_NORM MEL_CORR VAE_CORR <<< "$(python - <<PY
import json, pathlib
d = json.loads(pathlib.Path("$GATE_DIR/report.json").read_text())
best = d.get("best_norm_mode", "none")
mel = d.get("mel_roundtrip", {}).get("mel_corr", float("nan"))
vae = d.get("vae_roundtrip", {}).get("mel_corr", float("nan"))
print(best, f"{mel:.4f}", f"{vae:.4f}")
PY
)"
echo
echo ">>> GATE: norm_mode=$BEST_NORM mel_corr=$MEL_CORR vae_corr=$VAE_CORR"

# Gate on BOTH round-trips. The mel check alone is not sufficient: a previous run passed
# mel_corr=0.9895 and still produced vae_corr=-0.1520, i.e. the VAE was being handed
# mis-scaled input and its latents were meaningless. Training on those latents would run
# happily and the loss would fall -- toward a target with no signal in it.
python - <<PY
import sys
mel, vae = float("$MEL_CORR"), float("$VAE_CORR")
if not (mel >= 0.80):
    sys.exit(
        f"ABORT: mel round-trip corr {mel:.4f} < 0.80. The mel does not match AudioLDM's "
        "vocoder. Listen to $GATE_DIR/01_melroundtrip.wav and fix frontend.py."
    )
if not (vae >= 0.80):
    sys.exit(
        f"ABORT: VAE round-trip corr {vae:.4f} < 0.80 (mel was fine at {mel:.4f}). The VAE "
        "cannot reconstruct our mel, so precomputed latents would carry no usable signal. "
        "Inspect $GATE_DIR/02_vae_roundtrip_*.wav and the per-norm scores in report.json."
    )
print(f"gate OK -- mel {mel:.4f}, VAE {vae:.4f}")
PY

# ========================================================= 2. Precompute latents
echo
echo "### [2/4] Precompute VAE latents (norm_mode=$BEST_NORM) $(stage_elapsed) ###"
PRE_ARGS=(--direction s2a --norm-mode "$BEST_NORM" --overwrite --load-workers 12)
[[ "$SMOKE" == "1" ]] && PRE_ARGS+=(--limit 64)

CUDA_VISIBLE_DEVICES=0 python scripts/precompute_latents.py \
    "${PRE_ARGS[@]}" --split train > "$LOG_DIR/02_precompute_train.log" 2>&1 &
PRE_TRAIN=$!
CUDA_VISIBLE_DEVICES=1 python scripts/precompute_latents.py \
    "${PRE_ARGS[@]}" --split val > "$LOG_DIR/02_precompute_val.log" 2>&1 &
PRE_VAL=$!

PRE_FAIL=0
wait "$PRE_TRAIN" || PRE_FAIL=1
wait "$PRE_VAL" || PRE_FAIL=1
tail -n 5 "$LOG_DIR/02_precompute_train.log" "$LOG_DIR/02_precompute_val.log"
[[ "$PRE_FAIL" != "0" ]] && { echo "ERROR: precompute failed." >&2; exit 1; }

# ================================================================= 3. Training
echo
echo "### [3/4] Rung 3 training (AudioLDM eps-MSE + CFG dropout, DDP x4) $(stage_elapsed) ###"
TRAIN_MINUTES="${TRAIN_MINUTES:-45}"
EPOCHS="${EPOCHS:-40}"
if [[ "$SMOKE" == "1" ]]; then TRAIN_MINUTES=4; EPOCHS=2; fi
echo "internal deadline: ${TRAIN_MINUTES} min (stops cleanly, validates, checkpoints)"
echo "epoch cap ${EPOCHS} -- the deadline is expected to bind first"

torchrun --nproc_per_node=4 --master_port=29517 -m salm.train_ldm \
    --config configs/s2a_rung3.yaml \
    --override \
        "run_name=$RUN_NAME" \
        "output_root=$OUT_ROOT/checkpoints" \
        "max_train_minutes=$TRAIN_MINUTES" \
        "epochs=$EPOCHS" \
    2>&1 | tee "$LOG_DIR/03_train.log"

CKPT="$RUN_DIR/checkpoint_best.pt"
[[ -f "$CKPT" ]] || { echo "WARNING: no best ckpt, using last" >&2; CKPT="$RUN_DIR/checkpoint_last.pt"; }
[[ -f "$CKPT" ]] || { echo "ERROR: training produced no checkpoint." >&2; exit 1; }

# =============================================================== 4. Generation
# Our model and the rung0 text ceiling on separate GPUs over the SAME val rows, so they
# are directly A/B comparable.
echo
echo "### [4/4] Generation $(stage_elapsed) ###"
NUM_GEN="${NUM_GEN:-16}"
[[ "$SMOKE" == "1" ]] && NUM_GEN=2

CUDA_VISIBLE_DEVICES=0 python -m salm.generate \
    --mode s2a --checkpoint "$CKPT" --split val --num-samples "$NUM_GEN" \
    --out-dir "$GEN_DIR/s2a" --copy-reference > "$LOG_DIR/04_gen_s2a.log" 2>&1 &
GEN_S2A=$!
CUDA_VISIBLE_DEVICES=1 python -m salm.generate \
    --mode rung0 --split val --num-samples "$NUM_GEN" \
    --out-dir "$GEN_DIR/rung0_text_ceiling" > "$LOG_DIR/04_gen_rung0.log" 2>&1 &
GEN_R0=$!

GEN_FAIL=0
wait "$GEN_S2A" || GEN_FAIL=1
wait "$GEN_R0" || GEN_FAIL=1
tail -n 5 "$LOG_DIR/04_gen_s2a.log" "$LOG_DIR/04_gen_rung0.log"

# ====================================================================== report
echo
echo "================== MEETING SUMMARY $(stage_elapsed) =================="
python - <<PY
import json, pathlib
run_dir = pathlib.Path("$RUN_DIR")
gen_dir = pathlib.Path("$GEN_DIR")
gate = pathlib.Path("$GATE_DIR/report.json")

if gate.exists():
    d = json.loads(gate.read_text())
    best = d.get("best_norm_mode")
    print("GATE (does AudioLDM's stack handle TTAS audio?)")
    print(f"  mel norm_mode selected : {best}")
    print(f"  mel  round-trip corr   : {d['mel_roundtrip']['mel_corr']:+.4f}  (vocoder inverts our mel)")
    print(f"  VAE  round-trip corr   : {d['vae_roundtrip']['mel_corr']:+.4f}  (VAE reconstructs our mel)")
    for norm, s in d.get("vae_roundtrip_by_norm", {}).items():
        print(f"     norm={norm:9s} VAE corr={s['mel_corr']:+.4f} audio_rms={s['rec_rms']:.5f}")
    print(f"  latent shape           : {d.get('latent_shape')}")

h = run_dir / "history.json"
if h.exists():
    hist = json.loads(h.read_text())
    print(f"\\nTRAINING ({len(hist)} epochs)")
    for r in hist:
        v = r["val_loss"]
        print(f"  ep{r['epoch']:02d} train={r['train_loss']:.4f} "
              f"val={'n/a' if v is None else f'{v:.4f}'}")
    vals = [r["val_loss"] for r in hist if r["val_loss"] is not None]
    if vals:
        print(f"  best val loss: {min(vals):.4f}  (first={vals[0]:.4f})")

print("\\nGENERATION")
for name, label in (("s2a", "our model (speech->audio)"),
                    ("rung0_text_ceiling", "ceiling (caption->audio)")):
    s = gen_dir / name / "summary.json"
    if s.exists():
        d = json.loads(s.read_text())
        print(f"  {label:28s} n={d['num_samples']:3d} rms={d['rms_mean']:.5f} "
              f"near_silent={d['near_silent_count']}")
    else:
        print(f"  {label:28s} MISSING")
PY

echo
echo "AUDIO TO PLAY:"
echo "  ours:    $GEN_DIR/s2a/audio/"
echo "  ceiling: $GEN_DIR/rung0_text_ceiling/audio/"
echo "  (files named <id>.wav and <id>__reference.wav for A/B)"
echo "logs: $LOG_DIR"
echo "=================================================================="

[[ "$GEN_FAIL" != "0" ]] && { echo "ERROR: a generation lane failed." >&2; exit 1; }
echo "OK"
