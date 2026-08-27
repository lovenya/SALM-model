#!/usr/bin/env bash
# A2S stage 2: train the conditioner + U-Net on ambience -> speech, from a fresh start.
#
# FRESH, NOT WARM-STARTED FROM S2A. User's decision, 2026-08-23: get A2S working as its own
# model first, then consider merging weights, joint bidirectional training, or a single model
# with a direction conditioner. "Fresh" means no S2A checkpoint -- the pretrained AudioLDM
# base (U-Net, VAE, vocoder) is still used, because a randomly initialised U-Net on 17.6k
# pairs is hopeless and we have the measurement to prove the pretrained one matters (a
# pristine, non-finetuned U-Net scored MARGIN -0.0036 against a finetuned +0.2707).
#
# TWO LANES, which is the whole design of the first cohort. It re-earns the S2A ablation for
# this direction instead of assuming it transfers:
#   A0  plain diffusion loss           expected to sit at chance, as L0 and L2 did on S2A
#   A1  + O2 alignment (lambda 1.0)    expected to work, as it did on S2A
# Run them as two submissions of this script so they queue independently:
#
#   RUN_NAME=probe40_A0_plain sbatch submission_scripts/run_tamia_a2s_stage2_train.sbatch
#   RUN_NAME=probe40_A1_o2 EXTRA=configs/exp/o2_align.yaml sbatch submission_scripts/run_tamia_a2s_stage2_train.sbatch
#
# AUTO-RESUME: RUN_NAME is stable, and the script resumes from checkpoint_last.pt if present.
# Resubmitting the same command after a walltime kill continues rather than restarting.
#
# SIZING IS AN ESTIMATE, not a measurement. There is no encoder in an A2S step -- the CLAP
# features are precomputed -- so the U-Net dominates and epochs should land well under S2A's
# 2.16 min at 274 steps/epoch. 40 epochs is budgeted at ~60 min inside a 2 h ask. The FIRST
# run of this script is also the measurement: read `mean epoch time` from the summary and
# tighten TRAIN_MINUTES afterwards.
#
# PREREQUISITE: stage 1 must have run (gate + both precomputes). The trainer refuses to start
# without the caches rather than training on a partial one.
#SBATCH --job-name=salm-a2s-train
#SBATCH --account=aip-csubakan
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus=h100:4
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

ARCH="${ARCH:-a2s_audioldm_clapseq_xformer}"
RUN_NAME="${RUN_NAME:-probe40_A0_plain}"
EXP="${EXP:-configs/exp/a2s_probe40.yaml}"
EPOCHS="${EPOCHS:-40}"
TRAIN_MINUTES="${TRAIN_MINUTES:-95}"
NUM_GEN="${NUM_GEN:-16}"
ALIGN_TARGET="${ALIGN_TARGET:-text}"
COND_ROOT="${COND_ROOT:-$OUTPUTS_ROOT/features_a2s_cond}"
# Both default to every earlier A2S probe's value (batch_size=16 -> 274 steps/epoch), so a
# plain invocation is unchanged. Override together -- lr_total_steps must equal
# steps_per_epoch * EPOCHS for THIS batch_size, or the LR curve lands at the wrong position.
BATCH_SIZE="${BATCH_SIZE:-16}"
LR_TOTAL_STEPS="${LR_TOTAL_STEPS:-}"

RUN_DIR="$RUNS_ROOT/$ARCH/$RUN_NAME"
LOG_DIR="$RUN_DIR/logs"
GEN_DIR="$RUN_DIR/generations"
mkdir -p "$LOG_DIR" "$GEN_DIR"

# ------------------------------------------------- prerequisites, checked up front
# Reject before loading a single model. The submit-jobs skill's rule: never discover a
# missing input at minute 30 of an allocation.
MISSING=0
for SPLIT in train val; do
    for F in "$FEATURES_ROOT/a2s/$SPLIT/latents_mean.npy" "$COND_ROOT/$SPLIT/cond_seq.npy"; do
        [[ -f "$F" ]] || { echo "MISSING: $F" >&2; MISSING=1; }
    done
done
if [[ "$MISSING" == "1" ]]; then
    echo "ERROR: run stage 1 first:" >&2
    echo "  sbatch submission_scripts/run_tamia_a2s_stage1_gate_precompute.sbatch" >&2
    exit 1
fi

# The A2S latents must be of the SPEECH. A cache built with the default --target audio is
# byte-plausible and would train the model to regenerate its own input, silently.
python - <<PY || exit 1
import json, pathlib, sys
for split in ("train", "val"):
    m = json.loads(pathlib.Path(f"$FEATURES_ROOT/a2s/{split}/meta.json").read_text())
    t = m.get("target")
    if t is not None and t != "speech":
        print(f"ERROR: a2s/{split} latents were built with target={t!r}, expected 'speech'.",
              file=sys.stderr)
        print("  Re-run: python scripts/precompute_latents.py --direction a2s "
              f"--target speech --split {split} --overwrite", file=sys.stderr)
        raise SystemExit(1)
    print(f"  a2s/{split}: {m['kept']} rows, target={t}, align={m.get('crop_align')}")
PY

cat > "$RUN_DIR/run_metadata.json" <<EOF
{
  "run_name": "$RUN_NAME",
  "arch": "$ARCH",
  "direction": "a2s",
  "exp": "$EXP",
  "extra": "${EXTRA:-}",
  "align_target": "$ALIGN_TARGET",
  "slurm_job_id": "${SLURM_JOB_ID:-none}",
  "account": "${SLURM_JOB_ACCOUNT:-unknown}",
  "hostname": "$(hostname)",
  "repo_path": "$REPO",
  "git_commit": "$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)",
  "git_dirty": $(if [[ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]]; then echo true; else echo false; fi),
  "venv": "$VENV",
  "epochs": $EPOCHS,
  "train_minutes": $TRAIN_MINUTES,
  "started_at": "$(TZ=America/Toronto date -Iseconds)"
}
EOF

salm_print_header "A2S train: $EPOCHS epochs  arch=$ARCH  run=$RUN_NAME  extra=${EXTRA:-none}"
salm_start_monitor "$LOG_DIR" "$RUN_NAME" 60

# ======================================================== 1. train (auto-resume)
echo
echo "### [1/2] Training ###"
CKPT="$RUN_DIR/checkpoints/checkpoint_last.pt"
RESUME_ARGS=()
if [[ -f "$CKPT" ]]; then
    echo "resuming from existing checkpoint: $CKPT"
    python -c "
import torch
p=torch.load('$CKPT',map_location='cpu',weights_only=False)
print(f\"  epoch={p['epoch']} step={p['step']} resumable={'optimizer' in p}\")"
    RESUME_ARGS=("resume_from=$CKPT")
else
    echo "fresh start (no checkpoint_last.pt in $RUN_DIR/checkpoints)"
fi

read -r -a EXTRA_CONFIGS <<< "${EXTRA:-}"
for f in ${EXTRA_CONFIGS[@]+"${EXTRA_CONFIGS[@]}"}; do
    [[ -f "$REPO/$f" ]] || { echo "ERROR: EXTRA config not found: $REPO/$f" >&2; exit 1; }
    echo "extra config: $f"
done

# lr_total_steps only overridden when explicitly set -- otherwise leave the EXP config's own
# default alone, since it's tuned to that config's own EPOCHS/batch_size already.
BATCH_ARGS=("batch_size=$BATCH_SIZE")
[[ -n "$LR_TOTAL_STEPS" ]] && BATCH_ARGS+=("lr_total_steps=$LR_TOTAL_STEPS")

torchrun --nproc_per_node=4 --master_port=29543 -m salm.train_ldm \
    --config configs/base/a2s_audioldm.yaml "configs/arch/${ARCH}.yaml" "$EXP" \
        ${EXTRA_CONFIGS[@]+"${EXTRA_CONFIGS[@]}"} \
    --override \
        "run_name=$RUN_NAME" \
        "arch=$ARCH" \
        "epochs=$EPOCHS" \
        "align_target=$ALIGN_TARGET" \
        "cond_features_root=$COND_ROOT" \
        "max_train_minutes=$TRAIN_MINUTES" \
        "${BATCH_ARGS[@]}" \
        ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
    2>&1 | tee -a "$LOG_DIR/01_train.log"

# =============================================================== 2. generation
# Two lanes so the output is interpretable the moment the job ends:
#   a2s   the model, conditioned on real ambience
#   b0    the unconditional floor -- the same U-Net driven by its own learned null embedding.
#         Without it, "the audio sounds like speech" proves nothing, because a finetuned
#         U-Net produces speech-shaped output whether or not it read the condition.
echo
echo "### [2/2] Generation ###"
BEST="$RUN_DIR/checkpoints/checkpoint_best.pt"
[[ -f "$BEST" ]] || BEST="$RUN_DIR/checkpoints/checkpoint_last.pt"

CUDA_VISIBLE_DEVICES=0 python -m salm.generate \
    --mode a2s --direction a2s --checkpoint "$BEST" --split val \
    --num-samples "$NUM_GEN" --out-dir "$GEN_DIR/a2s_best" --copy-reference \
    > "$LOG_DIR/02_gen_a2s.log" 2>&1 &
GEN_A=$!
CUDA_VISIBLE_DEVICES=1 python -m salm.generate \
    --mode b0 --direction a2s --checkpoint "$BEST" --split val \
    --num-samples "$NUM_GEN" --out-dir "$GEN_DIR/b0_uncond" \
    > "$LOG_DIR/02_gen_b0.log" 2>&1 &
GEN_B=$!
GEN_FAIL=0
wait "$GEN_A" || GEN_FAIL=1
wait "$GEN_B" || GEN_FAIL=1
tail -n 5 "$LOG_DIR/02_gen_a2s.log" "$LOG_DIR/02_gen_b0.log"

# ===================================================================== report
echo
echo "==================== A2S RUN SUMMARY ===================="
python - <<PY
import json, pathlib
run = pathlib.Path("$RUN_DIR")
hp = run / "history.json"
h = json.loads(hp.read_text()) if hp.exists() else []
print(f"epochs completed: {len(h)}")
if h:
    def series(k):
        return [(r["epoch"], r[k]) for r in h if r.get(k) is not None]

    vals = series("val_loss")
    if vals:
        best = min(vals, key=lambda x: x[1])
        print(f"val loss : first={vals[0][1]:.4f} last={vals[-1][1]:.4f} "
              f"best={best[1]:.4f} @ep{best[0]}   (expected to be UNINFORMATIVE)")

    r1 = series("val_clap_r1")
    chance = next((r["val_clap_r1_chance"] for r in h
                   if r.get("val_clap_r1_chance") is not None), None)
    if r1:
        top = max(r1, key=lambda x: x[1])
        print(f"val_clap_r1: first={r1[0][1]:.4f} last={r1[-1][1]:.4f} "
              f"best={top[1]:.4f} @ep{top[0]}")
        if chance:
            print(f"             chance={chance:.4f}  ->  best is {top[1]/chance:.1f}x chance")
            if top[1] < chance * 1.5:
                print("  >>> AT CHANCE. The conditioner is not learning ambience -> speech.")
                print("      This is what S2A's plain objective did. If this is the O2 lane,")
                print("      the S2A fix does NOT transfer and the interface is the suspect.")
            else:
                print("  >>> LEARNING. The conditioner reads the ambience.")

    cos = series("val_clap_cos")
    if cos:
        print(f"val_clap_cos: first={cos[0][1]:+.4f} last={cos[-1][1]:+.4f}")
    d2 = series("val_d2_gap")
    if d2:
        print(f"val_d2_gap : first={d2[0][1]:.4f} last={d2[-1][1]:.4f}   "
              "(how far conditioning moves the prediction)")
    et = [r.get("epoch_minutes", 0) for r in h]
    print(f"mean epoch time: {sum(et)/max(len(et),1):.2f} min  "
          "<-- use this to set TRAIN_MINUTES for the 100-epoch run")

for name in ("a2s_best", "b0_uncond"):
    s = pathlib.Path("$GEN_DIR") / name / "summary.json"
    if s.exists():
        d = json.loads(s.read_text())
        print(f"{name:10s} rms={d['rms_mean']:.5f} near_silent={d['near_silent_count']}"
              f"/{d['num_samples']} unet_loaded={d.get('unet_weights_loaded')}")
PY
echo "========================================================="
echo "run dir: $RUN_DIR"
echo "listen : $GEN_DIR/a2s_best/audio/   (vs $GEN_DIR/b0_uncond/audio/)"
echo "         each *__reference.wav is the TARGET speech for that row"
echo "W&B    : bash scripts/wandb_sync.sh $RUN_DIR"
echo "========================================================="
[[ "$GEN_FAIL" == "0" ]] || { echo "ERROR: a generation lane failed" >&2; exit 1; }
echo "OK"
