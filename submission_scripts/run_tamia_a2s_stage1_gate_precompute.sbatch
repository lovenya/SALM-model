#!/usr/bin/env bash
# A2S stage 1: the speech-stack GATE, then every precompute the A2S trainer needs.
#
# THE GATE IS THE POINT. AudioLDM's front-end was chosen for ambient audio -- 16 kHz, 64 mel
# bins, fmax 8000, with a VAE and vocoder trained on general audio. Speech vocoders normally
# use 80 mel bins. If that stack cannot round-trip real speech back into intelligible WORDS,
# no amount of A2S training can help, because the model can never beat its own decoder's
# reconstruction ceiling. So this runs first, it is pure inference, and it is scored as WER
# against the ground-truth transcript rather than as a spectrogram correlation.
#
# EXIT CODE IS THE GATE. PASS or MARGINAL -> exit 0, so a chained stage 2 proceeds.
# FAIL -> exit 1, so `--dependency=afterok` holds stage 2 back instead of burning hours on a
# decoder that cannot carry speech.
#
# The precompute runs regardless of the verdict (it is ~20 min of a 1 h allocation, and the
# conditioner-side work is reusable even if the decoder has to be replaced).
#
# Runtime estimate ~35 min against a 1 h ask:
#   gate                      ~8 min   (AudioLDM + Whisper-large-v3, 24 clips x 3 paths)
#   VAE latents of speech     ~2 min   (S2A measured 65 s for 15,033 rows; a2s has 17,596)
#   CLAP conditioning        ~20 min   (measured 31.6 rows/s for one audio forward; we do
#                                       two audio forwards + one text forward per row)
# Splits run concurrently on separate GPUs, which is what keeps the last item near 20 min.
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_stage1_gate_precompute.sbatch
#   SKIP_SPEECH_ALIGN=1 sbatch ...     # halve the CLAP pass, lose the align_speech control
#SBATCH --job-name=salm-a2s-stage1
#SBATCH --account=aip-csubakan
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
TAG="a2s_stage1_${RUN_ID}"
NUM_CLIPS="${NUM_CLIPS:-24}"
COND_ROOT="${COND_ROOT:-$OUTPUTS_ROOT/features_a2s_cond}"

SKIP_ALIGN_FLAG=()
[[ "${SKIP_SPEECH_ALIGN:-0}" == "1" ]] && SKIP_ALIGN_FLAG=(--skip-speech-align)

LOG_DIR="$OUTPUTS_ROOT/logs/salm_model/$TAG"
GATE_DIR="$OUTPUTS_ROOT/a2s_gate/$TAG"
mkdir -p "$LOG_DIR" "$GATE_DIR"

salm_print_header "A2S stage 1: speech-stack gate + precompute  ($TAG)"
salm_start_monitor "$LOG_DIR" "$TAG" 30

# ==================================================================== 1. gate
echo
echo "### [1/3] A2S gate: can this stack reconstruct intelligible speech? ###"
CUDA_VISIBLE_DEVICES=0 python scripts/a2s/check_speech_stack.py \
    --out-dir "$GATE_DIR" --split val --num-clips "$NUM_CLIPS" --device cuda \
    2>&1 | tee "$LOG_DIR/01_gate.log"

VERDICT=$(python -c "
import json,pathlib
print(json.loads(pathlib.Path('$GATE_DIR/report.json').read_text())['verdict'])")
echo "gate verdict: $VERDICT"

# ================================================= 2. VAE latents of the SPEECH
# --target speech is the whole point: A2S generates speech, so the target latents must be
# the utterance, not the ambience. The flag defaults correctly from --direction a2s, and is
# passed explicitly here so the command reads unambiguously in the log.
echo
echo "### [2/3] VAE latents of the TARGET speech ###"
PIDS=(); GPU=0
for SPLIT in train val test; do
    if [[ -f "$FEATURES_ROOT/a2s/$SPLIT/latents_mean.npy" ]]; then
        echo "  $SPLIT: already present, skipping"
        continue
    fi
    CUDA_VISIBLE_DEVICES=$GPU python scripts/precompute_latents.py \
        --direction a2s --target speech --split "$SPLIT" \
        > "$LOG_DIR/02_latents_$SPLIT.log" 2>&1 &
    PIDS+=($!); GPU=$((GPU+1))
done
LAT_FAIL=0
for p in ${PIDS[@]+"${PIDS[@]}"}; do wait "$p" || LAT_FAIL=1; done
tail -n 4 "$LOG_DIR"/02_latents_*.log 2>/dev/null || true
[[ "$LAT_FAIL" == "0" ]] || { echo "ERROR: a latent precompute lane failed" >&2; exit 1; }

# ========================================= 3. CLAP conditioning + align targets
echo
echo "### [3/3] CLAP condition frames + both O2 alignment candidates ###"
PIDS=(); GPU=0
for SPLIT in train val test; do
    if [[ -f "$COND_ROOT/$SPLIT/cond_seq.npy" ]]; then
        echo "  $SPLIT: already present, skipping"
        continue
    fi
    CUDA_VISIBLE_DEVICES=$GPU python scripts/a2s/precompute_a2s_conditioning.py \
        --split "$SPLIT" --out-dir "$COND_ROOT/$SPLIT" \
        ${SKIP_ALIGN_FLAG[@]+"${SKIP_ALIGN_FLAG[@]}"} \
        > "$LOG_DIR/03_cond_$SPLIT.log" 2>&1 &
    PIDS+=($!); GPU=$((GPU+1))
done
COND_FAIL=0
for p in ${PIDS[@]+"${PIDS[@]}"}; do wait "$p" || COND_FAIL=1; done
tail -n 14 "$LOG_DIR"/03_cond_train.log 2>/dev/null || true
[[ "$COND_FAIL" == "0" ]] || { echo "ERROR: a conditioning precompute lane failed" >&2; exit 1; }

# =================================================================== summary
echo
echo "==================== A2S STAGE 1 SUMMARY ===================="
python - <<PY
import json, pathlib

g = json.loads(pathlib.Path("$GATE_DIR/report.json").read_text())
print(f"GATE: {g['verdict']}")
print(f"  WER floor (untouched)  : {g['wer']['original']['wer']:.4f}")
print(f"  WER mel+vocoder        : {g['wer']['mel']['wer']:.4f} "
      f"(+{g['wer_degradation_mel']:.4f})")
print(f"  WER VAE+vocoder        : {g['wer']['vae']['wer']:.4f} "
      f"(+{g['wer_degradation_vae']:.4f})")
print(f"  {g['verdict_detail']}")
d = g["duration_scan"]
print(f"  target speech: mean {d['mean_s']:.2f}s, {d['frac_over_canvas']*100:.2f}% over the "
      f"{d['canvas_s']}s canvas, {d['mean_canvas_occupancy']*100:.0f}% mean occupancy")

print()
for split in ("train", "val", "test"):
    p = pathlib.Path("$COND_ROOT") / split / "meta.json"
    if not p.exists():
        continue
    m = json.loads(p.read_text())
    print(f"{split}: {m['rows']} rows, cond {m['cond_frames']}x{m['cond_feature_dim']} "
          f"({m['cond_seconds_per_frame']:.3f}s/frame)")

p = pathlib.Path("$COND_ROOT/train/meta.json")
if p.exists():
    m = json.loads(p.read_text())
    print()
    print("WHICH O2 ALIGNMENT TARGET? (S2A ambient reference: 0.291; near 1.0 = useless)")
    print(f"  align_text   (CLAP-text of transcript)  "
          f"{m['align_text']['pairwise_cosine_mean']:.4f}")
    if "align_speech" in m:
        print(f"  align_speech (CLAP-audio of speech)     "
              f"{m['align_speech']['pairwise_cosine_mean']:.4f}")
PY
echo "============================================================="
echo "gate dir:  $GATE_DIR   (listen/<sample_id>/*.wav)"
echo "latents:   $FEATURES_ROOT/a2s/{train,val,test}"
echo "condition: $COND_ROOT/{train,val,test}"
echo

if [[ "$VERDICT" == "FAIL" ]]; then
    echo "GATE FAILED -- do NOT start A2S training on this decoder." >&2
    echo "Exiting non-zero so any afterok dependency stays held." >&2
    exit 1
fi
echo "OK ($VERDICT)"
