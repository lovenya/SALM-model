#!/usr/bin/env bash
# Non-Qwen, non-retrieval TTAS-supervised scene -> utterance planner ablation.
#
# All four GPUs are used independently because each Flan-T5-large lane fits on one H100.
# The source audio is the existing cached CLAP *sequence*, not its pooled vector.
#
# ======================= EXPECTED RESULTS (pre-registered) ========================
# Primary raw-audio success requires ALL of:
#   1. val_speech_scene_gap > 0 durably: the exact speech targets become less likely when
#      paired with a different real scene, proving the planner uses scene identity;
#   2. shuffled_output_change_rate materially > 0: dependence survives free generation;
#   3. train_exact_copy_rate low and unique_output_rate high: this is generation, not hidden
#      retrieval/memorisation;
#   4. heldout_generations.jsonl is specific to audio_cap by human inspection, not merely
#      grammatical. Perplexity alone is explicitly not success.
#
# Lane meanings:
#   oracle_caption_speech -- upper control. If this cannot produce specific held-out speech,
#     Flan-T5/recipe is wrong for TTAS and raw-audio failures are uninterpretable.
#   audio_speech -- direct baseline. Success is the shortest route to the renderer.
#   audio_structured -- explicit caption->reasoning->speech chain. If its speech is more
#     specific than audio_speech, intermediate scene semantics are the missing bottleneck.
#   audio_multitask -- shared memory trained on each field separately. If this wins, keep a
#     latent scene representation but do not require verbose chain text at inference.
#
# Failure outcomes:
#   oracle works, all audio lanes have ~0 speech gap/generic text -> cached CLAP sequence is
#     inadequate for fine scene semantics; replace the ambient encoder, not the renderer.
#   scene gap positive but generations copy training rows -> conditional memorisation; add
#     stronger regularisation/more semantic pretraining, do not call it scene understanding.
#   structured/multitask beat direct -> retain explicit auxiliary semantic supervision.
#   all lanes including oracle fail -> training/API/target formulation issue; no A2S claim.
#
# Ambiguous:
#   lower val loss but near-zero gap, or changed outputs that remain generic. This means a
#   better language prior, not better grounding, and must not advance to speech rendering.
# ================================================================================
#SBATCH --job-name=salm-a2s-scene-text
#SBATCH --account=aip-ravanelm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
: "${SALM_REPO:?ERROR: submit with --export=ALL,SALM_REPO=/absolute/clean/checkout}"
REPO="$SALM_REPO"
[[ -d "$REPO" ]] || { echo "ERROR: SALM_REPO is not a directory: $REPO" >&2; exit 1; }
source "$REPO/submission_scripts/_salm_env.sh"

MODEL=/scratch/l/lovenya/salm_migration_clean/models/hf_cache/models--google--flan-t5-large/snapshots/0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a
EPOCHS="${EPOCHS:-8}"
MAX_MINUTES="${MAX_MINUTES:-105}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
STAMP="${STAMP:-$(TZ=America/Toronto date +%Y%m%d_%H%M)}"
RUN_GROUP="${RUN_GROUP:-scene_text_${STAMP}}"
export RUN_GROUP
RUN_ROOT="$OUTPUTS_ROOT/scene_text_planner/$RUN_GROUP"

for required in "$MODEL/config.json" "$MODEL/model.safetensors" \
    "$FEATURES_ROOT/a2s/train/index.jsonl" \
    "$OUTPUTS_ROOT/features_a2s_cond/train/cond_seq.npy" \
    "$OUTPUTS_ROOT/features_a2s_cond/val/cond_seq.npy"; do
  [[ -e "$required" ]] || { echo "MISSING: $required" >&2; exit 1; }
done

[[ ! -e "$RUN_ROOT" ]] || { echo "REFUSING existing run group: $RUN_ROOT" >&2; exit 1; }
mkdir -p "$RUN_ROOT"
git -C "$REPO" rev-parse HEAD > "$RUN_ROOT/job_start_commit.txt"
git -C "$REPO" status --porcelain > "$RUN_ROOT/job_start_dirty.txt"
pwd > "$RUN_ROOT/job_start_repo_path.txt"
salm_print_header "TTAS-supervised Flan-T5 scene-to-utterance planner (4 independent lanes)"
salm_start_monitor "$RUN_ROOT/resources" scene_text 30

run_lane() {
  local gpu="$1" name="$2" source="$3" target="$4"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/a2s/train_scene_text_planner.py \
    --run-name "$RUN_GROUP/$name" \
    --model-path "$MODEL" \
    --source "$source" \
    --target-mode "$target" \
    --train-scope decoder \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --save-every 0 \
    --max-minutes "$MAX_MINUTES" \
    --generate-count 64 \
    2>&1 | tee "$RUN_ROOT/${name}.log"
}

pids=()
run_lane 0 oracle_caption_speech caption speech & pids+=("$!")
run_lane 1 audio_speech audio speech & pids+=("$!")
run_lane 2 audio_structured audio structured & pids+=("$!")
run_lane 3 audio_multitask audio multitask & pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

echo "==================== PLANNER SUMMARIES ===================="
python - "$RUN_ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for name in ("oracle_caption_speech", "audio_speech", "audio_structured", "audio_multitask"):
    p = root / name / "summary.json"
    if not p.exists():
        print(f"{name}: MISSING summary")
        continue
    x = json.loads(p.read_text())
    b, g = x["best_val"], x["generation"]
    print(f"{name:27s} loss={b['val_loss']:.3f} gap={b['val_scene_gap']:+.3f} "
          f"speech_gap={b['val_speech_scene_gap']:+.3f} changed={g['shuffled_output_change_rate']:.2f} "
          f"copies={g['train_exact_copy_rate']:.2f} unique={g['unique_output_rate']:.2f} "
          f"target_f1={g['mean_target_token_f1']:.2f}")
PY

if (( failed )); then
  echo "one or more planner lanes failed; inspect the per-lane logs" >&2
  exit 1
fi
echo "OK"
