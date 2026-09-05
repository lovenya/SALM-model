#!/usr/bin/env bash
# Compare grounded planner training surfaces without changing the language model unnecessarily.
#
# EXPECTED RESULTS (pre-registered)
# Success:
#   - audio_adapter_speech or audio_crossattn_speech produces less generic, scene-specific
#     utterances than the completed full-decoder baseline, while retaining a positive
#     shuffled-scene loss gap and low exact-copy rate;
#   - caption lanes name audible events correctly, change under shuffled audio, and do not
#     copy training captions. Caption target F1 is measured against captions, not speech.
# Failure:
#   - all four lanes remain generic or lose their scene gap. Then frozen CLAP sequence is
#     the limiting representation; increasing decoder freedom is not the answer.
# Ambiguous:
#   - caption grounding improves but direct speech stays generic. This validates a modular
#     audio -> caption -> response planner, but is not yet an end-to-end result.
#SBATCH --job-name=salm-a2s-plan-compare
#SBATCH --account=aip-ravanelm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail
: "${SALM_REPO:?submit with --export=ALL,SALM_REPO=/absolute/clean/checkout}"
REPO="$SALM_REPO"
[[ -d "$REPO" ]] || { echo "missing repository: $REPO" >&2; exit 1; }
source "$REPO/submission_scripts/_salm_env.sh"
cd "$REPO"

MODEL=/scratch/l/lovenya/salm_migration_clean/models/hf_cache/models--google--flan-t5-large/snapshots/0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a
EPOCHS="${EPOCHS:-8}"
MAX_MINUTES="${MAX_MINUTES:-50}"
STAMP="${STAMP:-$(TZ=America/Toronto date +%Y%m%d_%H%M)}"
RUN_GROUP="${RUN_GROUP:-planner_compare_${STAMP}}"
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
salm_print_header "Grounded planner comparison: adapter and cross-attention"
salm_start_monitor "$RUN_ROOT/resources" planner_compare 30

run_lane() {
  local gpu="$1" name="$2" target="$3" scope="$4"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/a2s/train_scene_text_planner.py \
    --run-name "$RUN_GROUP/$name" --model-path "$MODEL" \
    --source audio --target-mode "$target" --train-scope "$scope" \
    --epochs "$EPOCHS" --batch-size 8 --grad-accum 4 \
    --save-every 0 --max-minutes "$MAX_MINUTES" --generate-count 64 \
    2>&1 | tee "$RUN_ROOT/${name}.log"
}

pids=()
run_lane 0 audio_adapter_speech speech adapter & pids+=("$!")
run_lane 1 audio_crossattn_speech speech cross_attention & pids+=("$!")
run_lane 2 audio_adapter_caption caption adapter & pids+=("$!")
run_lane 3 audio_crossattn_caption caption cross_attention & pids+=("$!")
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done

python - "$RUN_ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for name in ("audio_adapter_speech", "audio_crossattn_speech", "audio_adapter_caption", "audio_crossattn_caption"):
    path = root / name / "summary.json"
    if not path.exists():
        print(f"{name}: MISSING")
        continue
    x = json.loads(path.read_text()); b=x["best_val"]; g=x["generation"]
    print(f"{name:26s} loss={b['val_loss']:.3f} gap={b['val_scene_gap']:+.3f} "
          f"changed={g['shuffled_output_change_rate']:.2f} copies={g['train_exact_copy_rate']:.2f} "
          f"unique={g['unique_output_rate']:.2f} target_f1={g['mean_target_token_f1']:.2f}")
PY
(( failed == 0 )) || { echo "one or more lanes failed" >&2; exit 1; }
echo OK
