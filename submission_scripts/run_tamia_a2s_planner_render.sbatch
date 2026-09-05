#!/usr/bin/env bash
# Honest composition of the completed raw-audio planner with the proven speech renderer.
# No oracle caption, reasoning, transcript, target code, or retrieval result is conditioning.
# The cached planner JSONL is an intermediate produced from ambient audio alone.
#
# EXPECTED RESULTS (pre-registered)
# Success:
#   Whisper is closer to each supplied planner utterance than to mismatched utterances,
#   and human listening confirms intelligible speech whose content matches that utterance.
# Failure:
#   audio is unintelligible or ignores the supplied planner text despite the oracle text
#   bridge working; this identifies a planner-text distribution or sampling failure.
# Ambiguous (and likely):
#   speech faithfully renders the supplied text but remains bland/generic. Then the honest
#   end-to-end plumbing works and scene planning—not speech generation—is the bottleneck.
# The shuffled-audio planner lanes must render different content; they are a binding control.
#SBATCH --job-name=salm-a2s-plan-render
#SBATCH --account=aip-ravanelm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail
: "${SALM_REPO:?submit with --export=ALL,SALM_REPO=/absolute/clean/checkout}"
REPO="$SALM_REPO"
[[ -d "$REPO" ]] || { echo "missing repository: $REPO" >&2; exit 1; }
source "$REPO/submission_scripts/_salm_env.sh"
cd "$REPO"

RUNS_ROOT="$OUTPUTS_ROOT/runs"
PLANNER_MANIFEST="${PLANNER_MANIFEST:-$OUTPUTS_ROOT/scene_text_planner/scene_text_quota_safe_20260905_0700/audio_speech/heldout_generations.jsonl}"
BRIDGE="${BRIDGE:-$OUTPUTS_ROOT/text_content_lm/text_bridge_k512_20260904_0311/checkpoints/checkpoint_ep39.pt}"
RENDERER="${RENDERER:-$RUNS_ROOT/a2s_audioldm_clapseq_xformer/content_concat_v4/checkpoints/checkpoint_best.pt}"
TEXT_FEATURES="${TEXT_FEATURES:-$OUTPUTS_ROOT/text_content_features}"
STAMP="${STAMP:-$(TZ=America/Toronto date +%Y%m%d_%H%M)}"
RUN_NAME="${RUN_NAME:-planner_render_${STAMP}}"
OUT="$OUTPUTS_ROOT/generations/$RUN_NAME"

for required in "$PLANNER_MANIFEST" "$BRIDGE" "$RENDERER" \
  "$TEXT_FEATURES/val_hidden.npy" "$TEXT_FEATURES/val_mask.npy"; do
  [[ -e "$required" ]] || { echo "MISSING: $required" >&2; exit 1; }
done
[[ ! -e "$OUT" ]] || { echo "REFUSING existing output: $OUT" >&2; exit 1; }
mkdir -p "$OUT"
git -C "$REPO" rev-parse HEAD > "$OUT/job_start_commit.txt"
git -C "$REPO" status --porcelain > "$OUT/job_start_dirty.txt"
salm_print_header "Raw-audio planner -> text bridge -> AudioLDM speech renderer"
salm_start_monitor "$OUT/resources" planner_render 15

render() {
  local gpu="$1" dir="$2" lane="$3" mode="$4"
  local greedy=()
  [[ "$mode" == greedy ]] && greedy=(--greedy)
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/a2s/generate_text_content_bridge.py \
    --bridge-checkpoint "$BRIDGE" --renderer-checkpoint "$RENDERER" \
    --features "$TEXT_FEATURES" --planner-manifest "$PLANNER_MANIFEST" \
    --lane "$lane" --split val --num-samples 64 --batch-size 4 \
    --steps 100 --temperature 0.8 --top-k 32 --seed 1337 \
    --out-dir "$OUT/$dir" "${greedy[@]}" \
    2>&1 | tee "$OUT/${dir}.log"
}

pids=()
render 0 planner_sampled planner_text sampled & pids+=("$!")
render 1 shuffled_sampled planner_shuffled sampled & pids+=("$!")
render 2 planner_greedy planner_text greedy & pids+=("$!")
render 3 shuffled_greedy planner_shuffled greedy & pids+=("$!")
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || { echo "one or more renderer lanes failed" >&2; exit 1; }

for dir in planner_sampled shuffled_sampled planner_greedy shuffled_greedy; do
  [[ $(wc -l < "$OUT/$dir/manifest.jsonl") -eq 64 ]] || {
    echo "$dir has an incomplete manifest" >&2; exit 1;
  }
done
CUDA_VISIBLE_DEVICES=0 python scripts/a2s/score_text_content_bridge.py \
  --dirs "$OUT/planner_sampled" "$OUT/shuffled_sampled" \
         "$OUT/planner_greedy" "$OUT/shuffled_greedy" \
  --out "$OUT/asr_binding.json" --batch-size 8

echo "Artifacts: $OUT"
echo OK
