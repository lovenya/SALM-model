#!/usr/bin/env bash
# Oracle transcript -> categorical CTC-code bridge -> proven content_concat_v4 renderer.
#
# This is an F2 GATE, not the final ambient->speech system. It never reads ambient audio.
# It tests whether a scene planner that produces words can hand those words to our existing
# AudioLDM speech renderer without retrieval, Qwen Talker, or a second TTS stack.
#
# PRE-REGISTERED VERDICT
#   SUCCESS:
#     (1) held-out val_text_gap > 0 and pair_win_rate well above 0.5;
#     (2) oracle_text audio is closer by WER to its supplied text than to a deranged text;
#     (3) oracle_text WER approaches the target-speech oracle_codes ceiling and the user can
#         hear the supplied words. shuffled_text must follow the shuffled supplied words,
#         not the validation row's original transcript.
#   FAILURE:
#     paired text loss is indistinguishable from rolled text, OR generated audio does not
#     bind to supplied text despite the oracle_codes lane remaining intelligible. That means
#     this frame-code bridge cannot translate lexical plans into renderer content.
#   AMBIGUOUS:
#     strong teacher-forced text binding but weak free-running/audio binding. That diagnoses
#     exposure bias or the fixed 256-frame duration entanglement; next test is run-length
#     (code,duration) prediction, not an ambient-scene change.
#
# Resource basis: the existing single-GPU ContentLM measured ~37 min/100 epochs. This job
# uses four-rank DDP (same ~25M decoder) after one frozen-T5 cache pass, then packs independent
# checkpoint/lane generations four at a time across all four H100s. 80 min internal deadline
# + 10 min completion buffer inside the 1:30 allocation. No job is submitted by this file's
# creation; run only after committing a clean checkout.
#SBATCH --job-name=salm-a2s-text-bridge
#SBATCH --account=aip-csubakan
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
: "${SALM_REPO:?ERROR: submit with --export=ALL,SALM_REPO=/absolute/clean/checkout}"
REPO="$SALM_REPO"
[[ -d "$REPO" ]] || { echo "ERROR: SALM_REPO is not a directory: $REPO" >&2; exit 1; }
source "$REPO/submission_scripts/_salm_env.sh"

STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"
RUN_NAME="${RUN_NAME:-text_bridge_k512_${STAMP}}"
FEATURE_DIR="${FEATURE_DIR:-$OUTPUTS_ROOT/text_content_features}"
CODEBOOK="${CODEBOOK:-$OUTPUTS_ROOT/content_codebook/codebook_k512.pt}"
RENDERER="${RENDERER:-$RUNS_ROOT/a2s_audioldm_clapseq_xformer/content_concat_v4/checkpoints/checkpoint_best.pt}"
TEXT_MODEL="${TEXT_MODEL:-/scratch/l/lovenya/salm_migration_clean/models/hf_cache/models--google--flan-t5-large/snapshots/0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a}"
SALVAGE_TRAIN_DIR="${SALVAGE_TRAIN_DIR:-}"
if [[ -n "$SALVAGE_TRAIN_DIR" ]]; then
    TRAIN_DIR="$SALVAGE_TRAIN_DIR"
else
    TRAIN_DIR="$OUTPUTS_ROOT/text_content_lm/$RUN_NAME"
fi
GEN_ROOT="$OUTPUTS_ROOT/generations/${RUN_NAME}"
NUM_GEN="${NUM_GEN:-16}"

[[ -z "$(git status --porcelain)" ]] || {
    echo "ERROR: full gate requires a clean committed checkout" >&2; exit 1;
}
for P in "$CODEBOOK" "$RENDERER" "$TEXT_MODEL" \
         "$FEATURES_ROOT/a2s/train/index.jsonl" "$FEATURES_ROOT/a2s/val/index.jsonl" \
         "$OUTPUTS_ROOT/content_codes/codes_codebook_k512_train.npy" \
         "$OUTPUTS_ROOT/content_codes/codes_codebook_k512_val.npy"; do
    [[ -e "$P" ]] || { echo "MISSING: $P" >&2; exit 1; }
done
if [[ -n "$SALVAGE_TRAIN_DIR" ]]; then
    [[ -d "$TRAIN_DIR" ]] || { echo "MISSING salvage run: $TRAIN_DIR" >&2; exit 1; }
else
    [[ ! -e "$TRAIN_DIR" ]] || { echo "REFUSING existing run: $TRAIN_DIR" >&2; exit 1; }
fi
[[ ! -e "$GEN_ROOT" ]] || { echo "REFUSING existing generation: $GEN_ROOT" >&2; exit 1; }

salm_print_header "Oracle text -> K=512 content codes -> frozen AudioLDM renderer"
salm_start_monitor "$OUTPUTS_ROOT/logs/resources/$RUN_NAME" "$RUN_NAME" 30
mkdir -p "$GEN_ROOT"

if [[ -z "$SALVAGE_TRAIN_DIR" ]]; then
# Cache token-level frozen Flan-T5 states once. These scripts expose no ambient argument and
# read only speech_text from the A2S index.
for SPLIT in train val; do
    CUDA_VISIBLE_DEVICES=0 python scripts/a2s/precompute_text_content_features.py \
        --split "$SPLIT" --model "$TEXT_MODEL" --out-root "$FEATURE_DIR" \
        --max-tokens 64 --batch-size 64 --device cuda
done

# Four H100s work on one DDP model; 45 min is the bounded training budget. The remaining
# allocation is reserved for the full checkpoint spread, Whisper scoring, and output flush.
timeout --signal=TERM --kill-after=120s 50m torchrun --standalone --nproc_per_node=4 \
    scripts/a2s/train_text_content_lm.py \
    --codebook "$CODEBOOK" --features "$FEATURE_DIR" --run-name "$RUN_NAME" \
    --epochs 100 --batch-size 64 --eval-every 5 --save-every 20 --max-minutes 45

TAGS=(checkpoint_best_loss checkpoint_ep19 checkpoint_ep39 checkpoint_ep59 \
      checkpoint_ep79 checkpoint_ep99 checkpoint_best_binding)
[[ -f "$TRAIN_DIR/checkpoints/checkpoint_ep99.pt" ]] || {
    echo "ERROR: training did not complete the registered 100 epochs" >&2; exit 1;
}
else
TAGS=(checkpoint_ep19 checkpoint_ep39)
for TAG in "${TAGS[@]}"; do
    [[ -f "$TRAIN_DIR/checkpoints/${TAG}.pt" ]] || {
        echo "MISSING salvage checkpoint: $TRAIN_DIR/checkpoints/${TAG}.pt" >&2; exit 1;
    }
done
fi
LANES=(oracle_text shuffled_text oracle_codes)

# Pack independent (checkpoint,lane) renderer calls four at a time on explicit devices.
pids=()
gpu=0
for TAG in "${TAGS[@]}"; do
    CKPT="$TRAIN_DIR/checkpoints/${TAG}.pt"
    [[ -f "$CKPT" ]] || { echo "MISSING checkpoint in full spread: $CKPT" >&2; exit 1; }
    for LANE in "${LANES[@]}"; do
        OUT="$GEN_ROOT/${TAG#checkpoint_}__${LANE}"
        CUDA_VISIBLE_DEVICES="$gpu" python scripts/a2s/generate_text_content_bridge.py \
            --bridge-checkpoint "$CKPT" --renderer-checkpoint "$RENDERER" \
            --features "$FEATURE_DIR" --lane "$LANE" --split val \
            --num-samples "$NUM_GEN" --temperature 0.8 --top-k 32 \
            --out-dir "$OUT" --device cuda &
        pids+=("$!")
        gpu=$((gpu + 1))
        if (( gpu == 4 )); then
            for pid in "${pids[@]}"; do wait "$pid"; done
            pids=()
            gpu=0
        fi
    done
done
for pid in "${pids[@]}"; do wait "$pid"; done

mapfile -t SCORE_DIRS < <(find "$GEN_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)
CUDA_VISIBLE_DEVICES=0 python scripts/a2s/score_text_content_bridge.py \
    --dirs "${SCORE_DIRS[@]}" --out "$GEN_ROOT/wer_and_binding.json" \
    --batch-size 8 --device cuda

# Lightweight artifact sanity: every registered directory must have exactly NUM_GEN generated
# WAVs (reference copies are counted separately) and a manifest with the same row count.
python - "$GEN_ROOT" "$NUM_GEN" <<'PY'
import json, pathlib, sys
root, expected = pathlib.Path(sys.argv[1]), int(sys.argv[2])
reports = []
for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / "manifest.jsonl").exists()):
    rows = [json.loads(x) for x in (d / "manifest.jsonl").read_text().splitlines()]
    wavs = [p for p in (d / "audio").glob("*.wav") if "__reference" not in p.name]
    if len(rows) != expected or len(wavs) != expected:
        raise SystemExit(f"{d}: rows={len(rows)} wavs={len(wavs)} expected={expected}")
    reports.append({"directory": str(d), "rows": len(rows), "generated_wavs": len(wavs),
                    "rms_min": min(r["generated_rms"] for r in rows),
                    "rms_max": max(r["generated_rms"] for r in rows)})
(root / "artifact_report.json").write_text(json.dumps(reports, indent=2) + "\n")
print(f"validated {len(reports)} generation lanes")
PY

echo "OK: $TRAIN_DIR"
echo "LISTEN/SCORES: $GEN_ROOT"
