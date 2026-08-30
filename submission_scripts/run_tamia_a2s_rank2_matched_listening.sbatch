#!/usr/bin/env bash
# Corrected Rank-2 falsification: matched oracle/retrieval/shuffled/zero listening outputs.
#
# This file PREPARES the experiment; it must not be submitted without the user's explicit
# instruction. The four content sources run concurrently, one per H100. Every source uses
# identical validation rows, seeds, sampler settings, and unique saved checkpoints.
#
# Safe first run:
#   EVAL_KIND=smoke sbatch submission_scripts/run_tamia_a2s_rank2_matched_listening.sbatch
# Decisive run after listening to smoke:
#   EVAL_KIND=full sbatch submission_scripts/run_tamia_a2s_rank2_matched_listening.sbatch
#
# Measured basis: prior Rank-2 generation was ~14 s / 16 samples and Rank-2 training peaked
# near 12.8 GB/GPU. The 30-minute request is deliberately >1.4x the expected generation and
# model-load time, and the internal 20-minute deadline preserves a 10-minute cleanup buffer.
#SBATCH --job-name=salm-a2s-r2listen
#SBATCH --account=aip-csubakan
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=240G
#SBATCH --gpus=h100:4
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

RUN_NAME="${RUN_NAME:-content_xattn_falsify_100ep}"
CKPT_DIR="${CKPT_DIR:-$RUNS_ROOT/a2s_audioldm_clapseq_xformer/$RUN_NAME/checkpoints}"
CTC_CKPT="${CTC_CKPT:-$OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt}"
COND_ROOT="${COND_ROOT:-$OUTPUTS_ROOT/features_a2s_cond}"
EVAL_KIND="${EVAL_KIND:-smoke}"
STEPS="${STEPS:-100}"
SEED="${SEED:-1337}"

case "$EVAL_KIND" in
    smoke) NUM_SAMPLES="${NUM_SAMPLES:-4}"; CKPT_NAMES=(checkpoint_best.pt) ;;
    full)  NUM_SAMPLES="${NUM_SAMPLES:-16}"; CKPT_NAMES=(
        checkpoint_best.pt checkpoint_ep19.pt checkpoint_ep39.pt
        checkpoint_ep59.pt checkpoint_ep79.pt checkpoint_ep99.pt
    ) ;;
    *) echo "ERROR: EVAL_KIND must be smoke or full, got $EVAL_KIND" >&2; exit 2 ;;
esac
export RUN_NAME CKPT_DIR CTC_CKPT COND_ROOT EVAL_KIND NUM_SAMPLES STEPS SEED

if [[ "${1:-}" == "--lane" ]]; then
    SOURCE="$2"
    LANE_ROOT="$3"
    for CKPT_NAME in "${CKPT_NAMES[@]}"; do
        CKPT="$CKPT_DIR/$CKPT_NAME"
        TAG="${CKPT_NAME#checkpoint_}"
        TAG="${TAG%.pt}"
        DEST="$LANE_ROOT/$SOURCE/$TAG"
        [[ ! -e "$DEST" ]] || { echo "ERROR: output exists: $DEST" >&2; exit 3; }
        python -m salm.generate \
            --mode a2s --direction a2s --checkpoint "$CKPT" --split val \
            --num-samples "$NUM_SAMPLES" --batch-size 4 --steps "$STEPS" --seed "$SEED" \
            --content-source "$SOURCE" --ctc-checkpoint "$CTC_CKPT" \
            --content-retrieval-root "$COND_ROOT" --out-dir "$DEST"
    done
    exit 0
fi

for REQUIRED in "$CTC_CKPT" "$COND_ROOT/train/cond_seq.npy" "$COND_ROOT/val/cond_seq.npy"; do
    [[ -f "$REQUIRED" ]] || { echo "MISSING: $REQUIRED" >&2; exit 4; }
done
for CKPT_NAME in "${CKPT_NAMES[@]}"; do
    [[ -f "$CKPT_DIR/$CKPT_NAME" ]] || { echo "MISSING: $CKPT_DIR/$CKPT_NAME" >&2; exit 4; }
done

RUN_TAG="${RUN_TAG:-rank2_matched_${EVAL_KIND}_${SLURM_JOB_ID:-manual}}"
GEN_ROOT="${GEN_ROOT:-$CKPT_DIR/../generations/$RUN_TAG}"
[[ ! -e "$GEN_ROOT" ]] || { echo "ERROR: refusing existing generation root: $GEN_ROOT" >&2; exit 5; }
mkdir -p "$GEN_ROOT/logs"

cat > "$GEN_ROOT/run_metadata.json" <<EOF
{
  "schema_version": 1,
  "experiment": "a2s_rank2_matched_content_listening",
  "eval_kind": "$EVAL_KIND",
  "content_sources": ["oracle", "retrieval", "shuffled", "zero"],
  "num_samples": $NUM_SAMPLES,
  "steps": $STEPS,
  "seed": $SEED,
  "checkpoint_dir": "$CKPT_DIR",
  "ctc_checkpoint": "$CTC_CKPT",
  "slurm_job_id": "${SLURM_JOB_ID:-none}",
  "account": "${SLURM_JOB_ACCOUNT:-unknown}",
  "hostname": "$(hostname)",
  "git_commit": "$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)",
  "git_dirty": $(if [[ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]]; then echo true; else echo false; fi),
  "started_at": "$(TZ=America/Toronto date -Iseconds)"
}
EOF

salm_print_header "A2S Rank-2 matched listening: $EVAL_KIND"
salm_start_monitor "$GEN_ROOT/logs" "$RUN_TAG" 15

# Hard gate: generation is invalid unless the live pretrained CLAP tower reproduces the
# fp16 training cache and retrieves each selected row back from that cache at rank 1.
CUDA_VISIBLE_DEVICES=0 python scripts/a2s/check_clap_live_cache_parity.py \
    --split val --num-samples "$NUM_SAMPLES" --batch-size 4 \
    --cache-root "$COND_ROOT" --report "$GEN_ROOT/clap_live_cache_parity.json"

SOURCES=(oracle retrieval shuffled zero)
PIDS=()
for GPU in 0 1 2 3; do
    SOURCE="${SOURCES[$GPU]}"
    timeout --signal=TERM --kill-after=60s 20m \
        env CUDA_VISIBLE_DEVICES="$GPU" bash "$0" --lane "$SOURCE" "$GEN_ROOT" \
        > "$GEN_ROOT/logs/${SOURCE}.log" 2>&1 &
    PIDS+=("$!")
done

set +e
FAILED=0
for I in 0 1 2 3; do
    wait "${PIDS[$I]}"
    STATUS=$?
    echo "lane ${SOURCES[$I]} exit=$STATUS"
    [[ "$STATUS" -eq 0 ]] || FAILED=1
done
set -e
[[ "$FAILED" -eq 0 ]] || { echo "FAILED: one or more generation lanes" >&2; exit 6; }

for SOURCE in "${SOURCES[@]}"; do
    for CKPT_NAME in "${CKPT_NAMES[@]}"; do
        TAG="${CKPT_NAME#checkpoint_}"; TAG="${TAG%.pt}"
        MANIFEST="$GEN_ROOT/$SOURCE/$TAG/manifest.jsonl"
        [[ -s "$MANIFEST" ]] || { echo "MISSING/EMPTY: $MANIFEST" >&2; exit 7; }
        COUNT="$(wc -l < "$MANIFEST")"
        [[ "$COUNT" -eq "$NUM_SAMPLES" ]] || {
            echo "ERROR: $MANIFEST has $COUNT rows, expected $NUM_SAMPLES" >&2; exit 7;
        }
    done
done

touch "$GEN_ROOT/SUCCESS"
echo "SUCCESS: matched listening outputs validated at $GEN_ROOT"
