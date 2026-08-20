#!/usr/bin/env bash
# Answer the two questions the loss cannot: is the speech used, and does generation beat lookup?
#
# Four lanes, one per GPU:
#   D1/D2 on the LAST checkpoint  -- is the conditioning used after 91 epochs?
#   D1/D2 on the EARLIEST snapshot -- was it ever used? A rising D1 across training would be the
#                                     first real evidence of learning we have.
#   retrieval in Whisper space     -- the honest "could a lookup replace the conditioner?" test,
#                                     since it retrieves in the same space the conditioner consumes
#   retrieval in CLAP space        -- also tests, for free, whether CLAP separates these utterances
#                                     at all (the rung1 premise we have asserted without evidence)
#
# Retrieval embeds the whole 15,033-row training pool, which is the slow part (~10 min).
#
# Submit:
#   sbatch submission_scripts/run_tamia_s2a_diagnostics.sbatch
#   NUM_GEN=64 sbatch ...        # more retrieval queries
#   MAX_BATCHES=20 sbatch ...    # cap the diagnostics for a quick look
#SBATCH --job-name=salm-s2a-diagnostics
#SBATCH --account=aip-ravanelm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

ARCH="${ARCH:-s2a_audioldm_whisper_xformer}"
RUN_NAME="${RUN_NAME:-long100}"
NUM_GEN="${NUM_GEN:-16}"
MAX_BATCHES="${MAX_BATCHES:-0}"
STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"

RUN_DIR="$RUNS_ROOT/$ARCH/$RUN_NAME"
CKPT_DIR="$RUN_DIR/checkpoints"
OUT="$RUN_DIR/diagnostics_${STAMP}"
LOG_DIR="$RUN_DIR/logs/diagnostics_${STAMP}"
mkdir -p "$OUT" "$LOG_DIR"

LAST="$CKPT_DIR/checkpoint_last.pt"
FIRST="$(ls -1 "$CKPT_DIR"/checkpoint_ep*.pt 2>/dev/null | sort -V | head -1 || true)"
[[ -f "$LAST" ]] || { echo "ERROR: no $LAST" >&2; exit 1; }

echo "==================================================================="
echo "S2A diagnostics + retrieval baseline"
echo "job:    ${SLURM_JOB_ID:-none}   host: $(hostname)"
echo "commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "run:    $ARCH/$RUN_NAME"
echo "last:   $LAST"
echo "first:  ${FIRST:-none}"
echo "==================================================================="

T0=$(date +%s)

# GPU 0 -- D1/D2 at the end of training
( CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_conditioning.py \
    --checkpoint "$LAST" --out-dir "$OUT/d1d2_last" --max-batches "$MAX_BATCHES" \
    > "$LOG_DIR/d1d2_last.log" 2>&1 ) & P0=$!

# GPU 1 -- D1/D2 at the earliest snapshot, for a start-vs-end comparison
if [[ -n "$FIRST" ]]; then
    ( CUDA_VISIBLE_DEVICES=1 python scripts/diagnose_conditioning.py \
        --checkpoint "$FIRST" --out-dir "$OUT/d1d2_first" --max-batches "$MAX_BATCHES" \
        > "$LOG_DIR/d1d2_first.log" 2>&1 ) & P1=$!
else
    P1=""
fi

# GPU 2/3 -- retrieval in each embedding space
( CUDA_VISIBLE_DEVICES=2 python scripts/retrieval_baseline.py \
    --space whisper --split "${SPLIT:-val}" --num-samples "$NUM_GEN" \
    --out-dir "$OUT/retrieval_whisper" \
    > "$LOG_DIR/retrieval_whisper.log" 2>&1 ) & P2=$!
( CUDA_VISIBLE_DEVICES=3 python scripts/retrieval_baseline.py \
    --space clap --split "${SPLIT:-val}" --num-samples "$NUM_GEN" \
    --out-dir "$OUT/retrieval_clap" \
    > "$LOG_DIR/retrieval_clap.log" 2>&1 ) & P3=$!

FAIL=0
for p in $P0 $P1 $P2 $P3; do wait "$p" || FAIL=1; done
TOTAL=$(( $(date +%s) - T0 ))

echo
echo "### D1/D2 -- is the conditioning used? ###"
for d in d1d2_last d1d2_first; do
    [[ -f "$OUT/$d/diagnostics.json" ]] || { echo "  $d: MISSING (see $LOG_DIR/$d.log)"; continue; }
    echo "--- $d"
    sed -n '/^VERDICT/,$p' "$LOG_DIR/${d}.log" 2>/dev/null || true
    grep -E "^(mean D1|mean D2|low-t D1|low-t D2|val CLAP|  To reach|  current D2)" "$LOG_DIR/${d}.log" 2>/dev/null || true
done

echo
echo "### retrieval baseline ###"
python - "$OUT" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
for space in ("whisper", "clap"):
    p = out / f"retrieval_{space}" / "summary.json"
    if not p.exists():
        print(f"  {space}: MISSING")
        continue
    d = json.loads(p.read_text())
    print(f"  {space:8s} n={d['num_samples']:3d} pool={d['train_pool']}  "
          f"sim={d['mean_retrieval_similarity']:.4f}  "
          f"caption_jaccard={d['mean_caption_jaccard']:.4f}  "
          f"exact={d['exact_caption_match']}/{d['num_samples']}")
print()
print("  High similarity with LOW caption agreement = the embedding clusters utterances that do")
print("  not share a scene. That is the case where FAD flatters retrieval and per-example metrics")
print("  expose it -- report both together.")
PY

# ------------------------------------------------------- semantic scoring of retrieval
# Similarity and caption-Jaccard say how CLOSE the retrieved utterance was; they say nothing
# about whether the returned AUDIO is right for the query. Retrieval returns real recordings, so
# it sits on the real manifold by construction and is expected to beat us on distributional
# metrics while losing per-example. MARGIN and audio_r1 are what demonstrate that, so score the
# retrieved clips the same way every generated lane is scored.
RDIRS=()
for space in whisper clap; do
    [[ -f "$OUT/retrieval_${space}/manifest.jsonl" ]] && RDIRS+=("$OUT/retrieval_${space}")
done
if [[ ${#RDIRS[@]} -gt 0 ]]; then
    echo
    echo "### CLAP scoring of the retrieval lanes ###"
    CUDA_VISIBLE_DEVICES=0 python scripts/score_generations_clap.py \
        --dirs "${RDIRS[@]}" --split "${SPLIT:-val}" --out "$OUT/retrieval_scores.json" \
        2>&1 | tee "$LOG_DIR/retrieval_score.log" || echo "WARN: retrieval scoring failed" >&2
fi

echo
echo "wall-clock: ${TOTAL}s ($(python -c "print(f'{$TOTAL/60:.1f}')") min)"
echo "results: $OUT"
echo "logs:    $LOG_DIR"
[[ "$FAIL" == "0" ]] || { echo "ERROR: a lane failed" >&2; exit 1; }
echo "OK"
