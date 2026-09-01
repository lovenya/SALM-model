#!/usr/bin/env bash
# PHASE 2 of end-to-end sequential conditioning -- train + generate in one job.
#
# WHAT WENT WRONG IN PHASE 1 (job 434308, 2026-09-01). The renderer (unet + content_projector)
# and the scene-understanding head (conditioner + sequence_head) trained SIMULTANEOUSLY from
# scratch. The renderer was never once shown real content -- only the head's own uninformative
# guesses -- while the head had no fixed, competent target to push toward. val_content_bind_gap
# never left +0.0000/+0.0004 across 250 epochs; ASR scoring confirmed conditioner-derived
# content and literal zero content were indistinguishable at every checkpoint. Root cause
# documented in docs/LEARNINGS.md and docs/EXPERIMENT_LOG.md (2026-09-01 entries).
#
# THE FIX. Freeze the renderer at content_concat_v4's own proven weights -- the checkpoint
# that turned oracle content into near-verbatim speech (job 433314: "Let's see if she starts
# this time... wake up!" -> "Let's see if she stops this time... wake up."). Warm-start the
# conditioner from the same checkpoint (still trainable). Only conditioner + a fresh
# sequence_head train, against a renderer guaranteed capable. See
# configs/exp/conditioner_sequence_phase2.yaml and src/salm/a2s/conditioner_sequence.py.
#
# ======================= EXPECTED RESULTS (decided before running) =======================
# Same instrument as phase 1: val_content_bind_gap and val_x0_rel_mse. NOT val_clap_r1/
# train_align -- pooled-pathway only, blind to this mechanism regardless of freeze state.
#
#   SUCCESS -- bind_gap rises durably above ~+0.05. This bar is now MEANINGFUL in a way phase
#     1's wasn't: this run's renderer is proven capable, so a rise here is real evidence the
#     head emits position-differentiated content a working renderer uses. Confirm by ear:
#     `conditioner` should sound more speech-like than `zero` at the same checkpoint.
#
#   FAILURE -- bind_gap stays ~0 again. A MUCH stronger negative than phase 1's, because the
#     "was the renderer ever shown real content" confound is gone. Argues the bottleneck is
#     genuinely upstream (data scale or the CLAP ambient representation itself, per 434204's
#     thin/overfitting retrieval signal). Next move: VoiceLDM warm start, not another
#     mechanism on the same ~24h of data.
#
#   AMBIGUOUS -- bind_gap small but positive. Next step: lambda_content_aux > 0, now on top
#     of a working renderer -- a materially different, better-controlled version of the idea
#     phase 1 left untested.
#
#   CONTROL: generates `zero` from every checkpoint alongside `conditioner`. Also generates
#     `oracle` on the last checkpoint -- and unlike phase 1, this IS a valid upper bound now,
#     since the renderer is v4's own proven one: it should reproduce something close to
#     v4's original oracle-content intelligibility, confirming the freeze/warm-start loaded
#     correctly before trusting anything else this job reports.
# =========================================================================================
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_conditioner_sequence_phase2.sbatch
#SBATCH --job-name=salm-a2s-condseq-p2
#SBATCH --account=aip-ravanelm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=02:45:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

ARCH="${ARCH:-a2s_audioldm_clapseq_xformer}"
RUN_NAME="${RUN_NAME:-conditioner_sequence_phase2_v1}"
EPOCHS="${EPOCHS:-250}"
# Same batch/step budget as phase 1 (job 434308): 68 steps/epoch, 250 epochs completed in
# 1:44:39 against a 2:45 limit there. Kept identical rather than shortened -- only the
# trainable surface shrank (conditioner + sequence_head vs. the full renderer), which does
# not by itself justify a shorter epoch budget without a measurement to back it.
BATCH_SIZE="${BATCH_SIZE:-64}"
LR_TOTAL_STEPS="${LR_TOTAL_STEPS:-17000}"
TRAIN_MINUTES="${TRAIN_MINUTES:-140}"
NUM_GEN="${NUM_GEN:-16}"

RENDERER_CKPT="${RENDERER_CKPT:-$RUNS_ROOT/a2s_audioldm_clapseq_xformer/content_concat_v4/checkpoints/checkpoint_best.pt}"

RUN_DIR="$RUNS_ROOT/$ARCH/$RUN_NAME"
LOG_DIR="$RUN_DIR/logs"
CKPT_DIR="$RUN_DIR/checkpoints"
mkdir -p "$LOG_DIR"

MISSING=0
for SPLIT in train val; do
    for F in "$FEATURES_ROOT/a2s/$SPLIT/latents_mean.npy" "$OUTPUTS_ROOT/features_a2s_cond/$SPLIT/cond_seq.npy"; do
        [[ -f "$F" ]] || { echo "MISSING: $F" >&2; MISSING=1; }
    done
done
CTC_CKPT="$OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt"
[[ -f "$CTC_CKPT" ]] || { echo "MISSING: $CTC_CKPT" >&2; MISSING=1; }
[[ -f "$RENDERER_CKPT" ]] || { echo "MISSING: $RENDERER_CKPT" >&2; MISSING=1; }
[[ "$MISSING" == "0" ]] || { echo "ERROR: missing prerequisites, see above" >&2; exit 1; }

cat > "$RUN_DIR/run_metadata.json" <<EOF
{
  "run_name": "$RUN_NAME", "arch": "$ARCH",
  "purpose": "phase 2: scene-understanding head trained against a FROZEN, proven renderer",
  "renderer_checkpoint": "$RENDERER_CKPT",
  "content_source": "conditioner pre-pool sequence (NO oracle, NO retrieval, NO predictor)",
  "slurm_job_id": "${SLURM_JOB_ID:-none}", "account": "${SLURM_JOB_ACCOUNT:-unknown}",
  "hostname": "$(hostname)", "repo_path": "$REPO",
  "git_commit": "$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)",
  "git_dirty": $(if [[ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]]; then echo true; else echo false; fi),
  "epochs": $EPOCHS, "batch_size": $BATCH_SIZE, "train_minutes": $TRAIN_MINUTES,
  "started_at": "$(TZ=America/Toronto date -Iseconds)"
}
EOF

salm_print_header "Phase 2 sequential conditioning (frozen renderer): $EPOCHS epochs, run=$RUN_NAME"
salm_start_monitor "$LOG_DIR" "$RUN_NAME" 60

CKPT="$CKPT_DIR/checkpoint_last.pt"
RESUME_ARGS=()
if [[ -f "$CKPT" ]]; then
    echo "resuming from existing checkpoint: $CKPT"
    RESUME_ARGS=("resume_from=$CKPT")
fi

torchrun --nproc_per_node=4 --master_port=29548 -m salm.train_ldm \
    --config configs/base/a2s_audioldm.yaml "configs/arch/${ARCH}.yaml" \
        configs/exp/o2_align.yaml configs/exp/content_concat.yaml \
        configs/exp/conditioner_sequence_phase2.yaml \
    --override \
        "run_name=$RUN_NAME" "arch=$ARCH" "epochs=$EPOCHS" \
        "max_train_minutes=$TRAIN_MINUTES" \
        "batch_size=$BATCH_SIZE" "lr_total_steps=$LR_TOTAL_STEPS" \
        "renderer_checkpoint=$RENDERER_CKPT" \
        ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
    2>&1 | tee -a "$LOG_DIR/train.log"

echo
echo "==================== TRAINING SUMMARY ===================="
python - <<PY
import json, pathlib
run = pathlib.Path("$RUN_DIR")
h = json.loads((run / "history.json").read_text()) if (run / "history.json").exists() else []
print(f"epochs completed: {len(h)}")
if not h:
    print("NO epochs completed -- check the log above for a crash.")
else:
    def series(k):
        return [(r["epoch"], r[k]) for r in h if r.get(k) is not None]
    bg = series("val_content_bind_gap")
    if bg:
        best = max(bg, key=lambda x: x[1])
        print(f"val_content_bind_gap : first={bg[0][1]:+.4f} last={bg[-1][1]:+.4f} "
              f"best={best[1]:+.4f} @ep{best[0]}")
        print("    >= +0.05 durably = the head produces content this proven renderer binds to")
        print("    ~0 the whole run  = failure is upstream (data scale / CLAP features),")
        print("                        not the renderer -- that confound is now eliminated")
        print("    (phase-1 reference: pinned +0.0000/+0.0004 the entire 250 epochs)")
    else:
        print("val_content_bind_gap : NOT RECORDED -- this run cannot answer its own question")
    for key, ref in (("val_x0_rel_mse", "phase 1: flat ~0.35-0.36 the whole run"),
                     ("val_loss", None), ("train_diffusion", None)):
        v = series(key)
        if v:
            line = f"{key:21s}: first={v[0][1]:.4f} last={v[-1][1]:.4f}"
            print(line + (f"   (reference: {ref})" if ref else ""))
    aux = series("train_content_aux")
    if aux:
        print(f"train_content_aux    : first={aux[0][1]:.4f} last={aux[-1][1]:.4f} "
              "(diagnostic only unless lambda_content_aux > 0)")
    for key in ("val_clap_r1", "train_align"):
        v = series(key)
        if v:
            print(f"[pooled-pathway only, NOT this run's test] {key}: "
                  f"first={v[0][1]:.4f} last={v[-1][1]:.4f}")
PY
echo "=========================================================="

# ---------------------------------------------------------------- generation
STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"
OUT_ROOT="$OUTPUTS_ROOT/generations/conditioner_sequence_phase2_${STAMP}"

CKPTS=()
for name in checkpoint_ep19.pt checkpoint_ep59.pt checkpoint_ep99.pt checkpoint_ep139.pt \
            checkpoint_ep179.pt checkpoint_ep219.pt checkpoint_ep249.pt \
            checkpoint_last.pt checkpoint_best.pt; do
    [[ -f "$CKPT_DIR/$name" ]] && CKPTS+=("$name")
done

if [[ ${#CKPTS[@]} -eq 0 ]]; then
    echo "no checkpoints found in $CKPT_DIR -- skipping generation" >&2
else
    echo "generating from ${#CKPTS[@]} checkpoint(s): ${CKPTS[*]}"
    for name in "${CKPTS[@]}"; do
        tag="${name%.pt}"
        for src in conditioner zero; do
            out_dir="$OUT_ROOT/${tag}__${src}"
            mkdir -p "$out_dir"      # tee opens its file immediately; python's mkdir is late
            echo
            echo "=== $tag / $src ==="
            python -m salm.generate --mode a2s --direction a2s \
                --checkpoint "$CKPT_DIR/$name" \
                --content-source "$src" --ctc-checkpoint "$CTC_CKPT" \
                --split val --num-samples "$NUM_GEN" \
                --out-dir "$out_dir" --copy-reference \
                2>&1 | tee "$out_dir.log" || echo "  FAILED: $tag/$src (continuing)" >&2
        done
    done

    # NOW a valid upper bound (unlike phase 1): the renderer here IS content_concat_v4's own
    # proven one, frozen. Should land close to v4's original oracle-content result -- if it
    # doesn't, the freeze/warm-start loading is suspect and everything else this job reports
    # needs re-checking before drawing conclusions from it.
    if [[ -f "$CKPT_DIR/checkpoint_last.pt" ]]; then
        out_dir="$OUT_ROOT/checkpoint_last__oracle"
        mkdir -p "$out_dir"
        echo
        echo "=== checkpoint_last / oracle (renderer sanity check -- should sound like v4) ==="
        python -m salm.generate --mode a2s --direction a2s \
            --checkpoint "$CKPT_DIR/checkpoint_last.pt" \
            --content-source oracle --ctc-checkpoint "$CTC_CKPT" \
            --split val --num-samples "$NUM_GEN" \
            --out-dir "$out_dir" --copy-reference \
            2>&1 | tee "$out_dir.log" || echo "  FAILED: oracle sanity check (continuing)" >&2
    fi
fi

echo
echo "=========================================================="
echo "listen: $OUT_ROOT/<tag>__<source>/audio/"
echo "  <id>.wav                  generated speech"
echo "  <id>__input_ambient.wav   the ambient scene it was generated FROM"
echo "  <id>__reference.wav       the real paired speech"
echo
echo "SANITY CHECK FIRST: does checkpoint_last__oracle sound like v4's original oracle result"
echo "(near-verbatim)? If not, the renderer freeze/warm-start is broken -- fix that before"
echo "trusting anything else here."
echo "THE ACTUAL TEST: at the same checkpoint, does __conditioner sound more speech-like than"
echo "__zero? If identical, the head still isn't learning anything usable, even with a"
echo "working renderer to push against."
echo "=========================================================="
echo "OK"
