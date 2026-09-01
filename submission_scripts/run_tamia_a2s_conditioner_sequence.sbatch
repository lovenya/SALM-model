#!/usr/bin/env bash
# END-TO-END SEQUENTIAL CONDITIONING -- train + generate in one job.
#
# THE QUESTION. Can the conditioner infer the content channel from the AMBIENT AUDIO ALONE?
# Everything that has produced intelligible speech in this project so far got its content
# from somewhere it will not have at deployment: the oracle path reads the target speech,
# the retrieval path copies a training utterance. This run removes both. The only input is
# this row's own 10.24s ambient recording; the conditioner's own pre-pool sequence (which it
# already computes and currently throws away in self.pool) is upsampled 32 -> 256 and drives
# the proven concat channel. Trained on the diffusion loss alone -- no oracle target, no
# regression teacher (lambda_content_aux is 0 here on purpose).
#
# See configs/exp/conditioner_sequence.yaml and src/salm/a2s/conditioner_sequence.py for the
# full reasoning. Summary of why this and not the alternatives:
#   - pooled vector cannot carry content        (established three independent ways)
#   - concat channel renders content correctly  (job 433314, oracle -> near-verbatim speech)
#   - regression predicting content fails       (MSE -> E[content|ambient], the blurry mean)
#   - retrieval works but copies training words (rejected as a deliverable)
#
# ======================= EXPECTED RESULTS (decided before running) =======================
# The metrics that actually reflect this mechanism are val_content_bind_gap and
# val_x0_rel_mse. NOT val_clap_r1 and NOT train_align: both score the POOLED conditioner
# output, which this change never touches, so both would look identical whether the new head
# works or is stone dead. Verified by reading the metric computation, not by trusting a
# template's summary block. The summary at the bottom of this script prints the right ones.
#
#   SUCCESS -- val_content_bind_gap rises durably above ~+0.05 (oracle-fed concat reached
#     +0.36) AND val_x0_rel_mse improves on the zero-content baseline (0.360 at init).
#     Meaning: the U-Net is binding the head's output frame-to-frame, i.e. the conditioner
#     is emitting time-localized content it inferred from ambience. Confirm by ear on the
#     generations this job also produces: `conditioner` should be audibly more speech-like
#     than the `zero` control at the same checkpoint. That would be the first content in
#     this project that is neither read off the target nor looked up from the training set.
#
#   FAILURE -- bind_gap flat near 0 through epoch ~15 and never recovers. Meaning: a third
#     independent route (after cross-attention and after regression) has produced content
#     with timing/energy but no phonetic identity. That is informative, not wasted: it
#     argues the bottleneck is neither the injection interface nor the objective, but that
#     ~24 h of paired data cannot support learning phonetic content from ambience at all --
#     and the next move is a warm start from a model that already has speech (VoiceLDM,
#     Apache-2.0, ~6,750 h), not another mechanism. Note bind_gap can be watched live in
#     history.json; the run does not have to finish before that call can be made.
#
#   AMBIGUOUS -- bind_gap small but clearly positive while x0_rel_mse stays flat. Meaning:
#     partial signal. The designated follow-up is already wired: rerun with
#     lambda_content_aux > 0 to add the auxiliary teacher, which is deliberately off here so
#     the first measurement is of the pure diffusion-loss objective and nothing else.
#
#   CONTROL that makes the above readable: the same job generates `zero` content from every
#     checkpoint. If `conditioner` and `zero` sound identical, the head is dead regardless
#     of what any curve does -- this is exactly the oracle-vs-shuffled trap that job 432420
#     fell into, where x0_rel_mse improved while the mechanism carried no identity. It also
#     generates `oracle` from the last checkpoint as an upper bound, confirming the concat
#     channel in THIS model still renders good content correctly.
# =========================================================================================
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_conditioner_sequence.sbatch
#SBATCH --job-name=salm-a2s-condseq
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
RUN_NAME="${RUN_NAME:-conditioner_sequence_v1}"
EPOCHS="${EPOCHS:-250}"
# batch_size 64 per GPU x 4 = 68 steps/epoch, measured 0.49 min/epoch on content_concat_v4.
# 250 epochs ~= 2h05. lr_total_steps MUST equal steps_per_epoch * EPOCHS for this batch size
# or the cosine curve lands at the wrong position; pass both together if overriding either.
BATCH_SIZE="${BATCH_SIZE:-64}"
LR_TOTAL_STEPS="${LR_TOTAL_STEPS:-17000}"
# Budget, from measurement rather than padding: 250 ep x 0.493 min/ep (content_concat_v4,
# 200 ep in 98.7 min) x 1.15 for this run's second conditioner forward pass = ~142 min of
# training; 19 generation invocations x ~30 s each (job 432123 did 7 in 3:29 wall, model
# load included) = ~10 min; ~5 min setup. 165 min requested against ~157 needed. The cap
# below is the safety valve: if the node runs slower, training truncates gracefully at 140
# min with checkpoints saved and generation still gets its slot, rather than the whole job
# being killed mid-write.
TRAIN_MINUTES="${TRAIN_MINUTES:-140}"
NUM_GEN="${NUM_GEN:-16}"

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
[[ "$MISSING" == "0" ]] || { echo "ERROR: missing prerequisites, see above" >&2; exit 1; }

cat > "$RUN_DIR/run_metadata.json" <<EOF
{
  "run_name": "$RUN_NAME", "arch": "$ARCH",
  "purpose": "end-to-end sequential conditioning: conditioner infers content from ambient audio alone",
  "content_source": "conditioner pre-pool sequence (NO oracle, NO retrieval, NO predictor)",
  "slurm_job_id": "${SLURM_JOB_ID:-none}", "account": "${SLURM_JOB_ACCOUNT:-unknown}",
  "hostname": "$(hostname)", "repo_path": "$REPO",
  "git_commit": "$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)",
  "git_dirty": $(if [[ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]]; then echo true; else echo false; fi),
  "epochs": $EPOCHS, "batch_size": $BATCH_SIZE, "train_minutes": $TRAIN_MINUTES,
  "started_at": "$(TZ=America/Toronto date -Iseconds)"
}
EOF

salm_print_header "End-to-end sequential conditioning: $EPOCHS epochs, run=$RUN_NAME"
salm_start_monitor "$LOG_DIR" "$RUN_NAME" 60

CKPT="$CKPT_DIR/checkpoint_last.pt"
RESUME_ARGS=()
if [[ -f "$CKPT" ]]; then
    echo "resuming from existing checkpoint: $CKPT"
    RESUME_ARGS=("resume_from=$CKPT")
fi

torchrun --nproc_per_node=4 --master_port=29547 -m salm.train_ldm \
    --config configs/base/a2s_audioldm.yaml "configs/arch/${ARCH}.yaml" \
        configs/exp/o2_align.yaml configs/exp/content_concat.yaml \
        configs/exp/conditioner_sequence.yaml \
    --override \
        "run_name=$RUN_NAME" "arch=$ARCH" "epochs=$EPOCHS" \
        "max_train_minutes=$TRAIN_MINUTES" \
        "batch_size=$BATCH_SIZE" "lr_total_steps=$LR_TOTAL_STEPS" \
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
    # THE metric for this run. Everything else below is context.
    bg = series("val_content_bind_gap")
    if bg:
        best = max(bg, key=lambda x: x[1])
        print(f"val_content_bind_gap : first={bg[0][1]:+.4f} last={bg[-1][1]:+.4f} "
              f"best={best[1]:+.4f} @ep{best[0]}")
        print("    >= +0.05 durably = the U-Net binds the head's output frame-to-frame")
        print("    ~0 through ep15  = head emits timing/energy but no phonetic identity")
        print("    (oracle-fed concat reference: +0.0087 at ep0 -> +0.358 at ep199)")
    else:
        print("val_content_bind_gap : NOT RECORDED -- this run cannot answer its own question")
    for key, ref in (("val_x0_rel_mse", "0.360 at init, 0.292 oracle-fed at ep199"),
                     ("val_loss", None), ("train_diffusion", None)):
        v = series(key)
        if v:
            line = f"{key:21s}: first={v[0][1]:.4f} last={v[-1][1]:.4f}"
            print(line + (f"   (reference: {ref})" if ref else ""))
    aux = series("train_content_aux")
    if aux:
        print(f"train_content_aux    : first={aux[0][1]:.4f} last={aux[-1][1]:.4f} "
              "(diagnostic only unless lambda_content_aux > 0)")
    # Printed ONLY to confirm the pooled pathway did not regress. Neither can show
    # whether the new sequential head works -- both score the pooled output it never touches.
    for key in ("val_clap_r1", "train_align"):
        v = series(key)
        if v:
            print(f"[pooled-pathway only, NOT this run's test] {key}: "
                  f"first={v[0][1]:.4f} last={v[-1][1]:.4f}")
PY
echo "=========================================================="

# ---------------------------------------------------------------- generation
# Full saved spread, not one pick (standing repo policy): "best" is chosen by
# val_x0_rel_mse, which need not coincide with the epoch where the head's content is most
# audible. Two sources everywhere -- `conditioner` (the real thing) and `zero` (the control
# that makes it readable) -- plus `oracle` on the last checkpoint as an upper bound.
STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"
OUT_ROOT="$OUTPUTS_ROOT/generations/conditioner_sequence_${STAMP}"

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
            # || true: one bad checkpoint must not take down the other 8. A crash here is
            # visible in the per-run log and in the missing audio directory.
            python -m salm.generate --mode a2s --direction a2s \
                --checkpoint "$CKPT_DIR/$name" \
                --content-source "$src" --ctc-checkpoint "$CTC_CKPT" \
                --split val --num-samples "$NUM_GEN" \
                --out-dir "$out_dir" --copy-reference \
                2>&1 | tee "$out_dir.log" || echo "  FAILED: $tag/$src (continuing)" >&2
        done
    done

    # Upper bound on the last checkpoint only: does the concat channel in THIS model still
    # render good content? If oracle is intelligible and conditioner is not, the renderer is
    # fine and the content source is what failed -- a different conclusion from both failing.
    if [[ -f "$CKPT_DIR/checkpoint_last.pt" ]]; then
        out_dir="$OUT_ROOT/checkpoint_last__oracle"
        mkdir -p "$out_dir"
        echo
        echo "=== checkpoint_last / oracle (upper bound) ==="
        python -m salm.generate --mode a2s --direction a2s \
            --checkpoint "$CKPT_DIR/checkpoint_last.pt" \
            --content-source oracle --ctc-checkpoint "$CTC_CKPT" \
            --split val --num-samples "$NUM_GEN" \
            --out-dir "$out_dir" --copy-reference \
            2>&1 | tee "$out_dir.log" || echo "  FAILED: oracle upper bound (continuing)" >&2
    fi
fi

echo
echo "=========================================================="
echo "listen: $OUT_ROOT/<tag>__<source>/audio/"
echo "  <id>.wav                  generated speech"
echo "  <id>__input_ambient.wav   the ambient scene it was generated FROM"
echo "  <id>__reference.wav       the real paired speech"
echo
echo "THE TEST: at the same checkpoint, does __conditioner sound more speech-like than"
echo "__zero? If identical, the head is dead no matter what the curves say."
echo "=========================================================="
echo "OK"
