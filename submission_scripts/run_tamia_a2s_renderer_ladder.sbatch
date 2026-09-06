#!/usr/bin/env bash
# RENDERER LADDER: 100 more epochs, a checkpoint every 20, and generate from every one so the
# choice is made BY EAR. Requested by Lovenya 2026-09-06.
#
# WHY BY EAR AND NOT BY METRIC. Today the metrics and the ear disagreed twice, and the ear was
# right both times. `inter_asr_dist` put continuous at 0.3223 and quantized at 0.5903 -- most of
# the distance from real speech (0.3146) to noise (0.967) -- and Lovenya heard them as
# "90/95% same". The reason is now understood: every lane here shares the same content LM and
# greedy decoding, so all lanes emit IDENTICAL content codes and therefore the SAME WORDS. A
# renderer can only change how those words sound, never which words they are. `inter_asr_dist`
# is very sensitive to acoustic clarity; a listener hearing the same sentence twice reasonably
# says they are the same. So this job scores nothing for ranking -- it builds a bundle.
#
# WHAT THIS CAN AND CANNOT DO. It cannot fix coherence. The greedy control (442847) showed the
# renderer reproduces the same words at 0.96/1.00/0.93 from fixed codes, and Lovenya's A/B
# today confirmed it from the other side. This settles the ACOUSTIC layer so it can be frozen,
# and everything after goes to the content LM -- where the measured 97% of the loss lives.
#
# ONE CLEAN RUN FROM v4, NOT A CHAIN. The first draft of this job warm-started from
# rquant_continuous (itself warm-started from v4), giving v4 -> 36 epochs -> 100 epochs: three
# separate OneCycleLR cycles, each re-warming the learning rate from near zero and annealing
# again. Worse, `renderer_checkpoint` restores only unet + content_projector, NOT the
# conditioner -- so every link in such a chain silently restarts the conditioner from scratch,
# and chaining compounds that. The resulting "ladder" would measure restart artefacts as much
# as training.
#
# So this starts from v4's best with a single schedule over the whole run. The epoch axis then
# means what it says. Not random init: the U-Net is 185M parameters against 17,596 clips and
# only works from AudioLDM's pretrained weights.
#
# (`resume_from` would be the other way to continue, and it is a trap here: it restores the
# OneCycleLR state whose total_steps was 4080, sized for the previous run's 60 epochs, so
# stepping further epochs past that raises.)
#
# ======================= EXPECTED RESULTS (decided before running) =======================
# The deliverable is a listening bundle, and Lovenya picks. Stated anyway so the outcome is
# interpretable rather than just collected:
#
#   THE HONEST PRIOR IS THAT THIS PLATEAUS. The previous run's `val_x0_rel_mse` peaked at epoch
#   14 (0.2786, itself better than v4's 0.2802 from 200 epochs) and then sat at 0.281-0.288 for
#   21 more epochs. The audio Lovenya liked came from that epoch-14 checkpoint. So the most
#   likely result is that the ladder sounds flat and the early checkpoint wins.
#
#   That is NOT an argument against running it, and the reason matters: the metric that
#   plateaued is the same class of metric that disagreed with the ear twice today. A plateau on
#   a measure that does not predict perception is not evidence the audio stopped improving.
#   The only way to know is to listen, which is what this produces.
#
#   LADDER IMPROVES with epoch -> keep the late checkpoint, freeze it, move on.
#   LADDER FLAT           -> keep epoch 14, freeze it, move on. The acoustic layer is settled
#                            and no further renderer training is justified.
#   LADDER DEGRADES       -> the previous run was already past its best; keep epoch 14 and
#                            record that renderer fine-tuning has a short useful window.
#   Every outcome ends the same way: one frozen renderer and the question closed.
#
#   EVERY LANE WILL SAY THE SAME WORDS. That is by construction, not a bug -- judge timbre,
#   articulation and clarity, not content. If two lanes differ in WORDS, something is wrong
#   with the generation and the comparison is invalid; say so rather than ranking them.
# =========================================================================================
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_renderer_ladder.sbatch
#SBATCH --job-name=salm-a2s-rend-ladder
#SBATCH --account=aip-csubakan
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=04:30:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

ARCH="${ARCH:-a2s_audioldm_clapseq_xformer}"
K="${K:-2048}"
EPOCHS="${EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-16}"
# 68 steps/epoch at batch 16 (the previous run used 4080 for 60). Sized for the WHOLE run so
# OneCycleLR completes exactly once, at the end -- the point of not chaining.
LR_TOTAL_STEPS="${LR_TOTAL_STEPS:-10200}"
TRAIN_MINUTES="${TRAIN_MINUTES:-200}"   # 150 x 1.21 = ~182 min; NOT the binding limit (that
                                        # cap is what truncated the previous run at 36 epochs)
NGEN="${NGEN:-64}"
PREV="$RUNS_ROOT/$ARCH/rquant_continuous_20260906_1425"   # kept only for its two listening rungs
BASE="$RUNS_ROOT/$ARCH/content_concat_v4/checkpoints/checkpoint_best.pt"
CTC_CKPT="$OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt"
LM_CKPT="$OUTPUTS_ROOT/content_lm/dedup_k${K}/checkpoints/checkpoint_best_loss.pt"
STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"
RUN_NAME="rladder_${STAMP}"
OUT_ROOT="$OUTPUTS_ROOT/generations/rladder_${STAMP}"

for F in "$BASE" "$PREV/checkpoints/checkpoint_last.pt" "$PREV/checkpoints/checkpoint_best.pt" \
         "$CTC_CKPT" "$LM_CKPT" "configs/arch/${ARCH}.yaml" configs/exp/content_concat.yaml; do
    [[ -e "$F" ]] || { echo "MISSING: $F" >&2; exit 1; }
done
[[ -e "$RUNS_ROOT/$ARCH/$RUN_NAME" ]] && { echo "ABORT: run dir exists" >&2; exit 1; }
FREE=$(df -Pm /scratch/l/lovenya | awk 'NR==2{print $4}')
echo "free space: ${FREE} MB (need >= 12000: ~9 GB of checkpoints plus ~450 MB of wavs)"
[[ "$FREE" -ge 12000 ]] || { echo "ABORT: only ${FREE} MB free" >&2; exit 1; }

salm_print_header "RENDERER LADDER: $EPOCHS epochs from content_concat_v4 best, ONE schedule"

echo
echo "==================== 1. TRAIN ===================="
torchrun --nproc_per_node=4 --master_port=29563 -m salm.train_ldm \
    --config configs/base/a2s_audioldm.yaml "configs/arch/${ARCH}.yaml" \
        configs/exp/o2_align.yaml configs/exp/content_concat.yaml \
    --override \
        "run_name=$RUN_NAME" "arch=$ARCH" "epochs=$EPOCHS" \
        "max_train_minutes=$TRAIN_MINUTES" "batch_size=$BATCH_SIZE" \
        "lr_total_steps=$LR_TOTAL_STEPS" "save_every_epochs=20" \
        "renderer_checkpoint=$BASE" "freeze_renderer=false" \
    2>&1 | tee "$OUTPUTS_ROOT/logs/slurm/rladder_${STAMP}.log"

RUN_DIR="$RUNS_ROOT/$ARCH/$RUN_NAME"
echo
echo "==================== 2. WHAT GOT SAVED ===================="
ls "$RUN_DIR/checkpoints" 2>/dev/null | sed 's/^/  /' || echo "  (no checkpoints -- training failed)"

echo
echo "==================== 3. GENERATE FROM EVERY RUNG ===================="
# The two rungs that already exist, so the ladder starts where the listening did.
declare -a LANES=( "ep14_prev:$PREV/checkpoints/checkpoint_best.pt"
                   "ep35_prev:$PREV/checkpoints/checkpoint_last.pt" )
for CK in "$RUN_DIR"/checkpoints/*.pt; do
    [[ -e "$CK" ]] || continue
    tag="$(basename "${CK%.pt}")"; tag="${tag#checkpoint_}"
    LANES+=( "new_${tag}:$CK" )
done
echo "  ${#LANES[@]} rungs to generate"

for pair in "${LANES[@]}"; do
    label="${pair%%:*}"; ck="${pair#*:}"
    [[ -f "$ck" ]] || { echo "  skip (absent): $label" >&2; continue; }
    if [[ "$(df -Pm /scratch/l/lovenya | awk 'NR==2{print $4}')" -lt 800 ]]; then
        echo "  STOPPING: under 800 MB free. Completed rungs are intact." >&2; break
    fi
    d="$OUT_ROOT/$label"; mkdir -p "$d"
    echo "  === $label ==="
    CUDA_VISIBLE_DEVICES=0 python -m salm.generate --mode a2s --direction a2s \
        --checkpoint "$ck" --content-source codebook_lm --content-lm-checkpoint "$LM_CKPT" \
        --content-lm-temperature 1.0 --content-lm-top-k 1 --ctc-checkpoint "$CTC_CKPT" \
        --split val --num-samples "$NGEN" --start-index 0 --seed 1337 \
        --out-dir "$d" --copy-reference > "$d.log" 2>&1 || echo "  FAILED: $label" >&2
done

echo
echo "==================== 4. THE BUNDLE (the deliverable) ===================="
# --require-all-lanes: a rung missing from some clips would produce folders holding half the
# comparison, which is worse than useless because it looks complete.
python scripts/a2s/build_listening_bundle.py --gen-root "$OUT_ROOT" \
    --out "$OUTPUTS_ROOT/listening_bundles/rladder_${STAMP}" \
    --max-clips 10 --require-all-lanes || echo "BUNDLE FAILED -- lanes intact" >&2

echo
echo "==================== 5. metrics, for the record only ===================="
# Recorded because they are cheap, NOT to rank the rungs. The ear ranks them; today these
# numbers disagreed with it twice.
CTRL="$OUTPUTS_ROOT/generations/control_lanes_20260831_0735"
DIRS="$CTRL/real_speech $CTRL/vae_roundtrip"
for d in "$OUT_ROOT"/*/; do [[ -f "$d/manifest.jsonl" ]] && DIRS="$DIRS ${d%/}"; done
# shellcheck disable=SC2086
python scripts/a2s/score_intelligibility.py --dirs $DIRS --ctc-checkpoint "$CTC_CKPT" \
    --device cuda --out "$OUTPUTS_ROOT/rladder_intel_${STAMP}.json" \
    || echo "SCORING FAILED -- generations intact" >&2

echo
echo "=========================================================="
echo "LISTEN AND PICK: $OUTPUTS_ROOT/listening_bundles/rladder_${STAMP}"
echo "  rsync -avzP -e \"ssh -J rorqual\" \\"
echo "    \"tamia:$OUTPUTS_ROOT/listening_bundles/rladder_${STAMP}/\" ~/salm_listen/rladder/"
echo
echo "Every rung says the SAME WORDS by construction -- same LM, same greedy codes. Judge"
echo "timbre and clarity only. If two rungs differ in WORDS, the comparison is broken."
echo "=========================================================="
echo "OK"
