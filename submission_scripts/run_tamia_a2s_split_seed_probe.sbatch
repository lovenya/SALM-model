#!/usr/bin/env bash
# TRAIN vs VAL vs TEST, and 6 seeds on the SAME clip. Generation only -- nothing is retrained.
#
# WHY. Two questions, one job, no new training.
#
# (1) IS THE PROBLEM OVERFITTING?  The user proposed this: "if train is fine, val is not, then
#     maybe what we need to solve is overfitting instead of something other task." The loss
#     curves already say overfitting is severe and were never in doubt -- dedup_k512 runs
#     train_code_loss 1.34 -> 0.21 while val_code_loss goes 1.50 -> 3.07, with the val minimum
#     at epoch 9 of 100:
#
#         epoch     0      9     19     39     59     79     99
#         train  5.051  1.345  0.966  0.488  0.315  0.232  0.210
#         val    3.078  1.500  1.704  2.377  2.797  3.027  3.072
#
#     What the curves CANNOT say is what memorisation sounds like, and that is the whole
#     question. A checkpoint that has all but memorised the train token sequences is a free
#     upper bound on the rest of the pipeline: whatever it produces from a TRAIN clip is the
#     best this codebook and renderer can do when the LM is essentially right.
#
# (2) HOW HARD DOES THE SCENE CONSTRAIN THE CONTENT?  Six seeds on one ambient clip. Never
#     measured. Same topic with different wording means the scene is doing real work and the
#     residual problem is fluency; six unrelated topics means the conditioning is weak and
#     scene_gap has been flattering us.
#
# A CONFOUND THIS JOB EXISTS PARTLY TO KEEP HONEST. val_scene_gap RISES MONOTONICALLY WITH
# OVERFITTING in every run measured -- dedup_k512 goes +0.041 at its val minimum to +0.133 at
# epoch 99, and frame-level k512 goes +0.022 to +0.087. So scene_gap is NOT a clean measure of
# scene conditioning; it also grows as the ambient becomes a sharper retrieval key into
# memorised training content. The published dedup-vs-frame-level comparison (+0.041 vs +0.022)
# survives this because both were read at their own val minimum, but scene_gap is from now on
# only comparable at matched val loss, and the epoch must be stated with the number.
#
# ======================= EXPECTED RESULTS (decided before running) =======================
# The decisive lane is TRAIN at `best` (~ep90, train loss 0.21), scored by
# scripts/a2s/score_memorisation.py, whose metrics are matched-minus-mismatched against a
# seeded derangement of the same lane's own references -- so the control text comes from the
# identical distribution and differs only in belonging to the clip.
#
#   A. TRAIN SHOWS A LARGE SPAN GAP (longest_span_gap >= ~6 words, ref_wer_gap clearly
#      negative) while VAL/TEST show ~0.
#        -> The codebook and the frozen renderer CAN carry long coherent speech. The islands
#           are a GENERALISATION failure, not a representation failure. Next move is data and
#           regularisation for the LM, and the two-level hierarchy becomes unnecessary.
#           This is the good outcome and it redirects the whole project.
#
#   B. TRAIN SHOWS ~NO GAP EITHER -- still 3-5 word islands even where the LM has memorised
#      the token sequence to 0.21 loss.
#        -> The bottleneck is DOWNSTREAM of the LM. Better planning, more data and less
#           overfitting all become irrelevant, because right codes are already not enough.
#           Next move is the codebook/renderer path itself. Note the oracle lane renders
#           near-verbatim from TRUE codes, so outcome B localises the failure precisely to
#           the gap between sampled-and-nearly-right codes and true codes -- meaning the
#           representation is not robust to small code errors, which is a concrete, testable
#           claim rather than a shrug.
#
#   C. VAL AND TEST DISAGREE with each other. Both are unseen, so they should not. This would
#      indicate a split problem (leakage or distribution shift), and every val-based number in
#      this project would need re-reading before anything else proceeds.
#
#   SEED SWEEP, read separately and at BOTH checkpoints: per-sample agreement across the 6
#   seeds. Same topic, different wording = scene conditioning is real. Six unrelated topics =
#   it is not, and scene_gap needs a replacement metric.
#
#   WHAT WOULD MAKE THIS JOB WORTHLESS, stated so it can be checked first: if the mismatched
#   arm scores near the matched arm in EVERY lane including train, the metric is not
#   discriminating and no conclusion above may be drawn from it. Check that before A/B/C.
#
#   NOT AN OUTCOME: absolute inter_asr_dist. It rewards saying something clearly over saying
#   the right thing and has already ranked LM lanes above their own oracle ceiling. It is
#   reported here only to place these lanes on the existing scale.
#
# HONEST CAVEAT. The renderer (content_concat_v4) was also trained on the train split, so the
# train lanes measure memorisation of the PIPELINE, not of the LM alone. That is the right
# scope for question (1) -- the upper bound we care about is end-to-end -- but it means a
# strong train result cannot be attributed to the LM by itself.
# =========================================================================================
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_split_seed_probe.sbatch
#SBATCH --job-name=salm-a2s-split-seed
#SBATCH --account=aip-csubakan
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

K="${K:-512}"                      # lowest val loss and largest scene_gap of the three
CTC_CKPT="$OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt"
RENDERER="$RUNS_ROOT/a2s_audioldm_clapseq_xformer/content_concat_v4/checkpoints/checkpoint_best.pt"
# best_loss is epoch 9, the val optimum. `best` is ~ep90, deep in the overfit regime and the
# latest checkpoint that survives on disk (ep99/last were deleted in the 2026-09-05 cleanup).
# BOTH are required: the comparison between them IS the overfitting measurement.
TAGS="${TAGS:-checkpoint_best_loss checkpoint_best}"
SPLITS="${SPLITS:-train val test}"
SEEDS="${SEEDS:-11 22 33 44 55 66}"
N_SPLIT="${N_SPLIT:-8}"            # split lanes: 8 clips, one seed
N_SEED="${N_SEED:-4}"              # seed lanes: 4 fixed clips, six seeds each
STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"
OUT_ROOT="$OUTPUTS_ROOT/generations/split_seed_${STAMP}"

for F in "$CTC_CKPT" "$RENDERER"; do
    [[ -f "$F" ]] || { echo "MISSING: $F" >&2; exit 1; }
done
for T in $TAGS; do
    [[ -f "$OUTPUTS_ROOT/content_lm/dedup_k${K}/checkpoints/${T}.pt" ]] \
        || { echo "MISSING LM: dedup_k${K}/${T}" >&2; exit 1; }
done

FREE=$(df -Pm /scratch/l/lovenya | awk 'NR==2{print $4}')
echo "free space: ${FREE} MB (need >= 1500)"
[[ "$FREE" -ge 1500 ]] || { echo "ABORT: only ${FREE} MB free; job 439442 died mid-write this way." >&2; exit 1; }

salm_print_header "SPLIT x SEED PROBE (K=$K, generation only, no retraining)"

# ---- build the lane list up front, so the GPU sharding below is a pure scheduling detail
LANES="$OUT_ROOT/lanes.txt"
mkdir -p "$OUT_ROOT"
: > "$LANES"
for TAG in $TAGS; do
  for SP in $SPLITS; do
    for TEMP in 0.7 1.0; do
      echo "split|$TAG|$SP|$TEMP|1337|$N_SPLIT|split__${TAG#checkpoint_}__${SP}__t${TEMP}" >> "$LANES"
    done
  done
  # Seed sweep on train and val only: test answers nothing here that val does not, and the
  # train-vs-val contrast is the entire point.
  for SP in train val; do
    for S in $SEEDS; do
      echo "seed|$TAG|$SP|1.0|$S|$N_SEED|seed__${TAG#checkpoint_}__${SP}__s${S}" >> "$LANES"
    done
  done
done
echo "planned lanes: $(wc -l < "$LANES")"

run_lane() {
    local gpu="$1" spec="$2"
    IFS='|' read -r _kind tag split temp seed n name <<< "$spec"
    local out_dir="$OUT_ROOT/$name"
    mkdir -p "$out_dir"
    echo "[gpu$gpu] $name"
    CUDA_VISIBLE_DEVICES="$gpu" python -m salm.generate --mode a2s --direction a2s \
        --checkpoint "$RENDERER" --content-source codebook_lm \
        --content-lm-checkpoint "$OUTPUTS_ROOT/content_lm/dedup_k${K}/checkpoints/${tag}.pt" \
        --content-lm-temperature "$temp" --ctc-checkpoint "$CTC_CKPT" \
        --split "$split" --num-samples "$n" --start-index 0 --seed "$seed" \
        --out-dir "$out_dir" --copy-reference \
        > "$out_dir.log" 2>&1 || echo "  FAILED: $name" >&2
}

# Four workers, one per GPU, striding the lane list. A lane failure must not abort its worker
# (that would silently drop every later lane on that GPU), hence the `|| echo` above.
NGPU=4
for g in $(seq 0 $((NGPU - 1))); do
  (
    i=0
    while IFS= read -r spec; do
      [[ $((i % NGPU)) -eq "$g" ]] && run_lane "$g" "$spec"
      i=$((i + 1))
    done < "$LANES"
  ) &
done
wait
echo "all lanes done"

echo
echo "==================== SCORING: memorisation / span ===================="
DIRS=""
for d in "$OUT_ROOT"/*/; do [[ -f "$d/manifest.jsonl" ]] && DIRS="$DIRS ${d%/}"; done
[[ -n "$DIRS" ]] || { echo "no lanes produced a manifest -- nothing to score" >&2; exit 1; }
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES=0 python scripts/a2s/score_memorisation.py \
    --dirs $DIRS --device cuda \
    --out "$OUTPUTS_ROOT/memorisation_${STAMP}.json" \
    2>&1 | tee "$OUTPUTS_ROOT/logs/slurm/split_seed_mem_${STAMP}.log"

echo
echo "==================== SCORING: intelligibility (scale only) ===================="
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES=0 python scripts/a2s/score_intelligibility.py \
    --dirs $DIRS --ctc-checkpoint "$CTC_CKPT" --device cuda \
    --out "$OUTPUTS_ROOT/intelligibility_split_seed_${STAMP}.json" \
    2>&1 | tee "$OUTPUTS_ROOT/logs/slurm/split_seed_intel_${STAMP}.log"

echo
echo "=========================================================="
echo "listen: $OUT_ROOT/*/audio/"
echo "  split__* : same seed, different split  -> is train better than val?"
echo "  seed__*  : same 4 clips, 6 seeds       -> does the scene pin the topic?"
echo
echo "READ FIRST: is longest_span_gap ~0 in EVERY lane including train? Then the metric did"
echo "not discriminate and no conclusion may be drawn. Only after that, compare train vs val."
echo "free space remaining: $(df -Pm /scratch/l/lovenya | awk 'NR==2{print $4}') MB"
echo "=========================================================="
echo "OK"
