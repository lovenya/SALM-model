#!/usr/bin/env bash
# PLAN VECTOR, REPLICATED ACROSS SEEDS. Recovering a result that was closed on the wrong metric
# and whose checkpoints were then deleted.
#
# WHAT HAPPENED. Job 443751 trained no-plan / lambda=1.0 / lambda=0.3 in one matched job and
# scored them. The agent judged it on `horizon_cross_seed` (1 / 0 / 1), called it null, closed
# the direction, and later recommended deleting the checkpoints. The job's own audio scores were
# written to `plan_intel_20260906_1222.json` and never opened. Re-analysed paired on the shared
# 64 clips:
#
#     lambda=0.3 vs its own control:  raw -0.0772  95% CI [-0.121, -0.034]
#                                     length-adjusted -0.0708, better on 46/64 clips
#     lambda=1.0 vs its own control:  raw -0.0514  95% CI [-0.096, -0.006]
#
# For scale, the greedy-decoding change that reframed this project was 0.083. Word counts were
# 17.4 -> 16.6, so this is NOT the "says less" artifact that explained the coarse-to-fine
# result. `horizon_cross_seed` has since been shown to return 1 or 0 for every arm and every
# control ever measured -- it has no resolution and should never have been the deliverable.
#
# WHY REPLICATE RATHER THAN RESTORE. The checkpoints are gone. More importantly the original
# was ONE SEED PER ARM, and three nominally identical flat controls on these same clips span
# 0.3434 / 0.3524 / 0.3650 (sd 0.011). A single -0.077 is ~5-7 sd against that spread, but it
# is two arms, not two seeds. Three seeds per arm settles it.
#
# ======================= EXPECTED RESULTS (decided before running) =======================
# PRIMARY: greedy `inter_asr_dist` on 128 val clips, PAIRED per clip against the same seed's
# own no-plan control, reported raw AND length-adjusted. The analysis script was validated by
# reproducing the audit's numbers on the original data to four decimals before this job existed.
#
#   SUCCESS -- lambda=0.3 wins in at least 2 of 3 seed pairings, mean raw delta <= -0.04, and
#       the length-adjusted delta also negative. Then the plan vector is real, it was closed in
#       error, and it becomes the main line.
#   FAILURE -- mean delta inside +/-0.02 (the measured run-to-run floor). The original was a
#       lucky seed and the null stands, now on the right metric.
#   AMBIGUOUS -- raw win but length-adjusted delta near zero. Then the plan only made the model
#       say less, which is the coarse-to-fine story again and not a planning result.
#
# MUST BE REPORTED ALONGSIDE, NOT BURIED: lexical diversity. In the original, lambda=0.3 bought
# its gain partly by retreating into stock phrases -- type/token 0.253 -> 0.214, unique openings
# 23 -> 20. A score improvement that comes entirely from saying safer things is not the win it
# looks like, and the comparison script prints these next to the effect.
#
# ALSO CHECKED, because the previous post-mortem turned on it: the plan head's BCE against a
# SCENE-BLIND per-code marginal (0.2073 on this data). Last time the head reached 0.2033 --
# beating "ignore the ambient entirely" by 0.0040 nats. If it lands there again while the audio
# still improves, the plan is NOT working by carrying scene information, and whatever it is
# doing (a regulariser? a drift brake?) is the actual finding and needs its own investigation.
#
# CONFOUND DELIBERATELY REMOVED: every arm renders through the SAME frozen renderer
# (rladder ep119). Job 444750 compared hier arms rendered through rladder against flat arms
# rendered through content_concat_v4, and that confound inflated its effect by ~3x.
# =========================================================================================
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_plan_replicate.sbatch
#SBATCH --job-name=salm-a2s-plan-rep
#SBATCH --account=aip-csubakan
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

K="${K:-2048}"
EPOCHS="${EPOCHS:-100}"
LAMBDA="${LAMBDA:-0.3}"
SEEDS="${SEEDS:-0 1 2}"
NGEN="${NGEN:-128}"
MAX_MINUTES="${MAX_MINUTES:-70}"
CB="$OUTPUTS_ROOT/content_codebook/codebook_k${K}.pt"
CTC_CKPT="$OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt"
RENDERER="$RUNS_ROOT/a2s_audioldm_clapseq_xformer/rladder_20260906_1654/checkpoints/checkpoint_ep119.pt"
STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"
OUT_ROOT="$OUTPUTS_ROOT/generations/planrep_${STAMP}"

for F in "$CB" "$CTC_CKPT" "$RENDERER"; do
    [[ -f "$F" ]] || { echo "MISSING: $F" >&2; exit 1; }
done
for S in $SEEDS; do
  for A in noplan p${LAMBDA}; do
    d="$OUTPUTS_ROOT/content_lm/planrep_${A}_s${S}"
    [[ -e "$d" ]] && { echo "ABORT: $d exists -- refusing to overwrite" >&2; exit 1; }
  done
done
FREE=$(df -Pm /scratch/l/lovenya | awk 'NR==2{print $4}')
echo "free space: ${FREE} MB (need >= 4000: ~1.4 GB checkpoints + ~750 MB wavs)"
[[ "$FREE" -ge 4000 ]] || { echo "ABORT: only ${FREE} MB free" >&2; exit 1; }

salm_print_header "PLAN VECTOR REPLICATION: 3 seeds x {no-plan, lambda=$LAMBDA}, K=$K"
echo "renderer FROZEN at rladder ep119 for EVERY arm -- the 444750 confound, removed"

echo
echo "==================== 0. THE SCENE-BLIND BAR, stated before training ===================="
# LEARNINGS: an auxiliary head must be scored against a baseline that ignores the conditioning
# input, and that baseline must be stated BEFORE the run, not computed afterwards to explain a
# result. Computed here from the training targets alone.
python - "$K" <<'PY' || true
import sys, numpy as np, torch
sys.path.insert(0, "src")
from salm.a2s.content_lm_dedup import plan_targets
K = int(sys.argv[1])
z = np.load(f"/scratch/l/lovenya/salm_model_outputs/content_codes_dedup/dedup_k{K}_train_r32.npz")
tgt = plan_targets(torch.from_numpy(z["tokens"][:4000]), torch.from_numpy(z["lengths"][:4000]), K)
dens = tgt.mean().item()
marg = tgt.mean(0, keepdim=True).expand_as(tgt).clamp(1e-6, 1 - 1e-6)
bce = torch.nn.functional.binary_cross_entropy(marg, tgt).item()
print(f"  bag-of-codes density {dens:.4f} ({tgt.sum(1).mean():.0f} of {K} codes per utterance)")
print(f"  SCENE-BLIND per-code marginal BCE: {bce:.4f}  <- the bar the plan head must beat")
print(f"  (job 443751's head reached 0.2033, i.e. +0.0040 over blind -- essentially nothing)")
PY

echo
echo "==================== 1. TRAIN (6 arms, 4 GPUs, two waves) ===================="
i=0
for S in $SEEDS; do
  for A in noplan p${LAMBDA}; do
    NAME="planrep_${A}_s${S}"
    EXTRA=""
    [[ "$A" != "noplan" ]] && EXTRA="--use-plan --lambda-plan $LAMBDA"
    # save-every 0: only best_loss and last. The epoch spread is not the question here and
    # /scratch is tight.
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$((i % 4)) python -u scripts/a2s/train_content_lm_dedup.py \
        --codebook "$CB" --run-name "$NAME" --seed "$S" \
        --epochs "$EPOCHS" --batch-size 64 \
        --d-model 512 --layers 6 --heads 8 --dropout 0.1 --weight-decay 0.01 \
        --max-run 32 --save-every 0 --max-minutes "$MAX_MINUTES" \
        $EXTRA > "$OUTPUTS_ROOT/logs/slurm/planrep_${NAME}_${STAMP}.log" 2>&1 &
    i=$((i + 1))
    [[ $((i % 4)) -eq 0 ]] && wait      # 4 GPUs, so 4 at a time
  done
done
wait
echo "training done"

echo
echo "==================== 2. DID THE PLAN HEAD BEAT SCENE-BLIND? ===================="
python - "$OUTPUTS_ROOT" "$LAMBDA" $SEEDS <<'PY' || true
import json, sys
from pathlib import Path
R = Path(sys.argv[1]) / "content_lm"; lam = sys.argv[2]
print(f"  {'run':22s} {'ep':>4s} {'val_loss':>9s} {'plan BCE':>9s} {'vs blind 0.2073':>16s}")
for s in sys.argv[3:]:
    for a in ("noplan", f"p{lam}"):
        h = R / f"planrep_{a}_s{s}" / "history.json"
        if not h.exists():
            print(f"  planrep_{a}_s{s}: no history"); continue
        hist = json.loads(h.read_text())
        lo = min(hist, key=lambda e: e["val_loss"])
        p = lo.get("train_plan_loss", 0.0)
        note = "" if a == "noplan" else f"{0.2073 - p:+16.4f}"
        print(f"  planrep_{a}_s{s:<8s} {lo['epoch']:4d} {lo['val_loss']:9.4f} {p:9.4f} {note}")
print()
print("  A plan head that only ties the blind baseline is NOT carrying scene information.")
print("  If the audio still improves, the mechanism is something else and that is the finding.")
PY

echo
echo "==================== 3. GENERATE (greedy, 128 clips, matched) ===================="
i=0
for S in $SEEDS; do
  for A in noplan p${LAMBDA}; do
    NAME="planrep_${A}_s${S}"
    LM="$OUTPUTS_ROOT/content_lm/$NAME/checkpoints/checkpoint_best_loss.pt"
    [[ -f "$LM" ]] || { echo "  skip (absent): $NAME" >&2; continue; }
    d="$OUT_ROOT/$NAME"; mkdir -p "$d"
    CUDA_VISIBLE_DEVICES=$((i % 4)) python -m salm.generate --mode a2s --direction a2s \
        --checkpoint "$RENDERER" --content-source codebook_lm --content-lm-checkpoint "$LM" \
        --content-lm-temperature 1.0 --content-lm-top-k 1 --ctc-checkpoint "$CTC_CKPT" \
        --split val --num-samples "$NGEN" --start-index 0 --seed 1337 \
        --out-dir "$d" --copy-reference > "$d.log" 2>&1 &
    i=$((i + 1))
    [[ $((i % 4)) -eq 0 ]] && wait
  done
done
wait
echo "generation done"

echo
echo "==================== 4. SCORE ===================="
CTRL="$OUTPUTS_ROOT/generations/control_lanes_20260831_0735"
DIRS="$CTRL/real_speech $CTRL/vae_roundtrip"
for d in "$OUT_ROOT"/*/; do [[ -f "$d/manifest.jsonl" ]] && DIRS="$DIRS ${d%/}"; done
SCORES="$OUTPUTS_ROOT/planrep_intel_${STAMP}.json"
# shellcheck disable=SC2086
python scripts/a2s/score_intelligibility.py --dirs $DIRS --ctc-checkpoint "$CTC_CKPT" \
    --device cuda --out "$SCORES" \
    || { echo "SCORING FAILED -- generations intact under $OUT_ROOT" >&2; exit 1; }

echo
echo "==================== 5. PAIRED, LENGTH-ADJUSTED (the deliverable) ===================="
for S in $SEEDS; do
    echo; echo "  --- seed $S ---"
    python scripts/a2s/compare_arms.py --scores "$SCORES" \
        --control "planrep_noplan_s${S}" --treatment "planrep_p${LAMBDA}_s${S}" \
        --out "$OUTPUTS_ROOT/planrep_compare_s${S}_${STAMP}.json" || true
done

echo
echo "==================== 6. ACROSS SEEDS ===================="
python - "$OUTPUTS_ROOT" "$STAMP" $SEEDS <<'PY' || true
import json, sys, numpy as np
from pathlib import Path
R, stamp = Path(sys.argv[1]), sys.argv[2]
raw, adj = [], []
for s in sys.argv[3:]:
    f = R / f"planrep_compare_s{s}_{stamp}.json"
    if not f.exists(): continue
    r = json.loads(f.read_text())[0]
    raw.append(r["raw_delta"]); adj.append(r["length_adjusted"])
    print(f"  seed {s}: raw {r['raw_delta']:+.4f}  len-adj {r['length_adjusted']:+.4f}  "
          f"{r['clips_better']}/{r['n']} clips better")
if raw:
    print(f"\n  MEAN across {len(raw)} seeds: raw {np.mean(raw):+.4f}  "
          f"len-adj {np.mean(adj):+.4f}  (sd {np.std(raw):.4f})")
    print(f"  wins in {sum(1 for x in raw if x < 0)}/{len(raw)} seed pairings")
    print()
    if np.mean(raw) <= -0.04 and sum(1 for x in raw if x < 0) >= 2 and np.mean(adj) < 0:
        print("  -> SUCCESS by the pre-registered criteria. The plan vector is real and was")
        print("     closed in error. Check the diversity columns before celebrating.")
    elif abs(np.mean(raw)) < 0.02:
        print("  -> FAILURE: inside the measured run-to-run floor. The original was a lucky")
        print("     seed and the null stands, now on the right metric.")
    else:
        print("  -> AMBIGUOUS. Read the length-adjusted column: a raw win with len-adj near")
        print("     zero means the plan only made the model say less.")
PY

python scripts/a2s/build_listening_bundle.py --gen-root "$OUT_ROOT" \
    --out "$OUTPUTS_ROOT/listening_bundles/planrep_${STAMP}" --max-clips 10 --require-all-lanes || true

echo
echo "=========================================================="
echo "listen: $OUTPUTS_ROOT/listening_bundles/planrep_${STAMP}"
echo "READ: section 6 for the verdict, section 2 for whether the plan head learned anything"
echo "scene-specific, and the diversity columns in section 5 before treating a win as a win."
echo "=========================================================="
echo "OK"
