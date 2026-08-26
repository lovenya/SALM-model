#!/usr/bin/env bash
# Oracle-conditioning ablation for A2S -- decomposes conditioner error from target quality.
#
# WHAT THIS ANSWERS. A1 (job 425738, +O2, plain cosine) reached val_clap_r1=0.327, well
# above chance but well below S2A's 0.564. Two very different explanations are consistent
# with that gap, and they point to opposite next steps:
#   (a) the CONDITIONER underfits a good target      -> fix architecture/loss/training
#   (b) the TARGET (CLAP-text) or the U-Net's ability to render it is the ceiling
#       -> fix the target/interface itself, no amount of conditioner tuning helps
#
# This generates with --oracle-align text, which skips the conditioner ENTIRELY and feeds
# the row's own true align_text vector straight to class_labels. If oracle generations are
# ALSO not word-accurate (checked by ear / by a future WER pass), that is direct evidence
# for (b), independent of how well the conditioner ever learns to predict that vector.
#
# Short and single-GPU generation only -- no training. Requests a whole node because TamIA
# only allocates h100 by node, same constraint as run_tamia_s2a_generate_only.sbatch.
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_oracle_ablation.sbatch
#   CKPT=/path/to/other/checkpoint_best.pt sbatch submission_scripts/run_tamia_a2s_oracle_ablation.sbatch
#SBATCH --job-name=salm-a2s-oracle
#SBATCH --account=aip-csubakan
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

# A1 is the only A2S checkpoint an oracle ablation is informative for -- it is the only one
# trained WITH an alignment target at all (A0 never used lambda_align, so there is no
# target its conditioner was ever asked to approach, and "oracle" would just mean "some
# vector unrelated to what A0 optimized").
CKPT="${CKPT:-$RUNS_ROOT/a2s_audioldm_clapseq_xformer/probe40_A1_o2/checkpoints/checkpoint_best.pt}"
NUM_GEN="${NUM_GEN:-16}"
SPLIT="${SPLIT:-val}"
STEPS="${STEPS:-100}"
GUIDANCE="${GUIDANCE:-3.0}"
RUN_ID="$(TZ=America/Toronto date +%Y%m%d_%H%M)"

if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: checkpoint not found: $CKPT" >&2
    exit 1
fi
echo "checkpoint: $CKPT"
ls -la "$CKPT"

GEN_DIR="$OUTPUTS_ROOT/generations/a2s_oracle_ablation_${RUN_ID}"
LOG_DIR="$OUTPUTS_ROOT/logs/salm_model/a2s_oracle_ablation_${RUN_ID}"
mkdir -p "$GEN_DIR" "$LOG_DIR"

salm_print_header "A2S oracle-conditioning ablation: n=$NUM_GEN split=$SPLIT"
salm_start_monitor "$LOG_DIR" "a2s_oracle_${RUN_ID}" 30

# Two lanes on separate GPUs: the oracle (conditioner bypassed) and, for a same-job direct
# comparison, the REAL conditioner's output on the identical rows/seed. The real lane's
# audio already exists under the source run's generations/a2s_best/, but re-running it here
# with IDENTICAL --seed/--steps/--guidance-scale as the oracle lane removes sampling-noise
# as a confound when comparing the two by ear.
CUDA_VISIBLE_DEVICES=0 python -m salm.generate \
    --mode a2s --direction a2s --checkpoint "$CKPT" \
    --oracle-align text \
    --split "$SPLIT" --num-samples "$NUM_GEN" --steps "$STEPS" --guidance-scale "$GUIDANCE" \
    --out-dir "$GEN_DIR/oracle_text" --copy-reference \
    > "$LOG_DIR/01_gen_oracle.log" 2>&1 &
GEN_A=$!
CUDA_VISIBLE_DEVICES=1 python -m salm.generate \
    --mode a2s --direction a2s --checkpoint "$CKPT" \
    --split "$SPLIT" --num-samples "$NUM_GEN" --steps "$STEPS" --guidance-scale "$GUIDANCE" \
    --out-dir "$GEN_DIR/real_conditioner" --copy-reference \
    > "$LOG_DIR/02_gen_real.log" 2>&1 &
GEN_B=$!
GEN_FAIL=0
wait "$GEN_A" || GEN_FAIL=1
wait "$GEN_B" || GEN_FAIL=1
tail -n 5 "$LOG_DIR/01_gen_oracle.log" "$LOG_DIR/02_gen_real.log"

echo
echo "==================== ORACLE ABLATION SUMMARY ===================="
python - <<PY
import json, pathlib
for name in ("oracle_text", "real_conditioner"):
    s = pathlib.Path("$GEN_DIR") / name / "summary.json"
    if s.exists():
        d = json.loads(s.read_text())
        print(f"{name:16s} oracle_align={d.get('oracle_align')!r:10s} "
              f"rms={d['rms_mean']:.5f} near_silent={d['near_silent_count']}/{d['num_samples']}")
    else:
        print(f"{name:16s} MISSING summary.json")
PY
echo "==================================================================="
echo "listen: $GEN_DIR/oracle_text/audio/  vs  $GEN_DIR/real_conditioner/audio/"
echo "        each *__reference.wav is the TRUE target speech for that row"
echo "==================================================================="
[[ "$GEN_FAIL" == "0" ]] || { echo "ERROR: a generation lane failed" >&2; exit 1; }
echo "OK"
