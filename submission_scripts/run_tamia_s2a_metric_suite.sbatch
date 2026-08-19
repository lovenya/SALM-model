#!/usr/bin/env bash
# FAD / KAD / KL-PaSST / IS / caption-CLAP on the TEST split, via the SpeechAudioLM metric suite.
#
# WHY: these are the metrics an ICASSP reviewer looks at first, and we have had ZERO of them for the
# entire project. Everything so far is our own CLAP-based MARGIN/audio_r1, which is (a) not
# comparable to any published number and (b) partly self-referential, since O2 trains against CLAP.
# FAD here uses PANNs, which DCASE 2024 found correlates with human ratings at rho = -0.94, above
# both CLAP and VGGish.
#
# HOW THE INTERFACE WORKS (learned by reading that repo and dry-running; do not guess at it):
#   * `--manifest` IS the row source, despite its help text saying "baseline". Our exported manifest
#     is what gets evaluated.
#   * EVERY key of our `input` block becomes a key of the evaluator's `source` dict
#     (evaluation/io.py:_v1_source copies record["input"] wholesale for schema_version 1). A
#     top-level "source" key is IGNORED. So the reference clip must be `input.audio_file`; FAD/KAD
#     look for `source.audio_file`. Getting this wrong rejected all 1,878 rows.
#   * ONE METRIC PER INVOCATION. runner.py: "Each eval run must contain exactly one metric; split
#     multi-metric commands into per-metric runs." Passing a comma list fails.
#   * Each metric shells out to its own backend venv, resolved from configs/eval_metrics.json
#     ({environment:audioldm_eval}, {environment:kadtk}, {environment:stable_audio_metrics}). Those
#     already exist; nothing needs installing.
#   * We only READ that repo. The single write ever made there is the one-line `salm_e2e`
#     registration in configs/model_capabilities.json, which the user approved.
#
# `prompt_adherence_clap_score` is deliberately excluded: it scores audio against a persisted TEXT
# generation prompt, and direct S2A has no text prompt, so the number would be meaningless.
# The geval_* metrics are excluded too -- they are LLM-judged and need a served model.
#
# WHAT TO EXPECT, so a surprising number gets questioned rather than believed:
#   * retrieval should WIN on FAD while losing on MARGIN -- it returns real recordings, so it sits on
#     the real manifold by construction. That contrast is the point, not an embarrassment.
#   * rung0 (caption-conditioned) should be catastrophically bad: its MARGIN is ~0 and its `matched`
#     CLAP cosine of 0.159 is BELOW the 0.29 random-real-clip floor.
#   * b0 (unconditional) should show poor diversity: its self_sim is 0.73 against 0.29 for real clips.
#
# Usage:
#   sbatch submission_scripts/run_tamia_s2a_metric_suite.sbatch
#   LANES="s2a_s7.5" METRICS="fad" sbatch ...       # single cell, for a quick check
#SBATCH --job-name=salm-s2a-metric-suite
#SBATCH --account=aip-ravanelm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

EVAL_REPO="${EVAL_REPO:-/scratch/l/lovenya/SpeechAudioLM}"
ARCH="${ARCH:-s2a_audioldm_whisper_xformer}"
RUN_NAME="${RUN_NAME:-long100_L2_o2}"
SPLIT="${SPLIT:-test}"
LANES="${LANES:-s2a_s7.5 rung0_caption rung1_clap b0_unconditional}"
METRICS="${METRICS:-fad kad kl_passt inception_score target_caption_clap_score}"
STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"

RUN_DIR="$RUNS_ROOT/$ARCH/$RUN_NAME"
FE="$(ls -1d "$RUN_DIR"/final_eval_*/ 2>/dev/null | tail -1)"
[[ -n "$FE" ]] || { echo "ERROR: no final_eval_* dir under $RUN_DIR" >&2; exit 1; }
LOG_DIR="$RUN_DIR/logs/metric_suite_${STAMP}"
SUM="$RUN_DIR/metric_suite_${STAMP}"
mkdir -p "$LOG_DIR" "$SUM"

echo "==================================================================="
echo "Metric suite: FAD / KAD / KL-PaSST / IS / caption-CLAP"
echo "job:    ${SLURM_JOB_ID:-none}   host: $(hostname)"
echo "commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "gen:    $FE"
echo "split:  $SPLIT"
echo "lanes:  $LANES"
echo "metrics:$METRICS"
echo "==================================================================="

salm_start_monitor "$LOG_DIR" "metric_suite" 120

read -r -a LANE_ARR <<< "$LANES"
read -r -a METRIC_ARR <<< "$METRICS"

# ---------------------------------------------- 1. re-export with the corrected schema
# Cheap (symlinks + json) and idempotent. Done here rather than relying on whatever schema the
# generation job happened to write, so this job is self-contained and reproducible.
echo
echo "### [1/2] Export manifests (input.audio_file is the key that matters) ###"
for lane in "${LANE_ARR[@]}"; do
    [[ -d "$FE$lane" ]] || { echo "  SKIP $lane: not generated"; continue; }
    python scripts/export_eval_manifest.py \
        --generation-dir "$FE$lane" --out-dir "$FE/eval_export/$lane" \
        --stage s2a --model salm_e2e --split "$SPLIT" \
        2>&1 | grep -E "wrote|ERROR" | sed "s|^|  $lane: |"
done

# ---------------------------------------------- 2. one invocation per (lane, metric)
echo
echo "### [2/2] Evaluate ###"
GPU=0
PIDS=()
for lane in "${LANE_ARR[@]}"; do
    MF="$FE/eval_export/$lane/manifest.jsonl"
    [[ -f "$MF" ]] || { echo "  SKIP $lane: no manifest"; continue; }
    # One background worker per lane; it walks its metrics sequentially because each loads a
    # different backend model and they would otherwise contend for the same GPU.
    (
        for metric in "${METRIC_ARR[@]}"; do
            echo "  [$lane] $metric on GPU $GPU"
            # MUST run with cwd = the evaluator's repo root. evaluate.py resolves
            # `configs/model_capabilities.json` RELATIVELY, so invoking it by absolute path from
            # our repo made every one of the 20 invocations die instantly with
            # FileNotFoundError: 'configs/model_capabilities.json' -- the job "completed" in 9 s
            # having computed nothing. A subshell keeps the cd local to this lane.
            ( cd "$EVAL_REPO" && CUDA_VISIBLE_DEVICES=$GPU python evaluate.py \
                --model salm_e2e --stage s2a --direction s2a \
                --evaluated-run-id "salm_e2e_${SPLIT}_${lane}" \
                --split "$SPLIT" --manifest "$MF" --metrics "$metric" \
                > "$LOG_DIR/${lane}__${metric}.log" 2>&1 ) \
                || echo "  WARN [$lane] $metric FAILED (see $LOG_DIR/${lane}__${metric}.log)" >&2
        done
    ) &
    PIDS+=($!)
    GPU=$(( (GPU + 1) % 4 ))
    if [[ "$GPU" == "0" ]]; then
        for p in "${PIDS[@]}"; do wait "$p" || true; done
        PIDS=()
    fi
done
for p in ${PIDS[@]+"${PIDS[@]}"}; do wait "$p" || true; done

# ---------------------------------------------- collect
echo
echo "### Results ###"
python - "$EVAL_REPO" "$SPLIT" "$SUM" <<'PY'
import json, pathlib, sys, re
repo, split, out = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
root = pathlib.Path("/scratch/l/lovenya/salm_migration_clean/results/evals/stages/s2a/s2a/salm_e2e/ttas") / split
rows = {}
for p in root.glob("*/*/*/**/*.json"):
    m = re.search(rf"{split}/([^/]+)/([^/]+)/", str(p))
    if not m:
        continue
    run_id, metric = m.group(1), m.group(2)
    lane = run_id.replace(f"salm_e2e_{split}_", "")
    try:
        d = json.loads(p.read_text())
    except Exception:
        continue
    # Metric summaries nest their number under varying keys; take the first float-ish scalar.
    def find(o):
        if isinstance(o, (int, float)) and not isinstance(o, bool):
            return float(o)
        if isinstance(o, dict):
            for k in ("value", "score", "mean", metric, "result"):
                if k in o:
                    v = find(o[k])
                    if v is not None:
                        return v
            for v in o.values():
                r = find(v)
                if r is not None:
                    return r
        return None
    v = find(d)
    if v is not None:
        rows.setdefault(lane, {}).setdefault(metric, v)

if not rows:
    print("  no metric outputs found -- check the per-(lane,metric) logs")
else:
    metrics = sorted({m for v in rows.values() for m in v})
    hdr = f"  {'lane':20s} " + " ".join(f"{m[:16]:>17s}" for m in metrics)
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for lane in sorted(rows):
        print(f"  {lane:20s} " + " ".join(
            (f"{rows[lane][m]:>17.4f}" if m in rows[lane] else f"{'--':>17s}") for m in metrics))
    (out / "metric_suite.json").write_text(json.dumps(rows, indent=2))
    print(f"\n  wrote {out/'metric_suite.json'}")
print()
print("  Read FAD/KAD alongside MARGIN, not instead of it: retrieval and other real-audio lanes")
print("  can win distributional metrics while being wrong per example.")
PY

echo
echo "logs:    $LOG_DIR"
echo "summary: $SUM"
echo "OK"
