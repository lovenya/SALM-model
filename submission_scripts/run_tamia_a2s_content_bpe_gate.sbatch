#!/usr/bin/env bash
# Lossless acoustic-code BPE COMPRESSION GATE. This does not train a generator and does not
# claim speech quality. It asks whether reversible macro-tokens can reduce the 256-frame
# ContentLM target below the planning horizon that defeated frame-level and run-length forms.
#
# Inputs: the already-built K=512 and K=1024 train/val integer-code caches.  One BPE model is
# learned on TRAIN for each K.  Prefixes at total vocabularies 2,048 / 4,096 / 8,192 are
# evaluated on both splits.  VAL is never used to choose merges.
#
# ========================= PRE-REGISTERED INTERPRETATION =========================
# Primary metric: mean VAL macro-token length at total vocabulary 8,192.
# Safety invariant: every TRAIN and VAL row must survive encode -> decode bit-for-bit.
#
# PASS: both K values have mean VAL length < 80 tokens and all round-trips are exact.
#   Interpretation: BPE supplies >=3.2x lossless shortening, enough to justify a ContentLM
#   predictor over macro-tokens. K=1024 is preferred if it passes because its quantized-oracle
#   renderer is more intelligible than K=512.
# FAILURE: either K remains >=80 mean VAL tokens, while round-trips are exact.
#   Interpretation: local acoustic motifs do not compress enough; do not train another
#   single-level predictor. Move to a semantic/acoustic hierarchy or text/phoneme planner.
# AMBIGUOUS: train passes but val fails, or the 8k curve is still falling steeply.
#   Interpretation: likely motif overfit or an unresolved vocabulary tradeoff. Inspect the
#   2k/4k/8k train-vs-val curve before choosing one narrowly scoped follow-up.
# IMPLEMENTATION FAILURE: any non-exact round-trip, missing cache, traceback, or incomplete
#   report. This says nothing about the hypothesis and must be fixed before interpretation.
# =================================================================================
#
# This is CPU preprocessing, but TamIA grants GPU nodes whole (4x H100). Both independent K
# fits run concurrently to use more of the allocated node; no GPU model is loaded.
#
# Usage (DO NOT submit from this file):
#   sbatch submission_scripts/run_tamia_a2s_content_bpe_gate.sbatch
#SBATCH --job-name=salm-a2s-bpe-gate
#SBATCH --account=aip-csubakan
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SALM_REPO="$REPO"
source "$REPO/submission_scripts/_salm_env.sh"

STAMP="${STAMP:-$(TZ=America/Toronto date +%Y%m%d_%H%M)}"
GATE_ROOT="${GATE_ROOT:-$OUTPUTS_ROOT/content_bpe_gate/bpe_gate_${STAMP}}"
CACHE_ROOT="${CACHE_ROOT:-$OUTPUTS_ROOT/content_codes}"
export CACHE_ROOT

[[ ! -e "$GATE_ROOT" ]] || {
    echo "REFUSING OVERWRITE: $GATE_ROOT already exists" >&2
    exit 1
}
for K in 512 1024; do
    for SPLIT in train val; do
        FILE="$CACHE_ROOT/codes_codebook_k${K}_${SPLIT}.npy"
        [[ -f "$FILE" ]] || { echo "MISSING: $FILE" >&2; exit 1; }
    done
done

mkdir -p "$GATE_ROOT"
salm_print_header "LOSSLESS ACOUSTIC-CODE BPE GATE: K=512/1024, vocab=2k/4k/8k"
salm_start_monitor "$GATE_ROOT/resources" "content_bpe_gate" 60

python - "$GATE_ROOT/job_start.json" <<'PY'
import json
import os
from pathlib import Path
import subprocess
import sys

repo = Path("/scratch/l/lovenya/SALM-model")
payload = {
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "repository": str(repo),
    "commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
    "dirty": bool(subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()),
    "cache_root": os.environ.get("CACHE_ROOT", "/scratch/l/lovenya/salm_model_outputs/content_codes"),
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n")
PY

# Reserve five minutes before Slurm termination for JSON flush, validation, and monitoring
# shutdown. Two independent K fits run in separate worker processes.
timeout --signal=TERM --kill-after=60s 40m \
    python -u scripts/a2s/evaluate_content_bpe_gate.py \
        --ks 512 1024 \
        --target-vocabs 2048 4096 8192 \
        --cache-root "$CACHE_ROOT" \
        --out-dir "$GATE_ROOT/results" \
        --workers 2 \
        --pass-mean-val-tokens 80

SUMMARY="$GATE_ROOT/results/summary.json"
[[ -s "$SUMMARY" ]] || { echo "IMPLEMENTATION FAILURE: missing $SUMMARY" >&2; exit 1; }
python - "$SUMMARY" <<'PY'
import json
from pathlib import Path
import sys

report = json.loads(Path(sys.argv[1]).read_text())
for result in report["results"]:
    for budget in result["budgets"]:
        assert budget["roundtrip"]["exact"] is True
        assert budget["roundtrip"]["train_rows_verified"] == result["train_shape"][0]
        assert budget["roundtrip"]["val_rows_verified"] == result["val_shape"][0]
print("validated complete report; scientific gate:", "PASS" if report["gate"]["overall_pass"] else "FAIL")
PY

echo "summary: $SUMMARY"
echo "Scientific FAIL is a valid result; only missing/non-exact artifacts make this job fail."
