#!/usr/bin/env bash
# COSYVOICE TOKEN-SPACE GATE. Pure inference, no training, writes one JSON. ~20 minutes.
#
# WHY. Measured 2026-09-06 from the dedup runs' own history: `scene_gap` accounts for 1.8-2.8%
# of the content LM's val loss. The other 97%+ is LANGUAGE MODELLING over code sequences,
# learned from 17,596 utterances. That is why every conditioning intervention landed where it
# did -- CFG (442847) bought two words, the plan vector (443751) was null, guidance saturated.
# They were trying to extract information the scene does not contain.
#
# So the live variable is the token space itself, and CosyVoice3 is the one candidate reachable
# without a download: its tokenizer is SINGLE-STREAM (verified from the ONNX signature -- one
# `indices` output, not the 8-32 parallel codebooks an RVQ acoustic codec emits, which would
# need a coarse/fine cascade and a full pipeline rebuild), runs at 25 Hz matching our latent
# rate exactly, and ships `llm.pt`, a language model pretrained over that token space. That
# last part is the point: it is the prior the 97% measurement says we lack, and we cannot get
# it any other way -- `datasets/` holds only TTAS (22,793 clips, 17,596 already train), so
# pretraining on unpaired speech would require acquiring a corpus.
#
# ======================= EXPECTED RESULTS (decided before running) =======================
# NOT MEASURED: resynthesis quality. CosyVoice is a shipped TTS system and its vocoder works;
# testing that would answer a question nobody is asking. The open question is whether ITS
# tokens are easier to MODEL than ours.
#
# THE METRIC: both token streams are extracted for the SAME utterances, and each is scored on
#
#     (unigram loss - trigram loss) / unigram loss
#
# the share of its own uncertainty that context removes. Absolute losses are NOT comparable
# across streams with different vocabulary sizes; this relative reduction is. It is the same
# discipline the plan-vector post-mortem forced: score against the baseline that ignores the
# thing being tested. Validated on synthetic streams before submission -- i.i.d. noise scores
# -0.7%, a deterministic period-8 cycle scores +98.7%, and an i.i.d. stream with vocab 4096
# scores +0.7%, so a large vocabulary cannot fake structure.
#
#   COSYVOICE CLEARLY HIGHER (>5 points) -- its tokens carry more exploitable structure, an LM
#       over them is a better-posed problem, and llm.pt is a pretrained prior for exactly that
#       space. Next step is the resynthesis gate (needs `uv pip install hyperpyyaml matcha-tts`)
#       and then an ambient->CosyVoice-token model. This is the outcome that would redirect the
#       project, so it needs the second gate before anything is rebuilt.
#   OURS HIGHER (>5 points) -- swapping would make the LM's job harder. The direction closes on
#       one cheap job, and the codebook work (D3) inherits the attention instead.
#   COMPARABLE (within 5 points) -- the representation is not the lever. Any gain from
#       CosyVoice would have to come from llm.pt's pretrained WEIGHTS rather than the token
#       space, which is a much narrower claim and a much bigger engineering commitment. I would
#       not take it on this evidence alone.
#
#   WATCH: token count per clip. CosyVoice at 25 Hz over 10.24 s should give ~256, matching our
#   grid. A materially different length means the rate assumption is wrong and the comparison
#   is not like-for-like -- check that before reading anything else.
# =========================================================================================
#
# Usage:
#   sbatch submission_scripts/run_tamia_a2s_cosyvoice_gate.sbatch
#SBATCH --job-name=salm-a2s-cosy-gate
#SBATCH --account=aip-csubakan
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/l/lovenya/salm_model_outputs/logs/slurm/%x-%j.out

set -euo pipefail

# shellcheck disable=SC1091
source /scratch/l/lovenya/SALM-model/submission_scripts/_salm_env.sh

NUM_TRAIN="${NUM_TRAIN:-2000}"
NUM_VAL="${NUM_VAL:-500}"
K="${K:-2048}"
STAMP="$(TZ=America/Toronto date +%Y%m%d_%H%M)"
TOK="/scratch/l/lovenya/salm_migration_clean/models/cascaded/tts/fun-cosyvoice3-0.5b-2512/speech_tokenizer_v3.onnx"

[[ -f "$TOK" ]] || { echo "MISSING tokenizer: $TOK" >&2; exit 1; }
[[ -f "$OUTPUTS_ROOT/content_codes/codes_codebook_k${K}_train.npy" ]] \
    || { echo "MISSING our cached codes for K=$K" >&2; exit 1; }
python -c "import onnxruntime" 2>/dev/null \
    || { echo "MISSING onnxruntime -- run: uv pip install onnxruntime-gpu" >&2; exit 1; }

salm_print_header "COSYVOICE TOKEN GATE: is their token space easier to model than ours?"
echo "writes one JSON and no audio -- /scratch is at 98% and this job adds nothing to it"

python scripts/a2s/cosyvoice_token_gate.py \
    --num-train "$NUM_TRAIN" --num-val "$NUM_VAL" --k "$K" \
    --out "$OUTPUTS_ROOT/cosyvoice_token_gate_${STAMP}.json"

echo
echo "=========================================================="
echo "READ: the RELATIVE reduction column, not the absolute losses -- the two streams have"
echo "different vocabulary sizes and only the relative figure is comparable."
echo "First check token count per clip is ~256; if not, the 25 Hz assumption is wrong and"
echo "nothing below is like-for-like."
echo "=========================================================="
echo "OK"
