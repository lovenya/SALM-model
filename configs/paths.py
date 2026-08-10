"""
Path registry for SALM-model.

This repo is clean-room: we do NOT import code from the SpeechAudioLM repo.
We DO read its data (manifests + audio/speech files) by absolute path, since
that repo is where dataset cleaning/filtering/splitting already happened.

Keep this file dumb: plain path constants + short comments. No logic,
no argparse, no env-var magic. Training/eval scripts import from here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# External repos / roots (read-only from this repo's point of view)
# ---------------------------------------------------------------------------

# The data-prep + eval repo. We only ever READ manifests/paths from here.
SPEECHAUDIOLM_REPO_ROOT = Path("/scratch/l/lovenya/SpeechAudioLM")

# Where the actual TTAS audio/*.wav and speech/*.wav files live. All
# `audio_file` / `speech_file` values in the manifests are relative to this.
TTAS_DATASET_ROOT = Path(
    "/scratch/l/lovenya/salm_migration_clean/datasets/ttas/extracted"
)

# ---------------------------------------------------------------------------
# Manifests (final, model-training inputs)
#
# clean_v2 = G-Eval audio v6 filtered manifests. IMPORTANT: a2s and s2a were
# filtered INDEPENDENTLY per direction, so they are NOT the same set of
# (audio, speech) pairs -- a row can pass the audio->speech direction's score
# threshold and fail the speech->audio direction's (or vice versa). See
# CONTEXT.md "Data" section for the audited overlap numbers.
# ---------------------------------------------------------------------------

_MANIFEST_ROOT = SPEECHAUDIOLM_REPO_ROOT / "data" / "manifests" / "clean_v2"

A2S_MANIFESTS = {
    "train": _MANIFEST_ROOT / "a2s" / "ttas_train.jsonl",
    "val": _MANIFEST_ROOT / "a2s" / "ttas_val.jsonl",
    "test": _MANIFEST_ROOT / "a2s" / "ttas_test.jsonl",
}

S2A_MANIFESTS = {
    "train": _MANIFEST_ROOT / "s2a" / "ttas_train.jsonl",
    "val": _MANIFEST_ROOT / "s2a" / "ttas_val.jsonl",
    "test": _MANIFEST_ROOT / "s2a" / "ttas_test.jsonl",
}

# ---------------------------------------------------------------------------
# Model assets (all local; compute nodes have no internet)
# ---------------------------------------------------------------------------

# HuggingFace cache root. Must be exported as HF_HOME before loading anything,
# or transformers/diffusers will try to reach the network and fail on a
# compute node.
HF_HOME = Path("/scratch/l/lovenya/salm_migration_clean/models/hf_cache")

# AudioLDM-S full v2: mel VAE + U-Net + CLAP text tower + HiFi-GAN vocoder.
# The "-s-" (small, ~180M U-Net) variant is deliberate: with 15k training pairs
# and a short wall-clock budget, the large variant costs more per step without a
# guaranteed quality win. Swapping to `cvssp/audioldm-l-full` is a one-line change.
AUDIOLDM_MODEL_ID = "cvssp/audioldm-s-full-v2"

# CLAP audio tower. Must be the NON-fusion ("unfused") variant: AudioLDM was
# trained against non-fusion HTSAT-base CLAP, and the locally cached
# 630k-audioset-FUSION-best.pt has a different architecture and embedding space.
# Using the fusion checkpoint would silently break the premise that we are aiming
# at the conditioning space the U-Net already understands.
CLAP_MODEL_ID = "laion/clap-htsat-unfused"

# Frozen speech encoders.
WHISPER_LARGE_V3_PATH = Path(
    "/scratch/l/lovenya/salm_migration_clean/models/cascaded/asr/whisper-large-v3"
)
QWEN25_OMNI_3B_PATH = Path(
    "/scratch/l/lovenya/salm_migration_clean/models/qwen2.5-omni-3b"
)
# Reference implementation of the frozen-MLLM conditioning pattern we mirror
# (thinker hidden states at layer_idx=-2 + zero-init linear adapter).
AUDIO_OMNI_CODE_ROOT = Path(
    "/scratch/l/lovenya/salm_migration_clean/code/Audio-Omni"
)

# ---------------------------------------------------------------------------
# This repo's own roots
# ---------------------------------------------------------------------------

SALM_MODEL_REPO_ROOT = Path(__file__).resolve().parents[1]

# Heavy artifacts live outside git, in a root dedicated to THIS repo.
# Deliberately not nested under salm_migration_clean: that tree belongs to the
# SpeechAudioLM repo (its datasets, metric caches, venvs), and mixing our training
# outputs into it makes both harder to reason about and to clean up.
OUTPUTS_ROOT = Path("/scratch/l/lovenya/salm_model_outputs")

# Precomputed VAE latents for the ambient targets, so mel extraction and VAE
# encoding happen once instead of every epoch.
FEATURES_ROOT = OUTPUTS_ROOT / "features"
# CLAP audio embeddings of the target clips. These are the regression target for the
# in-loop val cosine metric, which exists because the diffusion loss is flat and
# therefore useless as a progress signal.
FEATURES_CLAP_ROOT = OUTPUTS_ROOT / "features_clap"

# One directory per run, grouped by architecture: runs/<arch>/<run_name>/
RUNS_ROOT = OUTPUTS_ROOT / "runs"
# Slurm's own %x-%j.out files, so nothing this repo produces lands in
# salm_migration_clean.
SLURM_LOGS_ROOT = OUTPUTS_ROOT / "logs" / "slurm"


def run_dir(arch: str, run_name: str) -> Path:
    """Canonical run directory. Everything for one run lives under here."""
    return RUNS_ROOT / arch / run_name


# ---------------------------------------------------------------------------
# Legacy roots (pre-2026-08-06). Kept only so old artifacts remain findable;
# do not write here.
# ---------------------------------------------------------------------------

_LEGACY_RUNS_ROOT = Path("/scratch/l/lovenya/salm_migration_clean/salm_model")

# ---------------------------------------------------------------------------
# Training environment (see configs/runtime_environments.json)
# ---------------------------------------------------------------------------

TRAIN_VENV = Path(
    "/scratch/l/lovenya/salm_migration_clean/.venvs/salm-model"
    "/.venv-salm-train-audioldm-py312"
)
