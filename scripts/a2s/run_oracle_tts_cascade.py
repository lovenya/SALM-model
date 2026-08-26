"""
Rank 1 of the F1/F2 plan (2026-08-26b): render REAL target transcripts with off-the-shelf TTS,
to establish the F2 (phonetic realizability) ceiling independent of F1 (content origination).

WHY THIS EXISTS. The oracle-conditioning ablation (2026-08-24) proved F2 is broken in our own
pipeline even with a perfect CONTENT-EMBEDDING. This script asks a different, complementary
question: what does F2 look like with the REAL WORDS as an explicit TEXT input, using models
built specifically for text-to-speech? That is the ceiling any A2S system's rendering half
should be measured against -- Kokoro-82M is the reliability reference (small, fast, was NEVER
part of AudioLDM's lineage, so its number is an independent ceiling), audioldm2-gigaspeech is
the same-lineage comparison (does the family our own model comes from actually solve F2 when
given the words, and how good is its own realizability).

Pre-registered predictions, written before any number exists (from the 2026-08-26b research):
Kokoro: inter-ASR distance ~0.1-0.2 (see scripts/a2s/score_intelligibility.py), real_word_rate
high. audioldm2-gigaspeech: worse and higher-variance -- the paper reports no WER for TTS at
all, and its own README warns about seed/hardware sensitivity.

CRITICAL TRAP, verified against the actual downloaded checkpoint (not assumed): the shipped
language_model/config.json has "max_new_tokens": 8 -- a leftover from the text-to-audio
conversion. TTS needs 512 (verified: AudioLDM2's own paper uses lambda=1, 512 LOA tokens per
10s, for speech specifically). Passing max_new_tokens=8 at call time silently reproduces
OUR OWN current failure mode inside their model -- an ~8-token generation cannot carry a
full utterance's content, so a wrong run here would falsely look like AudioLDM2-speech also
fails at F2, when actually the call was just misconfigured.

Kokoro is optional at import time (needs `pip install kokoro misaki[en]`, not yet always
present) -- this script runs whatever's available and reports clearly what was skipped, rather
than hard-failing the whole job over one missing package.

Usage:
    python scripts/a2s/run_oracle_tts_cascade.py --split val --num-samples 16 --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from configs import paths as project_paths  # noqa: E402
from salm.data.manifest import load_manifest  # noqa: E402

AUDIOLDM2_SR = 16000     # verified: matches our own pipeline, no resample needed for scoring
KOKORO_SR = 24000
AUDIOLDM2_GIGASPEECH_ID = "anhnct/audioldm2_gigaspeech"
KOKORO_MODEL_ID = "hexgrad/Kokoro-82M"


def log(msg: str) -> None:
    print(msg, flush=True)


def run_audioldm2_gigaspeech(rows, out_dir: Path, device: str, seed: int) -> list[dict]:
    from diffusers import AudioLDM2Pipeline

    log("Loading audioldm2_gigaspeech ...")
    # fp16 on CUDA, fp32 elsewhere -- matches the existing convention in salm/generate.py.
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    pipe = AudioLDM2Pipeline.from_pretrained(
        AUDIOLDM2_GIGASPEECH_ID, torch_dtype=dtype
    ).to(device)

    manifest_rows = []
    generator = torch.Generator(device=device).manual_seed(seed)
    for row in rows:
        # max_new_tokens=512 is MANDATORY, not the checkpoint's own default (8) -- see module
        # docstring. A silent default here would misconfigure the model, not fail loudly.
        audio = pipe(
            prompt="",
            transcription=row.speech_text,
            audio_length_in_s=10.24,
            num_inference_steps=200,
            max_new_tokens=512,
            generator=generator,
        ).audios[0]
        out_path = out_dir / "audio" / f"{row.sample_id}.wav"
        sf.write(out_path, audio, AUDIOLDM2_SR)
        manifest_rows.append({
            "sample_id": row.sample_id, "status": "success",
            "generated_file_rel": str(out_path.relative_to(out_dir)),
            "speech_text": row.speech_text,
        })
    return manifest_rows


def run_kokoro(rows, out_dir: Path, device: str) -> list[dict] | None:
    try:
        from kokoro import KPipeline
    except ImportError:
        log("SKIPPED: `kokoro` not installed (pip install kokoro misaki[en]). "
            "AudioLDM2-gigaspeech lane still runs.")
        return None

    log("Loading Kokoro-82M ...")
    kpipe = KPipeline(lang_code="a", device=device)  # 'a' = American English
    # `voice` is REQUIRED -- verified against installed source (KPipeline.__call__ raises
    # ValueError("Specify a voice...") if None). af_heart is the exact example the source's
    # own error message uses for lang_code='a'; not otherwise a principled choice, and worth
    # revisiting once we care about voice diversity rather than a single F2 ceiling number.
    voice = "af_heart"

    manifest_rows = []
    for row in rows:
        # KPipeline.Result implements __iter__ for backward compat, yielding EXACTLY
        # (graphemes, phonemes, audio) -- verified against installed source, not assumed.
        # It is a generator over chunks for long text; our utterances are short (mean
        # 4.86s), so one chunk is expected -- concatenate defensively in case Kokoro still
        # splits it, and skip any chunk whose audio came back None (KPipeline.Result.audio
        # is None when a segment's inference was skipped, e.g. an empty/whitespace-only
        # split -- silently including that would corrupt the concatenated waveform).
        chunks = [audio for _, _, audio in kpipe(row.speech_text, voice=voice) if audio is not None]
        if not chunks:
            log(f"    WARNING: no audio produced for {row.sample_id!r}, skipping")
            continue
        audio = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
        audio = audio.numpy() if torch.is_tensor(audio) else np.asarray(audio)
        out_path = out_dir / "audio" / f"{row.sample_id}.wav"
        sf.write(out_path, audio, KOKORO_SR)
        manifest_rows.append({
            "sample_id": row.sample_id, "status": "success",
            "generated_file_rel": str(out_path.relative_to(out_dir)),
            "speech_text": row.speech_text,
        })
    return manifest_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-audioldm2", action="store_true")
    ap.add_argument("--skip-kokoro", action="store_true")
    args = ap.parse_args()

    rows = load_manifest("a2s", args.split)[: args.num_samples]
    log(f"rendering {len(rows)} REAL target transcripts (oracle content, F2-ceiling probe)")

    out_root = Path(args.out_dir)
    t0 = time.time()

    if not args.skip_audioldm2:
        d = out_root / "audioldm2_gigaspeech"
        (d / "audio").mkdir(parents=True, exist_ok=True)
        rows_out = run_audioldm2_gigaspeech(rows, d, args.device, args.seed)
        (d / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows_out) + "\n")
        log(f"    wrote {len(rows_out)} clips to {d}")

    if not args.skip_kokoro:
        d = out_root / "kokoro_82m"
        (d / "audio").mkdir(parents=True, exist_ok=True)
        rows_out = run_kokoro(rows, d, args.device)
        if rows_out is not None:
            (d / "manifest.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows_out) + "\n"
            )
            log(f"    wrote {len(rows_out)} clips to {d}")

    log(f"\nelapsed {(time.time()-t0)/60:.1f} min")
    log(f"Next: score both with scripts/a2s/score_intelligibility.py --dirs "
        f"{out_root}/audioldm2_gigaspeech {out_root}/kokoro_82m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
