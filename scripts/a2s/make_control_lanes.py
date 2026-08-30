#!/usr/bin/env python3
"""Build POSITIVE-CONTROL lanes for the intelligibility scorer.

Why this exists: every lane ever scored by score_intelligibility.py landed in
0.955-0.993 inter-ASR distance -- a 3.8% range against a ceiling of 1.0 -- and the
script's own docstring pre-registers "on REAL target speech, inter-ASR distance
~0.2-0.3". That control lane was never actually built or run, so there is no way to
tell "no improvement" from "improvement below the metric's resolution".

Note the pre-registered floor is itself optimistic: the CTC probe's own val WER on real
speech is 0.4145, so the real-speech floor is ~0.42-0.45, not 0.2-0.3.

Emits generate.py-compatible dirs (audio/ + manifest.jsonl) so the existing scorer runs
on them unchanged:
  real_speech   the untouched target speech -- the true floor
  vae_roundtrip target speech -> mel -> VAE encode/decode -> vocoder, i.e. the best any
                latent-diffusion model on this stack could possibly sound
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT)); sys.path.insert(0, str(REPO_ROOT / "src"))

from configs import paths as project_paths  # noqa: E402
from salm.data.frontend import (  # noqa: E402
    AUDIOLDM_CLIP_SECONDS, AUDIOLDM_SR, MelConfig, MelExtractor, load_waveform_mono_16k,
)
from salm.data.manifest import load_manifest  # noqa: E402


def write_lane(out_dir: Path, rows: list[dict]) -> None:
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    with (out_dir / "manifest.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(
        {"mode": out_dir.name, "num_samples": len(rows), "control_lane": True}, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rows = load_manifest("a2s", args.split)[: args.num_samples]
    out_root = Path(args.out_root)

    from diffusers import AudioLDMPipeline
    pipe = AudioLDMPipeline.from_pretrained(
        project_paths.AUDIOLDM_MODEL_ID, torch_dtype=torch.float32).to(args.device)
    extractor = MelExtractor(MelConfig(norm_mode="none"), device=args.device)
    sf_scale = float(pipe.vae.config.scaling_factor)

    real_rows, vae_rows = [], []
    for r in rows:
        # align="left" matches how the a2s speech latents were cached.
        wav = load_waveform_mono_16k(
            str(r.speech_path), target_seconds=AUDIOLDM_CLIP_SECONDS, align="left")

        p_real = out_root / "real_speech" / "audio" / f"{r.sample_id}.wav"
        p_real.parent.mkdir(parents=True, exist_ok=True)
        sf.write(p_real, wav, AUDIOLDM_SR)

        with torch.no_grad():
            mel = extractor(wav)   # accepts np.ndarray directly, see frontend.py:145
            mel_in = mel.unsqueeze(0).unsqueeze(0)
            lat = pipe.vae.encode(mel_in).latent_dist.sample() * sf_scale
            mel_dec = pipe.vae.decode(lat / sf_scale).sample.squeeze(1)
            rec = pipe.vocoder(extractor.denormalize(mel_dec)).squeeze(0).cpu().numpy()

        p_vae = out_root / "vae_roundtrip" / "audio" / f"{r.sample_id}.wav"
        p_vae.parent.mkdir(parents=True, exist_ok=True)
        sf.write(p_vae, rec.astype(np.float32), AUDIOLDM_SR)

        for lane_rows, rel in ((real_rows, p_real), (vae_rows, p_vae)):
            lane_rows.append({
                "sample_id": r.sample_id, "status": "success",
                "generated_file_rel": f"audio/{r.sample_id}.wav",
                "generated_file": str(rel),
                "speech_file": str(r.speech_path), "audio_cap": r.audio_cap,
                "speech_text": r.speech_text,
            })

    write_lane(out_root / "real_speech", real_rows)
    write_lane(out_root / "vae_roundtrip", vae_rows)
    print(f"wrote {len(real_rows)} rows to {out_root}/{{real_speech,vae_roundtrip}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
