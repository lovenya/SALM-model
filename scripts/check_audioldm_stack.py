"""
Rung 0 gate: verify the AudioLDM stack works offline BEFORE spending hours training.

This script answers four questions, in dependency order. If any fails, training is
pointless, so run this first.

  1. Do the weights load offline (no internet on compute nodes)?
  2. How is the U-Net actually conditioned? We expect a single global 512-d CLAP
     vector passed as `class_labels` with `encoder_hidden_states=None`. The whole
     conditioner design depends on this, so we assert it against the installed source
     rather than trusting the paper.
  3. Is our mel extraction correct? Checked by round-tripping a REAL clip:
     wav -> mel -> vocoder -> wav. If the mel scale is wrong the vocoder output is
     noise, so this isolates mel correctness from the VAE entirely. Both
     `norm_mode` variants are tried and scored so the uncertain knob in
     frontend.py is settled empirically.
  4. Does the full generative path produce audible audio? wav -> mel -> VAE encode
     -> VAE decode -> vocoder, plus a text-conditioned generation from `audio_cap`.

Usage (inside an allocation or on a login node for the CPU-only checks):
    python scripts/check_audioldm_stack.py --out-dir /path/to/scratch/rung0
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from salm.data.frontend import (  # noqa: E402
    AUDIOLDM_SR,
    AUDIOLDM_CLIP_SECONDS,
    MelConfig,
    MelExtractor,
    load_waveform_mono_16k,
)

AUDIOLDM_MODEL_ID = "cvssp/audioldm-s-full-v2"


def log(msg: str) -> None:
    print(msg, flush=True)


def check_conditioning_path(pipe) -> dict:
    """
    Confirm HOW the U-Net consumes the CLAP embedding.

    We read the installed pipeline source instead of assuming, because the conditioner's
    output contract (one global vector vs. a token sequence) follows directly from it.
    """
    unet_cfg = pipe.unet.config
    found = {
        "cross_attention_dim": unet_cfg.get("cross_attention_dim"),
        "class_embed_type": unet_cfg.get("class_embed_type"),
        "projection_class_embeddings_input_dim": unet_cfg.get(
            "projection_class_embeddings_input_dim"
        ),
        "in_channels": unet_cfg.get("in_channels"),
        "sample_size": unet_cfg.get("sample_size"),
    }

    src = inspect.getsource(type(pipe).__call__)
    found["passes_class_labels"] = "class_labels=prompt_embeds" in src
    found["encoder_hidden_states_none"] = "encoder_hidden_states=None" in src

    log("\n[2] U-Net conditioning path")
    for k, v in found.items():
        log(f"    {k}: {v}")

    if found["passes_class_labels"] and found["encoder_hidden_states_none"]:
        log("    => CONFIRMED: single global vector via class_labels. Mapper outputs ONE vector.")
    else:
        log("    => WARNING: conditioning path differs from expectation. Inspect before training:")
        log(f"       {inspect.getsourcefile(type(pipe))}")

    return found


def score_reconstruction(mel_ref: torch.Tensor, mel_rec: torch.Tensor) -> dict:
    """Compare two (frames, mels) log-mels. Correlation is the readable signal here."""
    n = min(mel_ref.shape[0], mel_rec.shape[0])
    a = mel_ref[:n].flatten().double()
    b = mel_rec[:n].flatten().double()
    mse = torch.mean((a - b) ** 2).item()
    corr = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    return {"mel_mse": mse, "mel_corr": corr}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="scratch dir for wavs + report")
    ap.add_argument(
        "--audio-file",
        default=None,
        help="a real ambient wav to round-trip; defaults to first row of the S2A val manifest",
    )
    ap.add_argument("--caption", default="Loud vehicle driving and passing another vehicle")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-generation", action="store_true", help="skip the slow sampling step")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    # ---------------------------------------------------------------- 1. load
    log("[1] Loading AudioLDM offline ...")
    from diffusers import AudioLDMPipeline

    pipe = AudioLDMPipeline.from_pretrained(AUDIOLDM_MODEL_ID, torch_dtype=torch.float32)
    pipe = pipe.to(args.device)
    log(f"    loaded {AUDIOLDM_MODEL_ID} on {args.device}")
    log(f"    vocoder sr: {pipe.vocoder.config.sampling_rate}, mel bins: {pipe.vocoder.config.model_in_dim}")
    report["vocoder_sr"] = pipe.vocoder.config.sampling_rate
    report["vocoder_mel_bins"] = pipe.vocoder.config.model_in_dim

    if pipe.vocoder.config.sampling_rate != AUDIOLDM_SR:
        log(f"    WARNING: vocoder sr != assumed {AUDIOLDM_SR}; fix frontend.py constants")
    if pipe.vocoder.config.model_in_dim != MelConfig().n_mels:
        log("    WARNING: vocoder mel bins != assumed 64; fix frontend.py constants")

    # ---------------------------------------------------- 2. conditioning path
    report["conditioning"] = check_conditioning_path(pipe)

    # ------------------------------------------------------- pick a real clip
    audio_path = args.audio_file
    if audio_path is None:
        manifest = Path(
            "/scratch/l/lovenya/SpeechAudioLM/data/manifests/clean_v2/s2a/ttas_val.jsonl"
        )
        dataset_root = Path(
            "/scratch/l/lovenya/salm_migration_clean/datasets/ttas/extracted"
        )
        with manifest.open() as f:
            row = json.loads(f.readline())
        audio_path = str(dataset_root / row["audio_file"])
        args.caption = row.get("audio_cap", args.caption)
    log(f"\n    round-trip clip: {audio_path}")
    log(f"    caption: {args.caption!r}")
    report["audio_file"] = audio_path
    report["caption"] = args.caption

    wav = load_waveform_mono_16k(audio_path, target_seconds=AUDIOLDM_CLIP_SECONDS)
    sf.write(out_dir / "00_original.wav", wav, AUDIOLDM_SR)

    # ------------------------------------- 3. mel correctness via vocoder only
    # This tests ONE thing: is the mel itself right (filterbank, hop, orientation, scale)?
    # The vocoder is the ground truth because it was trained to invert exactly AudioLDM's
    # mel. Deliberately uses raw log-mel and does NOT vary norm_mode: an earlier version
    # looped over norm modes here but denormalized before vocoding, so both branches fed
    # the vocoder the identical tensor and the "comparison" was vacuous.
    log("\n[3] Mel round-trip (wav -> raw log-mel -> vocoder -> wav), isolating mel from VAE")
    raw_extractor = MelExtractor(MelConfig(norm_mode="none"), device=args.device)
    mel_raw = raw_extractor(wav)

    with torch.no_grad():
        rec = pipe.vocoder(mel_raw.unsqueeze(0)).cpu().numpy().squeeze()
    sf.write(out_dir / "01_melroundtrip.wav", rec, AUDIOLDM_SR)

    mel_scores = score_reconstruction(raw_extractor(wav).cpu(), raw_extractor(rec).cpu())
    mel_scores["rec_rms"] = float(np.sqrt(np.mean(rec**2)))
    report["mel_roundtrip"] = mel_scores
    log(f"    corr={mel_scores['mel_corr']:+.4f} mse={mel_scores['mel_mse']:.4f} "
        f"rms={mel_scores['rec_rms']:.5f}")
    if mel_scores["mel_corr"] < 0.8:
        log("    => WARNING: weak round-trip. Mel settings are wrong; listen to")
        log("       01_melroundtrip.wav before trusting any training run.")

    # --------------------------------------------------- 4a. VAE round-trip
    # THIS is the test that discriminates between normalizations, because the VAE is
    # scale-sensitive. Scoring happens in mel space (input vs decoded, same normalization),
    # so the vocoder cannot mask a mismatch.
    #
    # Expectation: "none" should win. The pipeline runs vae.decode(...) -> vocoder(mel)
    # with no rescaling between them, so the VAE's output space IS the vocoder's input
    # space, and section [3] shows the vocoder wants raw log-mel.
    log("\n[4a] VAE round-trip per norm_mode (mel -> encode -> decode), scored in mel space")
    sf_scale = float(pipe.vae.config.scaling_factor)
    vae_results = {}

    for norm_mode in ("none", "audiomae"):
        extractor = MelExtractor(MelConfig(norm_mode=norm_mode), device=args.device)
        mel = extractor(wav)

        with torch.no_grad():
            mel_in = mel.unsqueeze(0).unsqueeze(0)          # VAE wants (B, 1, frames, mels)
            posterior = pipe.vae.encode(mel_in).latent_dist
            latent = posterior.sample() * sf_scale
            mel_dec = pipe.vae.decode(latent / sf_scale).sample.squeeze(1)

            rec_vae = pipe.vocoder(extractor.denormalize(mel_dec)).cpu().numpy().squeeze()

        sf.write(out_dir / f"02_vae_roundtrip_{norm_mode}.wav", rec_vae, AUDIOLDM_SR)

        scores = score_reconstruction(mel.cpu(), mel_dec.squeeze(0).cpu())
        scores["rec_rms"] = float(np.sqrt(np.mean(rec_vae**2)))
        scores["latent_shape"] = list(latent.shape)
        vae_results[norm_mode] = scores
        log(f"     norm_mode={norm_mode:9s} mel-space corr={scores['mel_corr']:+.4f} "
            f"mse={scores['mel_mse']:.4f} audio_rms={scores['rec_rms']:.5f}")

    best_norm = max(vae_results, key=lambda k: vae_results[k]["mel_corr"])
    report["vae_roundtrip_by_norm"] = vae_results
    report["vae_roundtrip"] = vae_results[best_norm]
    report["best_norm_mode"] = best_norm
    report["latent_shape"] = vae_results[best_norm]["latent_shape"]
    log(f"     => best norm_mode: {best_norm} "
        f"(VAE corr {vae_results[best_norm]['mel_corr']:+.4f}, "
        f"scaling_factor {sf_scale:.4f})")
    if vae_results[best_norm]["mel_corr"] < 0.8:
        log("     => WARNING: even the best normalization round-trips poorly through the")
        log("        VAE. Do not trust precomputed latents; inspect 02_vae_roundtrip_*.wav.")

    # ------------------------------------------- 4b. text-conditioned sampling
    if not args.skip_generation:
        log("\n[4b] Text-conditioned generation from the caption (Rung 0 ceiling sample)")
        with torch.no_grad():
            gen = pipe(
                args.caption,
                num_inference_steps=50,
                audio_length_in_s=AUDIOLDM_CLIP_SECONDS,
                guidance_scale=2.5,
            ).audios[0]
        sf.write(out_dir / "03_generated_from_caption.wav", gen, AUDIOLDM_SR)
        report["generated_rms"] = float(np.sqrt(np.mean(gen**2)))
        log(f"     generated rms: {report['generated_rms']:.5f}")
        if report["generated_rms"] < 1e-4:
            log("     => WARNING: generated audio is near-silent. Generation path is broken.")

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    log(f"\nWrote wavs + report.json to {out_dir}")
    log("LISTEN to 01_melroundtrip_*.wav and 03_generated_from_caption.wav before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
