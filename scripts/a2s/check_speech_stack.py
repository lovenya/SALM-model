"""
A2S GATE: can AudioLDM's mel + VAE + HiFi-GAN reconstruct intelligible SPEECH?

This is the A2S analogue of `scripts/check_audioldm_stack.py`, and it is a hard gate for the
same reason: it is pure inference, and if it fails, every hour spent training A2S on this
stack is wasted. Run it before anything else.

THE RISK IT TESTS. The whole front-end was chosen for ambient audio: 16 kHz, **64 mel bins**,
fmax 8000, and a VAE + vocoder trained on general audio. Speech vocoders normally use 80 mel
bins. If 64 bins at this hop cannot carry formant detail, the round trip returns something
that sounds like speech but is not words -- and a generative model trained on top of it could
never do better than its own reconstruction ceiling.

WHY THIS GATE IS BETTER THAN THE S2A ONE. The S2A gate scored a mel-space correlation, which
answers "did the spectrogram survive?" but not "did the MEANING survive?". Here we hold
`speech_text` for every row, so the round-tripped audio can be transcribed with Whisper and
scored as **WER against the ground-truth transcript**. That is the quantity that actually
matters, and it is interpretable without a baseline table.

THREE PATHS, so a failure is attributable rather than just a failure:

    original    the real speech file, untouched          -> WER floor
    mel         wav -> mel -> vocoder -> wav             -> isolates mel + vocoder
    vae         wav -> mel -> VAE enc/dec -> vocoder     -> the full generative path

`original` is the control that makes the other two readable: ASR is not perfect and TTAS
transcripts are not guaranteed verbatim, so a raw WER of, say, 0.18 on the untouched audio is
the floor, not a defect. What matters is the DEGRADATION -- wer(vae) - wer(original).

Usage:
    python scripts/a2s/check_speech_stack.py --out-dir /path/to/gate --device cuda
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
from salm.data.frontend import (  # noqa: E402
    AUDIOLDM_CLIP_SECONDS,
    AUDIOLDM_SR,
    MelConfig,
    MelExtractor,
    load_waveform_mono_16k,
)
from salm.data.manifest import load_manifest  # noqa: E402
from salm.a2s.wer import corpus_wer  # noqa: E402
from salm.a2s.asr import WhisperASR  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


def score_mel(mel_ref: torch.Tensor, mel_rec: torch.Tensor) -> dict:
    a = mel_ref.flatten().float().numpy()
    b = mel_rec.flatten().float().numpy()
    return {
        "mel_corr": float(np.corrcoef(a, b)[0, 1]),
        "mel_mse": float(np.mean((a - b) ** 2)),
    }


# ------------------------------------------------------------------- duration scan
def duration_scan(rows, limit: int) -> dict:
    """
    How much A2S speech does the 10.24 s canvas actually truncate?

    Header-only reads (`sf.info`), so this is I/O rather than decode. It answers a design
    question the S2A pipeline never had to ask: S2A targets were ~10 s ambience against a
    10.24 s canvas, but speech averages 4.86 s with a tail past 16 s, and every truncated
    row is one whose audio no longer matches its transcript.
    """
    secs = []
    errors = 0
    for row in rows[:limit]:
        try:
            info = sf.info(str(row.speech_path))
            secs.append(info.frames / float(info.samplerate))
        except Exception:
            errors += 1
    a = np.array(secs) if secs else np.zeros(1)
    over = float((a > AUDIOLDM_CLIP_SECONDS).mean())
    return {
        "scanned": len(secs),
        "read_errors": errors,
        "mean_s": float(a.mean()),
        "median_s": float(np.median(a)),
        "p95_s": float(np.percentile(a, 95)),
        "max_s": float(a.max()),
        "frac_over_canvas": over,
        "canvas_s": AUDIOLDM_CLIP_SECONDS,
        "mean_canvas_occupancy": float(np.minimum(a, AUDIOLDM_CLIP_SECONDS).mean()
                                       / AUDIOLDM_CLIP_SECONDS),
    }



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--num-clips", type=int, default=24,
                    help="clips to round-trip and transcribe")
    ap.add_argument("--scan-limit", type=int, default=4000,
                    help="rows for the duration scan (header reads only)")
    ap.add_argument("--norm-mode", default="none",
                    help="mel normalization; the S2A gate measured 'none' as correct")
    ap.add_argument("--save-wavs", type=int, default=8, help="how many clips to write out")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"split": args.split, "norm_mode": args.norm_mode}
    t0 = time.time()

    # ------------------------------------------------------------------ rows
    rows = load_manifest("a2s", args.split)
    log(f"[0] A2S {args.split}: {len(rows)} rows")

    log(f"\n[1] Duration scan of the TARGET speech (first {args.scan_limit} rows, headers only)")
    scan = duration_scan(rows, args.scan_limit)
    report["duration_scan"] = scan
    log(f"    mean {scan['mean_s']:.2f}s  median {scan['median_s']:.2f}s  "
        f"p95 {scan['p95_s']:.2f}s  max {scan['max_s']:.2f}s")
    log(f"    over the {scan['canvas_s']}s canvas: {scan['frac_over_canvas']*100:.2f}% of rows "
        f"(these get left-cropped, so their audio stops matching speech_text)")
    log(f"    mean canvas occupancy: {scan['mean_canvas_occupancy']*100:.1f}% "
        f"-- the rest is the silence the model must learn to emit")

    # ------------------------------------------------------------- load stack
    log("\n[2] Loading AudioLDM (VAE + vocoder) offline ...")
    from diffusers import AudioLDMPipeline

    pipe = AudioLDMPipeline.from_pretrained(
        project_paths.AUDIOLDM_MODEL_ID, torch_dtype=torch.float32
    ).to(args.device)
    log(f"    vocoder sr={pipe.vocoder.config.sampling_rate} "
        f"mel_bins={pipe.vocoder.config.model_in_dim}")
    report["vocoder_sr"] = int(pipe.vocoder.config.sampling_rate)
    report["vocoder_mel_bins"] = int(pipe.vocoder.config.model_in_dim)

    extractor = MelExtractor(MelConfig(norm_mode=args.norm_mode), device=args.device)
    sf_scale = float(pipe.vae.config.scaling_factor)

    # ------------------------------------------------------------ round trips
    picked = rows[: args.num_clips]
    log(f"\n[3] Round-tripping {len(picked)} clips through both paths ...")

    originals, mel_recs, vae_recs, refs, mel_scores = [], [], [], [], []
    for i, row in enumerate(picked):
        # LEFT-aligned: an over-long utterance must keep its start, or the audio no longer
        # begins where speech_text begins.
        wav = load_waveform_mono_16k(
            str(row.speech_path), target_seconds=AUDIOLDM_CLIP_SECONDS, align="left"
        )
        mel = extractor(wav)

        with torch.no_grad():
            # path 'mel': vocoder only. The vocoder was trained to invert exactly this mel,
            # so it is the ground truth for whether the mel itself is right.
            rec_mel = pipe.vocoder(
                extractor.denormalize(mel.unsqueeze(0))
            ).cpu().numpy().squeeze()

            # path 'vae': the full generative path the diffusion model would decode through.
            mel_in = mel.unsqueeze(0).unsqueeze(0)          # VAE wants (B, 1, frames, mels)
            latent = pipe.vae.encode(mel_in).latent_dist.sample() * sf_scale
            mel_dec = pipe.vae.decode(latent / sf_scale).sample.squeeze(1)
            rec_vae = pipe.vocoder(
                extractor.denormalize(mel_dec)
            ).cpu().numpy().squeeze()

        mel_scores.append(score_mel(mel.cpu(), mel_dec.squeeze(0).cpu()))
        originals.append(wav.astype(np.float32))
        mel_recs.append(rec_mel.astype(np.float32))
        vae_recs.append(rec_vae.astype(np.float32))
        refs.append(row.speech_text)

        if i < args.save_wavs:
            d = out_dir / "listen" / row.sample_id
            d.mkdir(parents=True, exist_ok=True)
            sf.write(d / "00_original.wav", wav, AUDIOLDM_SR)
            sf.write(d / "01_mel_vocoder.wav", rec_mel, AUDIOLDM_SR)
            sf.write(d / "02_vae_vocoder.wav", rec_vae, AUDIOLDM_SR)
            (d / "transcript.txt").write_text(row.speech_text + "\n")

    report["mel_space"] = {
        "corr_mean": float(np.mean([s["mel_corr"] for s in mel_scores])),
        "mse_mean": float(np.mean([s["mel_mse"] for s in mel_scores])),
    }
    log(f"    VAE round-trip in mel space: corr={report['mel_space']['corr_mean']:+.4f} "
        f"mse={report['mel_space']['mse_mean']:.4f}")

    # -------------------------------------------------------------------- ASR
    log(f"\n[4] Transcribing all three paths with Whisper-large-v3 ...")
    asr = WhisperASR(project_paths.WHISPER_LARGE_V3_PATH, args.device)

    hyps = {}
    for name, wavs in (("original", originals), ("mel", mel_recs), ("vae", vae_recs)):
        hyps[name] = asr.transcribe(wavs)
        log(f"    transcribed {name}")

    wer = {name: corpus_wer(refs, h) for name, h in hyps.items()}
    report["wer"] = wer
    report["examples"] = [
        {
            "sample_id": picked[i].sample_id,
            "reference": refs[i],
            "original": hyps["original"][i],
            "mel": hyps["mel"][i],
            "vae": hyps["vae"][i],
        }
        for i in range(min(6, len(picked)))
    ]

    log("\n    path        WER     edits/words")
    for name in ("original", "mel", "vae"):
        w = wer[name]
        log(f"    {name:10s} {w['wer']:.4f}   {w['edits']}/{w['ref_words']}")

    floor = wer["original"]["wer"]
    degradation = wer["vae"]["wer"] - floor
    report["wer_floor"] = floor
    report["wer_degradation_vae"] = degradation
    report["wer_degradation_mel"] = wer["mel"]["wer"] - floor

    log("\n    examples (reference | vae round-trip):")
    for ex in report["examples"][:4]:
        log(f"      REF: {ex['reference'][:88]}")
        log(f"      VAE: {ex['vae'][:88]}")

    # ---------------------------------------------------------------- verdict
    # Thresholds are judgement calls, stated so the verdict can be argued with rather than
    # deferred to. The degradation, not the absolute WER, is what the stack is responsible
    # for: the floor already absorbs ASR error and non-verbatim transcripts.
    if degradation < 0.15:
        verdict, detail = "PASS", "words survive the round trip; A2S on this stack is viable"
    elif degradation < 0.40:
        verdict, detail = (
            "MARGINAL",
            "intelligibility is degraded but not destroyed -- the reconstruction ceiling will "
            "cap any A2S result, so report it alongside every generation metric",
        )
    else:
        verdict, detail = (
            "FAIL",
            "the stack cannot carry speech. Do not train A2S on it. Next levers, cheapest "
            "first: (a) raise n_mels to 80 and check whether the pretrained vocoder still "
            "accepts it -- it will not, model_in_dim is fixed, so this means a different "
            "vocoder; (b) swap in a speech vocoder/codec for the A2S decoder while keeping "
            "the conditioner work; (c) drop the AudioLDM decoder for A2S entirely",
        )

    report["verdict"] = verdict
    report["verdict_detail"] = detail
    report["elapsed_seconds"] = time.time() - t0

    log("\n" + "=" * 78)
    log(f"A2S GATE: {verdict}")
    log(f"  WER floor (untouched audio) : {floor:.4f}")
    log(f"  WER after mel+vocoder       : {wer['mel']['wer']:.4f}  "
        f"(+{report['wer_degradation_mel']:.4f})")
    log(f"  WER after VAE+vocoder       : {wer['vae']['wer']:.4f}  (+{degradation:.4f})")
    log(f"  {detail}")
    log("=" * 78)

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    log(f"\nwrote {out_dir}/report.json  ({(time.time()-t0)/60:.1f} min)")
    log(f"listen: {out_dir}/listen/<sample_id>/{{00_original,01_mel_vocoder,02_vae_vocoder}}.wav")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
