"""
Generate ambient audio, in any of the three rungs.

Modes:
  rung0  text  -> audio.  Conditions on the CLAP TEXT embedding of the dataset's
                  `audio_cap`. No training involved. This is the ceiling: how well can
                  AudioLDM render TTAS-like scenes when handed a perfect description?
  rung1  speech-> audio via CLAP AUDIO embedding of the speech, straight into frozen
                  AudioLDM. The control. Expect generic output, because CLAP encodes
                  "someone is speaking", not what they said -- confirming that is a
                  result, and it is what motivates the mapper.
  s2a    speech-> audio via a trained checkpoint (frozen encoder -> mapper -> U-Net).
                  The actual model.

All three share the same sampler, so differences come only from the conditioning vector.
We pass `prompt_embeds` directly to AudioLDMPipeline, which skips its text encoder --
verified against pipeline_audioldm.py, where the U-Net receives the vector as
`class_labels` with `encoder_hidden_states=None`.

Usage:
    python -m salm.generate --mode rung0 --split val --num-samples 8 --out-dir <dir>
    python -m salm.generate --mode s2a --checkpoint <ckpt.pt> --split val --num-samples 8 --out-dir <dir>
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
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from configs import paths as project_paths  # noqa: E402
from salm.data.frontend import AUDIOLDM_CLIP_SECONDS, AUDIOLDM_SR, load_waveform_mono_16k  # noqa: E402
from salm.data.manifest import load_manifest  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


def load_pipeline(device: str, dtype: torch.dtype):
    from diffusers import AudioLDMPipeline

    pipe = AudioLDMPipeline.from_pretrained(
        project_paths.AUDIOLDM_MODEL_ID, torch_dtype=dtype
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def clap_text_embeddings(pipe, captions: list[str], device: str) -> torch.Tensor:
    """CLAP text embedding, normalized exactly as the pipeline does internally."""
    inputs = pipe.tokenizer(
        captions, padding="max_length", max_length=512, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        out = pipe.text_encoder(
            inputs.input_ids.to(device), attention_mask=inputs.attention_mask.to(device)
        )
    return F.normalize(out.text_embeds, dim=-1)


class ClapAudioEncoder:
    """
    CLAP audio tower, for the rung1 control.

    Must be the NON-fusion checkpoint to match the space AudioLDM was trained against.
    LAION-CLAP operates at 48 kHz, so speech is resampled up from 16 kHz here.
    """

    SR = 48000

    def __init__(self, device: str, dtype: torch.dtype = torch.float32):
        from transformers import ClapModel, ClapProcessor

        self.device = device
        self.processor = ClapProcessor.from_pretrained(project_paths.CLAP_MODEL_ID)
        self.model = ClapModel.from_pretrained(
            project_paths.CLAP_MODEL_ID, torch_dtype=dtype
        ).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, wavs_16k: list[np.ndarray]) -> torch.Tensor:
        import librosa

        wavs_48k = [
            librosa.resample(w, orig_sr=AUDIOLDM_SR, target_sr=self.SR, res_type="soxr_hq")
            for w in wavs_16k
        ]
        inputs = self.processor(audios=wavs_48k, sampling_rate=self.SR, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self.model.get_audio_features(**inputs)
        return F.normalize(feats, dim=-1)


def load_s2a_checkpoint(checkpoint: Path, device: str):
    """Restore the trained mapper (and finetuned U-Net weights, if present)."""
    from salm.encoders.whisper_encoder import WhisperSpeechEncoder
    from salm.mapper import SpeechToClapMapper

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    log(f"    checkpoint epoch={payload.get('epoch')} val_loss={payload.get('val_loss')}")

    encoder = WhisperSpeechEncoder(
        cfg.get("whisper_path", str(project_paths.WHISPER_LARGE_V3_PATH)),
        dtype=torch.float16,
        device=device,
    )
    mapper = SpeechToClapMapper(
        input_dim=encoder.feature_dim,
        output_dim=payload["mapper"]["null_embedding"].shape[0],
        hidden_dim=cfg.get("mapper_hidden_dim", 1024),
        num_layers=cfg.get("mapper_layers", 2),
        dropout=0.0,
    ).to(device)
    mapper.load_state_dict(payload["mapper"])
    mapper.eval()

    return encoder, mapper, payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["rung0", "rung1", "s2a"])
    ap.add_argument("--checkpoint", default=None, help="required for --mode s2a")
    ap.add_argument("--direction", default="s2a", choices=["s2a", "a2s"])
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--num-samples", type=int, default=8, help="-1 for the whole split")
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--guidance-scale", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--copy-reference", action="store_true",
                    help="also write the reference ambient audio, for A/B listening")
    args = ap.parse_args()

    if args.mode == "s2a" and not args.checkpoint:
        log("ERROR: --mode s2a requires --checkpoint")
        return 1

    out_dir = Path(args.out_dir)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)

    rows = load_manifest(args.direction, args.split)
    rows = rows[args.start_index :]
    if args.num_samples > 0:
        rows = rows[: args.num_samples]
    log(f"{args.mode}: generating {len(rows)} samples from {args.direction}/{args.split}")

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    log("Loading AudioLDM ...")
    pipe = load_pipeline(args.device, dtype)

    encoder = mapper = clap = None
    if args.mode == "s2a":
        log("Loading trained checkpoint ...")
        encoder, mapper, _payload = load_s2a_checkpoint(Path(args.checkpoint), args.device)
    elif args.mode == "rung1":
        log("Loading CLAP audio tower (control) ...")
        clap = ClapAudioEncoder(args.device)

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    manifest_rows: list[dict] = []
    t0 = time.time()

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]

        # ---------------------------------------------- build the condition vector
        if args.mode == "rung0":
            cond = clap_text_embeddings(pipe, [r.audio_cap for r in batch], args.device)
            negative = None
        else:
            wavs = [load_waveform_mono_16k(str(r.speech_path)) for r in batch]

            if args.mode == "rung1":
                cond = clap(wavs)
                negative = None
            else:
                maxlen = max(w.shape[0] for w in wavs)
                padded = np.zeros((len(wavs), maxlen), dtype=np.float32)
                lengths = []
                for i, w in enumerate(wavs):
                    padded[i, : w.shape[0]] = w
                    lengths.append(w.shape[0])
                with torch.no_grad():
                    feats, mask = encoder.encode_with_mask(
                        torch.from_numpy(padded).to(args.device),
                        torch.tensor(lengths),
                    )
                    cond = mapper(feats, mask=mask)
                    negative = mapper.null_condition(len(batch), torch.device(args.device))

        cond = cond.to(dtype=pipe.unet.dtype)
        if negative is not None:
            negative = negative.to(dtype=pipe.unet.dtype)

        # ------------------------------------------------------------- sample
        with torch.no_grad():
            result = pipe(
                prompt_embeds=cond,
                negative_prompt_embeds=negative,
                num_inference_steps=args.steps,
                audio_length_in_s=AUDIOLDM_CLIP_SECONDS,
                guidance_scale=args.guidance_scale,
                generator=generator,
            )
        audios = result.audios

        for row, audio in zip(batch, audios):
            gen_path = out_dir / "audio" / f"{row.sample_id}.wav"
            sf.write(gen_path, audio, AUDIOLDM_SR)

            ref_path = None
            if args.copy_reference:
                ref = load_waveform_mono_16k(
                    str(row.audio_path), target_seconds=AUDIOLDM_CLIP_SECONDS
                )
                ref_path = out_dir / "audio" / f"{row.sample_id}__reference.wav"
                sf.write(ref_path, ref, AUDIOLDM_SR)

            manifest_rows.append(
                {
                    "sample_id": row.sample_id,
                    "status": "success",
                    "mode": args.mode,
                    # Relative to this manifest's directory. Absolute paths were used
                    # originally and broke as soon as a run directory was moved, so the
                    # portable form is authoritative and the absolute one is a convenience.
                    "generated_file_rel": str(gen_path.relative_to(out_dir)),
                    "generated_file": str(gen_path),
                    "reference_audio_file": str(row.audio_path),
                    "reference_copy": str(ref_path) if ref_path else None,
                    "speech_file": str(row.speech_path),
                    "audio_cap": row.audio_cap,
                    "speech_text": row.speech_text,
                    "generated_rms": float(np.sqrt(np.mean(np.square(audio)))),
                }
            )

        done = len(manifest_rows)
        rate = done / max(time.time() - t0, 1e-6)
        log(f"    {done}/{len(rows)}  {rate:.2f} samples/s")

    with (out_dir / "manifest.jsonl").open("w") as f:
        for r in manifest_rows:
            f.write(json.dumps(r) + "\n")

    rms = [r["generated_rms"] for r in manifest_rows]
    summary = {
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "direction": args.direction,
        "split": args.split,
        "num_samples": len(manifest_rows),
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "audioldm_model_id": project_paths.AUDIOLDM_MODEL_ID,
        "rms_mean": float(np.mean(rms)) if rms else 0.0,
        "rms_min": float(np.min(rms)) if rms else 0.0,
        "near_silent_count": int(sum(1 for r in rms if r < 1e-4)),
        "elapsed_seconds": time.time() - t0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    log(f"\nwrote {len(manifest_rows)} wavs to {out_dir/'audio'}")
    log(f"rms mean={summary['rms_mean']:.5f} min={summary['rms_min']:.5f} "
        f"near_silent={summary['near_silent_count']}")
    if summary["near_silent_count"]:
        log("WARNING: near-silent generations present -- inspect before reporting results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
