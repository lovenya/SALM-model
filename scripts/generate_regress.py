"""
Generate audio from a T0 latent-regression checkpoint.

No diffusion, so this bypasses the AudioLDM pipeline's sampling loop entirely and uses only its
frozen decoder halves, exactly as verified against the installed source
(diffusers/pipelines/deprecated/audioldm/pipeline_audioldm.py):

    latents / vae.config.scaling_factor  ->  vae.decode(...).sample  ->  vocoder(mel)

That is one forward pass instead of 50 denoising steps, which is itself a result worth recording:
this baseline's latency is the floor the diffusion models must justify.

Output layout matches salm.generate exactly -- audio/<sample_id>.wav, manifest.jsonl with
`generated_file_rel`, summary.json -- so the packaging, reporting and eval tooling all work
against it unchanged.

Usage:
    python scripts/generate_regress.py --checkpoint <ckpt.pt> --out-dir <dir> \
        --split val --num-samples 16 --copy-reference
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from configs import paths as project_paths  # noqa: E402
from salm.data.dataset import S2ALatentDataset, collate  # noqa: E402

AUDIOLDM_SR = 16000
AUDIOLDM_CLIP_SECONDS = 10.24


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--direction", default="s2a")
    ap.add_argument("--split", default="val")
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--copy-reference", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    kind = payload.get("model_kind")
    if kind != "latent_regressor":
        # A diffusion checkpoint loaded here would silently produce garbage, since its state dict
        # is a conditioner writing into a 512-d slot, not a latent predictor.
        raise SystemExit(
            f"{args.checkpoint} has model_kind={kind!r}, expected 'latent_regressor'. "
            "Use salm.generate for diffusion checkpoints."
        )
    cfg = payload["config"]
    latent_shape = tuple(payload["latent_shape"])
    log(f"checkpoint epoch={payload.get('epoch')} arch={cfg.get('arch')} latent={latent_shape}")

    feat_dir = project_paths.FEATURES_ROOT / args.direction / args.split
    ds = S2ALatentDataset(feat_dir, speech_seconds=cfg.get("speech_seconds", 16.0))
    lo = args.start_index
    hi = min(len(ds), lo + args.num_samples)
    rows = list(range(lo, hi))
    log(f"{args.split}: generating rows {lo}..{hi-1} of {len(ds)}")

    # ------------------------------------------------------------------- models
    from diffusers import AudioLDMPipeline
    from salm.architectures import ENCODERS, parse_encoder
    from salm.latent_regressor import build_latent_regressor

    pipe = AudioLDMPipeline.from_pretrained(
        project_paths.AUDIOLDM_MODEL_ID, torch_dtype=torch.float32
    )
    vae, vocoder = pipe.vae.to(device).eval(), pipe.vocoder.to(device).eval()
    scaling = float(vae.config.scaling_factor)
    del pipe.unet, pipe.text_encoder            # never used on this path; free the memory
    log(f"frozen VAE + vocoder loaded (vae scaling_factor={scaling:.4f})")

    encoder = ENCODERS[parse_encoder(cfg["arch"])](cfg, device)
    encoder.eval()
    model = build_latent_regressor(encoder.feature_dim, latent_shape, cfg).to(device).eval()
    model.load_state_dict(payload["model"])
    log("regressor weights loaded")

    manifest: list[dict] = []
    pred_sq = tgt_sq = n_el = 0.0
    t0 = time.time()

    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start : start + args.batch_size]
        batch = collate([ds[i] for i in chunk])

        with torch.no_grad():
            feats, mask = encoder.encode_with_mask(
                batch["speech"].to(device), batch["speech_length"]
            )
            pred = model(feats, mask=mask).float()          # (B, C, T, F), scaled space
            # Same decode the pipeline performs, minus the sampling loop.
            mel = vae.decode(pred / scaling).sample
            if mel.dim() == 4:
                mel = mel.squeeze(1)
            wavs = vocoder(mel).cpu().float().numpy()

        # Record the variance ratio -- the number this whole baseline exists to produce.
        tgt = ((batch["latent_mean"] + batch["latent_std"] * torch.randn_like(batch["latent_std"]))
               * ds.vae_scaling_factor).float()
        pred_sq += pred.pow(2).sum().item()
        tgt_sq += tgt.pow(2).sum().item()
        n_el += pred.numel()

        n_target = int(AUDIOLDM_SR * AUDIOLDM_CLIP_SECONDS)
        for i, row_idx in enumerate(chunk):
            row = ds.rows[row_idx]
            sid = row["sample_id"]
            wav = wavs[i][:n_target]
            gen_rel = f"audio/{sid}.wav"
            sf.write(out_dir / gen_rel, wav, AUDIOLDM_SR)

            entry = {
                "sample_id": sid,
                "generated_file_rel": gen_rel,     # relative: absolute paths broke on a root move
                "audio_cap": row.get("audio_cap"),
                "speech_text": row.get("speech_text"),
                "rms": float(np.sqrt(np.mean(wav.astype(np.float64) ** 2))),
            }
            if args.copy_reference:
                from salm.data.frontend import load_waveform_mono_16k

                ref = load_waveform_mono_16k(
                    str(row["audio_file"]), target_seconds=AUDIOLDM_CLIP_SECONDS
                )
                ref_rel = f"audio/{sid}__reference.wav"
                sf.write(out_dir / ref_rel, ref, AUDIOLDM_SR)
                entry["reference_file_rel"] = ref_rel
            manifest.append(entry)

    elapsed = time.time() - t0
    pred_std = (pred_sq / max(n_el, 1)) ** 0.5
    tgt_std = (tgt_sq / max(n_el, 1)) ** 0.5

    with (out_dir / "manifest.jsonl").open("w") as f:
        for e in manifest:
            f.write(json.dumps(e) + "\n")
    summary = {
        "mode": "t0_regress",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "arch": cfg.get("arch"),
        "num_samples": len(manifest),
        "split": args.split,
        "start_index": args.start_index,
        "seed": args.seed,
        "elapsed_seconds": elapsed,
        "seconds_per_sample": elapsed / max(len(manifest), 1),
        "pred_latent_std": pred_std,
        "target_latent_std": tgt_std,
        "std_ratio": pred_std / max(tgt_std, 1e-8),
        "rms_mean": float(np.mean([e["rms"] for e in manifest])) if manifest else 0.0,
        "diffusion_steps": 0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    log(f"\n{len(manifest)} samples in {elapsed:.1f}s "
        f"({summary['seconds_per_sample']:.2f} s/sample, ONE forward pass -- no denoising loop)")
    log(f"predicted latent std {pred_std:.4f} vs target {tgt_std:.4f}  "
        f"ratio {summary['std_ratio']:.3f}")
    log("  ratio near 1.0 = predictions match the scale real latents have.")
    log("  well below 1.0 = the variance collapse that makes regression sound washed out.")
    log("  reference points measured on train latents: blend of 2 = 0.73, blend of 16 = 0.41.")
    log(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
