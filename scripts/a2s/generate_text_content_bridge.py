#!/usr/bin/env python
"""Render held-out oracle-text bridge lanes through ``content_concat_v4``.

All lanes use the renderer's learned-null global condition.  They differ only in the
content sequence: codes sampled from the supplied transcript, codes sampled from another
row's transcript, or target-speech oracle codes.  Ambient audio is never loaded.
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from configs import paths as project_paths  # noqa: E402
from salm.a2s.content_codebook import ContentCodebook  # noqa: E402
from salm.a2s.content_concat import ContentProjector, concat_content, widen_conv_in  # noqa: E402
from salm.a2s.text_content_lm import TextContentLM  # noqa: E402
from salm.data.frontend import AUDIOLDM_CLIP_SECONDS, AUDIOLDM_SR, load_waveform_mono_16k  # noqa: E402
from salm.data.manifest import load_manifest  # noqa: E402
from salm.generate import deranged_indices, load_pipeline, load_s2a_checkpoint, load_trained_unet  # noqa: E402


def build_model(payload: dict, device: str) -> TextContentLM:
    model = TextContentLM(
        num_codes=int(payload["num_codes"]),
        text_dim=int(payload["text_dim"]),
        content_frames=int(payload["content_frames"]),
        max_text_tokens=int(payload["max_text_tokens"]),
        d_model=int(payload["d_model"]),
        n_layers=int(payload["layers"]),
        n_heads=int(payload["heads"]),
        dropout=0.0,
    ).to(device).eval()
    model.load_state_dict(payload["model"])
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge-checkpoint", type=Path, required=True)
    ap.add_argument("--renderer-checkpoint", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--lane", choices=["oracle_text", "shuffled_text", "oracle_codes"], required=True)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--guidance-scale", type=float, default=3.0)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.out_dir.exists():
        raise SystemExit(f"refusing existing output directory: {args.out_dir}")
    for p in (args.bridge_checkpoint, args.renderer_checkpoint):
        if not p.is_file():
            raise SystemExit(f"missing checkpoint: {p}")
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "audio").mkdir()

    rows_all = load_manifest("a2s", args.split)
    stop = min(len(rows_all), args.start_index + args.num_samples)
    indices = list(range(args.start_index, stop))
    rows = [rows_all[i] for i in indices]
    if len(rows) < 2:
        raise SystemExit("need at least two rows for a paired shuffled-text control")

    hidden = np.load(args.features / f"{args.split}_hidden.npy", mmap_mode="r")
    mask = np.load(args.features / f"{args.split}_mask.npy", mmap_mode="r")
    target_codes = np.load(
        project_paths.OUTPUTS_ROOT / "content_codes" /
        f"codes_codebook_k512_{args.split}.npy", mmap_mode="r"
    )
    if not (len(hidden) == len(mask) == len(target_codes) == len(rows_all)):
        raise SystemExit("text/code cache rows do not match the held-out manifest")

    bridge_payload = torch.load(args.bridge_checkpoint, map_location="cpu", weights_only=False)
    if bridge_payload.get("bridge_kind") != "oracle_text_content_lm":
        raise SystemExit("checkpoint is not an oracle text-content bridge")
    if int(bridge_payload["num_codes"]) != 512:
        raise SystemExit("this gate requires its registered K=512 checkpoint")
    bridge = build_model(bridge_payload, args.device)
    book = ContentCodebook(512, int(bridge_payload["codes"].shape[1])).to(args.device)
    book.codes.copy_(bridge_payload["codes"].to(args.device))

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    pipe = load_pipeline(args.device, dtype)
    unused_encoder, conditioner, renderer_payload = load_s2a_checkpoint(
        args.renderer_checkpoint, args.device
    )
    del unused_encoder
    if "content_projector" not in renderer_payload:
        raise SystemExit("renderer checkpoint has no content-concat projector")
    proj_state = renderer_payload["content_projector"]
    if "proj.bias" in proj_state:
        raise SystemExit("biased old projector is not a valid exact-null renderer")
    extra = int(renderer_payload.get("content_concat_channels", 8))
    mode = str(renderer_payload.get("content_concat_mode", "broadcast"))
    out_features = int(proj_state["proj.weight"].shape[0])
    n_freq = 16 if mode == "broadcast" else out_features // extra
    projector = ContentProjector(
        content_dim=book.content_dim, extra_channels=extra, n_freq=n_freq, mode=mode,
    ).to(args.device).eval()
    projector.load_state_dict(proj_state)
    widen_conv_in(pipe.unet, extra)
    pipe.unet.to(args.device)
    loaded = load_trained_unet(pipe, renderer_payload, dtype)
    if not loaded:
        raise SystemExit("renderer checkpoint contains no trained U-Net")

    if args.lane == "shuffled_text":
        local_perm = deranged_indices(len(rows), args.seed + 41)
        source_indices = [indices[j] for j in local_perm]
    else:
        source_indices = indices

    diffusion_rng = torch.Generator(device=args.device).manual_seed(args.seed)
    code_rng = torch.Generator(device=args.device).manual_seed(args.seed + 9973)
    manifest = []
    started = time.time()
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        target_idx = indices[start : start + len(batch_rows)]
        source_idx = source_indices[start : start + len(batch_rows)]
        if args.lane == "oracle_codes":
            code_t = torch.from_numpy(
                np.asarray(target_codes[target_idx], dtype=np.int64)
            ).to(args.device)
        else:
            h = torch.from_numpy(np.asarray(hidden[source_idx], dtype=np.float32)).to(args.device)
            m = torch.from_numpy(np.asarray(mask[source_idx], dtype=np.bool_)).to(args.device)
            code_t = bridge.sample(
                h, m, temperature=args.temperature, top_k=args.top_k,
                greedy=args.greedy, generator=code_rng,
            )
        content = book.dequantize(code_t)
        cond = conditioner.null_condition(len(batch_rows), torch.device(args.device)).to(dtype)

        original_forward = pipe.unet.forward

        def patched_forward(sample, timestep, encoder_hidden_states=None, **kwargs):
            reps = sample.shape[0] // content.shape[0]
            seq = content.repeat(reps, 1, 1) if reps > 1 else content
            cmap = projector(seq.float()).to(sample.dtype)
            return original_forward(
                concat_content(sample, cmap), timestep, encoder_hidden_states=None, **kwargs
            )

        pipe.unet.forward = patched_forward
        try:
            with torch.inference_mode():
                audios = pipe(
                    prompt_embeds=cond,
                    negative_prompt_embeds=cond,
                    num_inference_steps=args.steps,
                    audio_length_in_s=AUDIOLDM_CLIP_SECONDS,
                    guidance_scale=args.guidance_scale,
                    generator=diffusion_rng,
                ).audios
        finally:
            pipe.unet.forward = original_forward

        for row, audio, ti, si, sampled in zip(
            batch_rows, audios, target_idx, source_idx, code_t.detach().cpu().numpy()
        ):
            path = args.out_dir / "audio" / f"{row.sample_id}.wav"
            ref = args.out_dir / "audio" / f"{row.sample_id}__reference.wav"
            sf.write(path, audio, AUDIOLDM_SR)
            sf.write(
                ref,
                load_waveform_mono_16k(
                    str(row.speech_path), target_seconds=AUDIOLDM_CLIP_SECONDS, align="left"
                ),
                AUDIOLDM_SR,
            )
            supplied = rows_all[si].speech_text
            manifest.append({
                "sample_id": row.sample_id,
                "status": "success",
                "lane": args.lane,
                "generated_file": str(path),
                "generated_file_rel": str(path.relative_to(args.out_dir)),
                "reference_copy": str(ref),
                "reference_audio_file": str(row.speech_path),
                "speech_text": supplied,
                "supplied_text": supplied,
                "paired_target_text": row.speech_text,
                "text_source_sample_id": rows_all[si].sample_id,
                "target_row_index": ti,
                "text_source_row_index": si,
                "num_distinct_codes": int(np.unique(sampled).size),
                "generated_rms": float(np.sqrt(np.mean(np.square(audio)))),
                "ambient_read": False,
            })
        print(f"{args.lane}: {len(manifest)}/{len(rows)}", flush=True)

    with (args.out_dir / "manifest.jsonl").open("w") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")
    summary = {
        "lane": args.lane,
        "num_samples": len(manifest),
        "bridge_checkpoint": str(args.bridge_checkpoint),
        "bridge_epoch": bridge_payload.get("epoch"),
        "renderer_checkpoint": str(args.renderer_checkpoint),
        "renderer_epoch": renderer_payload.get("epoch"),
        "temperature": args.temperature,
        "top_k": args.top_k,
        "greedy": args.greedy,
        "seed": args.seed,
        "ambient_read": False,
        "elapsed_seconds": time.time() - started,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
