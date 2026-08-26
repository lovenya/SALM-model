"""
Precompute AudioLDM VAE latents for the TARGET clip of a split.

Which file is the target depends on direction, hence `--target`:
    s2a  --target audio   the ambient clip   (the default; every S2A run used this)
    a2s  --target speech  the utterance      (A2S generates speech)

Why precompute: mel extraction and VAE encoding are the two steps most likely to be
subtly wrong, and they are deterministic. Running them once means (a) a mistake shows
up in minutes rather than mid-training, and (b) the training loop never touches the
VAE, so epochs are fast and the VAE never occupies VRAM during training.

We store the posterior mean and std rather than a single sample, so training can sample
from the latent distribution the way AudioLDM does instead of reusing one frozen draw.

Storage is modest: ~128 KB per sample => under 2 GB for the 15k-row S2A train split.

Outputs, under <out-dir>/<direction>/<split>/:
    latents_mean.npy   (N, C, H, W) float16
    latents_std.npy    (N, C, H, W) float16
    index.jsonl        one row per KEPT sample, with paths + metadata
    meta.json          shapes, scaling factor, mel settings, drop counts

Usage:
    python scripts/precompute_latents.py --direction s2a --split train
    python scripts/precompute_latents.py --direction a2s --split train --target speech
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from configs import paths as project_paths  # noqa: E402
from salm.data.frontend import (  # noqa: E402
    AUDIOLDM_CLIP_SECONDS,
    MelConfig,
    MelExtractor,
    is_near_silent,
    load_waveform_mono_16k,
)
from salm.data.manifest import load_manifest  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


def load_rows_prefetched(
    rows: list,
    num_workers: int,
    prefetch: int,
    target_attr: str = "audio_path",
    align: str = "center",
) -> Iterator[tuple[int, object, np.ndarray | None, str | None]]:
    """
    Yield (row_index, row, waveform, error) in manifest order, loading ahead in threads.

    Audio loading dominates this script -- decode from Lustre plus a 24k->16k resample per
    file -- and it was originally serial, capping throughput at ~1.8 rows/s regardless of
    whether the VAE ran on CPU or GPU. Threads are the right tool because soundfile and
    librosa's resampler spend their time in C code with the GIL released.

    Order is preserved so `latent_index` stays aligned with index.jsonl, and only
    `prefetch` files are held in flight so memory stays bounded (each 10.24 s clip is
    ~650 KB, so a few hundred in flight is cheap).
    """
    def load_one(item: tuple[int, object]):
        row_i, row = item
        try:
            wav = load_waveform_mono_16k(
                str(getattr(row, target_attr)),
                target_seconds=AUDIOLDM_CLIP_SECONDS,
                align=align,
            )
        except Exception as exc:
            return row_i, row, None, f"read_error: {exc}"
        if is_near_silent(wav):
            return row_i, row, None, "near_silent_target"
        return row_i, row, wav, None

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        pending: list = []
        items = iter(list(enumerate(rows)))

        for item in items:
            pending.append(pool.submit(load_one, item))
            if len(pending) >= prefetch:
                break

        while pending:
            future = pending.pop(0)
            try:
                nxt = next(items)
            except StopIteration:
                pass
            else:
                pending.append(pool.submit(load_one, nxt))
            yield future.result()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default="s2a", choices=["s2a", "a2s"])
    ap.add_argument(
        "--target",
        default=None,
        choices=["audio", "speech"],
        help="which file to encode as the generation target. Default: audio for s2a, "
             "speech for a2s -- i.e. whatever that direction generates.",
    )
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--out-dir", default=str(project_paths.FEATURES_ROOT))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--norm-mode",
        default="none",
        choices=["none", "audiomae"],
        help="mel normalization; use whatever check_audioldm_stack.py reported as best",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="debug: only process N rows")
    ap.add_argument("--overwrite", action="store_true")
    # Audio decode + resample is the bottleneck here, not the VAE, and it is I/O plus C
    # code with the GIL released -- so threads genuinely help.
    ap.add_argument("--load-workers", type=int, default=12,
                    help="threads decoding/resampling audio ahead of the VAE")
    ap.add_argument("--load-prefetch", type=int, default=96,
                    help="max clips held in flight (bounds memory)")
    args = ap.parse_args()

    # Default the target from the direction rather than requiring the flag: passing
    # --direction a2s and silently encoding ambient latents would produce a cache that looks
    # right, trains without error, and teaches the model to generate the wrong modality.
    target = args.target or ("speech" if args.direction == "a2s" else "audio")
    target_attr = "speech_path" if target == "speech" else "audio_path"
    # Speech is left-aligned: its transcript is the supervision, so an over-long utterance
    # must keep its start. Ambient targets nearly fill the canvas, so centre is fine there.
    align = "left" if target == "speech" else "center"

    out_dir = Path(args.out_dir) / args.direction / args.split
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        log(f"ERROR: {out_dir} already populated. Pass --overwrite to replace.")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(args.direction, args.split)
    if args.limit:
        rows = rows[: args.limit]
    log(f"{args.direction}/{args.split}: {len(rows)} manifest rows")
    log(f"    target = {target} ({target_attr}), crop align = {align}")

    # ------------------------------------------------------------------ models
    log("Loading AudioLDM VAE ...")
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(
        project_paths.AUDIOLDM_MODEL_ID, subfolder="vae", torch_dtype=torch.float32
    )
    vae = vae.to(args.device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    scaling_factor = float(vae.config.scaling_factor)
    log(f"    vae scaling_factor={scaling_factor}")

    mel_cfg = MelConfig(norm_mode=args.norm_mode)
    extractor = MelExtractor(mel_cfg, device=args.device)

    # ------------------------------------------- discover latent shape once
    probe_wav = load_waveform_mono_16k(
        str(getattr(rows[0], target_attr)),
        target_seconds=AUDIOLDM_CLIP_SECONDS,
        align=align,
    )
    with torch.no_grad():
        probe_mel = extractor(probe_wav).unsqueeze(0).unsqueeze(0)
        probe_post = vae.encode(probe_mel).latent_dist
    latent_shape = tuple(probe_post.mean.shape[1:])
    log(f"    mel shape={tuple(probe_mel.shape[1:])} -> latent shape={latent_shape}")

    n = len(rows)
    mean_mm = np.lib.format.open_memmap(
        out_dir / "latents_mean.npy", mode="w+", dtype=np.float16, shape=(n, *latent_shape)
    )
    std_mm = np.lib.format.open_memmap(
        out_dir / "latents_std.npy", mode="w+", dtype=np.float16, shape=(n, *latent_shape)
    )

    # ------------------------------------------------------------------- loop
    index_rows: list[dict] = []
    dropped: list[dict] = []
    written = 0
    t0 = time.time()

    batch: list[tuple[int, np.ndarray]] = []

    def flush(batch: list[tuple[int, np.ndarray]]) -> None:
        nonlocal written
        if not batch:
            return
        wavs = torch.from_numpy(np.stack([w for _, w in batch]))
        with torch.no_grad():
            mels = extractor.batch(wavs.to(args.device))       # (B, frames, mels)
            posterior = vae.encode(mels.unsqueeze(1)).latent_dist
            mean = posterior.mean.cpu().numpy().astype(np.float16)
            std = posterior.std.cpu().numpy().astype(np.float16)
        for j, (_row_i, _w) in enumerate(batch):
            mean_mm[written] = mean[j]
            std_mm[written] = std[j]
            written += 1

    for row_i, row, wav, error in load_rows_prefetched(
        rows,
        num_workers=args.load_workers,
        prefetch=args.load_prefetch,
        target_attr=target_attr,
        align=align,
    ):
        # Unreadable or degenerate rows are dropped, not fatal -- one bad file should not
        # discard a 20-minute pass.
        if error is not None:
            dropped.append({"row_index": row_i, "sample_id": row.sample_id, "reason": error})
            continue

        # The latent slot for this row is `written + len(batch)`, assigned in flush().
        index_rows.append(
            {
                "latent_index": written + len(batch),
                "sample_id": row.sample_id,
                "audio_file": str(row.audio_path),
                "speech_file": str(row.speech_path),
                "speech_text": row.speech_text,
                "audio_cap": row.audio_cap,
                "speaker_gender": row.speaker_gender,
                "speaker_emotion": row.speaker_emotion,
                "manifest_row_index": row_i,
            }
        )
        batch.append((row_i, wav))

        if len(batch) >= args.batch_size:
            flush(batch)
            batch = []
            if written % (args.batch_size * 20) == 0:
                rate = written / max(time.time() - t0, 1e-6)
                eta = (len(rows) - written) / max(rate, 1e-6)
                log(f"    {written}/{len(rows)}  {rate:.1f} rows/s  eta {eta/60:.1f} min")

    flush(batch)

    # Trim the memmaps to what we actually wrote (dropped rows leave a tail).
    mean_mm.flush()
    std_mm.flush()
    del mean_mm, std_mm
    if written != n:
        log(f"    trimming memmaps from {n} to {written} rows")
        for name in ("latents_mean.npy", "latents_std.npy"):
            arr = np.load(out_dir / name, mmap_mode="r")
            trimmed = np.array(arr[:written])
            np.save(out_dir / name, trimmed)
            del arr, trimmed

    with (out_dir / "index.jsonl").open("w") as f:
        for r in index_rows:
            f.write(json.dumps(r) + "\n")

    meta = {
        "direction": args.direction,
        "split": args.split,
        # Recorded so the cache is self-describing: a2s latents built from the ambient clip
        # by mistake are byte-plausible and would train silently. Consumers should check it.
        "target": target,
        "target_field": target_attr,
        "crop_align": align,
        "manifest_rows": len(rows),
        "kept": written,
        "dropped": len(dropped),
        "drop_reasons": dropped[:50],
        "latent_shape": list(latent_shape),
        "vae_scaling_factor": scaling_factor,
        "audioldm_model_id": project_paths.AUDIOLDM_MODEL_ID,
        "mel_config": {
            "sample_rate": mel_cfg.sample_rate,
            "n_fft": mel_cfg.n_fft,
            "hop_length": mel_cfg.hop_length,
            "win_length": mel_cfg.win_length,
            "n_mels": mel_cfg.n_mels,
            "fmin": mel_cfg.fmin,
            "fmax": mel_cfg.fmax,
            "clip_seconds": mel_cfg.clip_seconds,
            "n_frames": mel_cfg.n_frames,
            "norm_mode": mel_cfg.norm_mode,
        },
        "elapsed_seconds": time.time() - t0,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    log(f"\nkept {written} / {len(rows)}  (dropped {len(dropped)})")
    log(f"wrote {out_dir}")
    log(f"elapsed {(time.time() - t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
