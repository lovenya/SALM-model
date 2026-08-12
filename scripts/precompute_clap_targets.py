"""
Precompute CLAP audio embeddings for the TARGET ambient clips.

Why this exists: the diffusion training loss is flat and therefore useless as a progress
signal (measured: val loss 0.1905 -> 0.1949 over 12 epochs, best at epoch 0). That is
normal for eps-MSE averaged over random timesteps, but it means we cannot tell whether a
100-epoch run is better than a 12-epoch one.

These embeddings give us a signal that *does* move: during validation we compare the
mapper's output vector against the CLAP embedding of the ground-truth ambient clip. High
cosine means "the mapper is pointing at the right scene", which is exactly the thing the
model has to learn and exactly what the diffusion loss hides.

Uses `laion/clap-htsat-unfused` -- the NON-fusion variant, matching what AudioLDM was
trained against. The locally cached 630k-audioset-FUSION-best.pt is a different
architecture with a different embedding space and must not be substituted.

Alignment guarantee: rows are written in the same order as `index.jsonl` from
`precompute_latents.py`, and the sample_id of every row is cross-checked, so
`clap[i]` always corresponds to `latents_mean[i]`.

Usage:
    python scripts/precompute_clap_targets.py --direction s2a --split train
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from configs import paths as project_paths  # noqa: E402
from salm.data.frontend import load_waveform_mono_16k  # noqa: E402

# LAION-CLAP operates at 48 kHz, unlike AudioLDM's 16 kHz.
CLAP_SR = 48000


def log(msg: str) -> None:
    print(msg, flush=True)


def clap_audio_embeddings(model, inputs: dict) -> torch.Tensor:
    """
    Get the CLAP audio embedding as a plain tensor.

    In transformers 5.x, `ClapModel.get_audio_features` returns a
    `BaseModelOutputWithPooling`, not a tensor, and the embedding sits on
    `.pooler_output` -- already L2-normalized by the model itself:

        audio_features = self.audio_projection(audio_outputs.pooler_output)
        audio_outputs.pooler_output = F.normalize(audio_features, dim=-1)
        return audio_outputs

    (verified in transformers/models/clap/modeling_clap.py). Older versions returned the
    tensor directly, so both shapes are handled. The final `normalize` is therefore a
    no-op on 5.x but keeps the contract explicit and correct on older builds.
    """
    out = model.get_audio_features(**inputs)
    feats = out if torch.is_tensor(out) else out.pooler_output
    return F.normalize(feats, dim=-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default="s2a", choices=["s2a", "a2s"])
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument(
        "--features-dir",
        default=None,
        help="latents dir holding index.jsonl (default: FEATURES_ROOT/<dir>/<split>)",
    )
    ap.add_argument("--out-dir", default=None, help="default: FEATURES_CLAP_ROOT/<dir>/<split>")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    features_dir = Path(
        args.features_dir or project_paths.FEATURES_ROOT / args.direction / args.split
    )
    out_dir = Path(
        args.out_dir or project_paths.FEATURES_CLAP_ROOT / args.direction / args.split
    )

    index_path = features_dir / "index.jsonl"
    if not index_path.exists():
        log(f"ERROR: {index_path} not found. Run precompute_latents.py first.")
        return 1

    if (out_dir / "clap_target.npy").exists() and not args.overwrite:
        log(f"ERROR: {out_dir} already populated. Pass --overwrite.")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in index_path.read_text().splitlines() if l.strip()]
    log(f"{args.direction}/{args.split}: {len(rows)} rows from {index_path}")

    # Rows must already be ordered by latent_index; assert rather than assume, because a
    # silent misalignment here would make the cosine metric meaningless.
    for i, r in enumerate(rows):
        if r["latent_index"] != i:
            log(f"ERROR: index.jsonl not ordered by latent_index at row {i} "
                f"(got {r['latent_index']}). Cannot guarantee alignment.")
            return 1

    log("Loading CLAP (laion/clap-htsat-unfused) ...")
    from transformers import ClapModel, ClapProcessor

    processor = ClapProcessor.from_pretrained(project_paths.CLAP_MODEL_ID)
    model = ClapModel.from_pretrained(project_paths.CLAP_MODEL_ID).to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    dim = int(model.config.projection_dim)
    log(f"    projection_dim={dim}")

    embeddings = np.zeros((len(rows), dim), dtype=np.float32)
    t0 = time.time()

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        wavs = []
        for r in batch:
            w = load_waveform_mono_16k(r["audio_file"])
            wavs.append(
                librosa.resample(w, orig_sr=16000, target_sr=CLAP_SR, res_type="soxr_hq")
            )

        inputs = processor(audio=wavs, sampling_rate=CLAP_SR, return_tensors="pt")
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = clap_audio_embeddings(model, inputs)
        embeddings[start : start + len(batch)] = feats.float().cpu().numpy()

        done = start + len(batch)
        if done % (args.batch_size * 20) == 0 or done == len(rows):
            rate = done / max(time.time() - t0, 1e-6)
            log(f"    {done}/{len(rows)}  {rate:.1f} rows/s  eta "
                f"{(len(rows)-done)/max(rate,1e-6)/60:.1f} min")

    np.save(out_dir / "clap_target.npy", embeddings)

    norms = np.linalg.norm(embeddings, axis=1)
    # Pairwise cosine spread tells us whether CLAP even separates these scenes. If the
    # mean off-diagonal similarity is very high, the targets are nearly identical and the
    # cosine metric would be uninformative -- worth knowing before trusting it.
    sample = embeddings[:: max(1, len(embeddings) // 512)][:512]
    sim = sample @ sample.T
    off = sim[~np.eye(len(sim), dtype=bool)]

    meta = {
        "direction": args.direction,
        "split": args.split,
        "rows": len(rows),
        "dim": dim,
        "clap_model_id": project_paths.CLAP_MODEL_ID,
        "features_dir": str(features_dir),
        "l2_norm_min": float(norms.min()),
        "l2_norm_max": float(norms.max()),
        "pairwise_cosine_mean": float(off.mean()),
        "pairwise_cosine_std": float(off.std()),
        "elapsed_seconds": time.time() - t0,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    log(f"\nwrote {out_dir}/clap_target.npy  shape={embeddings.shape}")
    log(f"target spread: pairwise cosine mean={meta['pairwise_cosine_mean']:.4f} "
        f"std={meta['pairwise_cosine_std']:.4f}")
    if meta["pairwise_cosine_mean"] > 0.9:
        log("WARNING: target embeddings are nearly collinear; the cosine metric will have")
        log("         little dynamic range and a high score would not mean much.")
    log(f"elapsed {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
