"""
Is the model actually using the speech? (diagnostics D1 and D2)

The diffusion loss cannot answer this. eps-MSE averages over random (t, eps), and that variance
dominates: our measured loss tracks the noise schedule's signal:noise almost perfectly and inversely
(0.559 at t=125 down to 0.001 at t=875), so anything the conditioner contributes is a sub-percent
correction on a large constant. Over 91 epochs both train and val loss stayed flat at ~0.198 with
sigma 0.007 and no trend -- equally consistent with "learning fine" and "ignoring the condition".

These two tests distinguish those cases directly:

  D1  condition shuffle. Give each latent ANOTHER sample's conditioning vector. If the loss does
      not get worse, the condition carries no sample-specific information -- the model is ignoring
      which utterance it was given.

  D2  conditional-vs-null gap. Measure ||eps(c) - eps(null)|| relative to ||eps(c)||. If ~0, the
      condition has no influence on the prediction at all, whatever the loss says. This is also
      exactly the quantity classifier-free guidance amplifies, so a zero gap means CFG is a no-op.

D2 == 0 implies D1 == 0, but D1 can be 0 while D2 is not: the model may exploit the conditioning
distribution generically without using the specific utterance.

CRITICAL CONTROL: for a given batch and timestep, all three variants (correct / shuffled / null) see
the SAME noise and the SAME timestep. Otherwise the differences would be resampling noise rather
than the effect of conditioning.

Reported per timestep bucket as well as overall, because conditioning matters most where signal is
high (low t) and the global average hides it.

Usage:
    python scripts/diagnose_conditioning.py --checkpoint <ckpt.pt> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from configs import paths as project_paths  # noqa: E402
from salm.data.dataset import S2ALatentDataset, collate  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--direction", default="s2a")
    ap.add_argument("--split", default="val")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-batches", type=int, default=0, help="0 = whole split")
    ap.add_argument("--buckets", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # ------------------------------------------------------------------- data
    feat = project_paths.FEATURES_ROOT / args.direction / args.split
    clap = project_paths.FEATURES_CLAP_ROOT / args.direction / args.split
    clap_dir = clap if (clap / "clap_target.npy").exists() else None
    ds = S2ALatentDataset(feat, clap_dir=clap_dir)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate
    )
    log(f"val rows {len(ds)}  latent {ds.latent_shape}  scaling {ds.vae_scaling_factor:.4f}")

    # ----------------------------------------------------------------- models
    from diffusers import DDPMScheduler, UNet2DConditionModel
    from salm.architectures import build as build_arch

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    arch = cfg.get("arch", "s2a_audioldm_whisper_xformer")
    epoch = payload.get("epoch")
    log(f"checkpoint epoch={epoch} arch={arch}")

    unet = UNet2DConditionModel.from_pretrained(
        project_paths.AUDIOLDM_MODEL_ID, subfolder="unet", torch_dtype=torch.float32
    ).to(device).eval()
    sched = DDPMScheduler.from_pretrained(project_paths.AUDIOLDM_MODEL_ID, subfolder="scheduler")

    # The trained U-Net matters here: diagnosing the pretrained U-Net would answer a different
    # question than the one we are asking about our run.
    unet_trained = "unet" in payload
    if unet_trained:
        st = {k: (v.float() if v.is_floating_point() else v) for k, v in payload["unet"].items()}
        unet.load_state_dict(st, strict=False)
        log("loaded TRAINED U-Net weights")
    else:
        log("WARNING checkpoint has no U-Net weights; diagnosing the PRETRAINED U-Net")

    cond_dim = unet.config.projection_class_embeddings_input_dim
    comps = build_arch(arch, {**cfg, "cond_dropout": 0.0, "mapper_dropout": 0.0}, device,
                       output_dim=cond_dim)
    encoder, conditioner = comps.encoder, comps.conditioner
    cond_state = payload.get("conditioner") or payload["mapper"]
    conditioner.load_state_dict(
        {k: (v.float() if v.is_floating_point() else v) for k, v in cond_state.items()}
    )
    conditioner.eval()

    T = sched.config.num_train_timesteps
    edges = [round(i * T / args.buckets) for i in range(args.buckets + 1)]
    bucket_ts = [((lo + hi) // 2, lo, hi) for lo, hi in zip(edges[:-1], edges[1:])]
    log(f"timestep buckets: {[f'{lo}-{hi} (t={t})' for t, lo, hi in bucket_ts]}")

    # accumulators keyed by timestep
    acc = {t: {k: 0.0 for k in ("loss_cond", "loss_shuf", "loss_null",
                                "gap_null", "gap_shuf", "n")} for t, _, _ in bucket_ts}
    cos_sum, cos_n = 0.0, 0

    t0 = time.time()
    nb = 0
    with torch.no_grad():
        for batch in loader:
            n = batch["speech"].shape[0]
            if n < 2:
                continue  # shuffling is meaningless for a single sample

            feats, mask = encoder.encode_with_mask(
                batch["speech"].to(device), batch["speech_length"]
            )
            cond = conditioner(feats, mask=mask)

            # Shuffle so NO sample keeps its own condition. A plain randperm can leave fixed
            # points, which would dilute D1 by giving some samples the correct condition anyway.
            # Building a single random n-cycle guarantees a derangement for n >= 2 and is still a
            # valid permutation -- patching fixed points in place is not (it can duplicate indices).
            order = torch.randperm(n, device=device)
            perm = torch.empty_like(order)
            perm[order] = order.roll(-1)
            shuf = cond[perm]
            null = conditioner.null_condition(n, device)

            if "clap_target" in batch:
                tgt = F.normalize(batch["clap_target"].to(device).float(), dim=-1)
                cos_sum += (cond.float() * tgt).sum(-1).sum().item()
                cos_n += n

            mean = batch["latent_mean"].to(device)
            std = batch["latent_std"].to(device)
            latents = (mean + std * torch.randn_like(std)) * ds.vae_scaling_factor
            noise = torch.randn_like(latents)   # SAME noise across all three variants

            for t, _, _ in bucket_ts:
                ts = torch.full((n,), t, device=device, dtype=torch.long)
                noisy = sched.add_noise(latents, noise, ts)

                e_c = unet(noisy, ts, encoder_hidden_states=None, class_labels=cond).sample
                e_s = unet(noisy, ts, encoder_hidden_states=None, class_labels=shuf).sample
                e_n = unet(noisy, ts, encoder_hidden_states=None, class_labels=null).sample

                a = acc[t]
                a["loss_cond"] += F.mse_loss(e_c, noise).item() * n
                a["loss_shuf"] += F.mse_loss(e_s, noise).item() * n
                a["loss_null"] += F.mse_loss(e_n, noise).item() * n
                # Relative gaps, so the numbers are comparable across timesteps despite the
                # prediction magnitude changing enormously with t.
                denom = e_c.flatten(1).norm(dim=1).clamp(min=1e-8)
                a["gap_null"] += ((e_c - e_n).flatten(1).norm(dim=1) / denom).sum().item()
                a["gap_shuf"] += ((e_c - e_s).flatten(1).norm(dim=1) / denom).sum().item()
                a["n"] += n

            nb += 1
            if args.max_batches and nb >= args.max_batches:
                break

    # ------------------------------------------------------------------ report
    log(f"\nbatches {nb}  samples {acc[bucket_ts[0][0]]['n']}  "
        f"elapsed {time.time()-t0:.1f}s")

    hdr = (f"\n{'t':>5} {'loss_cond':>10} {'loss_shuf':>10} {'D1 delta':>10} "
           f"{'loss_null':>10} {'D2 gap':>9} {'shuf gap':>9}")
    log(hdr)
    log("-" * len(hdr))
    results = {}
    for t, lo, hi in bucket_ts:
        a = acc[t]
        k = a["n"] or 1
        lc, ls, ln = a["loss_cond"]/k, a["loss_shuf"]/k, a["loss_null"]/k
        gn, gs = a["gap_null"]/k, a["gap_shuf"]/k
        results[f"t{lo}_{hi}"] = {
            "timestep": t, "loss_cond": lc, "loss_shuf": ls, "d1_delta": ls-lc,
            "loss_null": ln, "d2_gap_null_rel": gn, "gap_shuffled_rel": gs,
        }
        log(f"{t:>5} {lc:>10.4f} {ls:>10.4f} {ls-lc:>+10.4f} {ln:>10.4f} "
            f"{gn:>9.4f} {gs:>9.4f}")

    d1 = sum(r["d1_delta"] for r in results.values()) / len(results)
    d2 = sum(r["d2_gap_null_rel"] for r in results.values()) / len(results)
    # Low-t is where conditioning should matter most, so report it separately.
    low = results[f"t{bucket_ts[0][1]}_{bucket_ts[0][2]}"]

    log(f"\nmean D1 delta (shuffled - correct): {d1:+.5f}")
    log(f"mean D2 relative null gap:          {d2:.5f}")
    log(f"low-t bucket D1 delta:              {low['d1_delta']:+.5f}")
    log(f"low-t bucket D2 gap:                {low['d2_gap_null_rel']:.5f}")
    if cos_n:
        log(f"val CLAP cosine:                    {cos_sum/cos_n:+.5f}  (chance +/-0.0442)")

    log("\nVERDICT")
    if d2 < 0.01:
        log("  D2 ~ 0: the conditioning has essentially NO influence on the prediction.")
        log("  The U-Net is ignoring the condition entirely, so CFG is also a no-op.")
        log("  More epochs and more conditioner capacity are both pointless. Change the")
        log("  objective (auxiliary alignment loss) or the conditioning interface.")
    elif d1 < 0.001:
        log("  D2 > 0 but D1 ~ 0: the condition changes the prediction, but swapping in ANOTHER")
        log("  sample's condition costs nothing -- so it carries no sample-specific information.")
        log("  The model has learned the conditioning DISTRIBUTION, not the mapping.")
    else:
        log(f"  D1 = {d1:+.5f} > 0: shuffling the conditioning measurably hurts, so the model IS")
        log("  using which utterance it was given. The mapping is being learned; the flat loss")
        log("  curve was hiding it. Scaling up is justified.")

    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": epoch,
        "arch": arch,
        "unet_weights_loaded": unet_trained,
        "split": args.split,
        "batches": nb,
        "samples": acc[bucket_ts[0][0]]["n"],
        "mean_d1_delta": d1,
        "mean_d2_gap": d2,
        "low_t_d1_delta": low["d1_delta"],
        "low_t_d2_gap": low["d2_gap_null_rel"],
        "val_clap_cos": (cos_sum / cos_n) if cos_n else None,
        "per_bucket": results,
        "elapsed_seconds": time.time() - t0,
    }
    (out_dir / "diagnostics.json").write_text(json.dumps(summary, indent=2))
    log(f"\nwrote {out_dir/'diagnostics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
