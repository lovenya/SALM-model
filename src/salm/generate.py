"""
Generate ambient audio, in any of the three rungs.

Modes:
  rung0  text  -> audio.  Conditions on the CLAP TEXT embedding of the dataset's
                  `audio_cap`. No training involved. This is the ceiling: how well can
                  AudioLDM render TTAS-like scenes when handed a perfect description?
  rung1  speech-> audio via CLAP AUDIO embedding of the speech, straight into frozen
                  AudioLDM. The control. Expect generic output, because CLAP encodes
                  "someone is speaking", not what they said -- confirming that is a
                  result, and it is what motivates the conditioner.
  s2a    speech-> audio via a trained checkpoint (frozen encoder -> conditioner -> U-Net).
  a2s    audio -> speech, the reverse direction. Same path, conditioned on the AMBIENT clip
         through the frozen CLAP tower. A checkpoint trained on cached features ('clapseq')
         has its arch remapped to the live tower here; see load_s2a_checkpoint.
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


def deranged_indices(n: int, seed: int) -> list[int]:
    """Deterministic no-fixed-point permutation for shuffled-content controls."""
    if n < 2:
        raise ValueError("shuffled content requires at least two selected samples")
    rng = np.random.default_rng(seed)
    base = np.arange(n)
    for _ in range(100):
        candidate = rng.permutation(n)
        if np.all(candidate != base):
            return candidate.tolist()
    # Deterministic fallback; a one-place rotation is always a derangement for n > 1.
    return np.roll(base, 1).tolist()


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
        inputs = self.processor(audio=wavs_48k, sampling_rate=self.SR, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # transformers 5.x returns BaseModelOutputWithPooling, with the (already
        # L2-normalized) embedding on .pooler_output; older versions returned a tensor.
        out = self.model.get_audio_features(**inputs)
        feats = out if torch.is_tensor(out) else out.pooler_output
        return F.normalize(feats, dim=-1)


def _derangement(n: int, seed: int) -> "np.ndarray":
    """
    A permutation of 0..n-1 with NO fixed points, for n >= 2.

    A plain randperm can return the identity, which makes a "shuffle" lane silently a no-op. That is
    the worst possible failure for an ablation: the lane would report full performance and be read as
    "order does not matter" when in fact nothing was scrambled. Caught exactly this with 1 s segments
    on a 3.75 s clip (3 chunks, seed 1234 drew the identity). A random n-cycle is deranged by
    construction and is still a valid permutation.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    perm = np.empty(n, dtype=np.int64)
    perm[order] = np.roll(order, -1)
    return perm


def degrade_speech(wav: np.ndarray, mode: str, seed: int) -> np.ndarray:
    """
    Destroy a chosen property of the speech while preserving the others.

    THE ABLATION THIS EXISTS FOR. The paper's central claim is that the model infers an *implied*
    scene from what was SAID -- "you almost hit me, slow down" has no sound-event noun in it, so the
    mapping requires a linguistic inference step. Nothing we currently measure supports that claim.
    Our model could equally be reading speaker identity, recording channel, background leakage, or
    prosodic energy. Every metric we have would look identical.

    Each mode removes one candidate explanation:

      reverse   Time-reverses the waveform. Destroys ALL linguistic content (phonemes run backwards
                and are not words) while preserving the long-term spectrum, the speaker's voice
                timbre, the channel, and the energy distribution almost exactly. This is the
                cleanest single control: if performance survives reversal, the model is not using
                language at all.
      shuffle   Chops into 250 ms segments and permutes them. Word identities largely survive at
                this scale but order and syntax do not. Separates "bag of words is enough" from
                "the model needs the sentence".
      lowpass   4th-order Butterworth at 400 Hz. Removes the formant/consonant detail that carries
                phonetic identity while keeping F0, rhythm and the energy envelope -- i.e. prosody
                without words. If performance survives this, prosody is doing the work.
      noise     Replaces speech with Gaussian noise matched to its per-clip RMS. Keeps only the
                energy level. A near-total destruction control that should land at the
                unconditional floor; it verifies the pipeline responds to the input at all.

    All modes preserve LENGTH exactly, so the encoder's valid-frame mask is unchanged and cannot
    itself explain a difference.
    """
    if mode == "none":
        return wav
    if mode == "reverse":
        return wav[::-1].copy()
    if mode == "shuffle":
        seg = 4000                       # 250 ms at 16 kHz
        n_full = (len(wav) // seg) * seg
        if n_full < 2 * seg:
            return wav                   # too short to shuffle meaningfully; leave it alone
        segs = wav[:n_full].reshape(-1, seg)
        return np.concatenate([segs[_derangement(len(segs), seed)].reshape(-1),
                               wav[n_full:]]).astype(np.float32)
    if mode == "local_reverse":
        # Reverse WITHIN each 250 ms window but keep the windows in order. Destroys local phonetic
        # identity while preserving global temporal order, the envelope, and the spectrum. This is
        # the complement of `shuffle` (which preserves local content and destroys order), and the
        # pair forms a 2x2 that separates the two factors `reverse` destroys together:
        #     none           local content OK,   order OK    -> 100%
        #     shuffle        local content OK,   order GONE  ->  38%
        #     local_reverse  local content GONE, order OK    ->    ?
        #     reverse        local content GONE, order GONE  ->  -2%
        seg = 4000
        n_full = (len(wav) // seg) * seg
        if n_full < seg:
            return wav[::-1].copy()
        out = wav[:n_full].reshape(-1, seg)[:, ::-1].reshape(-1)
        return np.concatenate([out, wav[n_full:][::-1]]).astype(np.float32)
    if mode == "shuffle_500ms":
        # Coarser than 250 ms so a short word is more likely to survive intact, but still fine enough
        # to give ~9 chunks at our 4.63 s mean duration. A 1 s variant was tried and DISCARDED: at
        # these durations it yields only 3-4 chunks, too few to scramble order meaningfully, and it
        # silently returned the input unchanged on the first clip tested.
        seg = 8000
        n_full = (len(wav) // seg) * seg
        if n_full < 2 * seg:
            return wav
        segs = wav[:n_full].reshape(-1, seg)
        return np.concatenate([segs[_derangement(len(segs), seed)].reshape(-1),
                               wav[n_full:]]).astype(np.float32)
    if mode == "lowpass":
        from scipy.signal import butter, filtfilt
        b, a = butter(4, 400.0 / (AUDIOLDM_SR / 2), btype="low")
        return filtfilt(b, a, wav).astype(np.float32)
    if mode == "noise":
        rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2))) or 1e-6
        g = np.random.default_rng(seed).standard_normal(len(wav)).astype(np.float32)
        return (g / (np.sqrt(np.mean(g ** 2)) + 1e-9) * rms).astype(np.float32)
    raise ValueError(f"unknown degrade mode {mode!r}")


def load_trained_unet(pipe, payload: dict, dtype: torch.dtype) -> bool:
    """
    Load the finetuned U-Net weights from a checkpoint into the pipeline.

    This existed only as a claim in a docstring until 2026-08-12: `load_s2a_checkpoint` restored
    the conditioner and never touched `pipe.unet`, so EVERY s2a sample generated before that date
    used the pristine pretrained AudioLDM U-Net. Since our runs train the U-Net too
    (`freeze_unet=False`, `lr_unet=1e-5`), those samples did not represent the trained model at all.

    Snapshot checkpoints store weights in fp16, so cast to float before loading and then return the
    module to the pipeline's dtype.

    Returns True if weights were loaded, False if the checkpoint had none.
    """
    if "unet" not in payload:
        log("    checkpoint contains NO unet weights (freeze_unet was true?) -- "
            "using pretrained AudioLDM U-Net")
        return False

    state = {
        k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
        for k, v in payload["unet"].items()
    }
    missing, unexpected = pipe.unet.load_state_dict(state, strict=False)
    if missing or unexpected:
        log(f"    WARNING: unet load had {len(missing)} missing / {len(unexpected)} unexpected keys")
    pipe.unet.to(dtype)
    log(f"    loaded TRAINED U-Net weights ({len(state)} tensors)")
    return True


def load_s2a_checkpoint(checkpoint: Path, device: str):
    """
    Restore the trained conditioner. The U-Net is loaded separately by `load_trained_unet`,
    because the pipeline must exist first.

    The architecture is rebuilt through the registry using the `arch` recorded in the
    checkpoint's own config, so any rung of the ladder loads without this function knowing
    which conditioner was used. An earlier version hardcoded the transformer conditioner and
    would silently fail to load an L0/L1/L3 checkpoint.
    """
    from salm.architectures import build as build_arch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    arch = cfg.get("arch", "s2a_audioldm_whisper_xformer")
    # A2S trains on PRECOMPUTED CLAP features ('clapseq'), whose encoder is a stand-in that
    # refuses to encode. Generation has no cache to read from, so swap in the live CLAP tower.
    # Same weights, same features -- see salm/encoders/precomputed.py.
    if "clapseq" in arch:
        arch = arch.replace("clapseq", "clap")
        log(f"    arch remapped to {arch} (live CLAP tower for generation)")
    # Checkpoints written before the mapper->conditioner rename use the "mapper" key.
    cond_state = payload.get("conditioner") or payload["mapper"]
    output_dim = cond_state["null_embedding"].shape[0]
    log(f"    checkpoint epoch={payload.get('epoch')} val_loss={payload.get('val_loss')}")
    log(f"    arch={arch} cond_dim={output_dim}")

    # Dropout off at inference; the saved value would otherwise perturb sampling. Both key
    # spellings are set so a checkpoint saved before the rename still resolves.
    cfg = {**cfg, "cond_dropout": 0.0, "mapper_dropout": 0.0}
    components = build_arch(arch, cfg, torch.device(device), output_dim=output_dim)
    encoder = components.encoder
    conditioner = components.conditioner

    # Snapshots are saved in fp16, so cast back to float before loading.
    state = {k: v.float() if v.is_floating_point() else v for k, v in cond_state.items()}
    conditioner.load_state_dict(state)
    conditioner.eval()

    return encoder, conditioner, payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["rung0", "rung1", "s2a", "a2s", "b0"])
    ap.add_argument("--checkpoint", default=None,
                    help="required for --mode s2a and --mode b0 (b0 needs the learned null)")
    ap.add_argument("--direction", default="s2a", choices=["s2a", "a2s"])
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--num-samples", type=int, default=8, help="-1 for the whole split")
    ap.add_argument("--start-index", type=int, default=0)
    # Language-degradation ablation. Applied to the speech waveform before the frozen encoder, so
    # the model and every other setting are untouched -- only the information in the input changes.
    ap.add_argument("--degrade", default="none",
                    choices=["none", "reverse", "shuffle", "shuffle_500ms", "local_reverse",
                             "lowpass", "noise"],
                    help="destroy one property of the input speech; see degrade_speech()")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--guidance-scale", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--copy-reference", action="store_true",
                    help="also write the reference TARGET audio for A/B listening -- the "
                         "ambient clip for s2a, the speech for a2s")
    ap.add_argument("--skip-unet-weights", action="store_true",
                    help="do NOT load the finetuned U-Net from the checkpoint; use the pretrained "
                         "AudioLDM U-Net instead. Isolates the conditioner's contribution, and "
                         "reproduces the behaviour of samples generated before 2026-08-12.")
    # 2026-08-24: oracle-conditioning ablation (mode=a2s only). Bypasses the CONDITIONER
    # entirely and feeds the true precomputed alignment embedding straight to class_labels,
    # so generation happens with the target O2 was training toward, not the conditioner's
    # attempt to reach it. Decomposes "the conditioner is bad at predicting the target" from
    # "the target itself is a poor generation signal" -- the two have very different fixes,
    # and confusing them would send the next round of work in the wrong direction.
    ap.add_argument("--oracle-align", default="none", choices=["none", "text", "speech"],
                    help="a2s only: skip the conditioner, feed the row's own "
                         "align_text.npy/align_speech.npy vector as class_labels instead")
    ap.add_argument("--cond-features-root", default=None,
                    help="a2s only: root of the precomputed cond cache "
                         "(default: OUTPUTS_ROOT/features_a2s_cond); needed to look up "
                         "--oracle-align vectors and, for --mode a2s with a 'clapseq' "
                         "checkpoint, has no other effect since generation always uses the "
                         "live CLAP tower")
    # Rank-2 sequence-conditioning falsification. Keep the old boolean for compatibility,
    # but make the actual content source explicit so oracle/retrieval/shuffled/zero outputs
    # cannot be mistaken for one another.
    ap.add_argument("--use-content-xattn", action="store_true",
                    help="deprecated compatibility flag; equivalent to "
                         "--content-source retrieval when no source is specified")
    ap.add_argument("--content-source", default="none",
                    choices=["none", "oracle", "retrieval", "shuffled", "zero"],
                    help="a2s Rank-2 content sequence: target speech (oracle), nearest "
                         "training speech by ambient CLAP (retrieval), another selected "
                         "row's target speech (shuffled), literal zeros, or no adapter")
    ap.add_argument("--ctc-checkpoint", default=None,
                    help="required with --use-content-xattn: the frozen CTC probe whose "
                         "encoder produces the content sequence (same checkpoint the "
                         "Rank-2 training run used)")
    ap.add_argument("--content-retrieval-root", default=None,
                    help="required with --use-content-xattn: root of the precomputed A2S "
                         "conditioning cache for the TRAIN split (default: "
                         "OUTPUTS_ROOT/features_a2s_cond) -- its cond_seq.npy, mean-pooled, "
                         "is the retrieval database of ambient scenes to match against")
    args = ap.parse_args()

    if args.use_content_xattn and args.content_source == "none":
        args.content_source = "retrieval"
        log("WARNING: --use-content-xattn is deprecated; using --content-source retrieval")
    use_content_xattn = args.content_source != "none"

    if args.mode in ("s2a", "a2s", "b0") and not args.checkpoint:
        log(f"ERROR: --mode {args.mode} requires --checkpoint")
        return 1
    if args.oracle_align != "none" and args.mode != "a2s":
        log(f"ERROR: --oracle-align is a2s-only, got --mode {args.mode}")
        return 1
    if use_content_xattn:
        if args.mode != "a2s":
            log(f"ERROR: --content-source is a2s-only, got --mode {args.mode}")
            return 1
        if not args.ctc_checkpoint:
            log("ERROR: non-none --content-source requires --ctc-checkpoint")
            return 1

    out_dir = Path(args.out_dir)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)

    rows = load_manifest(args.direction, args.split)
    rows = rows[args.start_index :]
    if args.num_samples > 0:
        rows = rows[: args.num_samples]
    shuffled_row_by_id = None
    if args.content_source == "shuffled":
        try:
            perm = deranged_indices(len(rows), args.seed + 7919)
        except ValueError as exc:
            log(f"ERROR: {exc}")
            return 1
        shuffled_row_by_id = {row.sample_id: rows[j] for row, j in zip(rows, perm)}
    log(f"{args.mode}: generating {len(rows)} samples from {args.direction}/{args.split}")

    # Oracle vectors, keyed by sample_id rather than positionally: index.jsonl's row order
    # can differ from load_manifest's if any rows were ever dropped by the latent precompute,
    # so a positional match would silently pair the wrong oracle vector with the wrong row
    # under --start-index/--num-samples slicing. sample_id is the one thing guaranteed
    # stable across both.
    oracle_by_id: dict[str, np.ndarray] | None = None
    if args.oracle_align != "none":
        cond_root = Path(
            args.cond_features_root or (project_paths.OUTPUTS_ROOT / "features_a2s_cond")
        )
        cond_dir = cond_root / args.split
        oracle_file = cond_dir / f"align_{args.oracle_align}.npy"
        # features_a2s_cond has no index.jsonl of its own -- it was built row-aligned to the
        # LATENTS' index.jsonl (scripts/precompute_latents.py --direction a2s), which is
        # exactly the file to key by.
        index_file = project_paths.FEATURES_ROOT / "a2s" / args.split / "index.jsonl"
        if not oracle_file.exists() or not index_file.exists():
            log(f"ERROR: --oracle-align {args.oracle_align} needs {oracle_file} and "
                f"{index_file}. Run scripts/a2s/precompute_a2s_conditioning.py --split "
                f"{args.split} first.")
            return 1
        oracle_arr = np.load(oracle_file)
        ids = [json.loads(l)["sample_id"] for l in index_file.read_text().splitlines() if l]
        if len(ids) != oracle_arr.shape[0]:
            log(f"ERROR: {index_file} has {len(ids)} rows but {oracle_file} has "
                f"{oracle_arr.shape[0]} -- caches out of sync, refusing to guess an alignment.")
            return 1
        oracle_by_id = {sid: oracle_arr[i] for i, sid in enumerate(ids)}
        missing = [r.sample_id for r in rows if r.sample_id not in oracle_by_id]
        if missing:
            log(f"ERROR: {len(missing)} requested rows have no oracle vector "
                f"(e.g. {missing[0]}) -- were they dropped by the latent precompute?")
            return 1
        log(f"    oracle conditioning ACTIVE: align_{args.oracle_align}, "
            f"{len(oracle_by_id)} vectors from {oracle_file}")

    # Retrieval database for --use-content-xattn: mean-pool the TRAIN split's already-cached
    # CLAP sequence (no new precompute needed -- scripts/a2s/precompute_a2s_conditioning.py
    # already wrote this for training) into one 768-d "scene key" per row, so a query's own
    # mean-pooled scene can be matched by cosine similarity at generation time.
    retrieval_keys = retrieval_rows = None
    if args.content_source == "retrieval":
        retrieval_root = Path(
            args.content_retrieval_root or (project_paths.OUTPUTS_ROOT / "features_a2s_cond")
        )
        train_cond_dir = retrieval_root / "train"
        train_index_file = project_paths.FEATURES_ROOT / "a2s" / "train" / "index.jsonl"
        if not (train_cond_dir / "cond_seq.npy").exists() or not train_index_file.exists():
            log(f"ERROR: --content-source retrieval needs {train_cond_dir / 'cond_seq.npy'} and "
                f"{train_index_file}. Run scripts/a2s/precompute_a2s_conditioning.py and "
                f"scripts/precompute_latents.py --direction a2s --target speech --split train.")
            return 1
        train_seq = np.load(train_cond_dir / "cond_seq.npy", mmap_mode="r")  # (N, 32, 768)
        retrieval_keys = torch.from_numpy(
            np.asarray(train_seq.mean(axis=1), dtype=np.float32)
        )
        retrieval_keys = F.normalize(retrieval_keys, dim=-1)
        retrieval_rows = [json.loads(l) for l in train_index_file.read_text().splitlines() if l]
        if len(retrieval_rows) != retrieval_keys.shape[0]:
            log(f"ERROR: {train_index_file} has {len(retrieval_rows)} rows but "
                f"cond_seq.npy has {retrieval_keys.shape[0]} -- caches out of sync.")
            return 1
        log(f"    content retrieval bank: {len(retrieval_rows)} training scenes")

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    log("Loading AudioLDM ...")
    pipe = load_pipeline(args.device, dtype)

    encoder = conditioner = clap = None
    unet_weights_loaded = False
    checkpoint_epoch = None
    # b0 loads the same checkpoint as s2a: it needs the conditioner for its learned null
    # embedding, and it must use the SAME U-Net, because B0 is the floor for THIS model rather
    # than a generic unconditional sample.
    ctc_head = content_extractor = None
    if args.mode in ("s2a", "a2s", "b0"):
        log("Loading trained checkpoint ...")
        encoder, conditioner, payload = load_s2a_checkpoint(Path(args.checkpoint), args.device)
        checkpoint_epoch = payload.get("epoch")
        if use_content_xattn:
            # MUST install the adapter's to_k_ip/to_v_ip submodules BEFORE loading the
            # checkpoint's unet weights -- load_trained_unet uses strict=False, so on a
            # vanilla (un-adapted) U-Net it would silently DROP every adapter weight as
            # "unexpected" instead of loading it, and generation would run as if Rank 2 had
            # never been trained. Same device-placement fix as training (afb41d7): the new
            # submodules are constructed on CPU and never moved until this explicit .to().
            from salm.a2s.content_adapter import install_content_cross_attention
            from salm.a2s.ctc_probe import SimpleCTCASR
            from salm.data.frontend import MelConfig, MelExtractor

            log(f"    loading frozen CTC probe from {args.ctc_checkpoint} ...")
            ctc_head = SimpleCTCASR().to(args.device).eval()
            ctc_payload = torch.load(args.ctc_checkpoint, map_location=args.device, weights_only=False)
            ctc_head.load_state_dict(ctc_payload["model"])
            for p in ctc_head.parameters():
                p.requires_grad_(False)
            content_extractor = MelExtractor(MelConfig(norm_mode="none"), device=args.device)

            adapted = install_content_cross_attention(pipe.unet, content_dim=ctc_head.encoder_dim)
            pipe.unet.to(args.device)
            log(f"    installed content cross-attention on {len(adapted)} attn2 layers "
                f"(content_dim={ctc_head.encoder_dim})")
        if args.skip_unet_weights:
            log("    --skip-unet-weights: keeping the PRETRAINED U-Net (ablation baseline)")
        else:
            unet_weights_loaded = load_trained_unet(pipe, payload, dtype)
    elif args.mode == "rung1":
        log("Loading CLAP audio tower (control) ...")
        clap = ClapAudioEncoder(args.device)

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    manifest_rows: list[dict] = []
    t0 = time.time()

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        content_seq = None
        content_meta = [
            {
                "content_source": "none",
                "content_sample_id": None,
                "content_speech_file": None,
                "content_retrieval_similarity": None,
            }
            for _ in batch
        ]

        # ---------------------------------------------- build the condition vector
        if args.mode == "b0":
            # B0 -- the UNCONDITIONAL floor. Every sample gets the model's own learned null
            # embedding, so the speech is never consulted. This is the empirical answer to "what
            # does zero information score?", and it is what makes every other number readable:
            # with the mismatched-target control in scripts/score_generations_clap.py, B0's MARGIN
            # is ~0 BY CONSTRUCTION, so any real model must beat it or it has learned nothing.
            #
            # Deliberately uses the trained conditioner's null embedding rather than zeros: that is
            # the vector this specific model actually falls back to under classifier-free guidance,
            # so it is the honest floor for THIS model rather than a generic one.
            cond = conditioner.null_condition(len(batch), torch.device(args.device))
            negative = None
        elif args.mode == "rung0":
            cond = clap_text_embeddings(pipe, [r.audio_cap for r in batch], args.device)
            negative = None
        elif args.mode == "a2s" and oracle_by_id is not None:
            # ORACLE ABLATION: skip the conditioner and the encoder entirely. class_labels
            # gets the row's own precomputed target embedding -- the exact vector O2 was
            # training the conditioner to reach -- so generation quality here reflects the
            # TARGET and the U-Net alone, with conditioner error removed from the picture.
            # Compared against the real a2s lane, this decomposes "conditioner underfits the
            # target" from "the target itself is a weak generation signal": if oracle
            # generations are ALSO not word-accurate, the target embedding (or the U-Net's
            # ability to decode it) is the bottleneck, not the conditioner -- in which case
            # improving the conditioner (better architecture, InfoNCE, more epochs) cannot
            # help, and effort belongs on the target/interface instead.
            oracle = np.stack([oracle_by_id[r.sample_id] for r in batch]).astype(np.float32)
            cond = F.normalize(torch.from_numpy(oracle).to(args.device), dim=-1)
            negative = conditioner.null_condition(len(batch), torch.device(args.device))
        elif args.mode == "a2s":
            # The mirror of the s2a branch: condition on the AMBIENT clip and let the U-Net
            # generate speech. Fixed 10.24 s canvas with centre alignment, matching exactly
            # what scripts/a2s/precompute_a2s_conditioning.py fed the encoder at train time --
            # a different crop here would put the condition out of distribution.
            wavs = [
                load_waveform_mono_16k(
                    str(r.audio_path), target_seconds=AUDIOLDM_CLIP_SECONDS
                )
                for r in batch
            ]
            lengths = [w.shape[0] for w in wavs]
            with torch.no_grad():
                feats, mask = encoder.encode_with_mask(
                    torch.from_numpy(np.stack(wavs)).to(args.device),
                    torch.tensor(lengths),
                )
                cond = conditioner(feats, mask=mask)
                negative = conditioner.null_condition(
                    len(batch), torch.device(args.device)
                )

                if args.content_source == "retrieval":
                    # Honest inference-time guess: use the speech paired with the closest
                    # TRAINING ambience in the correctly cached CLAP sequence space.
                    m = mask.unsqueeze(-1).to(feats.dtype)
                    query_key = (feats * m).sum(1) / m.sum(1).clamp(min=1.0)
                    query_key = F.normalize(query_key.float(), dim=-1).cpu()
                    sims = query_key @ retrieval_keys.T          # (b, N)
                    best = sims.argmax(dim=1).tolist()
                    source_rows = [retrieval_rows[i] for i in best]
                    content_meta = [
                        {
                            "content_source": "retrieval",
                            "content_sample_id": source_rows[i]["sample_id"],
                            "content_speech_file": source_rows[i]["speech_file"],
                            "content_retrieval_similarity": float(sims[i, best[i]].item()),
                        }
                        for i in range(len(batch))
                    ]
                elif args.content_source == "oracle":
                    source_rows = batch
                    content_meta = [
                        {
                            "content_source": "oracle",
                            "content_sample_id": row.sample_id,
                            "content_speech_file": str(row.speech_path),
                            "content_retrieval_similarity": None,
                        }
                        for row in batch
                    ]
                elif args.content_source == "shuffled":
                    source_rows = [shuffled_row_by_id[row.sample_id] for row in batch]
                    content_meta = [
                        {
                            "content_source": "shuffled",
                            "content_sample_id": row.sample_id,
                            "content_speech_file": str(row.speech_path),
                            "content_retrieval_similarity": None,
                        }
                        for row in source_rows
                    ]
                elif args.content_source == "zero":
                    from salm.a2s.ctc_probe import CTC_FRAMES_OUT

                    content_seq = torch.zeros(
                        len(batch), CTC_FRAMES_OUT, ctc_head.encoder_dim,
                        device=args.device, dtype=torch.float32,
                    )
                    content_meta = [
                        {
                            "content_source": "zero",
                            "content_sample_id": None,
                            "content_speech_file": None,
                            "content_retrieval_similarity": None,
                        }
                        for _ in batch
                    ]

                if args.content_source in ("retrieval", "oracle", "shuffled"):
                    content_wavs = [
                        load_waveform_mono_16k(
                            str(row["speech_file"] if isinstance(row, dict) else row.speech_path),
                            target_seconds=AUDIOLDM_CLIP_SECONDS,
                            align="left",
                        )
                        for row in source_rows
                    ]
                    content_wav_t = torch.from_numpy(np.stack(content_wavs)).float().to(args.device)
                    content_mel = content_extractor.batch(content_wav_t)
                    content_seq = ctc_head.encode(content_mel)   # (b, 256, 512)
        else:
            wavs = [load_waveform_mono_16k(str(r.speech_path)) for r in batch]
            if args.degrade != "none":
                wavs = [degrade_speech(w, args.degrade, args.seed + i)
                        for i, w in enumerate(wavs)]

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
                    cond = conditioner(feats, mask=mask)
                    negative = conditioner.null_condition(len(batch), torch.device(args.device))

        cond = cond.to(dtype=pipe.unet.dtype)
        if negative is not None:
            negative = negative.to(dtype=pipe.unet.dtype)

        # ------------------------------------- inject content, if this checkpoint has it
        # AudioLDMPipeline.__call__ hardcodes `encoder_hidden_states=None` on every one of
        # its own internal unet(...) calls (verified against installed diffusers source,
        # pipeline_audioldm.py) -- there is no pipeline argument to pass it through. Rather
        # than reimplement the pipeline's own (already correct) guidance/scheduling/decode
        # loop by hand, wrap unet.forward for the duration of this one call, so every
        # internal call the pipeline makes receives our content tuple instead of None, and
        # everything else about the pipeline's tested behaviour is untouched. Restored
        # immediately after, so a later batch without content-xattn is unaffected.
        restore_forward = None
        if content_seq is not None:
            original_forward = pipe.unet.forward

            def patched_forward(sample, timestep, encoder_hidden_states=None, **kwargs):
                from salm.a2s.content_adapter import content_encoder_hidden_states

                # `sample`'s batch dim is doubled to 2B under classifier-free guidance
                # (torch.cat([latents]*2) inside the pipeline) -- match it here. Content is
                # NEVER dropped for a "null" branch (training never saw a null-content
                # variant either -- content_kwargs is always the real tuple, see
                # train_ldm.py's diagnose()), so both halves get the identical content.
                reps = sample.shape[0] // content_seq.shape[0]
                seq = content_seq.repeat(reps, 1, 1) if reps > 1 else content_seq
                # to_k_ip/to_v_ip were cast to the pipeline's working dtype (fp16 on cuda)
                # by load_trained_unet's pipe.unet.to(dtype); the content sequence itself
                # came from the CTC probe, which is never cast and stays fp32. Match
                # `sample`'s dtype (the actual dtype every other tensor in this forward
                # call is already using) rather than hardcoding fp16, so this also works
                # unchanged on CPU (--device cpu, dtype=float32).
                seq = seq.to(dtype=sample.dtype)
                return original_forward(
                    sample, timestep,
                    encoder_hidden_states=content_encoder_hidden_states(seq),
                    **kwargs,
                )

            pipe.unet.forward = patched_forward
            restore_forward = original_forward

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
        if restore_forward is not None:
            pipe.unet.forward = restore_forward
        audios = result.audios

        for row, audio, row_content_meta in zip(batch, audios, content_meta):
            gen_path = out_dir / "audio" / f"{row.sample_id}.wav"
            sf.write(gen_path, audio, AUDIOLDM_SR)

            # The reference is whatever this direction is trying to GENERATE: ambience for
            # S2A, speech for A2S. Getting this backwards would score every A2S clip against
            # its own input.
            is_a2s = args.direction == "a2s"
            ref_source = row.speech_path if is_a2s else row.audio_path
            ref_path = None
            if args.copy_reference:
                ref = load_waveform_mono_16k(
                    str(ref_source),
                    target_seconds=AUDIOLDM_CLIP_SECONDS,
                    align="left" if is_a2s else "center",
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
                    # Named for the S2A case, but it always means "the target". The metric
                    # suite maps this to source.audio_file.
                    "reference_audio_file": str(ref_source),
                    "condition_file": str(row.audio_path if is_a2s else row.speech_path),
                    "reference_copy": str(ref_path) if ref_path else None,
                    "speech_file": str(row.speech_path),
                    "audio_cap": row.audio_cap,
                    "speech_text": row.speech_text,
                    "generated_rms": float(np.sqrt(np.mean(np.square(audio)))),
                    **row_content_meta,
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
        "degrade": args.degrade,
        "seed": args.seed,
        "audioldm_model_id": project_paths.AUDIOLDM_MODEL_ID,
        # Provenance: which weights actually produced these files. Without this, a directory of
        # wavs is ambiguous about whether the trained U-Net was used.
        "checkpoint_epoch": checkpoint_epoch,
        "unet_weights_loaded": unet_weights_loaded,
        "skip_unet_weights": bool(args.skip_unet_weights),
        # "none" = the real conditioner's own output was used; anything else means the
        # CONDITIONER was bypassed and the row's precomputed target embedding was fed to
        # class_labels directly -- see --oracle-align's help text for what that isolates.
        "oracle_align": args.oracle_align,
        "content_source": args.content_source,
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
