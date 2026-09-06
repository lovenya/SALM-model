#!/usr/bin/env python
"""
Is CosyVoice3's speech-token space better structured for language modelling than our codes?

THE QUESTION THIS ANSWERS. Measured 2026-09-06: `scene_gap` accounts for 1.8-2.8% of the
content LM's val loss. The other 97%+ is LANGUAGE MODELLING over code sequences, learned from
17,596 utterances. Every conditioning intervention tried (CFG 442847, plan vector 443751,
guidance) moved little, because the scene simply does not determine word 7. So the bottleneck
is the quality of the LM over the token space -- and that makes the token space itself a live
variable for the first time.

CosyVoice3's tokenizer is a plausible replacement: SINGLE stream (verified from the ONNX
signature -- one `indices` output, not the 8-32 parallel codebooks an RVQ acoustic codec would
give, which would need a coarse/fine cascade and a pipeline rebuild), at 25 Hz, matching our
latent rate exactly. It also ships `llm.pt`, a language model pretrained over that token space
on far more speech than we have -- precisely the prior the 97% measurement says we lack, and
one we cannot otherwise obtain since `datasets/` holds only TTAS.

WHAT IS MEASURED, and why it is the right cheap gate. Not resynthesis quality: CosyVoice is a
shipped TTS system and its vocoder works. The open question is whether ITS tokens are easier to
model than OURS. So both token streams are extracted for the SAME utterances and compared on
how much a sequence model can gain from context:

    unigram loss  - trigram loss     (nats/token)

That reduction is the structure a language model has to exploit. Absolute losses are NOT
comparable across streams with different vocabulary sizes; the reduction from each stream's own
unigram baseline is. This is the same discipline the plan-vector post-mortem forced: score
against the baseline that ignores the thing being tested.

Usage:
  python scripts/a2s/cosyvoice_token_gate.py --num-train 2000 --num-val 500
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from configs import paths as project_paths  # noqa: E402

TOKENIZER = ("/scratch/l/lovenya/salm_migration_clean/models/cascaded/tts/"
             "fun-cosyvoice3-0.5b-2512/speech_tokenizer_v3.onnx")


def whisper_log_mel(wav: np.ndarray, n_mels: int = 128) -> np.ndarray:
    """
    Whisper's own log-mel, reimplemented rather than imported.

    The `whisper` package is not installed and installing is the user's call, but the transform
    is fully specified: 16 kHz, n_fft 400, hop 160, `n_mels` filters, magnitude squared, log10,
    clamped to within 8 dB of the maximum, then scaled to roughly [-1, 1] as (x + 4) / 4.
    CosyVoice's frontend calls exactly `whisper.log_mel_spectrogram(speech, n_mels=128)`, so
    any deviation here silently feeds the tokenizer out-of-distribution features and every
    number below would be wrong while looking plausible.
    """
    import librosa

    x = torch.from_numpy(wav).float()
    window = torch.hann_window(400)
    stft = torch.stft(x, 400, 160, window=window, return_complex=True)
    mag = stft[..., :-1].abs() ** 2
    filters = torch.from_numpy(
        librosa.filters.mel(sr=16000, n_fft=400, n_mels=n_mels)
    ).float()
    mel = filters @ mag
    log_spec = torch.clamp(mel, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).numpy()


def ngram_reduction(seqs: list[np.ndarray], vocab: int, order: int = 3) -> dict:
    """
    Held-out loss of unigram / bigram / trigram models fitted on the same sequences.

    Add-one smoothing, and a strict 80/20 split so the trigram counts cannot memorise the
    evaluation rows. What matters is the REDUCTION from unigram, which is comparable across
    streams of different vocabulary size in a way the absolute losses are not.
    """
    n_fit = max(1, int(0.8 * len(seqs)))
    fit, held = seqs[:n_fit], seqs[n_fit:]
    if not held:
        held = fit[-1:]

    uni = np.ones(vocab)
    bi: dict[int, Counter] = defaultdict(Counter)
    tri: dict[tuple[int, int], Counter] = defaultdict(Counter)
    for s in fit:
        for i, t in enumerate(s):
            uni[t] += 1
            if i >= 1:
                bi[int(s[i - 1])][int(t)] += 1
            if i >= 2:
                tri[(int(s[i - 2]), int(s[i - 1]))][int(t)] += 1
    uni_p = uni / uni.sum()

    def loss(fn) -> float:
        tot, n = 0.0, 0
        for s in held:
            for i, t in enumerate(s):
                tot += -np.log(fn(s, i, int(t)))
                n += 1
        return tot / max(n, 1)

    def p_uni(s, i, t):
        return uni_p[t]

    def p_bi(s, i, t):
        if i < 1:
            return uni_p[t]
        c = bi.get(int(s[i - 1]))
        if not c:
            return uni_p[t]
        return (c.get(t, 0) + 1.0) / (sum(c.values()) + vocab)

    def p_tri(s, i, t):
        if i < 2:
            return p_bi(s, i, t)
        c = tri.get((int(s[i - 2]), int(s[i - 1])))
        if not c:
            return p_bi(s, i, t)
        return (c.get(t, 0) + 1.0) / (sum(c.values()) + vocab)

    lu, lb, lt = loss(p_uni), loss(p_bi), loss(p_tri)
    return {"vocab": vocab, "unigram": lu, "bigram": lb, "trigram": lt,
            "reduction_bigram": lu - lb, "reduction_trigram": lu - lt,
            "relative_reduction": (lu - lt) / lu if lu > 0 else float("nan")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-train", type=int, default=2000)
    ap.add_argument("--num-val", type=int, default=500)
    ap.add_argument("--k", type=int, default=2048, help="our codebook, for the comparison")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import onnxruntime as ort

    prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(TOKENIZER, providers=prov)
    print(f"tokenizer providers in use: {sess.get_providers()}", flush=True)
    in0, in1 = [i.name for i in sess.get_inputs()]

    idx = [json.loads(l) for l in
           (project_paths.OUTPUTS_ROOT / "features/a2s/train/index.jsonl").read_text().splitlines()
           if l.strip()]
    n = min(args.num_train + args.num_val, len(idx))
    print(f"tokenizing {n} clips ...", flush=True)

    cosy: list[np.ndarray] = []
    for i in range(n):
        wav, sr = sf.read(idx[i]["speech_file"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            raise SystemExit(f"expected 16 kHz, got {sr} -- the tokenizer assumes 16 kHz")
        feat = whisper_log_mel(wav)[None].astype(np.float32)
        out = sess.run(None, {in0: feat, in1: np.array([feat.shape[2]], dtype=np.int32)})[0]
        cosy.append(np.asarray(out).flatten().astype(np.int64))
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n}", flush=True)

    ours_all = np.load(project_paths.OUTPUTS_ROOT / "content_codes" /
                       f"codes_codebook_k{args.k}_train.npy")[:n]
    ours = [row for row in ours_all]

    cosy_vocab = int(max(int(s.max()) for s in cosy)) + 1
    lens_c = np.array([len(s) for s in cosy])
    lens_o = np.array([len(s) for s in ours])
    print()
    print(f"  CosyVoice tokens: vocab seen {cosy_vocab}, length {lens_c.mean():.1f} "
          f"(min {lens_c.min()}, max {lens_c.max()})")
    print(f"  our k{args.k} codes : vocab {args.k}, length {lens_o.mean():.1f}")
    for name, seqs in (("cosyvoice", cosy), (f"ours_k{args.k}", ours)):
        rl = np.mean([1.0 / max(1e-9, (np.diff(s) != 0).mean()) if len(s) > 1 else 1.0
                      for s in seqs])
        print(f"  {name:14s} mean run {rl:.2f} frames")

    res = {}
    print("\n  fitting n-gram models (held-out 20%) ...", flush=True)
    res["cosyvoice"] = ngram_reduction(cosy, cosy_vocab)
    res[f"ours_k{args.k}"] = ngram_reduction(ours, args.k)

    print()
    print("  HOW MUCH STRUCTURE IS THERE FOR A SEQUENCE MODEL TO EXPLOIT?")
    print("  (absolute loss is not comparable across vocabulary sizes; the REDUCTION is)")
    print()
    print(f"  {'stream':14s} {'vocab':>7s} {'unigram':>9s} {'trigram':>9s} "
          f"{'reduction':>10s} {'relative':>9s}")
    for k, r in res.items():
        print(f"  {k:14s} {r['vocab']:7d} {r['unigram']:9.4f} {r['trigram']:9.4f} "
              f"{r['reduction_trigram']:10.4f} {r['relative_reduction']:8.1%}")
    print()
    a = res["cosyvoice"]["relative_reduction"]
    b = res[f"ours_k{args.k}"]["relative_reduction"]
    if a > b + 0.05:
        print(f"  -> CosyVoice tokens are MORE structured ({a:.1%} vs {b:.1%}). An LM over them")
        print(f"     is a better-posed problem, and llm.pt is a pretrained prior for that space.")
    elif b > a + 0.05:
        print(f"  -> OUR codes are more structured ({b:.1%} vs {a:.1%}). Swapping the token")
        print(f"     space would make the LM's job harder, not easier. Direction closes.")
    else:
        print(f"  -> COMPARABLE ({a:.1%} vs {b:.1%}). The token space is not the lever; any gain")
        print(f"     would have to come from llm.pt's pretrained weights, not the representation.")

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
