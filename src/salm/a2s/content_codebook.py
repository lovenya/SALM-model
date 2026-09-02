"""
ContentCodebook -- a discrete vocabulary over real CTC-probe content frames.

WHY THIS EXISTS. Four attempts to produce the content channel from ambient audio alone have
failed, and the two honest ones (regression, 2026-08-31; end-to-end diffusion loss, jobs
434308 + 434417) failed the SAME way for a reason that is mathematical, not architectural:

    argmin_f E[(f(x) - y)^2]  =  f(x) = E[y | x]

A deterministic function trained against specific targets converges on the conditional MEAN.
The averaging does not come from duplicate targets -- the dataset is a strict bijection, one
ambience to one utterance. It comes from NEIGHBOURS: two kitchen recordings sit close in CLAP
space with unrelated transcripts, and a smooth function cannot emit wildly different content
for near-identical inputs, so the two pulls balance at the mean. And at validation the input
is unseen, where the only thing a deterministic map can do is interpolate between known
neighbours -- averaging by definition, which is why extra capacity never helped.

In content space that mean is OFF-MANIFOLD: the average of the encodings of "dinner's ready"
and "where are my keys" is not a blurry utterance, it is not an utterance at all. Feeding it
to a renderer trained exclusively on real encodings plausibly explains why job 434417's
frozen renderer returned almost no gradient -- its input was outside everything it had seen.

WHAT CHANGES WITH A CODEBOOK. Quantize content to K real frames learned from actual speech.
Content becomes 256 discrete indices instead of 256 continuous vectors, and:

  1. Every possible output is a real content frame that some real utterance produced, so the
     renderer never sees off-manifold input. On-manifold BY CONSTRUCTION.
  2. The predictor's objective becomes cross-entropy over categories, whose optimum is the
     true conditional DISTRIBUTION rather than its average -- and the mean of a distribution
     over categories is not a category, so there is no way to accidentally emit the average.
  3. At inference you SAMPLE. One plausible utterance, committed to, instead of the blur.

Direct precedent for this exact failure and this exact fix: lip-to-speech went from 65-78%
WER with mel regression to 50.2/44.8% with discrete units (Choi et al., arXiv 2305.19603).

THE GATE. All of the above is worthless if quantization itself destroys intelligibility.
Before any predictor is trained, `--content-source oracle_quantized` feeds quantized ORACLE
content to the already-proven renderer (content_concat_v4). If that is not about as
intelligible as unquantized oracle (0.389 against a 0.372 VAE ceiling), K is too small or
this representation is too lossy, and no amount of predictor work can recover it.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _nearest_codes(
    flat: torch.Tensor, codes: torch.Tensor, chunk: int = 32768
) -> torch.Tensor:
    """
    Nearest code index for each row of `flat` (N, D) against `codes` (K, D), by squared
    Euclidean distance.

    Chunked because the full (N, K) distance matrix is the memory blow-up here: 4.5M frames
    against 2048 codes is 9.2e9 floats. ||x||^2 is dropped since it is constant per row and
    cannot change the argmin.
    """
    code_sq = (codes * codes).sum(dim=1)                       # (K,)
    out = torch.empty(flat.shape[0], dtype=torch.long, device=flat.device)
    for start in range(0, flat.shape[0], chunk):
        block = flat[start : start + chunk]                    # (n, D)
        # -2 x.c + ||c||^2, argmin over K
        d = code_sq.unsqueeze(0) - 2.0 * (block @ codes.t())   # (n, K)
        out[start : start + chunk] = d.argmin(dim=1)
    return out


class ContentCodebook(nn.Module):
    """
    K real content frames, learned from the training split's own CTC-probe encodings.

    Not a VQ-VAE: nothing is trained through this and there is no commitment loss. The CTC
    probe is frozen and its output space is fixed, so the codebook is a fitted constant --
    k-means centroids over frames that actually occurred. That keeps it inspectable (every
    code is a real frame's neighbourhood) and makes it reusable across predictor variants.
    """

    def __init__(self, num_codes: int, content_dim: int) -> None:
        super().__init__()
        self.num_codes = int(num_codes)
        self.content_dim = int(content_dim)
        # A buffer, not a Parameter: it is fitted once and then frozen, and it must ride
        # along in state_dict so a predictor checkpoint is self-contained.
        self.register_buffer("codes", torch.zeros(num_codes, content_dim))

    @torch.no_grad()
    def indices(self, content_seq: torch.Tensor) -> torch.Tensor:
        """(B, T, D) continuous content -> (B, T) long code indices."""
        b, t, d = content_seq.shape
        if d != self.content_dim:
            raise RuntimeError(
                f"content width {d} does not match codebook width {self.content_dim}"
            )
        idx = _nearest_codes(content_seq.reshape(-1, d).float(), self.codes)
        return idx.view(b, t)

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """(B, T) long code indices -> (B, T, D) content, every frame a real code."""
        return self.codes[indices]

    @torch.no_grad()
    def quantize(self, content_seq: torch.Tensor) -> torch.Tensor:
        """Round-trip: (B, T, D) -> nearest codes -> (B, T, D). The gate's operation."""
        return self.dequantize(self.indices(content_seq))

    @torch.no_grad()
    def reconstruction_stats(self, content_seq: torch.Tensor) -> dict:
        """
        How much does quantization actually cost? Reported by the build script per K so the
        choice of K is made on measurement rather than a round number.
        """
        q = self.quantize(content_seq)
        err = (q - content_seq).flatten(0, 1).norm(dim=-1)
        ref = content_seq.flatten(0, 1).norm(dim=-1).clamp(min=1e-8)
        cos = torch.nn.functional.cosine_similarity(
            q.flatten(0, 1), content_seq.flatten(0, 1), dim=-1
        )
        return {
            "rel_l2": float((err / ref).mean()),
            "cosine": float(cos.mean()),
        }


@torch.no_grad()
def _kmeanspp_init(
    frames: torch.Tensor, num_codes: int, g: torch.Generator, log=print
) -> torch.Tensor:
    """
    k-means++ seeding (Arthur & Vassilvitskii 2007): first centre uniform, each subsequent
    centre drawn with probability proportional to its squared distance to the nearest centre
    chosen so far.

    Not a refinement -- a correctness fix. Plain random-sample init was measurably getting
    stuck: on 12 well-separated synthetic clusters it recovered only ~9.4 effective codes,
    merging some clusters while splitting others. On real content frames that shows up as
    wasted codebook capacity, which is exactly the thing this representation cannot afford.

    Cost is one pass per centre, but each pass only measures distance to the ONE new centre
    (a running minimum is kept), so it is K matrix-vector products, not K full k-means steps.
    """
    n, d = frames.shape
    codes = torch.empty(num_codes, d, device=frames.device, dtype=frames.dtype)
    first = int(torch.randint(0, n, (1,), generator=g).item())
    codes[0] = frames[first]
    # Squared distance from every frame to the nearest chosen centre.
    closest = ((frames - codes[0]) ** 2).sum(dim=1)
    for i in range(1, num_codes):
        total = float(closest.sum())
        if total <= 0.0:
            # Every remaining frame coincides with a chosen centre: fall back to uniform.
            rest = torch.randint(0, n, (num_codes - i,), generator=g).to(frames.device)
            codes[i:] = frames[rest]
            break
        probs = (closest / closest.sum()).cpu()
        pick = int(torch.multinomial(probs, 1, generator=g).item())
        codes[i] = frames[pick]
        closest = torch.minimum(closest, ((frames - codes[i]) ** 2).sum(dim=1))
        if num_codes >= 512 and i % 512 == 0:
            log(f"      k-means++ seeded {i}/{num_codes}")
    return codes


@torch.no_grad()
def fit_codebook(
    frames: torch.Tensor,
    num_codes: int,
    iters: int = 60,
    seed: int = 0,
    tol: float = 1e-5,
    log=print,
) -> tuple[ContentCodebook, dict]:
    """
    Lloyd's algorithm (full-batch k-means) over `frames` (N, D), on whatever device it is on.

    Full-batch rather than minibatch k-means: with the frame count subsampled to fit in GPU
    memory this converges deterministically in well under a minute, and determinism matters
    more here than the last few percent of speed -- the codebook is a fixed artifact that
    every downstream experiment is compared against, so it should be exactly reproducible
    from (frames, num_codes, seed).

    Dead codes (zero members) are re-seeded to random frames, which is standard and prevents
    a slow leak of usable capacity across iterations.
    """
    n, d = frames.shape
    if n < num_codes:
        raise ValueError(f"need at least {num_codes} frames to fit {num_codes} codes, got {n}")

    g = torch.Generator(device="cpu").manual_seed(seed)
    codes = _kmeanspp_init(frames, num_codes, g, log=log)

    prev = None
    history = []
    for it in range(iters):
        assign = _nearest_codes(frames, codes)
        # Sum frames per code, then divide by count -- index_add_ rather than a loop over K.
        sums = torch.zeros_like(codes)
        sums.index_add_(0, assign, frames)
        counts = torch.bincount(assign, minlength=num_codes).to(frames.dtype)
        dead = counts == 0
        codes = sums / counts.clamp(min=1.0).unsqueeze(1)
        if bool(dead.any()):
            n_dead = int(dead.sum())
            reseed = torch.randint(0, n, (n_dead,), generator=g).to(frames.device)
            codes[dead] = frames[reseed]

        # Mean squared distance to the assigned code -- the objective being minimized.
        inertia = float(((frames - codes[assign]) ** 2).sum(dim=1).mean())
        history.append(inertia)
        if it % 10 == 0 or it == iters - 1:
            log(f"      iter {it:3d}  inertia={inertia:.5f}  dead={int(dead.sum())}")
        if prev is not None and abs(prev - inertia) < tol * max(prev, 1e-8):
            log(f"      converged at iter {it} (relative change < {tol})")
            break
        prev = inertia

    book = ContentCodebook(num_codes, d).to(frames.device)
    book.codes.copy_(codes)

    # Usage statistics. A codebook where 90% of codes are never used is really a much
    # smaller codebook, and its effective size is what matters for the predictor's job.
    assign = _nearest_codes(frames, codes)
    counts = torch.bincount(assign, minlength=num_codes).float()
    p = counts / counts.sum()
    nz = p[p > 0]
    stats = {
        "num_codes": num_codes,
        "inertia": history[-1],
        "dead_codes": int((counts == 0).sum()),
        # exp(entropy) -- the "effective number of codes actually in use"
        "usage_perplexity": float(torch.exp(-(nz * nz.log()).sum())),
        "top1_share": float(p.max()),
    }
    return book, stats
