"""
ContentLM -- autoregressive prediction of discrete content codes from an ambient scene.

THE ONE THING THIS IS FOR. Every previous content predictor was a deterministic map trained
against specific targets, and therefore converged on E[content | ambient] -- the conditional
mean, which in content space is not an utterance at all. See content_codebook.py for the full
argument. This model never emits a content vector directly. It emits a DISTRIBUTION over K
real codes per frame, trained by cross-entropy (whose optimum is the true conditional
distribution rather than its average), and at inference it SAMPLES. The mean of a
distribution over categories is not a category, so the collapse that killed regression and
both end-to-end runs is not merely discouraged here -- it is unrepresentable.

WHY AUTOREGRESSIVE AND NOT PER-FRAME INDEPENDENT. Sampling each of the 256 positions
independently from its own marginal produces phoneme soup: frame 5 commits to one utterance,
frame 6 to a different one, and the result is locally plausible and globally incoherent.
Conditioning each code on the codes already chosen -- P(c_t | c_<t, scene) -- is what makes a
sampled sequence a single utterance. Same reason language models are autoregressive.

WHAT WOULD MAKE THIS LOOK LIKE IT WORKS WHILE FAILING. Content frames are ~53% silence and
English phoneme statistics are highly predictable, so a model that completely ignores the
ambient scene can still reach a healthy-looking cross-entropy by learning the marginal
distribution of speech. That is the exact analogue of the failure `val_content_bind_gap` was
built to catch on the previous mechanism, and it is why `scene_gap` below is the metric this
model is actually judged on, not train loss and not raw perplexity.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ContentLM(nn.Module):
    """
    (B, A, ambient_dim) ambient CLAP frames -> autoregressive distribution over (B, T) codes.

    Cross-attention to the ambient sequence rather than prefixing it onto the code stream:
    the scene is only 32 frames against 256 code positions, and keeping them in separate
    streams means the ablation below (swap the memory, keep everything else) is a clean
    intervention on exactly one thing.
    """

    def __init__(
        self,
        num_codes: int,
        ambient_dim: int = 768,
        content_frames: int = 256,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_codes = int(num_codes)
        self.content_frames = int(content_frames)
        self.d_model = int(d_model)
        # Index `num_codes` is BOS -- it can never collide with a real code, which run
        # 0..num_codes-1, so no reserved-token bookkeeping is needed in the codebook itself.
        self.bos_id = int(num_codes)

        self.ambient_proj = nn.Linear(ambient_dim, d_model)
        self.ambient_pos = nn.Parameter(torch.zeros(1, 512, d_model))

        self.code_embed = nn.Embedding(num_codes + 1, d_model)
        self.code_pos = nn.Parameter(torch.zeros(1, content_frames, d_model))

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_codes)

        nn.init.normal_(self.ambient_pos, std=0.02)
        nn.init.normal_(self.code_pos, std=0.02)

    def encode_ambient(self, ambient: torch.Tensor) -> torch.Tensor:
        """(B, A, ambient_dim) -> (B, A, d_model) memory for cross-attention."""
        x = self.ambient_proj(ambient)
        return x + self.ambient_pos[:, : x.shape[1]]

    def forward(
        self,
        ambient: torch.Tensor,
        codes_in: torch.Tensor,
        memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Teacher-forced logits. `codes_in` is [BOS, c_0 ... c_{T-2}] (B, T); returns (B, T, K)
        whose position t predicts c_t.

        `memory` may be passed pre-computed so the ablation can substitute a DIFFERENT
        batch's memory without re-encoding, guaranteeing the two passes differ in exactly
        one thing.
        """
        if memory is None:
            memory = self.encode_ambient(ambient)
        t = codes_in.shape[1]
        x = self.code_embed(codes_in) + self.code_pos[:, :t]
        causal = torch.triu(
            torch.ones(t, t, device=codes_in.device, dtype=torch.bool), diagonal=1
        )
        h = self.decoder(tgt=x, memory=memory, tgt_mask=causal)
        return self.head(self.norm(h))

    @torch.no_grad()
    def sample(
        self,
        ambient: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Autoregressively sample (B, T) codes. THE point of this whole model -- argmax here
        would walk straight back into the mean-collapse this design exists to avoid, so
        sampling is the default and there is deliberately no `greedy=True` convenience flag.

        No KV cache: 256 steps over a ~20M-parameter model for a handful of rows is seconds,
        and a hand-rolled cache is a silent-correctness risk for no meaningful gain here.
        """
        b = ambient.shape[0]
        device = ambient.device
        memory = self.encode_ambient(ambient)
        codes = torch.full((b, 1), self.bos_id, dtype=torch.long, device=device)

        for _ in range(self.content_frames):
            logits = self.forward(ambient, codes, memory=memory)[:, -1]      # (B, K)
            logits = logits / max(temperature, 1e-5)

            if top_k > 0:
                k = min(top_k, logits.shape[-1])
                kth = logits.topk(k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))

            if top_p > 0.0:
                order = logits.argsort(dim=-1, descending=True)
                sorted_logits = logits.gather(1, order)
                probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                # Keep the first token that crosses the threshold, drop everything after it.
                drop = probs - sorted_logits.softmax(dim=-1) > top_p
                sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(1, order, sorted_logits)

            probs = logits.softmax(dim=-1)
            nxt = torch.multinomial(probs, num_samples=1, generator=generator)
            codes = torch.cat([codes, nxt], dim=1)

        return codes[:, 1:]                                                   # drop BOS

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def shuffled_memory(memory: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """
    Roll the batch so every sequence is scored against a DIFFERENT scene's ambient memory.

    The scene ablation, and the single most important measurement this stage makes. Rolling
    rather than zeroing is deliberate: a zeroed memory is out of distribution, so a model
    could score worse on it for reasons that have nothing to do with using the scene, which
    would read as a false positive. A rolled memory is a real ambient encoding that simply
    belongs to the wrong row, so any loss difference is attributable to the PAIRING alone.

    scene_gap = loss(rolled) - loss(true). Near zero means the model is ignoring the scene
    and has merely learned generic speech statistics -- a total failure that looks perfectly
    healthy in the training curve.
    """
    if memory.shape[0] < 2:
        raise RuntimeError("scene ablation needs batch >= 2; a batch of 1 rolls onto itself")
    return torch.roll(memory, shifts=1, dims=0)
