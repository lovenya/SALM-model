"""
ContentLMDedup -- autoregressive (code, duration) pairs over run-length-encoded content.

WHY THIS EXISTS. The frame-level ContentLM works: it generates per-scene, syntactically real
speech from ambient audio alone, and cross-checkpoint consistency proves the scene genuinely
determines the content (jobs 436344/438059). What it cannot do is hold a plan: short
utterances come out clean, long ones fragment after ~3-5 words, and the user's listening
across the full epoch ladder found every checkpoint alike -- "all the same".

Both alternative explanations are eliminated by measurement, not argument:
  - more training      -> job 438059, flat epoch ladder (ep99 no better than ep39)
  - regularization     -> job 438062, scene_gap at the loss minimum invariant at +0.021
                          across seven configs (3 codebook sizes, 3x dropout, 10x weight
                          decay, two depths, label smoothing)

What remains is that the model plans over 256 tokens at 25 fps -- one 40 ms acoustic state per
step -- and holds a plan for roughly 30 of them. This model plans over run-length tokens
instead, ~128-151 of them, and emits a duration alongside each code so the 256-frame grid the
concat channel requires can be reconstructed exactly.

HONEST EXPECTATION. Measured compression is 1.7-2.0x, NOT the 3-5x first assumed: 45% of
adjacent frames are already identical and the rest sit at cosine ~0.79, so they are genuinely
distinct centroids. Halving the horizon is a real improvement and is not expected to be
sufficient alone. If the 3-4 word islands become 6-8 word phrases, that is the win; if they do
not move, that is the evidence for a two-level semantic/acoustic hierarchy rather than a
surprise.

TWO HEADS, ONE MODEL. Duration is predicted by a second head off the same hidden state rather
than by a separate model. A standalone duration predictor would have to be conditioned on the
sampled code sequence anyway, so it would duplicate this model's context for no gain, and two
models means two things to keep in sync at generation time.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ContentLMDedup(nn.Module):
    """
    (B, A, ambient_dim) ambient CLAP frames -> autoregressive (code, duration) pairs.

    Vocabulary layout, chosen so no index is ever ambiguous:
      codes      0 .. K-1   real codebook entries
      BOS        K          input only, never a prediction target
      EOS        K+1        prediction target only, never an input
      durations  1 .. max_run   real; 0 is the "no duration" slot used for BOS

    The code head therefore has K+1 outputs (K real codes plus EOS) and the duration head
    max_run outputs (duration d maps to class d-1).
    """

    def __init__(
        self,
        num_codes: int,
        max_run: int = 32,
        ambient_dim: int = 768,
        max_tokens: int = 256,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_codes = int(num_codes)
        self.max_run = int(max_run)
        self.max_tokens = int(max_tokens)
        self.d_model = int(d_model)
        self.bos_id = int(num_codes)
        self.eos_id = int(num_codes) + 1

        self.ambient_proj = nn.Linear(ambient_dim, d_model)
        self.ambient_pos = nn.Parameter(torch.zeros(1, 512, d_model))

        # Code and duration are embedded separately and SUMMED: the model must know how long
        # the previous token lasted, or it cannot track how much of the 256-frame canvas it
        # has already spent, and duration prediction would be blind to the budget.
        self.code_embed = nn.Embedding(num_codes + 2, d_model)
        self.dur_embed = nn.Embedding(max_run + 1, d_model)
        self.token_pos = nn.Parameter(torch.zeros(1, max_tokens + 1, d_model))

        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * ff_mult,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.code_head = nn.Linear(d_model, num_codes + 1)   # +1 for EOS
        self.dur_head = nn.Linear(d_model, max_run)

        nn.init.normal_(self.ambient_pos, std=0.02)
        nn.init.normal_(self.token_pos, std=0.02)

    def encode_ambient(self, ambient: torch.Tensor) -> torch.Tensor:
        x = self.ambient_proj(ambient)
        return x + self.ambient_pos[:, : x.shape[1]]

    def forward(
        self,
        ambient: torch.Tensor,
        codes_in: torch.Tensor,
        durs_in: torch.Tensor,
        memory: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Teacher-forced. `codes_in`/`durs_in` are (B, S) with BOS prepended; returns
        (code_logits (B, S, K+1), dur_logits (B, S, max_run)) where position t predicts the
        t-th target.

        `memory` may be supplied pre-computed so the scene ablation can substitute a different
        batch's memory without re-encoding -- guaranteeing the two passes differ in exactly
        one tensor. `pad_mask` is True at padded positions.
        """
        if memory is None:
            memory = self.encode_ambient(ambient)
        s = codes_in.shape[1]
        if s > self.token_pos.shape[1]:
            # Otherwise this surfaces as an opaque broadcast error deep in the embedding sum.
            # The width needed is (longest padded token count + 1) for the prepended BOS, so
            # max_tokens must be >= the dedup cache's max_tokens.
            raise RuntimeError(
                f"sequence width {s} exceeds max_tokens+1={self.token_pos.shape[1]}; "
                f"build the model with max_tokens >= {s - 1}"
            )
        x = self.code_embed(codes_in) + self.dur_embed(durs_in) + self.token_pos[:, :s]
        causal = torch.triu(
            torch.ones(s, s, device=codes_in.device, dtype=torch.bool), diagonal=1
        )
        h = self.decoder(
            tgt=x, memory=memory, tgt_mask=causal, tgt_key_padding_mask=pad_mask
        )
        h = self.norm(h)
        return self.code_head(h), self.dur_head(h)

    @torch.no_grad()
    def sample(
        self,
        ambient: torch.Tensor,
        total_frames: int = 256,
        temperature: float = 1.0,
        dur_temperature: float | None = None,
        top_k: int = 0,
        top_p: float = 0.0,
        cfg_scale: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> tuple[list[list[int]], list[list[int]]]:
        """
        Autoregressively sample (codes, durations) per row until EOS, the frame budget is
        filled, or max_tokens is reached. Returns ragged python lists -- rows genuinely differ
        in length, and padding them here would only force the caller to undo it.

        Sampling, never argmax: argmax walks straight back into the conditional mean that the
        discrete formulation exists to escape. `dur_temperature` defaults to `temperature`;
        it is separable because duration is a far lower-entropy variable than code identity
        and may want sharper sampling.

        CLASSIFIER-FREE GUIDANCE (`cfg_scale` > 1). Measured on job 442814, cross-seed
        agreement on a fixed ambient clip collapses after roughly the fourth word: 0.20, 0.20,
        0.17, 0.12, then 0.00. Positional inter-ASR agreement over the same audio stays near
        the real-speech curve throughout, so the later words are clearly articulated -- they
        are simply no longer pinned by the scene. The model stops using the ambient long
        before it stops speaking.

        Guidance attacks exactly that. At every step the step logits are extrapolated away
        from what the model would say under a DIFFERENT scene:

            log p  =  log p(mismatched)  +  w * [ log p(true) - log p(mismatched) ]

        so whatever the true ambient contributes is amplified at every position, not only
        where the model was already inclined to use it. w = 1 is exactly the unguided model.

        The unconditional branch is a rolled batch, not a zeroed memory. That is the same
        choice `shuffled_memory` documents for the scene ablation: a zeroed memory is out of
        distribution, so extrapolating away from it would push the sequence off-manifold for
        reasons unrelated to the scene, whereas a rolled memory is a real ambient encoding
        that simply belongs to another row. It therefore needs a batch of at least 2.

        Only the CODE head is guided. Duration is low-entropy and controls the frame budget;
        extrapolating it risks run lengths that no longer sum to a valid 256-frame canvas,
        which would corrupt the render for reasons that have nothing to do with content.
        """
        b = ambient.shape[0]
        device = ambient.device
        dt = temperature if dur_temperature is None else dur_temperature
        memory = self.encode_ambient(ambient)
        mem_uncond = None
        if cfg_scale != 1.0:
            if b < 2:
                raise ValueError(
                    f"cfg_scale={cfg_scale} needs a batch of >= 2 so the unconditional branch "
                    f"can be a DIFFERENT scene's memory; got batch {b}. Roll on a batch of 1 "
                    f"returns the same memory and guidance would silently do nothing."
                )
            mem_uncond = torch.roll(memory, shifts=1, dims=0)

        codes_in = torch.full((b, 1), self.bos_id, dtype=torch.long, device=device)
        durs_in = torch.zeros((b, 1), dtype=torch.long, device=device)

        out_codes: list[list[int]] = [[] for _ in range(b)]
        out_durs: list[list[int]] = [[] for _ in range(b)]
        spent = torch.zeros(b, dtype=torch.long, device=device)
        done = torch.zeros(b, dtype=torch.bool, device=device)

        def _pick(logits: torch.Tensor, temp: float) -> torch.Tensor:
            logits = logits / max(temp, 1e-5)
            if top_k > 0:
                k = min(top_k, logits.shape[-1])
                kth = logits.topk(k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if 0.0 < top_p < 1.0:
                # Nucleus: keep the smallest set of codes whose probability mass reaches
                # top_p, drop the rest. The middle ground greedy and temperature do not
                # cover -- greedy always takes the single most likely continuation, which is
                # why 39 of 128 clips opened "hey can you hear" (job 443475), while lowering
                # temperature rescales the WHOLE distribution and leaves the long tail
                # reachable. top_p removes the tail without flattening the head.
                srt, idx = logits.sort(dim=-1, descending=True)
                cum = srt.softmax(dim=-1).cumsum(dim=-1)
                # Shift so the token that CROSSES the threshold is itself kept; otherwise a
                # distribution whose top code already exceeds top_p would have everything
                # masked and multinomial would see all -inf.
                drop = cum - srt.softmax(dim=-1) >= top_p
                drop[:, 0] = False
                # Map the sorted-order mask back to vocabulary order. Explicit zeros_like
                # base: scattering into `drop` itself happens to work only because idx is a
                # permutation, which is not obvious to a reader.
                mask = torch.zeros_like(drop).scatter(1, idx, drop)
                logits = logits.masked_fill(mask, float("-inf"))
            return torch.multinomial(logits.softmax(dim=-1), 1, generator=generator)

        for _ in range(self.max_tokens):
            cl, dl = self.forward(ambient, codes_in, durs_in, memory=memory)
            cl, dl = cl[:, -1], dl[:, -1]
            if mem_uncond is not None:
                cu, _ = self.forward(ambient, codes_in, durs_in, memory=mem_uncond)
                # Combine in LOG-PROBABILITY space, not raw logits: the two branches have no
                # reason to share a normalisation constant, so extrapolating unnormalised
                # logits would mix in an arbitrary per-branch offset.
                lp_c = torch.log_softmax(cl, dim=-1)
                lp_u = torch.log_softmax(cu[:, -1], dim=-1)
                cl = lp_u + cfg_scale * (lp_c - lp_u)

            nxt_code = _pick(cl, temperature).squeeze(1)              # (B,) in 0..K (K=EOS)
            nxt_dur = _pick(dl, dt).squeeze(1) + 1                    # (B,) in 1..max_run

            hit_eos = nxt_code == self.num_codes
            # Do not let a duration overshoot the remaining budget: expand() would truncate
            # it anyway, and feeding the model a duration it did not emit would desynchronise
            # its own view of how much canvas is left.
            remaining = (total_frames - spent).clamp(min=1)
            nxt_dur = torch.minimum(nxt_dur, remaining)

            for i in range(b):
                if done[i] or hit_eos[i]:
                    continue
                out_codes[i].append(int(nxt_code[i]))
                out_durs[i].append(int(nxt_dur[i]))
            spent = torch.where(done | hit_eos, spent, spent + nxt_dur)
            done = done | hit_eos | (spent >= total_frames)
            if bool(done.all()):
                break

            # A finished row is fed EOS-as-BOS padding; its outputs are ignored from here.
            step_code = torch.where(done, torch.full_like(nxt_code, self.bos_id), nxt_code)
            step_dur = torch.where(done, torch.zeros_like(nxt_dur), nxt_dur)
            codes_in = torch.cat([codes_in, step_code.unsqueeze(1)], dim=1)
            durs_in = torch.cat([durs_in, step_dur.unsqueeze(1)], dim=1)

        # A row that emitted EOS immediately would expand to nothing; give it one silent-ish
        # token so downstream expansion has something real to work with rather than raising.
        for i in range(b):
            if not out_codes[i]:
                out_codes[i] = [0]
                out_durs[i] = [total_frames]
        return out_codes, out_durs

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_teacher_forcing(
    tokens: torch.Tensor,
    durations: torch.Tensor,
    lengths: torch.Tensor,
    bos_id: int,
    eos_id_in_head: int,
) -> dict:
    """
    Padded (B, L) tokens/durations + (B,) lengths -> the tensors a training step needs.

    Returns inputs of width L+1 (BOS prepended) and targets of the same width, where target
    position `lengths[i]` is EOS and everything past it is masked out. Built here rather than
    inline in the training loop because every off-by-one in this construction is silent: the
    loss still decreases, the model just learns to predict the wrong thing.
    """
    b, l = tokens.shape
    device = tokens.device
    width = l + 1

    # ---- inputs: BOS, then each token/duration shifted right by one
    codes_in = torch.full((b, width), bos_id, dtype=torch.long, device=device)
    durs_in = torch.zeros((b, width), dtype=torch.long, device=device)
    codes_in[:, 1:] = tokens.clamp(min=0)      # padded slots arbitrary; masked out below
    durs_in[:, 1:] = durations.clamp(min=0)

    # ---- targets: position t predicts token t, then one EOS, then ignore (-100)
    pad_col = torch.zeros((b, 1), dtype=torch.long, device=device)
    tok_wide = torch.cat([tokens.clamp(min=0), pad_col], dim=1)              # (b, width)
    dur_wide = torch.cat([durations.clamp(min=1) - 1, pad_col], dim=1)       # 1..R -> 0..R-1

    ar = torch.arange(width, device=device).unsqueeze(0)
    lens = lengths.unsqueeze(1)
    valid = ar < lens                                                        # real tokens
    ign = torch.full_like(tok_wide, -100)
    code_tgt = torch.where(valid, tok_wide, ign)
    dur_tgt = torch.where(valid, dur_wide, ign)
    # exactly one EOS per row, immediately after its last real token. Duration stays -100
    # there: there is no duration to predict for an end-of-sequence marker.
    code_tgt.scatter_(1, lens, eos_id_in_head)

    # ---- decoder padding mask: input 0 is BOS and input t+1 holds token t, so a row of
    # length L has real inputs 0..L and padding strictly beyond. BOS is always real, so no
    # row can be fully masked (which would produce NaNs in attention).
    pad_mask = ar > lens
    return {
        "codes_in": codes_in, "durs_in": durs_in,
        "code_tgt": code_tgt, "dur_tgt": dur_tgt,
        "pad_mask": pad_mask,
    }
