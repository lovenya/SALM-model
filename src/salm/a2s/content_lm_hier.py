"""
ContentLMHier -- coarse-to-fine over run-length content tokens, in one autoregressive model.

THE ARITHMETIC THIS EXISTS FOR. Measured across jobs 442814 / 443475: cross-seed agreement on a
fixed ambient clip runs 0.20 0.20 0.17 0.12 and then 0.00, so the model holds roughly 20
run-length tokens of plan. One utterance needs about 56. A 2.8x shortfall.

    full sequence   151 tokens needed vs 20-token horizon  ->  7.6x short
    coarse (1/8)     19 tokens needed vs 20-token horizon  ->  FITS

THE 151 IS MEASURED AND THE FIRST VERSION OF THIS FILE HAD IT WRONG. An earlier draft claimed
56 tokens and a 2.8x shortfall, which came from the SPEECH REGION only (111 of 256 frames). The
model must emit the whole 256-frame sequence including the silent tail, and the k2048 dedup
cache says that is 151.4 tokens on average. At the stride 3 that error implied, the coarse pass
would be 50 tokens -- still 2.5x beyond the horizon -- so the design would have failed for
exactly the reason it exists to avoid, and the null would have been written up as
"coarse-to-fine does not help".

At stride 8 each coarse anchor covers ~13.5 frames = ~540 ms, which at the measured 2.85
words/s is roughly one anchor per 1.5 words. The planning grid is word-scale by accident of the
arithmetic rather than by design, which is at least a good sign.

Every previous attempt tried to make the model use those 20 tokens better and all of them were
null -- classifier-free guidance (442847) bought two words, the plan vector (443751) was null,
think tokens (443989) were null. None of them changed what 20 tokens SPAN. This does: at 1/3
resolution the whole utterance fits inside the horizon the model actually has, so it can commit
to a trajectory across the entire utterance before emitting a single fine token.

WHY THIS IS NOT THE PLAN VECTOR AGAIN. That predicted a bag of codes -- one summary, no time
structure -- and it beat a scene-blind marginal by 0.0040 nats, so the model was asked to
commit to something that carried almost nothing. A coarse SEQUENCE is different in three ways
that matter: it has time structure (it is a trajectory, not a summary), it spans the whole
utterance, and every one of its tokens is supervised by real data rather than by a target the
agent invented.

WHY NOT TWO MODELS. A separate planner would have to be conditioned on the ambient anyway,
duplicating this model's encoder, and would need its own sampling loop kept in sync at
generation. Here the coarse tokens are simply a PREFIX of the same sequence:

    inputs   [BOS] c0 c1 .. c_{m-1} [SEP] f0 f1 .. f_{n-1}
    targets   c0   c1 ..  c_{m-1} [SEP]  f0 f1 ..  f_{n-1} [EOS]

so the fine pass reads the skeleton through ordinary self-attention over its own context. One
model, one objective, and at generation it does coarse-then-fine for free.

DURATIONS BELONG ONLY TO THE FINE PASS. The coarse tokens are a plan, not a rendering: they
carry no duration, their duration inputs are the zero slot, and their duration targets are
ignored. Only the fine tokens reconstruct the 256-frame grid the concat channel requires.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def coarse_of(tokens: np.ndarray, length: int, stride: int) -> np.ndarray:
    """
    The low-resolution sketch: every `stride`-th real token.

    A strided subsequence of REAL codes, deliberately, rather than a pooled or re-clustered
    summary. Pooling would need a second codebook and a second thing to validate, and the whole
    claim here is about sequence LENGTH rather than about inventing a new representation.
    """
    n = int(length)
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    return tokens[:n][::stride].astype(np.int64)


def build_hier_batch(
    tokens: torch.Tensor,
    durations: torch.Tensor,
    lengths: torch.Tensor,
    stride: int,
    bos_id: int,
    sep_id: int,
    eos_id: int,
) -> dict:
    """
    Padded (B, L) run-length tokens -> the coarse+SEP+fine teacher-forcing tensors.

    Built here rather than inline in the trainer for the same reason `build_teacher_forcing`
    was: every off-by-one in this construction is SILENT. The loss still falls, the model just
    learns to predict the wrong position, and the experiment reports a plausible number for a
    model that was never trained on the task described.
    """
    b, _ = tokens.shape
    device = tokens.device
    tok_np = tokens.detach().cpu().numpy()
    dur_np = durations.detach().cpu().numpy()
    len_np = lengths.detach().cpu().numpy()

    rows_c, rows_d, widths, coarse_lens = [], [], [], []
    for i in range(b):
        n = int(len_np[i])
        c = coarse_of(tok_np[i], n, stride)
        f = tok_np[i][:n].astype(np.int64)
        fd = dur_np[i][:n].astype(np.int64)
        # code stream and duration stream, aligned position for position
        seq = np.concatenate([c, [sep_id], f])
        dseq = np.concatenate([np.zeros(len(c), dtype=np.int64), [0], fd])
        rows_c.append(seq)
        rows_d.append(dseq)
        widths.append(len(seq))
        coarse_lens.append(len(c))

    w = max(widths) if widths else 1
    codes_in = np.full((b, w + 1), bos_id, dtype=np.int64)
    durs_in = np.zeros((b, w + 1), dtype=np.int64)
    code_tgt = np.full((b, w + 1), -100, dtype=np.int64)
    dur_tgt = np.full((b, w + 1), -100, dtype=np.int64)
    pad_mask = np.ones((b, w + 1), dtype=bool)
    is_fine = np.zeros((b, w + 1), dtype=bool)

    for i, (seq, dseq) in enumerate(zip(rows_c, rows_d)):
        k = len(seq)
        codes_in[i, 1 : k + 1] = seq
        durs_in[i, 1 : k + 1] = dseq
        code_tgt[i, :k] = seq
        code_tgt[i, k] = eos_id                      # exactly one EOS, after the last token
        # durations are supervised ONLY on fine positions: the coarse prefix is a plan and has
        # no rendering, and SEP/EOS have nothing to render either.
        sep_at = int(np.where(seq == sep_id)[0][0])
        dur_tgt[i, sep_at + 1 : k] = dseq[sep_at + 1 : k] - 1     # 1..R -> 0..R-1
        is_fine[i, sep_at + 1 : k] = True
        pad_mask[i, : k + 1] = False                 # BOS plus k real inputs

    t = lambda a, dt: torch.from_numpy(a).to(device=device, dtype=dt)   # noqa: E731
    return {
        "codes_in": t(codes_in, torch.long), "durs_in": t(durs_in, torch.long),
        "code_tgt": t(code_tgt, torch.long), "dur_tgt": t(dur_tgt, torch.long),
        "pad_mask": t(pad_mask, torch.bool), "is_fine": t(is_fine, torch.bool),
        # the COARSE prefix length, not the width of the whole row -- the first
        # version measured the concatenated sequence and reported 81 where the
        # answer was 20, which would have made the horizon claim unfalsifiable.
        "mean_coarse": float(np.mean(coarse_lens)) if coarse_lens else 0.0,
    }


class ContentLMHier(nn.Module):
    """
    Vocabulary layout, chosen so no index is ever ambiguous:
        codes  0 .. K-1
        BOS    K        input only
        SEP    K+1      both an input and a target -- it is a real emitted symbol
        EOS    K+2      target only
    The code head has K+3 outputs so EOS=K+2 is in range; the BOS slot is unused.
    """

    def __init__(
        self,
        num_codes: int,
        max_run: int = 32,
        ambient_dim: int = 768,
        max_tokens: int = 384,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
        stride: int = 8,
    ) -> None:
        super().__init__()
        self.num_codes = int(num_codes)
        self.max_run = int(max_run)
        self.max_tokens = int(max_tokens)
        self.stride = int(stride)
        self.bos_id = int(num_codes)
        self.sep_id = int(num_codes) + 1
        self.eos_id = int(num_codes) + 2

        self.ambient_proj = nn.Linear(ambient_dim, d_model)
        self.ambient_pos = nn.Parameter(torch.zeros(1, 512, d_model))
        self.code_embed = nn.Embedding(num_codes + 3, d_model)
        self.dur_embed = nn.Embedding(max_run + 1, d_model)
        self.token_pos = nn.Parameter(torch.zeros(1, max_tokens + 1, d_model))

        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * ff_mult,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        # K+3, not K+2: the ids are codes 0..K-1, BOS=K, SEP=K+1, EOS=K+2, so EOS
        # indexes K+2 and the head must be that wide. The BOS slot is never a
        # target and is simply unused -- cheaper than remapping ids into a tighter
        # head, which is the kind of off-by-one that trains happily and wrong.
        self.code_head = nn.Linear(d_model, num_codes + 3)
        self.dur_head = nn.Linear(d_model, max_run)

        nn.init.normal_(self.ambient_pos, std=0.02)
        nn.init.normal_(self.token_pos, std=0.02)

    def encode_ambient(self, ambient: torch.Tensor) -> torch.Tensor:
        x = self.ambient_proj(ambient)
        return x + self.ambient_pos[:, : x.shape[1]]

    def forward(self, ambient, codes_in, durs_in, memory=None, pad_mask=None):
        if memory is None:
            memory = self.encode_ambient(ambient)
        s = codes_in.shape[1]
        if s > self.token_pos.shape[1]:
            raise RuntimeError(
                f"sequence width {s} exceeds max_tokens+1={self.token_pos.shape[1]}. The "
                f"hierarchical sequence is coarse+1+fine, so max_tokens must be at least "
                f"ceil(L/stride) + 1 + L for the longest L in the cache."
            )
        x = self.code_embed(codes_in) + self.dur_embed(durs_in) + self.token_pos[:, :s]
        causal = torch.triu(
            torch.ones(s, s, device=codes_in.device, dtype=torch.bool), diagonal=1
        )
        h = self.norm(self.decoder(tgt=x, memory=memory, tgt_mask=causal,
                                   tgt_key_padding_mask=pad_mask))
        return self.code_head(h), self.dur_head(h)

    @torch.no_grad()
    def sample(self, ambient, total_frames=256, temperature=1.0, top_k=0,
               generator=None, max_coarse_frac=0.75):
        """
        Two phases, one loop: emit coarse tokens until SEP, then fine tokens until the frame
        budget is full. Returns (fine_codes, fine_durations, n_coarse) per row -- the coarse
        prefix is a plan and is never rendered, but its LENGTH is returned because a run where
        the coarse pass collapsed to nothing is a failed run wearing a plausible output.
        """
        b, device = ambient.shape[0], ambient.device
        memory = self.encode_ambient(ambient)
        codes_in = torch.full((b, 1), self.bos_id, dtype=torch.long, device=device)
        durs_in = torch.zeros((b, 1), dtype=torch.long, device=device)

        out_c: list[list[int]] = [[] for _ in range(b)]
        out_d: list[list[int]] = [[] for _ in range(b)]
        n_coarse = [0] * b
        in_fine = torch.zeros(b, dtype=torch.bool, device=device)
        done = torch.zeros(b, dtype=torch.bool, device=device)
        spent = torch.zeros(b, dtype=torch.long, device=device)
        # A row that never emits SEP would spend the whole budget planning. Force the switch.
        coarse_cap = int(self.max_tokens * max_coarse_frac)

        def pick(logits, temp):
            logits = logits / max(temp, 1e-5)
            if top_k > 0:
                k = min(top_k, logits.shape[-1])
                kth = logits.topk(k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            return torch.multinomial(logits.softmax(dim=-1), 1, generator=generator).squeeze(1)

        for _ in range(self.max_tokens):
            cl, dl = self.forward(ambient, codes_in, durs_in, memory=memory)
            nxt = pick(cl[:, -1], temperature)
            nd = pick(dl[:, -1], temperature) + 1

            hit_sep = (nxt == self.num_codes + 1) & ~in_fine
            forced = torch.tensor([n_coarse[i] >= coarse_cap for i in range(b)], device=device)
            hit_sep = hit_sep | (forced & ~in_fine)
            hit_eos = (nxt == self.num_codes + 2) | (nxt == self.num_codes + 1) & in_fine

            remaining = (total_frames - spent).clamp(min=1)
            nd = torch.minimum(nd, remaining)
            for i in range(b):
                if done[i] or hit_sep[i] or hit_eos[i]:
                    continue
                if in_fine[i]:
                    out_c[i].append(int(nxt[i])); out_d[i].append(int(nd[i]))
                else:
                    n_coarse[i] += 1
            spent = torch.where(done | ~in_fine | hit_eos, spent, spent + nd)
            in_fine = in_fine | hit_sep
            done = done | (hit_eos & in_fine) | (in_fine & (spent >= total_frames))
            if bool(done.all()):
                break

            step_code = torch.where(hit_sep, torch.full_like(nxt, self.sep_id), nxt)
            step_code = torch.where(done, torch.full_like(nxt, self.bos_id), step_code)
            step_dur = torch.where(in_fine & ~hit_sep & ~done, nd, torch.zeros_like(nd))
            codes_in = torch.cat([codes_in, step_code.unsqueeze(1)], dim=1)
            durs_in = torch.cat([durs_in, step_dur.unsqueeze(1)], dim=1)

        for i in range(b):
            if not out_c[i]:
                out_c[i], out_d[i] = [0], [total_frames]
        return out_c, out_d, n_coarse

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
