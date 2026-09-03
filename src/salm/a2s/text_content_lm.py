"""Autoregressive transcript-to-content-code bridge for the proven A2S renderer.

This is deliberately an *oracle text* component, not an ambient-audio model.  It answers a
single architectural question: if an upstream scene planner supplies an utterance as text,
can a small categorical model turn that text into the 256 on-manifold CTC-code frames that
``content_concat_v4`` already knows how to render?  Keeping ambient audio out of this gate
separates the lexical planner from the speech renderer instead of confounding both again.

The text encoder is frozen and evaluated outside this module.  The bridge consumes its
token-level hidden states, cross-attends to them while autoregressively predicting K=512
codes, and therefore keeps both kinds of sequence structure: the supplied words in memory
and the already-generated speech-code prefix in the decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TextContentLM(nn.Module):
    """Frozen text-encoder states -> autoregressive fixed-length content-code sequence."""

    def __init__(
        self,
        num_codes: int,
        text_dim: int = 1024,
        content_frames: int = 256,
        max_text_tokens: int = 64,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_codes = int(num_codes)
        self.content_frames = int(content_frames)
        self.max_text_tokens = int(max_text_tokens)
        self.text_dim = int(text_dim)
        self.d_model = int(d_model)
        self.bos_id = self.num_codes

        self.text_proj = nn.Linear(text_dim, d_model)
        self.text_pos = nn.Parameter(torch.zeros(1, max_text_tokens, d_model))
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

        nn.init.normal_(self.text_pos, std=0.02)
        nn.init.normal_(self.code_pos, std=0.02)

    def encode_text(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project frozen text states ``(B,S,text_dim)`` into decoder memory."""
        if hidden.shape[-1] != self.text_dim:
            raise RuntimeError(
                f"text width {hidden.shape[-1]} != configured text_dim {self.text_dim}"
            )
        if hidden.shape[1] > self.max_text_tokens:
            raise RuntimeError(
                f"text length {hidden.shape[1]} exceeds max_text_tokens={self.max_text_tokens}"
            )
        return self.text_proj(hidden) + self.text_pos[:, : hidden.shape[1]]

    def forward(
        self,
        text_hidden: torch.Tensor,
        text_mask: torch.Tensor,
        codes_in: torch.Tensor,
        memory: torch.Tensor | None = None,
        memory_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Teacher-forced logits; position ``t`` predicts code ``c_t``.

        ``text_mask`` is True for real text tokens.  ``memory`` and ``memory_pad_mask`` may
        be supplied together for a paired-vs-shuffled intervention that changes only which
        transcript memory accompanies a fixed target sequence.
        """
        if text_mask.dtype is not torch.bool:
            text_mask = text_mask.bool()
        if memory is None:
            memory = self.encode_text(text_hidden)
        if memory_pad_mask is None:
            memory_pad_mask = ~text_mask

        t = codes_in.shape[1]
        if t > self.content_frames:
            raise RuntimeError(
                f"code length {t} exceeds content_frames={self.content_frames}"
            )
        x = self.code_embed(codes_in) + self.code_pos[:, :t]
        causal = torch.triu(
            torch.ones(t, t, device=codes_in.device, dtype=torch.bool), diagonal=1
        )
        h = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=causal,
            memory_key_padding_mask=memory_pad_mask,
        )
        return self.head(self.norm(h))

    @torch.no_grad()
    def sample(
        self,
        text_hidden: torch.Tensor,
        text_mask: torch.Tensor,
        temperature: float = 0.8,
        top_k: int = 32,
        top_p: float = 0.0,
        greedy: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Generate ``(B, content_frames)`` codes from text, left to right."""
        b = text_hidden.shape[0]
        device = text_hidden.device
        memory = self.encode_text(text_hidden)
        pad = ~text_mask.bool()
        codes = torch.full((b, 1), self.bos_id, dtype=torch.long, device=device)

        for _ in range(self.content_frames):
            logits = self.forward(
                text_hidden,
                text_mask,
                codes,
                memory=memory,
                memory_pad_mask=pad,
            )[:, -1]
            if greedy:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / max(float(temperature), 1e-5)
                if top_k > 0:
                    k = min(int(top_k), logits.shape[-1])
                    kth = logits.topk(k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                if top_p > 0.0:
                    order = logits.argsort(dim=-1, descending=True)
                    sorted_logits = logits.gather(1, order)
                    sorted_probs = sorted_logits.softmax(dim=-1)
                    drop = sorted_probs.cumsum(dim=-1) - sorted_probs > float(top_p)
                    sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
                    logits = torch.full_like(logits, float("-inf")).scatter(
                        1, order, sorted_logits
                    )
                nxt = torch.multinomial(
                    logits.softmax(dim=-1), num_samples=1, generator=generator
                )
            codes = torch.cat([codes, nxt], dim=1)
        return codes[:, 1:]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def roll_text_pair(
    memory: torch.Tensor, pad_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pair each target with another row's real transcript memory and matching mask."""
    if memory.shape[0] < 2:
        raise RuntimeError("text binding ablation needs batch >= 2")
    return torch.roll(memory, 1, 0), torch.roll(pad_mask, 1, 0)
