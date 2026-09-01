"""
Learned ambient<->content retrieval, replacing raw-CLAP-cosine nearest-neighbour with a
TRAINED matching function -- while keeping retrieval's one genuinely important property: it
always hands back a real, complete, un-averaged training example's content, never a blurry
regression average.

WHY THIS EXISTS. AmbientContentPredictor (2026-08-31) failed decisively: predicting a
continuous 512-d content vector via MSE+cosine regression converges toward a blurry average
that scores reasonable cosine similarity but carries no committed phonetic identity --
generated intelligibility landed at 0.94-0.96 (noise-band), far worse than plain nearest-
neighbour retrieval's 0.50. Two independent capacity/regularization settings (v1, v2) hit the
SAME ~0.70 val_cos ceiling, ruling out "needs more/less capacity" as the explanation -- the
failure is the OUTPUT TYPE, not the architecture size.

Retrieval never has this failure mode: whatever it returns is always SOME real training
example's actual, fully-formed content trajectory. The only thing worth improving is WHICH
example gets picked -- currently decided by raw CLAP cosine similarity, a feature space never
trained for this task. This module replaces that with two small projection heads trained
contrastively (CLIP-style): push the correct (ambient, content) pairing's similarity above
every other pairing's in the same batch. At inference, use the learned embedding to find the
best match, then reuse THAT row's real content sequence -- identical downstream usage to
plain retrieval, just with a trained metric instead of an off-the-shelf one.

Both towers pool their input to ONE vector before projecting -- the retrieval DECISION only
needs "which training example is the best match", a per-clip judgement, not a sequence.
Whatever gets retrieved still supplies its full, real 256-frame content sequence downstream.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentRetriever(nn.Module):
    def __init__(
        self,
        ambient_dim: int = 768,
        content_dim: int = 512,
        embed_dim: int = 256,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.ambient_proj = nn.Sequential(
            nn.Linear(ambient_dim, embed_dim * 2), nn.GELU(), nn.Linear(embed_dim * 2, embed_dim),
        )
        self.content_proj = nn.Sequential(
            nn.Linear(content_dim, embed_dim * 2), nn.GELU(), nn.Linear(embed_dim * 2, embed_dim),
        )
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))

    def embed_ambient(self, ambient_seq: torch.Tensor, ambient_mask: torch.Tensor) -> torch.Tensor:
        """ambient_seq: (B, 32, ambient_dim), ambient_mask: (B, 32) bool. Returns (B, embed_dim), L2-normalized."""
        m = ambient_mask.unsqueeze(-1).to(ambient_seq.dtype)
        pooled = (ambient_seq * m).sum(1) / m.sum(1).clamp(min=1.0)
        return F.normalize(self.ambient_proj(pooled), dim=-1)

    def embed_content(self, content_seq: torch.Tensor) -> torch.Tensor:
        """content_seq: (B, 256, content_dim), mean-pooled -- see module docstring for why a
        pooled comparison is enough for the retrieval DECISION. Returns (B, embed_dim), L2-normalized."""
        pooled = content_seq.mean(dim=1)
        return F.normalize(self.content_proj(pooled), dim=-1)

    def forward(self, ambient_seq: torch.Tensor, ambient_mask: torch.Tensor, content_seq: torch.Tensor):
        return self.embed_ambient(ambient_seq, ambient_mask), self.embed_content(content_seq)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def retrieval_contrastive_loss(ambient_emb: torch.Tensor, content_emb: torch.Tensor, log_temperature: torch.Tensor) -> torch.Tensor:
    """
    Symmetric InfoNCE, same construction as train_ldm.py's infonce_loss for O2 -- negatives
    are the other B-1 rows of THIS batch only. Both embeddings are already L2-normalized.
    """
    temperature = log_temperature.exp().clamp(min=1e-3)
    logits = ambient_emb @ content_emb.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_a2c = F.cross_entropy(logits, labels)
    loss_c2a = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_a2c + loss_c2a)
