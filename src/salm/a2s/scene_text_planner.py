"""Flan-T5 scene-to-utterance planner driven by cached ambient CLAP frames.

This module deliberately stops at text.  Speech rendering is already handled by the
content-concat AudioLDM stack; the missing variable is a scene-grounded linguistic plan.
The adapter maps the *sequence* of ambient CLAP frames into the hidden-state distribution
expected by a pretrained T5 decoder, preserving time and avoiding retrieval entirely.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn


TASK_NAMES = ("speech", "caption", "reasoning", "structured")
TASK_TO_ID = {name: i for i, name in enumerate(TASK_NAMES)}


@dataclass(frozen=True)
class PlannerRow:
    sample_id: str
    audio_cap: str
    reasoning: str
    speech_text: str


def normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


def format_target(row: PlannerRow, task: str) -> str:
    if task == "speech":
        return row.speech_text.strip()
    if task == "caption":
        return row.audio_cap.strip()
    if task == "reasoning":
        return row.reasoning.strip()
    if task == "structured":
        return (
            f"caption: {row.audio_cap.strip()}\n"
            f"reasoning: {row.reasoning.strip()}\n"
            f"speech: {row.speech_text.strip()}"
        )
    raise ValueError(f"unknown planner task: {task!r}")


def task_for_example(target_mode: str, epoch: int, sample_index: int) -> str:
    """Deterministic rotating supervision for reproducible multi-task training."""
    if target_mode != "multitask":
        if target_mode not in TASK_TO_ID:
            raise ValueError(f"unknown target mode: {target_mode!r}")
        return target_mode
    choices = ("speech", "caption", "reasoning")
    return choices[(int(epoch) + int(sample_index)) % len(choices)]


def build_targets(
    rows: Sequence[PlannerRow], target_mode: str, epoch: int, indices: Sequence[int]
) -> tuple[list[str], torch.LongTensor]:
    tasks = [task_for_example(target_mode, epoch, i) for i in indices]
    targets = [format_target(row, task) for row, task in zip(rows, tasks)]
    ids = torch.tensor([TASK_TO_ID[t] for t in tasks], dtype=torch.long)
    return targets, ids


class AudioMemoryAdapter(nn.Module):
    """Map (B, ambient_frames, ambient_dim) into T5 encoder-memory states.

    A learned task token is prepended so caption/reasoning/speech supervision can share the
    same scene memory without asking the decoder to infer which textual view is requested.
    """

    def __init__(
        self,
        ambient_dim: int,
        t5_dim: int,
        layers: int = 2,
        heads: int = 8,
        dropout: float = 0.1,
        num_tasks: int = len(TASK_NAMES),
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(ambient_dim)
        self.input_proj = nn.Linear(ambient_dim, t5_dim)
        block = nn.TransformerEncoderLayer(
            d_model=t5_dim,
            nhead=heads,
            dim_feedforward=4 * t5_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(block, num_layers=layers)
        self.task_embed = nn.Embedding(num_tasks, t5_dim)
        self.output_norm = nn.LayerNorm(t5_dim)

    def forward(self, ambient: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        if ambient.ndim != 3:
            raise ValueError(f"ambient must be (B,T,D), got {tuple(ambient.shape)}")
        if task_ids.shape != (ambient.shape[0],):
            raise ValueError(
                f"task_ids must be ({ambient.shape[0]},), got {tuple(task_ids.shape)}"
            )
        x = self.input_proj(self.input_norm(ambient))
        x = self.context(x)
        task = self.task_embed(task_ids).unsqueeze(1)
        return self.output_norm(torch.cat([task, x], dim=1))


class AudioConditionedT5(nn.Module):
    """Thin wrapper that supplies projected audio as T5 ``encoder_outputs``.

    The installed Transformers 5.14 T5 implementation explicitly accepts a
    ``BaseModelOutput(last_hidden_state=...)`` in both ``forward`` and ``generate``.  Keeping
    the wrapper here makes the intervention and the true-vs-shuffled diagnostic identical.
    """

    def __init__(
        self,
        t5: nn.Module,
        ambient_dim: int,
        adapter_layers: int = 2,
        adapter_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.t5 = t5
        self.audio_adapter = AudioMemoryAdapter(
            ambient_dim=ambient_dim,
            t5_dim=int(t5.config.d_model),
            layers=adapter_layers,
            heads=adapter_heads,
            dropout=dropout,
        )

    @staticmethod
    def _encoder_output(memory: torch.Tensor):
        from transformers.modeling_outputs import BaseModelOutput

        return BaseModelOutput(last_hidden_state=memory)

    def encode_audio(self, ambient: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        return self.audio_adapter(ambient, task_ids)

    def forward_audio(
        self,
        ambient: torch.Tensor,
        task_ids: torch.Tensor,
        labels: torch.Tensor,
        memory: torch.Tensor | None = None,
    ):
        if memory is None:
            memory = self.encode_audio(ambient, task_ids)
        mask = torch.ones(memory.shape[:2], dtype=torch.long, device=memory.device)
        return self.t5(
            encoder_outputs=self._encoder_output(memory),
            attention_mask=mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )

    @torch.no_grad()
    def generate_audio(
        self,
        ambient: torch.Tensor,
        task_ids: torch.Tensor,
        **generate_kwargs,
    ) -> torch.Tensor:
        memory = self.encode_audio(ambient, task_ids)
        mask = torch.ones(memory.shape[:2], dtype=torch.long, device=memory.device)
        return self.t5.generate(
            encoder_outputs=self._encoder_output(memory),
            attention_mask=mask,
            **generate_kwargs,
        )

    def set_train_scope(self, scope: str, source: str) -> None:
        """Choose an explicit fine-tuning surface; the audio adapter always trains."""
        if scope not in {"adapter", "cross_attention", "decoder", "full"}:
            raise ValueError(f"unknown train scope: {scope}")
        for p in self.t5.parameters():
            p.requires_grad_(False)
        for p in self.audio_adapter.parameters():
            p.requires_grad_(source == "audio")
        if scope == "cross_attention":
            # T5 decoder blocks are [self-attention, cross-attention, feed-forward].
            # Updating the complete cross-attention layer (including its layer norm)
            # lets the decoder learn to read audio memory without rewriting its
            # pretrained language model or output head.
            for block in self.t5.decoder.block:
                for p in block.layer[1].parameters():
                    p.requires_grad_(True)
        if scope in {"decoder", "full"}:
            for p in self.t5.decoder.parameters():
                p.requires_grad_(True)
        if scope == "full":
            for p in self.t5.parameters():
                p.requires_grad_(True)
            # The text encoder is bypassed by raw audio, so training it would only allocate
            # optimizer state without receiving gradients.
            if source == "audio":
                for p in self.t5.encoder.parameters():
                    p.requires_grad_(False)


def token_f1(prediction: str, reference: str) -> float:
    p, r = normalize_text(prediction).split(), normalize_text(reference).split()
    if not p or not r:
        return float(p == r)
    overlap = sum((Counter(p) & Counter(r)).values())
    precision, recall = overlap / len(p), overlap / len(r)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def novelty_metrics(generated: Sequence[str], train_texts: Iterable[str]) -> dict[str, float]:
    """Exact-copy and lexical-overlap guards for a small held-out generation set."""
    train_sets: list[set[str]] = []
    train_norm: set[str] = set()
    for text in train_texts:
        norm = normalize_text(text)
        if not norm:
            continue
        train_norm.add(norm)
        train_sets.append(set(norm.split()))

    gen_norm = [normalize_text(x) for x in generated]
    exact = sum(x in train_norm for x in gen_norm) / max(len(gen_norm), 1)
    nearest = []
    for norm in gen_norm:
        words = set(norm.split())
        if not words or not train_sets:
            nearest.append(0.0)
            continue
        nearest.append(max(len(words & t) / max(len(words | t), 1) for t in train_sets))

    tokens = [w for x in gen_norm for w in x.split()]
    bigrams = [(ws[i], ws[i + 1]) for x in gen_norm for ws in [x.split()] for i in range(len(ws) - 1)]
    return {
        "generated_count": float(len(generated)),
        "train_exact_copy_rate": float(exact),
        "mean_nearest_train_jaccard": float(sum(nearest) / max(len(nearest), 1)),
        "unique_output_rate": float(len(set(gen_norm)) / max(len(gen_norm), 1)),
        "distinct_1": float(len(set(tokens)) / max(len(tokens), 1)),
        "distinct_2": float(len(set(bigrams)) / max(len(bigrams), 1)),
    }
