from __future__ import annotations

import json

import pytest
import torch

from salm.a2s.scene_text_planner import (
    AudioConditionedT5,
    AudioMemoryAdapter,
    PlannerRow,
    TASK_TO_ID,
    build_targets,
    format_target,
    normalize_text,
    novelty_metrics,
    token_f1,
)


ROW = PlannerRow(
    sample_id="x",
    audio_cap="Rain falls on a metal roof",
    reasoning="The speaker reacts to the worsening rain.",
    speech_text="We should get inside before it gets worse.",
)


def test_structured_target_keeps_all_three_supervision_fields():
    text = format_target(ROW, "structured")
    assert text == (
        "caption: Rain falls on a metal roof\n"
        "reasoning: The speaker reacts to the worsening rain.\n"
        "speech: We should get inside before it gets worse."
    )


def test_multitask_rotates_deterministically_across_epochs():
    one, ids_one = build_targets([ROW], "multitask", epoch=0, indices=[0])
    two, ids_two = build_targets([ROW], "multitask", epoch=1, indices=[0])
    three, ids_three = build_targets([ROW], "multitask", epoch=2, indices=[0])
    assert one == [ROW.speech_text]
    assert two == [ROW.audio_cap]
    assert three == [ROW.reasoning]
    assert [ids_one.item(), ids_two.item(), ids_three.item()] == [
        TASK_TO_ID["speech"], TASK_TO_ID["caption"], TASK_TO_ID["reasoning"]
    ]


def test_audio_adapter_preserves_frames_and_prepends_task_token():
    adapter = AudioMemoryAdapter(ambient_dim=12, t5_dim=16, layers=1, heads=4, dropout=0)
    out = adapter(torch.randn(3, 7, 12), torch.tensor([0, 1, 2]))
    assert out.shape == (3, 8, 16)
    assert torch.isfinite(out).all()


def test_novelty_metrics_detect_copy_and_diversity():
    m = novelty_metrics(["Hello there", "new words here"], ["hello, there!", "old phrase"])
    assert m["train_exact_copy_rate"] == 0.5
    assert m["unique_output_rate"] == 1.0
    assert m["mean_nearest_train_jaccard"] >= 0.5
    assert normalize_text("Hello, THERE!") == "hello there"
    assert token_f1("one two", "one three") == pytest.approx(0.5)


def test_external_encoder_outputs_work_with_installed_t5_api():
    transformers = pytest.importorskip("transformers")
    config = transformers.T5Config(
        vocab_size=32,
        d_model=16,
        d_ff=32,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        decoder_start_token_id=0,
        eos_token_id=1,
        pad_token_id=0,
    )
    wrapper = AudioConditionedT5(
        transformers.T5ForConditionalGeneration(config),
        ambient_dim=12,
        adapter_layers=1,
        adapter_heads=4,
        dropout=0,
    )
    audio = torch.randn(2, 5, 12)
    tasks = torch.tensor([0, 0])
    labels = torch.tensor([[4, 5, 1], [6, 7, 1]])
    out = wrapper.forward_audio(audio, tasks, labels)
    assert out.logits.shape == (2, 3, 32)
    generated = wrapper.generate_audio(audio, tasks, max_new_tokens=2)
    assert generated.shape[0] == 2
