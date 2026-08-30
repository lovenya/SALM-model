from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

import torch

from salm.generate import deranged_indices


class _FakeAudioModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=768)
        self.weight = torch.nn.Parameter(torch.ones(1))


class ContentControlTests(unittest.TestCase):
    def test_derangement_is_deterministic_and_has_no_fixed_points(self):
        first = deranged_indices(16, 123)
        self.assertEqual(first, deranged_indices(16, 123))
        self.assertEqual(sorted(first), list(range(16)))
        self.assertTrue(all(i != j for i, j in enumerate(first)))

    def test_single_sample_shuffled_control_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            deranged_indices(1, 123)

    def test_zero_content_projects_to_exact_zero(self):
        from salm.a2s.content_concat import ContentProjector

        for mode in ("broadcast", "full"):
            projector = ContentProjector(12, extra_channels=3, n_freq=5, mode=mode)
            out = projector(torch.zeros(2, 7, 12))
            self.assertEqual(tuple(out.shape), (2, 3, 7, 5))
            self.assertTrue(torch.equal(out, torch.zeros_like(out)))
            self.assertIsNone(projector.proj.bias)


class ClapLoaderTests(unittest.TestCase):
    def _fake_transformers(self):
        module = types.ModuleType("transformers")
        feature_cls = mock.Mock()
        feature_cls.from_pretrained.return_value = object()
        clap_cls = mock.Mock()
        clap_cls.from_pretrained.return_value = types.SimpleNamespace(
            audio_model=_FakeAudioModel()
        )
        module.ClapFeatureExtractor = feature_cls
        module.ClapModel = clap_cls
        return module, feature_cls, clap_cls

    def test_live_loader_uses_full_clap_model_audio_tower(self):
        from salm.encoders.clap_encoder import ClapAudioSequenceEncoder

        module, feature_cls, clap_cls = self._fake_transformers()
        with mock.patch.dict(sys.modules, {"transformers": module}):
            encoder = ClapAudioSequenceEncoder("fake-clap", device="cpu")
        feature_cls.from_pretrained.assert_called_once_with("fake-clap")
        clap_cls.from_pretrained.assert_called_once()
        self.assertEqual(encoder.feature_dim, 768)

    def test_preloaded_audio_tower_avoids_second_full_model_load(self):
        from salm.encoders.clap_encoder import ClapAudioSequenceEncoder

        module, _, clap_cls = self._fake_transformers()
        injected = _FakeAudioModel()
        with mock.patch.dict(sys.modules, {"transformers": module}):
            encoder = ClapAudioSequenceEncoder(
                "fake-clap", device="cpu", audio_model=injected
            )
        clap_cls.from_pretrained.assert_not_called()
        self.assertIs(encoder.audio_model, injected)


if __name__ == "__main__":
    unittest.main()
