"""Tests for the bag-of-codes plan target. Pure tensor logic -- no model, no GPU."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from salm.a2s.content_lm_dedup import plan_targets  # noqa: E402


def test_multi_hot_marks_exactly_the_codes_present():
    tok = torch.tensor([[3, 7, 3, 0], [1, 1, 1, 0]])
    lens = torch.tensor([3, 3])
    p = plan_targets(tok, lens, num_codes=8)
    assert p[0].nonzero().flatten().tolist() == [3, 7]
    assert p[1].nonzero().flatten().tolist() == [1]


def test_padding_is_excluded():
    """The whole point: a padded slot holding code 5 must not put 5 in the plan. Without the
    length mask every row inherits its padding and the target goes nearly constant."""
    tok = torch.tensor([[2, 5, 5, 5]])          # only the first token is real
    p_short = plan_targets(tok, torch.tensor([1]), num_codes=8)
    p_full = plan_targets(tok, torch.tensor([4]), num_codes=8)
    assert p_short.nonzero()[:, 1].tolist() == [2]
    assert p_full.nonzero()[:, 1].tolist() == [2, 5]


def test_values_are_binary_not_counts():
    """A code appearing 30 times must score 1.0, not 30 -- BCE targets have to be in [0,1]."""
    tok = torch.zeros((1, 30), dtype=torch.long)
    p = plan_targets(tok, torch.tensor([30]), num_codes=4)
    assert p.max().item() == 1.0 and p[0, 0].item() == 1.0


def test_zero_length_row_is_all_zero():
    p = plan_targets(torch.tensor([[4, 4]]), torch.tensor([0]), num_codes=8)
    assert p.sum().item() == 0.0


def test_order_free():
    """A plan must not encode order, or it is just a copy of the sequence."""
    a = plan_targets(torch.tensor([[1, 2, 3, 0]]), torch.tensor([3]), 8)
    b = plan_targets(torch.tensor([[3, 1, 2, 0]]), torch.tensor([3]), 8)
    assert torch.equal(a, b)


def test_shape_and_range():
    p = plan_targets(torch.randint(0, 64, (5, 20)), torch.full((5,), 20), num_codes=64)
    assert p.shape == (5, 64)
    assert p.min() >= 0.0 and p.max() <= 1.0


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    bad = 0
    for name, fn in fns:
        try:
            fn(); print(f"  PASS  {name}")
        except AssertionError as e:
            bad += 1; print(f"  FAIL  {name}: {e}")
    print(f"\n{len(fns)-bad}/{len(fns)} passed")
    raise SystemExit(1 if bad else 0)
