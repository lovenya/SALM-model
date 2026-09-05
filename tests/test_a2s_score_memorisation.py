"""Pure-function tests for the memorisation scorer. No model, no audio."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "a2s"))
from score_memorisation import derangement, longest_common_span  # noqa: E402


def test_derangement_has_no_fixed_point():
    """A clip scored against its own reference would fake the whole memorisation signal."""
    for n in range(2, 25):
      for seed in range(30):
        p = derangement(n, seed)
        assert sorted(p.tolist()) == list(range(n)), "not a permutation"
        assert not (p == np.arange(n)).any(), f"fixed point at n={n} seed={seed}"


def test_derangement_degenerate_sizes():
    assert derangement(0, 0).tolist() == []
    assert derangement(1, 0).tolist() == [0]  # nothing to derange; must not hang


def test_derangement_is_seed_stable():
    assert derangement(10, 7).tolist() == derangement(10, 7).tolist()


def test_longest_span_basics():
    assert longest_common_span([], ["a"]) == 0
    assert longest_common_span(["a", "b", "c"], ["x", "y"]) == 0
    assert longest_common_span(["a", "b", "c"], ["a", "b", "c"]) == 3


def test_longest_span_requires_contiguity():
    """The whole point: scattered hits are the failure mode, not a success."""
    a = ["the", "cat", "sat", "on", "the", "mat"]
    assert longest_common_span(a, ["the", "dog", "sat", "under", "the", "rug"]) == 1
    assert longest_common_span(a, ["and", "the", "cat", "sat", "quietly"]) == 3


def test_longest_span_takes_the_longest_of_several_runs():
    a = ["one", "two", "x", "three", "four", "five"]
    b = ["one", "two", "y", "three", "four", "five"]
    assert longest_common_span(a, b) == 3


def test_longest_span_is_symmetric():
    a, b = ["p", "q", "r", "s"], ["z", "q", "r", "z"]
    assert longest_common_span(a, b) == longest_common_span(b, a) == 2


# pytest is not installed in the project venv, and CLAUDE.md forbids the agent installing it,
# so these run standalone. Same plain `def test_*` shape as the rest of tests/, so a pytest
# run elsewhere still collects them.
if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    bad = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            bad += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    raise SystemExit(1 if bad else 0)
