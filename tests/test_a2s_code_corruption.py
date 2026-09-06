"""Tests for nearest-neighbour code corruption. Pure tensor logic -- no model, no GPU."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from salm.a2s.code_corruption import (  # noqa: E402
    corrupt_codes, measure_code_error_rate, neighbour_table,
)


def _ring(n=8, d=4):
    """Codes on a circle: each one's nearest neighbours are its ring-adjacent codes."""
    ang = torch.arange(n).float() * (2 * np.pi / n)
    c = torch.zeros(n, d)
    c[:, 0], c[:, 1] = torch.cos(ang), torch.sin(ang)
    return c


def test_neighbours_exclude_self():
    nb = neighbour_table(_ring(), k=3)
    for i in range(nb.shape[0]):
        assert i not in nb[i].tolist(), f"code {i} is its own neighbour"


def test_neighbours_are_actually_nearest():
    nb = neighbour_table(_ring(8), k=2)
    for i in range(8):
        assert set(nb[i].tolist()) == {(i - 1) % 8, (i + 1) % 8}


def test_rate_zero_is_bit_identical():
    idx = torch.randint(0, 8, (4, 20))
    assert torch.equal(corrupt_codes(idx, neighbour_table(_ring()), 0.0), idx)


def test_corruption_only_ever_substitutes_neighbours():
    """The whole design: a random code would make the target audio wrong for the codes."""
    nb = neighbour_table(_ring(8), k=2)
    idx = torch.randint(0, 8, (16, 40))
    g = torch.Generator().manual_seed(0)
    out = corrupt_codes(idx, nb, 0.5, generator=g)
    changed = out != idx
    for a, b in zip(idx[changed].tolist(), out[changed].tolist()):
        assert b in nb[a].tolist(), f"{a} -> {b} is not a nearest neighbour"


def test_rate_is_approximately_honoured():
    nb = neighbour_table(_ring(16), k=4)
    idx = torch.randint(0, 16, (64, 128))
    g = torch.Generator().manual_seed(1)
    frac = (corrupt_codes(idx, nb, 0.25, generator=g) != idx).float().mean().item()
    assert 0.20 < frac < 0.30, f"asked for 0.25, changed {frac:.3f}"


def test_shape_and_dtype_preserved():
    nb = neighbour_table(_ring())
    idx = torch.randint(0, 8, (3, 11))
    out = corrupt_codes(idx, nb, 0.4, generator=torch.Generator().manual_seed(2))
    assert out.shape == idx.shape and out.dtype == idx.dtype


def test_error_rate_measurement():
    nbrs = neighbour_table(_ring(8), k=2).numpy()
    true = np.array([[0, 1, 2, 3]])
    perfect = measure_code_error_rate(true.copy(), true, nbrs)
    assert perfect["error_rate"] == 0.0
    near = measure_code_error_rate(np.array([[1, 2, 3, 4]]), true, nbrs)   # each off by one
    assert near["error_rate"] == 1.0 and near["share_of_errors_that_are_near"] == 1.0
    far = measure_code_error_rate(np.array([[4, 5, 6, 7]]), true, nbrs)    # opposite side
    assert far["error_rate"] == 1.0 and far["share_of_errors_that_are_near"] == 0.0


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
