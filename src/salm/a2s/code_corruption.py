"""
Nearest-neighbour code corruption -- augmentation that makes a renderer robust to the code
errors its language model will actually make.

WHY NOT JUST TRAIN ON LM-SAMPLED CODES. That is the obvious experiment and it is ill-posed.
Pairing LM-sampled codes with the ORIGINAL audio teaches the renderer to produce the true
utterance from codes that say something else -- that is, to ignore its content channel. It
would collapse back to `argmin E[(f(x)-y)^2] = E[y|x]`, the conditional-mean failure this
project exists to escape, and it would do so while every training curve looked healthy.

The problem is that for an LM-sampled code sequence we have no matching target audio, and
inventing one is the whole difficulty.

WHAT THIS DOES INSTEAD. Corrupt the TRUE codes by substituting each chosen frame with one of
its nearest neighbours in codebook space. Neighbours in a k-means codebook over CTC-probe
features are acoustically similar by construction, so the original audio remains an
approximately valid target -- the pairing stays honest -- while the renderer is forced to
tolerate the kind of local error the LM produces. Substitution is deliberately NOT uniform
random: a random code is acoustically unrelated, the target audio becomes wrong for it, and
the augmentation would teach exactly the "ignore the content channel" lesson it exists to
avoid.

CALIBRATION MATTERS. The corruption rate should match the error rate the LM actually produces,
which is measurable (`measure_code_error_rate`) rather than guessable. Too low and the
augmentation is a no-op; too high and the target genuinely stops matching the codes.
"""

from __future__ import annotations

import numpy as np
import torch


def neighbour_table(codes: torch.Tensor, k: int = 8) -> torch.Tensor:
    """
    (K, D) codebook -> (K, k) indices of each code's k nearest OTHER codes.

    Cosine distance on L2-normalised rows, matching how the codebook itself was fitted; a code
    is never its own neighbour, so a substitution always changes something.
    """
    x = torch.nn.functional.normalize(codes.float(), dim=1)
    sim = x @ x.t()
    sim.fill_diagonal_(-2.0)
    return sim.topk(min(k, sim.shape[0] - 1), dim=1).indices


def corrupt_codes(
    indices: torch.Tensor,
    nbrs: torch.Tensor,
    rate: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    (B, T) code indices -> same shape with `rate` of the frames replaced by a random one of
    that frame's nearest neighbours. rate <= 0 returns the input unchanged, so the zero arm of
    an ablation is bit-identical to no augmentation rather than merely similar.
    """
    if rate <= 0.0:
        return indices
    device = indices.device
    mask = torch.rand(indices.shape, device=device, generator=generator) < rate
    if not bool(mask.any()):
        return indices
    k = nbrs.shape[1]
    pick = torch.randint(0, k, indices.shape, device=device, generator=generator)
    swapped = nbrs.to(device)[indices.clamp(min=0), pick]
    return torch.where(mask, swapped, indices)


def measure_code_error_rate(pred: np.ndarray, true: np.ndarray, nbrs: np.ndarray) -> dict:
    """
    Compare LM-sampled code frames against the truth, so the corruption rate is calibrated
    rather than chosen. Reports both the raw disagreement and how much of it lands on a NEAR
    neighbour -- the latter is the part nearest-neighbour corruption actually simulates, and if
    it is small then this augmentation is modelling the wrong error and should be rethought.
    """
    n = min(pred.shape[0], true.shape[0])
    p, t = pred[:n].reshape(-1), true[:n].reshape(-1)
    wrong = p != t
    near = np.zeros_like(wrong)
    if wrong.any():
        near[wrong] = np.array([bool(pi in nbrs[ti]) for pi, ti in zip(p[wrong], t[wrong])])
    return {
        "frames": int(p.size),
        "error_rate": float(wrong.mean()),
        "near_miss_rate": float(near.mean()),
        "share_of_errors_that_are_near": float(near.sum() / max(wrong.sum(), 1)),
    }
