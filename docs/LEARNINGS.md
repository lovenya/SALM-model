# Learnings

## What this document is

A running list of **portable technical lessons** — gotchas, mechanisms, and fixes that are
worth remembering independent of this project. Where `EXPERIMENT_LOG.md` records *why we made
a decision here*, this records *the general thing that would trip you up again on a different
project* — the kind of thing worth being able to explain cleanly in an interview: what broke,
why it broke at the mechanism level, and how it was actually fixed (not just "it works now").

Each entry should stand on its own without needing this project's context. A pointer back into
`EXPERIMENT_LOG.md` is enough provenance; the entry itself should be the reusable part.

Organized by topic as entries accumulate. Newest entry per topic goes on top.

---

## PyTorch / deep learning engineering

### A frozen module's `.eval()` can break `.backward()` through it — `requires_grad=False` and `.eval()` are independent, and RNNs make that bite

**The symptom:** `RuntimeError: cudnn RNN backward can only be called in training mode` — raised
on `.backward()`, not on the forward pass, and only when the offending module contains an
`nn.RNN`/`nn.GRU`/`nn.LSTM` running on a CUDA device with cuDNN enabled (the default). Nothing
about the error message points at the actual cause.

**The mechanism:** two axes of "frozen" get conflated, and they're genuinely independent:

- `requires_grad_(False)` on a module's parameters controls whether *that module's own weights*
  receive gradient and get updated by the optimizer.
- `.eval()` vs `.train()` controls *forward-pass behavior* — BatchNorm uses running stats
  instead of batch stats, Dropout becomes a no-op, and (the part that isn't obvious) **cuDNN's
  fused RNN kernel decides whether to allocate the "reserve space" its own backward pass will
  need, based on whether the forward pass ran in training mode.**

A common pattern — load a pretrained submodule, freeze its parameters, call `.eval()` on it
because "it's frozen, we don't want dropout/batchnorm noise," then use it as a differentiable
step inside a larger graph you DO want to backprop through — works for almost everything
(Linear, Conv, LayerNorm, attention...) but silently sets up this exact crash the moment the
frozen submodule contains an RNN, because `eval()` skips the reserve-space allocation
regardless of whether upstream code will later try to run `.backward()` through it. The
parameters being frozen is irrelevant to this — the crash happens even though nothing about the
RNN's own weights was ever going to change; the issue is purely about gradient *flowing through*
it to reach something else upstream.

**The fix:** don't call a blanket `.eval()`/`.train()` on the whole module. Put the parts that
need frozen-statistics behavior (BatchNorm, etc.) in `.eval()`, and put the RNN specifically
back in `.train()`:

```python
frozen_model.eval()          # BatchNorm: use running stats, the actual reason to freeze at all
frozen_model.rnn.train()     # GRU/LSTM/RNN: needs train-mode for cuDNN's backward reserve space
for p in frozen_model.parameters():
    p.requires_grad_(False)  # this is what actually keeps it frozen -- orthogonal to the above
```

Side effect worth knowing about: `.train()` on an RNN with `dropout > 0` between layers
re-enables that inter-layer dropout. If the frozen module is meant to be a fully deterministic
scorer, either accept that small stochasticity (harmless if its weights never update and it's
just providing a training signal to something else) or construct it with `dropout=0` if
determinism matters more than matching how it was originally trained.

**The general lesson, for an interview:** "frozen" is not one flag in PyTorch — parameter-level
freezing (`requires_grad`) and module-mode (`train`/`eval`) are separate mechanisms that usually
happen to move together but aren't the same thing, and backend-specific kernels (cuDNN here) can
depend on the mode flag for reasons that have nothing to do with dropout/batchnorm — in this
case, whether backward-pass memory was even allocated during the forward pass. When gradient
needs to flow *through* a frozen component to reach something upstream, both axes need to be
reasoned about independently, not just "set eval and requires_grad=False and move on."

(Caught in: SALM-model, 2026-08-26, wiring a frozen CTC head into a diffusion training loop for
a content-supervision loss — see `docs/EXPERIMENT_LOG.md` 2026-08-26 for the full story.)
