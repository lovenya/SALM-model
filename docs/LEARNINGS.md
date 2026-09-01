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

## `from_pretrained` on the wrong class can silently give you a randomly-initialized model

**The bug.** A checkpoint published as `ClapModel` stores audio weights under
`audio_model.audio_encoder.*`. Loading it through `ClapAudioModel.from_pretrained(...)` --
which expects `audio_encoder.*` -- does not raise. HuggingFace reports the checkpoint's keys
as UNEXPECTED, reports the model's own keys as MISSING, initializes them fresh, and returns a
**perfectly runnable model with random weights**.

**Why it survived for a week.** The warning is printed, but it looks exactly like the routine
head-mismatch noise you see whenever you load a base checkpoint into a task model
("MISSING: those params were newly initialized because missing from the checkpoint. Consider
training on your downstream task"). In our case it was 33 lines of it, buried in a job log, and
read past repeatedly.

Worse, one code path accidentally worked: a precompute script constructed the wrapper and then
**overwrote** `.audio_model` with a correctly-loaded tower, so the cached features were right.
Only the second code path -- generation -- used the broken constructor. Training metrics
therefore looked completely normal while every generated sample was conditioned on noise.

**The general lesson.** In PyTorch/HF, "it loaded and it runs" is not evidence that weights
were loaded. `strict=False` semantics are pervasive: `from_pretrained` across mismatched
classes, `load_state_dict(..., strict=False)`, and adapter installs that must happen *before*
a load or get silently dropped as unexpected keys.

**What to actually do:**
1. **Never let a load warning scroll past unexamined.** If a load prints MISSING keys for
   anything other than a task head you intend to train, treat it as an error.
2. **Assert a numeric invariant, not the absence of an exception.** Our fix added a
   fail-fast parity check: encode a handful of files live, compare against the cached
   features, require cosine > 0.999 and rank-1 self-retrieval. It runs before every
   generation job and would have caught this on day one.
3. **Suspect any component that exists on two code paths.** The one where a bug hides is the
   one nobody looks at -- here, the path where the object was constructed and used directly
   rather than being patched up afterwards.

## A checkpoint selector that measures the wrong component silently ruins every downstream result

`select_metric: val_clap_r1` scored the *conditioner's* retrieval accuracy. The question the
project actually cared about was whether the *U-Net* could render intelligible speech. Those
are different components, and their curves have different shapes: retrieval peaked at epoch
2-8 and then declined for the rest of every run, while the U-Net's own reconstruction error
kept improving to epoch ~92.

So `checkpoint_best` was **always** a near-untrained renderer -- epoch 6, 6, 3, 2, 4 across
five runs -- and every listening test, every generation comparison, and the single ablation
that redirected the entire research programme were all run on models that had barely trained.
Nothing errored. The numbers all looked plausible.

**The lesson:** your model-selection metric silently defines what "the model" means for every
downstream artifact. Before trusting any generated output, check which epoch produced it and
whether the metric that chose it is causally connected to the property you are judging. A
metric can be perfectly valid (`val_clap_r1` genuinely measured the conditioner) and still be
the wrong selector.

**Corollary that cost us weeks:** when adding a new mechanism, add a metric that is *sensitive
to that mechanism* in the same commit. We added a content channel and judged it with metrics
computed from the conditioner's output, which the channel never touches. The metric we
eventually needed -- roll the conditioning sequence along time and check whether the loss gets
worse -- is ~20 lines, runs in validation, and would have falsified the mechanism at epoch 3
instead of after two training runs, a generation job, and a listening session.

## A "sanity check" checkpoint that shares a variable with the thing it's checking isn't independent

The end-to-end sequential-conditioning job (2026-09-01) generated `oracle` content from its
own last checkpoint as a designed upper bound: if the concat channel still renders real
content correctly, the failure is in the content source, not the renderer. It scored at the
noise floor -- as bad as everything else -- which looked like the concat mechanism itself had
broken.

It hadn't. `diffusion_loss` routed `content_seq` from the new sequence head's own output
*unconditionally* whenever the feature was active, with no code path that ever showed the
renderer real content during training. So the "upper bound" checkpoint was never trained on
the thing the upper-bound test fed it -- oracle content was out-of-distribution for that
specific checkpoint, and a bad score there said nothing about the mechanism a prior run
(`content_concat_v4`) had already validated with the same mechanism trained the normal way.

**The lesson:** an upper-bound / sanity-check generation using a checkpoint from the run under
test is only independent evidence if that checkpoint was actually exposed to the reference
condition during training. If a new training mode reroutes what a shared component learns
from, reusing that component's checkpoint to "check against the old baseline" silently tests
something else -- here, out-of-distribution generalization, not renderer capability. Trace
the actual data flow (what did THIS checkpoint's loss ever compute against?) before trusting
a checkpoint-reuse control, not just what the generation flag says it's doing.
