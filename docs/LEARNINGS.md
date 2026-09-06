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

## A deterministic predictor cannot be taught a one-to-many mapping, no matter the loss or the capacity

Three attempts to predict speech content from an ambient scene failed the same way, and the
reason is a closed-form property of the objective, not a bug:

    argmin_f E[(f(x) - y)^2]  =  f(x) = E[y | x]

Squared error is minimized by the conditional mean. Cosine loss behaves the same way. And
critically, the diffusion loss did too, because the stochasticity in a diffusion model lives
in the **sampler's noise** -- it varies the *realisation* of a given conditioning. A
deterministic network sitting outside that loop, emitting one fixed conditioning vector
trained against specific targets, collapses exactly like a regressor. Putting randomness
somewhere in the system is not the same as putting it in the component that has to make the
choice.

Three corrections to the intuitions that delayed this diagnosis, all worth keeping:

1. **It is not caused by duplicate targets.** Our dataset was a strict bijection -- one input,
   one target. The averaging happens across **neighbours**: near-identical inputs with
   unrelated targets, which a smooth function cannot separate, so the competing gradients
   balance at the mean.
2. **It shows up on validation regardless of memorisation.** Even a model that fits every
   training pair perfectly has no memorised answer for an unseen input, and interpolating
   between known neighbours *is* averaging. This is why more capacity never helped, and why
   train/val divergence is the wrong thing to look at for this failure.
3. **Whether the mean is harmful depends on the target's nature.** For texture-like targets
   (ambient audio, characterised by its statistics) the mean of plausible outputs is still a
   plausible output -- averaging degrades gracefully, which is why the reverse direction of the
   same architecture worked fine. For symbolic targets (phonemes, words) the mean is
   **off-manifold**: the average of two utterances is not a blurry utterance, it is not an
   utterance. Ask which kind of target you have before choosing a regression objective.

**The fix is to make the prediction categorical and then sample.** Quantize the target space
into K real units, predict a distribution over them with cross-entropy (whose optimum is the
true conditional distribution, not its mean), and sample at inference. The mean of a
distribution over categories is not a category, so the collapse becomes impossible rather than
merely discouraged -- and every output is on-manifold by construction. Make it autoregressive,
or independent per-position sampling reintroduces incoherence from a different direction.

Published precedent for the identical failure and identical fix: lip-to-speech, 65-78% WER
with mel regression to 50.2/44.8% with discrete units (Choi et al., arXiv 2305.19603).

## When an ablation gives the SAME answer under every hyperparameter, that invariance is the result

A regularization sweep was run to fix severe overfitting (train loss ~9x below val loss). It
failed its stated success criterion: the validation-loss minimum barely moved. But the
conditioning-strength metric, measured at each run's own loss minimum, came back at +0.0220,
+0.0203, +0.0210, +0.0203, +0.0216, +0.0222, +0.0217 across seven configurations spanning
three vocabulary sizes, 3x dropout, 10x weight decay, two model depths, and label smoothing.

A spread of 0.0019 under interventions that large is not a weak result to be improved on. It
is a **measurement of the data**, not of the model: that quantity is how much the conditioning
input genuinely predicts the target, and no amount of model tuning moves it. Everything above
it in the unregularized runs was memorisation, which the regularization correctly suppressed
while leaving the floor untouched.

**The lesson:** a sweep that "fails" by moving nothing may be delivering a more valuable
number than one that succeeds. Before concluding a lever is wrong, check whether the metric
was *invariant* rather than merely *unimproved* — invariance across genuinely different
interventions is evidence about the problem, and it separates the irreducible signal from the
part that was overfitting. Design sweeps so that this is readable: measure the metric at each
run's own operating point (here, its own loss minimum), not at a fixed epoch, or the
invariance is hidden by differences in where each run happens to be in its trajectory.

**Corollary on where to look next.** Once an invariance like this is established, tuning the
model is settled business — the remaining levers are the representation, the objective's
structure, or the data. That redirection is worth more than the failed sweep cost.

---

## A conditioning metric that grows with overfitting is measuring retrieval, not conditioning

*2026-09-05, from the dedup and frame-level history curves (jobs 439442, 436344)*

`val_scene_gap` — `loss(rolled ambient) − loss(true ambient)` — was the headline number for
the run-length experiment: +0.041 for dedup_k512 against +0.021 for frame-level, with ~0.001
seed variance. Reading the full epoch curves rather than the value at the loss minimum shows
it **rises monotonically as the model overfits**, in every run measured:

| run | val min | scene_gap @ val min | scene_gap @ ep99 | val loss @ ep99 |
|---|---|---|---|---|
| dedup_k512 | ep9 | +0.0414 | **+0.1329** | 3.07 (from 1.50) |
| dedup_k1024 | ep10 | +0.0370 | +0.1060 | 3.21 (from 1.62) |
| dedup_k2048 | ep9 | +0.0322 | +0.0946 | 3.41 (from 1.76) |
| frame k512 | ep7 | +0.0220 | +0.0873 | 2.52 (from 1.14) |
| frame k2048 | ep7 | +0.0210 | +0.0586 | 2.95 (from 1.44) |

So an overfit frame-level model posts a *higher* scene_gap (+0.087) than the run-length model
at its optimum (+0.041), while being decisively worse by val loss. Taken at face value the
metric would have ranked them backwards.

**Why it happens, and why it is not a bug.** The ablation is computed on val, where nothing is
memorised — but memorisation still sharpens the *mapping* from ambient to content, because the
ambient becomes a better retrieval key into memorised training utterances. Rolling it destroys
the key, so the penalty grows. The metric is honestly reporting "how much the model leans on
the ambient"; it was our reading of that as "how well the scene is understood" that was wrong.
Leaning harder on the ambient as a lookup index is not scene understanding.

**What survives.** The dedup-vs-frame-level comparison stands, because both arms were read at
their own val minimum — which the earlier regularization lesson on this page had already made
standard practice. That convention is the only reason this confound did not invalidate the
result outright.

**The rule:** an ablation-gap metric is only comparable at matched val loss, and the epoch must
be quoted with the number. More generally, before trusting any metric as a measure of quality,
plot it against the overfitting axis. If it improves monotonically while val loss doubles, it
is measuring something the model gains by memorising, and the direction it points is only
trustworthy at a fixed point on that axis. This is cheap to check — the numbers were already
sitting in `history.json` — and it should be checked once per metric, not once per project.

---

## An argument about continuous means does not transfer to discrete tokens

*2026-09-06, jobs 443379 / 443475*

**The principle:** when a project is built around avoiding a specific failure mode, check that
the argument still applies after you change representation. Inherited reasoning is invisible —
it lives in a comment nobody re-reads, and it keeps steering decisions long after its premise
has gone.

A speech-generation system failed at regression for a well-understood reason:

    argmin_f E[(f(x) − y)²]  =  f(x) = E[y | x]

A squared-error model returns the conditional mean, and for symbolic targets the mean of
several plausible outputs is not a plausible output — the average of five sentences is not a
sentence. That diagnosis was correct and it motivated a move to discrete codes with
autoregressive sampling, which was the right move.

It also left this comment in the sampler, and the comment then governed every generation the
project produced for weeks:

> Sampling, never argmax: argmax walks straight back into the conditional mean that the
> discrete formulation exists to escape.

**That is false, and the falseness is the whole lesson.** Argmax over a categorical
distribution returns the most likely *sequence* — a real utterance, a member of the output
space. The conditional mean of a distribution over discrete tokens is not what argmax computes
and is not reachable by it; a mean requires a metric space, and category labels have none. The
argument was carried across the boundary between continuous and discrete outputs, where its
premise stops holding, and nobody re-derived it because it had been right once.

Measured cost of the mistake: switching to greedy decoding moved the project's intelligibility
metric from 0.452 to 0.369 on a scale where real speech is 0.315 and the autoencoder's own
round-trip ceiling is 0.372. One flag, no retraining, no new data — the single largest
improvement in the project's history, sitting behind a comment.

**How this gets caught earlier.** A comment that justifies a *decision* by citing a *theorem*
should name the theorem's assumptions, so that changing representation makes the mismatch
visible. "Never argmax" is unfalsifiable folklore. "Never argmax **because squared error
returns E[y|x] and the mean of plausible mel trajectories is off-manifold**" carries its own
expiry date: the moment the loss is cross-entropy over categories, the stated reason plainly no
longer applies and the decision is up for review. Write the *because* into the code, not just
the rule.

**The generalisation.** Inherited constraints deserve periodic re-derivation, especially the
ones that feel settled. The dangerous ones are not the decisions still being argued about —
those get revisited. They are the ones that were correct in their original context, were
recorded as rules rather than as arguments, and survived the change that invalidated them.

---

## A checkpoint selector inherits every pathology of its selection metric

*2026-09-06, from the dedup content-LM runs*

**The principle:** `checkpoint_best.pt` is only "best" with respect to whatever you chose to
maximise. If that metric has a failure mode, the selector silently converts it into a shipped
artifact, and the filename then hides the problem behind a word that sounds like a verdict.

Three runs saved `checkpoint_best` on highest `val_scene_gap` — an ablation gap measuring how
much the model relies on its conditioning input. Separately, that metric turned out to rise
**monotonically with overfitting** (it grows as the conditioning input becomes a sharper
retrieval key into memorised training examples). The consequence:

| run | `best_loss` | val_loss | `best` | val_loss |
|---|---|---|---|---|
| k512 | ep9 | 2.37 | ep87 | **5.01** |
| k1024 | ep10 | 2.42 | ep91 | **4.93** |
| k2048 | ep9 | 2.49 | ep85 | **4.95** |

`checkpoint_best` was, by construction, close to the most overfit checkpoint of each run —
val_loss more than double the optimum — because it was selected by a criterion that rewards
memorisation. Under a stochastic decoder the two tags scored alike, so the problem stayed
hidden; under a deterministic one the honest checkpoint won by a wide, consistent margin.

**What to do instead.** Save on the metric you will defend in writing, and name files after the
criterion rather than after a judgement: `checkpoint_minval.pt`, `checkpoint_maxgap.pt`. A
reader then cannot mistake a selection rule for a conclusion. Where a secondary metric is worth
tracking, track it — but do not let it name a file `best`, because six weeks later "we used the
best checkpoint" is what gets written in the paper.

**The deeper point, and the reason this pairs with the entry above on scene_gap:** any metric
used for *selection* must first be validated against the overfitting axis, not merely against
intuition about what it measures. A quantity that improves while held-out loss doubles is
measuring something the model gains by memorising. It may still be diagnostically useful. It
must never be the thing you optimise or select on.
