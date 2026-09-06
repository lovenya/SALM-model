# Research directions

## What this document is

**The idea space** -- alternatives, architectures, and open research questions we have
considered for A2S, with the reasoning for and against each. Deliberately separate from
`EXPERIMENT_LOG.md`, which records what was actually run and what it measured.

Things land here when they are *discussed*; they graduate to `OPEN_ITEMS.md` when they become
a concrete plan, and to `EXPERIMENT_LOG.md` when a job produces a number. Rejected ideas stay,
with the reason, because the easiest thing to lose is why we chose against something -- and
rejected ideas get re-proposed otherwise.

Sources are cited so a claim can be re-checked rather than inherited.

---

## The problem, restated

A2S = ambient audio -> speech. Given a 10.24 s non-speech scene, generate plausible speech.
Two separable failures, and keeping them separate is the single most useful frame we have:

- **F1 -- content origination.** WHICH words to say. Information-theoretic, not architectural:
  an ambient scene does not determine a transcript. Our ridge probe quantified the available
  signal at R@1@16 ~ 0.45 vs 0.0625 chance -- real, but topic-level.
- **F2 -- phonetic realizability.** CAN the model render crisp words at all, given content.
  Purely architectural, and fixable.

Word-error-rate against the true transcript conflates the two and cannot separate them: a
model saying wrong-but-real words scores the same as one producing babble.

**Where we are (2026-08-30):** F2 is the active work. The calibrated intelligibility floor is
0.3146 (real speech) / 0.3624 (VAE round-trip, i.e. the ceiling for this stack); every
generation we have produced sits at 0.944-0.990. That gap is the whole problem.

---

## The scale problem, which contextualizes everything below

| system | speech hours | ratio to ours |
|---|---|---|
| **Ours** | **~24 h** | 1x |
| BVS (video -> intelligible speech) | ~1,611 h | ~68x |
| VoiceLDM | ~6,750 h | ~284x |
| DiTTo-TTS | 82,000 h | ~3,400x |

Our "42 h" figure is the 10.24 s canvas; speech averages 4.86 s inside it, so the real speech
content is ~24 h. **Every architectural conclusion should be read against a 100-300x data gap.**
It is the strongest single argument for borrowing a pretrained F2 rather than learning it.

---

## Live directions

### D1. Frame-aligned channel concatenation -- ACTIVE, currently the main line

Concatenate the content sequence onto the U-Net's input along the channel axis instead of
cross-attending to it. Content frame *t* is wired to latent frame *t* as an architectural
fact rather than something attention must learn.

**Why:** every modern speech generator does this -- F5-TTS (verified in local source,
`dit.py:162`), E2-TTS, VoiceBox (forced-aligned phones), NaturalSpeech 2 (length-expanded
phonemes). Cross-attention appears only where the sequence is ALREADY frame-aligned
(VoiceLDM's durator, AudioLDM 2's LOA). **The invariant is frame alignment, not
concat-vs-attention.**

**Why we are unusually well placed:** our latent (8, 256, 16) runs at exactly 25 fps and the
CTC probe emits exactly 256 frames -- a 1:1 match, no durator or forced aligner needed. F5
fakes this with filler-token padding; we have it exactly.

**Status:** built, trained 100 epochs (job 432493). `val_content_bind_gap` +0.0087 -> +0.33
and `val_x0_rel_mse` 0.2884 (best ever). Binding confirmed; **intelligibility not yet
measured.**

### D2. Content classifier-free guidance -- DONE 2026-09-06 (job 442847), PARTIAL, not the main line

We have never generated A2S audio with content guidance. The content channel had no null
branch and no separate scale, while the pooled channel the oracle test discredited is
amplified 3x. VoiceLDM uses **dual CFG** with independent `w_desc` and `w_cont`, and reports
that raising `w_cont` is precisely what buys intelligibility (at the cost of description
adherence). InstructPix2Pix is the same two-condition shape and solves it identically.

**RESULT (job 442847).** Implemented on the content LM's code head rather than the diffusion
sampler, with the unconditional branch a rolled batch (a different scene's memory) rather than
a zeroed one, since a zeroed memory is out of distribution. Swept w in {1.0, 1.5, 2.5, 4.0}.

w=1.5 roughly doubles scene control at the first two words (cross-seed agreement 0.11 -> 0.27
and 0.04 -> 0.21) with clarity intact. w >= 2.5 destroys the audio, buying determinism by
pushing codes off-manifold (inter-ASR 0.46 / 0.15 / 0.06 by the third word). So: a real but
small effect with a hard ceiling on w, kept available and not pursued further.

**The control was worth more than the treatment.** A greedy lane included only to calibrate the
cross-seed scale showed that with content codes FIXED, the renderer reproduces the same words at
0.96 / 1.00 / 0.93 for the first three positions. The renderer is not the bottleneck;
essentially all output variance is the LM's sampling. That observation led directly to D8.

### D3. A better content representation

Our content comes from a CTC-trained BiGRU. CTC representations are notoriously **peaky** --
most frames confidently blank, information in sparse spikes -- which is a poor match for a
per-frame conditioning signal. Options:

- ~~**A temporal pre-net** (dilated Conv1d / ConvNeXt-1D) to smear spikes into phone-duration
  plateaus.~~ **KILLED ON MEASUREMENT, 2026-09-06.** The premise is false for our codes. CTC
  *posteriors* are peaky and blank-dominated; we use the *encoder hidden states* before the
  output layer, and they are not. Measured on the cached train codes: every code used (512/512
  and 2048/2048), top-1 code covers 2.2% / 0.7%, top-10 covers 18.5% / 6.0%, effective
  vocabulary 239/512 and 883/2048. Mean run length 2.00 / 1.70 frames = **80 / 68 ms**, which
  is already phone-scale. The plateaus this pre-net would create are there.
- **Self-supervised speech features** -- HuBERT / WavLM / w2v-BERT at 50 Hz, pooled x2 to our
  25 fps. BVS's ablation found SSL tokens beat a text/transcript representation decisively
  (26.4% vs 98.7% WER) on the closest published task to ours.
- **SpeechT5's TTS content encoder** -- VoiceLDM's own choice. DiTTo-TTS's ablation puts the
  content-encoder axis at WER 6.22 -> 3.07, larger than most architecture changes.

### D4. VoiceLDM warm start -- the honest fallback if F2 resists

[VoiceLDM](https://arxiv.org/abs/2309.13664) (ICASSP 2024) is our architecture, already
working: AudioLDM conditioned on BOTH an environmental description AND speech content, using
**AudioLDM's own frozen VAE and HiFi-GAN at 16 kHz**, reporting **3.909% WER** -- better than
ground-truth recordings through the same ASR. **Apache-2.0, training code public since May
2024** (license verified directly).

If we adopt it as the A2S backbone and drive its *content* slot from our ambient conditioner,
our 24 h only has to learn **F1**, because F2 arrives pretrained at 6,750 h. This is the only
direction that plausibly produces intelligible A2S output soon.

**Costs, stated honestly:** VoiceLDM-S is 280M and VoiceLDM-M is 508M against our 185M U-Net,
so weights will not drop into our existing model -- adoption means taking VoiceLDM wholesale,
which sacrifices the shared S2A/A2S backbone narrative. That is a real trade and the user's
call, not a technical default.

### D5. Two-stage semantic-token cascade

Stage 1 (ours, on 24 h): ambient scene -> discrete speech tokens. Stage 2 (**frozen,
pretrained**): tokens -> waveform. This is the AudioLM / SPEAR-TTS / BVS pattern, and it makes
the F1/F2 split architectural rather than analytical.

**Why it is attractive at our scale:** all acoustic, phonetic and speaker burden sits behind a
frozen boundary, so our data only trains the part that is genuinely ours. The earlier
"42 h is a failure zone for codec LMs" objection does not transfer -- that concerned
fine-tuning a full acoustic codec LM end to end.

**Local assets, no downloads needed:** CosyVoice3-0.5B weights (`llm.pt`, `flow.pt`, `hift.pt`)
and training code, Apache-2.0, already on scratch. Its speech tokens run at **25 Hz** --
exactly our latent rate.

**Cheap gate before any building:** tokenize 16 real speech files, resynthesize through the
frozen stage 2, run ASR. ~15 min, pure inference. If round-trip WER is low, stage 2 is proven
and only stage 1 is unknown.

**Supporting evidence:** Choi et al., *Intelligible Lip-to-Speech Synthesis with Speech Units*
(Interspeech 2023, [arXiv 2305.19603](https://arxiv.org/abs/2305.19603)) -- a non-speech-input
-> speech task where mel-regression sat at 65-78% WER and switching the *prediction target* to
discrete units cut it to 50.2/44.8, using no text labels. Their diagnosis is ours: per-frame
L2 in a continuous space is minimized by the blurry mean of the phoneme posterior;
cross-entropy over a discrete vocabulary cannot be gamed that way.

**Caveat on transfer:** lip video *determines* the words; an ambient scene does not. Do not
expect low WER-vs-reference and do not steer on it.

### D6. F1 as the actual research contribution

Confirmed across the literature: **no system infers speech content from a non-speech scene.**
Every environment-aware TTS takes text as mandatory input -- UmbraTTS, VoiceDiT, IDEA-TTS,
I2TTS, ImmersiveTTS, Audio-Omni, AudioLDM2-TTS, and VoiceLDM itself. That is this project's
unclaimed problem.

Approaches: retrieval against training transcripts (we have a baseline); scene caption -> LLM
-> plausible utterance; a learned scene->text decoder; or implicit prediction of a content
*representation* that never materializes as words.

**Evaluation is the hard part** and needs designing before results exist: verbatim WER is
wrong for an inherently one-to-many task. Candidates: R-precision / self-retrieval (does the
generated speech match its OWN scene better than a random one?), distributional metrics,
LLM-as-judge, human protocols from dialogue and story generation.

### D7. Discrete content units + autoregressive sampling -- ACTIVE, the current main line

**The diagnosis this direction answers.** Four content sources have been tried. Retrieval and
oracle work but cheat (copy a training utterance / read the target). The two honest ones --
regression (2026-08-31) and end-to-end diffusion-loss training (jobs 434308, 434417) --
failed identically, and the reason is mathematical rather than architectural:

    argmin_f E[(f(x) - y)^2]  =  f(x) = E[y | x]

A deterministic map trained against specific targets converges on the conditional MEAN. Three
things about this were initially mis-stated and are worth keeping straight:

  - It is **not** caused by one-to-many data. The dataset is a strict bijection: one ambience,
    one utterance. The averaging happens across **neighbours** -- two kitchen recordings sit
    close in CLAP space with unrelated transcripts, and a smooth function cannot emit wildly
    different content for near-identical inputs, so the competing pulls balance at the mean.
  - The decisive version: `bind_gap` is measured on **validation**. Even perfect memorisation
    of all 17,596 training pairs leaves an unseen clip with no memorised answer, and the only
    thing a deterministic map can do there is interpolate between known neighbours -- which is
    averaging by definition. That is why extra capacity never helped.
  - The diffusion loss did not rescue it because **the stochasticity is in the wrong place**.
    Diffusion's randomness lives in the sampler's noise, varying the *realisation* of a given
    conditioning. The thinker sits outside that loop feeding one fixed vector, trained by
    gradient descent against specific targets, and collapses exactly like the MSE predictor.

And in content space that mean is **off-manifold**: the average of the encodings of "dinner's
ready" and "where are my keys" is not a blurry utterance, it is not an utterance. This also
plausibly explains job 434417's flat-zero result -- a renderer frozen on real encodings was
handed input outside everything it had ever seen, and returned almost no gradient.

**The mechanism.** Quantize content to K real frames learned from actual training speech
(`src/salm/a2s/content_codebook.py`). Content becomes 256 discrete indices, and both halves of
the failure are addressed at once:

  1. Every possible output is a real content frame some real utterance produced -- on-manifold
     by construction, so the renderer never sees input outside its training distribution.
  2. The predictor's objective becomes cross-entropy over categories, whose optimum is the
     true conditional **distribution** rather than its average. The mean of a distribution
     over categories is not a category, so the average cannot be emitted even in principle.
  3. At inference you **sample** -- one plausible utterance, committed to, instead of blur.

**Must be autoregressive.** Sampling all 256 positions independently from their marginals
gives phoneme soup: frame 5 committing to one utterance, frame 6 to another. `P(code_t |
code_<t, scene)` is what keeps a sampled utterance internally coherent, for the same reason
language models are autoregressive rather than predicting all tokens at once.

**Precedent for exactly this failure and exactly this fix:** lip-to-speech went from 65-78%
WER with mel regression to 50.2 / 44.8% with discrete units (Choi et al., arXiv 2305.19603).
Same collapse, same remedy. Related: the semantic-token cascade of D5, of which this is the
concrete, gated version.

**Status: gated on purpose.** Job 436244 fits codebooks at K = 256/512/1024/2048 and then
feeds **quantized oracle** content through the proven `content_concat_v4` renderer, scoring
the audio. No predictor is written until quantization is shown not to destroy intelligibility
-- reconstruction cosine is explicitly not accepted as the test, since this project has
repeatedly seen a metric move while the audio did not.

**The instrument the predictor stage will need**, designed before results exist: perplexity
with the ambient input **ablated** (zeroed or shuffled across the batch). If conditioned and
ablated perplexity are equal, the model has learned generic English phoneme statistics and is
ignoring the scene -- which would look like healthy training loss while being a total failure.
This is the `bind_gap` analogue for the discrete stage, and it is cheap to compute per epoch.

**What this direction does NOT fix.** The residual roughness of the oracle-content generations
is a separate problem -- VAE round-trip plus vocoder plus data scale, already measured at
0.389 against a 0.372 ceiling, so there is almost no headroom there to win. Deliberately
deprioritized (user call, 2026-09-02).

---

---

### D8. Deterministic decoding -- ACTIVE, the current best result

Greedy (`top_k=1`) over the discrete content codes, replacing temperature sampling. Free: one
flag, no retraining.

**Why it was not tried for weeks.** `ContentLM.sample` carried "Sampling, never argmax: argmax
walks straight back into the conditional mean that the discrete formulation exists to escape."
That is an argument about CONTINUOUS targets -- where `argmin E[(f(x)-y)^2] = E[y|x]` and the
mean of plausible mel trajectories is off-manifold -- imported into discrete token space, where
argmax returns the most likely SEQUENCE, a real utterance rather than an average. Written up in
LEARNINGS as the general lesson about inherited constraints.

**RESULT (jobs 443379, 443475).** With `checkpoint_best_loss` (epoch 9) and K=2048, over 128
val clips: `inter_asr_dist` 0.3688 against real speech 0.3146 and the VAE round-trip ceiling
0.3722. 48% of clips at or below that ceiling, 35% at or below real speech, 7% clearly
degraded, none near the noise floor. Previous project best was 0.452. Listening (Lovenya):
"the best loss ones... they are super better".

**The cost, found by reading the transcripts rather than the metrics.** Four stock openings
cover 46% of 128 clips ("hey can you hear ..." alone is 39). The perfect-agreement clips are
disproportionately the ones where the model reached for a safe phrase -- four different engine
scenes all produced variants of "hold on tight, we're picking up speed". Failure correlates
with utterance length (r = +0.37): it breaks down when it attempts something beyond its stock
inventory. `cross_clip_diversity` scored 0.931 and missed this entirely, because it averages
over all pairs and most pairs are unrelated scenes -- the collapse is per scene FAMILY, and the
metric is too coarse to see it. A per-scene-family diversity measure is the open gap.

**Also required with it:** `checkpoint_best_loss`, not `checkpoint_best`. See LEARNINGS on
selection metrics -- `best` is chosen on `val_scene_gap`, which rises with overfitting.

### D9. The content LM is ~97% a LANGUAGE model -- and that is the bottleneck

*Measured 2026-09-06, from the dedup runs' own history.json*

| run | val_code_loss | scene_gap | the scene's share |
|---|---|---|---|
| dedup_k512 | 1.5002 | +0.0414 | **2.8%** |
| dedup_k1024 | 1.6213 | +0.0370 | **2.3%** |
| dedup_k2048 | 1.7588 | +0.0322 | **1.8%** |

`scene_gap` is the entire contribution of knowing WHICH scene this is. Everything else -- 97%
and up -- is language modelling over code sequences, learned from 17,596 utterances.

**Four independent measurements now agree, and they were not designed to:**
- exact-word recovery f1 0.16 even from a ground-truth caption (job 442799)
- the plan head beat a scene-blind marginal by 0.0040 nats (443751)
- cross-seed agreement collapses after ~4 words (442814)
- the scene accounts for under 3% of the LM's loss (above)

**The reframe.** The scene constrains the OPENING and the TOPIC. It was never going to
constrain word 7, and no conditioning machinery changes that -- which is exactly why CFG
(442847), the plan vector (443751) and guidance all landed where they did. Those were not
failures of implementation; they were attempts to extract information that is not present.

So the coherence complaint -- "3-4 words make sense, then another group starts" -- is a
LANGUAGE-MODELLING failure, not a conditioning failure. After word 4 the model should be
producing fluent speech constrained only by topic. It cannot, because it is a weak LM over
2048 acoustic codes trained on 17k utterances.

**What follows, and the constraint that shapes it.** The obvious move is to pretrain the
content LM on unpaired speech -- any audio converts to codes through the existing CTC probe and
codebook, no ambient needed. **We do not have unpaired speech.** `datasets/` holds only TTAS
(22,793 clips, of which 17,596 are already train). Acquiring a corpus is a download decision,
not an engineering one.

That is the strongest argument for D5: CosyVoice3's `llm.pt` is a pretrained language model
over its own 25 Hz token space, already on scratch. It supplies exactly the prior this
measurement says we lack, without needing a speech corpus we do not have. Gate first
(tokenize -> resynthesize through frozen flow+hift -> ASR, ~15 min) before committing.
BLOCKED on `uv pip install onnxruntime` -- the tokenizer is ONNX and the runtime is absent.

## Considered and rejected

- **BEATs as a content encoder.** Its stated design goal is discarding fine detail for
  high-level semantics -- the opposite of what content needs. Same family as CLAP. Only
  legitimate use is as an alternative *scene* encoder, decided by the existing ridge probe.
- **Audio-Omni as backbone.** Transcript is a mandatory input to its speech path, so it does
  not solve F1. *Correction to the 2026-08-26 assessment:* the repo does now ship
  inference/model code and links released weights -- but exhaustive search finds no optimizer,
  dataset, loss, or training entry point. Adapting its generator means implementing and
  validating a rectified-flow fine-tuning recipe ourselves, not invoking an official trainer.
  Revisit only after a released-checkpoint round-trip gate.
- **Switching wholesale to AudioLDM 2.** Validates full-rate sequence conditioning for TTS
  (λ=1 for speech vs λ=8 for general audio) but ships separate speech/general checkpoints and
  has no official full training recipe -- so it is not the shared bidirectional base we want.
- **Changing the mel/VAE/vocoder (64-bin, 16 kHz).** Refuted twice: our own gate measured
  WER 6.27% -> 5.64% through the full VAE+vocoder path on real speech, and VoiceLDM reaches
  3.9% WER on these exact frozen components. Do not spend a job here.
- **More conditioner tuning on the pooled interface.** Settled by the oracle test and
  confirmed by AudioLDM 2's own λ ablation.
- **Flow matching as a stack rewrite.** Its documented advantage is *sampling*-step
  efficiency and training stability; claims about *data* efficiency are asserted, not
  measured, and no controlled same-data comparison in audio was found. Not worth discarding a
  pretrained U-Net for. Better acquired *for free* inside CosyVoice's `flow.pt` or F5's DiT
  via D5/D4.
- **The oracle-transcript TTS cascade (Rank 1).** Dropped as redundant with SpeechAudioLM's
  existing cascaded ASR+TTS baseline. *Nuance worth preserving:* dropping it as a **ceiling
  measurement** was right; the two-stage semantic->acoustic **architecture** it resembles is
  not rejected -- that is D5.

---

## Open questions worth an opinion, not yet answered

- **Is our failure the injection mechanism, the content representation, or the data scale?**
  D1 tests the first. If concatenation binds (it does) but audio stays unintelligible, that
  points past the interface toward D3 and then D4/D5. The steelman for scale is strong: no
  interface fix closes a 100-300x gap.
- **"World models" / JEPA.** Raised as a direction. Current assessment: JEPA is a
  representation-learning objective, not a waveform generator, and there is no justified
  world-model framing for static (scene, speech) pairs with no temporal interaction. It could
  serve as an *auxiliary shared-memory objective* across S2A/A2S, which is a real but much
  narrower claim than "world model". Needs a concrete formulation before it can be planned.
- **Actionable architecture ladder from the research pass:** UNISON first, UniFlow-Audio
  second, Audio-Omni only after a released-checkpoint inference/round-trip gate.
- **The bidirectional endgame.** Merged weights, joint training, or a one-hot direction
  conditioner -- deferred since 2026-08-23 and still unaddressed. Note that D4/D5 both cost
  the shared-backbone story, so this question and those directions interact.
