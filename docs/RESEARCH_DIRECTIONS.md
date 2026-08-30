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

### D2. Content classifier-free guidance -- NOT YET DONE, cheap, high expected value

We have never generated A2S audio with content guidance. The content channel had no null
branch and no separate scale, while the pooled channel the oracle test discredited is
amplified 3x. VoiceLDM uses **dual CFG** with independent `w_desc` and `w_cont`, and reports
that raising `w_cont` is precisely what buys intelligibility (at the cost of description
adherence). InstructPix2Pix is the same two-condition shape and solves it identically.

`content_dropout_prob=0.1` now creates the null during training; the **sampler still needs the
dual-CFG term** and a `w_cont` sweep. Also unswept: plain `guidance_scale` for A2S at all --
measured `val_d2_gap` of 0.0086-0.018 means CFG at 3.0 moves the trajectory only ~3.6% off
unconditional. S2A swept this and shipped at 7.5.

### D3. A better content representation

Our content comes from a CTC-trained BiGRU. CTC representations are notoriously **peaky** --
most frames confidently blank, information in sparse spikes -- which is a poor match for a
per-frame conditioning signal. Options:

- **A temporal pre-net** (dilated Conv1d / ConvNeXt-1D over the 256x512 sequence) to smear
  spikes into phone-duration plateaus. F5's ConvNeXt refinement is the precedent, though
  their motivation differs (they are fixing *mis*-alignment; ours is already correct).
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

---

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
