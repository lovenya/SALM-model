# Architecture and recipe menu

Options for what to try next, organised so that **every added component is licensed by a measured
gap against something simpler** — because the reviewer question is "why did you introduce
complexity, and did you show the simple thing fails?"

Each tier is ordered by how much uncertainty it removes per GPU-hour, not by how interesting it is.
Findings marked *verified* were checked against a primary source; everything else is flagged.

Companion docs: `EXPERIMENT_LOG.md` (what already happened, with numbers), `OPEN_ITEMS.md` (what is
waiting on a decision).

---

## Tier −1 · Diagnostics — is the condition used at all?

**Status: running (job 406885).** Everything below is easier to choose once this lands.

| id | test | what zero means |
|---|---|---|
| **D1** | shuffle the conditioning within a batch (a true derangement) | the condition carries no **sample-specific** information |
| **D2** | `‖ε(c) − ε(null)‖ / ‖ε(c)‖` | the condition has **no influence at all**; CFG is also a no-op |

`D2 = 0` implies `D1 = 0`, but not the reverse: the model can exploit the conditioning
*distribution* generically without using the specific utterance. Both are evaluation-time, so they
run on existing checkpoints. Run on **both** the earliest and the last snapshot — a D1 that grows
across training would be the first genuine evidence of learning we have.

**Why this gates everything.** ε-MSE cannot answer it. Measured: the loss tracks the noise
schedule's signal:noise almost perfectly and inversely (0.559 at t≈125 → 0.001 at t≈875), train and
val both sat at ~0.198 ± 0.007 with no trend over 91 epochs, and "best val" matched the expected
minimum of 91 noise draws. A flat curve is equally consistent with learning and with ignoring the
speech entirely.

---

## Tier 0 · Baselines with no training

Cheap, and what a reviewer checks first.

| id | baseline | tests | status |
|---|---|---|---|
| B0 | **unconditional** — sample from the learned null embedding | what "zero information" scores; any model must beat it | not built |
| B1 | **retrieval by speech** — nearest training utterance → its paired ambient clip, verbatim | whether *generation* is needed at all | **running (406885)**, both Whisper and CLAP spaces |
| B2 | **retrieval by transcript** — ASR → nearest caption → its clip | whether *audio* input is needed or text suffices | not built |
| B3 | **caption oracle (rung0)** — condition on ground-truth `audio_cap` | the ceiling | done |
| B5 | **majority-scene prior** — always generate the most frequent caption | dataset bias | not built |

**Expect retrieval to win FAD, and plan for it.** Retrieval emits *real* recordings, so it sits on
the real manifold by construction while FAD only measures distributional distance. It can score
beautifully and be wrong on every example. Report three things together: FAD (retrieval may win), a
**per-example** metric such as CLAP cosine to the ground-truth paired audio (retrieval should lose),
and a novelty check that our outputs are not nearest-neighbour copies. That combination is a
*result*, not an embarrassment. Only one published paper reports such a baseline at all ("Sounding
that Object", arXiv 2506.04214, where retrieval loses on everything), so there is no citation to
hide behind. *Verified.*

Also: **FAD's encoder choice dominates its correlation with human judgement** — DCASE 2024 found
PANNs FAD at ρ = −0.94 against human ratings, above CLAP and VGGish. Choose it deliberately.

**Not the 4-stage cascade.** User decision, 2026-08-12: do not evaluate the SpeechAudioLM cascade
(transcriber → captioner → Llama → generator) as a quality bar. The claim is that this is the only
end-to-end model for the task and that it wins on latency, memory and compute footprint. An earlier
draft treated beating the cascade as existential; that was wrong.

**Still open:** the *minimal two-stage* pipeline (ASR → transcript → off-the-shelf text-to-audio) is
a different thing, and the literature is emphatic it is the row reviewers demand, because it decides
whether end-to-end is needed. V2A-Mapper's analogue loses badly (FD 56.40 vs 24.17). See
`OPEN_ITEMS.md`.

### Efficiency as a first-class result

Because the claim is end-to-end + efficiency, these are primary numbers, not afterthoughts:
parameter count, peak VRAM for one inference, wall-clock latency per sample, number of models and
stages to orchestrate. Cheap to measure, hard to argue with, and they hold even if quality ties.

---

## Tier 1 · The ladder we lack

All share the precomputed latents, so each rung is cheap. **The gap between rungs is the point.**

| id | conditioner | params | U-Net |
|---|---|---|---|
| L0f | masked mean-pool + one linear | 0.66 M | **frozen** ← the true minimum |
| L0 | masked mean-pool + one linear | 0.66 M | finetuned |
| L1 | attention-pool + MLP | 7.1 M | finetuned |
| L2 | transformer + attention-pool *(what we ran)* | 32.3 M | finetuned |
| L3 | k learned queries (Q-Former-ish) | 18.6 M | finetuned |

**Comparability rule:** only compare runs sharing `epochs` **and** `lr_total_steps`
(`configs/exp/long100.yaml` pins 23,400). A derived LR curve makes a 5-epoch run fully annealed at
epoch 5 (factor 0.000) while a 100-epoch run is at 0.997 — not comparable at any epoch past warmup.

**L0/L1 are no longer optional.** V2A-Mapper's own table shows their **2.10 M** 1-layer MLP scoring
FD 35.06 *and the best CLIP-score in its family* (9.927, beating their 48.8 M final model's 9.720).
So a near-trivial mapper already works and complexity bought ~30% FD, not a qualitative change. Any
reviewer who reads that table will demand our equivalent row. At 15k pairs L2 carries ~2,150
params/example, so L0 matching L2 is a live possibility. *Verified.*

---

## Tier 2 · Change the objective, not the architecture

**My top recommendation**, and the cheapest thing with real upside.

The conditioner currently receives *only* a sparse gradient through the diffusion loss: one random
timestep per sample, averaged over a difficulty gradient that dominates the number. At 15k pairs that
may simply be too weak to learn "speech meaning → scene".

| id | recipe | note |
|---|---|---|
| **O2** | `L = L_DDPM + λ · MSE` to the frozen CLAP audio embedding, `w(n)=10^(−n/200)`, **λ = 2000** | published recipe, keeps AudioLDM's objective |
| O1 | two-stage: regress the target's CLAP embedding, then a **fully frozen** AudioLDM | minutes of compute; makes CLAP cosine a *valid* metric again |
| O1b | **diffusion mapper in CLAP space** (V2A-Mapper's own best variant) | samples rather than averaging — correct for a one-to-many task |
| O3 | contrastive pretrain (speech ↔ target CLAP), then diffusion finetune | Diff-Foley needed exactly this |

**O2 has a near-exact precedent.** Latent CLAP Loss (arXiv 2403.12182, EUSIPCO 2024) on almost our
setup (AudioLDM base, CLAP/VAE/vocoder frozen, LDM trainable): **FAD 8.74 → 7.10**, ≈19% relative;
human quality 6.00→6.40 and class relevance 6.55→7.03, p < .001. **λ = 2000 is the actionable
detail** — the auxiliary term needs an enormous weight relative to the diffusion loss, and a naive
λ≈1 would do nothing. MSE, not cosine. `w(n)` down-weights high-noise timesteps, matching our own
bucket finding that signal lives at low `t`. *Verified.*

**O1b is worth serious thought.** S2A is explicitly one-to-many — many scenes are valid for one
utterance — and deterministic MSE regression converges to the *mean* of them, which is the wrong
target. A diffusion mapper samples instead. V2A-Mapper's diffusion variant beat regression at every
size (FD 24.17 vs 29.4).

**The citation for "the diffusion loss is a weak semantic signal":** REPA (arXiv 2410.06940, ICLR'25
oral) gets **>17.5× faster convergence** by aligning intermediate hidden states to frozen features.
Transferred to audio in HunyuanVideo-Foley (2508.16930), best in *shallower* layers. "REPA Works
Until It Doesn't" (2505.16792) argues for early-stopping the term, i.e. a decaying λ.

**The counterexample that defines our regime:** Foley Control (2510.21581) trains a conditioner on
the diffusion loss alone with no auxiliary term — on **~700k clips**. Diff-Foley needed contrastive
pretraining. V2A-Mapper bypasses the diffusion loss for the conditioner entirely. **Nobody relies on
the diffusion loss alone at 15k pairs.** That is a defensible motivation paragraph.

---

## Tier 3 · The conditioning interface

AudioLDM 1 exposes exactly one global 512-d vector (`class_embed_type=simple_projection`, verified
from the checkpoint config), projected and added to the timestep embedding — so it modulates every
block globally rather than attending to positions.

**Correction to an earlier claim of mine.** I argued 512 floats is too few. A direct ablation
(arXiv 2501.14680, text-to-music) says otherwise:

```
local only (T5 sequence, cross-attn)            FAD 2.02
global + local (pooled + sequence)              FAD 1.94
mean-pooled global via FiLM, no extra params    FAD 1.89   ← best
```

Pooled and sequence conditioning are **complementary**: pooled steers globally, cross-attention
supplies detail. Our real limitation is **not** the 512 floats but that we have *only* the pooled
path and **no cross-attention path at all**. The evidence-backed move is to *add* a sequence path
while *keeping* pooling.

Corroborating: **VoiceLDM** conditions on speech content via a pooled encoding and its documented
failure is misalignment producing repetitive output, blamed on lost temporal structure — our risk in
miniature.

| id | option | note |
|---|---|---|
| C1 | **AudioLDM 2** — cross-attention over a sequence (AudioMAE "LOA" 512×768 per 10 s, plus a separate T5 cross-attention; a GPT-2 translator produces the LOA sequence) | the principled fix. `haoheliu/AudioLDM-training-finetuning` supports finetuning both v1 and v2 |
| C2 | enable cross-attention in AudioLDM 1's U-Net (its config carries `cross_attention_dim`) | cheaper but off the pretrained path; unverified that it trains at all |
| C3 | Audio-Omni's DiT + rectified flow | proven MLLM→generator pattern; abandons AudioLDM |

---

## Tier 4 · The encoder

| id | encoder | note |
|---|---|---|
| E1 | Whisper-large-v3 encoder *(current)* | linguistic content; local |
| E2 | **frozen Qwen2.5-Omni thinker**, layer **−2** | reasoning, not transcription. Weights local |
| E3 | hybrid: Whisper features + explicit transcript embedding | cheap; tests whether paralinguistics add anything over text |
| E4 | CLAP audio tower | control, expected to fail — **and now being measured** by the CLAP-space retrieval lane |

**Take layer −2, not the final layer.** Audio-Omni Table 7: penultimate vs final → T2A IS **11.26 vs
9.36**, FD **2.75 vs 4.21**. The last layer is over-specialised for next-token prediction. *Verified.*

**Frozen MLLM features beat simpler encoders** — Audio-Omni Table A1: frozen Qwen2.5-Omni vs T5 vs
CLAP on T2A → KL **1.25** vs 1.51 / 1.81. No Whisper baseline, so not a literal Whisper-vs-Qwen test.

**But elaborate adapters lost to raw hidden states** in the same ablation: MetaQuery and a generic
learned-query mechanism scored *worse* than simply taking penultimate hidden states (IS 7.44 / 8.55
vs **11.26**). Independently supports starting simple; a BLIP-2 Q-Former is 188 M params.

**Zero-init is about training dynamics, not capacity** (ControlNet): a zeroed branch outputs exactly
0 at step 0, so the pretrained backbone is preserved and conditioning fades in instead of injecting
random gradient noise. Worth using for any adapter bolted onto the frozen U-Net.

Two cautions: **Pengi's ablation** found freezing its audio encoder measurably *hurt* downstream
performance versus finetuning; and **LTU-AS attends over multiple Whisper layers**, whereas we use
only the final one — a cheap untried improvement.

**Pair Tier 4 with Tier 3.** A frozen MLLM's rich sequence gets crushed by a single pooled vector, so
spending on the encoder before fixing the interface wastes most of the benefit.

---

## Tier 5 · The generator family

AudioLDM-S (current, 185 M) · AudioLDM-L · **AudioLDM 2** · Audio-Omni DiT (local) · TangoFlux
(local) · AudioGen — **deprioritised**.

### AudioGen is a worse bet than I first proposed

- **AudioCraft explicitly does not support finetuning with changed layers** — and replacing T5 text
  conditioning with speech features *is* that. No dataset tooling either; you build your own
  `AudioDataset` with per-clip JSON sidecars on top of Dora + Flashy.
- **Our data scale is the risky case.** Narrow style adaptation works on ~1 h. Learning a *new
  conditioning pathway* on ~42 h is where community reports break: 3 h single-instrument was fine,
  a broader dataset gave **crackling in >80% of generations with `grad_norm=INF`** (audiocraft#447).
- **Practitioners at our scale chose diffusion.** DCASE 2023 Task 7 (~5.4 h): nearly all competitive
  entries were diffusion + transfer learning, explicitly for data scarcity.
- **Diffusion wins the benchmark** (Tango's table, AudioCaps): AudioGen FAD 3.13 / KL 2.09 vs
  AudioLDM-L 2.08 / 1.86 vs Tango 1.59 / 1.37. Caveat: cross-paper variance is real (AudioGen-Large
  reported elsewhere at 1.82), so treat the gap as 0.2–1.5 FAD, config-dependent.
- The released `audiogen-medium` is a **reimplementation**, not the paper model. And the delay
  pattern is **MusicGen's** contribution, not AudioGen's — I attributed it wrongly at first.

**If AR is ever revisited:** FoleyGen's trick (arXiv 2309.10537) *prepends* conditioning to the token
sequence and lets causal self-attention consume it — no new cross-attention weights, which sidesteps
AudioCraft's restriction. FAD 1.65 vs Im2Wav's 6.41, so prepending is not a bottleneck. Tokenizer:
**DAC** is the defensible choice for environmental audio (universal codec, SOTA reconstruction);
WavTokenizer trades fidelity for short sequences (PESQ 2.17 vs DAC 3.69).

---

## The differentiating argument, independent of any metric win

**What can end-to-end do that a text cascade structurally cannot?** A transcript discards
paralinguistics — emotion, urgency, prosody. Our manifests carry `speaker_emotion` and
`speaker_gender`. So: does the model's output vary with emotion **at matched transcript**?

If yes, that is a real contribution *even if FAD ties*, and it directly motivates the architecture.
It reframes the paper from "ours is better" (risky) to "text is a lossy bottleneck for this task, and
here is the measurement" (defensible).

## Novelty, phrased so it survives review

Do **not** claim "nobody generates ambience from language" — text-to-audio does, and
narrative→soundscape pipelines exist (Sound of Story 2310.19264, ReelWave 2503.07217,
BackgroundMellow 2607.11364). Claim the two things that are unclaimed:

1. **conditioning on speech audio directly**, with no ASR-to-text bottleneck;
2. **conditioning on an *implied* scene rather than a *described* one** — "you almost hit me, slow
   down" contains no sound-event noun, so the model must *infer*. Every prior system consumes a
   description of the sound.

Supporting: **VoiceLDM** takes a transcript *plus a separate user-supplied* environment description,
explicitly not inferred from content. **Sound of Story** deliberately *strips* speech. **Soundscape
Stylization** takes ambience from an example clip. Each routed around this inference step.

Confidence ~85%; residual risk is a 2025–26 workshop/Interspeech paper under idiosyncratic phrasing.
Also unclaimed: no work conditions an AR audio LM on speech embeddings for scene audio.

**One claim of mine that does NOT hold up:** that CLAP encodes "someone is speaking" rather than what
was said. There is **no controlled probing study** supporting it; the argument is structural (CLAP
trains on captions that tag speech rather than transcribe it). Cite it as an inferred limitation —
or better, substantiate it ourselves with the CLAP-space retrieval and rung1 control.
