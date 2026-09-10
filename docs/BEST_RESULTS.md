# Best A2S results — the deliverable shortlist

**Purpose.** A durable record of the generations Lovenya has rated best by ear, with the exact
artifacts needed to reproduce each one. If the coherence problem is not fully solved, the
deliverable is chosen from this list.

**Why it exists.** On 2026-09-06 a working result was closed on the wrong metric and its
checkpoints were deleted on the agent's recommendation; recovering it required retraining. The
files listed under PROTECTED below must not be deleted without checking this page first.

*Ranked by Lovenya's listening, with `inter_asr_dist` alongside. The ear ranks; the number
places the lane on a scale. Anchors: real speech **0.3146**, VAE round-trip ceiling **0.3722**,
zero-content floor **0.967**. Lower is better.*

---

## #1 — Coarse-to-fine content LM (stride 12), epoch 10

**`inter_asr_dist` 0.2970** · real_word_rate 0.982 · 64 clips
**Lovenya, 2026-09-07:** *"one of the best speech generations I have heard until now"*,
and earlier on the same lane: *"stride12 was pretty nice... the least unintelligible out of the
lot, that I've been hearing across our so many runs"*

| component | path (under `salm_model_outputs/`) |
|---|---|
| content LM | `content_lm/hier_k2048_s12/checkpoints/checkpoint_best_loss.pt` (epoch 10) |
| renderer | `runs/a2s_audioldm_clapseq_xformer/rladder_20260906_1654/checkpoints/checkpoint_ep119.pt` |
| codebook | `content_codebook/codebook_k2048.pt` |
| CTC probe | `ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt` |
| decoding | **greedy** (`--content-lm-top-k 1`), temperature 1.0, seed 1337 |
| audio | `generations/hierlate_20260907_1602/best_loss/` and `generations/hier_20260907_1223/s12__greedy/` |

```bash
python -m salm.generate --mode a2s --direction a2s \
  --checkpoint $RUNS_ROOT/a2s_audioldm_clapseq_xformer/rladder_20260906_1654/checkpoints/checkpoint_ep119.pt \
  --content-source codebook_lm \
  --content-lm-checkpoint $OUTPUTS_ROOT/content_lm/hier_k2048_s12/checkpoints/checkpoint_best_loss.pt \
  --content-lm-temperature 1.0 --content-lm-top-k 1 \
  --ctc-checkpoint $OUTPUTS_ROOT/ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt \
  --split val --num-samples 64 --start-index 0 --seed 1337 --out-dir OUT --copy-reference
```

**Why it wins: UNEXPLAINED.** The hierarchy is not the reason — the ordered stride prediction
job 444750 was built to test failed outright (horizons flat at 1 across strides 6/8/12).

The obvious candidate was length: this model produces 11.1 words against the flat model's 17.6,
where the human reference is 13.3, and only 11% of its utterances run past 1.5x their reference
against the flat model's 42%.

**That candidate is now in doubt.** Job 448455 clamped the flat model's decode budget to 111 and
90 frames, bringing its word count to 13.7 and 13.1 — essentially the human rate — and the audio
got WORSE, not better (0.3404 → 0.4216 and 0.4155, all CIs excluding zero, and worse still after
length adjustment). If shortness were the mechanism, that should have moved toward 0.2949.

But that job tested FORCEFUL TRUNCATION, not learned stopping: it cuts the model off mid-utterance
whether or not it had finished, so the tail is severed rather than completed. It therefore does
not settle the question. **The property that matters may be "coherent completion" rather than
"length", and a decode-time cap cannot produce it.** The experiment that would separate the two —
training with the targets trimmed at the speech boundary, so the model learns to emit EOS there
itself (`--truncate-speech`) — has not been run.

Until it has, treat the reason this lane wins as OPEN.

---

## #2 — Flat run-length content LM, epoch 9

**`inter_asr_dist` 0.3161** · 64 clips · the "A" lane in the 2026-09-07 A/B

| component | path |
|---|---|
| content LM | `content_lm/dedup_k2048/checkpoints/checkpoint_best_loss.pt` (epoch 9) |
| renderer | same as #1 (`rladder ep119`) |
| decoding | greedy, temperature 1.0, seed 1337 |
| audio | `generations/rladder_20260906_1654/new_ep119/` |

Identical to #1 except the content model. Useful as the reference point every other result is
measured against.

---

## #3 — Flat LM through the first fine-tuned renderer

**`inter_asr_dist` 0.3223** · **Lovenya, 2026-09-06:** *"it is the best speech I have heard
until now in our project"* (at that time)

Renderer `runs/a2s_audioldm_clapseq_xformer/rquant_continuous_20260906_1425/checkpoints/checkpoint_best.pt`
(epoch 14), same LM and decoding as #2. Superseded by #2's renderer, kept because it was rated
best at the time and the 2.5 GB checkpoint still exists.

---

## #4 — Flat LM through the original v4 renderer

**`inter_asr_dist` 0.3688** over 128 clips · **Lovenya, 2026-09-06:** *"super better"*

Renderer `content_concat_v4/checkpoints/checkpoint_best.pt`, same LM and decoding.
The first result that crossed into recognisable speech; the 128-clip number makes it the most
statistically solid figure on this page.

---

## PROTECTED — do not delete without checking this file

```
content_lm/hier_k2048_s12/checkpoints/checkpoint_best_loss.pt                     111 MB   #1
content_lm/dedup_k2048/checkpoints/checkpoint_best_loss.pt                        111 MB   #2,#3,#4
runs/.../rladder_20260906_1654/checkpoints/checkpoint_ep119.pt                    414 MB   #1,#2
runs/.../rquant_continuous_20260906_1425/checkpoints/checkpoint_best.pt          2482 MB   #3
runs/.../content_concat_v4/checkpoints/checkpoint_best.pt                        2482 MB   #4
content_codebook/codebook_k2048.pt                                                  4 MB   all
ctc_probe/ctc_probe_v1/checkpoints/checkpoint_best.pt                              27 MB   all
```

Total 5.6 GB. All verified present 2026-09-07.

---

## Reading the numbers on this page

`inter_asr_dist` is the edit distance between Whisper and a CTC probe on the same audio — it
measures whether two independent recognisers hear the same words, not whether those words are
correct. It has **disagreed with the ear twice** (two lanes 0.27 apart were heard as "90/95%
same"), it rises about **+0.009 per generated word** so a model that says less scores better
without being better, and the run-to-run floor from diffusion noise alone is **~0.02–0.03**.
Differences below that are not evidence. The ranking here is Lovenya's; the numbers are context.
