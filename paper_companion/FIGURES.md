# Model figures — 2026-10-05

The website model page is a first architecture draft describing the implemented
source-only generation paths. It does not change models, experiments or results.
Figures are original code-generated vector diagrams; AudioLDM Figure 1's
separation of training and sampling informed the presentation, not the model
content. Primary reference: https://proceedings.mlr.press/v202/liu23f/liu23f.pdf.

`scripts/website/build_model_figures.mjs` writes `figures/generation`, `training`
and `content-lm` in SVG, PDF and 2× PNG formats. SVGs retain text, shapes and
named component groups. PDFs are 174 mm wide and have embedded fonts. Each
artifact's bytes and SHA-256 are in `figures/manifest.json`. Raster previews
are inspected separately from the DOM checks. No model inference is involved.

## Evidence checked

| Diagram claim | Source |
| --- | --- |
| Global conditioner: two Transformer layers, attention pooling, 512-dimensional output | `src/salm/conditioners.py`, `XFormerConditioner`; saved S2A/A2S run configs |
| AudioLDM global condition is class conditioning; selected A2S uses channel concat | `src/salm/train_ldm.py`, `ddpm_loss`; `src/salm/generate.py`; selected renderer config |
| 512→128 full projection and eight content channels, concatenated with eight latent channels | `src/salm/a2s/content_concat.py`, `ContentProjector`, `concat_content`; renderer config |
| Frozen CTC features: 256×512 before the character head | `src/salm/a2s/ctc_probe.py`, `SimpleCTCASR.encode`; `scripts/a2s/train_ctc_probe.py` |
| Single k-means centroid codebook, K=2048 | `src/salm/a2s/content_codebook.py`; `scripts/a2s/build_content_codebook.py`; retained codebook metadata |
| Run-length durations bounded to 32; expansion holds the last code to fill the canvas | `src/salm/a2s/content_dedup.py`; `src/salm/generate.py`, codebook LM path |
| Six-layer causal decoder, width 512, eight heads, summed code/duration/position embeddings, two heads | `src/salm/a2s/content_lm_dedup.py`; `src/salm/a2s/content_lm_hier.py`; current run histories and trainer defaults |
| Hierarchy is a coarse prefix in one LM, stride 12, followed by SEP and fine pairs | `content_lm_hier.py`, `coarse_of`, `build_hier_batch`, `sample`; current hierarchical run history |
| LM training uses code CE and fine-duration CE, independently of renderer optimization | `scripts/a2s/train_content_lm_dedup.py`; `scripts/a2s/train_content_lm_hier.py` |
| S2A cosine alignment target is the target audio embedding, not transcript text | `scripts/precompute_clap_targets.py`; `src/salm/train_ldm.py`; saved S2A config |
| A2S alignment target is paired transcript text; renderer sees continuous oracle content | `scripts/a2s/precompute_a2s_conditioning.py`; `train_ldm.py`; current A2S renderer config |

Saved current artifacts inspected, without deserializing checkpoint tensors:

- `salm_model_outputs/runs/s2a_audioldm_whisper_xformer/long100_L2_o2_rorqual_b64_2gpu/config.json`
- `salm_model_outputs/runs/a2s_audioldm_clapseq_xformer/content_concat_v4_rorqual/config.json`
- `salm_model_outputs/content_lm/dedup_k2048_rorqual/history.json`
- `salm_model_outputs/content_lm/hier_k2048_s12_rorqual/history.json`
- `salm_model_outputs/user_survey_bundle_20260917/provenance.json` and its A2S generation manifests

These use the destination v4 renderer lineage. Older rladder experiments are
not described as a required second renderer-training phase. The diagrams
describe architecture, not new evaluations or an explanation for hierarchical
performance differences.

## Deliberate simplifications

- Figure 1 separates the two direction-specific models; it does not imply shared
  weights or a jointly trained bidirectional network. Sampling steps are folded
  into the U-Net block. Classifier-free guidance and preprocessing are omitted.
- Figure 2 shows the two renderer objectives. Frozen VAE encoding includes the
  mel frontend; decoding and vocoding appear in Figure 1. Target embeddings
  provide a loss target rather than a replacement inference condition.
- Figure 3 shows the LM's common structure and sequence formats. LayerNorm,
  residual connections and dropout are summarized rather than drawn in detail.
  SEP exists only in the hierarchical model. The renderer is marked frozen in
  this figure because it is fixed during LM fitting.
- A2S renderer training uses continuous target-speech features; inference uses
  centroid features predicted from the source. This handoff is stated explicitly
  in the page. The CTC character head is bypassed for feature extraction.
- Main SVG text is designed for full-page-width figures. On narrow screens the
  figure frame scrolls horizontally; the image also links to the full-size SVG.
  These are drafts for author review, not a claim that three figures fit the
  four-page paper. The generation figure is the compact overview candidate.

## Paper wording to reconcile later

The supplied `docs/ICASSP2027_audiogeneration.pdf`, section 3.3.2, says the
pipeline remains continuous end to end and uses source cross-attention plus
transcript alignment. That wording does not fully describe these systems:

1. A2S has discrete acoustic code/duration prediction before continuous latent
   diffusion; its selected renderer uses frame-aligned channel concatenation.
2. The pooled global condition is AudioLDM class conditioning. Cross-attention
   to scene frames occurs in the content LM, not as the selected U-Net content
   mechanism.
3. S2A aligns to target-audio CLAP embeddings; A2S aligns to paired-transcript
   CLAP text embeddings.
4. A2S training is staged; LM and renderer are not jointly optimized through
   waveform generation.

The PDF itself was not edited. Keep this ledger when refining the paper and the
model-page wording with the author.
