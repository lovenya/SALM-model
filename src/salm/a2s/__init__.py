"""
A2S: audio -> speech. Kept as its own module rather than folded into the S2A code.

The split is deliberate and was the user's call (2026-08-23). A2S is the direction where a lot
of things will be tried -- precomputed conditioning, a different alignment target, and later
sequence/cross-attention conditioning that changes how the U-Net is called -- so giving it its
own surface keeps that churn away from the S2A pipeline that currently produces the results.

What is NOT duplicated: the conditioner ladder (`salm.conditioners`), the mel front-end
(`salm.data.frontend`), the manifest reader, and the training loop itself. Those are shared,
and the direction-specific parts plug into them:

    encoder      salm.encoders.clap_encoder      / salm.encoders.precomputed
    dataset      salm.a2s.dataset                (this module)
    registry     arch name 'a2s_audioldm_clapseq_<rung>'

Forking `train_ldm.py` was considered and rejected: it carries the DDP collective-deadline fix,
the resume path and the selection-metric logic, all of which were bought with debugging time,
and two copies would drift.
"""
