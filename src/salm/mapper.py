"""
Deprecated shim. The real implementations live in `salm.conditioners`.

"mapper" was the original name for what is now consistently called a **conditioner** (it
produces AudioLDM's conditioning vector). This module re-exports the new names so any old
import path keeps working; new code should import from `salm.conditioners` directly.
"""

from salm.conditioners import (  # noqa: F401
    CLAP_EMBED_DIM,
    AttentionPool,
    BaseConditioner,
    SpeechToClapMapper,
    XFormerConditioner,
)

__all__ = [
    "CLAP_EMBED_DIM",
    "AttentionPool",
    "BaseConditioner",
    "SpeechToClapMapper",
    "XFormerConditioner",
]
