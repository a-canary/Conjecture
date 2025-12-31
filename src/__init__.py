"""
Conjecture: Evidence-based AI reasoning system
Simplified Universal Claim Architecture with LLM-Driven Instruction Support
"""

__version__ = "1.0.0"

# Core components
from .core.models import (
    Claim,
    ClaimState,
    Relationship,
    ClaimFilter,
)

# High-level API
__all__ = [
    "Claim",
    "ClaimType",
    "ClaimState",
]
