"""
Conjecture: Evidence-based AI reasoning system
Simplified Universal Claim Architecture with LLM-Driven Instruction Support
"""

__version__ = "1.0.0"

# Core components
from .core import (
    Conjecture,
    ExplorationResult,
)
from .core.models import (
    Claim,
    ClaimType,
    ClaimState,
)

# High-level API
__all__ = [
    "Conjecture",
    "ExplorationResult",
    "Claim",
    "ClaimType",
    "ClaimState",
]
