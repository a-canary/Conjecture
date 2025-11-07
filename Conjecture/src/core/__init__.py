"""
Conjecture: Unified models package
Single, elegant implementation for all claim types
"""

from .unified_models import (
    Claim,
    ClaimBatch,
    ClaimState,
    ClaimType,
    ProcessingResult,
    validate_unified_models,
)

__all__ = [
    "Claim",
    "ClaimType",
    "ClaimState",
    "ClaimBatch",
    "ProcessingResult",
    "validate_unified_models",
]
