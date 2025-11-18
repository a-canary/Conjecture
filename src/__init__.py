"""
Conjecture: Evidence-based AI reasoning system
Simplified Universal Claim Architecture with LLM-Driven Instruction Support
"""

__version__ = "1.0.0"

# Core components
from .core import (
    UnifiedClaim,
    SupportRelationshipManager,
    RelationshipMetrics,
    TraversalResult,
)

# Context building
from .context import CompleteContextBuilder

# LLM integration
from .llm import InstructionSupportProcessor

# High-level API
__all__ = [
    "UnifiedClaim",
    "SupportRelationshipManager", 
    "CompleteContextBuilder",
    "InstructionSupportProcessor",
    "RelationshipMetrics",
    "TraversalResult",
]
