"""
Conjecture: Core package for Simplified Universal Claim Architecture
Provides simplified claim models and relationship management
"""

# Legacy models (for existing compatibility)
from .unified_models import (
    Claim,
    ClaimBatch,
    ClaimState,
    ClaimType,
    ProcessingResult,
    validate_unified_models,
)

# New simplified architecture components
from .unified_claim import (
    UnifiedClaim,
    create_claim_index,
    get_orphaned_claims,
    get_root_claims,
    get_leaf_claims,
    filter_claims_by_tags,
    filter_claims_by_confidence,
    create_instruction_claim,
    create_concept_claim,
    create_evidence_claim,
)

from .support_relationship_manager import (
    SupportRelationshipManager,
    RelationshipMetrics,
    TraversalResult,
)

# Export both legacy and new components
__all__ = [
    # Legacy components
    "Claim",
    "ClaimType",
    "ClaimState",
    "ClaimBatch",
    "ProcessingResult",
    "validate_unified_models",
    
    # New simplified architecture
    "UnifiedClaim",
    "SupportRelationshipManager",
    "RelationshipMetrics",
    "TraversalResult",
    "create_claim_index",
    "get_orphaned_claims",
    "get_root_claims",
    "get_leaf_claims",
    "filter_claims_by_tags",
    "filter_claims_by_confidence",
    "create_instruction_claim",
    "create_concept_claim",
    "create_evidence_claim",
]
