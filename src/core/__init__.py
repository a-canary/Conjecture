"""
Conjecture: Core package for Simplified Universal Claim Architecture
Provides unified claim models and relationship management
"""

# Core models - single source of truth
from .models import (
    Claim,
    ClaimBatch,
    ClaimState,
    ClaimType,
    DirtyReason,
    ProcessingResult,
    ToolCall,
    ExecutionResult,
    ParsedResponse,
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

# Export core components
__all__ = [
    # Core claim models
    "Claim",
    "ClaimType",
    "ClaimState",
    "ClaimBatch",
    "DirtyReason",
    "ProcessingResult",
    "ToolCall",
    "ExecutionResult",
    "ParsedResponse",
    # Relationship management
    "SupportRelationshipManager",
    "RelationshipMetrics",
    "TraversalResult",
    # Helper functions
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
