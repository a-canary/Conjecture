"""
Conjecture: Core package for Simplified Universal Claim Architecture
Provides unified claim models and relationship management
"""

# Import the Conjecture class from the main module
from ..conjecture import Conjecture, ExplorationResult

# Core models - single source of truth
from .models import (
    Claim,
    ClaimBatch,
    ClaimState,
    ClaimState,
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
    create_claim,
)

from .support_relationship_manager import (
    SupportRelationshipManager,
    RelationshipMetrics,
    TraversalResult,
)

# Export core components
__all__ = [
    # Conjecture class
    "Conjecture",
    "ExplorationResult",
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
    "create_claim",
]
