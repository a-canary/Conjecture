"""
Conjecture: Core package for Simplified Universal Claim Architecture
Provides unified claim models and relationship management
"""

# Core models - single source of truth
from .models import (
    Claim,
    ClaimBatch,
    ClaimState,
    ClaimScope,
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
    generate_claim_id,
)

# Export core components
__all__ = [
    # Core claim models
    "Claim",
    "ClaimType",
    "ClaimState",
    "ClaimScope",
    "ClaimBatch",
    "DirtyReason",
    "ProcessingResult",
    "ToolCall",
    "ExecutionResult",
    "ParsedResponse",
    # Helper functions
    "create_claim_index",
    "get_orphaned_claims",
    "get_root_claims",
    "get_leaf_claims",
    "filter_claims_by_tags",
    "filter_claims_by_confidence",
    "create_claim",
    "generate_claim_id",
]
