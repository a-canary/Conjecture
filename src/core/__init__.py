"""
Conjecture: Core package for Unified Claim Architecture
Provides unified claim models and relationship management
"""

# Core models - single source of truth
from .models import (
    Claim,
    ClaimBatch,
    ClaimState,
    ClaimType,
    ClaimScope,
    DirtyReason,
    ProcessingResult,
    ToolCall,
    ExecutionResult,
    ParsedResponse,
    Relationship,
    get_orphaned_claims,
    get_root_claims,
    get_leaf_claims,
    filter_claims_by_tags,
    filter_claims_by_confidence,
    create_claim,
    generate_claim_id,
    validate_claim_id,
    validate_confidence,
)

# DataConfig from data layer (single source of truth)
from src.data.models import DataConfig

# Unified models (consolidated from multiple sources)
from .unified_models import (
    DataSource,
    DataItem,
    ContextItemType,
    ContextItem,
    ContextResult,
    TokenUsage,
    CacheEntry,
    PromptTemplateStatus,
    PromptTemplateType,
    PromptVariable,
    PromptTemplate,
    LLMResponse,
    ResponseSchema,
    ParsedLLMResponse,
    FallbackResponse,
    PromptMetrics,
)

# Export all components
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
    "Relationship",
    "DataConfig",
    
    # Helper functions
    "get_orphaned_claims",
    "get_root_claims",
    "get_leaf_claims",
    "filter_claims_by_tags",
    "filter_claims_by_confidence",
    "create_claim",
    "generate_claim_id",
    "validate_claim_id",
    "validate_confidence",
    
    # Unified models
    "DataSource",
    "DataItem",
    "ContextItemType",
    "ContextItem",
    "ContextResult",
    "TokenUsage",
    "CacheEntry",
    "PromptTemplateStatus",
    "PromptTemplateType",
    "PromptVariable",
    "PromptTemplate",
    "LLMResponse",
    "ResponseSchema",
    "ParsedLLMResponse",
    "FallbackResponse",
    "PromptMetrics",
]