"""
Unified Core Models for Conjecture
Consolidates all essential data models into a single, clean system
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

# Import core models to maintain backward compatibility
from .models import (
    Claim, ClaimState, ClaimType, ClaimScope, DirtyReason,
    ClaimBatch, ProcessingResult, ToolCall, ExecutionResult,
    ParsedResponse, ClaimFilter, Relationship,
    create_claim, validate_claim_id, validate_confidence, generate_claim_id
)

# DataConfig from data layer (single source of truth)
from src.data.models import DataConfig

class DataSource(str, Enum):
    """Data source types"""
    USER_INPUT = "user_input"
    EXTERNAL_API = "external_api"
    FILE_SYSTEM = "file_system"
    DATABASE = "database"
    WEB_SEARCH = "web_search"
    EXISTING_CLAIMS = "existing_claims"
    TOOL_RESULT = "tool_result"
    KNOWLEDGE_BASE = "knowledge_base"

class DataItem(BaseModel):
    """Represents a piece of collected data"""
    id: str = Field(..., description="Data item ID")
    source: DataSource = Field(..., description="Data source")
    content: Any = Field(..., description="Data content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    confidence: float = Field(default=1.0, description="Confidence score (0.0-1.0)")
    relevance_score: Optional[float] = Field(None, description="Relevance to query")
    size_bytes: int = Field(default=0, description="Data size in bytes")

class ContextItemType(str, Enum):
    """Context item types"""
    EXAMPLE = "example"
    CLAIM = "claim"
    TOOL = "tool"
    DATA = "data"
    KNOWLEDGE = "knowledge"

class ContextItem(BaseModel):
    """Represents an item in context"""
    id: str = Field(..., description="Context item ID")
    item_type: ContextItemType = Field(..., description="Type of context item")
    content: str = Field(..., description="Context content")
    relevance_score: float = Field(..., description="Relevance score (0.0-1.0)")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    token_estimate: int = Field(0, description="Estimated token count")
    source: Optional[str] = Field(None, description="Source of context item")
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ContextResult(BaseModel):
    """Result of context building"""
    query: str = Field(..., description="Original query")
    context_items: List[ContextItem] = Field(..., description="Collected context items")
    total_tokens: int = Field(0, description="Total token count")
    processing_time_ms: int = Field(0, description="Processing time in milliseconds")
    collection_method: str = Field("default", description="Method used for collection")
    optimization_applied: bool = Field(False, description="Whether optimization was applied")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class TokenUsage(BaseModel):
    """Token usage information"""
    input_tokens: int = Field(0)
    output_tokens: int = Field(0)
    total_tokens: int = Field(0)
    context_tokens: int = Field(0)

class CacheEntry(BaseModel):
    """Cache entry for data"""
    key: str = Field(..., description="Cache key")
    data: Any = Field(..., description="Cached data")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    ttl_seconds: int = Field(300, description="Time to live in seconds")
    hit_count: int = Field(0, description="Number of cache hits")
    size_bytes: int = Field(0, description="Cached data size")

    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return (datetime.utcnow() - self.timestamp).total_seconds() > self.ttl_seconds

    def increment_hit(self) -> None:
        """Increment hit count"""
        self.hit_count += 1

class PromptTemplateStatus(str, Enum):
    """Prompt template status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DRAFT = "draft"
    DEPRECATED = "deprecated"

class PromptTemplateType(str, Enum):
    """Prompt template types"""
    RESEARCH = "research"
    CODING = "coding"
    TESTING = "testing"
    EVALUATION = "evaluation"
    GENERAL = "general"
    CUSTOM = "custom"

class PromptVariable(BaseModel):
    """Represents a prompt template variable"""
    name: str = Field(..., description="Variable name")
    type: str = Field(default="string", description="Variable type")
    required: bool = Field(default=True, description="Whether variable is required")
    default_value: Optional[Any] = Field(None, description="Default value")
    description: str = Field("", description="Variable description")
    validation_rule: Optional[str] = Field(None, description="Validation rule")

class PromptTemplate(BaseModel):
    """Represents a prompt template"""
    id: str = Field(..., description="Template ID")
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    template_type: PromptTemplateType = Field(..., description="Template type")
    status: PromptTemplateStatus = Field(default=PromptTemplateStatus.ACTIVE)
    template_content: str = Field(..., description="Template content with placeholders")
    variables: List[PromptVariable] = Field(default_factory=list, description="Template variables")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    version: str = Field(default="1.0", description="Template version")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    usage_count: int = Field(0, description="Number of times used")
    success_rate: float = Field(default=1.0, description="Success rate")
    
    @field_validator('variables')
    @classmethod
    def validate_content_placeholders(cls, v, values):
        """Ensure all placeholders in content have corresponding variables"""
        if 'template_content' in values:
            import re
            # Find {{variable}} placeholders
            placeholders = re.findall(r'\{\{(\w+)\}\}', values.get('template_content', ''))
            variable_names = [var.name for var in v]
            
            missing_vars = set(placeholders) - set(variable_names)
            if missing_vars:
                raise ValueError(f"Template contains undefined variables: {missing_vars}")
        
        return v
    
    def render(self, variables: Dict[str, Any]) -> str:
        """Render template with provided variables"""
        # Validate required variables
        for var in self.variables:
            if var.required and var.name not in variables:
                missing_var = var.name
                var_description = var.description or missing_var
                raise ValueError(f"Required variable '{missing_var}' ({var_description}) not provided")
        
        # Set default values
        resolved_vars = {}
        for var in self.variables:
            if var.name in variables:
                resolved_vars[var.name] = variables[var.name]
            elif var.default_value is not None:
                resolved_vars[var.name] = var.default_value
        
        # Render template
        rendered = self.template_content
        for name, value in resolved_vars.items():
            placeholder = "{{" + name + "}}"
            rendered = rendered.replace(placeholder, str(value))
        
        return rendered

class LLMResponse(BaseModel):
    """Represents an LLM response"""
    content: str = Field(..., description="Response content")
    model: str = Field(..., description="Model used")
    token_usage: Dict[str, int] = Field(..., description="Token usage information")
    response_time_ms: int = Field(0, description="Response time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class ResponseSchema(BaseModel):
    """Schema for validating LLM responses"""
    schema_name: str = Field(..., description="Schema name")
    required_fields: List[str] = Field(default_factory=list)
    field_types: Dict[str, str] = Field(default_factory=dict)
    validation_rules: Dict[str, str] = Field(default_factory=dict)
    custom_validator: Optional[str] = Field(None, description="Custom validator name")

class ParsedLLMResponse(BaseModel):
    """Represents a parsed and validated LLM response"""
    is_valid: bool = Field(default=False)
    parsed_data: Optional[Dict[str, Any]] = Field(None)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, description="Confidence in parsing")
    extracted_claims: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class FallbackResponse(BaseModel):
    """Represents a fallback response when parsing fails"""
    response_type: str = Field(..., description="Type of fallback")
    message: str = Field(..., description="Fallback message")
    original_response: str = Field(..., description="Original LLM response")
    parsing_errors: List[str] = Field(default_factory=list)
    should_retry: bool = Field(default=False, description="Whether to retry original request")
    retry_hints: List[str] = Field(default_factory=list)

class PromptMetrics(BaseModel):
    """Metrics for prompt performance"""
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        frozen=False,
        protected_namespaces=()
    )
    
    template_id: str = Field(..., description="Template ID")
    usage_count: int = Field(default=0)
    success_rate: float = Field(default=0.0)
    average_response_time_ms: float = Field(default=0.0)
    average_tokens_used: float = Field(default=0.0)
    last_used: Optional[datetime] = Field(None)
    error_rate: float = Field(default=0.0)
    user_satisfaction_score: Optional[float] = Field(None)

# Re-export all models for backward compatibility
__all__ = [
    # Core models
    'Claim', 'ClaimState', 'ClaimType', 'ClaimScope', 'DirtyReason',
    'ClaimBatch', 'ProcessingResult', 'ToolCall', 'ExecutionResult',
    'ParsedResponse', 'ClaimFilter', 'Relationship', 'DataConfig',
    
    # Utility functions
    'create_claim', 'validate_claim_id', 'validate_confidence', 'generate_claim_id',
    
    # Unified models
    'DataSource', 'DataItem', 'ContextItemType', 'ContextItem', 'ContextResult',
    'TokenUsage', 'CacheEntry', 'PromptTemplateStatus', 'PromptTemplateType',
    'PromptVariable', 'PromptTemplate', 'LLMResponse', 'ResponseSchema',
    'ParsedLLMResponse', 'FallbackResponse', 'PromptMetrics'
]