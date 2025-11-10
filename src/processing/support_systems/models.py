"""
Data models for Support Systems
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


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


class DataSchema(BaseModel):
    """Schema definition for data validation"""
    required_fields: List[str] = Field(default_factory=list)
    optional_fields: List[str] = Field(default_factory=list)
    field_types: Dict[str, str] = Field(default_factory=dict)
    validators: Dict[str, str] = Field(default_factory=dict)
    custom_validation: Optional[str] = Field(None, description="Custom validation rule")


class ValidationResult(BaseModel):
    """Result of data validation"""
    is_valid: bool = Field(default=False)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    transformed_data: Optional[Dict[str, Any]] = Field(None)


class ProcessedData(BaseModel):
    """Represents processed data after validation and transformation"""
    original_item: DataItem = Field(..., description="Original data item")
    validated_data: Any = Field(..., description="Validated data")
    processing_operations: List[str] = Field(default_factory=list)
    validation_result: Optional[ValidationResult] = Field(None)


class ContextItemType(str, Enum):
    """Context item types"""
    SKILL = "skill"
    EXAMPLE = "example"
    CLAIM = "claim"
    TOOL = "tool"
    DATA = "data"
    KNOWLEDGE = "knowledge"


class ContextItem(BaseModel):
    """Represents an item in the context"""
    id: str = Field(..., description="Context item ID")
    item_type: ContextItemType = Field(..., description="Type of context item")
    content: str = Field(..., description="Context content")
    relevance_score: float = Field(..., description="Relevance score (0.0-1.0)")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    token_estimate: int = Field(0, description="Estimated token count")
    source: Optional[str] = Field(None, description="Source of the context item")
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


class OptimizedContext(BaseModel):
    """Optimized context for LLM consumption"""
    items: List[ContextItem] = Field(..., description="Included context items")
    excluded_items: List[ContextItem] = Field(default_factory=list, description="Excluded items")
    total_tokens: int = Field(0, description="Total token count")
    token_limit: int = Field(0, description="Token limit used")
    optimization_strategy: str = Field("relevance", description="Optimization strategy")
    optimization_score: float = Field(0.0, description="Optimization quality score")


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