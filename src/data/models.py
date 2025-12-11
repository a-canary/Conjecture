"""
Data models and configuration for the Conjecture data layer.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict, field_serializer
from src.core.models import ClaimScope

class DataLayerError(Exception):
    """Base exception for data layer operations."""

    pass

class ClaimNotFoundError(DataLayerError):
    """Raised when a claim is not found."""

    pass

class InvalidClaimError(DataLayerError):
    """Raised when claim validation fails."""

    pass

class RelationshipError(DataLayerError):
    """Raised when relationship operations fail."""

    pass

class EmbeddingError(DataLayerError):
    """Raised when embedding operations fail."""

    pass

class ClaimState(str, Enum):
    """Claim state enumeration"""

    EXPLORE = "Explore"
    VALIDATED = "Validated"
    ORPHANED = "Orphaned"
    QUEUED = "Queued"

class ClaimType(str, Enum):
    """Claim type enumeration"""

    CONCEPT = "concept"
    REFERENCE = "reference"
    THESIS = "thesis"
    SKILL = "skill"
    EXAMPLE = "example"
    GOAL = "goal"

class Claim(BaseModel):
    """Enhanced Claim model for data layer operations"""

    # Core fields
    id: str = Field(..., description="Unique claim identifier (format: c########)")
    content: str = Field(
        ..., min_length=10, max_length=5000, description="Claim content"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score (0.0-1.0)"
    )
    dirty: bool = Field(default=True, description="Whether claim needs synchronization")
    created: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )

    # Optional fields
    updated: Optional[datetime] = Field(
        default=None, description="Last update timestamp"
    )
    tags: List[str] = Field(default_factory=list, description="Topic tags")
    state: ClaimState = Field(
        default=ClaimState.EXPLORE, description="Current claim state"
    )
    type: List[ClaimType] = Field(
        default_factory=lambda: [ClaimType.CONCEPT], description="Claim types"
    )
    embedding: Optional[List[float]] = Field(
        default=None, description="Vector embedding"
    )
    
    # Scope field for access control
    scope: ClaimScope = Field(
        default=ClaimScope.USER_WORKSPACE, description="Claim scope for access control"
    )
    
    # Dirty flag fields
    is_dirty: bool = Field(
        default=True, description="Whether claim needs re-evaluation"
    )
    dirty_reason: Optional[str] = Field(
        default=None, description="Reason why claim was marked dirty"
    )
    dirty_timestamp: Optional[datetime] = Field(
        default=None, description="When claim was marked dirty"
    )
    dirty_priority: int = Field(
        default=0, description="Priority for dirty evaluation (higher = more urgent)"
    )

    # Relationship fields (for SQLite)
    supported_by: List[str] = Field(
        default_factory=list, description="Claim IDs that support this claim"
    )
    supports: List[str] = Field(
        default_factory=list, description="Claim IDs this claim supports"
    )

    model_config = ConfigDict()

    @field_serializer("created", "updated")
    def serialize_datetime(self, v: Optional[datetime]) -> Optional[str]:
        """Serialize datetime to ISO format"""
        return v.isoformat() if v else None

    @field_validator("id")
    @classmethod
    def validate_id(cls, v):
        """Validate claim ID format"""
        if not v.startswith("c") or not v[1:].isdigit():
            raise ValueError(
                "Claim ID must be in format c######## (c followed by 8 digits)"
            )
        return v

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v):
        """Validate tags are clean strings and remove duplicates"""
        if v:
            for tag in v:
                if not isinstance(tag, str) or not tag.strip():
                    raise ValueError("Tags must be non-empty strings")
        # Remove duplicates while preserving order
        seen = set()
        unique_tags = []
        for tag in [tag.strip() for tag in v if tag.strip()]:
            if tag not in seen:
                seen.add(tag)
                unique_tags.append(tag)
        return unique_tags

    @field_validator("updated")
    @classmethod
    def validate_updated_timestamp(cls, v, info):
        """Ensure updated timestamp is not before creation"""
        if v and "created" in info.data and v < info.data["created"]:
            raise ValueError("Updated timestamp cannot be before creation timestamp")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return self.dict(exclude_none=True)

    @property
    def created_at(self) -> datetime:
        """Backward compatibility property for created timestamp"""
        return self.created

    def to_chroma_metadata(self) -> Dict[str, Any]:
        """Convert claim to ChromaDB metadata format"""
        return {
            "confidence": self.confidence,
            "state": self.state.value,
            "supported_by": ",".join(self.supported_by) if self.supported_by else "",
            "supports": ",".join(self.supports) if self.supports else "",
            "type": ",".join([t.value for t in self.type]),
            "tags": ",".join(self.tags) if self.tags else "",
            "created": self.created.isoformat(),
            "updated": self.updated.isoformat()
            if self.updated
            else self.created.isoformat(),
            "dirty": self.dirty,
        }

    @classmethod
    def from_chroma_result(
        cls, id: str, content: str, metadata: Dict[str, Any]
    ) -> "Claim":
        """Create claim from ChromaDB query result"""
        return cls(
            id=id,
            content=content,
            confidence=metadata["confidence"],
            state=ClaimState(metadata["state"]),
            supported_by=metadata.get("supported_by", "").split(",")
            if metadata.get("supported_by")
            else [],
            supports=metadata.get("supports", "").split(",")
            if metadata.get("supports")
            else [],
            type=[
                ClaimType(t) for t in metadata.get("type", "concept").split(",") if t
            ],
            tags=metadata.get("tags", "").split(",") if metadata.get("tags") else [],
            created=datetime.fromisoformat(metadata["created"]),
            updated=datetime.fromisoformat(metadata["updated"])
            if metadata.get("updated")
            else None,
            dirty=metadata.get("dirty", False),
        )

    def mark_dirty(self):
        """Mark claim as needing synchronization"""
        self.dirty = True
        self.updated = datetime.utcnow()

    def mark_clean(self):
        """Mark claim as synchronized"""
        self.dirty = False
        self.updated = datetime.utcnow()

    def update_confidence(self, new_confidence: float):
        """Update confidence score"""
        if not (0.0 <= new_confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        self.confidence = new_confidence
        self.mark_dirty()

    def add_support(self, supporting_claim_id: str):
        """Add a supporting claim ID"""
        if supporting_claim_id and supporting_claim_id not in self.supported_by:
            self.supported_by.append(supporting_claim_id)
            self.mark_dirty()

    def add_supports(self, supported_claim_id: str):
        """Add a claim this claim supports"""
        if supported_claim_id and supported_claim_id not in self.supports:
            self.supports.append(supported_claim_id)
            self.mark_dirty()

    def __repr__(self):
        return f"Claim(id={self.id}, confidence={self.confidence:.2f}, state={self.state.value})"

class Relationship(BaseModel):
    """Simplified claim relationship model - only supports 'A supports B' connections"""

    supporter_id: str = Field(..., description="ID of supporting claim (A)")
    supported_id: str = Field(..., description="ID of supported claim (B)")
    created: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )

    @property
    def created_at(self) -> datetime:
        """Backward compatibility property for created timestamp"""
        return self.created

    def __repr__(self) -> str:
        return f"Relationship({self.supporter_id} supports {self.supported_id})"

class ClaimFilter(BaseModel):
    """Filter options for claim queries"""

    # Basic filters
    tags: Optional[List[str]] = Field(
        default=None, description="Filter by tags (any match)"
    )
    confidence_min: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Minimum confidence"
    )
    confidence_max: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Maximum confidence"
    )
    dirty_only: Optional[bool] = Field(
        default=None, description="Filter dirty claims only"
    )
    content_contains: Optional[str] = Field(
        default=None, description="Content contains text"
    )

    # Pagination
    limit: Optional[int] = Field(
        default=100, ge=1, le=1000, description="Maximum results"
    )
    offset: Optional[int] = Field(default=0, ge=0, description="Result offset")

    # Date range
    created_after: Optional[datetime] = Field(
        default=None, description="Created after this date"
    )
    created_before: Optional[datetime] = Field(
        default=None, description="Created before this date"
    )

    # State and type filters
    states: Optional[List[ClaimState]] = Field(
        default=None, description="Filter by states"
    )
    types: Optional[List[ClaimType]] = Field(
        default=None, description="Filter by claim types"
    )

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, v):
        """Set reasonable default limit"""
        if v is None:
            return 100
        return v

    @field_validator("offset")
    @classmethod
    def validate_offset(cls, v):
        """Default offset to 0"""
        if v is None:
            return 0
        return v

    @field_validator("confidence_max")
    @classmethod
    def validate_confidence_range(cls, v, info):
        """Validate confidence range and log errors for out-of-range values"""
        if v is not None:
            # Check if value is outside 0-1 range (assuming 0-1 scale, not 0-100)
            if v < 0.0 or v > 1.0:
                import logging

                logger = logging.getLogger(__name__)
                logger.error(f"Confidence value {v} is outside valid range 0.0-1.0")
                # For tool response context, we'll still allow it but log the error
            # Check range consistency - this should raise ValidationError for test compatibility
            if (
                "confidence_min" in info.data
                and info.data["confidence_min"] is not None
            ):
                if v < info.data["confidence_min"]:
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.error(
                        f"confidence_max ({v}) must be >= confidence_min ({info.data['confidence_min']})"
                    )
                    raise ValueError(
                        f"confidence_max ({v}) must be >= confidence_min ({info.data['confidence_min']})"
                    )
        return v

class DataConfig(BaseModel):
    """Configuration for data layer components"""

    # SQLite configuration
    sqlite_path: str = Field(
        default="./data/conjecture.db", description="SQLite database path"
    )

    # ChromaDB configuration
    chroma_path: str = Field(default="./data/vector_db", description="ChromaDB path")
    chroma_collection: str = Field(
        default="claims", description="ChromaDB collection name"
    )

    # Embedding configuration
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", description="Embedding model name"
    )
    embedding_dim: Optional[int] = Field(
        default=None, description="Embedding dimension"
    )
    embedding_service: Optional[Any] = Field(default=None, description="Embedding service instance")
    vector_store: Optional[Any] = Field(default=None, description="Vector store instance")
    
    # Performance configuration
    max_tokens: int = Field(default=8000, ge=1000, description="Maximum context tokens")
    cache_size: int = Field(default=1000, ge=0, description="Cache size")
    cache_ttl: int = Field(default=300, ge=0, description="Cache TTL in seconds")
    batch_size: int = Field(default=100, ge=1, le=1000, description="Batch size")
    max_connections: Optional[int] = Field(default=10, ge=1, description="Maximum database connections")
    query_timeout: int = Field(default=30, description="Query timeout in seconds")

    # Feature flags
    use_chroma: bool = Field(default=True, description="Whether to use ChromaDB for vector storage")
    use_embeddings: bool = Field(default=True, description="Whether to use embeddings")
    auto_sync: bool = Field(default=True, description="Auto-sync dirty claims")

class QueryResult(BaseModel):
    """Standardized query result"""

    items: List[Dict[str, Any]] = Field(
        default_factory=list, description="Query results"
    )
    total_count: Optional[int] = Field(
        default=None, description="Total available items"
    )
    query_time: Optional[float] = Field(
        default=None, description="Query execution time"
    )
    has_more: bool = Field(default=False, description="More results available")

    def __len__(self):
        return len(self.items)

# Import common result classes to maintain backward compatibility
from src.core.common_results import ProcessingResult, BatchResult

class ProcessingStats(BaseModel):
    """Processing statistics and metrics"""

    operation: str = Field(..., description="Operation name")
    start_time: datetime = Field(..., description="Start timestamp")
    end_time: Optional[datetime] = Field(default=None, description="End timestamp")
    items_processed: int = Field(default=0, description="Items processed")
    items_succeeded: int = Field(default=0, description="Items successfully processed")
    items_failed: int = Field(default=0, description="Items failed")

    @property
    def duration(self) -> Optional[float]:
        """Duration in seconds"""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def success_rate(self) -> float:
        """Success rate as percentage"""
        if self.items_processed == 0:
            return 0.0
        return (self.items_succeeded / self.items_processed) * 100

    @property
    def items_per_second(self) -> Optional[float]:
        """Items processed per second"""
        if self.duration and self.duration > 0:
            return self.items_processed / self.duration
        return None

    def start(self):
        """Start processing"""
        self.start_time = datetime.utcnow()

    def end(self):
        """End processing"""
        self.end_time = datetime.utcnow()

    def add_success(self, count: int = 1):
        """Add successful items"""
        self.items_processed += count
        self.items_succeeded += count

    def add_failure(self, count: int = 1):
        """Add failed items"""
        self.items_processed += count
        self.items_failed += count

class ProcessingResult(BaseModel):
    """Result of processing a single claim"""

    claim_id: str = Field(..., description="ID of the processed claim")
    success: bool = Field(..., description="Whether processing was successful")
    message: Optional[str] = Field(default=None, description="Processing message")
    updated_confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Updated confidence score"
    )
    processing_time: Optional[float] = Field(
        default=None, ge=0.0, description="Processing time in seconds"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional processing metadata"
    )

# Utility functions for claim validation and processing
def validate_claim_id(claim_id: str) -> bool:
    """Validate claim ID format"""
    return claim_id.startswith("c") and len(claim_id) == 8 and claim_id[1:].isdigit()

def validate_confidence(confidence: float) -> bool:
    """Validate confidence score"""
    return 0.0 <= confidence <= 1.0

def generate_claim_id(counter: int = 1) -> str:
    """Generate a claim ID from a counter"""
    return f"c{counter:07d}"
