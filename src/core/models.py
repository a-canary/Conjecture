"""
Core Pydantic models for Conjecture claims and data structures
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

class RelationshipError(Exception):
    """Exception raised for relationship-related errors"""

    pass

class DataLayerError(Exception):
    """Exception raised for data layer operation errors"""

    pass

class ClaimNotFoundError(Exception):
    """Exception raised when a claim is not found"""

    pass

class InvalidClaimError(Exception):
    """Exception raised for invalid claim data"""

    pass

class ClaimState(str, Enum):
    """Claim state enumeration"""

    EXPLORE = "Explore"
    VALIDATED = "Validated"
    ORPHANED = "Orphaned"
    QUEUED = "Queued"

class ClaimType(str, Enum):
    """Claim type enumeration - All claims are impressions, assumptions, observations, or conjectures with variable truth"""

    IMPRESSION = "impression"  # Initial impression or intuition about something
    ASSUMPTION = "assumption"  # Something taken as true without proof for reasoning
    OBSERVATION = "observation"  # Something noticed or perceived through senses/data
    CONJECTURE = "conjecture"  # Conclusion formed on incomplete evidence
    CONCEPT = "concept"  # Abstract idea or general notion
    EXAMPLE = "example"  # Specific instance or case
    GOAL = "goal"  # Desired outcome or objective
    REFERENCE = "reference"  # Pointer to external information
    ASSERTION = "assertion"  # Strong statement made with confidence
    THESIS = "thesis"  # Proposition put forward for consideration
    HYPOTHESIS = "hypothesis"  # Proposed explanation for investigation
    QUESTION = "question"  # Query seeking information
    TASK = "task"  # Action item or work to be done

class ClaimScope(str, Enum):
    """Claim scope enumeration - simplified for local-first design"""

    # Most restrictive (default) to least restrictive
    USER_WORKSPACE = (
        "user-{workspace}"  # Personal work: tasks, questions, hypotheses, bugs
    )
    TEAM_WORKSPACE = (
        "team-{workspace}"  # Team project info: features, discoveries, insights
    )
    TEAM_WIDE = "team-wide"  # Global team knowledge: best practices, decisions
    PUBLIC = "public"  # Shareable global truths for broader audience

    @classmethod
    def get_default(cls) -> str:
        """Get the default scope (most restrictive for security)"""
        return cls.USER_WORKSPACE.value

    @classmethod
    def get_hierarchy(cls) -> list:
        """Get scope hierarchy from most to least restrictive"""
        return [
            cls.USER_WORKSPACE.value,
            cls.TEAM_WORKSPACE.value,
            cls.TEAM_WIDE.value,
            cls.PUBLIC.value,
        ]

    @classmethod
    def normalize_scope(cls, scope: str, workspace: str = None) -> str:
        """Normalize scope with workspace context where applicable"""
        if workspace and scope in [cls.USER_WORKSPACE.value, cls.TEAM_WORKSPACE.value]:
            return scope.format(workspace=workspace)
        return scope

    @classmethod
    def is_valid_scope(cls, scope: str) -> bool:
        """Check if scope is valid"""
        return scope in [s.value for s in cls]

    @classmethod
    def get_scope_level(cls, scope: str) -> int:
        """Get restriction level (0=most restrictive, 3=least restrictive)"""
        hierarchy = cls.get_hierarchy()
        try:
            return hierarchy.index(scope)
        except ValueError:
            return -1  # Invalid scope

class DirtyReason(str, Enum):
    """Dirty flag reason enumeration"""

    NEW_CLAIM_ADDED = "new_claim_added"
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    SUPPORTING_CLAIM_CHANGED = "supporting_claim_changed"
    RELATIONSHIP_CHANGED = "relationship_changed"
    MANUAL_MARK = "manual_mark"
    BATCH_EVALUATION = "batch_evaluation"
    SYSTEM_TRIGGER = "system_trigger"

class Claim(BaseModel):
    """Core claim model with validation"""
    
    model_config = {"protected_namespaces": ()}

    id: str = Field(..., description="Unique claim identifier")
    content: str = Field(
        ..., min_length=5, max_length=2000, description="Claim content"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score (0.0-1.0)"
    )
    state: ClaimState = Field(
        default=ClaimState.EXPLORE, description="Current claim state"
    )
    supported_by: List[str] = Field(
        default_factory=list, description="Claims that support this claim"
    )
    supports: List[str] = Field(
        default_factory=list, description="Claims this claim supports"
    )
    tags: List[str] = Field(default_factory=list, description="Topic tags")
    scope: ClaimScope = Field(
        default=ClaimScope.USER_WORKSPACE,
        description="Claim scope: user-workspace (personal), team-workspace (project), team-wide (global team), or public (shareable)",
    )
    created: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    updated: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )
    embedding: Optional[List[float]] = Field(
        default=None, description="Vector embedding"
    )
    # Dirty flag fields
    is_dirty: bool = Field(
        default=True, description="Whether claim needs re-evaluation"
    )
    dirty_reason: Optional[DirtyReason] = Field(
        default=None, description="Reason why claim was marked dirty"
    )
    dirty_timestamp: Optional[datetime] = Field(
        default=None, description="When claim was marked dirty"
    )
    dirty_priority: int = Field(
        default=0, description="Priority for dirty evaluation (higher = more urgent)"
    )

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v):
        """Validate tags are strings and not empty"""
        if v:
            for tag in v:
                if not isinstance(tag, str) or not tag.strip():
                    raise ValueError("Tags must be non-empty strings")
        return list(dict.fromkeys(v))  # Remove duplicates while preserving order

    @model_validator(mode="after")
    def validate_updated_timestamp(self):
        """Ensure updated timestamp is not before creation"""
        if self.updated < self.created:
            raise ValueError("Updated timestamp cannot be before creation timestamp")
        return self

    def to_chroma_metadata(self) -> Dict[str, Any]:
        """Convert claim to ChromaDB metadata format"""
        return {
            "confidence": self.confidence,
            "state": self.state.value,
            "supported_by": self.supported_by,
            "supports": self.supports,
            "tags": self.tags,
            "created": self.created.isoformat(),
            "updated": self.updated.isoformat(),
            "is_dirty": self.is_dirty,
            "dirty_reason": self.dirty_reason.value if self.dirty_reason else None,
            "dirty_timestamp": self.dirty_timestamp.isoformat()
            if self.dirty_timestamp
            else None,
            "dirty_priority": self.dirty_priority,
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
            supported_by=metadata.get("supported_by", []),
            supports=metadata.get("supports", []),
            tags=metadata.get("tags", []),
            created=datetime.fromisoformat(metadata["created"]),
            updated=datetime.fromisoformat(metadata["updated"]),
            is_dirty=metadata.get("is_dirty", False),
            dirty_reason=DirtyReason(metadata["dirty_reason"])
            if metadata.get("dirty_reason")
            else None,
            dirty_timestamp=datetime.fromisoformat(metadata["dirty_timestamp"])
            if metadata.get("dirty_timestamp")
            else None,
            dirty_priority=metadata.get("dirty_priority", 0),
        )

    def format_for_context(self) -> str:
        """Format claim for LLM context in standard [c{id} | content | / confidence] format"""
        return f"[c{self.id} | {self.content} | / {self.confidence:.2f}]"

    def format_for_llm_analysis(self) -> str:
        """Format claim for detailed LLM analysis with metadata"""
        tags_str = ",".join(self.tags) if self.tags else "none"
        return (
            f"Claim ID: {self.id}\n"
            f"Content: {self.content}\n"
            f"Confidence: {self.confidence:.2f}\n"
            f"State: {self.state.value}\n"
            f"Tags: {tags_str}\n"
            f"Supports: {', '.join(self.supports) if self.supports else 'none'}\n"
            f"Supported By: {', '.join(self.supported_by) if self.supported_by else 'none'}"
        )

    def format_for_output(self) -> str:
        """Format claim for output in standard [c{id} | content | / confidence] format"""
        return f"[c{self.id} | {self.content} | / {self.confidence:.2f}]"

    def __repr__(self) -> str:
        return f"Claim(id={self.id}, confidence={self.confidence}, state={self.state.value}, dirty={self.is_dirty})"

    @classmethod
    def create_claim_index(cls, claims: List['Claim']) -> Dict[str, 'Claim']:
        """Create claim index for fast lookups"""
        return {claim.id: claim for claim in claims}
    
    @classmethod
    def get_orphaned_claims(cls, claims: List['Claim']) -> List['Claim']:
        """Get claims that are not supported by any other claims"""
        supported_ids = set()
        for claim in claims:
            supported_ids.update(claim.supports)
        
        return [claim for claim in claims if claim.id not in supported_ids]
    
    @property
    def dirty(self) -> bool:
        """Backward compatibility property for dirty flag"""
        return self.is_dirty

    @property
    def created_at(self) -> datetime:
        """Backward compatibility property for created timestamp"""
        return self.created

    @property
    def updated_at(self) -> datetime:
        """Backward compatibility property for updated timestamp"""
        return self.updated

    def __hash__(self) -> int:
        """Make Claim hashable for use in sets and as dict keys"""
        return hash((self.id, self.content, self.confidence))

    @property
    def is_confident(self, threshold: Optional[float] = None) -> bool:
        """Check if claim meets confidence threshold"""
        effective_threshold = threshold or self._get_default_threshold()
        return self.confidence >= effective_threshold

    @property
    def needs_evaluation(self) -> bool:
        """Check if claim needs further evaluation"""
        return not self.is_confident

    def _get_default_threshold(self) -> float:
        """Get effective confidence threshold"""
        try:
            from ..config.config import get_config

            config = get_config()
            return config.get_effective_confident_threshold()
        except ImportError:
            # Fallback if config not available
            return 0.8

# Backward compatibility alias for BasicClaim
# BasicClaim was the old name for Claim before the model consolidation
BasicClaim = Claim

class ClaimBatch(BaseModel):
    """Batch model for processing multiple claims"""

    claims: List[Claim] = Field(..., min_length=1, description="List of claims")
    batch_id: str = Field(..., description="Batch identifier")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Batch timestamp"
    )

    def to_chroma_batch(self) -> tuple:
        """Convert batch to ChromaDB batch format"""
        ids = [claim.id for claim in self.claims]
        documents = [claim.content for claim in self.claims]
        embeddings = [claim.embedding for claim in self.claims if claim.embedding]
        metadatas = [claim.to_chroma_metadata() for claim in self.claims]
        return ids, documents, embeddings, metadatas

class ProcessingResult(BaseModel):
    """Result of claim processing"""

    success: bool = Field(..., description="Processing success status")
    processed_claims: int = Field(..., description="Number of claims processed")
    updated_claims: int = Field(..., description="Number of claims updated")
    errors: List[str] = Field(default_factory=list, description="Processing errors")
    execution_time: Optional[float] = Field(
        None, description="Execution time in seconds"
    )
    message: str = Field(default="", description="Processing message")

# Tool execution models (moved from unified_models)
class ToolCall(BaseModel):
    """Represents a tool invocation."""

    name: str = Field(..., description="Name of the tool to invoke")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Tool parameters"
    )

class ExecutionResult(BaseModel):
    """Result of tool execution."""

    success: bool = Field(..., description="Execution success status")
    outcome: str = Field(..., description="Execution outcome or error message")
    duration: float = Field(..., description="Execution duration in seconds")
    tool_name: str = Field(..., description="Name of the tool that was executed")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

class ParsedResponse(BaseModel):
    """Represents a parsed response from LLM"""

    tool_calls: List[ToolCall] = Field(
        default_factory=list, description="Extracted tool calls"
    )
    errors: List[str] = Field(default_factory=list, description="Parsing errors")

class ClaimFilter(BaseModel):
    """Filter for querying claims"""

    tags: Optional[List[str]] = Field(default=None, description="Filter by tags")
    confidence_min: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Minimum confidence"
    )
    confidence_max: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Maximum confidence"
    )
    dirty_only: Optional[bool] = Field(
        default=None, description="Filter only dirty claims"
    )
    content_contains: Optional[str] = Field(
        default=None, description="Content contains text"
    )
    limit: Optional[int] = Field(
        default=100, ge=1, description="Maximum results to return"
    )
    offset: Optional[int] = Field(default=0, ge=0, description="Results offset")
    created_after: Optional[datetime] = Field(
        default=None, description="Created after timestamp"
    )
    created_before: Optional[datetime] = Field(
        default=None, description="Created before timestamp"
    )
    states: Optional[List[ClaimState]] = Field(
        default=None, description="Filter by states"
    )

    @model_validator(mode="after")
    def validate_confidence_range(self):
        """Validate confidence_max is >= confidence_min"""
        if (
            self.confidence_max is not None
            and self.confidence_min is not None
            and self.confidence_max < self.confidence_min
        ):
            raise ValueError("confidence_max must be >= confidence_min")
        return self

class Relationship(BaseModel):
    """Relationship between claims"""

    supporter_id: str = Field(..., description="ID of supporting claim")
    supported_id: str = Field(..., description="ID of supported claim")
    relationship_type: str = Field(..., description="Type of relationship")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Relationship confidence"
    )
    created: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    created_by: Optional[str] = Field(
        default=None, description="User who created relationship"
    )

    @field_validator("relationship_type")
    @classmethod
    def validate_relationship_type(cls, v):
        """Validate relationship type"""
        valid_types = ["supports", "contradicts", "relates_to", "depends_on"]
        if v not in valid_types:
            raise ValueError(f"Relationship type must be one of: {valid_types}")
        return v

    @property
    def supporter(self) -> str:
        """Backward compatibility property"""
        return self.supporter_id

    @property
    def supported(self) -> str:
        """Backward compatibility property"""
        return self.supported_id

    @property
    def created_at(self) -> datetime:
        """Backward compatibility property for created timestamp"""
        return self.created

class DataConfig(BaseModel):
    """Configuration for data layer"""

    sqlite_path: str = Field(
        default="./data/conjecture.db", description="SQLite database path"
    )
    chroma_path: str = Field(default="data/chroma", description="ChromaDB path")
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", description="Embedding model name"
    )
    max_tokens: int = Field(default=8000, ge=1000, description="Maximum context tokens")
    max_connections: Optional[int] = Field(default=10, ge=1, description="Maximum database connections")
    

# Helper functions for claim management
def get_orphaned_claims(claims: List[Claim]) -> List[Claim]:
    """Get claims that are not supported by any other claims"""
    supported_ids = set()
    for claim in claims:
        supported_ids.update(claim.supports)
    
    return [claim for claim in claims if claim.id not in supported_ids]

def get_root_claims(claims: List[Claim]) -> List[Claim]:
    """Get claims that don't support any other claims (root claims)"""
    supporting_ids = set()
    for claim in claims:
        supporting_ids.update(claim.supports)
    
    return [claim for claim in claims if claim.id not in supporting_ids]

def get_leaf_claims(claims: List[Claim]) -> List[Claim]:
    """Get claims that are not supported by any other claims (leaf claims)"""
    supported_by_ids = set()
    for claim in claims:
        supported_by_ids.update(claim.supported_by)
    
    return [claim for claim in claims if claim.id not in supported_by_ids]

def filter_claims_by_tags(claims: List[Claim], tags: List[str]) -> List[Claim]:
    """Filter claims by tags"""
    return [claim for claim in claims if any(tag in claim.tags for tag in tags)]

def filter_claims_by_confidence(claims: List[Claim], min_confidence: float, max_confidence: float = 1.0) -> List[Claim]:
    """Filter claims by confidence range"""
    return [claim for claim in claims if min_confidence <= claim.confidence <= max_confidence]

def create_claim(content: str, confidence: float = 0.5, **kwargs) -> Claim:
    """Create a new claim with auto-generated ID"""
    claim_id = generate_claim_id()
    return Claim(id=claim_id, content=content, confidence=confidence, **kwargs)

def generate_claim_id() -> str:
    """Generate a new claim ID"""
    import time
    import random
    timestamp = int(time.time() * 1000)
    random_suffix = random.randint(1000, 9999)
    return f"c{timestamp}{random_suffix}"

def validate_claim_id(claim_id: str) -> bool:
    """Validate claim ID format"""
    import re
    pattern = r'^c\d{13,}$'
    return re.match(pattern, claim_id) is not None

def validate_confidence(confidence: float) -> bool:
    """Validate confidence value"""
    return 0.0 <= confidence <= 1.0