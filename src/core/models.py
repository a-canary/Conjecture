"""
Core Pydantic models for Conjecture claims and data structures
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


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


class ClaimScope(str, Enum):
    """Claim scope enumeration"""

    USER_WORKSPACE = "user-{workspace}"  # User-specific, workspace-bound
    TEAM_WORKSPACE = "team-{workspace}"  # Team-specific, workspace-bound
    TEAM_WIDE = "team-wide"  # Team-wide across workspaces
    PUBLIC = "public"  # Publicly accessible


class DirtyReason(str, Enum):
    """Dirty flag reason enumeration"""

    NEW_CLAIM_ADDED = "new_claim_added"
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    SUPPORTING_CLAIM_CHANGED = "supporting_claim_changed"
    RELATIONSHIP_CHANGED = "relationship_changed"
    MANUAL_MARK = "manual_mark"
    BATCH_EVALUATION = "batch_evaluation"
    SYSTEM_TRIGGER = "system_trigger"
    RELATIONSHIP_CHANGE = "relationship_change"
    CONTENT_UPDATE = "content_update"
    CONFIDENCE_CHANGE = "confidence_change"
    STATE_CHANGE = "state_change"
    MANUAL_FLAG = "manual_flag"


class Claim(BaseModel):
    """Core claim model with validation"""

    model_config = ConfigDict(
        validate_assignment=True, extra="allow", frozen=False, protected_namespaces=()
    )

    id: str = Field(..., description="Unique claim identifier")
    content: str = Field(
        ..., min_length=5, max_length=1000, description="Claim content"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence level (0.0-1.0)"
    )
    state: ClaimState = Field(default=ClaimState.EXPLORE, description="Claim state")
    type: List[ClaimType] = Field(
        default_factory=lambda: [ClaimType.CONCEPT], description="Claim type(s)"
    )
    tags: List[str] = Field(default_factory=list, description="Claim tags")
    scope: ClaimScope = Field(
        default=ClaimScope.USER_WORKSPACE, description="Claim scope"
    )

    # Bidirectional relationship fields
    supports: List[str] = Field(
        default_factory=list,
        description="Claims this claim supports (upward relationships)",
    )
    supported_by: List[str] = Field(
        default_factory=list,
        description="Claims that support this claim (downward relationships)",
    )

    # Metadata fields
    created: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Claim creation time",
    )
    updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update time",
    )
    embedding: Optional[List[float]] = Field(
        default=None, description="Vector embedding for semantic search"
    )

    # Dirty flag fields for change tracking
    is_dirty: bool = Field(
        default=True, description="Whether claim has unsaved changes"
    )
    dirty: bool = Field(default=True, description="Backward compatibility flag")
    dirty_reason: Optional[DirtyReason] = Field(
        default=None, description="Reason for dirty flag"
    )
    dirty_timestamp: Optional[datetime] = Field(
        default=None, description="When claim was marked dirty"
    )
    dirty_priority: int = Field(
        default=0, description="Priority for dirty evaluation (higher = more urgent)"
    )

    @field_validator("content")
    @classmethod
    def validate_content(cls, v):
        """Validate claim content"""
        if not v or not v.strip():
            raise ValueError("Claim content cannot be empty")
        if len(v.strip()) < 5:
            raise ValueError("Claim content must be at least 5 characters long")
        return v.strip()

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v):
        """Validate claim tags"""
        if v and len(v) > 20:
            raise ValueError("Cannot have more than 20 tags")
        # Remove duplicates and empty strings
        seen = set()
        unique_tags = []
        for tag in v:
            tag_clean = tag.lower().strip()
            if tag_clean and tag_clean not in seen:
                seen.add(tag_clean)
                unique_tags.append(tag_clean)
        return unique_tags

    @model_validator(mode="after")
    def validate_relationships(self):
        """Validate bidirectional relationships"""
        # Check for self-references
        if self.id in self.supports or self.id in self.supported_by:
            raise ValueError("Claim cannot support itself")

        return self

    @model_validator(mode="after")
    def validate_timestamps(self):
        """Validate timestamp relationships"""
        # Normalize timestamps to handle naive vs aware comparison
        created = self.created
        updated = self.updated
        if created and updated:
            # Make both naive for comparison if mixed
            if created.tzinfo is not None and updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            elif created.tzinfo is None and updated.tzinfo is not None:
                created = created.replace(tzinfo=timezone.utc)
            if updated < created:
                raise ValueError("Updated time cannot be before created time")
        return self

    def to_chroma_metadata(self) -> Dict[str, Any]:
        """Convert claim to ChromaDB metadata format"""
        return {
            "id": self.id,
            "state": self.state.value,
            "type": [t.value for t in self.type],
            "tags": self.tags,
            "scope": self.scope.value,
            "confidence": self.confidence,
            "created": self.created.isoformat(),
            "supports": self.supports,
            "supported_by": self.supported_by,
            "is_dirty": self.is_dirty,
            "dirty_reason": self.dirty_reason.value if self.dirty_reason else None,
            "dirty_timestamp": self.dirty_timestamp.isoformat()
            if self.dirty_timestamp
            else None,
            "dirty_priority": self.dirty_priority,
        }

    def format_for_context(self) -> str:
        """Format claim for context inclusion"""
        return f"[c{self.id} | {self.content} | / {self.confidence}]"

    def format_for_output(self) -> str:
        """Format claim for output display"""
        return self.format_for_context()

    def format_for_llm_analysis(self) -> str:
        """Format claim for LLM analysis"""
        return f"""Claim ID: {self.id}
Content: {self.content}
Confidence: {self.confidence}
State: {self.state.value}
Type: {[t.value for t in self.type]}
Tags: {",".join(self.tags)}
Scope: {self.scope.value}
Supports: {self.supports}
Supported By: {self.supported_by}"""

    def __hash__(self) -> int:
        """Make Claim hashable for use in sets"""
        return hash((self.id, self.content, self.confidence))

    def __eq__(self, other) -> bool:
        """Claim equality based on hashable attributes"""
        if not isinstance(other, Claim):
            return False
        return hash(self) == hash(other)


# Backward compatibility alias for BasicClaim
# BasicClaim was old name for Claim before model consolidation
BasicClaim = Claim


class ClaimBatch(BaseModel):
    """Batch model for processing multiple claims"""

    claims: List[Claim] = Field(..., min_length=1, description="List of claims")
    batch_id: str = Field(..., description="Batch identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Batch timestamp",
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

    model_config = ConfigDict(
        validate_assignment=True, extra="forbid", frozen=False, protected_namespaces=()
    )

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

    model_config = ConfigDict(
        validate_assignment=True, extra="forbid", frozen=False, protected_namespaces=()
    )

    name: str = Field(..., description="Name of tool to invoke")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Tool parameters"
    )


class ExecutionResult(BaseModel):
    """Result of tool execution."""

    model_config = ConfigDict(
        validate_assignment=True, extra="forbid", frozen=False, protected_namespaces=()
    )

    success: bool = Field(..., description="Execution success status")
    outcome: str = Field(..., description="Execution outcome or error message")
    duration: float = Field(..., description="Execution duration in seconds")
    tool_name: str = Field(..., description="Name of tool that was executed")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class ParsedResponse(BaseModel):
    """Represents a parsed response from LLM"""

    model_config = ConfigDict(
        validate_assignment=True, extra="forbid", frozen=False, protected_namespaces=()
    )

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
    state: Optional[ClaimState] = Field(default=None, description="Filter by state")
    type: Optional[List[ClaimType]] = Field(
        default=None, description="Filter by claim type"
    )
    scope: Optional[ClaimScope] = Field(default=None, description="Filter by scope")
    created_after: Optional[datetime] = Field(
        default=None, description="Filter claims created after this date"
    )
    created_before: Optional[datetime] = Field(
        default=None, description="Filter claims created before this date"
    )
    limit: Optional[int] = Field(
        default=None, ge=1, le=1000, description="Maximum number of results"
    )


class Relationship(BaseModel):
    """Relationship between claims"""

    source_claim_id: str = Field(..., description="Source claim ID")
    target_claim_id: str = Field(..., description="Target claim ID")
    relationship_type: str = Field(..., description="Type of relationship")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in relationship"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional relationship metadata"
    )
    created: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Relationship creation time",
    )


# Utility functions for claim management
def create_claim(
    content: str,
    confidence: float = 0.5,
    claim_type: List[ClaimType] = None,
    tags: List[str] = None,
    scope: ClaimScope = ClaimScope.USER_WORKSPACE,
    claim_id: str = None,
) -> Claim:
    """Create a new claim with validation"""
    if claim_type is None:
        claim_type = [ClaimType.CONCEPT]

    if claim_id is None:
        claim_id = generate_claim_id()

    return Claim(
        id=claim_id,
        content=content,
        confidence=confidence,
        type=claim_type,
        tags=tags or [],
        scope=scope,
    )


def validate_claim_id(claim_id: str) -> bool:
    """Validate claim ID format"""
    if not claim_id or not isinstance(claim_id, str):
        return False

    # Check for valid claim ID pattern (alphanumeric with optional hyphens/underscores)
    import re

    pattern = r"^[a-zA-Z0-9_-]+$"
    return bool(re.match(pattern, claim_id))


def validate_confidence(confidence: float) -> bool:
    """Validate confidence value"""
    return isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0


def generate_claim_id() -> str:
    """Generate a unique claim ID"""
    import uuid
    import time

    timestamp = int(time.time() * 1000)
    unique_id = str(uuid.uuid4())[:8]
    return f"c{timestamp}_{unique_id}"


def get_orphaned_claims(claims: List[Claim]) -> List[Claim]:
    """Get claims that have no supporting relationships (orphans)"""
    return [
        claim
        for claim in claims
        if not claim.supported_by and claim.state != ClaimState.ORPHANED
    ]


def get_root_claims(claims: List[Claim]) -> List[Claim]:
    """Get root claims (claims that support others but are not supported by any)"""
    return [claim for claim in claims if claim.supports and not claim.supported_by]


def get_leaf_claims(claims: List[Claim]) -> List[Claim]:
    """Get leaf claims (claims that are supported but don't support others)"""
    return [claim for claim in claims if claim.supported_by and not claim.supports]


def filter_claims_by_tags(claims: List[Claim], tags: List[str]) -> List[Claim]:
    """Filter claims by matching tags"""
    if not tags:
        return claims
    tag_set = set(tag.lower() for tag in tags)
    return [
        claim for claim in claims if any(tag.lower() in tag_set for tag in claim.tags)
    ]


def filter_claims_by_confidence(
    claims: List[Claim], min_confidence: float, max_confidence: float = 1.0
) -> List[Claim]:
    """Filter claims by confidence range"""
    return [
        claim
        for claim in claims
        if min_confidence <= claim.confidence <= max_confidence
    ]


def create_claim(content: str, confidence: float = 0.5, **kwargs) -> Claim:
    """Create a new claim with validation"""
    claim_type = kwargs.get("claim_type", [ClaimType.CONCEPT])
    claim_scope = kwargs.get("scope", ClaimScope.USER_WORKSPACE)
    claim_tags = kwargs.get("tags", [])
    claim_id = kwargs.get("claim_id", None)

    if claim_id is None:
        claim_id = generate_claim_id()

    return Claim(
        id=claim_id,
        content=content,
        confidence=confidence,
        type=claim_type,
        tags=claim_tags,
        scope=claim_scope,
    )


def generate_claim_id() -> str:
    """Generate a unique claim ID"""
    import uuid
    import time

    timestamp = int(time.time() * 1000)
    unique_id = str(uuid.uuid4())[:8]
    return f"c{timestamp}_{unique_id}"


def validate_claim_id(claim_id: str) -> bool:
    """Validate claim ID format"""
    if not claim_id or not isinstance(claim_id, str):
        return False

    # Check for valid claim ID pattern (alphanumeric with optional hyphens/underscores)
    import re

    pattern = r"^[a-zA-Z0-9_-]+$"
    return bool(re.match(pattern, claim_id))


def validate_confidence(confidence: float) -> bool:
    """Validate confidence value"""
    return isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0
