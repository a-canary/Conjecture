"""
Core Pydantic models for Conjecture claims and data structures
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


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
    type: List[ClaimType] = Field(..., min_items=1, description="Claim types")
    tags: List[str] = Field(default_factory=list, description="Topic tags")
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
        default=False, description="Whether claim needs re-evaluation"
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

    @validator("tags")
    def validate_tags(cls, v):
        """Validate tags are strings and not empty"""
        if v:
            for tag in v:
                if not isinstance(tag, str) or not tag.strip():
                    raise ValueError("Tags must be non-empty strings")
        return v

    @validator("updated")
    def validate_updated_timestamp(cls, v, values):
        """Ensure updated timestamp is not before creation"""
        if "created" in values and v < values["created"]:
            raise ValueError("Updated timestamp cannot be before creation timestamp")
        return v

    def to_chroma_metadata(self) -> Dict[str, Any]:
        """Convert claim to ChromaDB metadata format"""
        return {
            "confidence": self.confidence,
            "state": self.state.value,
            "supported_by": self.supported_by,
            "supports": self.supports,
            "type": [t.value for t in self.type],
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
            type=[ClaimType(t) for t in metadata["type"]],
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
        """Format claim for LLM context"""
        type_str = ",".join([t.value for t in self.type])
        return f"- [{self.id},{self.confidence},{type_str},{self.state.value}]{self.content}"

    def __repr__(self) -> str:
        return f"Claim(id={self.id}, confidence={self.confidence}, state={self.state.value}, type={[t.value for t in self.type]}, dirty={self.is_dirty})"


class ClaimBatch(BaseModel):
    """Batch model for processing multiple claims"""

    claims: List[Claim] = Field(..., min_items=1, description="List of claims")
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


# Helper functions for working with Claim collections
def create_claim_index(claims: List[Claim]) -> Dict[str, Claim]:
    """Create an index mapping claim IDs to claim objects for efficient lookup"""
    return {claim.id: claim for claim in claims}


def get_orphaned_claims(claims: List[Claim]) -> List[Claim]:
    """Get all orphaned claims (no relationships)"""
    return [claim for claim in claims if not claim.supported_by and not claim.supports]


def get_root_claims(claims: List[Claim]) -> List[Claim]:
    """Get all root claims (support others but not supported)"""
    return [claim for claim in claims if claim.supports and not claim.supported_by]


def get_leaf_claims(claims: List[Claim]) -> List[Claim]:
    """Get all leaf claims (supported but don't support others)"""
    return [claim for claim in claims if claim.supported_by and not claim.supports]


def filter_claims_by_tags(claims: List[Claim], tags: List[str]) -> List[Claim]:
    """Filter claims by tag presence (claims must have at least one of the tags)"""
    if not tags:
        return claims

    tag_set = set(tags)
    return [claim for claim in claims if any(tag in tag_set for tag in claim.tags)]


def filter_claims_by_confidence(
    claims: List[Claim], min_confidence: float = 0.0, max_confidence: float = 1.0
) -> List[Claim]:
    """Filter claims by confidence range"""
    return [
        claim
        for claim in claims
        if min_confidence <= claim.confidence <= max_confidence
    ]


# Factory functions for common use cases
def create_instruction_claim(
    content: str,
    created_by: str = "system",
    confidence: float = 0.8,
    tags: Optional[List[str]] = None,
) -> Claim:
    """Create a claim for instruction/guidance content"""
    import uuid

    default_tags = ["instruction", "guidance"]
    claim_tags = tags if tags is not None else default_tags

    return Claim(
        id=f"instruction-{uuid.uuid4().hex[:8]}",
        content=content,
        confidence=confidence,
        type=[ClaimType.CONCEPT],
        tags=claim_tags,
    )


def create_concept_claim(
    content: str,
    created_by: str = "system",
    confidence: float = 0.7,
    tags: Optional[List[str]] = None,
) -> Claim:
    """Create a claim for conceptual content"""
    import uuid

    claim_tags = tags or ["concept"]

    return Claim(
        id=f"concept-{uuid.uuid4().hex[:8]}",
        content=content,
        confidence=confidence,
        type=[ClaimType.CONCEPT],
        tags=claim_tags,
    )


def create_evidence_claim(
    content: str,
    created_by: str = "system",
    confidence: float = 0.9,
    tags: Optional[List[str]] = None,
) -> Claim:
    """Create a claim for evidence/fact content"""
    import uuid

    claim_tags = tags or ["evidence", "fact"]

    return Claim(
        id=f"evidence-{uuid.uuid4().hex[:8]}",
        content=content,
        confidence=confidence,
        type=[ClaimType.REFERENCE],
        tags=claim_tags,
    )
