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
        )

    def format_for_context(self) -> str:
        """Format claim for LLM context"""
        type_str = ",".join([t.value for t in self.type])
        return f"- [{self.id},{self.confidence},{type_str},{self.state.value}]{self.content}"

    def update_confidence(self, new_confidence: float) -> None:
        """Update confidence and timestamp"""
        if not 0.0 <= new_confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        self.confidence = new_confidence
        self.updated = datetime.utcnow()

    def add_support(self, supporting_claim_id: str) -> None:
        """Add a supporting claim ID"""
        if supporting_claim_id not in self.supported_by:
            self.supported_by.append(supporting_claim_id)
            self.updated = datetime.utcnow()

    def add_supports(self, supported_claim_id: str) -> None:
        """Add a claim this claim supports"""
        if supported_claim_id not in self.supports:
            self.supports.append(supported_claim_id)
            self.updated = datetime.utcnow()

    def __repr__(self) -> str:
        return f"Claim(id={self.id}, confidence={self.confidence}, state={self.state.value}, type={[t.value for t in self.type]})"


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
