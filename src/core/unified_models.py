"""
Unified Claim Model - Single, elegant implementation
Combines the best features while eliminating complexity
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
    EXAMPLE = "example"
    GOAL = "goal"


class Claim(BaseModel):
    """
    Unified Claim Model - Single implementation for all Conjecture use cases
    Provides comprehensive functionality with elegant simplicity
    """

    id: str = Field(..., description="Unique claim identifier")
    content: str = Field(
        ..., min_length=10, max_length=2000, description="Claim content"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score (0.0-1.0)"
    )
    type: List[ClaimType] = Field(..., min_items=1, description="Claim types")

    # Optional fields with sensible defaults
    state: ClaimState = Field(
        default=ClaimState.EXPLORE, description="Current claim state"
    )
    tags: List[str] = Field(default_factory=list, description="Topic tags")
    supported_by: List[str] = Field(
        default_factory=list, description="Claims that support this claim"
    )
    supports: List[str] = Field(
        default_factory=list, description="Claims this claim supports"
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

    @validator("content")
    def validate_content_length(cls, v):
        """Ensure content is meaningful"""
        if len(v.strip()) < 10:
            raise ValueError("Content must be at least 10 characters long")
        return v.strip()

    @validator("tags")
    def validate_tags(cls, v):
        """Validate tags are non-empty strings"""
        if v:
            for tag in v:
                if not isinstance(tag, str) or not tag.strip():
                    raise ValueError("Tags must be non-empty strings")
        return [tag.strip() for tag in v if tag.strip()]

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
            "supported_by": ",".join(self.supported_by) if self.supported_by else "",
            "supports": ",".join(self.supports) if self.supports else "",
            "type": ",".join([t.value for t in self.type]),
            "tags": ",".join(self.tags) if self.tags else "",
            "created": self.created.isoformat(),
            "updated": self.updated.isoformat(),
        }

    @classmethod
    def from_chroma_result(
        cls, id: str, content: str, metadata: Dict[str, Any]
    ) -> "Claim":
        """Create claim from ChromaDB query result"""
        # Parse string metadata back to lists (for compatibility)
        supported_by = (
            [s for s in metadata.get("supported_by", "").split(",") if s]
            if metadata.get("supported_by")
            else []
        )
        supports = (
            [s for s in metadata.get("supports", "").split(",") if s]
            if metadata.get("supports")
            else []
        )
        types = (
            [ClaimType(t) for t in metadata.get("type", "").split(",") if t]
            if metadata.get("type")
            else []
        )
        tags = (
            [t for t in metadata.get("tags", "").split(",") if t]
            if metadata.get("tags")
            else []
        )

        return cls(
            id=id,
            content=content,
            confidence=metadata["confidence"],
            type=types,
            state=ClaimState(metadata["state"]),
            tags=tags,
            supported_by=supported_by,
            supports=supports,
            created=datetime.fromisoformat(metadata["created"]),
            updated=datetime.fromisoformat(metadata["updated"]),
        )

    def format_for_context(self) -> str:
        """Format claim for LLM context"""
        type_str = ",".join([t.value for t in self.type])
        return f"- [{self.id},{self.confidence},{type_str},{self.state.value}]{self.content}"

    def update_confidence(self, new_confidence: float) -> None:
        """Update confidence and timestamp"""
        if not (0.0 <= new_confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        self.confidence = new_confidence
        self.updated = datetime.utcnow()

    def add_support(self, supporting_claim_id: str) -> None:
        """Add a supporting claim ID"""
        if supporting_claim_id and supporting_claim_id not in self.supported_by:
            self.supported_by.append(supporting_claim_id)
            self.updated = datetime.utcnow()

    def add_supports(self, supported_claim_id: str) -> None:
        """Add a claim this claim supports"""
        if supported_claim_id and supported_claim_id not in self.supports:
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


# Simplified tool execution models (moved from skill_models)
class ToolCall(BaseModel):
    """Represents a tool invocation."""
    name: str = Field(..., description="Name of the tool to invoke")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")


class ExecutionResult(BaseModel):
    """Result of tool execution."""
    success: bool = Field(..., description="Execution success status")
    outcome: str = Field(..., description="Execution outcome or error message")
    duration: float = Field(..., description="Execution duration in seconds")
    tool_name: str = Field(..., description="Name of the tool that was executed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ParsedResponse(BaseModel):
    """Represents a parsed response from LLM"""
    tool_calls: List[ToolCall] = Field(default_factory=list, description="Extracted tool calls")
    errors: List[str] = Field(default_factory=list, description="Parsing errors")


def validate_unified_models() -> bool:
    """Validate the unified model implementation"""
    try:
        # Test creation with all features
        claim = Claim(
            id="test_001",
            content="Quantum encryption uses photon polarization states for secure communication",
            confidence=0.85,
            type=[ClaimType.CONCEPT],
            tags=["quantum", "encryption"],
        )
        print("✅ Unified claim creation: PASS")

        # Test validation
        try:
            Claim(
                id="bad",
                content="Too short",  # Less than 10 characters
                confidence=1.5,  # Out of range
                type=[ClaimType.CONCEPT],
            )
            print("❌ Should have failed validation")
            return False
        except ValueError:
            print("✅ Validation rules: PASS")

        # Test relationships
        claim.add_support("support_001")
        claim.add_supports("supported_001")
        assert "support_001" in claim.supported_by
        assert "supported_001" in claim.supports
        print("✅ Relationships: PASS")

        # Test confidence update
        claim.update_confidence(0.95)
        assert claim.confidence == 0.95
        print("✅ Confidence update: PASS")

        # Test metadata conversion and restoration
        metadata = claim.to_chroma_metadata()
        restored = Claim.from_chroma_result(
            id="restored_001",
            content=claim.content,
            metadata=metadata,
        )
        assert restored.confidence == 0.95
        assert restored.content == claim.content
        print("✅ Metadata conversion: PASS")

        # Test batch processing
        batch = ClaimBatch(claims=[claim], batch_id="batch_001")
        ids, documents, embeddings, metadatas = batch.to_chroma_batch()
        assert len(ids) == 1
        assert documents[0] == claim.content
        print("✅ Batch processing: PASS")

        return True
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing Unified Models")
    print("=" * 30)
    if validate_unified_models():
        print("🎉 All unified model tests passed!")
    else:
        print("❌ Unified model tests failed")
