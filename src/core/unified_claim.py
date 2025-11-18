"""
Simplified Unified Claim Model for the Simplified Universal Claim Architecture
Implements a single universal Claim structure with no enhanced fields
"""

from datetime import datetime
from typing import List, Dict, Any, Set, Optional
from pydantic import BaseModel, Field, validator


class UnifiedClaim(BaseModel):
    """
    Simplified Universal Claim Model
    
    This is the single, universal claim structure that supports all functionality.
    No enhanced fields - keeps the architecture simple and maintainable.
    """
    
    # Core identity and content
    id: str = Field(..., description="Unique claim identifier")
    content: str = Field(..., min_length=5, max_length=2000, description="Claim content")
    
    # Confidence and metadata
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    tags: List[str] = Field(default_factory=list, description="Topic tags")
    
    # Support relationships (bidirectional)
    supported_by: List[str] = Field(default_factory=list, description="Claims that support this claim")
    supports: List[str] = Field(default_factory=list, description="Claims this claim supports")
    
    # Audit fields
    created_by: str = Field(..., description="Creator identifier")
    created: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")

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

    def format_for_context(self) -> str:
        """Format claim for LLM context"""
        tags_str = ",".join(self.tags) if self.tags else "no-tags"
        return f"[{self.id}][{self.confidence:.2f}][{tags_str}] {self.content}"

    def format_for_llm_analysis(self) -> str:
        """Format claim for LLM analysis with relationships"""
        supported_by_str = ",".join(self.supported_by) if self.supported_by else "none"
        supports_str = ",".join(self.supports) if self.supports else "none"
        tags_str = ",".join(self.tags) if self.tags else "no-tags"
        
        return (f"Claim ID: {self.id}\n"
                f"Content: {self.content}\n"
                f"Confidence: {self.confidence:.2f}\n"
                f"Tags: {tags_str}\n"
                f"Supported by: {supported_by_str}\n"
                f"Supports: {supports_str}\n"
                f"Created: {self.created.isoformat()}\n")

    # Relationship traversal methods
    def get_supporting_ids(self) -> Set[str]:
        """Get set of claim IDs that support this claim"""
        return set(self.supported_by)

    def get_supported_ids(self) -> Set[str]:
        """Get set of claim IDs this claim supports"""
        return set(self.supports)

    def has_support_relationships(self) -> bool:
        """Check if claim has any support relationships"""
        return bool(self.supported_by or self.supports)

    def is_root_claim(self) -> bool:
        """Check if this is a root claim (supports others but not supported)"""
        return bool(self.supports) and not self.supported_by

    def is_leaf_claim(self) -> bool:
        """Check if this is a leaf claim (supported but doesn't support others)"""
        return bool(self.supported_by) and not self.supports

    def is_orphaned(self) -> bool:
        """Check if claim has no relationships"""
        return not self.supported_by and not self.supports

    # Compatibility methods for existing code
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "content": self.content,
            "confidence": self.confidence,
            "tags": self.tags,
            "supported_by": self.supported_by,
            "supports": self.supports,
            "created_by": self.created_by,
            "created": self.created.isoformat(),
            "updated": self.updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedClaim":
        """Create from dictionary representation"""
        if "created" in data and isinstance(data["created"], str):
            data["created"] = datetime.fromisoformat(data["created"])
        if "updated" in data and isinstance(data["updated"], str):
            data["updated"] = datetime.fromisoformat(data["updated"])
        return cls(**data)

    def __repr__(self) -> str:
        return f"UnifiedClaim(id={self.id}, confidence={self.confidence:.2f}, supports={len(self.supports)}, supported_by={len(self.supported_by)})"


# Helper functions for working with UnifiedClaim collections
def create_claim_index(claims: List[UnifiedClaim]) -> Dict[str, UnifiedClaim]:
    """Create an index mapping claim IDs to claim objects for efficient lookup"""
    return {claim.id: claim for claim in claims}


def get_orphaned_claims(claims: List[UnifiedClaim]) -> List[UnifiedClaim]:
    """Get all orphaned claims (no relationships)"""
    return [claim for claim in claims if claim.is_orphaned()]


def get_root_claims(claims: List[UnifiedClaim]) -> List[UnifiedClaim]:
    """Get all root claims (support others but not supported)"""
    return [claim for claim in claims if claim.is_root_claim()]


def get_leaf_claims(claims: List[UnifiedClaim]) -> List[UnifiedClaim]:
    """Get all leaf claims (supported but don't support others)"""
    return [claim for claim in claims if claim.is_leaf_claim()]


def filter_claims_by_tags(claims: List[UnifiedClaim], tags: List[str]) -> List[UnifiedClaim]:
    """Filter claims by tag presence (claims must have at least one of the tags)"""
    if not tags:
        return claims
    
    tag_set = set(tags)
    return [
        claim for claim in claims
        if any(tag in tag_set for tag in claim.tags)
    ]


def filter_claims_by_confidence(claims: List[UnifiedClaim], min_confidence: float = 0.0, max_confidence: float = 1.0) -> List[UnifiedClaim]:
    """Filter claims by confidence range"""
    return [
        claim for claim in claims
        if min_confidence <= claim.confidence <= max_confidence
    ]


# Simplified factory functions for common use cases
def create_instruction_claim(
    content: str, 
    created_by: str = "system",
    confidence: float = 0.8,
    tags: Optional[List[str]] = None
) -> UnifiedClaim:
    """Create a claim for instruction/guidance content"""
    import uuid
    default_tags = ["instruction", "guidance"]
    claim_tags = tags if tags is not None else default_tags
    
    return UnifiedClaim(
        id=f"instruction-{uuid.uuid4().hex[:8]}",
        content=content,
        confidence=confidence,
        tags=claim_tags,
        created_by=created_by
    )


def create_concept_claim(
    content: str, 
    created_by: str = "system",
    confidence: float = 0.7,
    tags: Optional[List[str]] = None
) -> UnifiedClaim:
    """Create a claim for conceptual content"""
    import uuid
    claim_tags = tags or ["concept"]
    
    return UnifiedClaim(
        id=f"concept-{uuid.uuid4().hex[:8]}",
        content=content,
        confidence=confidence,
        tags=claim_tags,
        created_by=created_by
    )


def create_evidence_claim(
    content: str, 
    created_by: str = "system",
    confidence: float = 0.9,
    tags: Optional[List[str]] = None
) -> UnifiedClaim:
    """Create a claim for evidence/fact content"""
    import uuid
    claim_tags = tags or ["evidence", "fact"]
    
    return UnifiedClaim(
        id=f"evidence-{uuid.uuid4().hex[:8]}",
        content=content,
        confidence=confidence,
        tags=claim_tags,
        created_by=created_by
    )