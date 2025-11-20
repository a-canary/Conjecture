"""
Basic claim models without external dependencies
Start simple, add complexity only when needed
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


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


class ClaimScope(str, Enum):
    """Claim scope enumeration (Req 1.2.4)"""

    SESSION = "Session"
    USER = "User"
    PROJECT = "Project"
    TEAM = "Team"
    GLOBAL = "Global"


class BasicClaim:
    """Simple claim class with basic validation"""

    def __init__(
        self,
        id: str,
        content: str,
        confidence: float,
        type: List[ClaimType],
        state: ClaimState = ClaimState.EXPLORE,
        scope: ClaimScope = ClaimScope.SESSION,
        tags: List[str] = None,
        supported_by: List[str] = None,
        supports: List[str] = None,
        created: datetime = None,
        updated: datetime = None,
    ):
        # Basic validation
        if not id or not isinstance(id, str):
            raise ValueError("ID must be a non-empty string")
        if not content or len(content.strip()) < 10:
            raise ValueError("Content must be at least 10 characters long")
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            raise ValueError("Confidence must be a float between 0.0 and 1.0")
        if not type or not isinstance(type, list) or len(type) == 0:
            raise ValueError("Type must be a non-empty list of ClaimType")

        self.id = id
        self.content = content.strip()
        self.confidence = float(confidence)
        self.type = type
        self.state = state
        self.scope = scope
        self.tags = tags or []
        self.supported_by = supported_by or []
        self.supports = supports or []
        self.created = created or datetime.utcnow()
        self.updated = updated or datetime.utcnow()

    def update_confidence(self, new_confidence: float) -> None:
        """Update confidence and timestamp"""
        if not (0.0 <= new_confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        self.confidence = float(new_confidence)
        self.updated = datetime.utcnow()

    def promote_scope(self, new_scope: ClaimScope) -> None:
        """Promote claim scope (Req 3.3)"""
        # Define promotion hierarchy order
        scope_order = {
            ClaimScope.SESSION: 0,
            ClaimScope.USER: 1,
            ClaimScope.PROJECT: 2,
            ClaimScope.TEAM: 3,
            ClaimScope.GLOBAL: 4,
        }

        current_level = scope_order.get(self.scope, 0)
        new_level = scope_order.get(new_scope, 0)

        if new_level > current_level:
            self.scope = new_scope
            self.updated = datetime.utcnow()
        else:
            raise ValueError(f"Cannot promote from {self.scope} to {new_scope}")

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

    def format_for_context(self) -> str:
        """Format claim for LLM context"""
        type_str = ",".join(
            [t.value if hasattr(t, "value") else str(t) for t in self.type]
        )
        state_value = (
            self.state.value if hasattr(self.state, "value") else str(self.state)
        )
        scope_value = (
            self.scope.value if hasattr(self.scope, "value") else str(self.scope)
        )
        return f"- [{self.id},{self.confidence},{type_str},{state_value},{scope_value}]{self.content}"

    def to_chroma_metadata(self) -> Dict[str, Any]:
        """Convert claim to ChromaDB metadata format"""
        return {
            "confidence": self.confidence,
            "state": self.state.value
            if hasattr(self.state, "value")
            else str(self.state),
            "scope": self.scope.value
            if hasattr(self.scope, "value")
            else str(self.scope),
            "supported_by": ",".join(self.supported_by) if self.supported_by else "",
            "supports": ",".join(self.supports) if self.supports else "",
            "type": ",".join(
                [t.value if hasattr(t, "value") else str(t) for t in self.type]
            ),
            "tags": ",".join(self.tags) if self.tags else "",
            "created": self.created.isoformat(),
            "updated": self.updated.isoformat(),
        }

    @classmethod
    def from_chroma_result(
        cls, id: str, content: str, metadata: Dict[str, Any]
    ) -> "BasicClaim":
        """Create claim from ChromaDB query result"""
        # Parse string metadata back to lists
        supported_by = (
            metadata.get("supported_by", "").split(",")
            if metadata.get("supported_by")
            else []
        )
        supports = (
            metadata.get("supports", "").split(",") if metadata.get("supports") else []
        )
        types = (
            [ClaimType(t) for t in metadata.get("type", "").split(",") if t]
            if metadata.get("type")
            else []
        )
        tags = metadata.get("tags", "").split(",") if metadata.get("tags") else []

        return cls(
            id=id,
            content=content,
            confidence=metadata["confidence"],
            type=types,
            state=ClaimState(metadata["state"]),
            scope=ClaimScope(metadata.get("scope", "Session")),
            tags=tags,
            supported_by=supported_by,
            supports=supports,
            created=datetime.fromisoformat(metadata["created"]),
            updated=datetime.fromisoformat(metadata["updated"]),
        )

    def __repr__(self) -> str:
        type_str = ",".join(
            [t.value if hasattr(t, "value") else str(t) for t in self.type]
        )
        state_value = (
            self.state.value if hasattr(self.state, "value") else str(self.state)
        )
        return f"BasicClaim(id={self.id}, confidence={self.confidence}, state={state_value}, scope={self.scope}, type={type_str})"


def validate_basic_models() -> bool:
    """Run basic validation tests"""
    try:
        # Test creation
        claim = BasicClaim(
            id="test_001",
            content="Quantum encryption uses photon polarization states",
            confidence=0.85,
            type=[ClaimType.CONCEPT],
            tags=["quantum", "encryption"],
        )
        print("✅ Basic claim creation: PASS")

        # Test validation
        try:
            BasicClaim(
                id="bad",
                content="Too short content",
                confidence=1.5,
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

        # Test scope promotion
        assert claim.scope == ClaimScope.SESSION
        claim.promote_scope(ClaimScope.PROJECT)
        assert claim.scope == ClaimScope.PROJECT
        try:
            claim.promote_scope(ClaimScope.SESSION)
            print("❌ Should have failed demotion")
            return False
        except ValueError:
            print("✅ Scope promotion rules: PASS")

        # Test metadata conversion
        metadata = claim.to_chroma_metadata()
        restored = BasicClaim.from_chroma_result(
            id="restored_001",
            content="Restored content",
            metadata=metadata,
        )
        assert restored.confidence == 0.95
        assert restored.scope == ClaimScope.PROJECT
        print("✅ Metadata conversion: PASS")

        return True
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 Testing Basic Models")
    print("=" * 30)
    if validate_basic_models():
        print("🎉 All basic model tests passed!")
    else:
        print("❌ Basic model tests failed")
