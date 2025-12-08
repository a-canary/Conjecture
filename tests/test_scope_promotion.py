import unittest
import sys
from pathlib import Path

# Ensure src is in python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from src.core.models import BasicClaim, ClaimScope, ClaimType, ClaimState


class TestScopePromotion(unittest.TestCase):
    """
    Validates Requirement 3.3: Scope Elevation
    """

    def test_scope_promotion_lifecycle(self):
        # Create a session scoped claim (default)
        claim = BasicClaim(
            id="scope_test_001",
            content="Initial session claim content that is long enough",
            confidence=0.5,
            type=[ClaimType.CONCEPT],
        )

        self.assertEqual(claim.scope, ClaimScope.SESSION)

        # Promote to User
        claim.promote_scope(ClaimScope.USER)
        self.assertEqual(claim.scope, ClaimScope.USER)

        # Promote to Project
        claim.promote_scope(ClaimScope.PROJECT)
        self.assertEqual(claim.scope, ClaimScope.PROJECT)

        # Verify demotion fails
        with self.assertRaises(ValueError):
            claim.promote_scope(ClaimScope.SESSION)

        # Promote to Global
        claim.promote_scope(ClaimScope.GLOBAL)
        self.assertEqual(claim.scope, ClaimScope.GLOBAL)

    def test_scope_persistence(self):
        # Verify scope survives serialization (to metadata)
        claim = BasicClaim(
            id="scope_persist_001",
            content="Persistent scope claim content",
            confidence=0.8,
            type=[ClaimType.THESIS],
            scope=ClaimScope.TEAM,
        )

        metadata = claim.to_chroma_metadata()
        self.assertEqual(metadata["scope"], "Team")

        # Deserialize
        restored = BasicClaim.from_chroma_result(
            id="restored_id", content="Content ignored", metadata=metadata
        )

        self.assertEqual(restored.scope, ClaimScope.TEAM)


if __name__ == "__main__":
    unittest.main()
