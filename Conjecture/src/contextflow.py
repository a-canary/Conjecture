"""
Conjecture: Simple API Interface
Provides elegant, unified access to all functionality
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from config.simple_config import Config
from core.unified_models import Claim, ClaimState, ClaimType


class Conjecture:
    """
    Simple, elegant API for Conjecture
    Single interface for all evidence-based AI reasoning
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize with optional custom configuration"""
        self.config = config or Config()

        # Initialize backend (will be implemented based on database_type)
        self._initialize_backend()

        print(f"🚀 Conjecture initialized: {self.config}")

    def _initialize_backend(self):
        """Initialize appropriate backend based on configuration"""
        if self.config.database_type == "chroma":
            print("📊 Using ChromaDB backend")
            # TODO: Initialize ChromaDB backend
        elif self.config.database_type == "file":
            print("💾 Using file-based backend")
            # TODO: Initialize file-based backend
        else:
            print("🎭 Using mock backend")
            # TODO: Initialize mock backend

    def explore(
        self,
        query: str,
        max_claims: int = 10,
        claim_types: Optional[List[str]] = None,
        confidence_threshold: Optional[float] = None,
    ) -> "ExplorationResult":
        """
        Explore knowledge base for claims related to query

        Args:
            query: Your question or topic to explore
            max_claims: Maximum number of claims to return
            claim_types: Specific claim types to include
            confidence_threshold: Minimum confidence level

        Returns:
            ExplorationResult: Comprehensive results with claims and insights
        """
        if not query or len(query.strip()) < 5:
            raise ValueError("Query must be at least 5 characters long")

        max_claims = max(1, min(max_claims, 50))  # Clamp between 1-50
        confidence_threshold = confidence_threshold or self.config.confidence_threshold

        print(f"🔍 Exploring: '{query}'")
        print(f"   Max claims: {max_claims}, Confidence: {confidence_threshold}")

        # TODO: Implement actual search logic
        # For now, return mock results
        mock_claims = self._generate_mock_claims(query, max_claims, claim_types)
        filtered_claims = [
            c for c in mock_claims if c.confidence >= confidence_threshold
        ]

        result = ExplorationResult(
            query=query,
            claims=filtered_claims[:max_claims],
            total_found=len(filtered_claims),
            search_time=0.1,  # Mock timing
            confidence_threshold=confidence_threshold,
            max_claims=max_claims,
        )

        print(f"✅ Found {len(result.claims)} claims in {result.search_time:.2f}s")
        return result

    def add_claim(
        self,
        content: str,
        confidence: float,
        claim_type: str,
        tags: Optional[List[str]] = None,
        **kwargs,
    ) -> Claim:
        """
        Add a new claim to the knowledge base

        Args:
            content: Claim content (minimum 10 characters)
            confidence: Confidence score (0.0-1.0)
            claim_type: Type of claim ("concept", "reference", etc.)
            tags: Optional topic tags
            **kwargs: Additional claim attributes

        Returns:
            Claim: The created claim with generated ID
        """
        # Validate inputs
        if len(content.strip()) < 10:
            raise ValueError("Content must be at least 10 characters long")
        if not (0.0 <= confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")

        try:
            claim_type_enum = ClaimType(claim_type.lower())
        except ValueError:
            valid_types = [t.value for t in ClaimType]
            raise ValueError(
                f"Invalid claim type: {claim_type}. Valid types: {valid_types}"
            )

        # Create claim
        claim = Claim(
            id=f"claim_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(content)}",
            content=content.strip(),
            confidence=confidence,
            type=[claim_type_enum],
            tags=tags or [],
            **kwargs,
        )

        print(f"✅ Created claim: {claim}")

        # TODO: Store claim in backend
        return claim

    def _generate_mock_claims(
        self, query: str, max_claims: int, claim_types: Optional[List[str]]
    ) -> List[Claim]:
        """Generate mock claims for demonstration"""
        mock_claims = [
            Claim(
                id="mock_001",
                content=f"{query} requires understanding of fundamental concepts and principles",
                confidence=0.85,
                type=[ClaimType.CONCEPT],
                tags=[
                    "fundamental",
                    query.lower().split()[0] if query.split() else "topic",
                ],
            ),
            Claim(
                id="mock_002",
                content=f"Research on {query} shows significant progress in recent years",
                confidence=0.92,
                type=[ClaimType.REFERENCE],
                tags=["research", "progress"],
            ),
            Claim(
                id="mock_003",
                content=f"Mastering {query} involves developing specific skills and competencies",
                confidence=0.78,
                type=[ClaimType.SKILL],
                tags=["mastery", "skills"],
            ),
            Claim(
                id="mock_004",
                content=f"The goal of studying {query} is to achieve comprehensive understanding",
                confidence=0.88,
                type=[ClaimType.GOAL],
                tags=["understanding", "comprehensive"],
            ),
            Claim(
                id="mock_005",
                content=f"An example of {query} can be found in practical applications",
                confidence=0.75,
                type=[ClaimType.EXAMPLE],
                tags=["example", "practical"],
            ),
            Claim(
                id="mock_006",
                content=f"The thesis regarding {query} suggests multiple perspectives exist",
                confidence=0.82,
                type=[ClaimType.THESIS],
                tags=["thesis", "perspectives"],
            ),
        ]

        # Filter by claim types if specified
        if claim_types:
            filtered_types = [t.lower() for t in claim_types]
            mock_claims = [
                c for c in mock_claims if any(t.value in filtered_types for t in c.type)
            ]

        return mock_claims[:max_claims]

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics and health metrics"""
        return {
            "config": self.config.to_dict(),
            "database_type": self.config.database_type,
            "database_path": self.config.database_path,
            "llm_enabled": self.config.llm_enabled,
            "claims_count": 0,  # TODO: Get actual count
            "system_healthy": True,
            "uptime": "N/A",  # TODO: Track actual uptime
        }


class ExplorationResult:
    """
    Result of an exploration query
    Provides comprehensive results with claims and insights
    """

    def __init__(
        self,
        query: str,
        claims: List[Claim],
        total_found: int,
        search_time: float,
        confidence_threshold: float,
        max_claims: int,
    ):
        self.query = query
        self.claims = claims
        self.total_found = total_found
        self.search_time = search_time
        self.confidence_threshold = confidence_threshold
        self.max_claims = max_claims
        self.timestamp = datetime.utcnow()

    def __str__(self) -> str:
        return f"ExplorationResult(query='{self.query}', claims={len(self.claims)}, time={self.search_time:.2f}s)"

    def summary(self) -> str:
        """Provide a human-readable summary of results"""
        if not self.claims:
            return f"No claims found for '{self.query}' above confidence threshold {self.confidence_threshold}"

        lines = [
            f"🎯 Explored: '{self.query}'",
            f"📊 Found: {len(self.claims)} claims (of {self.total_found} total)",
            f"⏱️  Time: {self.search_time:.2f}s",
            f"🎚️  Confidence: ≥{self.confidence_threshold}",
            "",
            "📋 Top Claims:",
        ]

        for i, claim in enumerate(self.claims[:5], 1):  # Show top 5
            type_str = ",".join([t.value for t in claim.type])
            lines.append(
                f"  {i}. [{claim.confidence:.2f}, {type_str}] {claim.content[:100]}{'...' if len(claim.content) > 100 else ''}"
            )

        if len(self.claims) > 5:
            lines.append(f"  ... and {len(self.claims) - 5} more claims")

        return "\n".join(lines)


# Convenience functions for immediate use
def explore(query: str, max_claims: int = 10, **kwargs) -> ExplorationResult:
    """
    Quick exploration function - no setup required
    """
    cf = Conjecture()
    return cf.explore(query, max_claims, **kwargs)


def add_claim(content: str, confidence: float, claim_type: str, **kwargs) -> Claim:
    """
    Quick add claim function - no setup required
    """
    cf = Conjecture()
    return cf.add_claim(content, confidence, claim_type, **kwargs)


if __name__ == "__main__":
    print("🧪 Testing Conjecture API")
    print("=" * 30)

    # Test initialization
    cf = Conjecture()
    print(f"✅ Conjecture initialized: {cf.config}")

    # Test exploration
    print("\n🔍 Testing exploration...")
    result = cf.explore("machine learning", max_claims=3)
    print(result.summary())

    # Test adding claim
    print("\n➕ Testing claim creation...")
    claim = cf.add_claim(
        content="Machine learning algorithms require substantial training data to achieve optimal performance",
        confidence=0.87,
        claim_type="concept",
        tags=["machine learning", "algorithms", "performance"],
    )
    print(f"✅ Created: {claim}")

    # Test statistics
    print("\n📊 Testing statistics...")
    stats = cf.get_statistics()
    print(f"System stats: {stats}")

    print("\n🎉 Conjecture API test completed!")
