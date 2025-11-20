"""
Simplified Data Management for Conjecture
Basic file-based claim storage and retrieval
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .core.models import Claim, ClaimType


class DataManager:
    """Simplified data management with file-based storage"""

    def __init__(self, data_path: str = "data/claims.json"):
        self.data_path = data_path
        self.data_dir = Path(data_path).parent
        self.data_dir.mkdir(exist_ok=True)
        self.claims: Dict[str, Claim] = {}
        self._load_data()

    def _load_data(self):
        """Load claims from file or start with empty data"""
        try:
            if os.path.exists(self.data_path):
                with open(self.data_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for claim_data in data:
                        claim = Claim(**claim_data)
                        self.claims[claim.id] = claim
                print(f"📁 Loaded {len(self.claims)} claims from {self.data_path}")
            else:
                print(f"📁 No existing data found, starting fresh at {self.data_path}")
        except Exception as e:
            print(f"⚠️  Error loading data: {e}. Starting fresh.")
            self.claims = {}

    def _save_data(self):
        """Save claims to file"""
        try:
            data = [claim.dict() for claim in self.claims.values()]
            with open(self.data_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"❌ Error saving data: {e}")

    def create_claim(
        self,
        content: str,
        confidence: float,
        claim_type: str,
        tags: Optional[List[str]] = None,
        **kwargs,
    ) -> Claim:
        """Create and store a new claim"""
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

        claim_id = f"claim_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(content)}"
        claim = Claim(
            id=claim_id,
            content=content.strip(),
            confidence=confidence,
            type=[claim_type_enum],
            tags=tags or [],
            **kwargs,
        )

        self.claims[claim_id] = claim
        self._save_data()
        return claim

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        """Get claim by ID"""
        return self.claims.get(claim_id)

    def get_recent_claims(self, limit: int = 10) -> List[Claim]:
        """Get most recent claims"""
        recent_claims = sorted(
            self.claims.values(), key=lambda c: c.created, reverse=True
        )
        return recent_claims[:limit]

    def search_claims(
        self,
        query: str,
        confidence_threshold: Optional[float] = None,
        claim_types: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Claim]:
        """Simple text search for claims"""
        query_lower = query.lower()
        matching_claims = []

        for claim in self.claims.values():
            # Check confidence threshold
            if confidence_threshold and claim.confidence < confidence_threshold:
                continue

            # Check claim types
            if claim_types:
                claim_type_strs = [t.value for t in claim.type]
                if not any(ct.lower() in claim_type_strs for ct in claim_types):
                    continue

            # Simple text matching
            if query_lower in claim.content.lower() or any(
                query_lower in tag.lower() for tag in claim.tags
            ):
                matching_claims.append(claim)

        return matching_claims[:limit]

    def update_claim(self, claim_id: str, **updates) -> Optional[Claim]:
        """Update claim fields"""
        if claim_id not in self.claims:
            return None

        claim = self.claims[claim_id]

        # Update allowed fields
        allowed_fields = ["content", "confidence", "tags", "supported_by", "supports"]
        for field, value in updates.items():
            if field in allowed_fields and hasattr(claim, field):
                setattr(claim, field, value)

        claim.updated = datetime.utcnow()
        self._save_data()
        return claim

    def delete_claim(self, claim_id: str) -> bool:
        """Delete a claim"""
        if claim_id in self.claims:
            del self.claims[claim_id]
            self._save_data()
            return True
        return False

    def get_statistics(self) -> Dict:
        """Get basic statistics"""
        if not self.claims:
            return {
                "total_claims": 0,
                "avg_confidence": 0.0,
                "claim_types": {},
                "oldest_claim": None,
                "newest_claim": None,
            }

        total_claims = len(self.claims)
        avg_confidence = sum(c.confidence for c in self.claims.values()) / total_claims

        # Count claim types
        type_counts = {}
        for claim in self.claims.values():
            for claim_type in claim.type:
                type_counts[claim_type.value] = type_counts.get(claim_type.value, 0) + 1

        # Find oldest and newest
        sorted_claims = sorted(self.claims.values(), key=lambda c: c.created)

        return {
            "total_claims": total_claims,
            "avg_confidence": avg_confidence,
            "claim_types": type_counts,
            "oldest_claim": sorted_claims[0].created.isoformat(),
            "newest_claim": sorted_claims[-1].created.isoformat(),
        }

    def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """Remove claims older than specified days"""
        cutoff_date = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        old_claims = [
            claim_id
            for claim_id, claim in self.claims.items()
            if claim.created.timestamp() < cutoff_date
        ]

        for claim_id in old_claims:
            del self.claims[claim_id]

        if old_claims:
            self._save_data()

        return len(old_claims)


# Global data manager instance
_data_manager = None


def get_data_manager() -> DataManager:
    """Get the global data manager instance"""
    global _data_manager
    if _data_manager is None:
        _data_manager = DataManager()
    return _data_manager


def create_claim(content: str, confidence: float, claim_type: str, **kwargs) -> Claim:
    """Convenience function to create a claim"""
    dm = get_data_manager()
    return dm.create_claim(content, confidence, claim_type, **kwargs)


def get_recent_claims(limit: int = 10) -> List[Claim]:
    """Convenience function to get recent claims"""
    dm = get_data_manager()
    return dm.get_recent_claims(limit)


def search_claims(query: str, **kwargs) -> List[Claim]:
    """Convenience function to search claims"""
    dm = get_data_manager()
    return dm.search_claims(query, **kwargs)


if __name__ == "__main__":
    print("🧪 Testing Data Manager")
    print("=" * 30)

    dm = DataManager("test_claims.json")

    # Test claim creation
    claim1 = dm.create_claim(
        content="Machine learning requires large datasets for training",
        confidence=0.85,
        claim_type="concept",
        tags=["ml", "data"],
    )
    print(f"✅ Created: {claim1}")

    claim2 = dm.create_claim(
        content="Python is the most popular language for machine learning",
        confidence=0.90,
        claim_type="reference",
        tags=["python", "ml"],
    )
    print(f"✅ Created: {claim2}")

    # Test search
    results = dm.search_claims("machine learning")
    print(f"✅ Search results: {len(results)} claims")

    # Test recent claims
    recent = dm.get_recent_claims(5)
    print(f"✅ Recent claims: {len(recent)} claims")

    # Test statistics
    stats = dm.get_statistics()
    print(f"✅ Statistics: {stats}")

    print("🎉 Data Manager tests passed!")
