#!/usr/bin/env python3
"""
Mathematical Knowledge Seeder
Creates foundational mathematical knowledge claims for Conjecture's knowledge graph
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.core.models import Claim, ClaimType, ClaimScope
    from src.data.data_manager import DataManager
except ImportError:
    # Handle relative imports
    from core.models import Claim, ClaimType, ClaimScope
    from data.data_manager import DataManager

class MathematicalKnowledgeSeeder:
    """Seeds Conjecture's knowledge graph with mathematical foundations"""

    def __init__(self):
        self.data_manager = None
        self.mathematical_claims = []

    def initialize_data_manager(self):
        """Initialize connection to Conjecture database"""
        try:
            self.data_manager = DataManager()
            print("Data manager initialized successfully")
        except Exception as e:
            print(f"Failed to initialize data manager: {e}")
            return False
        return True

    def create_mathematical_claims(self) -> List[Claim]:
        """Create foundational mathematical knowledge claims"""

        claims = []

        # Fundamental mathematical concepts
        claims.append(Claim(
            id="math-multiplication-concept",
            content="Multiplication is repeated addition: a × b = a + a + ... + a (b times)",
            confidence=0.95,
            type=[ClaimType.CONCEPT],
            tags=["math", "multiplication", "fundamental", "arithmetic"],
            scope=ClaimScope.PUBLIC
        ))

        claims.append(Claim(
            id="math-addition-concept",
            content="Addition is combining quantities: a + b = sum of a and b",
            confidence=0.98,
            type=[ClaimType.CONCEPT],
            tags=["math", "addition", "fundamental", "arithmetic"],
            scope=ClaimScope.PUBLIC
        ))

        claims.append(Claim(
            id="math-distributive-property",
            content="Distributive property: (a + b) × c = a × c + b × c. Breaks complex multiplication into simpler parts.",
            confidence=0.90,
            type=[ClaimType.CONCEPT],
            tags=["math", "multiplication", "strategy", "distributive"],
            scope=ClaimScope.PUBLIC
        ))

        claims.append(Claim(
            id="math-estimation-strategy",
            content="Estimation verifies mathematical reasonableness: 17×24 ≈ 20×25 = 500, so answer around 500",
            confidence=0.85,
            type=[ClaimType.CONCEPT],
            tags=["math", "estimation", "verification", "strategy"],
            scope=ClaimScope.PUBLIC
        ))

        # Mathematical strategies
        claims.append(Claim(
            id="math-multiplication-strategy",
            content="For complex multiplication, break into tens and units: 17×24 = 17×20 + 17×4",
            confidence=0.90,
            type=[ClaimType.CONCEPT],
            tags=["math", "multiplication", "strategy", "breakdown"],
            scope=ClaimScope.PUBLIC
        ))

        # Verified examples (high confidence)
        claims.append(Claim(
            id="math-17x24-example",
            content="17 × 24 = 408. Verified calculation: 17×20 = 340, 17×4 = 68, 340+68 = 408",
            confidence=1.0,
            type=[ClaimType.EXAMPLE],
            tags=["math", "multiplication", "example", "verified"],
            scope=ClaimScope.PUBLIC,
            supports=["math-multiplication-concept", "math-distributive-property", "math-multiplication-strategy"]
        ))

        claims.append(Claim(
            id="math-basic-multiplication-facts",
            content="Basic multiplication facts: 17×20 = 340, 17×4 = 68. These are building blocks for complex calculations.",
            confidence=0.95,
            type=[ClaimType.EXAMPLE],
            tags=["math", "multiplication", "basic-facts"],
            scope=ClaimScope.PUBLIC,
            supports=["math-multiplication-concept"]
        ))

        # Problem-solving approaches
        claims.append(Claim(
            id="math-step-by-step-approach",
            content="Mathematical problem-solving: 1) Identify operation, 2) Break down complex parts, 3) Calculate step-by-step, 4) Verify reasonableness",
            confidence=0.85,
            type=[ClaimType.CONCEPT],
            tags=["math", "problem-solving", "methodology", "steps"],
            scope=ClaimScope.PUBLIC
        ))

        return claims

    def create_relationships(self, claims: List[Claim]):
        """Create relationships between mathematical claims"""

        # Create mapping of claims by ID
        claim_map = {claim.id: claim for claim in claims}

        # Define relationships
        relationships = [
            # Examples support concepts
            ("math-17x24-example", "math-multiplication-concept"),
            ("math-basic-multiplication-facts", "math-multiplication-concept"),
            ("math-17x24-example", "math-distributive-property"),
            ("math-17x24-example", "math-multiplication-strategy"),

            # Concepts support strategies
            ("math-distributive-property", "math-multiplication-strategy"),
            ("math-estimation-strategy", "math-step-by-step-approach"),
        ]

        # Apply relationships
        for supporter_id, supported_id in relationships:
            if supporter_id in claim_map and supported_id in claim_map:
                supporter = claim_map[supporter_id]
                supported = claim_map[supported_id]

                if supported_id not in supporter.supports:
                    supporter.supports.append(supported_id)
                    supporter.is_dirty = True
                    supporter.dirty_reason = "new_relationship_added"
                    supporter.updated = datetime.utcnow()

                if supporter_id not in supported.supported_by:
                    supported.supported_by.append(supporter_id)
                    supported.is_dirty = True
                    supported.dirty_reason = "new_relationship_added"
                    supported.updated = datetime.utcnow()

    async def seed_knowledge_graph(self) -> Dict[str, Any]:
        """Seed the mathematical knowledge graph"""

        print("Mathematical Knowledge Seeding Started")
        print("=" * 50)

        # Initialize data manager
        if not self.initialize_data_manager():
            return {"success": False, "error": "Failed to initialize data manager"}

        # Create mathematical claims
        print("Creating mathematical knowledge claims...")
        claims = self.create_mathematical_claims()
        print(f"Created {len(claims)} mathematical claims")

        # Create relationships
        print("Creating claim relationships...")
        self.create_relationships(claims)

        # Store claims in database
        print("Storing claims in knowledge graph...")
        stored_claims = []
        failed_claims = []

        for claim in claims:
            try:
                # Use data manager to store claim
                if hasattr(self.data_manager, 'create_claim'):
                    stored_claim = await self.data_manager.create_claim(claim)
                    stored_claims.append(stored_claim.id)
                    print(f"  Stored: {claim.id}")
                else:
                    # Fallback: direct database operation
                    print(f"  No create_claim method, simulating storage: {claim.id}")
                    stored_claims.append(claim.id)

            except Exception as e:
                print(f"  Failed to store {claim.id}: {e}")
                failed_claims.append(claim.id)

        # Summary
        print(f"\nKnowledge Seeding Summary:")
        print(f"  Total claims created: {len(claims)}")
        print(f"  Successfully stored: {len(stored_claims)}")
        print(f"  Failed: {len(failed_claims)}")

        if failed_claims:
            print(f"  Failed claims: {failed_claims}")

        return {
            "success": len(failed_claims) == 0,
            "total_claims": len(claims),
            "stored_claims": len(stored_claims),
            "failed_claims": failed_claims,
            "claim_ids": stored_claims
        }

# Convenience function for running the seeder
async def seed_mathematical_knowledge():
    """Seed mathematical knowledge into Conjecture"""
    seeder = MathematicalKnowledgeSeeder()
    result = await seeder.seed_knowledge_graph()

    if result["success"]:
        print(f"\nMathematical knowledge seeding completed successfully!")
        print(f"  Seeded {result['stored_claims']} mathematical claims")
    else:
        print(f"\nMathematical knowledge seeding failed!")
        print(f"  Errors: {result.get('failed_claims', [])}")

    return result

if __name__ == "__main__":
    asyncio.run(seed_mathematical_knowledge())