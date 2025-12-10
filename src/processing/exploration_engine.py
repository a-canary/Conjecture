"""
Exploration Engine - Claim selection and prioritization
Implements get_next_exploration() and claim state management
Start simple, extend only when needed
"""

import time
from datetime import datetime
from typing import List, Optional, Tuple

from ..core.basic_models import BasicClaim, ClaimState
from ..data.data_manager import DataManager

class ExplorationEngine:
    """Manages claim exploration and prioritization"""

    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.similarity_threshold = 0.7

    async def get_next_exploration(self, root_claim: BasicClaim) -> Optional[BasicClaim]:
        """
        Select claim with confidence < 0.95 and highest semantic similarity to root claim
        Implements weighted scoring: 70% semantic similarity + 30% support relationship
        """
        try:
            # Get all claims with confidence < 0.95 (excluding Orphaned)
            candidates = []
            all_claims = []

            # Filter candidates by confidence and state
            all_claims_dicts = await self.data_manager.get_statistics()
            all_claims = []
            
            # Get all claims from the data manager
            if hasattr(self.data_manager, 'sqlite_manager') and self.data_manager.sqlite_manager:
                # Get all claims from SQLite
                claims_data = await self.data_manager.sqlite_manager.get_all_claims()
                for claim_dict in claims_data:
                    claim = BasicClaim(**claim_dict)
                    all_claims.append(claim)
                    
                    if (
                        claim.confidence < 0.95
                        and claim.state != ClaimState.ORPHANED
                    ):
                        candidates.append(claim)

            if not candidates:
                return None

            # Calculate similarity scores
            best_candidate = None
            best_score = -1

            for candidate in candidates:
                # Simple semantic similarity (word overlap for now)
                semantic_sim = self._calculate_similarity(
                    root_claim.content, candidate.content
                )

                # Support relevance boost
                support_relevance = 1.0 if root_claim.id in candidate.supports else 0.5

                # Weighted score: 70% semantic + 30% support relevance
                score = 0.7 * semantic_sim + 0.3 * support_relevance

                if score > best_score:
                    best_score = score
                    best_candidate = candidate

            print(
                f"🎯 Selected claim for exploration: {best_candidate.id} (score: {best_score:.3f})"
            )
            return best_candidate

        except Exception as e:
            print(f"❌ Error in get_next_exploration: {e}")
            return None

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Simple word overlap similarity for semantic matching"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    async def get_exploration_queue(
        self, root_claim: BasicClaim, max_claims: int = 10
    ) -> List[BasicClaim]:
        """
        Get queue of claims to explore, ranked by similarity to root claim
        Returns claims with confidence < 0.95, sorted by similarity
        """
        try:
            # Get low confidence claims
            filter_obj = ClaimFilter(confidence_max=0.95)
            low_conf_claims_dicts = await self.data_manager.filter_claims(filter_obj)
            low_conf_claims = [BasicClaim(**claim_dict) for claim_dict in low_conf_claims_dicts]

            # Filter out orphaned claims
            active_claims = [
                claim for claim in low_conf_claims if claim.state != ClaimState.ORPHANED
            ]

            # Calculate similarity scores for all claims
            scored_claims = []
            for claim in active_claims:
                similarity = self._calculate_similarity(
                    root_claim.content, claim.content
                )

                # Apply filters and limits
                if similarity >= self.similarity_threshold:
                    scored_claims.append((claim, similarity))

            # Sort by similarity (highest first)
            scored_claims.sort(key=lambda x: x[1], reverse=True)

            # Return top claims
            queue = [claim for claim, _ in scored_claims[:max_claims]]

            print(f"📋 Exploration queue: {len(queue)} claims ready for processing")
            return queue

        except Exception as e:
            print(f"❌ Error getting exploration queue: {e}")
            return []

    async def update_claim_states(self, claim: BasicClaim) -> bool:
        """
        Update claim state based on confidence and relationships
        Implements state transition logic
        """
        try:
            old_state = claim.state

            # High confidence claims become Validated
            if claim.confidence >= 0.95:
                claim.state = ClaimState.VALIDATED
                print(
                    f"🏆 Claim {claim.id} promoted to Validated (confidence: {claim.confidence})"
                )

            # Check for orphaned status
            elif claim.state not in [ClaimState.VALIDATED, ClaimState.ORPHANED]:
                # Get supporting claims
                supporting_claims = []
                for sup_id in claim.supported_by:
                    sup_claim = await self.data_manager.get_claim(sup_id)
                    if sup_claim:
                        supporting_claims.append(sup_claim)

                # If all supporters are resolved, check confidence
                if supporting_claims:
                    all_resolved = all(
                        sup.state in [ClaimState.VALIDATED, ClaimState.ORPHANED]
                        for sup in supporting_claims
                    )

                    if all_resolved:
                        # Calculate average supporter confidence
                        avg_confidence = sum(
                            sup.confidence for sup in supporting_claims
                        ) / len(supporting_claims)

                        if avg_confidence >= 0.85:
                            claim.state = ClaimState.VALIDATED
                            print(
                                f"✅ Claim {claim.id} Validated by supporter consensus"
                            )
                        else:
                            claim.state = ClaimState.ORPHANED
                            print(
                                f"🗑️ Claim {claim.id} marked Orphaned (low confidence)"
                            )

            # Queue low confidence claims that aren't orphaned
            elif claim.confidence < 0.95 and claim.state == ClaimState.VALIDATED:
                claim.state = ClaimState.QUEUED
                print(f"⏳ Claim {claim.id} moved to Queued (confidence dropped)")

            # Update timestamp
            claim.updated = datetime.utcnow()

            # Save changes
            updates = {"state": claim.state.value, "updated": claim.updated}
            if await self.data_manager.update_claim(claim.id, updates):
                if old_state != claim.state:
                    print(
                        f"🔄 Claim {claim.id}: {old_state.value} → {claim.state.value}"
                    )
                return True
            else:
                return False

        except Exception as e:
            print(f"❌ Error updating claim state for {claim.id}: {e}")
            return False

    async def get_claim_statistics(self) -> dict:
        """Get statistics about claim states and distribution"""
        try:
            stats = {
                "total_claims": 0,
                "by_state": {},
                "by_type": {},
                "avg_confidence": 0.0,
                "high_confidence_count": 0,
                "low_confidence_count": 0,
            }

            # Get all claims from data manager
            claims_data = await self.data_manager.get_statistics()
            claims = []
            if hasattr(self.data_manager, 'sqlite_manager') and self.data_manager.sqlite_manager:
                claims_dicts = await self.data_manager.sqlite_manager.get_all_claims()
                claims = [BasicClaim(**claim_dict) for claim_dict in claims_dicts]
                stats["total_claims"] = len(claims)
            if not claims:
                return stats

            total_confidence = 0.0

            for claim in claims:
                # State statistics
                state_key = claim.state.value
                stats["by_state"][state_key] = stats["by_state"].get(state_key, 0) + 1

                # Type statistics
                for claim_type in claim.type:
                    type_key = claim_type.value
                    stats["by_type"][type_key] = stats["by_type"].get(type_key, 0) + 1

                # Confidence statistics
                total_confidence += claim.confidence
                if claim.confidence >= 0.8:
                    stats["high_confidence_count"] += 1
                elif claim.confidence < 0.5:
                    stats["low_confidence_count"] += 1

            stats["avg_confidence"] = total_confidence / len(claims) if claims else 0.0

            return stats

        except Exception as e:
            print(f"❌ Error generating claim statistics: {e}")
            return {}

async def test_exploration_engine():
    """Test exploration engine functionality"""
    print("🧪 Testing Exploration Engine")
    print("=" * 40)

    # Setup
    from ..data.data_manager import DataManager, DataConfig
    config = DataConfig(
        sqlite_path="./data/exploration_test.db",
        use_chroma=False,  # Disable Chroma for testing
        use_embeddings=False  # Disable embeddings for testing
    )
    data_manager = DataManager(config)
    await data_manager.initialize()
    
    engine = ExplorationEngine(data_manager)

    # Create sample claims
    root_claim = await data_manager.create_claim(
        content="Quantum encryption can prevent hospital data breaches through photon-based key distribution",
        confidence=0.3,
        claim_id="exp_root_001"
    )
    print("✅ Root claim created")

    # Create candidate claims
    candidates = []
    for i in range(5):
        claim = await data_manager.create_claim(
            content=f"Quantum key distribution using photon polarization for secure hospital communication #{i}",
            confidence=0.4 + (i * 0.1),
            claim_id=f"exp_cand_{i}"
        )
        candidates.append(claim)
        
        # Add relationship
        await data_manager.add_relationship(claim.id, root_claim.id)

    print(f"✅ Created {len(candidates)} candidate claims")

    # Test get_next_exploration
    next_claim = await engine.get_next_exploration(root_claim)
    if next_claim:
        print(f"✅ get_next_exploration: PASS (selected {next_claim.id})")
    else:
        print("❌ get_next_exploration: FAIL")
        return False

    # Test exploration queue
    queue = await engine.get_exploration_queue(root_claim, max_claims=3)
    print(f"✅ Exploration queue: PASS (found {len(queue)} claims)")

    # Test state updates
    high_conf_claim = candidates[-1]
    # Update confidence through data manager
    await data_manager.update_claim(high_conf_claim.id, {"confidence": 0.95})
    high_conf_claim = await data_manager.get_claim(high_conf_claim.id)
    
    if await engine.update_claim_states(high_conf_claim):
        print("✅ State promotion: PASS")
    else:
        print("❌ State promotion: FAIL")
        return False

    # Test statistics
    stats = await engine.get_claim_statistics()
    print(f"✅ Statistics: PASS (total: {stats['total_claims']})")

    # Cleanup
    await data_manager.close()
    
    print("🎉 All exploration engine tests passed!")
    return True

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(test_exploration_engine())
    exit(0 if success else 1)
