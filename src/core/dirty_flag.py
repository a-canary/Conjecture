"""
Dirty Flag System for Conjecture Claims
Handles automatic detection and cascading of dirty flags for claim re-evaluation
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict

from .models import Claim, DirtyReason

class DirtyFlagSystem:
    """Core dirty flag management system for claims"""

    def __init__(self, confidence_threshold: float = 0.90, cascade_depth: int = 3):
        """
        Initialize dirty flag system

        Args:
            confidence_threshold: Confidence threshold for prioritized evaluation
            cascade_depth: Maximum depth for cascading dirty flags
        """
        self.confidence_threshold = confidence_threshold
        self.cascade_depth = cascade_depth
        self.logger = logging.getLogger(__name__)
        self._dirty_claim_cache: Dict[str, Claim] = {}
        self._cascade_tracker: Dict[str, int] = {}

    def mark_claim_dirty(
        self, claim: Claim, reason: DirtyReason, priority: int = 0, cascade: bool = True
    ) -> None:
        """
        Mark a claim as dirty and optionally cascade to related claims

        Args:
            claim: The claim to mark dirty
            reason: Reason for marking dirty
            priority: Priority for evaluation (higher = more urgent)
            cascade: Whether to cascade dirty status to related claims
        """
        self.logger.debug(f"Marking claim {claim.id} as dirty: {reason.value}")

        # Set priority based on confidence if not explicitly provided
        if priority == 0:
            priority = self._calculate_priority(claim, reason)

        claim.mark_dirty(reason, priority)
        self._dirty_claim_cache[claim.id] = claim
        self._cascade_tracker[claim.id] = 0

        # Cascade to related claims if enabled
        if cascade:
            self._cascade_dirty_flags(claim, reason, 1)

    def _calculate_priority(self, claim: Claim, reason: DirtyReason) -> int:
        """Calculate priority based on claim confidence and reason"""
        base_priority = 0

        # Higher priority for low confidence claims
        if claim.confidence < self.confidence_threshold:
            base_priority += 10

        # Priority based on reason
        reason_priorities = {
            DirtyReason.NEW_CLAIM_ADDED: 5,
            DirtyReason.CONFIDENCE_THRESHOLD: 15,
            DirtyReason.SUPPORTING_CLAIM_CHANGED: 8,
            DirtyReason.RELATIONSHIP_CHANGED: 6,
            DirtyReason.MANUAL_MARK: 20,
            DirtyReason.BATCH_EVALUATION: 3,
            DirtyReason.SYSTEM_TRIGGER: 4,
        }
        base_priority += reason_priorities.get(reason, 0)

        # Additional priority based on how far below threshold
        if claim.confidence < self.confidence_threshold:
            confidence_gap = self.confidence_threshold - claim.confidence
            base_priority += int(confidence_gap * 10)

        return base_priority

    def _cascade_dirty_flags(
        self, source_claim: Claim, reason: DirtyReason, current_depth: int
    ) -> None:
        """
        Cascade dirty flag to related claims

        Args:
            source_claim: The claim that triggered the cascade
            reason: Reason for dirty flag
            current_depth: Current cascade depth
        """
        if current_depth > self.cascade_depth:
            return

        # Get related claims (both supported by and supports)
        related_ids = set(source_claim.supported_by) | set(source_claim.supports)

        cascade_reason = DirtyReason.SUPPORTING_CLAIM_CHANGED
        priority_penalty = max(
            0, 10 - (current_depth * 2)
        )  # Decreasing priority with depth

        for related_id in related_ids:
            # Prevent infinite loops
            if related_id in self._cascade_tracker:
                if self._cascade_tracker[related_id] <= current_depth:
                    continue

            self._cascade_tracker[related_id] = current_depth

            # Get the related claim (this would typically come from a data store)
            # For now, we'll skip if not in cache
            if related_id in self._dirty_claim_cache:
                related_claim = self._dirty_claim_cache[related_id]
            else:
                # In a real implementation, you'd fetch this from your data store
                continue

            # Mark related claim as dirty with lower priority
            self.logger.debug(
                f"Cascading dirty flag to {related_id} at depth {current_depth}"
            )
            related_claim.mark_dirty(cascade_reason, priority_penalty)
            self._dirty_claim_cache[related_id] = related_claim

            # Continue cascade
            if current_depth < self.cascade_depth:
                self._cascade_dirty_flags(
                    related_claim, cascade_reason, current_depth + 1
                )

    def on_claim_updated(
        self, updated_claim: Claim, original_claim: Claim, all_claims: Dict[str, Claim]
    ) -> None:
        """
        Handle dirty flag propagation when a claim is updated

        Args:
            updated_claim: The new version of the claim
            original_claim: The original version before update
            all_claims: Dictionary of all claims for relationship lookup
        """
        # Check if the claim actually changed in meaningful ways
        if (
            updated_claim.content != original_claim.content
            or updated_claim.confidence != original_claim.confidence
        ):
            # Find claims that this claim supports (B claims where A supports B)
            supported_claim_ids = updated_claim.supports

            for supported_id in supported_claim_ids:
                if supported_id in all_claims:
                    supported_claim = all_claims[supported_id]
                    self.mark_claim_dirty(
                        supported_claim,
                        DirtyReason.SUPPORTING_CLAIM_CHANGED,
                        priority=8,  # Medium priority for supporting claim changes
                        cascade=False,  # Don't cascade from here to avoid infinite loops
                    )
                    self.logger.info(
                        f"Marked claim {supported_id} dirty due to update in supporting claim {updated_claim.id}"
                    )

    def mark_claims_dirty_by_confidence_threshold(
        self, claims: List[Claim], threshold: Optional[float] = None
    ) -> int:
        """
        Mark claims as dirty if confidence is below threshold

        Args:
            claims: List of claims to check
            threshold: Confidence threshold (uses instance threshold if None)

        Returns:
            Number of claims marked dirty
        """
        threshold = threshold or self.confidence_threshold
        marked_count = 0

        for claim in claims:
            if claim.confidence < threshold and not claim.is_dirty:
                self.mark_claim_dirty(
                    claim, DirtyReason.CONFIDENCE_THRESHOLD, priority=15, cascade=True
                )
                marked_count += 1

        self.logger.info(
            f"Marked {marked_count} claims dirty due to confidence threshold"
        )
        return marked_count

    def mark_dirty_on_new_claim(
        self,
        new_claim: Claim,
        existing_claims: List[Claim],
        similarity_threshold: float = 0.7,
    ) -> int:
        """
        Mark related claims dirty when a new claim is added

        Args:
            new_claim: The newly added claim
            existing_claims: List of existing claims
            similarity_threshold: Threshold for claim similarity

        Returns:
            Number of existing claims marked dirty
        """
        marked_count = 0

        for claim in existing_claims:
            # Check for potential relationships based on shared attributes
            if self._should_mark_related_on_new_claim(
                new_claim, claim, similarity_threshold
            ):
                # Mark existing claim dirty due to new claim
                self.mark_claim_dirty(
                    claim, DirtyReason.NEW_CLAIM_ADDED, priority=5, cascade=True
                )
                marked_count += 1

        self.logger.info(
            f"Marked {marked_count} existing claims dirty due to new claim {new_claim.id}"
        )
        return marked_count

    def _should_mark_related_on_new_claim(
        self, new_claim: Claim, existing_claim: Claim, similarity_threshold: float
    ) -> bool:
        """
        Determine if existing claim should be marked dirty due to new claim

        Args:
            new_claim: Newly added claim
            existing_claim: Existing claim to check
            similarity_threshold: Similarity threshold for marking dirty

        Returns:
            True if existing claim should be marked dirty
        """
        # Skip already dirty claims
        if existing_claim.is_dirty:
            return False

        # Check for shared tags
        shared_tags = set(new_claim.tags) & set(existing_claim.tags)
        if shared_tags:
            tag_similarity = len(shared_tags) / max(
                len(new_claim.tags), len(existing_claim.tags), 1
            )
            if tag_similarity > similarity_threshold:
                return True

        # Check for shared claim types
        shared_types = set(new_claim.type) & set(existing_claim.type)
        if shared_types:
            return True

        # Check for existing relationships
        if (
            existing_claim.id in new_claim.supported_by
            or existing_claim.id in new_claim.supports
        ):
            return True

        return False

    def get_dirty_claims(
        self,
        claims: Optional[List[Claim]] = None,
        prioritize: bool = True,
        max_count: Optional[int] = None,
    ) -> List[Claim]:
        """
        Get list of dirty claims, optionally prioritized

        Args:
            claims: List of claims to filter (checks cache if None)
            prioritize: Whether to sort by priority
            max_count: Maximum number of claims to return

        Returns:
            List of dirty claims
        """
        if claims is None:
            # Use cached dirty claims
            dirty_claims = list(self._dirty_claim_cache.values())
        else:
            # Filter provided claims
            dirty_claims = [claim for claim in claims if claim.is_dirty]
            # Update cache
            for claim in dirty_claims:
                self._dirty_claim_cache[claim.id] = claim

        if prioritize:
            # Sort by priority (descending) and timestamp (ascending for same priority)
            dirty_claims.sort(
                key=lambda c: (-c.dirty_priority, c.dirty_timestamp or datetime.min)
            )

        if max_count:
            dirty_claims = dirty_claims[:max_count]

        return dirty_claims

    def get_priority_dirty_claims(
        self,
        claims: Optional[List[Claim]] = None,
        confidence_threshold: Optional[float] = None,
        max_count: Optional[int] = None,
    ) -> List[Claim]:
        """
        Get prioritized dirty claims (confidence below threshold)

        Args:
            claims: List of claims to filter
            confidence_threshold: Confidence threshold for priority
            max_count: Maximum number of claims to return

        Returns:
            List of priority dirty claims
        """
        threshold = confidence_threshold or self.confidence_threshold

        dirty_claims = self.get_dirty_claims(claims, prioritize=True)

        # Filter for priority claims
        priority_claims = [
            claim for claim in dirty_claims if claim.should_prioritize(threshold)
        ]

        if max_count:
            priority_claims = priority_claims[:max_count]

        self.logger.info(
            f"Found {len(priority_claims)} priority dirty claims out of {len(dirty_claims)} total dirty claims"
        )

        return priority_claims

    def clear_dirty_flags(
        self,
        claims: Optional[List[Claim]] = None,
        reason_filter: Optional[DirtyReason] = None,
    ) -> int:
        """
        Clear dirty flags from claims

        Args:
            claims: List of claims to clean (cleans all if None)
            reason_filter: Only clear claims with specific reason

        Returns:
            Number of claims cleaned
        """
        if claims is None:
            claims_to_clean = list(self._dirty_claim_cache.values())
        else:
            claims_to_clean = [claim for claim in claims if claim.is_dirty]

        # Filter by reason if specified
        if reason_filter:
            claims_to_clean = [
                claim
                for claim in claims_to_clean
                if claim.dirty_reason == reason_filter
            ]

        # Clean claims
        for claim in claims_to_clean:
            claim.mark_clean()
            if claim.id in self._dirty_claim_cache:
                del self._dirty_claim_cache[claim.id]
            if claim.id in self._cascade_tracker:
                del self._cascade_tracker[claim.id]

        self.logger.info(f"Cleared dirty flags from {len(claims_to_clean)} claims")
        return len(claims_to_clean)

    def get_dirty_statistics(
        self, claims: Optional[List[Claim]] = None
    ) -> Dict[str, int]:
        """
        Get statistics about dirty claims

        Args:
            claims: List of claims to analyze (uses cache if None)

        Returns:
            Dictionary with dirty claim statistics
        """
        dirty_claims = self.get_dirty_claims(claims, prioritize=False)

        stats = {
            "total_dirty": len(dirty_claims),
            "priority_dirty": len(
                [
                    c
                    for c in dirty_claims
                    if c.should_prioritize(self.confidence_threshold)
                ]
            ),
            "reasons": defaultdict(int),
            "priority_ranges": defaultdict(int),
        }

        for claim in dirty_claims:
            # Count by reason
            if claim.dirty_reason:
                stats["reasons"][claim.dirty_reason.value] += 1

            # Count by priority ranges
            if claim.dirty_priority >= 20:
                stats["priority_ranges"]["high"] += 1
            elif claim.dirty_priority >= 10:
                stats["priority_ranges"]["medium"] += 1
            else:
                stats["priority_ranges"]["low"] += 1

        # Convert defaultdicts to regular dicts
        stats["reasons"] = dict(stats["reasons"])
        stats["priority_ranges"] = dict(stats["priority_ranges"])

        return stats

    def invalidate_claim(self, claim_id: str) -> None:
        """
        Invalidate claim from cache (useful when claim is deleted)

        Args:
            claim_id: ID of claim to invalidate
        """
        if claim_id in self._dirty_claim_cache:
            del self._dirty_claim_cache[claim_id]
        if claim_id in self._cascade_tracker:
            del self._cascade_tracker[claim_id]

    def rebuild_cache(self, claims: List[Claim]) -> None:
        """
        Rebuild dirty claim cache from full claim list

        Args:
            claims: List of all claims
        """
        self._dirty_claim_cache.clear()
        self._cascade_tracker.clear()

        for claim in claims:
            if claim.is_dirty:
                self._dirty_claim_cache[claim.id] = claim
                self._cascade_tracker[claim.id] = 0

        self.logger.info(
            f"Rebuilt dirty cache with {len(self._dirty_claim_cache)} dirty claims"
        )
