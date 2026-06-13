# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Pure functions for claim operations - Separated from data models
This is the Tools layer for claim manipulation operations.

Naming convention:
- subs: claims that provide evidence FOR this claim (children)
- supers: claims this claim provides evidence FOR (toward root, parents)
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Tuple
from .models import Claim, DirtyReason


def update_confidence(claim: Claim, new_confidence: float) -> Claim:
    """Pure function to update confidence and timestamp"""
    if not 0.0 <= new_confidence <= 1.0:
        raise ValueError("Confidence must be between 0.0 and 1.0")

    return Claim(
        id=claim.id,
        content=claim.content,
        confidence=new_confidence,
        state=claim.state,
        subs=claim.subs.copy(),
        supers=claim.supers.copy(),
        scope=claim.scope,
        tags=claim.tags.copy(),
        created=claim.created,
        updated=datetime.now(timezone.utc),
        embedding=claim.embedding,
        is_dirty=claim.is_dirty,
        dirty_reason=claim.dirty_reason,
        dirty_timestamp=claim.dirty_timestamp,
        dirty_priority=claim.dirty_priority,
    )


def add_sub(claim: Claim, sub_claim_id: str) -> Claim:
    """Pure function to add a sub claim ID (claim that provides evidence FOR this claim)"""
    subs = claim.subs.copy()
    if sub_claim_id not in subs:
        subs.append(sub_claim_id)

    return Claim(
        id=claim.id,
        content=claim.content,
        confidence=claim.confidence,
        state=claim.state,
        subs=subs,
        supers=claim.supers.copy(),
        scope=claim.scope,
        tags=claim.tags.copy(),
        created=claim.created,
        updated=datetime.now(timezone.utc),
        embedding=claim.embedding,
        is_dirty=True,  # Mark as dirty when sub is added
        dirty_reason=DirtyReason.SUPPORTING_CLAIM_CHANGED,
        dirty_timestamp=datetime.now(timezone.utc),
        dirty_priority=claim.dirty_priority,
    )


def add_super(claim: Claim, super_claim_id: str) -> Claim:
    """Pure function to add a super claim ID (claim this provides evidence FOR, toward root)"""
    supers = claim.supers.copy()
    if super_claim_id not in supers:
        supers.append(super_claim_id)

    return Claim(
        id=claim.id,
        content=claim.content,
        confidence=claim.confidence,
        state=claim.state,
        subs=claim.subs.copy(),
        supers=supers,
        scope=claim.scope,
        tags=claim.tags.copy(),
        created=claim.created,
        updated=datetime.now(timezone.utc),
        embedding=claim.embedding,
        is_dirty=True,  # Mark as dirty when super is added
        dirty_reason=DirtyReason.SUPPORTING_CLAIM_CHANGED,
        dirty_timestamp=datetime.now(timezone.utc),
        dirty_priority=claim.dirty_priority,
    )


def mark_dirty(claim: Claim, reason: DirtyReason, priority: int = 0) -> Claim:
    """Pure function to mark claim as dirty for re-evaluation"""
    return Claim(
        id=claim.id,
        content=claim.content,
        confidence=claim.confidence,
        state=claim.state,
        subs=claim.subs.copy(),
        supers=claim.supers.copy(),
        scope=claim.scope,
        tags=claim.tags.copy(),
        created=claim.created,
        updated=datetime.now(timezone.utc),
        embedding=claim.embedding,
        is_dirty=True,
        dirty_reason=reason,
        dirty_timestamp=datetime.now(timezone.utc),
        dirty_priority=priority,
    )


def mark_clean(claim: Claim) -> Claim:
    """Pure function to mark claim as clean (no longer needs re-evaluation)"""
    return Claim(
        id=claim.id,
        content=claim.content,
        confidence=claim.confidence,
        state=claim.state,
        subs=claim.subs.copy(),
        supers=claim.supers.copy(),
        scope=claim.scope,
        tags=claim.tags.copy(),
        created=claim.created,
        updated=datetime.now(timezone.utc),
        embedding=claim.embedding,
        is_dirty=False,
        dirty=False,
        dirty_reason=None,
        dirty_timestamp=None,
        dirty_priority=0,
    )


def set_dirty_priority(claim: Claim, priority: int) -> Claim:
    """Pure function to set dirty priority for evaluation ordering"""
    return Claim(
        id=claim.id,
        content=claim.content,
        confidence=claim.confidence,
        state=claim.state,
        subs=claim.subs.copy(),
        supers=claim.supers.copy(),
        scope=claim.scope,
        tags=claim.tags.copy(),
        created=claim.created,
        updated=datetime.now(timezone.utc),
        embedding=claim.embedding,
        is_dirty=claim.is_dirty,
        dirty_reason=claim.dirty_reason,
        dirty_timestamp=claim.dirty_timestamp,
        dirty_priority=priority if claim.is_dirty else claim.dirty_priority,
    )


def should_prioritize(claim: Claim, confidence_threshold: float = 0.90) -> bool:
    """Pure function to check if claim should be prioritized for evaluation"""
    return claim.is_dirty and claim.confidence < confidence_threshold


def find_sub_claims(claim: Claim, all_claims: List[Claim]) -> List[Claim]:
    """Pure function to find all sub claims (claims that provide evidence FOR this claim)"""
    return [c for c in all_claims if c.id in claim.subs]


def find_super_claims(claim: Claim, all_claims: List[Claim]) -> List[Claim]:
    """Pure function to find all super claims (claims this provides evidence FOR)"""
    return [c for c in all_claims if c.id in claim.supers]


def calculate_support_strength(
    claim: Claim, all_claims: List[Claim]
) -> Tuple[float, int]:
    """Pure function to calculate support strength from sub claims"""
    sub_claims = find_sub_claims(claim, all_claims)
    if not sub_claims:
        return 0.0, 0

    # Simple strength calculation: average confidence weighted by relationship
    total_confidence = sum(c.confidence for c in sub_claims)
    avg_confidence = total_confidence / len(sub_claims)

    return avg_confidence, len(sub_claims)


def validate_relationship_integrity(claim: Claim, all_claims: List[Claim]) -> List[str]:
    """Pure function to validate claim relationships"""
    errors = []
    claim_ids = {c.id for c in all_claims}

    # Check if sub claim IDs exist
    for claim_id in claim.subs:
        if claim_id not in claim_ids:
            errors.append(f"Sub claim {claim_id} not found")

    # Check if super claim IDs exist
    for claim_id in claim.supers:
        if claim_id not in claim_ids:
            errors.append(f"Super claim {claim_id} not found")

    return errors


def get_claim_hierarchy(
    claim: Claim, all_claims: List[Claim], max_depth: int = 5
) -> Dict[str, Any]:
    """Pure function to get claim hierarchy/relationships"""
    hierarchy = {
        "claim_id": claim.id,
        "confidence": claim.confidence,
        "state": claim.state.value,
        "supers_count": len(claim.supers),
        "subs_count": len(claim.subs),
        "subs": [],
        "supers": [],
    }

    # Get subs (claims that provide evidence FOR this claim)
    sub_details = []
    for sub in find_sub_claims(claim, all_claims):
        sub_details.append(
            {
                "id": sub.id,
                "confidence": sub.confidence,
                "content": sub.content[:100] + "..."
                if len(sub.content) > 100
                else sub.content,
            }
        )
    hierarchy["subs"] = sub_details

    # Get super claims (claims this provides evidence FOR)
    super_details = []
    for super_claim in find_super_claims(claim, all_claims):
        super_details.append(
            {
                "id": super_claim.id,
                "confidence": super_claim.confidence,
                "content": super_claim.content[:100] + "..."
                if len(super_claim.content) > 100
                else super_claim.content,
            }
        )
    hierarchy["supers"] = super_details

    return hierarchy


def batch_update_confidence(
    claims: List[Claim], updates: Dict[str, float]
) -> List[Claim]:
    """Pure function to update confidence for multiple claims"""
    updated_claims = []
    for claim in claims:
        if claim.id in updates:
            updated_claims.append(update_confidence(claim, updates[claim.id]))
        else:
            updated_claims.append(claim)
    return updated_claims


def find_dirty_claims(claims: List[Claim], priority_threshold: int = 0) -> List[Claim]:
    """Pure function to find dirty claims with optional priority filter"""
    return [c for c in claims if c.is_dirty and c.dirty_priority >= priority_threshold]


def filter_claims_by_confidence(
    claims: List[Claim], min_confidence: float = 0.0, max_confidence: float = 1.0
) -> List[Claim]:
    """Pure function to filter claims by confidence range"""
    return [c for c in claims if min_confidence <= c.confidence <= max_confidence]


def filter_claims_by_type(claims: List[Claim], claim_types: List[str]) -> List[Claim]:
    """Pure function to filter claims by type"""
    target_types = set(claim_types)
    return [c for c in claims if any(t.value in target_types for t in c.type)]


def filter_claims_by_tags(
    claims: List[Claim], tags: List[str], match_all: bool = False
) -> List[Claim]:
    """Pure function to filter claims by tags"""
    target_tags = set(tags)

    if match_all:
        return [c for c in claims if target_tags.issubset(set(c.tags))]
    else:
        return [c for c in claims if any(tag in target_tags for tag in c.tags)]


def update_claim_with_dirty_propagation(
    updated_claim: Claim,
    original_claim: Claim,
    all_claims: Dict[str, Claim],
    dirty_system: Optional["DirtyFlagSystem"] = None,
) -> Tuple[Claim, List[str]]:
    """
    Update a claim and propagate dirty flags to super claims (claims this provides evidence FOR).

    Args:
        updated_claim: The new version of the claim
        original_claim: The original version before update
        all_claims: Dictionary of all claims for relationship lookup
        dirty_system: Optional dirty flag system for propagation

    Returns:
        Tuple of (updated_claim, list_of_marked_dirty_claim_ids)
    """
    marked_dirty_ids = []

    # Check if claim actually changed in meaningful ways
    if (
        updated_claim.content != original_claim.content
        or abs(updated_claim.confidence - original_claim.confidence) > 0.01
    ):
        # Find claims that this claim provides evidence FOR (supers, toward root)
        super_claim_ids = updated_claim.supers

        for super_id in super_claim_ids:
            if super_id in all_claims:
                super_claim = all_claims[super_id]

                # Mark the super claim as dirty
                if dirty_system:
                    dirty_system.mark_claim_dirty(
                        super_claim,
                        DirtyReason.SUPPORTING_CLAIM_CHANGED,
                        priority=8,
                        cascade=False,
                    )
                else:
                    # Fallback: mark dirty directly
                    super_claim.mark_dirty(
                        DirtyReason.SUPPORTING_CLAIM_CHANGED, priority=8
                    )

                marked_dirty_ids.append(super_id)

    return updated_claim, marked_dirty_ids
