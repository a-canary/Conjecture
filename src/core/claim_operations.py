# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Pure functional claim operations for Conjecture claims.

All functions in this module are pure: they return new Claim instances
rather than mutating the input claim. This ensures immutability and
predictable behavior throughout the system.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

from .models import Claim, ClaimType, DirtyReason


def update_confidence(claim: Claim, confidence: float) -> Claim:
    """Update claim confidence, returning a new Claim.

    Args:
        claim: The claim to update.
        confidence: New confidence value (must be between 0.0 and 1.0).

    Returns:
        New Claim with updated confidence and updated timestamp.

    Raises:
        ValueError: If confidence is outside the range [0.0, 1.0].
    """
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError("Confidence must be between 0.0 and 1.0")

    data = claim.model_dump()
    data["confidence"] = confidence
    data["updated"] = datetime.now(timezone.utc)
    return Claim(**data)


def mark_dirty(claim: Claim, reason: DirtyReason, priority: int = 0) -> Claim:
    """Mark claim as dirty, returning a new Claim.

    Args:
        claim: The claim to mark dirty.
        reason: The reason for marking the claim dirty.
        priority: Priority for dirty evaluation (higher = more urgent). Defaults to 0.

    Returns:
        New Claim with dirty flag set.
    """
    data = claim.model_dump()
    now = datetime.now(timezone.utc)
    data["is_dirty"] = True
    data["dirty"] = True
    data["dirty_reason"] = reason
    data["dirty_priority"] = priority
    data["dirty_timestamp"] = now
    data["updated"] = now
    return Claim(**data)


def mark_clean(claim: Claim) -> Claim:
    """Mark claim as clean, returning a new Claim.

    Args:
        claim: The claim to mark clean.

    Returns:
        New Claim with dirty flag cleared.
    """
    data = claim.model_dump()
    now = datetime.now(timezone.utc)
    data["is_dirty"] = False
    data["dirty"] = False
    data["dirty_reason"] = None
    data["dirty_timestamp"] = None
    data["dirty_priority"] = 0
    data["updated"] = now
    return Claim(**data)


def add_sub(claim: Claim, sub_id: str) -> Claim:
    """Add a sub (child/supporting) claim relationship, returning a new Claim.

    If the sub_id already exists in claim.subs, the operation is idempotent
    and returns a new Claim with no duplicate entry.

    Adding a sub also marks the claim dirty with SUPPORTING_CLAIM_CHANGED reason.

    Args:
        claim: The claim to add a sub to.
        sub_id: The ID of the claim that provides evidence for this claim.

    Returns:
        New Claim with the sub relationship added and dirty flag set.
    """
    if sub_id in claim.subs:
        # Idempotent: return new claim with same subs (no mutation)
        data = claim.model_dump()
        return Claim(**data)

    data = claim.model_dump()
    now = datetime.now(timezone.utc)
    data["subs"] = list(claim.subs) + [sub_id]
    data["is_dirty"] = True
    data["dirty"] = True
    data["dirty_reason"] = DirtyReason.SUPPORTING_CLAIM_CHANGED
    data["dirty_timestamp"] = now
    data["updated"] = now
    return Claim(**data)


def add_super(claim: Claim, super_id: str) -> Claim:
    """Add a super (parent/supported) claim relationship, returning a new Claim.

    If the super_id already exists in claim.supers, the operation is idempotent
    and returns a new Claim with no duplicate entry.

    Adding a super also marks the claim dirty with SUPPORTING_CLAIM_CHANGED reason.

    Args:
        claim: The claim to add a super to.
        super_id: The ID of the claim this claim provides evidence FOR.

    Returns:
        New Claim with the super relationship added and dirty flag set.
    """
    if super_id in claim.supers:
        # Idempotent: return new claim with same supers (no mutation)
        data = claim.model_dump()
        return Claim(**data)

    data = claim.model_dump()
    now = datetime.now(timezone.utc)
    data["supers"] = list(claim.supers) + [super_id]
    data["is_dirty"] = True
    data["dirty"] = True
    data["dirty_reason"] = DirtyReason.SUPPORTING_CLAIM_CHANGED
    data["dirty_timestamp"] = now
    data["updated"] = now
    return Claim(**data)


def find_dirty_claims(
    claims: List[Claim], priority_threshold: Optional[int] = None
) -> List[Claim]:
    """Filter claims to find those that are dirty.

    Args:
        claims: The list of claims to search.
        priority_threshold: If provided, only return dirty claims whose
            dirty_priority is >= this threshold.

    Returns:
        List of dirty claims (optionally filtered by priority threshold).
    """
    dirty = [c for c in claims if c.is_dirty]
    if priority_threshold is not None:
        dirty = [c for c in dirty if c.dirty_priority >= priority_threshold]
    return dirty


def find_super_claims(claim: Claim, all_claims: List[Claim]) -> List[Claim]:
    """Find all claims that this claim provides evidence FOR (its supers).

    Args:
        claim: The claim whose supers to find.
        all_claims: The full collection of claims to search within.

    Returns:
        List of Claim objects corresponding to claim.supers IDs that exist
        in all_claims.
    """
    super_ids: Set[str] = set(claim.supers)
    return [c for c in all_claims if c.id in super_ids]


def find_sub_claims(claim: Claim, all_claims: List[Claim]) -> List[Claim]:
    """Find all claims that provide evidence FOR this claim (its subs).

    Args:
        claim: The claim whose subs to find.
        all_claims: The full collection of claims to search within.

    Returns:
        List of Claim objects corresponding to claim.subs IDs that exist
        in all_claims.
    """
    sub_ids: Set[str] = set(claim.subs)
    return [c for c in all_claims if c.id in sub_ids]


def filter_claims_by_confidence(
    claims: List[Claim],
    min_confidence: float = 0.0,
    max_confidence: float = 1.0,
) -> List[Claim]:
    """Filter claims by confidence range (inclusive).

    Args:
        claims: The list of claims to filter.
        min_confidence: Minimum confidence (inclusive).
        max_confidence: Maximum confidence (inclusive).

    Returns:
        Claims whose confidence is within [min_confidence, max_confidence].
    """
    return [
        c for c in claims if min_confidence <= c.confidence <= max_confidence
    ]


def filter_claims_by_tags(
    claims: List[Claim],
    tags: List[str],
    match_all: bool = False,
) -> List[Claim]:
    """Filter claims by tags.

    Args:
        claims: The list of claims to filter.
        tags: The tags to match against.
        match_all: If True, a claim must contain all specified tags.
                   If False (default), a claim must contain at least one tag.

    Returns:
        Claims that match the tag criteria.
    """
    search_tags = {t.lower() for t in tags}

    def matches(claim: Claim) -> bool:
        claim_tags = {t.lower() for t in claim.tags}
        if match_all:
            return search_tags.issubset(claim_tags)
        return bool(search_tags & claim_tags)

    return [c for c in claims if matches(c)]


def filter_claims_by_type(
    claims: List[Claim],
    types: List[str],
) -> List[Claim]:
    """Filter claims by claim type.

    Args:
        claims: The list of claims to filter.
        types: The type names to match (case-insensitive string values of ClaimType).

    Returns:
        Claims that have at least one matching type.
    """
    search_types = {t.lower() for t in types}

    def matches(claim: Claim) -> bool:
        claim_type_values = {ct.value.lower() for ct in claim.type}
        return bool(search_types & claim_type_values)

    return [c for c in claims if matches(c)]


def calculate_support_strength(
    claim: Claim, all_claims: List[Claim]
) -> Tuple[float, int]:
    """Calculate the support strength of a claim based on its sub claims.

    Support strength is the average confidence of all sub (supporting) claims.

    Args:
        claim: The claim to calculate support strength for.
        all_claims: The full collection of claims.

    Returns:
        Tuple of (average_confidence, supporter_count). If no supporters exist,
        returns (0.0, 0).
    """
    sub_claims = find_sub_claims(claim, all_claims)
    count = len(sub_claims)
    if count == 0:
        return 0.0, 0
    total = sum(c.confidence for c in sub_claims)
    return total / count, count


def get_claim_hierarchy(claim: Claim, all_claims: List[Claim]) -> dict:
    """Build a hierarchy representation of a claim and its immediate relationships.

    Args:
        claim: The claim to build hierarchy for.
        all_claims: The full collection of claims.

    Returns:
        Dictionary with claim_id, confidence, subs_count, supers_count,
        subs (list of dicts with id/content/confidence), and
        supers (list of dicts with id/content/confidence).
        Content longer than 100 characters is truncated to 100 chars + "...".
    """

    def truncate(content: str, max_len: int = 100) -> str:
        if len(content) > max_len:
            return content[:max_len] + "..."
        return content

    sub_claims = find_sub_claims(claim, all_claims)
    super_claims = find_super_claims(claim, all_claims)

    return {
        "claim_id": claim.id,
        "confidence": claim.confidence,
        "subs_count": len(sub_claims),
        "supers_count": len(super_claims),
        "subs": [
            {
                "id": c.id,
                "content": truncate(c.content),
                "confidence": c.confidence,
            }
            for c in sub_claims
        ],
        "supers": [
            {
                "id": c.id,
                "content": truncate(c.content),
                "confidence": c.confidence,
            }
            for c in super_claims
        ],
    }


def validate_relationship_integrity(
    claim: Claim, all_claims: List[Claim]
) -> List[str]:
    """Validate that all relationship references in a claim point to existing claims.

    Args:
        claim: The claim whose relationships to validate.
        all_claims: The full collection of claims to check against.

    Returns:
        List of error strings. Empty list means all relationships are valid.
        Each error describes a missing claim ID.
    """
    existing_ids = {c.id for c in all_claims}
    errors: List[str] = []

    for sub_id in claim.subs:
        if sub_id not in existing_ids:
            errors.append(f"Sub claim '{sub_id}' not found in claim collection")

    for super_id in claim.supers:
        if super_id not in existing_ids:
            errors.append(f"Super claim '{super_id}' not found in claim collection")

    return errors


def batch_update_confidence(
    claims: List[Claim], updates: Dict[str, float]
) -> List[Claim]:
    """Batch update confidence values for multiple claims.

    Args:
        claims: The list of claims to potentially update.
        updates: Dictionary mapping claim IDs to new confidence values.
                 Claims not present in the dictionary are returned unchanged.

    Returns:
        New list of claims with confidence values updated for those in updates.

    Raises:
        ValueError: If any confidence value in updates is outside [0.0, 1.0].
    """
    result = []
    for claim in claims:
        if claim.id in updates:
            result.append(update_confidence(claim, updates[claim.id]))
        else:
            result.append(claim)
    return result


def should_prioritize(claim: Claim, confidence_threshold: float = 0.9) -> bool:
    """Determine whether a claim should be prioritized for re-evaluation.

    A claim should be prioritized when it is dirty AND its confidence is
    below the given threshold.

    Args:
        claim: The claim to evaluate.
        confidence_threshold: The confidence threshold below which a dirty
            claim is considered priority. Defaults to 0.9.

    Returns:
        True if the claim is dirty and confidence < confidence_threshold.
    """
    return claim.is_dirty and claim.confidence < confidence_threshold


def set_dirty_priority(claim: Claim, priority: int) -> Claim:
    """Set the dirty priority on a claim, but only if the claim is dirty.

    If the claim is clean, the priority is left unchanged (at 0).

    Args:
        claim: The claim to update.
        priority: The priority value to set.

    Returns:
        New Claim with updated priority (if dirty) or unchanged (if clean).
    """
    if not claim.is_dirty:
        # Clean claims: return as-is (priority stays at 0 or whatever it is)
        data = claim.model_dump()
        return Claim(**data)

    data = claim.model_dump()
    data["dirty_priority"] = priority
    return Claim(**data)


def update_claim_with_dirty_propagation(
    updated: Claim,
    original: Claim,
    all_claims: Dict[str, Claim],
    confidence_change_threshold: float = 0.01,
) -> Tuple[Claim, Set[str]]:
    """Update a claim and propagate dirty flags to its supers if content or
    confidence changed significantly.

    Args:
        updated: The new version of the claim.
        original: The original version of the claim before the update.
        all_claims: Dictionary mapping claim IDs to Claim objects.
        confidence_change_threshold: Minimum absolute confidence change that
            triggers dirty propagation. Defaults to 0.01.

    Returns:
        Tuple of (updated_claim, marked_ids) where marked_ids is the set of
        claim IDs that were marked dirty due to propagation.
    """
    marked_ids: Set[str] = set()

    content_changed = updated.content != original.content
    confidence_changed = (
        abs(updated.confidence - original.confidence) >= confidence_change_threshold
    )

    if content_changed or confidence_changed:
        # Propagate to supers (claims this claim provides evidence FOR)
        for super_id in updated.supers:
            if super_id in all_claims:
                marked_ids.add(super_id)

    return updated, marked_ids
