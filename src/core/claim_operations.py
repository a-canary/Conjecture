"""
Pure functions for claim operations - Separated from data models
This is the Tools layer for claim manipulation operations.
"""
from datetime import datetime
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
        supported_by=claim.supported_by.copy(),
        supports=claim.supports.copy(),
        type=claim.type.copy(),
        tags=claim.tags.copy(),
        created=claim.created,
        updated=datetime.utcnow(),
        embedding=claim.embedding,
        is_dirty=claim.is_dirty,
        dirty_reason=claim.dirty_reason,
        dirty_timestamp=claim.dirty_timestamp,
        dirty_priority=claim.dirty_priority
    )


def add_support(claim: Claim, supporting_claim_id: str) -> Claim:
    """Pure function to add a supporting claim ID"""
    supported_by = claim.supported_by.copy()
    if supporting_claim_id not in supported_by:
        supported_by.append(supporting_claim_id)
    
    return Claim(
        id=claim.id,
        content=claim.content,
        confidence=claim.confidence,
        state=claim.state,
        supported_by=supported_by,
        supports=claim.supports.copy(),
        type=claim.type.copy(),
        tags=claim.tags.copy(),
        created=claim.created,
        updated=datetime.utcnow(),
        embedding=claim.embedding,
        is_dirty=claim.is_dirty,
        dirty_reason=claim.dirty_reason,
        dirty_timestamp=claim.dirty_timestamp,
        dirty_priority=claim.dirty_priority
    )


def add_supports(claim: Claim, supported_claim_id: str) -> Claim:
    """Pure function to add a claim this claim supports"""
    supports = claim.supports.copy()
    if supported_claim_id not in supports:
        supports.append(supported_claim_id)
    
    return Claim(
        id=claim.id,
        content=claim.content,
        confidence=claim.confidence,
        state=claim.state,
        supported_by=claim.supported_by.copy(),
        supports=supports,
        type=claim.type.copy(),
        tags=claim.tags.copy(),
        created=claim.created,
        updated=datetime.utcnow(),
        embedding=claim.embedding,
        is_dirty=claim.is_dirty,
        dirty_reason=claim.dirty_reason,
        dirty_timestamp=claim.dirty_timestamp,
        dirty_priority=claim.dirty_priority
    )


def mark_dirty(claim: Claim, reason: DirtyReason, priority: int = 0) -> Claim:
    """Pure function to mark claim as dirty for re-evaluation"""
    return Claim(
        id=claim.id,
        content=claim.content,
        confidence=claim.confidence,
        state=claim.state,
        supported_by=claim.supported_by.copy(),
        supports=claim.supports.copy(),
        type=claim.type.copy(),
        tags=claim.tags.copy(),
        created=claim.created,
        updated=datetime.utcnow(),
        embedding=claim.embedding,
        is_dirty=True,
        dirty_reason=reason,
        dirty_timestamp=datetime.utcnow(),
        dirty_priority=priority
    )


def mark_clean(claim: Claim) -> Claim:
    """Pure function to mark claim as clean (no longer needs re-evaluation)"""
    return Claim(
        id=claim.id,
        content=claim.content,
        confidence=claim.confidence,
        state=claim.state,
        supported_by=claim.supported_by.copy(),
        supports=claim.supports.copy(),
        type=claim.type.copy(),
        tags=claim.tags.copy(),
        created=claim.created,
        updated=datetime.utcnow(),
        embedding=claim.embedding,
        is_dirty=False,
        dirty_reason=None,
        dirty_timestamp=None,
        dirty_priority=0
    )


def set_dirty_priority(claim: Claim, priority: int) -> Claim:
    """Pure function to set dirty priority for evaluation ordering"""
    return Claim(
        id=claim.id,
        content=claim.content,
        confidence=claim.confidence,
        state=claim.state,
        supported_by=claim.supported_by.copy(),
        supports=claim.supports.copy(),
        type=claim.type.copy(),
        tags=claim.tags.copy(),
        created=claim.created,
        updated=datetime.utcnow(),
        embedding=claim.embedding,
        is_dirty=claim.is_dirty,
        dirty_reason=claim.dirty_reason,
        dirty_timestamp=claim.dirty_timestamp,
        dirty_priority=priority if claim.is_dirty else claim.dirty_priority
    )


def should_prioritize(claim: Claim, confidence_threshold: float = 0.90) -> bool:
    """Pure function to check if claim should be prioritized for evaluation"""
    return claim.is_dirty and claim.confidence < confidence_threshold


def find_supporting_claims(claim: Claim, all_claims: List[Claim]) -> List[Claim]:
    """Pure function to find all supporting claims"""
    return [c for c in all_claims if c.id in claim.supported_by]


def find_supported_claims(claim: Claim, all_claims: List[Claim]) -> List[Claim]:
    """Pure function to find all claims supported by this claim"""
    return [c for c in all_claims if c.id in claim.supports]


def calculate_support_strength(claim: Claim, all_claims: List[Claim]) -> Tuple[float, int]:
    """Pure function to calculate support strength from supporting claims"""
    supporting_claims = find_supporting_claims(claim, all_claims)
    if not supporting_claims:
        return 0.0, 0
    
    # Simple strength calculation: average confidence weighted by relationship
    total_confidence = sum(c.confidence for c in supporting_claims)
    avg_confidence = total_confidence / len(supporting_claims)
    
    return avg_confidence, len(supporting_claims)


def validate_relationship_integrity(claim: Claim, all_claims: List[Claim]) -> List[str]:
    """Pure function to validate claim relationships"""
    errors = []
    claim_ids = {c.id for c in all_claims}
    
    # Check if supported_by claim IDs exist
    for claim_id in claim.supported_by:
        if claim_id not in claim_ids:
            errors.append(f"Supporting claim {claim_id} not found")
    
    # Check if supports claim IDs exist
    for claim_id in claim.supports:
        if claim_id not in claim_ids:
            errors.append(f"Supported claim {claim_id} not found")
    
    return errors


def get_claim_hierarchy(claim: Claim, all_claims: List[Claim], max_depth: int = 5) -> Dict[str, Any]:
    """Pure function to get claim hierarchy/relationships"""
    hierarchy = {
        "claim_id": claim.id,
        "confidence": claim.confidence,
        "state": claim.state.value,
        "supports_count": len(claim.supports),
        "supported_by_count": len(claim.supported_by),
        "supporters": [],
        "supported": []
    }
    
    # Get supporters (supporting claims)
    supporter_details = []
    for supporter in find_supporting_claims(claim, all_claims):
        supporter_details.append({
            "id": supporter.id,
            "confidence": supporter.confidence,
            "content": supporter.content[:100] + "..." if len(supporter.content) > 100 else supporter.content
        })
    hierarchy["supporters"] = supporter_details
    
    # Get supported claims
    supported_details = []
    for supported in find_supported_claims(claim, all_claims):
        supported_details.append({
            "id": supported.id,
            "confidence": supported.confidence,
            "content": supported.content[:100] + "..." if len(supported.content) > 100 else supported.content
        })
    hierarchy["supported"] = supported_details
    
    return hierarchy


def batch_update_confidence(claims: List[Claim], updates: Dict[str, float]) -> List[Claim]:
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


def filter_claims_by_confidence(claims: List[Claim], min_confidence: float = 0.0, max_confidence: float = 1.0) -> List[Claim]:
    """Pure function to filter claims by confidence range"""
    return [c for c in claims if min_confidence <= c.confidence <= max_confidence]


def filter_claims_by_type(claims: List[Claim], claim_types: List[str]) -> List[Claim]:
    """Pure function to filter claims by type"""
    target_types = set(claim_types)
    return [c for c in claims if any(t.value in target_types for t in c.type)]


def filter_claims_by_tags(claims: List[Claim], tags: List[str], match_all: bool = False) -> List[Claim]:
    """Pure function to filter claims by tags"""
    target_tags = set(tags)
    
    if match_all:
        return [c for c in claims if target_tags.issubset(set(c.tags))]
    else:
        return [c for c in claims if any(tag in target_tags for tag in c.tags)]