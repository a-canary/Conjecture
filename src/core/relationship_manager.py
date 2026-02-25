"""
Claim Relationship Manager - Pure Functions for Claim Relationships
Manages subs and supers relationships with validation and analysis.
"""
from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timezone

from .models import Claim, ClaimType, ClaimState
from .claim_operations import (
    add_sub, add_super, find_sub_claims,
    find_super_claims, validate_relationship_integrity
)

@dataclass
class RelationshipAnalysis:
    """Pure data structure for relationship analysis results."""
    claim_id: str
    support_strength: float
    support_count: int
    supported_count: int
    circular_dependencies: List[str]
    orphaned_claims: bool
    relationship_depth: int
    completeness_score: float

@dataclass
class RelationshipChange:
    """Pure data structure for relationship changes."""
    claim_id: str
    change_type: str  # "add_sub", "remove_sub", "add_super", "remove_super"
    related_claim_id: str
    timestamp: datetime
    metadata: Dict[str, Any]

# Pure Functions for Relationship Management

def establish_bidirectional_relationship(claim: Claim, related_claim: Claim) -> Tuple[Claim, Claim]:
    """Pure function to establish bidirectional relationship between claims."""
    # Claim adds related_claim as a super (claim provides evidence FOR related_claim)
    updated_claim = add_super(claim, related_claim.id)
    # Related claim adds claim as a sub (claim provides evidence FOR related_claim)
    updated_related = add_sub(related_claim, claim.id)

    return updated_claim, updated_related

def remove_sub_relationship(claim: Claim, sub_claim_id: str) -> Claim:
    """Pure function to remove a sub relationship."""
    new_subs = [cid for cid in claim.subs if cid != sub_claim_id]

    return Claim(
        id=claim.id,
        content=claim.content,
        confidence=claim.confidence,
        state=claim.state,
        subs=new_subs,
        supers=claim.supers.copy(),
        type=claim.type.copy(),
        tags=claim.tags.copy(),
        created=claim.created,
        updated=datetime.now(timezone.utc),
        embedding=claim.embedding,
        is_dirty=claim.is_dirty,
        dirty_reason=claim.dirty_reason,
        dirty_timestamp=claim.dirty_timestamp,
        dirty_priority=claim.dirty_priority
    )

def remove_super_relationship(claim: Claim, super_claim_id: str) -> Claim:
    """Pure function to remove a super relationship."""
    new_supers = [cid for cid in claim.supers if cid != super_claim_id]

    return Claim(
        id=claim.id,
        content=claim.content,
        confidence=claim.confidence,
        state=claim.state,
        subs=claim.subs.copy(),
        supers=new_supers,
        type=claim.type.copy(),
        tags=claim.tags.copy(),
        created=claim.created,
        updated=datetime.now(timezone.utc),
        embedding=claim.embedding,
        is_dirty=claim.is_dirty,
        dirty_reason=claim.dirty_reason,
        dirty_timestamp=claim.dirty_timestamp,
        dirty_priority=claim.dirty_priority
    )

def create_super_map(claims: List[Claim]) -> Dict[str, Set[str]]:
    """Pure function to create a super relationship map (claim -> claims it provides evidence FOR)."""
    super_map = {}
    claim_ids = {claim.id for claim in claims}

    for claim in claims:
        super_map[claim.id] = set()
        for super_id in claim.supers:
            if super_id in claim_ids:
                super_map[claim.id].add(super_id)

    return super_map

def create_sub_map(claims: List[Claim]) -> Dict[str, Set[str]]:
    """Pure function to create a sub relationship map (claim -> claims that provide evidence FOR it)."""
    sub_map = {}
    claim_ids = {claim.id for claim in claims}

    for claim in claims:
        sub_map[claim.id] = set()
        for sub_id in claim.subs:
            if sub_id in claim_ids:
                sub_map[claim.id].add(sub_id)

    return sub_map

def detect_circular_dependencies(super_map: Dict[str, Set[str]]) -> List[List[str]]:
    """Pure function to detect circular dependencies using DFS."""
    visited = set()
    rec_stack = set()
    cycles = []

    def dfs(node: str, path: List[str]) -> bool:
        if node in rec_stack:
            # Found a cycle
            cycle_start = path.index(node)
            cycles.append(path[cycle_start:] + [node])
            return True

        if node in visited:
            return False

        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for neighbor in super_map.get(node, set()):
            if dfs(neighbor, path):
                return True

        rec_stack.remove(node)
        path.pop()
        return False

    for claim_id in super_map:
        if claim_id not in visited:
            dfs(claim_id, [])

    return cycles

def find_orphaned_claims(claims: List[Claim]) -> List[Claim]:
    """Pure function to find claims that have no relationships."""
    return [
        claim for claim in claims
        if not claim.subs and not claim.supers
    ]

def find_root_claims(claims: List[Claim]) -> List[Claim]:
    """Pure function to find root claims (claims that have supers but no subs - toward the root)."""
    return [
        claim for claim in claims
        if claim.supers and not claim.subs
    ]

def find_leaf_claims(claims: List[Claim]) -> List[Claim]:
    """Pure function to find leaf claims (claims that have subs but no supers - at the leaf level)."""
    return [
        claim for claim in claims
        if claim.subs and not claim.supers
    ]

def calculate_relationship_depth(super_map: Dict[str, Set[str]], root_claim_id: str) -> int:
    """Pure function to calculate the maximum depth from a root claim."""
    def dfs_depth(node: str, visited: Set[str]) -> int:
        if node in visited:
            return 0  # Cycle detected

        visited.add(node)

        if not super_map.get(node):
            return 1

        max_depth = 0
        for child in super_map[node]:
            child_depth = dfs_depth(child, visited.copy())
            max_depth = max(max_depth, child_depth)

        return max_depth + 1

    return dfs_depth(root_claim_id, set())

def analyze_claim_relationships(claim: Claim, all_claims: List[Claim]) -> RelationshipAnalysis:
    """Pure function to analyze a claim's relationships."""
    from .claim_operations import calculate_support_strength

    # Get support strength
    support_strength, support_count = calculate_support_strength(claim, all_claims)

    # Find circular dependencies affecting this claim
    super_map = create_super_map(all_claims)
    cycles = detect_circular_dependencies(super_map)

    claim_cycles = [
        cycle for cycle in cycles
        if claim.id in cycle
    ]

    # Check if orphaned
    is_orphaned = not claim.subs and not claim.supers

    # Calculate relationship depth
    if claim.supers:
        depth = calculate_relationship_depth(super_map, claim.id)
    else:
        depth = 0

    # Calculate completeness score
    sub_claims = find_sub_claims(claim, all_claims)
    super_claims = find_super_claims(claim, all_claims)

    # Completeness based on having both subs and supers
    completeness = 0.0
    if sub_claims and super_claims:
        completeness = 1.0
    elif sub_claims or super_claims:
        completeness = 0.5

    return RelationshipAnalysis(
        claim_id=claim.id,
        support_strength=support_strength,
        support_count=support_count,
        supported_count=len(claim.supers),
        circular_dependencies=[cycle for cycle in claim_cycles if len(cycle) > 0],
        orphaned_claims=is_orphaned,
        relationship_depth=depth,
        completeness_score=completeness
    )

def suggest_sub_relationships(claim: Claim, all_claims: List[Claim], max_suggestions: int = 5) -> List[Tuple[Claim, float]]:
    """Pure function to suggest sub relationships based on content similarity."""
    # Simple content-based suggestion (could be enhanced with embeddings)
    suggestions = []
    claim_words = set(claim.content.lower().split())

    for other_claim in all_claims:
        if other_claim.id == claim.id or other_claim.id in claim.subs:
            continue

        # Calculate word overlap as similarity score
        other_words = set(other_claim.content.lower().split())
        overlap = len(claim_words.intersection(other_words))
        similarity = overlap / max(len(claim_words), len(other_words), 1)

        # Only suggest if there's some similarity
        if similarity > 0.1:
            suggestions.append((other_claim, similarity))

    # Sort by similarity and return top suggestions
    suggestions.sort(key=lambda x: x[1], reverse=True)
    return suggestions[:max_suggestions]

def validate_relationship_consistency(claims: List[Claim]) -> List[str]:
    """Pure function to validate overall relationship consistency."""
    errors = []
    claim_ids = {claim.id for claim in claims}

    # Check each claim's relationships
    for claim in claims:
        claim_errors = validate_relationship_integrity(claim, claims)
        errors.extend([f"Claim {claim.id}: {error}" for error in claim_errors])

    # Check for circular dependencies
    super_map = create_super_map(claims)
    cycles = detect_circular_dependencies(super_map)
    if cycles:
        for i, cycle in enumerate(cycles):
            errors.append(f"Circular dependency #{i+1}: {' -> '.join(cycle)}")

    # Check for orphaned validated claims
    orphaned_validated = [
        claim for claim in find_orphaned_claims(claims)
        if claim.state == ClaimState.VALIDATED
    ]
    if orphaned_validated:
        errors.append(f"Found {len(orphaned_validated)} validated claims with no relationships")

    return errors

def propagate_confidence_updates(claim_updates: Dict[str, float], all_claims: List[Claim], propagation_factor: float = 0.1) -> List[Claim]:
    """Pure function to propagate confidence updates through sub relationships."""
    if not claim_updates:
        return all_claims.copy()

    updated_claims = all_claims.copy()
    claim_map = {claim.id: [i, claim] for i, claim in enumerate(updated_claims)}

    # Create sub map for propagation
    sub_map = create_sub_map(updated_claims)

    # Apply direct updates first
    for claim_id, new_confidence in claim_updates.items():
        if claim_id in claim_map:
            idx, claim = claim_map[claim_id]
            from .claim_operations import update_confidence
            updated_claims[idx] = update_confidence(claim, new_confidence)

    # Propagate updates to subs
    max_iterations = 3  # Limit propagation depth
    for iteration in range(max_iterations):
        changes_made = False

        for claim_id, new_confidence in claim_updates.items():
            for sub_id in sub_map.get(claim_id, set()):
                if sub_id in claim_map:
                    idx, sub_claim = claim_map[updated_claims[idx].id]
                    # Apply propagation factor
                    confidence_change = (new_confidence - sub_claim.confidence) * propagation_factor
                    new_sub_confidence = min(1.0, max(0.0, sub_claim.confidence + confidence_change))

                    if abs(new_sub_confidence - sub_claim.confidence) > 0.01:
                        from .claim_operations import update_confidence
                        updated_claims[idx] = update_confidence(updated_claims[idx], new_sub_confidence)
                        changes_made = True

        if not changes_made:
            break

    return updated_claims

def create_relationship_heatmap(claims: List[Claim]) -> Dict[str, Dict[str, float]]:
    """Pure function to create a relationship strength heatmap."""
    heatmap = {}

    for claim in claims:
        heatmap[claim.id] = {}
        subs = find_sub_claims(claim, claims)

        for sub in subs:
            # Use sub's confidence as relationship strength
            heatmap[claim.id][sub.id] = sub.confidence

    return heatmap

def get_relationship_statistics(claims: List[Claim]) -> Dict[str, Any]:
    """Pure function to get comprehensive relationship statistics."""
    total_claims = len(claims)
    if total_claims == 0:
        return {"error": "No claims provided"}

    # Basic counts
    with_relationships = [c for c in claims if c.subs or c.supers]
    supers_only = [c for c in claims if c.supers and not c.subs]
    subs_only = [c for c in claims if c.subs and not c.supers]
    both_relations = [c for c in claims if c.subs and c.supers]
    orphaned = find_orphaned_claims(claims)

    # Relationship chains
    root_claims = find_root_claims(claims)
    leaf_claims = find_leaf_claims(claims)

    # Circular dependencies
    super_map = create_super_map(claims)
    cycles = detect_circular_dependencies(super_map)

    # Average relationship counts
    avg_subs = sum(len(c.subs) for c in claims) / total_claims
    avg_supers = sum(len(c.supers) for c in claims) / total_claims

    return {
        "total_claims": total_claims,
        "with_relationships": len(with_relationships),
        "supers_only": len(supers_only),
        "subs_only": len(subs_only),
        "both_relations": len(both_relations),
        "orphaned": len(orphaned),
        "root_claims": len(root_claims),
        "leaf_claims": len(leaf_claims),
        "circular_dependencies": len(cycles),
        "avg_subs": round(avg_subs, 2),
        "avg_supers": round(avg_supers, 2),
        "relationship_coverage": round(len(with_relationships) / total_claims * 100, 1)
    }