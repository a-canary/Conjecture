"""
Claim Relationship Manager - Pure Functions for Claim Relationships
Manages supported_by and supports relationships with validation and analysis.
"""
from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

from .models import Claim, ClaimType, ClaimState
from .claim_operations import (
    add_support, add_supports, find_supporting_claims, 
    find_supported_claims, validate_relationship_integrity
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
    change_type: str  # "add_support", "remove_support", "add_supports", "remove_supports"
    related_claim_id: str
    timestamp: datetime
    metadata: Dict[str, Any]


# Pure Functions for Relationship Management

def establish_bidirectional_relationship(claim: Claim, related_claim: Claim) -> Tuple[Claim, Claim]:
    """Pure function to establish bidirectional relationship between claims."""
    # Claim supports related_claim
    updated_claim = add_supports(claim, related_claim.id)
    # Related claim is supported_by claim  
    updated_related = add_support(related_claim, claim.id)
    
    return updated_claim, updated_related


def remove_support_relationship(claim: Claim, supporting_claim_id: str) -> Claim:
    """Pure function to remove a support relationship."""
    new_supported_by = [cid for cid in claim.supported_by if cid != supporting_claim_id]
    
    return Claim(
        id=claim.id,
        content=claim.content,
        confidence=claim.confidence,
        state=claim.state,
        supported_by=new_supported_by,
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


def remove_supports_relationship(claim: Claim, supported_claim_id: str) -> Claim:
    """Pure function to remove a supports relationship."""
    new_supports = [cid for cid in claim.supports if cid != supported_claim_id]
    
    return Claim(
        id=claim.id,
        content=claim.content,
        confidence=claim.confidence,
        state=claim.state,
        supported_by=claim.supported_by.copy(),
        supports=new_supports,
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


def create_support_map(claims: List[Claim]) -> Dict[str, Set[str]]:
    """Pure function to create a support relationship map."""
    support_map = {}
    claim_ids = {claim.id for claim in claims}
    
    for claim in claims:
        support_map[claim.id] = set()
        for supported_id in claim.supports:
            if supported_id in claim_ids:
                support_map[claim.id].add(supported_id)
    
    return support_map


def create_supporter_map(claims: List[Claim]) -> Dict[str, Set[str]]:
    """Pure function to create a supporter relationship map."""
    supporter_map = {}
    claim_ids = {claim.id for claim in claims}
    
    for claim in claims:
        supporter_map[claim.id] = set()
        for supporter_id in claim.supported_by:
            if supporter_id in claim_ids:
                supporter_map[claim.id].add(supporter_id)
    
    return supporter_map


def detect_circular_dependencies(support_map: Dict[str, Set[str]]) -> List[List[str]]:
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
        
        for neighbor in support_map.get(node, set()):
            if dfs(neighbor, path):
                return True
        
        rec_stack.remove(node)
        path.pop()
        return False
    
    for claim_id in support_map:
        if claim_id not in visited:
            dfs(claim_id, [])
    
    return cycles


def find_orphaned_claims(claims: List[Claim]) -> List[Claim]:
    """Pure function to find claims that have no relationships."""
    return [
        claim for claim in claims
        if not claim.supported_by and not claim.supports
    ]


def find_root_claims(claims: List[Claim]) -> List[Claim]:
    """Pure function to find root claims (claims that support others but are not supported)."""
    return [
        claim for claim in claims
        if claim.supports and not claim.supported_by
    ]


def find_leaf_claims(claims: List[Claim]) -> List[Claim]:
    """Pure function to find leaf claims (claims that are supported but don't support others)."""
    return [
        claim for claim in claims
        if claim.supported_by and not claim.supports
    ]


def calculate_relationship_depth(support_map: Dict[str, Set[str]], root_claim_id: str) -> int:
    """Pure function to calculate the maximum depth from a root claim."""
    def dfs_depth(node: str, visited: Set[str]) -> int:
        if node in visited:
            return 0  # Cycle detected
        
        visited.add(node)
        
        if not support_map.get(node):
            return 1
        
        max_depth = 0
        for child in support_map[node]:
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
    support_map = create_support_map(all_claims)
    cycles = detect_circular_dependencies(support_map)
    
    claim_cycles = [
        cycle for cycle in cycles
        if claim.id in cycle
    ]
    
    # Check if orphaned
    is_orphaned = not claim.supported_by and not claim.supports
    
    # Calculate relationship depth
    if claim.supports:
        depth = calculate_relationship_depth(support_map, claim.id)
    else:
        depth = 0
    
    # Calculate completeness score
    supporting_claims = find_supporting_claims(claim, all_claims)
    supported_claims = find_supported_claims(claim, all_claims)
    
    # Completeness based on having both supporters and supported claims
    completeness = 0.0
    if supporting_claims and supported_claims:
        completeness = 1.0
    elif supporting_claims or supported_claims:
        completeness = 0.5
    
    return RelationshipAnalysis(
        claim_id=claim.id,
        support_strength=support_strength,
        support_count=support_count,
        supported_count=len(claim.supports),
        circular_dependencies=[cycle for cycle in claim_cycles if len(cycle) > 0],
        orphaned_claims=is_orphaned,
        relationship_depth=depth,
        completeness_score=completeness
    )


def suggest_support_relationships(claim: Claim, all_claims: List[Claim], max_suggestions: int = 5) -> List[Tuple[Claim, float]]:
    """Pure function to suggest support relationships based on content similarity."""
    # Simple content-based suggestion (could be enhanced with embeddings)
    suggestions = []
    claim_words = set(claim.content.lower().split())
    
    for other_claim in all_claims:
        if other_claim.id == claim.id or other_claim.id in claim.supported_by:
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
    support_map = create_support_map(claims)
    cycles = detect_circular_dependencies(support_map)
    if cycles:
        for i, cycle in enumerate(cycles):
            errors.append(f"Circular dependency #{i+1}: {' → '.join(cycle)}")
    
    # Check for orphaned validated claims
    orphaned_validated = [
        claim for claim in find_orphaned_claims(claims)
        if claim.state == ClaimState.VALIDATED
    ]
    if orphaned_validated:
        errors.append(f"Found {len(orphaned_validated)} validated claims with no relationships")
    
    return errors


def propagate_confidence_updates(claim_updates: Dict[str, float], all_claims: List[Claim], propagation_factor: float = 0.1) -> List[Claim]:
    """Pure function to propagate confidence updates through support relationships."""
    if not claim_updates:
        return all_claims.copy()
    
    updated_claims = all_claims.copy()
    claim_map = {claim.id: [i, claim] for i, claim in enumerate(updated_claims)}
    
    # Create supporter map for propagation
    supporter_map = create_supporter_map(updated_claims)
    
    # Apply direct updates first
    for claim_id, new_confidence in claim_updates.items():
        if claim_id in claim_map:
            idx, claim = claim_map[claim_id]
            from .claim_operations import update_confidence
            updated_claims[idx] = update_confidence(claim, new_confidence)
    
    # Propagate updates to supporters
    max_iterations = 3  # Limit propagation depth
    for iteration in range(max_iterations):
        changes_made = False
        
        for claim_id, new_confidence in claim_updates.items():
            for supporter_id in supporter_map.get(claim_id, set()):
                if supporter_id in claim_map:
                    idx, supporter_claim = claim_map[updated_claims[idx].id]
                    # Apply propagation factor
                    confidence_change = (new_confidence - supporter_claim.confidence) * propagation_factor
                    new_supporter_confidence = min(1.0, max(0.0, supporter_claim.confidence + confidence_change))
                    
                    if abs(new_supporter_confidence - supporter_claim.confidence) > 0.01:
                        from .claim_operations import update_confidence
                        updated_claims[idx] = update_confidence(updated_claims[idx], new_supporter_confidence)
                        changes_made = True
        
        if not changes_made:
            break
    
    return updated_claims


def create_relationship_heatmap(claims: List[Claim]) -> Dict[str, Dict[str, float]]:
    """Pure function to create a relationship strength heatmap."""
    heatmap = {}
    
    for claim in claims:
        heatmap[claim.id] = {}
        supporting = find_supporting_claims(claim, claims)
        
        for supporter in supporting:
            # Use supporter's confidence as relationship strength
            heatmap[claim.id][supporter.id] = supporter.confidence
    
    return heatmap


def get_relationship_statistics(claims: List[Claim]) -> Dict[str, Any]:
    """Pure function to get comprehensive relationship statistics."""
    total_claims = len(claims)
    if total_claims == 0:
        return {"error": "No claims provided"}
    
    # Basic counts
    with_relationships = [c for c in claims if c.supported_by or c.supports]
    supporting_only = [c for c in claims if c.supports and not c.supported_by]
    supported_only = [c for c in claims if c.supported_by and not c.supports]
    both_relations = [c for c in claims if c.supported_by and c.supports]
    orphaned = find_orphaned_claims(claims)
    
    # Support chains
    root_claims = find_root_claims(claims)
    leaf_claims = find_leaf_claims(claims)
    
    # Circular dependencies
    support_map = create_support_map(claims)
    cycles = detect_circular_dependencies(support_map)
    
    # Average support counts
    avg_supported_by = sum(len(c.supported_by) for c in claims) / total_claims
    avg_supports = sum(len(c.supports) for c in claims) / total_claims
    
    return {
        "total_claims": total_claims,
        "with_relationships": len(with_relationships),
        "supporting_only": len(supporting_only),
        "supported_only": len(supported_only),
        "both_relations": len(both_relations),
        "orphaned": len(orphaned),
        "root_claims": len(root_claims),
        "leaf_claims": len(leaf_claims),
        "circular_dependencies": len(cycles),
        "avg_supported_by": round(avg_supported_by, 2),
        "avg_supports": round(avg_supports, 2),
        "relationship_coverage": round(len(with_relationships) / total_claims * 100, 1)
    }