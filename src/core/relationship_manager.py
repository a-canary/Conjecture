# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Pure functions for claim relationship management.

All functions are side-effect free: they return new objects and do not
mutate their inputs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from src.core.models import Claim, ClaimState


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RelationshipAnalysis:
    """Summary of a single claim's relationship position in the graph."""

    claim_id: str
    support_strength: float
    support_count: int        # number of sub-claims (supporters)
    supported_count: int      # number of super-claims (claims this supports)
    circular_dependencies: List[List[str]]
    orphaned_claims: bool
    relationship_depth: int
    completeness_score: float


@dataclass
class RelationshipChange:
    """Record of a relationship modification."""

    claim_id: str
    change_type: str
    related_claim_id: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pure relationship helpers
# ---------------------------------------------------------------------------


def establish_bidirectional_relationship(
    claim1: Claim, claim2: Claim
) -> Tuple[Claim, Claim]:
    """Return copies of *claim1* and *claim2* linked bidirectionally.

    claim1 gains claim2.id in its ``supers`` list.
    claim2 gains claim1.id in its ``subs`` list.
    """
    new_supers1 = list(claim1.supers)
    if claim2.id not in new_supers1:
        new_supers1.append(claim2.id)

    new_subs2 = list(claim2.subs)
    if claim1.id not in new_subs2:
        new_subs2.append(claim1.id)

    updated1 = claim1.model_copy(update={"supers": new_supers1})
    updated2 = claim2.model_copy(update={"subs": new_subs2})
    return updated1, updated2


def remove_sub_relationship(claim: Claim, sub_id: str) -> Claim:
    """Return a copy of *claim* with *sub_id* removed from its ``subs`` list."""
    new_subs = [s for s in claim.subs if s != sub_id]
    return claim.model_copy(update={"subs": new_subs})


def remove_super_relationship(claim: Claim, super_id: str) -> Claim:
    """Return a copy of *claim* with *super_id* removed from its ``supers`` list."""
    new_supers = [s for s in claim.supers if s != super_id]
    return claim.model_copy(update={"supers": new_supers})


# ---------------------------------------------------------------------------
# Map builders
# ---------------------------------------------------------------------------


def create_super_map(claims: List[Claim]) -> Dict[str, Set[str]]:
    """Build a forward adjacency map: claim_id -> set of super (parent) ids.

    Only includes edges to claims that are present in *claims*.
    """
    existing_ids: Set[str] = {c.id for c in claims}
    result: Dict[str, Set[str]] = {}
    for claim in claims:
        result[claim.id] = {s for s in claim.supers if s in existing_ids}
    return result


def create_sub_map(claims: List[Claim]) -> Dict[str, Set[str]]:
    """Build a backward adjacency map: claim_id -> set of sub (child) ids.

    Only includes edges to claims that are present in *claims*.
    """
    existing_ids: Set[str] = {c.id for c in claims}
    result: Dict[str, Set[str]] = {}
    for claim in claims:
        result[claim.id] = {s for s in claim.subs if s in existing_ids}
    return result


# ---------------------------------------------------------------------------
# Graph analysis
# ---------------------------------------------------------------------------


def detect_circular_dependencies(
    super_map: Dict[str, Set[str]]
) -> List[List[str]]:
    """Detect cycles in a directed super-map using DFS.

    Returns a list of cycles; each cycle is a list of node IDs that form
    a closed loop.  An empty list means no cycles were found.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {node: WHITE for node in super_map}
    cycles: List[List[str]] = []
    stack: List[str] = []

    def dfs(node: str) -> None:
        color[node] = GRAY
        stack.append(node)
        for neighbour in super_map.get(node, set()):
            if neighbour not in color:
                # neighbour not in super_map at all — skip
                continue
            if color[neighbour] == GRAY:
                # Found a back-edge — record the cycle
                cycle_start = stack.index(neighbour)
                cycles.append(list(stack[cycle_start:]))
            elif color[neighbour] == WHITE:
                dfs(neighbour)
        stack.pop()
        color[node] = BLACK

    for node in list(super_map.keys()):
        if color[node] == WHITE:
            dfs(node)

    return cycles


def find_orphaned_claims(claims: List[Claim]) -> List[Claim]:
    """Return claims that have no sub *and* no super relationships."""
    return [c for c in claims if not c.subs and not c.supers]


def find_root_claims(claims: List[Claim]) -> List[Claim]:
    """Return claims that have supers but no subs (pure supporters / leaf nodes toward root)."""
    return [c for c in claims if c.supers and not c.subs]


def find_leaf_claims(claims: List[Claim]) -> List[Claim]:
    """Return claims that have subs but no supers (top-level claims with supporters)."""
    return [c for c in claims if c.subs and not c.supers]


def calculate_relationship_depth(
    super_map: Dict[str, Set[str]], start_id: str
) -> int:
    """Calculate the maximum depth reachable from *start_id* in *super_map*.

    Handles cycles by tracking visited nodes.  Returns 1 for an isolated node.
    """
    if start_id not in super_map:
        return 0

    visited: Set[str] = set()
    max_depth = [0]

    def dfs(node: str, depth: int) -> None:
        if node in visited:
            max_depth[0] = max(max_depth[0], depth)
            return
        visited.add(node)
        max_depth[0] = max(max_depth[0], depth)
        for neighbour in super_map.get(node, set()):
            dfs(neighbour, depth + 1)
        visited.discard(node)

    dfs(start_id, 1)
    return max_depth[0]


def analyze_claim_relationships(
    claim: Claim, all_claims: List[Claim]
) -> RelationshipAnalysis:
    """Produce a :class:`RelationshipAnalysis` for *claim* within *all_claims*."""
    existing_ids: Set[str] = {c.id for c in all_claims}

    valid_subs = [s for s in claim.subs if s in existing_ids]
    valid_supers = [s for s in claim.supers if s in existing_ids]

    support_count = len(valid_subs)
    supported_count = len(valid_supers)
    orphaned = (support_count == 0) and (supported_count == 0)

    # completeness: 0.5 per direction present
    if orphaned:
        completeness = 0.0
    elif support_count > 0 and supported_count > 0:
        completeness = 1.0
    else:
        completeness = 0.5

    super_map = create_super_map(all_claims)
    depth = calculate_relationship_depth(super_map, claim.id)

    avg_sub_confidence = 0.0
    if valid_subs:
        claim_by_id = {c.id: c for c in all_claims}
        confidences = [claim_by_id[s].confidence for s in valid_subs if s in claim_by_id]
        avg_sub_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    cycles = detect_circular_dependencies(super_map)

    return RelationshipAnalysis(
        claim_id=claim.id,
        support_strength=avg_sub_confidence,
        support_count=support_count,
        supported_count=supported_count,
        circular_dependencies=cycles,
        orphaned_claims=orphaned,
        relationship_depth=depth,
        completeness_score=completeness,
    )


# ---------------------------------------------------------------------------
# Suggestion
# ---------------------------------------------------------------------------


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Simple word-level Jaccard similarity between two strings."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a and not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def suggest_sub_relationships(
    claim: Claim,
    all_claims: List[Claim],
    max_suggestions: int = 5,
    similarity_threshold: float = 0.1,
) -> List[Tuple[Claim, float]]:
    """Suggest candidates that could act as sub-claims (supporters) for *claim*.

    Returns a sorted list of ``(candidate_claim, similarity_score)`` tuples,
    excluding *claim* itself and existing subs.
    """
    existing_subs: Set[str] = set(claim.subs)
    candidates: List[Tuple[Claim, float]] = []

    for candidate in all_claims:
        if candidate.id == claim.id:
            continue
        if candidate.id in existing_subs:
            continue
        score = _jaccard_similarity(claim.content, candidate.content)
        if score >= similarity_threshold:
            candidates.append((candidate, score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:max_suggestions]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_relationship_consistency(claims: List[Claim]) -> List[str]:
    """Check the claim graph for structural problems.

    Returns a list of human-readable error strings (empty = no errors).
    """
    errors: List[str] = []
    claim_ids: Set[str] = {c.id for c in claims}

    for claim in claims:
        # Check for references to non-existent claims
        for super_id in claim.supers:
            if super_id not in claim_ids:
                errors.append(
                    f"Claim {claim.id!r} references non-existent super claim {super_id!r}"
                )
        for sub_id in claim.subs:
            if sub_id not in claim_ids:
                errors.append(
                    f"Claim {claim.id!r} references non-existent sub claim {sub_id!r}"
                )

        # Orphaned but validated claims are suspicious
        if (
            not claim.supers
            and not claim.subs
            and claim.state == ClaimState.VALIDATED
        ):
            errors.append(
                f"Claim {claim.id!r} is validated but has no relationships (isolated validated claim)"
            )

    # Detect circular dependencies
    super_map = create_super_map(claims)
    cycles = detect_circular_dependencies(super_map)
    for cycle in cycles:
        errors.append(f"Circular dependency detected: {' -> '.join(cycle)}")

    return errors


# ---------------------------------------------------------------------------
# Confidence propagation
# ---------------------------------------------------------------------------


def propagate_confidence_updates(
    updates: Dict[str, float],
    claims: List[Claim],
    propagation_factor: float = 0.1,
    iterations: int = 3,
) -> List[Claim]:
    """Propagate confidence changes through the relationship graph.

    *updates* maps claim_id -> new confidence value for directly updated claims.
    Neighbouring claims receive a fractional adjustment based on *propagation_factor*.
    Returns a new list of :class:`Claim` objects (original objects are not mutated).
    """
    # Start with copies
    claim_map: Dict[str, Claim] = {c.id: c.model_copy() for c in claims}

    # Apply direct updates
    for cid, new_conf in updates.items():
        if cid in claim_map:
            claim_map[cid] = claim_map[cid].model_copy(update={"confidence": new_conf})

    if not updates:
        return list(claim_map.values())

    super_map = create_super_map(claims)
    sub_map = create_sub_map(claims)

    for _ in range(iterations):
        new_claim_map = {cid: c.model_copy() for cid, c in claim_map.items()}
        for cid, claim in claim_map.items():
            neighbours: Set[str] = (
                super_map.get(cid, set()) | sub_map.get(cid, set())
            )
            for neighbour_id in neighbours:
                if neighbour_id not in claim_map:
                    continue
                neighbour = claim_map[neighbour_id]
                # Nudge neighbour toward claim's confidence
                delta = (claim.confidence - neighbour.confidence) * propagation_factor
                new_conf = max(0.0, min(1.0, neighbour.confidence + delta))
                new_claim_map[neighbour_id] = new_claim_map[neighbour_id].model_copy(
                    update={"confidence": new_conf}
                )
        claim_map = new_claim_map

    return list(claim_map.values())


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------


def create_relationship_heatmap(
    claims: List[Claim],
) -> Dict[str, Dict[str, float]]:
    """Build a heatmap of sub-claim confidences per claim.

    For each claim, maps the IDs of its sub-claims to their confidence values.
    """
    claim_by_id: Dict[str, Claim] = {c.id: c for c in claims}
    heatmap: Dict[str, Dict[str, float]] = {}

    for claim in claims:
        row: Dict[str, float] = {}
        for sub_id in claim.subs:
            if sub_id in claim_by_id:
                row[sub_id] = claim_by_id[sub_id].confidence
        heatmap[claim.id] = row

    return heatmap


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def get_relationship_statistics(claims: List[Claim]) -> Dict[str, Any]:
    """Compute high-level relationship statistics for a collection of claims."""
    if not claims:
        return {"error": "No claims provided"}

    total = len(claims)
    orphaned_list = find_orphaned_claims(claims)
    orphaned_count = len(orphaned_list)
    with_relationships = total - orphaned_count

    root_list = find_root_claims(claims)
    leaf_list = find_leaf_claims(claims)

    super_map = create_super_map(claims)
    cycles = detect_circular_dependencies(super_map)

    avg_subs = sum(len(c.subs) for c in claims) / total
    avg_supers = sum(len(c.supers) for c in claims) / total

    relationship_coverage = (with_relationships / total) * 100.0

    return {
        "total_claims": total,
        "with_relationships": with_relationships,
        "orphaned": orphaned_count,
        "root_claims": len(root_list),
        "leaf_claims": len(leaf_list),
        "circular_dependencies": len(cycles),
        "relationship_coverage": relationship_coverage,
        "avg_subs": avg_subs,
        "avg_supers": avg_supers,
    }
