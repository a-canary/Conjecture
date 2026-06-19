# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Stateful manager for claim support relationships.

Wraps the pure functions in relationship_manager.py behind a class-based
interface that maintains an internal index and caches computed metrics.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from src.core.models import Claim
from src.core.relationship_manager import (
    create_super_map,
    create_sub_map,
    detect_circular_dependencies,
    find_orphaned_claims,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RelationshipMetrics:
    """Cached metrics about the relationship graph."""

    total_claims: int
    total_relationships: int
    orphaned_claims: int = 0
    relationship_density: float = 0.0
    avg_subs: float = 0.0
    avg_supers: float = 0.0
    circular_dependency_count: int = 0


@dataclass
class TraversalResult:
    """Result of a BFS/DFS traversal through the relationship graph."""

    visited_claims: List[str] = field(default_factory=list)
    traversal_path: List[str] = field(default_factory=list)
    depth: int = 0
    cycles: List[List[str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Manager class
# ---------------------------------------------------------------------------


class SupportRelationshipManager:
    """Manages support relationships among a collection of :class:`Claim` objects.

    The manager keeps an in-memory index and two adjacency maps:

    * ``_super_map``  – claim_id -> set of super-claim ids (forward direction)
    * ``_sub_map``    – claim_id -> set of sub-claim ids  (backward direction)

    Relationship mutations (``add_relationship`` / ``remove_relationship``)
    update the underlying :class:`Claim` objects **in-place** and invalidate
    the cached :class:`RelationshipMetrics`.
    """

    def __init__(self, claims: List[Claim]) -> None:
        self.claims: List[Claim] = list(claims)
        self.claim_index: Dict[str, Claim] = {c.id: c for c in self.claims}
        self._super_map: Dict[str, Set[str]] = create_super_map(self.claims)
        self._sub_map: Dict[str, Set[str]] = create_sub_map(self.claims)
        self._relationship_metrics: Optional[RelationshipMetrics] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _invalidate_metrics(self) -> None:
        self._relationship_metrics = None

    def _rebuild_maps(self) -> None:
        self._super_map = create_super_map(self.claims)
        self._sub_map = create_sub_map(self.claims)

    # ------------------------------------------------------------------
    # Direct relationship accessors
    # ------------------------------------------------------------------

    def get_sub_claims(self, claim_id: str) -> List[Claim]:
        """Return the direct sub-claims (supporters) of *claim_id*."""
        if claim_id not in self.claim_index:
            return []
        sub_ids = self._sub_map.get(claim_id, set())
        return [self.claim_index[sid] for sid in sub_ids if sid in self.claim_index]

    def get_super_claims(self, claim_id: str) -> List[Claim]:
        """Return the direct super-claims (claims supported by *claim_id*)."""
        if claim_id not in self.claim_index:
            return []
        super_ids = self._super_map.get(claim_id, set())
        return [self.claim_index[sid] for sid in super_ids if sid in self.claim_index]

    # ------------------------------------------------------------------
    # Transitive traversal
    # ------------------------------------------------------------------

    def get_all_sub_ancestors(
        self, claim_id: str, max_depth: int = 100
    ) -> TraversalResult:
        """BFS upward from *claim_id* following sub-edges (toward supporters).

        Returns all transitively reachable sub-claim IDs, excluding *claim_id*
        itself.
        """
        if claim_id not in self.claim_index:
            return TraversalResult()

        visited: List[str] = []
        path: List[str] = []
        cycles: List[List[str]] = []
        seen: Set[str] = {claim_id}
        queue: deque = deque([(claim_id, 0)])
        max_depth_reached = 0

        while queue:
            current_id, depth = queue.popleft()
            if depth > max_depth:
                continue
            max_depth_reached = max(max_depth_reached, depth)

            sub_ids = self._sub_map.get(current_id, set())
            for sub_id in sub_ids:
                if sub_id == claim_id:
                    cycles.append([current_id, sub_id])
                    continue
                if sub_id in seen:
                    cycles.append([current_id, sub_id])
                    continue
                seen.add(sub_id)
                visited.append(sub_id)
                path.append(sub_id)
                queue.append((sub_id, depth + 1))

        return TraversalResult(
            visited_claims=visited,
            traversal_path=path,
            depth=max_depth_reached,
            cycles=cycles,
        )

    def get_all_super_descendants(
        self, claim_id: str, max_depth: int = 100
    ) -> TraversalResult:
        """BFS downward from *claim_id* following super-edges (toward dependents).

        Returns all transitively reachable super-claim IDs, excluding *claim_id*
        itself.
        """
        if claim_id not in self.claim_index:
            return TraversalResult()

        visited: List[str] = []
        path: List[str] = []
        cycles: List[List[str]] = []
        seen: Set[str] = {claim_id}
        queue: deque = deque([(claim_id, 0)])
        max_depth_reached = 0

        while queue:
            current_id, depth = queue.popleft()
            if depth > max_depth:
                continue
            max_depth_reached = max(max_depth_reached, depth)

            super_ids = self._super_map.get(current_id, set())
            for super_id in super_ids:
                if super_id == claim_id:
                    cycles.append([current_id, super_id])
                    continue
                if super_id in seen:
                    cycles.append([current_id, super_id])
                    continue
                seen.add(super_id)
                visited.append(super_id)
                path.append(super_id)
                queue.append((super_id, depth + 1))

        return TraversalResult(
            visited_claims=visited,
            traversal_path=path,
            depth=max_depth_reached,
            cycles=cycles,
        )

    # ------------------------------------------------------------------
    # Cycle detection
    # ------------------------------------------------------------------

    def detect_all_cycles(self) -> List[List[str]]:
        """Return all cycles found in the combined (super + sub) adjacency graph."""
        # Build a combined adjacency map using both directions
        combined: Dict[str, Set[str]] = {}
        for cid in self.claim_index:
            combined[cid] = (
                self._super_map.get(cid, set()) | self._sub_map.get(cid, set())
            )
        return detect_circular_dependencies(combined)

    # ------------------------------------------------------------------
    # Shortest path
    # ------------------------------------------------------------------

    def find_shortest_path(
        self, from_id: str, to_id: str
    ) -> Optional[List[str]]:
        """Return the shortest path between two claims using BFS over all edges.

        Returns ``None`` if either claim does not exist or no path is found.
        Returns ``[from_id]`` if *from_id == to_id*.
        """
        if from_id not in self.claim_index or to_id not in self.claim_index:
            return None
        if from_id == to_id:
            return [from_id]

        # Build combined adjacency for undirected traversal
        visited: Set[str] = {from_id}
        queue: deque = deque([[from_id]])

        while queue:
            path = queue.popleft()
            current = path[-1]
            neighbours = (
                self._super_map.get(current, set())
                | self._sub_map.get(current, set())
            )
            for neighbour in neighbours:
                if neighbour == to_id:
                    return path + [neighbour]
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(path + [neighbour])

        return None

    # ------------------------------------------------------------------
    # Add / remove relationships
    # ------------------------------------------------------------------

    def add_relationship(self, supporter_id: str, supported_id: str) -> bool:
        """Add a support relationship from *supporter_id* to *supported_id*.

        Mutates the underlying :class:`Claim` objects and updates internal maps.
        Returns ``True`` on success, ``False`` on failure (missing claims,
        self-reference, or duplicate).
        """
        if supporter_id not in self.claim_index:
            return False
        if supported_id not in self.claim_index:
            return False
        if supporter_id == supported_id:
            return False

        supporter = self.claim_index[supporter_id]
        supported = self.claim_index[supported_id]

        # Check for duplicate
        if supported_id in supporter.supers:
            return False

        # Mutate in-place (tests assert on the original claim object)
        supporter.supers.append(supported_id)
        supported.subs.append(supporter_id)

        # Update adjacency maps
        self._super_map.setdefault(supporter_id, set()).add(supported_id)
        self._sub_map.setdefault(supported_id, set()).add(supporter_id)

        self._invalidate_metrics()
        return True

    def remove_relationship(self, supporter_id: str, supported_id: str) -> bool:
        """Remove the support relationship from *supporter_id* to *supported_id*.

        Mutates the underlying :class:`Claim` objects and updates internal maps.
        Returns ``True`` on success, ``False`` if either claim is missing or
        the relationship does not exist.
        """
        if supporter_id not in self.claim_index:
            return False
        if supported_id not in self.claim_index:
            return False

        supporter = self.claim_index[supporter_id]
        supported = self.claim_index[supported_id]

        if supported_id not in supporter.supers:
            return False

        supporter.supers.remove(supported_id)
        if supporter_id in supported.subs:
            supported.subs.remove(supporter_id)

        # Update adjacency maps
        self._super_map.get(supporter_id, set()).discard(supported_id)
        self._sub_map.get(supported_id, set()).discard(supporter_id)

        self._invalidate_metrics()
        return True

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_relationship_consistency(self) -> List[str]:
        """Return a list of consistency errors in the current claim graph."""
        errors: List[str] = []
        claim_ids: Set[str] = set(self.claim_index.keys())

        for claim in self.claims:
            for super_id in claim.supers:
                if super_id not in claim_ids:
                    errors.append(
                        f"Claim {claim.id!r} references non-existent super claim {super_id!r}"
                    )
                else:
                    # Check reciprocity
                    super_claim = self.claim_index[super_id]
                    if claim.id not in super_claim.subs:
                        errors.append(
                            f"Relationship {claim.id!r} -> {super_id!r} is not reciprocated "
                            f"(missing {claim.id!r} in subs of {super_id!r})"
                        )
            for sub_id in claim.subs:
                if sub_id not in claim_ids:
                    errors.append(
                        f"Claim {claim.id!r} references non-existent sub claim {sub_id!r}"
                    )
                else:
                    sub_claim = self.claim_index[sub_id]
                    if claim.id not in sub_claim.supers:
                        errors.append(
                            f"Relationship {sub_id!r} -> {claim.id!r} is not reciprocated "
                            f"(missing {claim.id!r} in supers of {sub_id!r})"
                        )

        # Detect cycles in the super_map
        cycles = detect_circular_dependencies(self._super_map)
        for cycle in cycles:
            errors.append(f"Circular dependency detected: {' -> '.join(cycle)}")

        return errors

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_relationship_metrics(self) -> RelationshipMetrics:
        """Return cached (or freshly computed) :class:`RelationshipMetrics`."""
        if self._relationship_metrics is not None:
            return self._relationship_metrics

        total = len(self.claims)
        # Count unique relationships (each super-edge counted once)
        total_relationships = sum(len(v) for v in self._super_map.values())

        orphaned = len(find_orphaned_claims(self.claims))

        if total > 1:
            # max possible directed relationships = n*(n-1)
            max_possible = total * (total - 1)
            density = (total_relationships * 2) / max_possible if max_possible else 0.0
        else:
            density = 0.0

        avg_subs = sum(len(c.subs) for c in self.claims) / total if total else 0.0
        avg_supers = sum(len(c.supers) for c in self.claims) / total if total else 0.0

        cycles = detect_circular_dependencies(self._super_map)

        self._relationship_metrics = RelationshipMetrics(
            total_claims=total,
            total_relationships=total_relationships,
            orphaned_claims=orphaned,
            relationship_density=min(density, 1.0),
            avg_subs=avg_subs,
            avg_supers=avg_supers,
            circular_dependency_count=len(cycles),
        )
        return self._relationship_metrics

    # ------------------------------------------------------------------
    # Depth organisation
    # ------------------------------------------------------------------

    def get_claims_by_depth(self, start_id: str) -> Dict[int, List[str]]:
        """BFS downward from *start_id* and group claim IDs by depth level.

        Depth 0 contains *start_id* itself.  Returns ``{}`` if *start_id*
        is not in the index.
        """
        if start_id not in self.claim_index:
            return {}

        depth_map: Dict[int, List[str]] = {0: [start_id]}
        visited: Set[str] = {start_id}
        queue: deque = deque([(start_id, 0)])

        while queue:
            current_id, depth = queue.popleft()
            super_ids = self._super_map.get(current_id, set())
            for super_id in super_ids:
                if super_id not in visited:
                    visited.add(super_id)
                    next_depth = depth + 1
                    depth_map.setdefault(next_depth, []).append(super_id)
                    queue.append((super_id, next_depth))

        return depth_map

    # ------------------------------------------------------------------
    # Utility operations
    # ------------------------------------------------------------------

    def optimize_for_performance(self) -> None:
        """Rebuild internal maps and invalidate cached metrics."""
        self._rebuild_maps()
        self._invalidate_metrics()

    def refresh(self, new_claims: List[Claim]) -> None:
        """Replace the managed claims with *new_claims* and reset all state."""
        self.claims = list(new_claims)
        self.claim_index = {c.id: c for c in self.claims}
        self._rebuild_maps()
        self._invalidate_metrics()

    def export_relationship_graph(self) -> Dict[str, Any]:
        """Export the relationship graph as plain dicts of sets/lists."""
        return {
            "supers": {cid: list(v) for cid, v in self._super_map.items()},
            "subs": {cid: list(v) for cid, v in self._sub_map.items()},
        }
