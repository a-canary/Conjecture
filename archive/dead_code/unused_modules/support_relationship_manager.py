# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Support Relationship Manager for Simplified Universal Claim Architecture
Handles efficient bidirectional relationship traversal, validation, and optimization

Naming convention:
- subs: claims that provide evidence FOR this claim (children)
- supers: claims this claim provides evidence FOR (toward root, parents)
"""

from typing import List, Dict, Set, Tuple, Optional
from datetime import datetime, timezone
from dataclasses import dataclass
from collections import defaultdict, deque

from .models import Claim


@dataclass
class RelationshipMetrics:
    """Performance and structure metrics for relationships"""

    total_claims: int
    total_relationships: int
    max_depth: int
    cycles_detected: int
    orphaned_claims: int
    avg_branching_factor: float
    relationship_density: float


@dataclass
class TraversalResult:
    """Result of relationship traversal operation"""

    visited_claims: List[str]
    traversal_path: List[str]
    depth: int
    cycles: List[List[str]]


class SupportRelationshipManager:
    """
    Manages support relationships between claims with optimized traversal and validation.

    Key features:
    - Efficient bidirectional relationship traversal
    - Circular dependency detection
    - Relationship validation and consistency checking
    - Performance optimization for large claim networks

    Relationship semantics:
    - subs: claims that provide evidence FOR this claim (children)
    - supers: claims this provides evidence FOR (toward root, parents)
    """

    def __init__(self, claims: List[Claim]):
        """Initialize with a list of claims"""
        self.claims = claims
        self.claim_index = {claim.id: claim for claim in claims}
        self._super_map = None
        self._sub_map = None
        self._relationship_metrics = None

        # Build optimized lookup structures
        self._build_relationship_maps()

    def _build_relationship_maps(self) -> None:
        """Build optimized relationship lookup maps"""
        self._super_map = defaultdict(set)  # claim_id -> set of super claim IDs (claims this provides evidence FOR)
        self._sub_map = defaultdict(set)  # claim_id -> set of sub claim IDs (claims that provide evidence FOR this)

        for claim in self.claims:
            # Build forward super map (claim -> claims it provides evidence FOR)
            for super_id in claim.supers:
                if super_id in self.claim_index:
                    self._super_map[claim.id].add(super_id)

            # Build backward sub map (claim -> claims that provide evidence FOR it)
            for sub_id in claim.subs:
                if sub_id in self.claim_index:
                    self._sub_map[claim.id].add(sub_id)

    def get_sub_claims(self, claim_id: str) -> List[Claim]:
        """Get all claims that provide evidence FOR the given claim (subs, children)"""
        if claim_id not in self.claim_index:
            return []

        subs = []
        for sub_id in self._sub_map[claim_id]:
            subs.append(self.claim_index[sub_id])
        return subs

    def get_super_claims(self, claim_id: str) -> List[Claim]:
        """Get all claims that this claim provides evidence FOR (supers, toward root)"""
        if claim_id not in self.claim_index:
            return []

        supers = []
        for super_id in self._super_map[claim_id]:
            supers.append(self.claim_index[super_id])
        return supers

    def get_all_sub_ancestors(
        self, claim_id: str, max_depth: int = 100
    ) -> TraversalResult:
        """
        Get all claims that provide evidence FOR the given claim (transitive closure of subs)
        Traverses down through the sub tree
        """
        if claim_id not in self.claim_index:
            return TraversalResult([], [], 0, [])

        visited = set()
        traversal_path = []
        cycles = []
        depth = 0
        queue = deque([(claim_id, 0, [claim_id])])  # (claim_id, depth, path)

        while queue and depth < max_depth:
            current_id, current_depth, current_path = queue.popleft()

            if current_id in visited:
                continue

            visited.add(current_id)
            traversal_path.append(current_id)
            depth = max(depth, current_depth)

            # Get subs (claims that provide evidence FOR current claim)
            subs = self._sub_map[current_id]
            for sub_id in subs:
                new_path = current_path + [sub_id]

                # Check for cycles
                if sub_id in current_path:
                    cycle_start = current_path.index(sub_id)
                    cycle = current_path[cycle_start:] + [sub_id]
                    cycles.append(cycle)
                    continue

                queue.append((sub_id, current_depth + 1, new_path))

        # Remove the starting claim from ancestors
        visited.discard(claim_id)
        traversal_path = [c for c in traversal_path if c != claim_id]

        return TraversalResult(list(visited), traversal_path, depth, cycles)

    def get_all_super_descendants(
        self, claim_id: str, max_depth: int = 100
    ) -> TraversalResult:
        """
        Get all claims this claim provides evidence FOR (transitive closure of supers)
        Traverses up toward root claims
        """
        if claim_id not in self.claim_index:
            return TraversalResult([], [], 0, [])

        visited = set()
        traversal_path = []
        cycles = []
        depth = 0
        queue = deque([(claim_id, 0, [claim_id])])  # (claim_id, depth, path)

        while queue and depth < max_depth:
            current_id, current_depth, current_path = queue.popleft()

            if current_id in visited:
                continue

            visited.add(current_id)
            traversal_path.append(current_id)
            depth = max(depth, current_depth)

            # Get super claims (claims this claim provides evidence FOR)
            supers = self._super_map[current_id]
            for super_id in supers:
                new_path = current_path + [super_id]

                # Check for cycles
                if super_id in current_path:
                    cycle_start = current_path.index(super_id)
                    cycle = current_path[cycle_start:] + [super_id]
                    cycles.append(cycle)
                    continue

                queue.append((super_id, current_depth + 1, new_path))

        # Remove the starting claim from descendants
        visited.discard(claim_id)
        traversal_path = [c for c in traversal_path if c != claim_id]

        return TraversalResult(list(visited), traversal_path, depth, cycles)

    def find_shortest_path(self, from_id: str, to_id: str) -> Optional[List[str]]:
        """Find shortest path between two claims using BFS"""
        if from_id not in self.claim_index or to_id not in self.claim_index:
            return None

        if from_id == to_id:
            return [from_id]

        visited = set()
        queue = deque([(from_id, [from_id])])

        while queue:
            current_id, path = queue.popleft()

            if current_id in visited:
                continue

            visited.add(current_id)

            # Check both subs and supers
            neighbors = list(self._sub_map[current_id]) + list(
                self._super_map[current_id]
            )

            for neighbor_id in neighbors:
                if neighbor_id == to_id:
                    return path + [neighbor_id]

                if neighbor_id not in visited:
                    queue.append((neighbor_id, path + [neighbor_id]))

        return None

    def detect_all_cycles(self) -> List[List[str]]:
        """Detect all circular dependencies in the relationship network"""
        visited = set()
        rec_stack = set()
        cycles = []

        for claim_id in self.claim_index:
            if claim_id not in visited:
                self._dfs_cycle_detection(claim_id, [], visited, rec_stack, cycles)

        return cycles

    def _dfs_cycle_detection(
        self,
        node: str,
        path: List[str],
        visited: Set[str],
        rec_stack: Set[str],
        cycles: List[List[str]],
    ) -> None:
        """DFS helper for cycle detection"""
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        # Check all neighbors (both subs and supers)
        neighbors = list(self._sub_map[node]) + list(self._super_map[node])

        for neighbor in neighbors:
            if neighbor in rec_stack:
                # Found a cycle
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                cycles.append(cycle)
            elif neighbor not in visited:
                self._dfs_cycle_detection(
                    neighbor, path.copy(), visited, rec_stack, cycles
                )

        rec_stack.remove(node)
        path.pop()

    def validate_relationship_consistency(self) -> List[str]:
        """Validate relationship consistency across all claims"""
        errors = []

        # Check each claim's relationships are bidirectional
        for claim in self.claims:
            claim_id = claim.id

            # Check subs relationships (claims that provide evidence FOR this claim)
            for sub_id in claim.subs:
                if sub_id not in self.claim_index:
                    errors.append(
                        f"Claim {claim_id} references non-existent sub {sub_id}"
                    )
                else:
                    sub = self.claim_index[sub_id]
                    if claim_id not in sub.supers:
                        errors.append(
                            f"Unidirectional relationship: {sub_id} is sub of {claim_id} but not reciprocated in supers"
                        )

            # Check supers relationships (claims this provides evidence FOR)
            for super_id in claim.supers:
                if super_id not in self.claim_index:
                    errors.append(
                        f"Claim {claim_id} references non-existent super {super_id}"
                    )
                else:
                    super_claim = self.claim_index[super_id]
                    if claim_id not in super_claim.subs:
                        errors.append(
                            f"Unidirectional relationship: {claim_id} is super of {super_id} but not reciprocated in subs"
                        )

        # Check for cycles
        cycles = self.detect_all_cycles()
        if cycles:
            for i, cycle in enumerate(cycles):
                errors.append(f"Circular dependency #{i + 1}: {' -> '.join(cycle)}")

        return errors

    def get_relationship_metrics(self) -> RelationshipMetrics:
        """Calculate performance and structure metrics"""
        if self._relationship_metrics is None:
            total_claims = len(self.claims)
            total_relationships = sum(len(claim.supers) for claim in self.claims)

            # Calculate max depth
            max_depth = 0
            for claim in self.claims:
                # Leaf claim has no subs (no claims provide evidence FOR it)
                if len(claim.subs) == 0:
                    descendants = self.get_all_super_descendants(claim.id)
                    max_depth = max(max_depth, descendants.depth)

            # Count cycles
            cycles = self.detect_all_cycles()

            # Count orphaned claims (no subs and no supers)
            orphaned = [
                claim
                for claim in self.claims
                if len(claim.subs) == 0 and len(claim.supers) == 0
            ]

            # Calculate average branching factor
            branching_factors = [
                len(claim.supers) for claim in self.claims if claim.supers
            ]
            avg_branching = (
                sum(branching_factors) / len(branching_factors)
                if branching_factors
                else 0
            )

            # Calculate relationship density
            max_possible_relationships = total_claims * (total_claims - 1)
            density = (
                (total_relationships * 2) / max_possible_relationships
                if max_possible_relationships > 0
                else 0
            )

            self._relationship_metrics = RelationshipMetrics(
                total_claims=total_claims,
                total_relationships=total_relationships,
                max_depth=max_depth,
                cycles_detected=len(cycles),
                orphaned_claims=len(orphaned),
                avg_branching_factor=avg_branching,
                relationship_density=density,
            )

        return self._relationship_metrics

    def add_relationship(self, sub_id: str, super_id: str) -> bool:
        """
        Add a bidirectional relationship between two claims.
        sub_id provides evidence FOR super_id.

        Args:
            sub_id: ID of claim that provides evidence (child)
            super_id: ID of claim that receives evidence (parent, toward root)

        Returns True if relationship was added successfully
        """
        if sub_id not in self.claim_index or super_id not in self.claim_index:
            return False

        if sub_id == super_id:
            return False

        sub = self.claim_index[sub_id]
        super_claim = self.claim_index[super_id]

        # Avoid duplicates
        if super_id in sub.supers or sub_id in super_claim.subs:
            return False

        # Add bidirectional relationship
        sub.supers.append(super_id)
        super_claim.subs.append(sub_id)

        # Update timestamps
        sub.updated = datetime.now(timezone.utc)
        super_claim.updated = datetime.now(timezone.utc)

        # Update internal maps
        self._super_map[sub_id].add(super_id)
        self._sub_map[super_id].add(sub_id)

        # Invalidate cached metrics
        self._relationship_metrics = None

        return True

    def remove_relationship(self, sub_id: str, super_id: str) -> bool:
        """
        Remove a bidirectional relationship between two claims.

        Args:
            sub_id: ID of claim that provides evidence (child)
            super_id: ID of claim that receives evidence (parent, toward root)

        Returns True if relationship was removed successfully
        """
        if sub_id not in self.claim_index or super_id not in self.claim_index:
            return False

        sub = self.claim_index[sub_id]
        super_claim = self.claim_index[super_id]

        # Remove if exists
        relationship_existed = False

        if super_id in sub.supers:
            sub.supers.remove(super_id)
            relationship_existed = True

        if sub_id in super_claim.subs:
            super_claim.subs.remove(sub_id)
            relationship_existed = True

        if relationship_existed:
            # Update timestamps
            sub.updated = datetime.now(timezone.utc)
            super_claim.updated = datetime.now(timezone.utc)

            # Update internal maps
            self._super_map[sub_id].discard(super_id)
            self._sub_map[super_id].discard(sub_id)

            # Invalidate cached metrics
            self._relationship_metrics = None

        return relationship_existed

    def get_claims_by_depth(self, root_claim_id: str) -> Dict[int, List[str]]:
        """Get claims organized by depth from the root claim"""
        if root_claim_id not in self.claim_index:
            return {}

        depth_map = defaultdict(list)
        visited = set()
        queue = deque([(root_claim_id, 0)])  # (claim_id, depth)

        while queue:
            current_id, depth = queue.popleft()

            if current_id in visited:
                continue

            visited.add(current_id)
            depth_map[depth].append(current_id)

            # Add all super claims to queue (claims this provides evidence FOR)
            for super_id in self._super_map[current_id]:
                if super_id not in visited:
                    queue.append((super_id, depth + 1))

        return dict(depth_map)

    def optimize_for_performance(self) -> None:
        """Optimize internal data structures for better performance"""
        # Rebuild relationship maps with current claim data
        self._build_relationship_maps()

        # Force recalculation of metrics next time they're requested
        self._relationship_metrics = None

    def refresh(self, new_claims: List[Claim]) -> None:
        """Refresh the relationship manager with new claim data"""
        self.claims = new_claims
        self.claim_index = {claim.id: claim for claim in new_claims}
        self._relationship_metrics = None
        self._build_relationship_maps()

    def export_relationship_graph(self) -> Dict[str, Dict[str, List[str]]]:
        """Export the relationship graph as a dictionary structure"""
        graph = {"supers": {}, "subs": {}}

        for claim_id in self.claim_index:
            graph["supers"][claim_id] = list(self._super_map[claim_id])
            graph["subs"][claim_id] = list(self._sub_map[claim_id])

        return graph
