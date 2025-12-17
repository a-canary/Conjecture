"""
Support Relationship Manager for Simplified Universal Claim Architecture
Handles efficient bidirectional relationship traversal, validation, and optimization
"""

from typing import List, Dict, Set, Tuple, Optional
from datetime import datetime
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
    """

    def __init__(self, claims: List[Claim]):
        """Initialize with a list of claims"""
        self.claims = claims
        self.claim_index = {claim.id: claim for claim in claims}
        self._support_map = None
        self._supporter_map = None
        self._relationship_metrics = None

        # Build optimized lookup structures
        self._build_relationship_maps()

    def _build_relationship_maps(self) -> None:
        """Build optimized relationship lookup maps"""
        self._support_map = defaultdict(set)  # claim_id -> set of supported claim IDs
        self._supporter_map = defaultdict(set)  # claim_id -> set of supporter claim IDs

        for claim in self.claims:
            # Build forward support map
            for supported_id in claim.supports:
                if supported_id in self.claim_index:
                    self._support_map[claim.id].add(supported_id)

            # Build backward supporter map
            for supporter_id in claim.supported_by:
                if supporter_id in self.claim_index:
                    self._supporter_map[claim.id].add(supporter_id)

    def get_supporting_claims(self, claim_id: str) -> List[Claim]:
        """Get all claims that directly support the given claim"""
        if claim_id not in self.claim_index:
            return []

        supporters = []
        for supporter_id in self._supporter_map[claim_id]:
            supporters.append(self.claim_index[supporter_id])
        return supporters

    def get_supported_claims(self, claim_id: str) -> List[Claim]:
        """Get all claims that are directly supported by the given claim"""
        if claim_id not in self.claim_index:
            return []

        supported = []
        for supported_id in self._support_map[claim_id]:
            supported.append(self.claim_index[supported_id])
        return supported

    def get_all_supporting_ancestors(
        self, claim_id: str, max_depth: int = 100
    ) -> TraversalResult:
        """
        Get all claims that support the given claim (transitive closure upward)
        Traverses to root claims
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

            # Get supporters
            supporters = self._supporter_map[current_id]
            for supporter_id in supporters:
                new_path = current_path + [supporter_id]

                # Check for cycles
                if supporter_id in current_path:
                    cycle_start = current_path.index(supporter_id)
                    cycle = current_path[cycle_start:] + [supporter_id]
                    cycles.append(cycle)
                    continue

                queue.append((supporter_id, current_depth + 1, new_path))

        # Remove the starting claim from ancestors
        visited.discard(claim_id)
        traversal_path = [c for c in traversal_path if c != claim_id]

        return TraversalResult(list(visited), traversal_path, depth, cycles)

    def get_all_supported_descendants(
        self, claim_id: str, max_depth: int = 100
    ) -> TraversalResult:
        """
        Get all claims supported by the given claim (transitive closure downward)
        Traverses to leaf claims
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

            # Get supported claims
            supported = self._support_map[current_id]
            for supported_id in supported:
                new_path = current_path + [supported_id]

                # Check for cycles
                if supported_id in current_path:
                    cycle_start = current_path.index(supported_id)
                    cycle = current_path[cycle_start:] + [supported_id]
                    cycles.append(cycle)
                    continue

                queue.append((supported_id, current_depth + 1, new_path))

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

            # Check both supporters and supported claims
            neighbors = list(self._supporter_map[current_id]) + list(
                self._support_map[current_id]
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

        # Check all neighbors (both supporters and supported)
        neighbors = list(self._supporter_map[node]) + list(self._support_map[node])

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

            # Check supported_by relationships
            for supporter_id in claim.supported_by:
                if supporter_id not in self.claim_index:
                    errors.append(
                        f"Claim {claim_id} references non-existent supporter {supporter_id}"
                    )
                else:
                    supporter = self.claim_index[supporter_id]
                    if claim_id not in supporter.supports:
                        errors.append(
                            f"Unidirectional relationship: {supporter_id} supports {claim_id} but not reciprocated"
                        )

            # Check supports relationships
            for supported_id in claim.supports:
                if supported_id not in self.claim_index:
                    errors.append(
                        f"Claim {claim_id} references non-existent supported claim {supported_id}"
                    )
                else:
                    supported = self.claim_index[supported_id]
                    if claim_id not in supported.supported_by:
                        errors.append(
                            f"Unidirectional relationship: {claim_id} supports {supported_id} but not reciprocated"
                        )

        # Check for cycles
        cycles = self.detect_all_cycles()
        if cycles:
            for i, cycle in enumerate(cycles):
                errors.append(f"Circular dependency #{i + 1}: {' → '.join(cycle)}")

        return errors

    def get_relationship_metrics(self) -> RelationshipMetrics:
        """Calculate performance and structure metrics"""
        if self._relationship_metrics is None:
            total_claims = len(self.claims)
            total_relationships = sum(len(claim.supports) for claim in self.claims)

            # Calculate max depth
            max_depth = 0
            for claim in self.claims:
                # Root claim has no supporters
                if len(claim.supported_by) == 0:
                    descendants = self.get_all_supported_descendants(claim.id)
                    max_depth = max(max_depth, descendants.depth)

            # Count cycles
            cycles = self.detect_all_cycles()

            # Count orphaned claims (no supporters and no supported)
            orphaned = [
                claim
                for claim in self.claims
                if len(claim.supported_by) == 0 and len(claim.supports) == 0
            ]

            # Calculate average branching factor
            branching_factors = [
                len(claim.supports) for claim in self.claims if claim.supports
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

    def add_support_relationship(self, supporter_id: str, supported_id: str) -> bool:
        """
        Add a bidirectional support relationship between two claims
        Returns True if relationship was added successfully
        """
        if supporter_id not in self.claim_index or supported_id not in self.claim_index:
            return False

        if supporter_id == supported_id:
            return False

        supporter = self.claim_index[supporter_id]
        supported = self.claim_index[supported_id]

        # Avoid duplicates
        if supported_id in supporter.supports or supporter_id in supported.supported_by:
            return False

        # Add bidirectional relationship
        supporter.supports.append(supported_id)
        supported.supported_by.append(supporter_id)

        # Update timestamps
        supporter.updated = datetime.utcnow()
        supported.updated = datetime.utcnow()

        # Update internal maps
        self._support_map[supporter_id].add(supported_id)
        self._supporter_map[supported_id].add(supporter_id)

        # Invalidate cached metrics
        self._relationship_metrics = None

        return True

    def remove_support_relationship(self, supporter_id: str, supported_id: str) -> bool:
        """
        Remove a bidirectional support relationship between two claims
        Returns True if relationship was removed successfully
        """
        if supporter_id not in self.claim_index or supported_id not in self.claim_index:
            return False

        supporter = self.claim_index[supporter_id]
        supported = self.claim_index[supported_id]

        # Remove if exists
        relationship_existed = False

        if supported_id in supporter.supports:
            supporter.supports.remove(supported_id)
            relationship_existed = True

        if supporter_id in supported.supported_by:
            supported.supported_by.remove(supporter_id)
            relationship_existed = True

        if relationship_existed:
            # Update timestamps
            supporter.updated = datetime.utcnow()
            supported.updated = datetime.utcnow()

            # Update internal maps
            self._support_map[supporter_id].discard(supported_id)
            self._supporter_map[supported_id].discard(supporter_id)

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

            # Add all supported claims to queue
            for supported_id in self._support_map[current_id]:
                if supported_id not in visited:
                    queue.append((supported_id, depth + 1))

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
        graph = {"supports": {}, "supported_by": {}}

        for claim_id in self.claim_index:
            claim = self.claim_index[claim_id]
            graph["supports"][claim_id] = list(self._support_map[claim_id])
            graph["supported_by"][claim_id] = list(self._supporter_map[claim_id])

        return graph
