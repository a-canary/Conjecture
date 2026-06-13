# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Graph Validator - Acyclic Graph Enforcement (D-0007)

Claims form a directed acyclic graph (DAG). This module detects cycles via
BFS traversal before a relationship is added, preventing A->B->A cycles.

Essential for context building without infinite loops.

Per CHOICES.md D-0007: "Claims form directed acyclic graph (DAG). Relationship
manager detects cycles via traversal. Prevents A->B->A relationships. Essential
for context building without infinite loops."

Relationship semantics:
  source supports target  =>  source.supers contains target
                          =>  target.subs contains source

  source -> target means source provides evidence FOR target.
  To detect a cycle, traverse target's supers upward; if source is reached, cycle exists.
"""

from collections import deque
from typing import Callable, List, Optional, Set


class CycleDetectedError(ValueError):
    """Raised when adding a relationship would create a cycle in the claim DAG.

    Inherits from ValueError so callers that catch ValueError also catch this.
    """

    def __init__(self, source_id: str, target_id: str, cycle_path: Optional[List[str]] = None):
        self.source_id = source_id
        self.target_id = target_id
        self.cycle_path = cycle_path or []
        path_str = " -> ".join(self.cycle_path) if self.cycle_path else f"{source_id} -> ... -> {source_id}"
        super().__init__(
            f"Adding relationship {source_id} -> {target_id} would create a cycle: {path_str}"
        )


def would_create_cycle(
    source_id: str,
    target_id: str,
    get_supers_fn: Callable[[str], List[str]],
) -> bool:
    """Check whether adding source -> target would create a cycle in the DAG.

    source -> target means source supports target (source is in target.subs,
    target is in source.supers).

    A cycle would exist if target can already reach source through its super
    chain, i.e., if source is an ancestor of target.  We detect this by
    traversing upward (through supers) from target and checking whether source
    is encountered.

    Args:
        source_id: The claim that will be doing the supporting.
        target_id: The claim that will be supported.
        get_supers_fn: Callable that accepts a claim ID and returns the list of
            super claim IDs (claims that the given claim provides evidence FOR).

    Returns:
        True if adding source -> target would create a cycle, False otherwise.
    """
    if source_id == target_id:
        # Self-loop is always a cycle.
        return True

    visited: Set[str] = set()
    queue: deque[str] = deque([target_id])

    while queue:
        current = queue.popleft()
        if current == source_id:
            return True
        if current in visited:
            continue
        visited.add(current)
        for super_id in get_supers_fn(current):
            if super_id not in visited:
                queue.append(super_id)

    return False


def assert_no_cycle(
    source_id: str,
    target_id: str,
    get_supers_fn: Callable[[str], List[str]],
) -> None:
    """Assert that adding source -> target does NOT create a cycle.

    Raises CycleDetectedError (a subclass of ValueError) if a cycle would result.

    Args:
        source_id: The claim that will be doing the supporting.
        target_id: The claim that will be supported.
        get_supers_fn: Callable returning super claim IDs for a given claim ID.

    Raises:
        CycleDetectedError: If adding the relationship would create a cycle.
    """
    if would_create_cycle(source_id, target_id, get_supers_fn):
        # Build the cycle path for a helpful error message.
        cycle_path = _trace_cycle_path(source_id, target_id, get_supers_fn)
        raise CycleDetectedError(source_id, target_id, cycle_path)


def _trace_cycle_path(
    source_id: str,
    target_id: str,
    get_supers_fn: Callable[[str], List[str]],
) -> List[str]:
    """Trace the cycle path from target back to source through supers.

    Used for producing a human-readable error message only; not for cycle
    detection itself.

    Returns the path as a list of claim IDs starting with target_id and ending
    with source_id, with the proposed new edge prepended as source_id.
    """
    # BFS from target to source, tracking predecessors to reconstruct path.
    predecessors: dict[str, Optional[str]] = {target_id: None}
    queue: deque[str] = deque([target_id])

    while queue:
        current = queue.popleft()
        if current == source_id:
            # Reconstruct path from source back to target.
            path: List[str] = []
            node: Optional[str] = source_id
            while node is not None:
                path.append(node)
                node = predecessors.get(node)
            # path is [source_id, ..., target_id] — reverse to get target->source.
            # Prepend proposed source_id to show the full cycle.
            return [source_id] + list(reversed(path))
        for super_id in get_supers_fn(current):
            if super_id not in predecessors:
                predecessors[super_id] = current
                queue.append(super_id)

    # Fallback if path not found (should not happen if called after would_create_cycle=True).
    return [source_id, target_id, source_id]
