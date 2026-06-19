# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Claim Visualization Utilities

Provides tree and trace visualization for claim relationships.
Supports both Rich terminal output and JSON for web interfaces.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Set
from src.data.models import Claim


@dataclass
class ClaimNode:
    """A node in the claim visualization tree."""
    claim: Claim
    children: List["ClaimNode"]
    depth: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.claim.id,
            "content": self.claim.content,
            "confidence": self.claim.confidence,
            "state": self.claim.state.value,
            "types": [t.value for t in self.claim.type],
            "depth": self.depth,
            "children": [child.to_dict() for child in self.children]
        }


@dataclass  
class ClaimTrace:
    """A trace from a claim back to the root."""
    nodes: List[ClaimNode]  # Ordered from root to the target claim
    path_ids: List[str]     # IDs in order from root to target
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "path": [node.to_dict() for node in self.nodes],
            "path_ids": self.path_ids,
            "length": len(self.nodes)
        }


@dataclass
class ClaimGraph:
    """Adjacency list representation of claim relationships."""
    nodes: Dict[str, Claim]      # claim_id -> Claim
    edges: Dict[str, List[str]]  # claim_id -> [child_ids]
    root_ids: List[str]          # Root claim IDs (no supers)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to adjacency list format for D3.js/vis.js."""
        nodes_list = [
            {
                "id": claim.id,
                "content": claim.content[:100] + "..." if len(claim.content) > 100 else claim.content,
                "confidence": claim.confidence,
                "state": claim.state.value,
            }
            for claim in self.nodes.values()
        ]
        
        edges_list = [
            {"from": source_id, "to": target_id}
            for source_id, child_ids in self.edges.items()
            for target_id in child_ids
        ]
        
        return {
            "nodes": nodes_list,
            "edges": edges_list,
            "root_ids": self.root_ids
        }


def build_claim_tree(
    claim: Claim,
    get_claim_func,
    max_depth: int = 3,
    current_depth: int = 0,
    visited: Optional[Set[str]] = None,
    min_confidence: float = 0.0
) -> ClaimNode:
    """
    Build a tree of claims starting from the given claim.
    
    Traverses subs (children) recursively to build the support tree.
    
    Args:
        claim: The root claim for the tree
        get_claim_func: Function to fetch a claim by ID
        max_depth: Maximum depth to traverse (default: 3)
        current_depth: Current recursion depth
        visited: Set of already visited claim IDs (for cycle detection)
        min_confidence: Minimum confidence threshold for inclusion
        
    Returns:
        ClaimNode with nested children
    """
    if visited is None:
        visited = set()
    
    if claim.id in visited or current_depth >= max_depth:
        return ClaimNode(claim=claim, children=[], depth=current_depth)
    
    visited.add(claim.id)
    
    children = []
    for sub_id in claim.subs:
        if sub_id in visited:
            continue
        try:
            sub_claim = get_claim_func(sub_id)
            if sub_claim and sub_claim.confidence >= min_confidence:
                child_node = build_claim_tree(
                    sub_claim, get_claim_func, max_depth, 
                    current_depth + 1, visited, min_confidence
                )
                children.append(child_node)
        except Exception:
            continue  # Skip claims that can't be fetched
    
    return ClaimNode(claim=claim, children=children, depth=current_depth)


def build_claim_trace(
    claim: Claim,
    get_claim_func,
    visited: Optional[Set[str]] = None
) -> ClaimTrace:
    """
    Build a trace from the given claim back to the root.
    
    Traverses supers (parents) until no more are found (root).
    
    Args:
        claim: The target claim for the trace
        get_claim_func: Function to fetch a claim by ID
        
    Returns:
        ClaimTrace with nodes in order from root to target
    """
    if visited is None:
        visited = set()
    
    nodes = []
    path_ids = []
    
    # Start from the given claim and traverse supers to root
    current = claim
    while current:
        if current.id in visited:
            break  # Cycle detected
        
        visited.add(current.id)
        node = ClaimNode(claim=current, children=[], depth=len(nodes))
        nodes.append(node)
        path_ids.append(current.id)
        
        # Move to parent (first super)
        if current.supers:
            try:
                current = get_claim_func(current.supers[0])
            except Exception:
                current = None
        else:
            current = None
    
    return ClaimTrace(nodes=list(reversed(nodes)), path_ids=list(reversed(path_ids)))


def build_claim_graph(
    claim_ids: List[str],
    get_claim_func,
    max_depth: int = 2
) -> ClaimGraph:
    """
    Build an adjacency list graph from a set of claim IDs.
    
    Args:
        claim_ids: Starting claim IDs
        get_claim_func: Function to fetch a claim by ID
        max_depth: Maximum depth to traverse
        
    Returns:
        ClaimGraph with nodes and edges
    """
    nodes: Dict[str, Claim] = {}
    edges: Dict[str, List[str]] = {}
    root_ids: List[str] = []
    
    to_process = list(claim_ids)
    visited = set()
    
    while to_process and len(visited) < 100:  # Limit to prevent runaway
        claim_id = to_process.pop(0)
        if claim_id in visited:
            continue
        
        try:
            claim = get_claim_func(claim_id)
            if not claim:
                continue
                
            nodes[claim_id] = claim
            visited.add(claim_id)
            
            # Track edges
            edges[claim_id] = claim.subs.copy()
            
            # Root claims have no supers
            if not claim.supers:
                root_ids.append(claim_id)
            
            # Add children to processing queue
            for sub_id in claim.subs[:5]:  # Limit children per node
                if sub_id not in visited and len(visited) < 100:
                    to_process.append(sub_id)
                    
        except Exception:
            continue
    
    return ClaimGraph(nodes=nodes, edges=edges, root_ids=root_ids)


def confidence_color(confidence: float) -> str:
    """Get a color based on confidence level."""
    if confidence >= 0.9:
        return "green"
    elif confidence >= 0.7:
        return "yellow"
    elif confidence >= 0.5:
        return "orange"
    else:
        return "red"


def format_claim_summary(claim: Claim, max_content: int = 60) -> str:
    """Format a one-line summary of a claim."""
    content = claim.content[:max_content]
    if len(claim.content) > max_content:
        content += "..."
    return f"[{confidence_color(claim.confidence)}]{claim.id}[/{confidence_color(claim.confidence)}] {content}"


def render_tree_ascii(node: ClaimNode, prefix: str = "", is_last: bool = True) -> str:
    """Render a claim tree as ASCII art."""
    lines = []
    
    # Build the connector
    connector = "└── " if is_last else "├── "
    branch = "    " if is_last else "│   "
    
    # Format the claim
    confidence_marker = "●" if node.claim.confidence >= 0.8 else "○"
    content = node.claim.content[:50]
    if len(node.claim.content) > 50:
        content += "..."
    
    line = f"{prefix}{connector}[{node.claim.id}] {confidence_marker} {content}"
    lines.append(line)
    
    # Render children
    for i, child in enumerate(node.children):
        child_prefix = prefix + branch
        child_is_last = i == len(node.children) - 1
        lines.append(render_tree_ascii(child, child_prefix, child_is_last))
    
    return "\n".join(lines)
