"""
Claim Management Tools for Conjecture
Provides tools for creating, managing, and querying claims and their relationships
"""

from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import json

# Import the registry system
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from src.tools.registry import register_tool

# Simple in-memory storage for claims (in a real system, this would be a database)
_claim_storage: Dict[str, Dict[str, Any]] = {}
_support_relationships: Dict[str, Set[str]] = {}
_next_claim_id = 1


def _generate_claim_id() -> str:
    """Generate a unique claim ID."""
    global _next_claim_id
    claim_id = f"claim_{_next_claim_id}"
    _next_claim_id += 1
    return claim_id


@register_tool(name="ClaimCreate", is_core=True)
def ClaimCreate(content: str, confidence: float = 0.8, tags: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Create a new claim with specified content and metadata.

    Args:
        content: The claim content text
        confidence: Confidence level (0.0 to 1.0, default: 0.8)
        tags: List of tags to categorize the claim (optional)

    Returns:
        Dictionary with claim ID and creation metadata
    """
    # Input validation
    if not content or not content.strip():
        return {
            'success': False,
            'error': 'Claim content cannot be empty',
            'claim_id': None
        }

    if len(content) > 10000:  # 10KB limit
        return {
            'success': False,
            'error': 'Claim content too long (max 10000 characters)',
            'claim_id': None
        }

    if not isinstance(confidence, (int, float)) or confidence < 0.0 or confidence > 1.0:
        confidence = 0.8

    if tags is None:
        tags = []

    if not isinstance(tags, list):
        return {
            'success': False,
            'error': 'Tags must be a list',
            'claim_id': None
        }

    # Limit number of tags
    if len(tags) > 20:
        tags = tags[:20]

    # Create claim
    claim_id = _generate_claim_id()
    timestamp = datetime.now().isoformat()

    claim = {
        'id': claim_id,
        'content': content.strip(),
        'confidence': float(confidence),
        'tags': [tag.strip() for tag in tags if tag.strip()],
        'created_at': timestamp,
        'updated_at': timestamp,
        'supports': [],  # Claims this claim supports
        'supported_by': []  # Claims that support this claim
    }

    _claim_storage[claim_id] = claim
    _support_relationships[claim_id] = set()

    return {
        'success': True,
        'claim_id': claim_id,
        'claim': claim
    }


@register_tool(name="ClaimAddSupport", is_core=True)
def ClaimAddSupport(supporter: str, supported: str) -> Dict[str, Any]:
    """
    Add a support relationship between two claims.

    Args:
        supporter: ID of the claim that provides support
        supported: ID of the claim that is being supported

    Returns:
        Dictionary with operation result and relationship details
    """
    # Input validation
    if not supporter or not supporter.strip():
        return {
            'success': False,
            'error': 'Supporter claim ID cannot be empty'
        }

    if not supported or not supported.strip():
        return {
            'success': False,
            'error': 'Supported claim ID cannot be empty'
        }

    # Check claims exist
    if supporter not in _claim_storage:
        return {
            'success': False,
            'error': f'Supporter claim not found: {supporter}'
        }

    if supported not in _claim_storage:
        return {
            'success': False,
            'error': f'Supported claim not found: {supported}'
        }

    if supporter == supported:
        return {
            'success': False,
            'error': 'A claim cannot support itself'
        }

    # Add support relationship
    if supporter not in _support_relationships:
        _support_relationships[supporter] = set()

    if supported not in _support_relationships:
        _support_relationships[supported] = set()

    # Add bidirectional references
    _support_relationships[supporter].add(supported)
    _claim_storage[supporter]['supports'].append(supported)
    _claim_storage[supported]['supported_by'].append(supporter)

    # Update timestamps
    timestamp = datetime.now().isoformat()
    _claim_storage[supporter]['updated_at'] = timestamp
    _claim_storage[supported]['updated_at'] = timestamp

    return {
        'success': True,
        'supporter': supporter,
        'supported': supported,
        'relationship_created': True
    }


@register_tool(name="ClaimGetSupport", is_core=True)
def ClaimGetSupport(claim_id: str) -> Dict[str, Any]:
    """
    Get support relationships for a claim.

    Args:
        claim_id: ID of the claim to query

    Returns:
        Dictionary with support relationships and claim details
    """
    # Input validation
    if not claim_id or not claim_id.strip():
        return {
            'success': False,
            'error': 'Claim ID cannot be empty'
        }

    # Check claim exists
    if claim_id not in _claim_storage:
        return {
            'success': False,
            'error': f'Claim not found: {claim_id}'
        }

    claim = _claim_storage[claim_id]

    # Get support relationships
    supports = claim.get('supports', [])
    supported_by = claim.get('supported_by', [])

    return {
        'success': True,
        'claim_id': claim_id,
        'claim': claim,
        'supports': supports,
        'supported_by': supported_by,
        'supports_count': len(supports),
        'supported_by_count': len(supported_by)
    }


@register_tool(name="ClaimAddTags", is_core=True)
def ClaimAddTags(claim_id: str, tags: List[str]) -> Dict[str, Any]:
    """
    Add tags to an existing claim.

    Args:
        claim_id: ID of the claim to tag
        tags: List of tags to add

    Returns:
        Dictionary with operation result and updated claim
    """
    # Input validation
    if not claim_id or not claim_id.strip():
        return {
            'success': False,
            'error': 'Claim ID cannot be empty'
        }

    if not isinstance(tags, list):
        return {
            'success': False,
            'error': 'Tags must be a list'
        }

    # Check claim exists
    if claim_id not in _claim_storage:
        return {
            'success': False,
            'error': f'Claim not found: {claim_id}'
        }

    # Add tags (avoid duplicates)
    claim = _claim_storage[claim_id]
    existing_tags = set(claim.get('tags', []))
    
    for tag in tags:
        tag = tag.strip()
        if tag and tag not in existing_tags:
            existing_tags.add(tag)
            claim['tags'].append(tag)

    # Update timestamp
    claim['updated_at'] = datetime.now().isoformat()

    return {
        'success': True,
        'claim_id': claim_id,
        'added_tags': len(tags),
        'total_tags': len(claim['tags']),
        'claim': claim
    }


@register_tool(name="ClaimsQuery", is_core=True)
def ClaimsQuery(filter_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Query claims with optional filtering criteria.

    Args:
        filter_dict: Dictionary with filter criteria (optional)
            - tags: List of tags to match (claims must have at least one)
            - confidence_min: Minimum confidence level (0.0 to 1.0)
            - confidence_max: Maximum confidence level (0.0 to 1.0)
            - content_contains: Text that must appear in claim content
            - limit: Maximum number of results to return

    Returns:
        Dictionary with matching claims and query metadata
    """
    if filter_dict is None:
        filter_dict = {}

    # Apply filters
    filtered_claims = []

    for claim_id, claim in _claim_storage.items():
        # Content filter
        if 'content_contains' in filter_dict:
            search_text = filter_dict['content_contains'].lower()
            if search_text not in claim['content'].lower():
                continue

        # Confidence filters
        if 'confidence_min' in filter_dict:
            min_conf = filter_dict['confidence_min']
            if claim['confidence'] < min_conf:
                continue

        if 'confidence_max' in filter_dict:
            max_conf = filter_dict['confidence_max']
            if claim['confidence'] > max_conf:
                continue

        # Tags filter
        if 'tags' in filter_dict:
            required_tags = set(filter_dict['tags'])
            claim_tags = set(claim.get('tags', []))
            if not required_tags.intersection(claim_tags):
                continue

        filtered_claims.append(claim)

    # Sort by creation time (newest first)
    filtered_claims.sort(key=lambda x: x['created_at'], reverse=True)

    # Apply limit
    if 'limit' in filter_dict:
        limit = filter_dict['limit']
        if isinstance(limit, int) and limit > 0:
            filtered_claims = filtered_claims[:limit]

    return {
        'success': True,
        'claims': filtered_claims,
        'total_count': len(filtered_claims),
        'filter_applied': filter_dict
    }


# Helper functions not exposed as tools
def _reset_claim_storage():
    """Reset claim storage (for testing)."""
    global _claim_storage, _support_relationships, _next_claim_id
    _claim_storage = {}
    _support_relationships = {}
    _next_claim_id = 1


def examples() -> List[str]:
    """
    Return example usage claims for LLM context
    These examples help the LLM understand when and how to use these tools
    """
    return [
        "ClaimCreate('Rust is a systems programming language with memory safety', confidence=0.9, tags=['rust', 'programming']) creates a new claim about Rust",
        "ClaimAddSupport('claim_1', 'claim_2') makes claim_1 support claim_2, establishing a logical relationship",
        "ClaimGetSupport('claim_1') returns all support relationships for claim_1, including what it supports and what supports it",
        "ClaimAddTags('claim_1', ['systems', 'memory-safety']) adds additional tags to categorize claim_1",
        "ClaimsQuery({'tags': ['rust'], 'confidence_min': 0.8}) returns all claims about Rust with high confidence",
        "ClaimsQuery({'content_contains': 'memory safety', 'limit': 5}) returns up to 5 claims mentioning memory safety",
        "ClaimsQuery({'confidence_min': 0.7, 'confidence_max': 0.9}) returns claims with confidence between 0.7 and 0.9",
        "ClaimCreate('WebSearch provides DuckDuckGo search capabilities', confidence=0.95, tags=['tool', 'search']) creates a claim about a tool"
    ]


if __name__ == "__main__":
    # Test the claim management functionality
    print("Testing claim management tools...")

    # Reset storage for testing
    _reset_claim_storage()

    # Test creating claims
    print("\n1. Creating claims:")
    result1 = ClaimCreate("Rust is a memory-safe systems language", confidence=0.9, tags=["rust", "systems"])
    print(f"   Created claim: {result1['claim_id']}")

    result2 = ClaimCreate("WebSearch tool provides DuckDuckGo search", confidence=0.95, tags=["tool", "search"])
    print(f"   Created claim: {result2['claim_id']}")

    # Test adding support
    print("\n2. Adding support:")
    support_result = ClaimAddSupport(result1['claim_id'], result2['claim_id'])
    print(f"   Support added: {support_result['success']}")

    # Test getting support relationships
    print("\n3. Getting support:")
    support_info = ClaimGetSupport(result1['claim_id'])
    print(f"   Supports count: {support_info['supports_count']}")
    print(f"   Supported by count: {support_info['supported_by_count']}")

    # Test adding tags
    print("\n4. Adding tags:")
    tag_result = ClaimAddTags(result1['claim_id'], ["memory-safety", "performance"])
    print(f"   Tags added: {tag_result['success']}, Total tags: {tag_result['total_tags']}")

    # Test querying claims
    print("\n5. Querying claims:")
    query_result = ClaimsQuery({"tags": ["rust"], "limit": 5})
    print(f"   Found claims: {query_result['total_count']}")

    print("\nExamples for LLM context:")
    for example in examples():
        print(f"- {example}")