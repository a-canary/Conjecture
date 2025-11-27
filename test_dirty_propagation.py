#!/usr/bin/env python3
"""
Test script for dirty flag propagation when A is updated, B is marked dirty
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.core.models import Claim, ClaimState, ClaimType, DirtyReason
from src.core.claim_operations import update_claim_with_dirty_propagation
from src.core.dirty_flag import DirtyFlagSystem


def test_dirty_propagation():
    """Test that when claim A is updated, claim B (which A supports) gets marked dirty"""

    # Create claim A (supporting claim)
    claim_a = Claim(
        id="c0000001",
        content="Quantum computing uses superposition",
        confidence=0.8,
        state=ClaimState.EXPLORE,
        type=[ClaimType.CONCEPT],
        supports=["c0000002"],  # A supports B
    )

    # Create claim B (supported claim)
    claim_b = Claim(
        id="c0000002",
        content="Quantum algorithms can solve certain problems faster",
        confidence=0.7,
        state=ClaimState.EXPLORE,
        type=[ClaimType.CONCEPT],
        supported_by=["c0000001"],  # B is supported by A
    )

    # All claims dictionary
    all_claims = {"c0000001": claim_a, "c0000002": claim_b}

    # Dirty flag system
    dirty_system = DirtyFlagSystem()

    print("=== Before Update ===")
    print(f"Claim A dirty: {claim_a.is_dirty}")
    print(f"Claim B dirty: {claim_b.is_dirty}")

    # Update claim A with new confidence
    updated_claim_a = Claim(
        id="c0000001",
        content="Quantum computing uses superposition and entanglement",  # Changed content
        confidence=0.9,  # Changed confidence
        state=ClaimState.EXPLORE,
        type=[ClaimType.CONCEPT],
        supports=["c0000002"],
    )

    # Apply update with dirty propagation
    final_claim_a, marked_dirty_ids = update_claim_with_dirty_propagation(
        updated_claim_a, claim_a, all_claims, dirty_system
    )

    print("\n=== After Update ===")
    print(f"Claim A dirty: {final_claim_a.is_dirty}")
    print(f"Claim B dirty: {claim_b.is_dirty}")
    print(f"Marked dirty IDs: {marked_dirty_ids}")

    # Verify the results
    assert claim_b.is_dirty == True, "Claim B should be marked dirty when A is updated"
    assert "c0000002" in marked_dirty_ids, "Claim B ID should be in marked dirty list"
    assert claim_b.dirty_reason == DirtyReason.SUPPORTING_CLAIM_CHANGED, (
        "Dirty reason should be SUPPORTING_CLAIM_CHANGED"
    )

    print("\n✅ Test passed! Dirty flag propagation working correctly.")


if __name__ == "__main__":
    test_dirty_propagation()
