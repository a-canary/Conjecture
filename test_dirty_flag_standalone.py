"""
Simple standalone test for Dirty Flag System functionality
Tests the core components without complex import chains
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_claim_model():
    """Test Claim model with dirty flag functionality"""
    from core.models import Claim, ClaimType, DirtyReason
    
    print("Testing Claim model...")
    
    # Create claim
    claim = Claim(
        id="test_claim",
        content="This is a test claim for dirty flag system",
        confidence=0.8,
        type=[ClaimType.CONCEPT],
        tags=["test"]
    )
    
    # Test initial state
    assert claim.is_dirty is False
    assert claim.dirty_reason is None
    assert claim.dirty_timestamp is None
    assert claim.dirty_priority == 0
    print("  + Initial clean state works")
    
    # Mark dirty
    claim.mark_dirty(DirtyReason.MANUAL_MARK, priority=15)
    assert claim.is_dirty is True
    assert claim.dirty_reason == DirtyReason.MANUAL_MARK
    assert claim.dirty_timestamp is not None
    assert claim.dirty_priority == 15
    print("  + Mark dirty works")
    
    # Test priority checking
    assert claim.should_prioritize(0.90) is True  # Below threshold
    assert claim.should_prioritize(0.70) is False  # Above threshold
    print("  ✓ Priority checking works")
    
    # Mark clean
    claim.mark_clean()
    assert claim.is_dirty is False
    assert claim.dirty_reason is None
    assert claim.dirty_timestamp is None
    assert claim.dirty_priority == 0
    print("  ✓ Mark clean works")
    
    # Test metadata conversion
    claim.mark_dirty(DirtyReason.CONFIDENCE_THRESHOLD, priority=10)
    metadata = claim.to_chroma_metadata()
    
    assert "is_dirty" in metadata
    assert "dirty_reason" in metadata
    assert "dirty_timestamp" in metadata
    assert "dirty_priority" in metadata
    assert metadata["is_dirty"] is True
    assert metadata["dirty_reason"] == "confidence_threshold"
    print("  ✓ Chroma metadata conversion works")
    
    # Test recreation from metadata
    recreated = Claim.from_chroma_result(
        id=claim.id,
        content=claim.content,
        metadata=metadata
    )
    assert recreated.is_dirty is True
    assert recreated.dirty_reason == DirtyReason.CONFIDENCE_THRESHOLD
    assert recreated.dirty_priority == 10
    print("  ✓ Recreation from metadata works")
    
    print("Claim model tests passed!\n")


def test_dirty_flag_system():
    """Test DirtyFlagSystem functionality"""
    from core.models import Claim, ClaimType, DirtyReason
    from core.dirty_flag import DirtyFlagSystem
    
    print("Testing DirtyFlagSystem...")
    
    # Create system
    system = DirtyFlagSystem(confidence_threshold=0.90, cascade_depth=3)
    print("  ✓ System initialization works")
    
    # Create test claims
    claims = [
        Claim(
            id="claim_1",
            content="High confidence claim",
            confidence=0.95,
            type=[ClaimType.CONCEPT]
        ),
        Claim(
            id="claim_2",
            content="Low confidence claim",
            confidence=0.75,
            type=[ClaimType.CONCEPT]
        ),
        Claim(
            id="claim_3",
            content="Medium confidence claim",
            confidence=0.85,
            type=[ClaimType.CONCEPT]
        )
    ]
    
    # Mark claim dirty
    system.mark_claim_dirty(claims[1], DirtyReason.MANUAL_MARK, priority=20)
    assert claims[1].is_dirty is True
    assert claims[1].dirty_priority == 20
    print("  ✓ Mark claim dirty works")
    
    # Test confidence threshold marking
    marked_count = system.mark_claims_dirty_by_confidence_threshold(claims, threshold=0.90)
    # Should mark claim_2 (0.75) and claim_3 (0.85)
    assert marked_count == 2
    print("  ✓ Confidence threshold marking works")
    
    # Get dirty claims
    dirty_claims = system.get_dirty_claims(claims, prioritize=True)
    assert len(dirty_claims) == 2  # claim_2 and claim_3
    # Should be sorted by priority
    priorities = [claim.dirty_priority for claim in dirty_claims]
    assert priorities == sorted(priorities, reverse=True)
    print("  ✓ Get dirty claims with prioritization works")
    
    # Get priority dirty claims
    priority_claims = system.get_priority_dirty_claims(claims, confidence_threshold=0.90)
    # Should include only claims with confidence < 0.90
    assert len(priority_claims) == 2
    for claim in priority_claims:
        assert claim.confidence < 0.90
    print("  ✓ Get priority dirty claims works")
    
    # Get statistics
    stats = system.get_dirty_statistics(claims)
    assert stats["total_dirty"] == 2
    assert stats["priority_dirty"] == 2
    assert "reasons" in stats
    assert "priority_ranges" in stats
    print("  ✓ Get statistics works")
    
    # Clear dirty flags
    cleared_count = system.clear_dirty_flags(claims)
    assert cleared_count == 2
    for claim in claims:
        assert claim.is_dirty is False
    print("  ✓ Clear dirty flags works")
    
    print("DirtyFlagSystem tests passed!\n")


def test_configuration():
    """Test dirty flag configuration"""
    from config.dirty_flag_config import DirtyFlagConfig, DirtyFlagConfigManager
    
    print("Testing DirtyFlagConfig...")
    
    # Test default config
    config = DirtyFlagConfig()
    assert config.confidence_threshold == 0.90
    assert config.batch_size == 5
    assert config.cascade_depth == 3
    assert config.priority_weights is not None
    print("  ✓ Default configuration works")
    
    # Test validation
    try:
        DirtyFlagConfig(confidence_threshold=1.5)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  ✓ Configuration validation works")
    
    # Test config from dict
    config_dict = {
        "confidence_threshold": 0.85,
        "batch_size": 10,
        "cascade_depth": 2
    }
    config = DirtyFlagConfig.from_dict(config_dict)
    assert config.confidence_threshold == 0.85
    assert config.batch_size == 10
    assert config.cascade_depth == 2
    print("  ✓ Config from dict works")
    
    # Test config manager
    manager = DirtyFlagConfigManager()
    initial_config = manager.get_config()
    
    # Update config
    updated = manager.update_config(confidence_threshold=0.95, batch_size=8)
    assert updated.confidence_threshold == 0.95
    assert updated.batch_size == 8
    print("  ✓ Config manager works")
    
    # Test summary
    summary = manager.get_config_summary()
    assert "confidence_threshold" in summary
    assert "config_source" in summary
    print("  ✓ Config summary works")
    
    print("Configuration tests passed!\n")


def test_dirty_reasons():
    """Test all dirty flag reasons"""
    from core.models import DirtyReason
    
    print("Testing dirty flag reasons...")
    
    # Test all reasons exist
    reasons = list(DirtyReason)
    expected_reasons = [
        DirtyReason.NEW_CLAIM_ADDED,
        DirtyReason.CONFIDENCE_THRESHOLD,
        DirtyReason.SUPPORTING_CLAIM_CHANGED,
        DirtyReason.RELATIONSHIP_CHANGED,
        DirtyReason.MANUAL_MARK,
        DirtyReason.BATCH_EVALUATION,
        DirtyReason.SYSTEM_TRIGGER
    ]
    
    assert len(reasons) == len(expected_reasons)
    for reason in expected_reasons:
        assert reason in reasons
    
    print("  ✓ All dirty reasons defined")
    print("Dirty flag reasons tests passed!\n")


def test_full_workflow():
    """Test complete dirty flag workflow"""
    from core.models import Claim, ClaimType, DirtyReason
    from core.dirty_flag import DirtyFlagSystem
    from config.dirty_flag_config import DirtyFlagConfig
    
    print("Testing full workflow...")
    
    # Create components with consistent config
    config = DirtyFlagConfig(confidence_threshold=0.90)
    system = DirtyFlagSystem(
        confidence_threshold=config.confidence_threshold,
        cascade_depth=config.cascade_depth
    )
    
    # Create test claims
    claims = [
        Claim(
            id="high_conf",
            content="High confidence scientific claim",
            confidence=0.95,
            type=[ClaimType.CONCEPT],
            tags=["science"]
        ),
        Claim(
            id="low_conf",
            content="Low confidence hypothesis",
            confidence=0.65,
            type=[ClaimType.CONCEPT],
            tags=["hypothesis"]
        ),
        Claim(
            id="medium_conf",
            content="Medium confidence observation",
            confidence=0.80,
            type=[ClaimType.CONCEPT],
            tags=["observation"]
        )
    ]
    
    # Step 1: Automatic detection by confidence threshold
    marked_count = system.mark_claims_dirty_by_confidence_threshold(claims)
    assert marked_count == 2  # low_conf and medium_conf
    print("  ✓ Automatic threshold detection works")
    
    # Step 2: Get priority claims for evaluation
    priority_claims = system.get_priority_dirty_claims(claims)
    assert len(priority_claims) == 2
    print("  ✓ Priority claim selection works")
    
    # Step 3: Manual marking
    high_conf_claim = claims[0]
    system.mark_claim_dirty(
        high_conf_claim,
        DirtyReason.MANUAL_MARK,
        priority=25
    )
    assert high_conf_claim.is_dirty is True
    print("  ✓ Manual dirty marking works")
    
    # Step 4: Get all dirty claims in priority order
    all_dirty = system.get_dirty_claims(claims, prioritize=True)
    assert len(all_dirty) == 3
    # High priority claim should be first
    assert all_dirty[0].dirty_priority == 25
    print("  ✓ Priority ordering works")
    
    # Step 5: Get statistics
    stats = system.get_dirty_statistics(claims)
    assert stats["total_dirty"] == 3
    assert stats["priority_dirty"] == 2  # Below threshold
    print("  ✓ Statistics generation works")
    
    # Step 6: Clean up
    cleared = system.clear_dirty_flags(claims)
    assert cleared == 3
    for claim in claims:
        assert claim.is_dirty is False
    print("  ✓ Cleanup works")
    
    print("Full workflow test passed!\n")


def main():
    """Run all tests"""
    print("=" * 60)
    print("DIRTY FLAG SYSTEM - STANDALONE TESTS")
    print("=" * 60)
    print()
    
    try:
        test_claim_model()
        test_dirty_flag_system()
        test_configuration()
        test_dirty_reasons()
        test_full_workflow()
        
        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("Dirty Flag System is working correctly.")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)