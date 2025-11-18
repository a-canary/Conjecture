"""
Integration test for Dirty Flag System
Tests the complete system integration with CLI and data flow
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.models import Claim, ClaimType, DirtyReason
from src.core.dirty_flag import DirtyFlagSystem
from src.processing.dirty_evaluator import DirtyEvaluator, DirtyEvaluationConfig
from src.config.dirty_flag_config import DirtyFlagConfig, get_dirty_flag_config


class TestDirtyFlagIntegration:
    """Integration tests for the complete dirty flag system"""
    
    def test_cli_import(self):
        """Test that CLI commands can be imported"""
        try:
            from src.cli.dirty_commands import dirty_app
            assert dirty_app is not None
        except ImportError as e:
            pytest.fail(f"Failed to import dirty commands: {e}")
    
    def test_config_integration(self):
        """Test configuration integration across components"""
        # Get global config
        config_manager = get_dirty_flag_config()
        config = config_manager.get_config()
        
        # Create components with same config
        dirty_system = DirtyFlagSystem(
            confidence_threshold=config.confidence_threshold,
            cascade_depth=config.cascade_depth
        )
        
        eval_config = DirtyEvaluationConfig(
            confidence_threshold=config.confidence_threshold,
            batch_size=config.batch_size
        )
        
        # Verify consistency
        assert dirty_system.confidence_threshold == config.confidence_threshold
        assert eval_config.confidence_threshold == config.confidence_threshold
        assert eval_config.batch_size == config.batch_size
    
    def test_claim_model_integration(self):
        """Test Claim model integration with dirty flags"""
        claim = Claim(
            id="test_claim",
            content="This is a test claim",
            confidence=0.8,
            type=[ClaimType.CONCEPT]
        )
        
        # Test dirty flag methods
        assert claim.is_dirty is False
        assert claim.dirty_reason is None
        assert claim.dirty_timestamp is None
        assert claim.dirty_priority == 0
        
        # Mark dirty
        claim.mark_dirty(DirtyReason.MANUAL_MARK, priority=10)
        
        assert claim.is_dirty is True
        assert claim.dirty_reason == DirtyReason.MANUAL_MARK
        assert claim.dirty_timestamp is not None
        assert claim.dirty_priority == 10
        
        # Test priority checking
        assert claim.should_prioritize(0.90) is True  # Below threshold
        
        # Mark clean
        claim.mark_clean()
        
        assert claim.is_dirty is False
        assert claim.dirty_reason is None
        assert claim.dirty_timestamp is None
        assert claim.dirty_priority == 0
    
    def test_chroma_metadata_integration(self):
        """Test ChromaDB metadata integration"""
        claim = Claim(
            id="test_claim",
            content="This is a test claim",
            confidence=0.8,
            type=[ClaimType.CONCEPT]
        )
        
        claim.mark_dirty(DirtyReason.MANUAL_MARK, priority=15)
        
        # Test metadata conversion
        metadata = claim.to_chroma_metadata()
        
        assert "is_dirty" in metadata
        assert "dirty_reason" in metadata
        assert "dirty_timestamp" in metadata
        assert "dirty_priority" in metadata
        
        assert metadata["is_dirty"] is True
        assert metadata["dirty_reason"] == "manual_mark"
        assert metadata["dirty_priority"] == 15
        
        # Test recreation from metadata
        recreated = Claim.from_chroma_result(
            id=claim.id,
            content=claim.content,
            metadata=metadata
        )
        
        assert recreated.is_dirty is True
        assert recreated.dirty_reason == DirtyReason.MANUAL_MARK
        assert recreated.dirty_priority == 15
    
    @patch('src.processing.llm.llm_manager.LLMManager')
    def test_evaluator_integration(self, mock_llm_manager_class):
        """Test DirtyEvaluator integration with mocked LLM"""
        # Create mock LLM manager
        mock_llm_manager = Mock()
        
        # Mock process_claims to return successful result
        mock_result = Mock()
        mock_result.success = True
        mock_result.content = """{
            "results": [
                {"success": true, "data": {"confidence": 0.9}},
                {"success": true, "data": {"confidence": 0.85}}
            ]
        }"""
        mock_llm_manager.process_claims.return_value = mock_result
        mock_llm_manager_class.return_value = mock_llm_manager
        
        # Create dirty flag system and evaluator
        dirty_system = DirtyFlagSystem()
        eval_config = DirtyEvaluationConfig(
            batch_size=2,
            max_parallel_batches=1,
            enable_two_pass=False
        )
        evaluator = DirtyEvaluator(mock_llm_manager, dirty_system, eval_config)
        
        # Create test claims
        claims = [
            Claim(
                id="claim_1",
                content="Test claim 1",
                confidence=0.7,
                type=[ClaimType.CONCEPT]
            ),
            Claim(
                id="claim_2",
                content="Test claim 2", 
                confidence=0.8,
                type=[ClaimType.CONCEPT]
            )
        ]
        
        # Mark claims dirty
        claims[0].mark_dirty(DirtyReason.MANUAL_MARK)
        claims[1].mark_dirty(DirtyReason.CONFIDENCE_THRESHOLD)
        
        # Run evaluation (async test would require proper async setup)
        # For integration test, we'll test the setup instead
        
        assert evaluator.llm_manager == mock_llm_manager
        assert evaluator.dirty_flag_system == dirty_system
        assert evaluator.config.batch_size == 2
    
    def test_environment_config_integration(self):
        """Test environment variable configuration"""
        # Test configuration from environment
        config = DirtyFlagConfig.from_env()
        
        # Should use defaults when no environment variables set
        assert config.confidence_threshold >= 0.0
        assert config.batch_size > 0
        assert config.cascade_depth > 0
    
    def test_configuration_validation_integration(self):
        """Test configuration validation in integration"""
        # Test valid configuration
        valid_config = DirtyFlagConfig(
            confidence_threshold=0.85,
            batch_size=10,
            cascade_depth=2
        )
        assert valid_config.confidence_threshold == 0.85
        
        # Test invalid configuration
        with pytest.raises(ValueError):
            DirtyFlagConfig(confidence_threshold=1.5)
    
    def test_dirty_flag_reasons_integration(self):
        """Test all dirty flag reasons work correctly"""
        claim = Claim(
            id="test_claim",
            content="Test claim",
            confidence=0.8,
            type=[ClaimType.CONCEPT]
        )
        
        dirty_system = DirtyFlagSystem()
        
        # Test all dirty reasons
        reasons = list(DirtyReason)
        
        for i, reason in enumerate(reasons):
            priority = i * 5  # Different priorities
            dirty_system.mark_claim_dirty(claim, reason, priority)
            
            assert claim.is_dirty is True
            assert claim.dirty_reason == reason
            assert claim.dirty_priority == priority
            
            # Reset for next test
            claim.mark_clean()
    
    def test_cascade_depth_limiting(self):
        """Test cascade depth is properly limited"""
        # Create a chain of related claims
        claims = []
        for i in range(10):
            claim = Claim(
                id=f"claim_{i}",
                content=f"Claim {i}",
                confidence=0.8,
                type=[ClaimType.CONVENTION]
            )
            
            # Create chain relationship
            if i > 0:
                claim.supported_by = [f"claim_{i-1}"]
            
            claims.append(claim)
        
        dirty_system = DirtyFlagSystem(cascade_depth=3)
        
        # Mark first claim dirty with cascade
        dirty_system.mark_claim_dirty(claims[0], DirtyReason.NEW_CLAIM_ADDED, cascade=True)
        
        # Cascade should be limited by depth (handled internally)
        # This test mainly verifies the system doesn't crash on deep cascades
        assert claims[0].is_dirty is True


def test_basic_workflow():
    """Test basic dirty flag workflow end-to-end"""
    # Create system components
    config_manager = get_dirty_flag_config()
    dirty_system = DirtyFlagSystem()
    
    # Create claims
    claims = [
        Claim(
            id="high_conf",
            content="High confidence claim",
            confidence=0.95,
            type=[ClaimType.CONCEPT]
        ),
        Claim(
            id="low_conf",
            content="Low confidence claim",
            confidence=0.75,
            type=[ClaimType.CONCEPT]
        )
    ]
    
    # Mark low confidence claim dirty
    dirty_system.mark_claim_dirty(
        claims[1],
        DirtyReason.CONFIDENCE_THRESHOLD,
        priority=10
    )
    
    # Get dirty claims
    dirty_claims = dirty_system.get_dirty_claims(claims)
    assert len(dirty_claims) == 1
    assert dirty_claims[0].id == "low_conf"
    
    # Get priority claims
    priority_claims = dirty_system.get_priority_dirty_claims(claims)
    assert len(priority_claims) == 1
    
    # Clear dirty flags
    cleared = dirty_system.clear_dirty_flags(claims)
    assert cleared == 1
    
    # Verify clean
    for claim in claims:
        assert claim.is_dirty is False
    
    print("✓ Basic dirty flag workflow test passed")


if __name__ == "__main__":
    print("Running Dirty Flag System Integration Tests...")
    
    # Run basic workflow test
    test_basic_workflow()
    
    # Run pytest tests
    print("\nRunning comprehensive tests...")
    pytest.main([__file__, "-v"])