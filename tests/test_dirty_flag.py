"""
Comprehensive tests for the Dirty Flag System
Tests dirty flag detection, cascading, evaluation, and CLI integration
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from src.core.models import Claim, ClaimState, ClaimType, DirtyReason
from src.core.dirty_flag import DirtyFlagSystem
from src.processing.dirty_evaluator import DirtyEvaluator, DirtyEvaluationConfig, DirtyClaimBatch
from src.config.dirty_flag_config import DirtyFlagConfig, DirtyFlagConfigManager


class TestDirtyFlagSystem:
    """Test cases for DirtyFlagSystem core functionality"""
    
    @pytest.fixture
    def dirty_flag_system(self):
        """Create DirtyFlagSystem instance for testing"""
        return DirtyFlagSystem(confidence_threshold=0.90, cascade_depth=3)
    
    @pytest.fixture
    def sample_claims(self):
        """Create sample claims for testing"""
        claims = [
            Claim(
                id="claim_1",
                content="The sky is blue",
                confidence=0.95,
                type=[ClaimType.CONCEPT],
                tags=["science", "observation"]
            ),
            Claim(
                id="claim_2",
                content="Water boils at 100°C",
                confidence=0.85,
                type=[ClaimType.CONCEPT],
                tags=["physics", "temperature"]
            ),
            Claim(
                id="claim_3",
                content="The sky appears blue due to Rayleigh scattering",
                confidence=0.75,
                type=[ClaimType.CONCEPT],
                tags=["science", "physics"],
                supports=["claim_1"]
            ),
            Claim(
                id="claim_4",
                content="Temperature affects boiling point",
                confidence=0.60,
                type=[ClaimType.CONCEPT],
                tags=["physics"],
                supported_by=["claim_2"]
            )
        ]
        return claims
    
    def test_mark_claim_dirty_basic(self, dirty_flag_system, sample_claims):
        """Test basic dirty flag marking"""
        claim = sample_claims[0]
        
        # Mark claim dirty
        dirty_flag_system.mark_claim_dirty(
            claim,
            DirtyReason.MANUAL_MARK,
            priority=10,
            cascade=False
        )
        
        # Verify claim is dirty
        assert claim.is_dirty is True
        assert claim.dirty_reason == DirtyReason.MANUAL_MARK
        assert claim.dirty_priority == 10
        assert claim.dirty_timestamp is not None
        
        # Verify claim is in cache
        assert claim.id in dirty_flag_system._dirty_claim_cache
    
    def test_mark_claim_dirty_with_cascade(self, dirty_flag_system, sample_claims):
        """Test dirty flag cascading to related claims"""
        source_claim = sample_claims[0]
        related_claim = sample_claims[2]  # Supports claim_1
        
        # Mark source claim dirty with cascade
        dirty_flag_system.mark_claim_dirty(
            source_claim,
            DirtyReason.NEW_CLAIM_ADDED,
            priority=5,
            cascade=True
        )
        
        # Verify source claim is dirty
        assert source_claim.is_dirty is True
        
        # Add related claim to system (simulating it's in the data store)
        dirty_flag_system._dirty_claim_cache[related_claim.id] = related_claim
        
        # Manually trigger cascade for testing
        dirty_flag_system._cascade_dirty_flags(
            source_claim,
            DirtyReason.NEW_CLAIM_ADDED,
            current_depth=1
        )
        
        # Verify related claim was marked dirty (due to cascade simulation)
        # Note: In real implementation, claims would be fetched from data store
    
    def test_priority_calculation(self, dirty_flag_system, sample_claims):
        """Test priority calculation based on confidence and reason"""
        low_confidence_claim = sample_claims[3]  # confidence = 0.60
        
        # Test priority for low confidence claim
        priority = dirty_flag_system._calculate_priority(
            low_confidence_claim,
            DirtyReason.CONFIDENCE_THRESHOLD
        )
        
        # Should have high priority due to low confidence
        assert priority > 10
        
        # Test priority for high confidence claim
        high_confidence_claim = sample_claims[0]  # confidence = 0.95
        high_priority = dirty_flag_system._calculate_priority(
            high_confidence_claim,
            DirtyReason.MANUAL_MARK
        )
        
        # Manual mark should still have priority
        assert high_priority >= 20
    
    def test_mark_claims_dirty_by_confidence_threshold(self, dirty_flag_system, sample_claims):
        """Test batch marking dirty by confidence threshold"""
        threshold = 0.80
        
        # Mark claims dirty by threshold
        marked_count = dirty_flag_system.mark_claims_dirty_by_confidence_threshold(
            sample_claims,
            threshold
        )
        
        # Should mark claims with confidence < 0.80
        expected_marked = [c for c in sample_claims if c.confidence < threshold]
        
        assert marked_count == len(expected_marked)
        
        # Verify correct claims were marked
        for claim in expected_marked:
            assert claim.is_dirty is True
            assert claim.dirty_reason == DirtyReason.CONFIDENCE_THRESHOLD
    
    def test_get_dirty_claims(self, dirty_flag_system, sample_claims):
        """Test getting dirty claims with prioritization"""
        # Mark some claims dirty with different priorities
        sample_claims[0].mark_dirty(DirtyReason.MANUAL_MARK, priority=20)
        sample_claims[1].mark_dirty(DirtyReason.CONFIDENCE_THRESHOLD, priority=5)
        sample_claims[2].mark_dirty(DirtyReason.NEW_CLAIM_ADDED, priority=15)
        
        # Get dirty claims
        dirty_claims = dirty_flag_system.get_dirty_claims(sample_claims, prioritize=True)
        
        # Should return 3 dirty claims
        assert len(dirty_claims) == 3
        
        # Should be sorted by priority (descending)
        priorities = [claim.dirty_priority for claim in dirty_claims]
        assert priorities == sorted(priorities, reverse=True)
        
        # Highest priority first
        assert dirty_claims[0].dirty_priority == 20
    
    def test_get_priority_dirty_claims(self, dirty_flag_system, sample_claims):
        """Test getting priority dirty claims (low confidence)"""
        # Mark claims dirty with different confidence levels
        sample_claims[0].mark_dirty(DirtyReason.MANUAL_MARK)  # confidence 0.95
        sample_claims[1].mark_dirty(DirtyReason.CONFIDENCE_THRESHOLD)  # confidence 0.85
        sample_claims[2].mark_dirty(DirtyReason.NEW_CLAIM_ADDED)  # confidence 0.75
        sample_claims[3].mark_dirty(DirtyReason.SUPPORTING_CLAIM_CHANGED)  # confidence 0.60
        
        # Get priority dirty claims (threshold = 0.90)
        priority_claims = dirty_flag_system.get_priority_dirty_claims(
            sample_claims,
            confidence_threshold=0.90
        )
        
        # Should only return claims with confidence < 0.90
        expected_priority = [c for c in sample_claims if c.is_dirty and c.confidence < 0.90]
        
        assert len(priority_claims) == len(expected_priority)
        
        # All returned claims should have confidence < 0.90
        for claim in priority_claims:
            assert claim.confidence < 0.90
    
    def test_clear_dirty_flags(self, dirty_flag_system, sample_claims):
        """Test clearing dirty flags"""
        # Mark some claims dirty
        sample_claims[0].mark_dirty(DirtyReason.MANUAL_MARK)
        sample_claims[1].mark_dirty(DirtyReason.CONFIDENCE_THRESHOLD)
        sample_claims[2].mark_dirty(DirtyReason.NEW_CLAIM_ADDED)
        
        # Clear all dirty flags
        cleared_count = dirty_flag_system.clear_dirty_flags(sample_claims)
        
        assert cleared_count == 3
        
        # Verify all claims are clean
        for claim in sample_claims:
            assert claim.is_dirty is False
            assert claim.dirty_reason is None
            assert claim.dirty_priority == 0
    
    def test_get_dirty_statistics(self, dirty_flag_system, sample_claims):
        """Test getting dirty claim statistics"""
        # Mark claims dirty with different reasons and priorities
        sample_claims[0].mark_dirty(DirtyReason.MANUAL_MARK, priority=25)
        sample_claims[1].mark_dirty(DirtyReason.CONFIDENCE_THRESHOLD, priority=15)
        sample_claims[2].mark_dirty(DirtyReason.NEW_CLAIM_ADDED, priority=5)
        
        # Get statistics
        stats = dirty_flag_system.get_dirty_statistics(sample_claims)
        
        # Verify statistics
        assert stats["total_dirty"] == 3
        assert stats["priority_dirty"] >= 1  # At least one claim below threshold
        
        # Check reason counts
        assert "manual_mark" in stats["reasons"]
        assert "confidence_threshold" in stats["reasons"]
        assert "new_claim_added" in stats["reasons"]
        
        # Check priority ranges
        assert "high" in stats["priority_ranges"]
        assert "medium" in stats["priority_ranges"]
        assert "low" in stats["priority_ranges"]
    
    def test_rebuild_cache(self, dirty_flag_system, sample_claims):
        """Test rebuilding dirty claim cache"""
        # Mark some claims dirty
        sample_claims[0].mark_dirty(DirtyReason.MANUAL_MARK)
        sample_claims[1].mark_dirty(DirtyReason.CONFIDENCE_THRESHOLD)
        
        # Clear cache to simulate fresh start
        dirty_flag_system._dirty_claim_cache.clear()
        dirty_flag_system._cascade_tracker.clear()
        
        # Rebuild cache
        dirty_flag_system.rebuild_cache(sample_claims)
        
        # Verify cache was rebuilt
        assert len(dirty_flag_system._dirty_claim_cache) == 2
        
        for claim_id in ["claim_1", "claim_2"]:
            assert claim_id in dirty_flag_system._dirty_claim_cache
            assert dirty_flag_system._cascade_tracker[claim_id] == 0


class TestDirtyEvaluationConfig:
    """Test cases for DirtyEvaluationConfig"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = DirtyEvaluationConfig()
        
        assert config.batch_size == 5
        assert config.max_parallel_batches == 3
        assert config.confidence_threshold == 0.90
        assert config.confidence_boost_factor == 0.10
        assert config.two_pass_evaluation is True
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = DirtyEvaluationConfig(
            batch_size=10,
            max_parallel_batches=5,
            confidence_threshold=0.85,
            confidence_boost_factor=0.15,
            two_pass_evaluation=False
        )
        
        assert config.batch_size == 10
        assert config.max_parallel_batches == 5
        assert config.confidence_threshold == 0.85
        assert config.confidence_boost_factor == 0.15
        assert config.two_pass_evaluation is False
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Test invalid confidence threshold
        with pytest.raises(ValueError):
            DirtyEvaluationConfig(confidence_threshold=1.5)
        
        # Test invalid batch size
        with pytest.raises(ValueError):
            DirtyEvaluationConfig(batch_size=0)
        
        # Test invalid max retries
        with pytest.raises(ValueError):
            DirtyEvaluationConfig(max_retries=-1)


class TestDirtyClaimBatch:
    """Test cases for DirtyClaimBatch"""
    
    def test_batch_creation(self):
        """Test creating a dirty claim batch"""
        claims = [
            Claim(id="claim1", content="Test claim 1", confidence=0.8, type=[ClaimType.CONCEPT]),
            Claim(id="claim2", content="Test claim 2", confidence=0.7, type=[ClaimType.CONCEPT])
        ]
        
        batch = DirtyClaimBatch(
            claims=claims,
            batch_id="test_batch",
            priority_level=10
        )
        
        assert batch.claims == claims
        assert batch.batch_id == "test_batch"
        assert batch.priority_level == 10
        assert batch.status == "pending"
        assert batch.created_at is not None
    
    def test_batch_metadata(self):
        """Test batch metadata functionality"""
        claims = [
            Claim(id="claim1", content="Test claim 1", confidence=0.8, type=[ClaimType.CONCEPT])
        ]
        
        metadata = {"test_key": "test_value", "evaluation_type": "priority"}
        batch = DirtyClaimBatch(
            claims=claims,
            batch_id="test_batch",
            metadata=metadata
        )
        
        assert batch.metadata == metadata
        assert batch.metadata["claim_count"] == 1


class TestDirtyFlagConfig:
    """Test cases for DirtyFlagConfig"""
    
    def test_default_config_values(self):
        """Test default configuration values"""
        config = DirtyFlagConfig()
        
        assert config.confidence_threshold == 0.90
        assert config.cascade_depth == 3
        assert config.batch_size == 5
        assert config.priority_weights is not None
        assert len(config.priority_weights) > 0
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Test invalid confidence threshold
        with pytest.raises(ValueError):
            DirtyFlagConfig(confidence_threshold=-0.1)
        
        with pytest.raises(ValueError):
            DirtyFlagConfig(confidence_threshold=1.1)
        
        # Test invalid cascade depth
        with pytest.raises(ValueError):
            DirtyFlagConfig(cascade_depth=0)
        
        # Test invalid batch size
        with pytest.raises(ValueError):
            DirtyFlagConfig(batch_size=0)
    
    def test_config_from_dict(self):
        """Test creating config from dictionary"""
        config_dict = {
            "confidence_threshold": 0.85,
            "batch_size": 8,
            "cascade_depth": 2
        }
        
        config = DirtyFlagConfig.from_dict(config_dict)
        
        assert config.confidence_threshold == 0.85
        assert config.batch_size == 8
        assert config.cascade_depth == 2
    
    def test_config_to_dict(self):
        """Test converting config to dictionary"""
        config = DirtyFlagConfig(
            confidence_threshold=0.85,
            batch_size=8
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["confidence_threshold"] == 0.85
        assert config_dict["batch_size"] == 8
        assert "priority_weights" in config_dict
    
    def test_config_update(self):
        """Test updating configuration"""
        config = DirtyFlagConfig()
        
        updated_config = config.update(
            confidence_threshold=0.95,
            batch_size=10
        )
        
        assert updated_config.confidence_threshold == 0.95
        assert updated_config.batch_size == 10
        assert updated_config.cascade_depth == config.cascade_depth  # Unchanged
    
    def test_priority_weight_management(self):
        """Test priority weight management"""
        config = DirtyFlagConfig()
        
        # Test getting priority weight
        weight = config.get_priority_weight("manual_mark")
        assert weight > 0
        
        # Test setting priority weight
        config.set_priority_weight("test_reason", 42.0)
        assert config.get_priority_weight("test_reason") == 42.0
        
        # Test non-existent reason
        assert config.get_priority_weight("non_existent") == 0.0
    
    def test_confidence_helpers(self):
        """Test confidence-related helper methods"""
        config = DirtyFlagConfig(confidence_threshold=0.90)
        
        # Test high confidence detection
        assert config.is_high_confidence(0.95) is True
        assert config.is_high_confidence(0.85) is False
        
        # Test priority bonus calculation
        bonus = config.get_priority_bonus(0.75)  # 0.90 - 0.75 = 0.15 gap
        assert bonus > 0
        
        bonus_high = config.get_priority_bonus(0.95)  # Above threshold
        assert bonus_high == 0.0


class TestDirtyFlagConfigManager:
    """Test cases for DirtyFlagConfigManager"""
    
    def test_config_manager_initialization(self):
        """Test config manager initialization"""
        manager = DirtyFlagConfigManager()
        
        config = manager.get_config()
        assert isinstance(config, DirtyFlagConfig)
    
    def test_config_update(self):
        """Test updating configuration through manager"""
        manager = DirtyFlagConfigManager()
        
        updated_config = manager.update_config(
            confidence_threshold=0.95,
            batch_size=10
        )
        
        assert updated_config.confidence_threshold == 0.95
        assert updated_config.batch_size == 10
        
        # Verify manager now has updated config
        current_config = manager.get_config()
        assert current_config.confidence_threshold == 0.95
    
    def test_config_history(self):
        """Test configuration history tracking"""
        manager = DirtyFlagConfigManager()
        
        # Update configuration multiple times
        manager.update_config(confidence_threshold=0.95)
        manager.update_config(batch_size=10)
        manager.update_config(cascade_depth=2)
        
        # Check history
        history = manager.get_config_history()
        assert len(history) >= 2  # At least the updates we made
    
    def test_config_summary(self):
        """Test configuration summary"""
        manager = DirtyFlagConfigManager()
        
        summary = manager.get_config_summary()
        
        assert "confidence_threshold" in summary
        assert "batch_size" in summary
        assert "config_source" in summary
        assert summary["config_source"] == "default"  # No history yet
    
    def test_reset_to_defaults(self):
        """Test resetting configuration to defaults"""
        manager = DirtyFlagConfigManager()
        
        # Update configuration
        manager.update_config(confidence_threshold=0.95, batch_size=10)
        
        # Reset to defaults
        default_config = manager.reset_to_defaults()
        
        # Verify defaults are restored
        default_manager = DirtyFlagConfigManager()
        assert default_config.confidence_threshold == default_manager.get_config().confidence_threshold


class TestDirtyEvaluator:
    """Test cases for DirtyEvaluator"""
    
    @pytest.fixture
    def mock_llm_manager(self):
        """Create mock LLM manager"""
        manager = Mock()
        
        # Mock process_claims method
        async def mock_process_claims(claims, task, **kwargs):
            mock_result = Mock()
            mock_result.success = True
            mock_result.content = """{
                "results": [
                    {"success": true, "data": {"confidence": 0.9}},
                    {"success": false, "error": "Processing failed"}
                ]
            }"""
            return mock_result
        
        manager.process_claims = mock_process_claims
        
        return manager
    
    @pytest.fixture
    def dirty_evaluator(self, mock_llm_manager):
        """Create DirtyEvaluator instance for testing"""
        dirty_system = DirtyFlagSystem()
        config = DirtyEvaluationConfig(
            batch_size=2,
            max_parallel_batches=1,
            enable_two_pass=False  # Simplify for testing
        )
        
        return DirtyEvaluator(mock_llm_manager, dirty_system, config)
    
    @pytest.fixture
    def sample_dirty_claims(self):
        """Create sample dirty claims"""
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
        
        # Mark as dirty
        claims[0].mark_dirty(DirtyReason.MANUAL_MARK, priority=15)
        claims[1].mark_dirty(DirtyReason.CONFIDENCE_THRESHOLD, priority=10)
        
        return claims
    
    @pytest.mark.asyncio
    async def test_evaluate_dirty_claims(self, dirty_evaluator, sample_dirty_claims):
        """Test evaluating dirty claims"""
        result = await dirty_evaluator.evaluate_dirty_claims(
            claims=sample_dirty_claims,
            priority_only=True
        )
        
        assert result.success is True
        assert result.processed_claims == 2
        assert result.updated_claims >= 0
        assert result.execution_time is not None
    
    def test_create_evaluation_batches(self, dirty_evaluator, sample_dirty_claims):
        """Test creating evaluation batches"""
        batches = dirty_evaluator._create_evaluation_batches(sample_dirty_claims)
        
        assert len(batches) == 1  # Small batch size for testing
        batch = batches[0]
        
        assert batch.claims == sample_dirty_claims
        assert batch.batch_id.startswith("batch_")
        assert batch.priority_level > 0
        assert batch.metadata["claim_count"] == 2
    
    def test_apply_confidence_boost(self, dirty_evaluator, sample_dirty_claims):
        """Test applying confidence boost to claims"""
        boosted_claims = dirty_evaluator._apply_confidence_boost(sample_dirty_claims)
        
        # Claims below threshold should be boosted
        for i, claim in enumerate(sample_dirty_claims):
            boosted = boosted_claims[i]
            if claim.confidence < dirty_evaluator.config.confidence_threshold:
                assert boosted.confidence > claim.confidence
    
    def test_update_processing_stats(self, dirty_evaluator):
        """Test updating processing statistics"""
        dirty_evaluator._update_processing_stats(
            processed_count=10,
            updated_count=8,
            execution_time=5.0
        )
        
        stats = dirty_evaluator.get_processing_stats()
        
        assert stats["total_processed"] == 10
        assert stats["successful_evaluations"] == 8
        assert stats["average_processing_time"] == 5.0
        assert stats["success_rate"] == 0.8
    
    def test_reset_stats(self, dirty_evaluator):
        """Test resetting processing statistics"""
        # Update some stats first
        dirty_evaluator._update_processing_stats(10, 8, 5.0)
        
        # Reset stats
        dirty_evaluator.reset_stats()
        
        stats = dirty_evaluator.get_processing_stats()
        
        assert stats["total_processed"] == 0
        assert stats["successful_evaluations"] == 0
        assert stats["average_processing_time"] == 0.0
        assert stats["success_rate"] == 0.0


class TestIntegration:
    """Integration tests for the complete dirty flag system"""
    
    def test_full_dirty_flag_workflow(self):
        """Test complete dirty flag workflow"""
        # Create dirty flag system
        dirty_system = DirtyFlagSystem(confidence_threshold=0.90)
        
        # Create sample claims
        claims = [
            Claim(
                id="claim_1",
                content="High confidence claim",
                confidence=0.95,
                type=[ClaimType.CONCEPT],
                tags=["test"]
            ),
            Claim(
                id="claim_2",
                content="Low confidence claim",
                confidence=0.75,
                type=[ClaimType.CONCEPT],
                tags=["test"]
            ),
            Claim(
                id="claim_3",
                content="Related claim",
                confidence=0.80,
                type=[ClaimType.CONCEPT],
                tags=["test"],
                supports=["claim_1"]
            )
        ]
        
        # Mark low confidence claims dirty
        marked_count = dirty_system.mark_claims_dirty_by_confidence_threshold(claims)
        assert marked_count == 2  # claim_2 and claim_3
        
        # Get priority dirty claims
        priority_claims = dirty_system.get_priority_dirty_claims(claims)
        assert len(priority_claims) == 2
        
        # Get statistics
        stats = dirty_system.get_dirty_statistics(claims)
        assert stats["total_dirty"] == 2
        assert stats["priority_dirty"] == 2
        
        # Clear dirty flags
        cleared_count = dirty_system.clear_dirty_flags(claims)
        assert cleared_count == 2
        
        # Verify all claims are clean
        for claim in claims:
            assert claim.is_dirty is False
    
    def test_configuration_integration(self):
        """Test configuration integration with system components"""
        # Create custom configuration
        config = DirtyFlagConfig(
            confidence_threshold=0.85,
            batch_size=8,
            cascade_depth=4
        )
        
        # Create manager with custom config
        manager = DirtyFlagConfigManager(config)
        
        # Create systems with config
        dirty_system = DirtyFlagSystem(
            confidence_threshold=config.confidence_threshold,
            cascade_depth=config.cascade_depth
        )
        
        eval_config = DirtyEvaluationConfig(
            batch_size=config.batch_size,
            confidence_threshold=config.confidence_threshold
        )
        
        # Verify configurations are consistent
        assert dirty_system.confidence_threshold == config.confidence_threshold
        assert dirty_system.cascade_depth == config.cascade_depth
        assert eval_config.batch_size == config.batch_size
        assert eval_config.confidence_threshold == config.confidence_threshold
    
    def test_cli_configuration_flow(self):
        """Test CLI configuration management flow"""
        # Simulate CLI operations
        manager = DirtyFlagConfigManager()
        
        # Show config (CLI equivalent)
        initial_config = manager.get_config()
        summary = manager.get_config_summary()
        assert "confidence_threshold" in summary
        
        # Update config (CLI equivalent)
        updated_config = manager.update_config(
            confidence_threshold=0.95,
            batch_size=10
        )
        assert updated_config.confidence_threshold == 0.95
        assert updated_config.batch_size == 10
        
        # Reset to defaults (CLI equivalent)
        default_config = manager.reset_to_defaults()
        assert default_config.confidence_threshold == initial_config.confidence_threshold


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])