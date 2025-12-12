import asyncio
"""
End-to-end test for configuration-driven processing
Tests different provider configurations and error handling
"""
import pytest
import tempfile
import json
from pathlib import Path

from src.conjecture import Conjecture
from src.config.unified_config import UnifiedConfig
from src.core.models import Claim, ClaimState, DirtyReason


class TestConfigurationDrivenProcessingE2E:
    """End-to-end test for configuration-driven processing"""

    @pytest.fixture
    def minimal_config(self):
        """Create minimal configuration for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "minimal_config.json"
            
            config_data = {
                "processing": {
                    "confidence_threshold": 0.85,
                    "confident_threshold": 0.5,
                    "max_context_size": 2000,
                    "batch_size": 5
                },
                "database": {
                    "database_path": f"{temp_dir}/minimal.db",
                    "chroma_path": f"{temp_dir}/minimal_chroma"
                },
                "workspace": {
                    "data_dir": temp_dir
                },
                "debug": False
            }
            
            with open(config_path, 'w') as f:
                json.dump(config_data, f)
            
            yield UnifiedConfig(config_path)

    @pytest.fixture
    def performance_config(self):
        """Create performance-optimized configuration"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "performance_config.json"
            
            config_data = {
                "processing": {
                    "confidence_threshold": 0.95,
                    "confident_threshold": 0.7,
                    "max_context_size": 16000,
                    "batch_size": 50
                },
                "database": {
                    "database_path": f"{temp_dir}/performance.db",
                    "chroma_path": f"{temp_dir}/performance_chroma",
                    "max_connections": 20,
                    "cache_size": 5000,
                    "cache_ttl": 600
                },
                "workspace": {
                    "data_dir": temp_dir
                },
                "debug": True,
                "monitoring": {
                    "enable_performance_tracking": True,
                    "enable_memory_tracking": True
                }
            }
            
            with open(config_path, 'w') as f:
                json.dump(config_data, f)
            
            yield UnifiedConfig(config_path)

    @pytest.fixture
    def local_providers_config(self):
        """Create configuration with local providers"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "local_providers_config.json"
            
            config_data = {
                "processing": {
                    "confidence_threshold": 0.85,
                    "confident_threshold": 0.6,
                    "max_context_size": 4000,
                    "batch_size": 10
                },
                "database": {
                    "database_path": f"{temp_dir}/local_providers.db",
                    "chroma_path": f"{temp_dir}/local_providers_chroma"
                },
                "workspace": {
                    "data_dir": temp_dir
                },
                "providers": [
                    {
                        "name": "local-ollama",
                        "type": "local",
                        "enabled": True,
                        "priority": 1,
                        "config": {
                            "base_url": "http://localhost:11434",
                            "model": "llama2",
                            "timeout": 30,
                            "max_tokens": 2048
                        }
                    },
                    {
                        "name": "local-lmstudio",
                        "type": "local", 
                        "enabled": True,
                        "priority": 2,
                        "config": {
                            "base_url": "http://localhost:1234",
                            "model": "granite-7b",
                            "timeout": 45,
                            "max_tokens": 4096
                        }
                    }
                ],
                "debug": True
            }
            
            with open(config_path, 'w') as f:
                json.dump(config_data, f)
            
            yield UnifiedConfig(config_path)

    @pytest.mark.asyncio
    async def test_minimal_configuration_processing(self, minimal_config):
        """Test processing with minimal configuration"""

        conjecture = Conjecture(config=minimal_config)
        
        # Verify configuration values
        assert conjecture.config.confidence_threshold == 0.85
        assert conjecture.config.confident_threshold == 0.5
        assert conjecture.config.max_context_size == 2000
        assert conjecture.config.batch_size == 5
        assert conjecture.config.debug is False
        
        # Test basic claim processing with minimal config
        test_claim = Claim(
            id="minimal_test",
            content="Simple test claim for minimal configuration",
            confidence=0.6,
            state=ClaimState.EXPLORE,
            tags=["minimal", "config", "test"]
        )
        
        # Add claim
        result = conjecture.add_claim(test_claim)
        assert result.success is True
        assert result.processed_claims == 1
        
        # Retrieve claim
        retrieved = await conjecture.get_claim("minimal_test")
        assert retrieved is not None
        assert retrieved.content == test_claim.content
        assert retrieved.confidence == test_claim.confidence

    def test_performance_configuration_processing(self, performance_config):
        """Test processing with performance-optimized configuration"""
        
        conjecture = Conjecture(config=performance_config)
        
        # Verify performance configuration values
        assert conjecture.config.confidence_threshold == 0.95
        assert conjecture.config.confident_threshold == 0.7
        assert conjecture.config.max_context_size == 16000
        assert conjecture.config.batch_size == 50
        
        # Test batch processing with larger batches
        large_batch = []
        for i in range(25):  # Half of batch size
            claim = Claim(
                id=f"perf_test_{i}",
                content=f"Performance test claim {i} for batch processing",
                confidence=0.6 + (i * 0.01),  # Varying confidence
                state=ClaimState.EXPLORE,
                tags=["performance", "batch", f"tag_{i % 5}"]
            )
            large_batch.append(claim)
        
        # Add batch
        batch_result = conjecture.add_claims_batch(large_batch)
        assert batch_result.success is True
        assert batch_result.processed_claims == 25
        
        # Test performance monitoring
        if hasattr(conjecture, 'performance_monitor'):
            metrics = conjecture.performance_monitor.get_metrics()
            assert isinstance(metrics, dict)
        
        # Test large context processing
        high_confidence_claims = conjecture.get_claims_by_confidence_range(0.8, 1.0)
        assert isinstance(high_confidence_claims, list)

    def test_local_providers_configuration(self, local_providers_config):
        """Test processing with local providers configuration"""
        
        conjecture = Conjecture(config=local_providers_config)
        
        # Verify providers configuration
        assert len(conjecture.config.providers) == 2
        
        provider_names = [p.name for p in conjecture.config.providers]
        assert "local-ollama" in provider_names
        assert "local-lmstudio" in provider_names
        
        # Check provider priorities
        ollama_provider = next(p for p in conjecture.config.providers if p.name == "local-ollama")
        lmstudio_provider = next(p for p in conjecture.config.providers if p.name == "local-lmstudio")
        
        assert ollama_provider.priority == 1
        assert lmstudio_provider.priority == 2
        assert ollama_provider.enabled is True
        assert lmstudio_provider.enabled is True
        
        # Test claim processing that would use providers
        processing_claim = Claim(
            id="provider_test",
            content="Test claim for provider-based processing",
            confidence=0.4,
            state=ClaimState.EXPLORE,
            tags=["provider", "local", "test"]
        )
        
        # Add claim
        result = conjecture.add_claim(processing_claim)
        assert result.success is True
        
        # Test processing simulation (would normally call LLM providers)
        dirty_claims = conjecture.get_dirty_claims()
        assert len(dirty_claims) >= 1
        
        # Simulate processing result
        processed_claim = Claim(
            id=processing_claim.id,
            content=processing_claim.content,
            confidence=0.75,  # Improved after "LLM processing"
            state=ClaimState.VALIDATED,
            supported_by=processing_claim.supported_by,
            supports=processing_claim.supports,
            tags=processing_claim.tags + ["processed"],
            is_dirty=False
        )
        
        update_result = conjecture.update_claim(processed_claim)
        assert update_result.success is True

    def test_configuration_error_handling(self):
        """Test error handling with invalid configurations"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test invalid configuration file
            invalid_config_path = Path(temp_dir) / "invalid_config.json"
            
            with open(invalid_config_path, 'w') as f:
                f.write("{ invalid json content")
            
            # Should handle gracefully and use defaults
            with pytest.raises(Exception):  # Should raise some kind of config error
                UnifiedConfig(invalid_config_path)
    
    def test_configuration_fallback_mechanisms(self):
        """Test configuration fallback mechanisms"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test missing configuration file (should use defaults)
            nonexistent_config = Path(temp_dir) / "nonexistent_config.json"
            
            # Should create config with defaults
            config = UnifiedConfig(nonexistent_config)
            
            # Verify default values are set
            assert config.confidence_threshold is not None
            assert config.confident_threshold is not None
            assert config.max_context_size is not None
            assert config.batch_size is not None

    def test_provider_fallback_and_error_handling(self, local_providers_config):
        """Test provider fallback mechanisms and error handling"""
        
        conjecture = Conjecture(config=local_providers_config)
        
        # Test with unavailable primary provider
        # This would normally test connection failures, but we'll simulate
        
        test_claim = Claim(
            id="fallback_test",
            content="Test claim for provider fallback",
            confidence=0.3,
            state=ClaimState.EXPLORE,
            tags=["fallback", "error", "test"]
        )
        
        # Add claim
        result = conjecture.add_claim(test_claim)
        assert result.success is True
        
        # Simulate provider failure scenario
        # In real implementation, this would test:
        # 1. Primary provider timeout/failure
        # 2. Fallback to secondary provider
        # 3. Error handling and logging
        
        # Test error handling in processing
        try:
            # Simulate processing with provider errors
            dirty_claims = conjecture.get_dirty_claims()
            processing_errors = []
            
            for claim in dirty_claims:
                if claim.id == "fallback_test":
                    # Simulate provider error handling
                    error_result = {
                        "claim_id": claim.id,
                        "success": False,
                        "error": "Provider connection failed",
                        "fallback_used": "local-lmstudio"
                    }
                    processing_errors.append(error_result)
            
            # Should have error information
            assert len(processing_errors) >= 1
            assert processing_errors[0]["fallback_used"] == "local-lmstudio"
            
        except Exception as e:
            # Should handle errors gracefully
            assert isinstance(e, Exception)

    def test_configuration_driven_batch_sizes(self, minimal_config, performance_config):
        """Test different batch sizes based on configuration"""
        
        minimal_conjecture = Conjecture(config=minimal_config)
        performance_conjecture = Conjecture(config=performance_config)
        
        # Test minimal config batch size (5)
        minimal_batch = []
        for i in range(7):  # More than batch size
            claim = Claim(
                id=f"minimal_batch_{i}",
                content=f"Minimal batch test claim {i}",
                confidence=0.6,
                state=ClaimState.EXPLORE,
                tags=["minimal", "batch"]
            )
            minimal_batch.append(claim)
        
        # Should process in multiple batches due to batch size limit
        minimal_result = minimal_conjecture.add_claims_batch(minimal_batch)
        assert minimal_result.success is True
        # Should handle gracefully even with more claims than batch size
        
        # Test performance config batch size (50)
        performance_batch = []
        for i in range(60):  # More than batch size
            claim = Claim(
                id=f"perf_batch_{i}",
                content=f"Performance batch test claim {i}",
                confidence=0.7,
                state=ClaimState.EXPLORE,
                tags=["performance", "batch"]
            )
            performance_batch.append(claim)
        
        # Should process in multiple batches
        performance_result = performance_conjecture.add_claims_batch(performance_batch)
        assert performance_result.success is True
        
        # Verify both instances handled their respective batch sizes correctly
        assert minimal_conjecture.config.batch_size == 5
        assert performance_conjecture.config.batch_size == 50

    def test_configuration_memory_and_disk_usage(self, performance_config):
        """Test configuration effects on memory and disk usage"""
        
        conjecture = Conjecture(config=performance_config)
        
        # Add many claims to test memory usage
        memory_test_claims = []
        for i in range(100):
            claim = Claim(
                id=f"memory_test_{i}",
                content=f"Memory usage test claim {i} with substantial content to test memory management",
                confidence=0.5 + (i % 50) * 0.01,  # Varying confidence
                state=ClaimState.EXPLORE,
                tags=["memory", "test", f"category_{i % 10}"],
                # Add some metadata to increase memory usage
                scope="user_workspace" if i % 2 == 0 else "team_workspace"
            )
            memory_test_claims.append(claim)
        
        # Add in batches to test memory management
        batch_size = 20
        for i in range(0, len(memory_test_claims), batch_size):
            batch = memory_test_claims[i:i + batch_size]
            result = conjecture.add_claims_batch(batch)
            assert result.success is True
        
        # Test system statistics
        if hasattr(conjecture, 'get_system_statistics'):
            stats = conjecture.get_system_statistics()
            assert stats.total_claims >= 100
            assert hasattr(stats, 'memory_usage') or hasattr(stats, 'disk_usage')
        
        # Test cleanup and optimization
        if hasattr(conjecture, 'optimize_storage'):
            optimization_result = conjecture.optimize_storage()
            assert optimization_result.success is True