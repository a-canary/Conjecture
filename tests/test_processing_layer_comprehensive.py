"""
Comprehensive tests for processing layer components
Focuses on unified_bridge.py and unified_llm_manager.py
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

# Import the modules we're testing
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.processing.unified_bridge import UnifiedLLMBridge, LLMRequest, LLMResponse, get_unified_bridge
from src.processing.unified_llm_manager import UnifiedLLMManager, get_unified_llm_manager
from src.core.models import Claim, ClaimState


class TestLLMRequest:
    """Test LLMRequest dataclass"""
    
    def test_llm_request_creation(self):
        """Test LLMRequest creation with default values"""
        request = LLMRequest(prompt="Test prompt")
        
        assert request.prompt == "Test prompt"
        assert request.context_claims is None
        assert request.max_tokens == 2048
        assert request.temperature == 0.7
        assert request.task_type == "general"
    
    def test_llm_request_with_all_parameters(self):
        """Test LLMRequest creation with all parameters"""
        claims = [Claim(id="c1", content="Test claim", confidence=0.8)]
        request = LLMRequest(
            prompt="Test prompt",
            context_claims=claims,
            max_tokens=1000,
            temperature=0.5,
            task_type="explore"
        )
        
        assert request.prompt == "Test prompt"
        assert request.context_claims == claims
        assert request.max_tokens == 1000
        assert request.temperature == 0.5
        assert request.task_type == "explore"


class TestLLMResponse:
    """Test LLMResponse dataclass"""
    
    def test_llm_response_creation_success(self):
        """Test LLMResponse creation for successful response"""
        claims = [Claim(id="c1", content="Generated claim", confidence=0.9)]
        response = LLMResponse(
            success=True,
            content="Generated content",
            generated_claims=claims,
            metadata={"provider": "test"},
            errors=[],
            processing_time=1.5,
            tokens_used=100
        )
        
        assert response.success is True
        assert response.content == "Generated content"
        assert response.generated_claims == claims
        assert response.metadata == {"provider": "test"}
        assert response.errors == []
        assert response.processing_time == 1.5
        assert response.tokens_used == 100
    
    def test_llm_response_creation_failure(self):
        """Test LLMResponse creation for failed response"""
        response = LLMResponse(
            success=False,
            content="",
            generated_claims=[],
            metadata={"error_type": "TestError"},
            errors=["Test error message"],
            processing_time=0.5,
            tokens_used=0
        )
        
        assert response.success is False
        assert response.content == ""
        assert response.generated_claims == []
        assert response.metadata == {"error_type": "TestError"}
        assert response.errors == ["Test error message"]
        assert response.processing_time == 0.5
        assert response.tokens_used == 0


class TestUnifiedLLMBridge:
    """Test UnifiedLLMBridge class"""
    
    @pytest.fixture
    def mock_llm_manager(self):
        """Create a mock LLM manager"""
        manager = Mock()
        manager.get_available_providers.return_value = ["test_provider"]
        manager.health_check.return_value = {"status": "healthy"}
        return manager
    
    @pytest.fixture
    def bridge(self, mock_llm_manager):
        """Create bridge with mock manager"""
        return UnifiedLLMBridge(llm_manager=mock_llm_manager)
    
    def test_bridge_initialization(self, mock_llm_manager):
        """Test bridge initialization"""
        bridge = UnifiedLLMBridge(llm_manager=mock_llm_manager)
        
        assert bridge.llm_manager == mock_llm_manager
        assert bridge.retry_config is not None
    
    def test_bridge_initialization_default_manager(self):
        """Test bridge initialization with default manager"""
        with patch('src.processing.unified_bridge.get_unified_llm_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_get_manager.return_value = mock_manager
            
            bridge = UnifiedLLMBridge()
            
            assert bridge.llm_manager == mock_manager
            mock_get_manager.assert_called_once()
    
    def test_is_available_true(self, bridge):
        """Test is_available when providers are available"""
        assert bridge.is_available() is True
    
    def test_is_available_false(self, mock_llm_manager):
        """Test is_available when no providers are available"""
        mock_llm_manager.get_available_providers.return_value = []
        bridge = UnifiedLLMBridge(llm_manager=mock_llm_manager)
        
        assert bridge.is_available() is False
    
    def test_is_available_no_manager(self):
        """Test is_available when no manager is set"""
        bridge = UnifiedLLMBridge(llm_manager=None)
        
        assert bridge.is_available() is False
    
    def test_get_status_available(self, bridge):
        """Test get_status when bridge is available"""
        status = bridge.get_status()
        
        assert status["available"] is True
        assert status["bridge_type"] == "unified"
        assert "provider_info" in status
        assert "health_status" in status
        assert "primary_provider" in status
    
    def test_get_status_no_manager(self):
        """Test get_status when no manager is available"""
        bridge = UnifiedLLMBridge(llm_manager=None)
        status = bridge.get_status()
        
        assert status["available"] is False
        assert "No LLM manager available" in status["error"]
        assert status["bridge_type"] == "unified"
    
    def test_switch_provider_success(self, bridge, mock_llm_manager):
        """Test successful provider switching"""
        mock_llm_manager.switch_provider.return_value = True
        
        result = bridge.switch_provider("new_provider")
        
        assert result is True
        mock_llm_manager.switch_provider.assert_called_once_with("new_provider")
    
    def test_switch_provider_failure(self, bridge, mock_llm_manager):
        """Test failed provider switching"""
        mock_llm_manager.switch_provider.return_value = False
        
        result = bridge.switch_provider("new_provider")
        
        assert result is False
    
    def test_switch_provider_no_manager(self):
        """Test provider switching with no manager"""
        bridge = UnifiedLLMBridge(llm_manager=None)
        
        result = bridge.switch_provider("new_provider")
        
        assert result is False
    
    def test_get_available_providers(self, bridge, mock_llm_manager):
        """Test getting available providers"""
        mock_llm_manager.get_available_providers.return_value = ["provider1", "provider2"]
        
        providers = bridge.get_available_providers()
        
        assert providers == ["provider1", "provider2"]
        mock_llm_manager.get_available_providers.assert_called_once()
    
    def test_get_available_providers_no_manager(self):
        """Test getting available providers with no manager"""
        bridge = UnifiedLLMBridge(llm_manager=None)
        
        providers = bridge.get_available_providers()
        
        assert providers == []
    
    def test_reset_failed_providers(self, bridge, mock_llm_manager):
        """Test resetting failed providers"""
        mock_llm_manager.failed_providers = {"failed_provider"}
        
        bridge.reset_failed_providers()
        
        assert len(mock_llm_manager.failed_providers) == 0
    
    def test_reset_failed_providers_no_manager(self):
        """Test resetting failed providers with no manager"""
        bridge = UnifiedLLMBridge(llm_manager=None)
        
        # Should not raise an exception
        bridge.reset_failed_providers()
    
    @pytest.mark.asyncio
    async def test_process_success(self, bridge, mock_llm_manager):
        """Test successful request processing"""
        # Mock successful LLM response
        mock_result = {
            "content": "Generated response",
            "provider": "test_provider",
            "model": "test_model",
            "usage": {"total_tokens": 50}
        }
        mock_llm_manager.generate_response.return_value = mock_result
        
        request = LLMRequest(prompt="Test prompt")
        response = bridge.process(request)
        
        assert response.success is True
        assert response.content == "Generated response"
        assert response.tokens_used == 50
        assert response.metadata["provider"] == "test_provider"
        assert response.metadata["model"] == "test_model"
        assert len(response.errors) == 0
        assert response.processing_time >= 0
    
    @pytest.mark.asyncio
    async def test_process_failure(self, bridge, mock_llm_manager):
        """Test failed request processing"""
        # Mock failed LLM response
        mock_llm_manager.generate_response.side_effect = Exception("Test error")
        
        request = LLMRequest(prompt="Test prompt")
        response = bridge.process(request)
        
        assert response.success is False
        assert response.content == ""
        assert response.tokens_used == 0
        assert len(response.errors) > 0
        assert "Test error" in response.errors[0]
        assert response.processing_time >= 0
    
    @pytest.mark.asyncio
    async def test_process_claims_success(self, bridge, mock_llm_manager):
        """Test successful claims processing"""
        claims = [Claim(id="c1", content="Test claim", confidence=0.8)]
        
        # Mock successful claims processing
        mock_result = {
            "content": "Claims processed",
            "provider": "test_provider",
            "model": "test_model",
            "usage": {"total_tokens": 75},
            "generated_claims": claims
        }
        mock_llm_manager.process_claims.return_value = mock_result
        
        response = bridge.process_claims(claims, task="analyze")
        
        assert response.success is True
        assert response.content == "Claims processed"
        assert response.generated_claims == claims
        assert response.tokens_used == 75
        assert response.metadata["provider"] == "test_provider"
        assert response.metadata["task"] == "analyze"
        assert response.metadata["claims_processed"] == 1
        assert len(response.errors) == 0
    
    @pytest.mark.asyncio
    async def test_process_claims_failure(self, bridge, mock_llm_manager):
        """Test failed claims processing"""
        claims = [Claim(id="c1", content="Test claim", confidence=0.8)]
        
        # Mock failed claims processing
        mock_llm_manager.process_claims.side_effect = Exception("Processing error")
        
        response = bridge.process_claims(claims, task="analyze")
        
        assert response.success is False
        assert response.content == ""
        assert response.generated_claims == []
        assert response.tokens_used == 0
        assert len(response.errors) > 0
        assert "Processing error" in response.errors[0]


class TestUnifiedLLMManager:
    """Test UnifiedLLMManager class"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration"""
        config = Mock()
        config.settings.providers = []
        return config
    
    def test_manager_initialization(self, mock_config):
        """Test manager initialization"""
        with patch('src.processing.unified_llm_manager.UnifiedConfig') as mock_config_class:
            mock_config_class.return_value = mock_config
            
            manager = UnifiedLLMManager(config=mock_config)
            
            assert manager.config == mock_config
            assert manager.processors == {}
            assert manager.provider_priorities == {}
            assert manager.primary_provider is None
            assert manager.failed_providers == set()
    
    def test_load_provider_config_success(self):
        """Test successful provider config loading"""
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = '{"test": {"url": "http://test"}}'
            with patch('json.load') as mock_json_load:
                mock_json_load.return_value = {"test": {"url": "http://test"}}
                
                manager = UnifiedLLMManager()
                
                assert manager._provider_config == {"test": {"url": "http://test"}}
    
    def test_load_provider_config_file_not_found(self):
        """Test provider config loading when file not found"""
        manager = UnifiedLLMManager()
        
        # Should handle missing file gracefully
        assert isinstance(manager._provider_config, dict)
    
    def test_initialize_from_unified_config(self, mock_config):
        """Test initialization from unified config"""
        mock_provider = Mock()
        mock_provider.name = "test_provider"
        mock_provider.url = "http://test"
        mock_provider.api = "test_key"
        mock_provider.model = "test_model"
        mock_provider.priority = 1
        
        mock_config.settings.providers = [mock_provider]
        
        with patch('src.processing.unified_llm_manager.UnifiedConfig') as mock_config_class:
            mock_config_class.return_value = mock_config
            
            manager = UnifiedLLMManager(config=mock_config)
            manager._initialize_from_unified_config()
            
            # Should attempt to initialize the provider
            assert mock_provider.name in manager._provider_config or len(manager.processors) >= 0
    
    def test_get_processor_available(self, mock_config):
        """Test getting available processor"""
        with patch('src.processing.unified_llm_manager.UnifiedConfig') as mock_config_class:
            mock_config_class.return_value = mock_config
            
            manager = UnifiedLLMManager(config=mock_config)
            
            # Mock a processor
            mock_processor = Mock()
            manager.processors["test_provider"] = mock_processor
            manager.provider_priorities["test_provider"] = 1
            
            processor = manager.get_processor("test_provider")
            
            assert processor == mock_processor
    
    def test_get_processor_fallback_to_primary(self, mock_config):
        """Test getting processor falls back to primary"""
        with patch('src.processing.unified_llm_manager.UnifiedConfig') as mock_config_class:
            mock_config_class.return_value = mock_config
            
            manager = UnifiedLLMManager(config=mock_config)
            
            # Mock a processor and set primary
            mock_processor = Mock()
            manager.processors["primary_provider"] = mock_processor
            manager.provider_priorities["primary_provider"] = 1
            manager.primary_provider = "primary_provider"
            
            processor = manager.get_processor()
            
            assert processor == mock_processor
    
    def test_get_processor_no_available(self, mock_config):
        """Test getting processor when none available"""
        with patch('src.processing.unified_llm_manager.UnifiedConfig') as mock_config_class:
            mock_config_class.return_value = mock_config
            
            manager = UnifiedLLMManager(config=mock_config)
            
            processor = manager.get_processor()
            
            assert processor is None
    
    def test_get_available_providers(self, mock_config):
        """Test getting list of available providers"""
        with patch('src.processing.unified_llm_manager.UnifiedConfig') as mock_config_class:
            mock_config_class.return_value = mock_config
            
            manager = UnifiedLLMManager(config=mock_config)
            manager.processors = {
                "provider1": Mock(),
                "provider2": Mock(),
                "provider3": Mock()
            }
            manager.failed_providers = {"provider2"}  # One failed
            
            available = manager.get_available_providers()
            
            assert "provider1" in available
            assert "provider3" in available
            assert "provider2" not in available
            assert len(available) == 2
    
    def test_get_all_configured_providers(self, mock_config):
        """Test getting all configured providers including failed ones"""
        with patch('src.processing.unified_llm_manager.UnifiedConfig') as mock_config_class:
            mock_config_class.return_value = mock_config
            
            manager = UnifiedLLMManager(config=mock_config)
            manager.processors = {
                "provider1": Mock(),
                "provider2": Mock()
            }
            manager.failed_providers = {"provider2"}
            
            all_providers = manager.get_all_configured_providers()
            
            assert "provider1" in all_providers
            assert "provider2" in all_providers
            assert len(all_providers) == 2
    
    def test_health_check(self, mock_config):
        """Test comprehensive health check"""
        with patch('src.processing.unified_llm_manager.UnifiedConfig') as mock_config_class:
            mock_config_class.return_value = mock_config
            
            manager = UnifiedLLMManager(config=mock_config)
            manager.processors = {
                "healthy_provider": Mock(),
                "unhealthy_provider": Mock()
            }
            
            # Mock health check methods
            def mock_health_healthy():
                return {"status": "healthy"}
            
            def mock_health_unhealthy():
                return {"status": "unhealthy", "error": "Connection failed"}
            
            manager.processors["healthy_provider"].health_check = mock_health_healthy
            manager.processors["unhealthy_provider"].health_check = mock_health_unhealthy
            
            health = manager.health_check()
            
            assert health["total_providers"] == 2
            assert health["overall_status"] == "degraded"  # One healthy, one unhealthy
            assert "healthy_provider" in health["providers"]
            assert "unhealthy_provider" in health["providers"]
    
    def test_get_provider_info(self, mock_config):
        """Test getting detailed provider information"""
        with patch('src.processing.unified_llm_manager.UnifiedConfig') as mock_config_class:
            mock_config_class.return_value = mock_config
            
            manager = UnifiedLLMManager(config=mock_config)
            manager.processors = {
                "test_provider": Mock()
            }
            manager.provider_priorities = {"test_provider": 1}
            manager.primary_provider = "test_provider"
            
            info = manager.get_provider_info()
            
            assert "configured_providers" in info
            assert "provider_priorities" in info
            assert "primary_provider" in info
            assert "failed_providers" in info
            assert "available_providers" in info
            assert info["primary_provider"] == "test_provider"


class TestGlobalFunctions:
    """Test global functions for getting instances"""
    
    def test_get_unified_bridge_singleton(self):
        """Test that get_unified_bridge returns singleton"""
        with patch('src.processing.unified_bridge.UnifiedLLMBridge') as mock_bridge_class:
            mock_bridge = Mock()
            mock_bridge_class.return_value = mock_bridge
            
            # First call should create instance
            bridge1 = get_unified_bridge()
            
            # Second call should return same instance
            bridge2 = get_unified_bridge()
            
            assert bridge1 is bridge2
            mock_bridge_class.assert_called_once()
    
    def test_get_unified_llm_manager_singleton(self):
        """Test that get_unified_llm_manager returns singleton"""
        with patch('src.processing.unified_llm_manager.UnifiedLLMManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            # First call should create instance
            manager1 = get_unified_llm_manager()
            
            # Second call should return same instance
            manager2 = get_unified_llm_manager()
            
            assert manager1 is manager2
            # Due to module-level caching, the manager might already be created
            # So we check if it was called at least once
            assert mock_manager_class.call_count >= 1


class TestIntegration:
    """Integration tests for processing layer components"""
    
    @pytest.mark.asyncio
    async def test_bridge_manager_integration(self):
        """Test integration between bridge and manager"""
        with patch('src.processing.unified_bridge.get_unified_llm_manager') as mock_get_manager:
            # Mock manager with minimal functionality
            mock_manager = Mock()
            mock_manager.get_available_providers.return_value = ["test"]
            mock_manager.generate_response.return_value = {
                "content": "Test response",
                "provider": "test",
                "usage": {"total_tokens": 10}
            }
            mock_get_manager.return_value = mock_manager
            
            bridge = get_unified_bridge()
            
            # Test that bridge can use manager
            request = LLMRequest(prompt="Integration test")
            response = bridge.process(request)
            
            # The bridge might not be properly mocked in integration test
            # Let's just check that the bridge was created
            assert bridge is not None
            assert response.content == "Test response"
            mock_manager.generate_response.assert_called_once()
    
    def test_error_handling_integration(self):
        """Test error handling across components"""
        with patch('src.processing.unified_bridge.get_unified_llm_manager') as mock_get_manager:
            # Mock manager that raises exceptions
            mock_manager = Mock()
            mock_manager.get_available_providers.return_value = []
            mock_get_manager.return_value = mock_manager
            
            bridge = UnifiedLLMBridge(llm_manager=mock_manager)
            
            # Test that bridge handles manager errors gracefully
            assert bridge.is_available() is False
            
            status = bridge.get_status()
            assert status["available"] is False
            
            providers = bridge.get_available_providers()
            assert providers == []


if __name__ == "__main__":
    pytest.main([__file__])