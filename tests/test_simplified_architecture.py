"""
Test suite for simplified OpenAI-compatible architecture
"""

import pytest
import asyncio
from unittest.mock import Mock, patch

from src.config.simplified_config import SimplifiedConfigManager, ProviderConfig
from src.processing.simplified_llm_manager import SimplifiedLLMManager
from src.processing.llm.openai_compatible_provider import OpenAICompatibleProcessor


class TestSimplifiedConfig:
    """Test simplified configuration system"""

    def test_provider_config_validation(self):
        """Test provider configuration validation"""
        # Valid provider
        provider = ProviderConfig(
            name="test_provider",
            url="https://api.example.com/v1",
            api="test-key",
            model="test-model",
            priority=1
        )
        assert provider.name == "test_provider"
        assert provider.url == "https://api.example.com/v1"
        assert provider.api == "test-key"
        assert provider.model == "test-model"
        assert provider.priority == 1

    def test_provider_config_invalid_url(self):
        """Test provider configuration with invalid URL"""
        with pytest.raises(ValueError):
            ProviderConfig(
                name="test_provider",
                url="invalid-url",
                api="test-key",
                model="test-model",
                priority=1
            )

    def test_config_manager_default_creation(self):
        """Test config manager creates default configuration"""
        config_manager = SimplifiedConfigManager()
        assert config_manager.config is not None
        assert len(config_manager.config.providers) > 0

    def test_config_manager_add_provider(self):
        """Test adding provider to configuration"""
        config_manager = SimplifiedConfigManager()
        provider = ProviderConfig(
            name="test_provider",
            url="https://api.example.com/v1",
            api="test-key",
            model="test-model",
            priority=1
        )
        
        config_manager.add_provider(provider)
        added_provider = config_manager.get_provider("test_provider")
        assert added_provider is not None
        assert added_provider.name == "test_provider"

    def test_config_manager_remove_provider(self):
        """Test removing provider from configuration"""
        config_manager = SimplifiedConfigManager()
        provider = ProviderConfig(
            name="test_provider",
            url="https://api.example.com/v1",
            api="test-key",
            model="test-model",
            priority=1
        )
        
        config_manager.add_provider(provider)
        assert config_manager.get_provider("test_provider") is not None
        
        removed = config_manager.remove_provider("test_provider")
        assert removed is True
        assert config_manager.get_provider("test_provider") is None


class TestOpenAICompatibleProvider:
    """Test OpenAI-compatible provider"""

    def test_provider_initialization(self):
        """Test provider initialization"""
        provider = OpenAICompatibleProcessor(
            api_key="test-key",
            api_url="https://api.example.com/v1",
            model_name="test-model",
            provider_name="test_provider"
        )
        
        assert provider.api_key == "test-key"
        assert provider.api_url == "https://api.example.com/v1"
        assert provider.model_name == "test-model"
        assert provider.provider_name == "test_provider"

    def test_local_provider_detection(self):
        """Test local provider detection"""
        provider = OpenAICompatibleProcessor(
            api_url="http://localhost:1234",
            provider_name="local_test"
        )
        
        assert provider._is_local_provider() is True
        
        provider = OpenAICompatibleProcessor(
            api_url="https://api.openai.com/v1",
            provider_name="openai_test"
        )
        
        assert provider._is_local_provider() is False

    def test_endpoint_url_construction(self):
        """Test endpoint URL construction for different providers"""
        # OpenAI
        provider = OpenAICompatibleProcessor(
            api_url="https://api.openai.com/v1",
            provider_name="openai"
        )
        assert provider._get_endpoint_url() == "https://api.openai.com/v1/chat/completions"
        
        # LM Studio
        provider = OpenAICompatibleProcessor(
            api_url="http://localhost:1234",
            provider_name="lm_studio"
        )
        assert provider._get_endpoint_url() == "http://localhost:1234/v1/chat/completions"
        
        # OpenRouter
        provider = OpenAICompatibleProcessor(
            api_url="https://openrouter.ai/api/v1",
            provider_name="openrouter"
        )
        assert provider._get_endpoint_url() == "https://openrouter.ai/api/v1/chat/completions"

    def test_headers_construction(self):
        """Test header construction for different providers"""
        # With API key
        provider = OpenAICompatibleProcessor(
            api_key="test-key",
            api_url="https://api.example.com/v1",
            provider_name="test_provider"
        )
        headers = provider._get_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-key"
        
        # Without API key (local)
        provider = OpenAICompatibleProcessor(
            api_url="http://localhost:1234",
            provider_name="local_test"
        )
        headers = provider._get_headers()
        assert "Authorization" not in headers


class TestSimplifiedLLMManager:
    """Test simplified LLM manager"""

    def test_manager_initialization(self):
        """Test manager initialization with providers"""
        providers = [
            {
                "name": "test_provider1",
                "url": "https://api.example.com/v1",
                "api": "test-key-1",
                "model": "test-model-1",
                "priority": 1
            },
            {
                "name": "test_provider2",
                "url": "https://api.example.com/v1",
                "api": "test-key-2",
                "model": "test-model-2",
                "priority": 2
            }
        ]
        
        manager = SimplifiedLLMManager(providers)
        assert len(manager.processors) == 2
        assert "test_provider1" in manager.processors
        assert "test_provider2" in manager.processors

    def test_provider_priority_ordering(self):
        """Test provider priority ordering"""
        providers = [
            {
                "name": "low_priority",
                "url": "https://api.example.com/v1",
                "api": "test-key",
                "model": "test-model",
                "priority": 10
            },
            {
                "name": "high_priority",
                "url": "https://api.example.com/v1",
                "api": "test-key",
                "model": "test-model",
                "priority": 1
            }
        ]
        
        manager = SimplifiedLLMManager(providers)
        # High priority provider should be primary
        assert manager.primary_provider == "high_priority"

    def test_get_available_providers(self):
        """Test getting available providers"""
        providers = [
            {
                "name": "available_provider",
                "url": "https://api.example.com/v1",
                "api": "test-key",
                "model": "test-model",
                "priority": 1
            }
        ]
        
        manager = SimplifiedLLMManager(providers)
        available = manager.get_available_providers()
        assert "available_provider" in available
        assert len(available) == 1

    def test_health_check(self):
        """Test health check functionality"""
        providers = [
            {
                "name": "test_provider",
                "url": "https://api.example.com/v1",
                "api": "test-key",
                "model": "test-model",
                "priority": 1
            }
        ]
        
        manager = SimplifiedLLMManager(providers)
        health = manager.health_check()
        
        assert "total_providers" in health
        assert health["total_providers"] == 1
        assert "overall_status" in health


class TestIntegration:
    """Test integration between simplified components"""

    @patch('requests.post')
    def test_end_to_end_provider_communication(self, mock_post):
        """Test end-to-end communication with mocked provider"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Test response"
                    }
                }
            ],
            "usage": {
                "total_tokens": 10,
                "prompt_tokens": 5,
                "completion_tokens": 5
            }
        }
        mock_post.return_value = mock_response
        
        # Test provider communication
        provider = OpenAICompatibleProcessor(
            api_key="test-key",
            api_url="https://api.example.com/v1",
            model_name="test-model"
        )
        
        result = provider.generate_response("Hello, world!")
        
        assert result.success is True
        assert len(result.processed_claims) == 0  # generate_response doesn't create claims
        assert result.tokens_used == 10
        assert result.model_used == "test-model"

    def test_fallback_mechanism(self):
        """Test provider fallback mechanism"""
        providers = [
            {
                "name": "primary_provider",
                "url": "https://api.example.com/v1",
                "api": "test-key",
                "model": "test-model",
                "priority": 1
            },
            {
                "name": "fallback_provider",
                "url": "https://api.example.com/v1",
                "api": "test-key",
                "model": "test-model",
                "priority": 2
            }
        ]
        
        manager = SimplifiedLLMManager(providers)
        
        # Mock primary provider to fail
        with patch.object(manager.processors["primary_provider"], 'generate_response') as mock_primary:
            mock_primary.side_effect = Exception("Primary provider failed")
            
            # Mock fallback provider to succeed
            with patch.object(manager.processors["fallback_provider"], 'generate_response') as mock_fallback:
                mock_fallback.return_value = Mock(success=True)
                
                # This should use fallback provider
                result = manager.generate_response("Test message")
                
                # Verify fallback was called
                mock_fallback.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])