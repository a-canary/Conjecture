"""
Unit tests for config/common.py module
Tests ProviderConfig and CommonConfig classes without mocking
"""
import pytest
from pydantic import ValidationError

from src.config.common import ProviderConfig, CommonConfig


class TestProviderConfig:
    """Test ProviderConfig class"""

    def test_minimal_provider_config(self):
        """Test creating ProviderConfig with minimal required fields"""
        provider = ProviderConfig(
            name="test_provider",
            url="https://test.com",
            model="test-model"
        )
        
        assert provider.name == "test_provider"
        assert provider.url == "https://test.com"
        assert provider.model == "test-model"
        assert provider.api == ""  # Default value
        assert provider.available is True  # Default value
        assert provider.timeout == 30  # Default value
        assert provider.max_retries == 3  # Default value

    def test_full_provider_config(self):
        """Test creating ProviderConfig with all fields"""
        provider = ProviderConfig(
            name="full_provider",
            url="https://full.com",
            api="test-api-key",
            model="full-model",
            available=False,
            timeout=60,
            max_retries=5
        )
        
        assert provider.name == "full_provider"
        assert provider.url == "https://full.com"
        assert provider.api == "test-api-key"
        assert provider.model == "full-model"
        assert provider.available is False
        assert provider.timeout == 60
        assert provider.max_retries == 5

    def test_provider_config_validation(self):
        """Test ProviderConfig validation"""
        # Valid config should not raise exception
        provider = ProviderConfig(
            name="valid",
            url="https://valid.com",
            model="valid-model"
        )
        assert provider is not None
        
        # All fields should accept valid values
        provider2 = ProviderConfig(
            name="test",
            url="http://localhost:11434",
            api="",
            model="llama2",
            available=True,
            timeout=120,
            max_retries=10
        )
        assert provider2.timeout == 120
        assert provider2.max_retries == 10


class TestCommonConfig:
    """Test CommonConfig class"""

    def test_minimal_common_config(self):
        """Test creating CommonConfig with default values"""
        config = CommonConfig()
        
        assert config.providers == []  # Default empty list
        assert config.debug is False  # Default value
        assert config.confidence_threshold == 0.95  # Default value
        assert config.max_context_size == 10  # Default value
        assert config.database_path == "data/conjecture.db"  # Default value
        assert config.user == "user"  # Default value
        assert config.team == "default"  # Default value

    def test_common_config_with_providers(self):
        """Test creating CommonConfig with providers"""
        provider1 = ProviderConfig(name="provider1", url="https://p1.com", model="model1")
        provider2 = ProviderConfig(name="provider2", url="https://p2.com", model="model2")
        
        config = CommonConfig(
            providers=[provider1, provider2],
            debug=True,
            confidence_threshold=0.8,
            max_context_size=20,
            database_path="custom.db",
            user="custom_user",
            team="custom_team"
        )
        
        assert len(config.providers) == 2
        assert config.providers[0].name == "provider1"
        assert config.providers[1].name == "provider2"
        assert config.debug is True
        assert config.confidence_threshold == 0.8
        assert config.max_context_size == 20
        assert config.database_path == "custom.db"
        assert config.user == "custom_user"
        assert config.team == "custom_team"

    def test_from_dict_method(self):
        """Test CommonConfig.from_dict class method"""
        config_dict = {
            "providers": [
                {
                    "name": "test_provider",
                    "url": "https://test.com",
                    "api": "test_key",
                    "model": "test_model"
                }
            ],
            "debug": True,
            "confidence_threshold": 0.85,
            "max_context_size": 15,
            "database_path": "test.db",
            "user": "test_user",
            "team": "test_team"
        }
        
        config = CommonConfig.from_dict(config_dict)
        
        assert len(config.providers) == 1
        assert config.providers[0].name == "test_provider"
        assert config.providers[0].url == "https://test.com"
        assert config.providers[0].api == "test_key"
        assert config.providers[0].model == "test_model"
        assert config.debug is True
        assert config.confidence_threshold == 0.85
        assert config.max_context_size == 15
        assert config.database_path == "test.db"
        assert config.user == "test_user"
        assert config.team == "test_team"

    def test_from_dict_with_defaults(self):
        """Test CommonConfig.from_dict with missing fields (should use defaults)"""
        config_dict = {
            "providers": [
                {
                    "name": "minimal_provider",
                    "url": "https://minimal.com",
                    "model": "minimal_model"
                }
            ],
            "debug": True
        }
        
        config = CommonConfig.from_dict(config_dict)
        
        assert len(config.providers) == 1
        assert config.providers[0].name == "minimal_provider"
        assert config.debug is True
        assert config.confidence_threshold == 0.95  # Default value
        assert config.max_context_size == 10  # Default value
        assert config.database_path == "data/conjecture.db"  # Default value
        assert config.user == "user"  # Default value
        assert config.team == "default"  # Default value

    def test_from_dict_empty_providers(self):
        """Test CommonConfig.from_dict with no providers"""
        config_dict = {"debug": False}
        
        config = CommonConfig.from_dict(config_dict)
        
        assert config.providers == []
        assert config.debug is False
        assert config.confidence_threshold == 0.95  # Default value

    def test_config_immutability(self):
        """Test that config objects maintain their values"""
        provider = ProviderConfig(name="immutable", url="https://immutable.com", model="immutable_model")
        config = CommonConfig(providers=[provider], debug=True)
        
        # Values should remain unchanged
        assert config.providers[0].name == "immutable"
        assert config.debug is True
        assert len(config.providers) == 1