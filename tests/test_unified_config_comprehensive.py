#!/usr/bin/env python3
"""
Comprehensive Tests for Unified Configuration System
Tests provider configuration, loading, validation, and error handling
"""

import pytest
import sys
import os
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.config.unified_config import (
    UnifiedConfig, get_config, validate_config, reload_config
)
from src.config.settings_models import ProviderConfig, ConjectureSettings


class TestUnifiedConfig:
    """Test UnifiedConfig class"""
    
    def test_unified_config_creation_default(self):
        """Test creating UnifiedConfig with default settings"""
        config = UnifiedConfig()
        
        assert config is not None
        assert isinstance(config.settings, ConjectureSettings)
        assert config.confidence_threshold >= 0.0
        assert config.confidence_threshold <= 1.0
        assert config.max_context_size > 0
        assert config.batch_size > 0
        assert isinstance(config.debug, bool)
        assert isinstance(config.database_path, str)
        assert isinstance(config.data_dir, str)
        assert isinstance(config.workspace, str)
        assert isinstance(config.user, str)
        assert isinstance(config.team, str)
    
    def test_unified_config_creation_with_path(self):
        """Test creating UnifiedConfig with custom config path"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "providers": [
                    {
                        "name": "test_provider",
                        "url": "https://test.example.com",
                        "api_key": "test_key",
                        "model": "test_model"
                    }
                ],
                "confidence_threshold": 0.9,
                "max_context_size": 4000,
                "batch_size": 50,
                "debug": True,
                "database_path": "/test/path.db",
                "workspace": "test_workspace",
                "user": "test_user",
                "team": "test_team"
            }
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = UnifiedConfig(temp_path)
            
            assert config.confidence_threshold == 0.9
            assert config.max_context_size == 4000
            assert config.batch_size == 50
            assert config.debug == True
            assert config.database_path == "/test/path.db"
            assert config.workspace == "test_workspace"
            assert config.user == "test_user"
            assert config.team == "test_team"
            
            # Test providers
            providers = config.get_providers()
            assert len(providers) == 1
            assert providers[0]["name"] == "test_provider"
            assert providers[0]["url"] == "https://test.example.com"
            
        finally:
            os.unlink(temp_path)
    
    def test_backward_compatibility_properties(self):
        """Test backward compatibility property access"""
        config = UnifiedConfig()
        
        # Test all properties exist and return expected types
        assert isinstance(config.confidence_threshold, float)
        assert isinstance(config.confident_threshold, float)
        assert isinstance(config.max_context_size, int)
        assert isinstance(config.batch_size, int)
        assert isinstance(config.debug, bool)
        assert isinstance(config.database_path, str)
        assert isinstance(config.data_dir, str)
        assert isinstance(config.workspace, str)
        assert isinstance(config.user, str)
        assert isinstance(config.team, str)
    
    def test_providers_property(self):
        """Test providers property returns list of dictionaries"""
        config = UnifiedConfig()
        
        providers = config.providers
        assert isinstance(providers, list)
        
        if providers:  # If there are providers configured
            for provider in providers:
                assert isinstance(provider, dict)
                assert "name" in provider
                assert "url" in provider
                assert "model" in provider
    
    def test_get_providers_method(self):
        """Test get_providers method"""
        config = UnifiedConfig()
        
        providers = config.get_providers()
        assert isinstance(providers, list)
        # Should return same as providers property
        assert providers == config.providers
    
    def test_get_primary_provider(self):
        """Test get_primary_provider method"""
        # Test with custom config that has providers
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "providers": [
                    {
                        "name": "primary_provider",
                        "url": "https://primary.example.com",
                        "api_key": "primary_key",
                        "model": "primary_model"
                    },
                    {
                        "name": "secondary_provider",
                        "url": "https://secondary.example.com",
                        "api_key": "secondary_key",
                        "model": "secondary_model"
                    }
                ]
            }
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = UnifiedConfig(temp_path)
            
            primary = config.get_primary_provider()
            assert primary is not None
            assert primary["name"] == "primary_provider"
            assert primary["url"] == "https://primary.example.com"
            
        finally:
            os.unlink(temp_path)
        
        # Test with no providers
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {"providers": []}
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = UnifiedConfig(temp_path)
            primary = config.get_primary_provider()
            assert primary is None
            
        finally:
            os.unlink(temp_path)
    
    def test_is_workspace_config(self):
        """Test is_workspace_config method"""
        config = UnifiedConfig()
        
        # This should return a boolean
        result = config.is_workspace_config()
        assert isinstance(result, bool)
    
    def test_get_config_info(self):
        """Test get_config_info method"""
        config = UnifiedConfig()
        
        info = config.get_config_info()
        assert isinstance(info, dict)
        
        # Should contain some expected keys
        expected_keys = ["config_path", "has_providers", "provider_count"]
        for key in expected_keys:
            assert key in info
    
    def test_get_effective_confident_threshold(self):
        """Test get_effective_confident_threshold method"""
        config = UnifiedConfig()
        
        threshold = config.get_effective_confident_threshold()
        assert isinstance(threshold, float)
        assert 0.0 <= threshold <= 1.0
    
    def test_to_dict(self):
        """Test to_dict method"""
        config = UnifiedConfig()
        
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        
        # Should contain configuration data
        assert "providers" in config_dict
        assert "confidence_threshold" in config_dict
        assert "max_context_size" in config_dict
    
    def test_reload_config(self):
        """Test reload_config method"""
        config = UnifiedConfig()
        
        # Get initial state
        initial_threshold = config.confidence_threshold
        
        # Reload should not raise an exception
        config.reload_config()
        
        # Should still be a valid config
        assert isinstance(config.confidence_threshold, float)
        assert 0.0 <= config.confidence_threshold <= 1.0
    
    def test_save_config(self):
        """Test save_config method"""
        config = UnifiedConfig()
        
        # Test saving to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # This should not raise an exception
            config.save_config(temp_path)
            
            # Verify file was created and contains valid JSON
            assert os.path.exists(temp_path)
            with open(temp_path, 'r') as f:
                saved_data = json.load(f)
            assert isinstance(saved_data, dict)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestConfigurationValidation:
    """Test configuration validation"""
    
    def test_validate_config_success(self):
        """Test validate_config with valid configuration"""
        # Create a temporary valid config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "providers": [
                    {
                        "name": "valid_provider",
                        "url": "https://valid.example.com",
                        "api_key": "valid_key",
                        "model": "valid_model"
                    }
                ]
            }
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            # Temporarily set config path
            original_config = get_config()
            
            # Create new config with our test file
            test_config = UnifiedConfig(temp_path)
            
            # Mock the global config for validation
            import src.config.unified_config
            original_global = src.config.unified_config._config
            src.config.unified_config._config = test_config
            
            try:
                result = validate_config()
                assert result == True
            finally:
                # Restore global config
                src.config.unified_config._config = original_global
                
        finally:
            os.unlink(temp_path)
    
    def test_validate_config_no_providers(self):
        """Test validate_config fails with no providers"""
        # Create a temporary config with no providers
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {"providers": []}
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            test_config = UnifiedConfig(temp_path)
            
            # Mock the global config for validation
            import src.config.unified_config
            original_global = src.config.unified_config._config
            src.config.unified_config._config = test_config
            
            try:
                result = validate_config()
                assert result == False
            finally:
                # Restore global config
                src.config.unified_config._config = original_global
                
        finally:
            os.unlink(temp_path)
    
    def test_validate_config_missing_fields(self):
        """Test validate_config fails with missing required fields"""
        # Create a temporary config with missing fields
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "providers": [
                    {
                        "name": "incomplete_provider",
                        "url": "https://example.com"
                        # Missing model field
                    }
                ]
            }
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            test_config = UnifiedConfig(temp_path)
            
            # Mock the global config for validation
            import src.config.unified_config
            original_global = src.config.unified_config._config
            src.config.unified_config._config = test_config
            
            try:
                result = validate_config()
                assert result == False
            finally:
                # Restore global config
                src.config.unified_config._config = original_global
                
        finally:
            os.unlink(temp_path)


class TestGlobalConfigFunctions:
    """Test global configuration functions"""
    
    def test_get_config_singleton(self):
        """Test get_config returns singleton instance"""
        config1 = get_config()
        config2 = get_config()
        
        assert config1 is config2  # Should be the same instance
        assert isinstance(config1, UnifiedConfig)
    
    def test_reload_config_global(self):
        """Test reload_config global function"""
        # Get initial config
        initial_config = get_config()
        
        # Reload should not raise an exception
        reload_config()
        
        # Should get a new instance
        new_config = get_config()
        assert isinstance(new_config, UnifiedConfig)


class TestConfigurationEdgeCases:
    """Test configuration edge cases and error handling"""
    
    def test_config_with_invalid_json(self):
        """Test handling of invalid JSON config file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json content")
            temp_path = f.name
        
        try:
            # Should handle invalid JSON gracefully
            with pytest.raises(Exception):  # Should raise some kind of exception
                UnifiedConfig(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_config_with_missing_file(self):
        """Test handling of missing config file"""
        non_existent_path = "/tmp/non_existent_config.json"
        
        # Should handle missing file gracefully (use defaults)
        config = UnifiedConfig(non_existent_path)
        assert isinstance(config, UnifiedConfig)
    
    def test_config_with_partial_data(self):
        """Test handling of partial config data"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "confidence_threshold": 0.95
                # Missing many other fields
            }
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = UnifiedConfig(temp_path)
            
            # Should use provided value and defaults for missing fields
            assert config.confidence_threshold == 0.95
            assert isinstance(config.max_context_size, int)
            assert isinstance(config.debug, bool)
            
        finally:
            os.unlink(temp_path)
    
    def test_config_with_invalid_types(self):
        """Test handling of invalid data types in config"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "confidence_threshold": "invalid_string",  # Should be float
                "max_context_size": "invalid_string",    # Should be int
                "debug": "invalid_string",              # Should be bool
                "providers": "invalid_string"           # Should be list
            }
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            # Should handle type validation gracefully
            # This might raise validation errors or use defaults
            config = UnifiedConfig(temp_path)
            assert isinstance(config, UnifiedConfig)
            
        finally:
            os.unlink(temp_path)
    
    def test_config_with_out_of_range_values(self):
        """Test handling of out-of-range values"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "confidence_threshold": 1.5,  # Should be <= 1.0
                "max_context_size": -100,     # Should be positive
                "batch_size": 0               # Should be positive
            }
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            # Should handle range validation gracefully
            config = UnifiedConfig(temp_path)
            assert isinstance(config, UnifiedConfig)
            
        finally:
            os.unlink(temp_path)


class TestProviderConfiguration:
    """Test provider-specific configuration"""
    
    def test_multiple_providers(self):
        """Test configuration with multiple providers"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "providers": [
                    {
                        "name": "provider1",
                        "url": "https://provider1.example.com",
                        "api_key": "key1",
                        "model": "model1"
                    },
                    {
                        "name": "provider2",
                        "url": "https://provider2.example.com",
                        "api_key": "key2",
                        "model": "model2"
                    },
                    {
                        "name": "provider3",
                        "url": "https://provider3.example.com",
                        "api_key": "key3",
                        "model": "model3"
                    }
                ]
            }
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = UnifiedConfig(temp_path)
            
            providers = config.get_providers()
            assert len(providers) == 3
            
            # Test primary provider is first one
            primary = config.get_primary_provider()
            assert primary["name"] == "provider1"
            
            # Test all providers have required fields
            for provider in providers:
                assert "name" in provider
                assert "url" in provider
                assert "model" in provider
                
        finally:
            os.unlink(temp_path)
    
    def test_provider_without_api_key(self):
        """Test provider configuration without API key (for local providers)"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "providers": [
                    {
                        "name": "local_provider",
                        "url": "http://localhost:11434",
                        "model": "llama2"
                        # No api_key field
                    }
                ]
            }
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = UnifiedConfig(temp_path)
            
            providers = config.get_providers()
            assert len(providers) == 1
            assert providers[0]["name"] == "local_provider"
            assert "api_key" not in providers[0] or providers[0]["api_key"] == ""
            
        finally:
            os.unlink(temp_path)
    
    def test_provider_with_optional_fields(self):
        """Test provider configuration with optional fields"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "providers": [
                    {
                        "name": "full_provider",
                        "url": "https://example.com",
                        "api_key": "secret_key",
                        "model": "gpt-4",
                        "timeout": 30,
                        "max_retries": 3,
                        "temperature": 0.7,
                        "max_tokens": 2000
                    }
                ]
            }
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = UnifiedConfig(temp_path)
            
            providers = config.get_providers()
            assert len(providers) == 1
            
            provider = providers[0]
            assert provider["name"] == "full_provider"
            assert provider["timeout"] == 30
            assert provider["max_retries"] == 3
            assert provider["temperature"] == 0.7
            assert provider["max_tokens"] == 2000
            
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])