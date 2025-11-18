"""
Comprehensive Test Suite for Unified Configuration Validator
Tests all adapters, validation logic, migration utilities, and backward compatibility
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the modules we're testing
from src.config.unified_validator import (
    UnifiedConfigValidator, 
    ConfigFormat,
    get_unified_validator,
    validate_config,
    get_primary_provider,
    show_configuration_status
)
from src.config.adapters import (
    BaseAdapter,
    ProviderConfig,
    SimpleProviderAdapter,
    IndividualEnvAdapter,
    UnifiedProviderAdapter,
    SimpleValidatorAdapter
)
from src.config.migration_utils import ConfigMigrator, analyze_migration, execute_migration

class TestBaseAdapter:
    """Test the base adapter functionality"""

    def test_provider_config_post_processing(self):
        """Test ProviderConfig post-processing"""
        # Test with model but no models list
        config1 = ProviderConfig(
            name="Test", base_url="http://test", api_key="key", 
            model="test-model"
        )
        assert config1.models == ["test-model"]
        
        # Test with models but no model
        config2 = ProviderConfig(
            name="Test", base_url="http://test", api_key="key",
            models=["model1", "model2"]
        )
        assert config2.model == "model1"
        
        # Test with both
        config3 = ProviderConfig(
            name="Test", base_url="http://test", api_key="key",
            model="model1", models=["model1", "model2"]
        )
        assert config3.model == "model1"
        assert config3.models == ["model1", "model2"]

    def test_normalize_url(self):
        """Test URL normalization"""
        adapter = SimpleProviderAdapter()  # Use concrete adapter for testing
        
        # Test http URLs
        assert adapter.normalize_url("http://example.com") == "http://example.com"
        assert adapter.normalize_url("example.com") == "https://example.com"
        
        # Test localhost URLs
        assert adapter.normalize_url("localhost:1234") == "http://localhost:1234"
        assert adapter.normalize_url("127.0.0.1:8080") == "http://127.0.0.1:8080"
        
        # Test HTTPS URLs
        assert adapter.normalize_url("https://example.com") == "https://example.com"

    def test_detect_provider_type(self):
        """Test provider type detection"""
        adapter = SimpleProviderAdapter()
        
        # Test known providers
        ollama_info = adapter.detect_provider_type("http://localhost:11434")
        assert ollama_info["name"] == "Ollama"
        assert ollama_info["is_local"] == True
        
        openai_info = adapter.detect_provider_type("https://api.openai.com/v1")
        assert openai_info["name"] == "OpenAI"
        assert openai_info["is_local"] == False
        
        # Test unknown provider
        unknown_info = adapter.detect_provider_type("https://unknown.example.com")
        assert unknown_info["name"] == "Unknown"

class TestSimpleProviderAdapter:
    """Test the Simple Provider Adapter"""

    def test_detect_format(self):
        adapter = SimpleProviderAdapter()
        
        # Test with PROVIDER_ variables
        env_vars = {"PROVIDER_OLLAMA": "http://localhost:11434,,llama2,ollama"}
        assert adapter.detect_format(env_vars) == True
        
        # Test without PROVIDER_ variables
        env_vars = {"OTHER_VAR": "value"}
        assert adapter.detect_format(env_vars) == False

    def test_validate_format(self):
        adapter = SimpleProviderAdapter()
        
        # Test valid configuration
        env_vars = {"PROVIDER_OLLAMA": "http://localhost:11434,,llama2,ollama"}
        is_valid, errors = adapter.validate_format(env_vars)
        assert is_valid == True
        assert len(errors) == 0
        
        # Test invalid configuration (missing parts)
        env_vars = {"PROVIDER_OLLAMA": "http://localhost:11434,llama2"}  # Missing API_KEY and PROTOCOL
        is_valid, errors = adapter.validate_format(env_vars)
        assert is_valid == False
        assert len(errors) > 0
        
        # Test invalid URL
        env_vars = {"PROVIDER_OLLAMA": "invalid-url,,llama2,ollama"}
        is_valid, errors = adapter.validate_format(env_vars)
        assert is_valid == False
        assert any("must start with http://" in error for error in errors)

    def test_load_providers(self):
        adapter = SimpleProviderAdapter()
        
        # Test with valid provider
        env_vars = {"PROVIDER_OLLAMA": "http://localhost:11434,,llama2,ollama"}
        success, providers, errors = adapter.load_providers(env_vars)
        assert success == True
        assert len(providers) == 1
        assert providers[0].name == "Ollama"
        assert providers[0].base_url == "http://localhost:11434"
        assert providers[0].model == "llama2"
        assert providers[0].protocol == "ollama"
        
        # Test with no providers
        env_vars = {}
        success, providers, errors = adapter.load_providers(env_vars)
        assert success == False
        assert len(providers) == 0

    def test_migrate_to_unified_format(self):
        adapter = SimpleProviderAdapter()
        
        # Test migration
        env_vars = {"PROVIDER_OLLAMA": "http://localhost:11434,,llama2,ollama"}
        migration = adapter.migrate_to_unified_format(env_vars)
        
        assert "success" in migration
        assert len(migration["success"]) > 0
        assert any("PROVIDER_API_URL=http://localhost:11434" in line for line in migration["success"])
        assert any("PROVIDER_MODEL=llama2" in line for line in migration["success"])

class TestIndividualEnvAdapter:
    """Test the Individual Environment Adapter"""

    def test_detect_format(self):
        adapter = IndividualEnvAdapter()
        
        # Test with API_URL variables
        env_vars = {"OLLAMA_API_URL": "http://localhost:11434"}
        assert adapter.detect_format(env_vars) == True
        
        # Test without API_URL variables
        env_vars = {"OTHER_VAR": "value"}
        assert adapter.detect_format(env_vars) == False

    def test_parse_models(self):
        adapter = IndividualEnvAdapter()
        
        # Test JSON array
        models = adapter._parse_models('["model1", "model2"]', "test")
        assert models == ["model1", "model2"]
        
        # Test manually quoted array
        models = adapter._parse_models('["model1","model2"]', "test")
        assert models == ["model1", "model2"]
        
        # Test comma-separated
        models = adapter._parse_models("model1,model2", "test")
        assert models == ["model1", "model2"]
        
        # Test single model
        models = adapter._parse_models("model1", "test")
        assert models == ["model1"]
        
        # Test empty
        models = adapter._parse_models("", "test")
        assert models == []

    def test_load_providers(self):
        adapter = IndividualEnvAdapter()
        
        # Test with valid configuration
        env_vars = {
            "OLLAMA_API_URL": "http://localhost:11434",
            "OLLAMA_API_KEY": "",
            "OLLAMA_MODELS": '["llama2", "mistral"]'
        }
        success, providers, errors = adapter.load_providers(env_vars)
        assert success == True
        assert len(providers) == 1
        assert providers[0].name == "Ollama"
        assert providers[0].models == ["llama2", "mistral"]

class TestUnifiedProviderAdapter:
    """Test the Unified Provider Adapter"""

    def test_detect_format(self):
        adapter = UnifiedProviderAdapter()
        
        # Test with PROVIDER_API_URL
        env_vars = {"PROVIDER_API_URL": "http://localhost:11434"}
        assert adapter.detect_format(env_vars) == True
        
        # Test without PROVIDER_API_URL
        env_vars = {"OTHER_VAR": "value"}
        assert adapter.detect_format(env_vars) == False

    def test_validate_format(self):
        adapter = UnifiedProviderAdapter()
        
        # Test valid local configuration
        env_vars = {
            "PROVIDER_API_URL": "http://localhost:11434",
            "PROVIDER_API_KEY": "",
            "PROVIDER_MODEL": "llama2"
        }
        is_valid, errors = adapter.validate_format(env_vars)
        assert is_valid == True
        
        # Test valid cloud configuration
        env_vars = {
            "PROVIDER_API_URL": "https://api.openai.com/v1",
            "PROVIDER_API_KEY": "sk-test-key",
            "PROVIDER_MODEL": "gpt-3.5-turbo"
        }
        is_valid, errors = adapter.validate_format(env_vars)
        assert is_valid == True
        
        # Test missing API key for cloud service
        env_vars = {
            "PROVIDER_API_URL": "https://api.openai.com/v1",
            "PROVIDER_API_KEY": "",
            "PROVIDER_MODEL": "gpt-3.5-turbo"
        }
        is_valid, errors = adapter.validate_format(env_vars)
        assert is_valid == False
        assert any("API key is required" in error for error in errors)

    def test_load_providers(self):
        adapter = UnifiedProviderAdapter()
        
        # Test with valid configuration
        env_vars = {
            "PROVIDER_API_URL": "http://localhost:11434",
            "PROVIDER_API_KEY": "",
            "PROVIDER_MODEL": "llama2"
        }
        success, providers, errors = adapter.load_providers(env_vars)
        assert success == True
        assert len(providers) == 1
        assert providers[0].name == "Ollama"
        assert providers[0].base_url == "http://localhost:11434"
        assert providers[0].model == "llama2"

class TestSimpleValidatorAdapter:
    """Test the Simple Validator Adapter"""

    def test_detect_format(self):
        adapter = SimpleValidatorAdapter()
        
        # Test with OLLAMA_ENDPOINT
        env_vars = {"OLLAMA_ENDPOINT": "http://localhost:11434"}
        assert adapter.detect_format(env_vars) == True
        
        # Test with OPENAI_API_KEY
        env_vars = {"OPENAI_API_KEY": "sk-test-key"}
        assert adapter.detect_format(env_vars) == True
        
        # Test without known variables
        env_vars = {"OTHER_VAR": "value"}
        assert adapter.detect_format(env_vars) == False

    def test_load_providers(self):
        adapter = SimpleValidatorAdapter()
        
        # Test with Ollama configuration
        env_vars = {
            "OLLAMA_ENDPOINT": "http://localhost:11434",
            "OLLAMA_MODEL": "llama2"
        }
        success, providers, errors = adapter.load_providers(env_vars)
        assert success == True
        assert len(providers) == 1
        assert providers[0].name == "Ollama"
        
        # Test with OpenAI configuration
        env_vars = {
            "OPENAI_API_KEY": "sk-test-key",
            "OPENAI_MODEL": "gpt-3.5-turbo"
        }
        success, providers, errors = adapter.load_providers(env_vars)
        assert success == True
        assert len(providers) == 1
        assert providers[0].name == "OpenAI"

class TestUnifiedConfigValidator:
    """Test the main UnifiedConfigValidator"""

    @pytest.fixture
    def temp_env_file(self):
        """Create a temporary .env file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("# Test configuration\n")
            f.write("TEST_VAR=test_value\n")
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_initialization(self, temp_env_file):
        validator = UnifiedConfigValidator(temp_env_file)
        assert validator.env_file == Path(temp_env_file)
        assert isinstance(validator.env_vars, dict)
        assert len(validator.adapters) == 4

    def test_detect_formats(self, temp_env_file):
        validator = UnifiedConfigValidator(temp_env_file)
        
        # Test with no format
        detected_formats = validator.detect_formats()
        assert len(detected_formats) == 0
        
        # Test with unified provider format
        validator.env_vars["PROVIDER_API_URL"] = "http://localhost:11434"
        detected_formats = validator.detect_formats()
        assert ConfigFormat.UNIFIED_PROVIDER in detected_formats

    def test_get_active_format(self, temp_env_file):
        validator = UnifiedConfigValidator(temp_env_file)
        
        # Test with no formats
        assert validator.get_active_format() == ConfigFormat.UNKNOWN
        
        # Test with unified provider format
        validator.env_vars["PROVIDER_API_URL"] = "http://localhost:11434"
        assert validator.get_active_format() == ConfigFormat.UNIFIED_PROVIDER
        
        # Test priority (unified should win)
        validator.env_vars["PROVIDER_OLLAMA"] = "http://localhost:11434,,llama2,ollama"
        assert validator.get_active_format() == ConfigFormat.UNIFIED_PROVIDER

    def test_validate_configuration(self, temp_env_file):
        validator = UnifiedConfigValidator(temp_env_file)
        
        # Test with no configuration
        result = validator.validate_configuration()
        assert result.success == False
        assert len(result.errors) > 0
        
        # Test with valid unified configuration
        validator.env_vars.update({
            "PROVIDER_API_URL": "http://localhost:11434",
            "PROVIDER_API_KEY": "",
            "PROVIDER_MODEL": "llama2"
        })
        result = validator.validate_configuration()
        assert result.success == True
        assert result.active_format == ConfigFormat.UNIFIED_PROVIDER

    def test_get_primary_provider(self, temp_env_file):
        validator = UnifiedConfigValidator(temp_env_file)
        
        # Test with no configuration
        primary = validator.get_primary_provider()
        assert primary is None
        
        # Test with valid configuration
        validator.env_vars.update({
            "PROVIDER_API_URL": "http://localhost:11434",
            "PROVIDER_API_KEY": "",
            "PROVIDER_MODEL": "llama2"
        })
        primary = validator.get_primary_provider()
        assert primary is not None
        assert primary.name == "Ollama"

    def test_cache_functionality(self, temp_env_file):
        validator = UnifiedConfigValidator(temp_env_file)
        
        # Test that validation is cached
        validator.env_vars.update({
            "PROVIDER_API_URL": "http://localhost:11434",
            "PROVIDER_API_KEY": "",
            "PROVIDER_MODEL": "llama2"
        })
        
        result1 = validator.validate_configuration()
        result2 = validator.validate_configuration()
        assert result1 is result2  # Should be the same object (cached)
        
        # Test cache invalidation
        validator.clear_cache()
        result3 = validator.validate_configuration()
        assert result1 is not result3  # Should be different object

    def test_export_configuration(self, temp_env_file):
        validator = UnifiedConfigValidator(temp_env_file)
        
        # Set up configuration
        validator.env_vars.update({
            "PROVIDER_API_URL": "http://localhost:11434",
            "PROVIDER_API_KEY": "test-key",
            "PROVIDER_MODEL": "llama2"
        })
        
        # Test export to unified format
        exported = validator.export_configuration(ConfigFormat.UNIFIED_PROVIDER)
        assert "PROVIDER_API_URL" in exported
        assert exported["PROVIDER_API_URL"] == "http://localhost:11434"

class TestConfigMigrator:
    """Test the configuration migrator utilities"""

    @pytest.fixture
    def temp_env_file(self):
        """Create a temporary .env file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("PROVIDER_OLLAMA=http://localhost:11434,,llama2,ollama\n")
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_create_backup(self, temp_env_file):
        migrator = ConfigMigrator(temp_env_file)
        
        success, backup_path = migrator.create_backup()
        assert success == True
        assert Path(backup_path).exists()
        
        # Cleanup
        if os.path.exists(backup_path):
            os.unlink(backup_path)

    def test_analyze_migration(self, temp_env_file):
        migrator = ConfigMigrator(temp_env_file)
        
        analysis = migrator.analyze_migration()
        
        assert "current_format" in analysis
        assert "detected_formats" in analysis
        assert "complexity_score" in analysis
        assert "migration_difficulty" in analysis
        assert "benefits" in analysis
        assert len(analysis["benefits"]) > 0

    def test_execute_migration_dry_run(self, temp_env_file):
        migrator = ConfigMigrator(temp_env_file)
        
        result = migrator.execute_migration(dry_run=True)
        
        assert result["success"] == True
        assert result["changes_applied"] == False
        assert "DRY RUN MODE" in str(result["warnings"])

    def test_generate_migration_script(self, temp_env_file):
        migrator = ConfigMigrator(temp_env_file)
        
        script_data = migrator.generate_migration_script()
        
        assert script_data["success"] == True
        assert "script" in script_data
        assert len(script_data["script"]) > 0
        assert any("PROVIDER_API_URL" in line for line in script_data["script"])

    def test_export_migration_analysis(self, temp_env_file):
        migrator = ConfigMigrator(temp_env_file)
        
        # Test JSON export
        json_export = migrator.export_migration_analysis("json")
        assert "current_format" in json_export
        
        # Test markdown export
        md_export = migrator.export_migration_analysis("markdown")
        assert "# Configuration Migration Guide" in md_export

class TestGlobalFunctions:
    """Test global convenience functions"""

    @patch('src.config.unified_validator._global_validator', None)
    def test_get_unified_validator(self):
        # Test that global validator is created
        validator1 = get_unified_validator()
        validator2 = get_unified_validator()
        assert validator1 is validator2  # Should be the same instance

    @patch('src.config.unified_validator._global_validator', None)
    def test_validate_config(self):
        # Test global validate_config function
        with patch.dict(os.environ, {
            'PROVIDER_API_URL': 'http://localhost:11434',
            'PROVIDER_API_KEY': '',
            'PROVIDER_MODEL': 'llama2'
        }):
            result = validate_config()
            assert result is not None

class TestIntegration:
    """Integration tests for the complete system"""

    @pytest.fixture
    def complete_test_env(self):
        """Create a comprehensive test environment"""
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            
            # Write complex configuration with multiple formats
            env_content = """
# Simple provider format (high priority)
PROVIDER_OLLAMA=http://localhost:11434,,llama2,ollama
PROVIDER_OPENROUTER=https://openrouter.ai/api/v1,sk-test-key,openai/gpt-3.5-turbo,openai

# Individual env format (medium priority)  
OLLAMA_API_URL=http://localhost:11435
OLLAMA_API_KEY=
OLLAMA_MODELS=["llama2", "mistral"]

# Unified format (highest priority)
PROVIDER_API_URL=http://localhost:11436
PROVIDER_API_KEY=
PROVIDER_MODEL=llama3

# Simple validator format (lowest priority)
OPENAI_API_KEY=sk-old-openai-key
OPENAI_MODEL=gpt-4
"""
            env_file.write_text(env_content.strip())
            
            yield str(env_file)
    
    def test_complete_validation(self, complete_test_env):
        """Test validation with complex multi-format environment"""
        validator = UnifiedConfigValidator(complete_test_env)
        
        # Should detect all formats
        detected_formats = validator.detect_formats()
        assert len(detected_formats) >= 3
        
        # Should choose unified format as active
        active_format = validator.get_active_format()
        assert active_format == ConfigFormat.UNIFIED_PROVIDER
        
        # Should validate successfully
        result = validator.validate_configuration()
        assert result.success == True
        
        # Should identify format conflicts
        assert len(result.format_conflicts) > 0
        
        # Should have migration suggestions
        assert len(result.migration_suggestions) > 0

    def test_migration_workflow(self, complete_test_env):
        """Test complete migration workflow"""
        migrator = ConfigMigrator(complete_test_env)
        
        # Analyze migration
        analysis = migrator.analyze_migration()
        assert analysis["current_format"] == ConfigFormat.UNIFIED_PROVIDER
        
        # Since already unified, test migrating from a different format
        # Create test env with simple provider format
        simple_env = Path(complete_test_env).parent / "simple.env"
        simple_env.write_text("PROVIDER_OLLAMA=http://localhost:11434,,llama2,ollama")
        
        simple_migrator = ConfigMigrator(str(simple_env))
        analysis = simple_migrator.analyze_migration()
        assert analysis["current_format"] == ConfigFormat.SIMPLE_PROVIDER
        
        # Test dry run migration
        result = simple_migrator.execute_migration(dry_run=True)
        assert result["success"] == True
        assert len(result["changes"]) > 0

@pytest.mark.parametrize("format_type,env_vars", [
    (ConfigFormat.UNIFIED_PROVIDER, {
        "PROVIDER_API_URL": "http://localhost:11434",
        "PROVIDER_API_KEY": "",
        "PROVIDER_MODEL": "llama2"
    }),
    (ConfigFormat.SIMPLE_PROVIDER, {
        "PROVIDER_OLLAMA": "http://localhost:11434,,llama2,ollama"
    }),
    (ConfigFormat.INDIVIDUAL_ENV, {
        "OLLAMA_API_URL": "http://localhost:11434",
        "OLLAMA_API_KEY": "",
        "OLLAMA_MODELS": '["llama2"]'
    }),
    (ConfigFormat.SIMPLE_VALIDATOR, {
        "OLLAMA_ENDPOINT": "http://localhost:11434",
        "OLLAMA_MODEL": "llama2"
    })
])
def test_all_formats(format_type, env_vars):
    """Parameterized test for all supported formats"""
    with patch.dict(os.environ, env_vars):
        validator = UnifiedConfigValidator()
        
        # Should detect the format
        detected_formats = validator.detect_formats()
        assert format_type in detected_formats
        
        # Should validate successfully
        result = validator.validate_configuration()
        if format_type == validator.get_active_format():
            assert result.success == True

def test_error_handling():
    """Test error handling and edge cases"""
    validator = UnifiedConfigValidator("nonexistent.env")
    
    # Should handle missing file gracefully
    result = validator.validate_configuration()
    assert isinstance(result, type(validator.validate_configuration()))
    
    # Test with invalid URLs
    validator.env_vars = {"PROVIDER_API_URL": "invalid-url"}
    result = validator.validate_configuration()
    assert result.success == False
    assert any("must start with http://" in error for error in result.errors)

if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v", "--tb=short"])