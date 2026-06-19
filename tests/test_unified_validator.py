# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Tests for Unified Validator

Tests the UnifiedValidator class and related validation functions.
"""

import pytest
from src.config.unified_validator import (
    UnifiedValidator,
    UnifiedConfigValidator,
    ValidationResult,
    ConfigFormat,
    get_unified_validator,
    validate_config,
    get_primary_provider,
    show_configuration_status,
)


class TestValidationResult:
    """Tests for ValidationResult model."""

    def test_validation_result_defaults(self):
        """Test default validation result values."""
        result = ValidationResult()

        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_validation_result_with_errors(self):
        """Test validation result with errors."""
        result = ValidationResult(
            is_valid=False,
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"],
        )

        assert result.is_valid is False
        assert len(result.errors) == 2
        assert len(result.warnings) == 1


class TestUnifiedValidator:
    """Tests for UnifiedValidator class."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        validator = UnifiedValidator()

        assert validator.config == {}

    def test_init_with_config(self):
        """Test initialization with config."""
        config = {"key": "value"}
        validator = UnifiedValidator(config=config)

        assert validator.config == config

    def test_validate_config(self):
        """Test config validation."""
        validator = UnifiedValidator()

        result = validator.validate_config({"providers": []})

        assert isinstance(result, ValidationResult)
        assert result.is_valid is True

    def test_validate_provider(self):
        """Test provider validation."""
        validator = UnifiedValidator()

        result = validator.validate_provider({"name": "test"})

        assert isinstance(result, ValidationResult)
        assert result.is_valid is True

    def test_validate_all(self):
        """Test validate all."""
        validator = UnifiedValidator()

        result = validator.validate_all()

        assert isinstance(result, ValidationResult)
        assert result.is_valid is True


class TestConfigFormat:
    """Tests for ConfigFormat enum."""

    def test_config_format_values(self):
        """Test config format enum values."""
        assert ConfigFormat.JSON.value == "json"
        assert ConfigFormat.YAML.value == "yaml"
        assert ConfigFormat.ENV.value == "env"

    def test_config_format_members(self):
        """Test config format enum members."""
        members = list(ConfigFormat)

        assert len(members) == 3


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_get_unified_validator(self):
        """Test getting unified validator instance."""
        validator = get_unified_validator()

        assert isinstance(validator, UnifiedValidator)

    def test_get_unified_validator_with_path(self):
        """Test getting validator with config path."""
        validator = get_unified_validator(config_path="/some/path")

        assert isinstance(validator, UnifiedValidator)

    def test_validate_config_function(self):
        """Test validate_config function."""
        result = validate_config({"key": "value"})

        assert isinstance(result, ValidationResult)
        assert result.is_valid is True

    def test_get_primary_provider_exists(self):
        """Test getting primary provider when it exists."""
        config = {
            "providers": [
                {"name": "provider1"},
                {"name": "provider2"},
            ]
        }

        result = get_primary_provider(config)

        assert result == "provider1"

    def test_get_primary_provider_empty(self):
        """Test getting primary provider when none exist."""
        config = {"providers": []}

        result = get_primary_provider(config)

        assert result is None

    def test_get_primary_provider_no_providers(self):
        """Test getting primary provider when no providers key."""
        config = {}

        result = get_primary_provider(config)

        assert result is None

    def test_show_configuration_status_configured(self):
        """Test showing status for configured system."""
        config = {
            "providers": [
                {"name": "test_provider"},
            ]
        }

        status = show_configuration_status(config)

        assert status["configured"] is True
        assert status["provider_count"] == 1
        assert status["primary_provider"] == "test_provider"

    def test_show_configuration_status_unconfigured(self):
        """Test showing status for unconfigured system."""
        config = {}

        status = show_configuration_status(config)

        assert status["configured"] is False
        assert status["provider_count"] == 0
        assert status["primary_provider"] is None


class TestBackwardCompatibility:
    """Tests for backward compatibility aliases."""

    def test_unified_config_validator_alias(self):
        """Test UnifiedConfigValidator is alias for UnifiedValidator."""
        assert UnifiedConfigValidator is UnifiedValidator

    def test_alias_instance(self):
        """Test alias creates valid instance."""
        validator = UnifiedConfigValidator()

        assert isinstance(validator, UnifiedValidator)
