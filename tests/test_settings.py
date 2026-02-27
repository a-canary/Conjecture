"""
Tests for Settings Module

Tests the Settings class and constants.
"""

import pytest
from src.config.settings import (
    Settings,
    DIRTY_FLAG_CONFIDENCE_THRESHOLD,
    DIRTY_FLAG_ENABLED,
    DIRTY_FLAG_AUTO_CLEAN,
    DIRTY_FLAG_CASCADE_DEPTH,
    DIRTY_FLAG_BATCH_SIZE,
)


class TestDirtyFlagConstants:
    """Tests for dirty flag configuration constants."""

    def test_confidence_threshold(self):
        """Test confidence threshold constant."""
        assert DIRTY_FLAG_CONFIDENCE_THRESHOLD == 0.8

    def test_enabled_default(self):
        """Test enabled constant."""
        assert DIRTY_FLAG_ENABLED is True

    def test_auto_clean_default(self):
        """Test auto clean constant."""
        assert DIRTY_FLAG_AUTO_CLEAN is False

    def test_cascade_depth(self):
        """Test cascade depth constant."""
        assert DIRTY_FLAG_CASCADE_DEPTH == 3

    def test_batch_size(self):
        """Test batch size constant."""
        assert DIRTY_FLAG_BATCH_SIZE == 100


class TestSettings:
    """Tests for Settings class."""

    def test_settings_defaults(self):
        """Test default settings values."""
        settings = Settings()

        assert settings.debug is False
        assert settings.log_level == "INFO"
        assert settings.max_workers == 4
        assert settings.timeout == 30
        assert settings.retry_attempts == 3

    def test_settings_custom_values(self):
        """Test settings with custom values."""
        settings = Settings(
            debug=True,
            log_level="DEBUG",
            max_workers=8,
        )

        assert settings.debug is True
        assert settings.log_level == "DEBUG"
        assert settings.max_workers == 8

    def test_settings_from_dict(self):
        """Test creating settings from dictionary."""
        settings_dict = {
            "debug": True,
            "log_level": "WARNING",
            "timeout": 60,
        }

        settings = Settings.from_dict(settings_dict)

        assert settings.debug is True
        assert settings.log_level == "WARNING"
        assert settings.timeout == 60

    def test_settings_to_dict(self):
        """Test converting settings to dictionary."""
        settings = Settings(debug=True, log_level="ERROR")

        result = settings.to_dict()

        assert isinstance(result, dict)
        assert result["debug"] is True
        assert result["log_level"] == "ERROR"
