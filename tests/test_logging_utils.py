# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Tests for logging utilities (src/utils/logging.py)

Tests:
- get_logger configuration
- setup_logger customization
- setup_logging global config
- Custom exception classes
"""

import pytest
import logging
import sys

from src.utils.logging import (
    get_logger,
    setup_logger,
    setup_logging,
    ClaimParseError,
    ClaimValidationError,
    ClaimFormatError,
)


# ---------------------------------------------------------------------------
# get_logger Tests
# ---------------------------------------------------------------------------

class TestGetLogger:
    """Test get_logger function."""

    def test_returns_logger(self):
        """Test returns a Logger instance."""
        logger = get_logger("test")
        assert isinstance(logger, logging.Logger)

    def test_logger_has_name(self):
        """Test logger has correct name."""
        logger = get_logger("my_module")
        assert logger.name == "my_module"

    def test_logger_has_handler(self):
        """Test logger has at least one handler."""
        logger = get_logger("test_handler")
        assert len(logger.handlers) >= 1

    def test_logger_level_is_info(self):
        """Test logger level is INFO by default."""
        logger = get_logger("test_level")
        assert logger.level == logging.INFO

    def test_none_name_gets_root_logger(self):
        """Test None name returns root logger."""
        logger = get_logger(None)
        assert logger.name == "root"


class TestSetupLogger:
    """Test setup_logger function."""

    def test_returns_logger(self):
        """Test returns a Logger instance."""
        logger = setup_logger("custom")
        assert isinstance(logger, logging.Logger)

    def test_custom_level(self):
        """Test custom log level."""
        logger = setup_logger("debug_test", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_default_level_is_info(self):
        """Test default level is INFO."""
        logger = setup_logger("default_level_test")
        assert logger.level == logging.INFO

    def test_clears_existing_handlers(self):
        """Test existing handlers are cleared."""
        logger = setup_logger("handler_test")
        initial_count = len(logger.handlers)

        # Setup again - should still have same count (cleared and re-added)
        logger = setup_logger("handler_test")
        assert len(logger.handlers) == initial_count

    def test_custom_handlers(self):
        """Test custom handlers are added."""
        custom_handler = logging.StreamHandler(sys.stderr)
        logger = setup_logger("custom_handler_test", handlers=[custom_handler])

        assert custom_handler in logger.handlers


class TestSetupLogging:
    """Test setup_logging function."""

    def test_does_not_raise(self):
        """Test setup_logging doesn't raise exceptions."""
        # basicConfig is idempotent - just verify it doesn't error
        setup_logging(level="DEBUG")
        setup_logging(level="INFO")
        setup_logging(level="WARNING")

    def test_uppercase_level_accepted(self):
        """Test level string is case-insensitive (no error)."""
        # Just verify no exception - basicConfig may not override
        setup_logging(level="warning")
        setup_logging(level="WARNING")
        setup_logging(level="Warning")

    def test_custom_format_accepted(self):
        """Test custom format string is accepted."""
        # Just verify no exception
        setup_logging(level="INFO", format_string="%(message)s")


# ---------------------------------------------------------------------------
# Exception Classes Tests
# ---------------------------------------------------------------------------

class TestClaimParseError:
    """Test ClaimParseError exception."""

    def test_is_exception(self):
        """Test is an Exception subclass."""
        assert issubclass(ClaimParseError, Exception)

    def test_can_be_raised(self):
        """Test can be raised and caught."""
        with pytest.raises(ClaimParseError):
            raise ClaimParseError("Parse failed")

    def test_message_preserved(self):
        """Test error message is preserved."""
        try:
            raise ClaimParseError("Invalid JSON")
        except ClaimParseError as e:
            assert str(e) == "Invalid JSON"


class TestClaimValidationError:
    """Test ClaimValidationError exception."""

    def test_is_exception(self):
        """Test is an Exception subclass."""
        assert issubclass(ClaimValidationError, Exception)

    def test_can_be_raised(self):
        """Test can be raised and caught."""
        with pytest.raises(ClaimValidationError):
            raise ClaimValidationError("Validation failed")

    def test_message_preserved(self):
        """Test error message is preserved."""
        try:
            raise ClaimValidationError("Missing field")
        except ClaimValidationError as e:
            assert str(e) == "Missing field"


class TestClaimFormatError:
    """Test ClaimFormatError exception."""

    def test_is_exception(self):
        """Test is an Exception subclass."""
        assert issubclass(ClaimFormatError, Exception)

    def test_can_be_raised(self):
        """Test can be raised and caught."""
        with pytest.raises(ClaimFormatError):
            raise ClaimFormatError("Format error")

    def test_message_preserved(self):
        """Test error message is preserved."""
        try:
            raise ClaimFormatError("Bad format")
        except ClaimFormatError as e:
            assert str(e) == "Bad format"
