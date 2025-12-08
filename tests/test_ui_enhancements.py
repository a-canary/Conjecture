"""
Comprehensive tests for UI enhancements
Tests all new UI components and improvements
"""

import pytest
import asyncio
import time
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List
import sys
import os
from io import StringIO

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.cli.ui_enhancements import (
    UIConfig, OutputMode, VerbosityLevel,
    EnhancedProgressTracker, EnhancedOutputFormatter,
    InteractivePrompter, ContextualHelp, UIEnhancer,
    create_ui_enhancer
)


class TestUIConfig:
    """Test UI configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = UIConfig()
        assert config.output_mode == OutputMode.NORMAL
        assert config.verbosity == VerbosityLevel.INFO
        assert config.show_progress is True
        assert config.show_timings is True
        assert config.color_enabled is True
        assert config.animated is True
        assert config.accessibility_mode is False
        assert config.compact_mode is False
        assert config.show_tips is True
        assert config.auto_save is True

    def test_custom_config(self):
        """Test custom configuration"""
        config = UIConfig(
            output_mode=OutputMode.JSON,
            verbosity=VerbosityLevel.DEBUG,
            show_progress=False,
            accessibility_mode=True
        )
        assert config.output_mode == OutputMode.JSON
        assert config.verbosity == VerbosityLevel.DEBUG
        assert config.show_progress is False
        assert config.accessibility_mode is True


class TestEnhancedProgressTracker:
    """Test enhanced progress tracking"""

    def test_progress_tracker_init(self):
        """Test progress tracker initialization"""
        config = UIConfig(show_progress=True)
        tracker = EnhancedProgressTracker(config)
        assert tracker.config == config
        assert tracker._tasks == {}
        assert tracker._subtasks == {}
        assert tracker._active is False

    def test_start_task(self):
        """Test starting a task"""
        config = UIConfig(show_progress=False)  # Disable Rich for testing
        tracker = EnhancedProgressTracker(config)

        task_id = tracker.start_task("Test task", total=100)
        assert task_id in tracker._tasks
        assert tracker._tasks[task_id]["description"] == "Test task"
        assert tracker._tasks[task_id]["total"] == 100
        assert tracker._tasks[task_id]["completed"] == 0
        assert task_id in tracker._start_times

    def test_update_task(self):
        """Test updating task progress"""
        config = UIConfig(show_progress=False)
        tracker = EnhancedProgressTracker(config)

        task_id = tracker.start_task("Test task", total=100)
        tracker.update_task(task_id, advance=25, description="Updated task")

        assert tracker._tasks[task_id]["completed"] == 25
        assert tracker._tasks[task_id]["description"] == "Updated task"

    def test_complete_task(self):
        """Test completing a task"""
        config = UIConfig(show_progress=False)
        tracker = EnhancedProgressTracker(config)

        task_id = tracker.start_task("Test task", total=100)
        tracker.complete_task(task_id, "Task completed")

        assert tracker._tasks[task_id]["completed"] == 100
        assert tracker._tasks[task_id]["status"] == "completed"

    def test_fail_task(self):
        """Test failing a task"""
        config = UIConfig(show_progress=False)
        tracker = EnhancedProgressTracker(config)

        task_id = tracker.start_task("Test task", total=100)
        tracker.fail_task(task_id, "Task failed")

        assert tracker._tasks[task_id]["status"] == "failed"


class TestEnhancedOutputFormatter:
    """Test enhanced output formatting"""

    def test_formatter_init(self):
        """Test formatter initialization"""
        config = UIConfig(output_mode=OutputMode.NORMAL)
        formatter = EnhancedOutputFormatter(config)
        assert formatter.config == config

    def test_format_success_normal(self):
        """Test success message formatting in normal mode"""
        config = UIConfig(output_mode=OutputMode.NORMAL)
        formatter = EnhancedOutputFormatter(config)

        result = formatter.format_success("Operation successful", "Details here")
        assert "Operation successful" in str(result)
        assert "Details here" in str(result)

    def test_format_success_json(self):
        """Test success message formatting in JSON mode"""
        config = UIConfig(output_mode=OutputMode.JSON)
        formatter = EnhancedOutputFormatter(config)

        result = formatter.format_success("Operation successful", "Details here")
        parsed = json.loads(result)
        assert parsed["status"] == "success"
        assert parsed["message"] == "Operation successful"
        assert parsed["details"] == "Details here"

    def test_format_error_with_suggestion(self):
        """Test error message formatting with suggestion"""
        config = UIConfig(output_mode=OutputMode.NORMAL)
        formatter = EnhancedOutputFormatter(config)

        result = formatter.format_error(
            "Something went wrong",
            "Error details",
            "Try again later"
        )
        result_str = str(result)
        assert "Something went wrong" in result_str
        assert "Error details" in result_str
        assert "Suggestion" in result_str
        assert "Try again later" in result_str

    def test_format_claim_table_normal(self):
        """Test claim table formatting in normal mode"""
        config = UIConfig(output_mode=OutputMode.NORMAL)
        formatter = EnhancedOutputFormatter(config)

        claims = [
            {
                "id": "c001",
                "content": "Test claim content",
                "confidence": 0.85,
                "tags": ["test", "claim"],
                "state": "validated"
            }
        ]

        result = formatter.format_claim_table(claims, "test query")
        # Should return a Rich Table object or formatted string
        assert result is not None

    def test_format_claim_table_json(self):
        """Test claim table formatting in JSON mode"""
        config = UIConfig(output_mode=OutputMode.JSON)
        formatter = EnhancedOutputFormatter(config)

        claims = [
            {
                "id": "c001",
                "content": "Test claim content",
                "confidence": 0.85,
                "tags": ["test", "claim"],
                "state": "validated"
            }
        ]

        result = formatter.format_claim_table(claims, "test query")
        parsed = json.loads(result)
        assert parsed["claims"] == claims
        assert parsed["query"] == "test query"
        assert parsed["count"] == 1

    def test_format_claim_details(self):
        """Test claim details formatting"""
        config = UIConfig(output_mode=OutputMode.NORMAL)
        formatter = EnhancedOutputFormatter(config)

        claim = {
            "id": "c001",
            "content": "Test claim content",
            "confidence": 0.85,
            "tags": ["test", "claim"],
            "state": "validated",
            "created_at": "2023-01-01T00:00:00Z",
            "created_by": "test_user"
        }

        result = formatter.format_claim_details(claim)
        assert result is not None

    def test_format_warning(self):
        """Test warning message formatting"""
        config = UIConfig(output_mode=OutputMode.NORMAL)
        formatter = EnhancedOutputFormatter(config)

        result = formatter.format_warning("This is a warning", "Warning details")
        result_str = str(result)
        assert "This is a warning" in result_str
        assert "Warning details" in result_str

    def test_format_info(self):
        """Test info message formatting"""
        config = UIConfig(output_mode=OutputMode.NORMAL)
        formatter = EnhancedOutputFormatter(config)

        result = formatter.format_info("This is info", "Info details")
        result_str = str(result)
        assert "This is info" in result_str
        assert "Info details" in result_str


class TestInteractivePrompter:
    """Test interactive prompting"""

    def test_prompter_init(self):
        """Test prompter initialization"""
        config = UIConfig()
        prompter = InteractivePrompter(config)
        assert prompter.config == config

    def test_ask_yes_no_json_mode(self):
        """Test yes/no prompt in JSON mode returns default"""
        config = UIConfig(output_mode=OutputMode.JSON)
        prompter = InteractivePrompter(config)

        result = prompter.ask_yes_no("Continue?", default=True)
        assert result is True  # Should return default in JSON mode

    def test_ask_text_json_mode(self):
        """Test text prompt in JSON mode returns default"""
        config = UIConfig(output_mode=OutputMode.JSON)
        prompter = InteractivePrompter(config)

        result = prompter.ask_text("Enter name", default="default_name")
        assert result == "default_name"  # Should return default in JSON mode

    def test_ask_choice_json_mode(self):
        """Test choice prompt in JSON mode returns first option"""
        config = UIConfig(output_mode=OutputMode.JSON)
        prompter = InteractivePrompter(config)

        choices = ["option1", "option2", "option3"]
        result = prompter.ask_choice("Choose option", choices, default="option2")
        assert result == "option2"  # Should return default in JSON mode

    def test_ask_int_json_mode(self):
        """Test integer prompt in JSON mode returns default"""
        config = UIConfig(output_mode=OutputMode.JSON)
        prompter = InteractivePrompter(config)

        result = prompter.ask_int("Enter number", default=42)
        assert result == 42  # Should return default in JSON mode

    def test_ask_float_json_mode(self):
        """Test float prompt in JSON mode returns default"""
        config = UIConfig(output_mode=OutputMode.JSON)
        prompter = InteractivePrompter(config)

        result = prompter.ask_float("Enter value", default=3.14)
        assert result == 3.14  # Should return default in JSON mode


class TestContextualHelp:
    """Test contextual help system"""

    def test_help_init(self):
        """Test help system initialization"""
        config = UIConfig()
        help_system = ContextualHelp(config)
        assert help_system.config == config
        assert "create" in help_system.tips
        assert "search" in help_system.tips

    def test_get_random_tip(self):
        """Test getting random tips"""
        config = UIConfig(show_tips=True)
        help_system = ContextualHelp(config)

        tip = help_system.get_random_tip("create")
        assert tip is not None
        assert "Tip:" in tip

    def test_no_tips_in_quiet_mode(self):
        """Test that tips are disabled in quiet mode"""
        config = UIConfig(output_mode=OutputMode.QUIET, show_tips=True)
        help_system = ContextualHelp(config)

        tip = help_system.get_random_tip("create")
        assert tip is None

    def test_no_tips_when_disabled(self):
        """Test that tips are disabled when show_tips is False"""
        config = UIConfig(show_tips=False)
        help_system = ContextualHelp(config)

        tip = help_system.get_random_tip("create")
        assert tip is None

    def test_show_help_for_command(self):
        """Test showing help for a command"""
        config = UIConfig(show_tips=True)
        help_system = ContextualHelp(config)

        formatter = Mock()
        formatter.print_info = Mock()

        help_system.show_help_for_command("create", formatter)
        formatter.print_info.assert_called_once()


class TestUIEnhancer:
    """Test main UI enhancer"""

    def test_enhancer_init(self):
        """Test UI enhancer initialization"""
        config = UIConfig()
        enhancer = UIEnhancer(config)

        assert enhancer.config == config
        assert enhancer.progress is not None
        assert enhancer.formatter is not None
        assert enhancer.prompter is not None
        assert enhancer.help is not None

    def test_start_operation(self):
        """Test starting an operation"""
        config = UIConfig(show_progress=False)
        enhancer = UIEnhancer(config)

        task_id = enhancer.start_operation("Test operation")
        assert task_id is not None
        assert enhancer.operation_start_time is not None

    def test_complete_operation(self):
        """Test completing an operation"""
        config = UIConfig(show_progress=False, show_timings=False)
        enhancer = UIEnhancer(config)

        task_id = enhancer.start_operation("Test operation")
        enhancer.complete_operation(task_id, "Operation completed")

        # Should not raise any exceptions
        assert True

    def test_fail_operation(self):
        """Test failing an operation"""
        config = UIConfig(show_progress=False)
        enhancer = UIEnhancer(config)

        task_id = enhancer.start_operation("Test operation")
        enhancer.fail_operation(task_id, "Operation failed", "Try again")

        # Should not raise any exceptions
        assert True

    def test_update_progress(self):
        """Test updating progress"""
        config = UIConfig(show_progress=False)
        enhancer = UIEnhancer(config)

        task_id = enhancer.start_operation("Test operation")
        enhancer.update_progress(task_id, "Updated progress", 10)

        # Should not raise any exceptions
        assert True

    def test_show_command_help(self):
        """Test showing command help"""
        config = UIConfig(show_tips=True)
        enhancer = UIEnhancer(config)

        enhancer.show_command_help("create")

        # Should not raise any exceptions
        assert True

    def test_cleanup(self):
        """Test cleanup"""
        config = UIConfig()
        enhancer = UIEnhancer(config)
        enhancer.cleanup()

        # Should not raise any exceptions
        assert True


class TestCreateUIEnhancer:
    """Test UI enhancer creation helper"""

    def test_create_ui_enhancer_defaults(self):
        """Test creating UI enhancer with defaults"""
        enhancer = create_ui_enhancer()

        assert enhancer.config.output_mode == OutputMode.NORMAL
        assert enhancer.config.verbosity == VerbosityLevel.INFO
        assert enhancer.config.show_progress is True

    def test_create_ui_enhancer_custom(self):
        """Test creating UI enhancer with custom settings"""
        enhancer = create_ui_enhancer(
            output_mode="json",
            verbosity="debug",
            show_progress=False
        )

        assert enhancer.config.output_mode == OutputMode.JSON
        assert enhancer.config.verbosity == VerbosityLevel.DEBUG
        assert enhancer.config.show_progress is False


class TestUIEnhancementsIntegration:
    """Integration tests for UI enhancements"""

    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """Test a complete workflow with all UI components"""
        config = UIConfig(
            output_mode=OutputMode.NORMAL,
            show_progress=False,
            show_timings=False
        )
        enhancer = UIEnhancer(config)

        try:
            # Start operation
            task_id = enhancer.start_operation("Test workflow")

            # Update progress
            enhancer.update_progress(task_id, "Processing step 1", 1)
            enhancer.update_progress(task_id, "Processing step 2", 1)

            # Show messages
            enhancer.formatter.print_info("Processing data...")
            enhancer.formatter.print_success("Step completed")

            # Complete operation
            enhancer.complete_operation(task_id, "Workflow completed successfully")

            # Show command help
            enhancer.show_command_help("search")

            assert True  # If we get here, everything worked

        finally:
            enhancer.cleanup()

    def test_different_output_modes(self):
        """Test all output modes work without errors"""
        output_modes = [
            OutputMode.NORMAL,
            OutputMode.QUIET,
            OutputMode.JSON,
            OutputMode.MARKDOWN,
            OutputMode.ACCESSIBLE
        ]

        for mode in output_modes:
            config = UIConfig(output_mode=mode, show_progress=False)
            formatter = EnhancedOutputFormatter(config)

            # Test all message types
            formatter.print_success("Success message")
            formatter.print_error("Error message")
            formatter.print_warning("Warning message")
            formatter.print_info("Info message")

            # Test claim formatting
            claims = [
                {
                    "id": "test",
                    "content": "Test claim",
                    "confidence": 0.8,
                    "tags": ["test"],
                    "state": "validated"
                }
            ]

            table = formatter.format_claim_table(claims)
            details = formatter.format_claim_details(claims[0])

            # Should not raise any exceptions
            assert True

    def test_verbosity_levels(self):
        """Test different verbosity levels"""
        verbosity_levels = [
            VerbosityLevel.SILENT,
            VerbosityLevel.ERROR,
            VerbosityLevel.WARN,
            VerbosityLevel.INFO,
            VerbosityLevel.DEBUG,
            VerbosityLevel.TRACE
        ]

        for level in verbosity_levels:
            config = UIConfig(verbosity=level, show_progress=False)
            enhancer = UIEnhancer(config)

            # Start operation (should respect verbosity level)
            task_id = enhancer.start_operation("Test operation")
            enhancer.complete_operation(task_id, "Completed")

            enhancer.cleanup()

            # Should not raise any exceptions
            assert True

    def test_accessibility_features(self):
        """Test accessibility mode features"""
        config = UIConfig(
            output_mode=OutputMode.ACCESSIBLE,
            accessibility_mode=True,
            show_progress=False
        )
        enhancer = UIEnhancer(config)

        try:
            # Test accessible message formatting
            enhancer.formatter.print_success("Operation completed")
            enhancer.formatter.print_error("Error occurred", "Details here", "Try again")

            # Test accessible prompts in JSON mode (should return defaults)
            json_config = UIConfig(
                output_mode=OutputMode.JSON,
                accessibility_mode=True,
                show_progress=False
            )
            json_enhancer = UIEnhancer(json_config)

            result = json_enhancer.prompter.ask_yes_no("Continue?", default=True)
            assert result is True

            result = json_enhancer.prompter.ask_text("Enter name", default="test")
            assert result == "test"

            assert True  # All accessibility features worked

        finally:
            enhancer.cleanup()
            json_enhancer.cleanup()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])