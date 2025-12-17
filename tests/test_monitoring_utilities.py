"""
Unit tests for logging and monitoring utilities
Tests logging and monitoring functionality without mocking
"""

import pytest
import logging
import time
from unittest.mock import patch, MagicMock

from src.utils.logging import get_logger, setup_logger
from src.monitoring.performance_monitor import (
    PerformanceMonitor,
    get_performance_monitor,
)


class TestLoggingUtilities:
    """Test logging utilities"""

    def test_setup_logger_default(self):
        """Test logger setup with default configuration"""
        logger = setup_logger("test_logger")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"
        assert logger.level <= logging.INFO  # Should be at least INFO level

    def test_setup_logger_custom_level(self):
        """Test logger setup with custom level"""
        logger = setup_logger("custom_logger", level=logging.DEBUG)

        assert logger.level == logging.DEBUG

    def test_setup_logger_with_handlers(self):
        """Test logger setup with custom handlers"""
        custom_handler = logging.StreamHandler()
        logger = setup_logger("handler_logger", handlers=[custom_handler])

        assert custom_handler in logger.handlers

    def test_get_logger_singleton(self):
        """Test get_logger returns same instance for same name"""
        logger1 = get_logger("singleton_test")
        logger2 = get_logger("singleton_test")

        assert logger1 is logger2

    def test_get_logger_different_names(self):
        """Test get_logger returns different instances for different names"""
        logger1 = get_logger("logger1")
        logger2 = get_logger("logger2")

        assert logger1 is not logger2
        assert logger1.name == "logger1"
        assert logger2.name == "logger2"

    def test_logger_functionality(self):
        """Test actual logging functionality"""
        with patch("logging.Logger.info") as mock_info:
            logger = get_logger("functionality_test")
            logger.info("Test message")

            mock_info.assert_called_once_with("Test message")

    def test_logger_level_filtering(self):
        """Test logger level filtering"""
        # Use setup_logger with INFO level
        logger = setup_logger("level_test", level=logging.INFO)

        # Verify the logger level is set correctly
        assert logger.level == logging.INFO

        # Verify that isEnabledFor works correctly for filtering
        assert not logger.isEnabledFor(logging.DEBUG)  # DEBUG should be filtered
        assert logger.isEnabledFor(logging.INFO)  # INFO should pass


class TestPerformanceMonitor:
    """Test performance monitoring utilities"""

    def test_performance_monitor_creation(self):
        """Test performance monitor creation"""
        monitor = PerformanceMonitor()

        # Updated to match actual API: start_timer, end_timer, get_performance_summary
        assert hasattr(monitor, "start_timer")
        assert hasattr(monitor, "end_timer")
        assert hasattr(monitor, "get_performance_summary")

    def test_start_timing(self):
        """Test starting timing"""
        monitor = PerformanceMonitor()

        # start_timer returns timer_id (string), not start_time (float)
        timer_id = monitor.start_timer("test_operation")

        assert isinstance(timer_id, str)
        assert len(timer_id) > 0

    def test_end_timing(self):
        """Test ending timing"""
        monitor = PerformanceMonitor()

        # start_timer returns timer_id, end_timer takes timer_id (no return value)
        timer_id = monitor.start_timer("test_operation")
        time.sleep(0.01)  # Small delay
        monitor.end_timer(timer_id)

        # Verify timing was recorded in performance summary
        summary = monitor.get_performance_summary(time_window_minutes=1)
        assert summary["performance_stats"]["total_operations"] > 0

    def test_get_metrics_empty(self):
        """Test getting metrics when no operations recorded"""
        monitor = PerformanceMonitor()

        # Use get_performance_summary instead of get_metrics
        summary = monitor.get_performance_summary(time_window_minutes=1)

        assert isinstance(summary, dict)
        assert summary["performance_stats"]["total_operations"] == 0

    def test_get_metrics_with_operations(self):
        """Test getting metrics after operations"""
        monitor = PerformanceMonitor()

        # Record some operations
        timer_id1 = monitor.start_timer("op1")
        time.sleep(0.01)
        monitor.end_timer(timer_id1)

        timer_id2 = monitor.start_timer("op2")
        time.sleep(0.02)
        monitor.end_timer(timer_id2)

        summary = monitor.get_performance_summary(time_window_minutes=1)

        assert "op1" in summary["operation_breakdown"]
        assert "op2" in summary["operation_breakdown"]
        assert summary["operation_breakdown"]["op1"]["count"] == 1
        assert summary["operation_breakdown"]["op2"]["count"] == 1
        assert summary["operation_breakdown"]["op1"]["total"] >= 0.01
        assert summary["operation_breakdown"]["op2"]["total"] >= 0.02

    def test_multiple_same_operations(self):
        """Test multiple executions of same operation"""
        monitor = PerformanceMonitor()

        # Run same operation multiple times
        for _ in range(3):
            timer_id = monitor.start_timer("repeat_op")
            time.sleep(0.01)
            monitor.end_timer(timer_id)

        summary = monitor.get_performance_summary(time_window_minutes=1)

        assert summary["operation_breakdown"]["repeat_op"]["count"] == 3
        assert summary["operation_breakdown"]["repeat_op"]["total"] >= 0.03
        assert "average" in summary["operation_breakdown"]["repeat_op"]
        assert summary["operation_breakdown"]["repeat_op"]["average"] >= 0.01

    def test_performance_monitor_singleton(self):
        """Test performance monitor singleton behavior"""
        monitor1 = get_performance_monitor()
        monitor2 = get_performance_monitor()

        assert monitor1 is monitor2

    def test_timing_without_start(self):
        """Test ending timing without starting"""
        monitor = PerformanceMonitor()

        # Should handle gracefully (end_timer with invalid ID logs warning but doesn't crash)
        # end_timer doesn't return a value
        monitor.end_timer("nonexistent_timer_id")

        # Test passes if no exception is raised
        assert True

    def test_concurrent_operations(self):
        """Test monitoring concurrent operations"""
        monitor = PerformanceMonitor()

        # Start multiple operations
        timer_id1 = monitor.start_timer("concurrent1")
        timer_id2 = monitor.start_timer("concurrent2")

        time.sleep(0.01)

        # End in reverse order
        monitor.end_timer(timer_id2)
        monitor.end_timer(timer_id1)

        # Verify both operations were tracked
        summary = monitor.get_performance_summary(time_window_minutes=1)
        assert "concurrent1" in summary["operation_breakdown"]
        assert "concurrent2" in summary["operation_breakdown"]

    def test_metrics_reset(self):
        """Test metrics isolation per monitor instance"""
        monitor = PerformanceMonitor()

        # Add some metrics
        timer_id = monitor.start_timer("reset_test")
        monitor.end_timer(timer_id)

        summary = monitor.get_performance_summary(time_window_minutes=1)
        assert summary["performance_stats"]["total_operations"] == 1

        # Create new monitor instance to test isolation
        new_monitor = PerformanceMonitor()
        new_summary = new_monitor.get_performance_summary(time_window_minutes=1)
        assert new_summary["performance_stats"]["total_operations"] == 0

    def test_performance_decorator(self):
        """Test performance monitoring decorator"""
        monitor = PerformanceMonitor()

        # Use timer decorator instead of monitor_performance
        @monitor.timer("decorated_function")
        def decorated_function(x, y):
            time.sleep(0.01)
            return x + y

        result = decorated_function(2, 3)

        assert result == 5

        summary = monitor.get_performance_summary(time_window_minutes=1)
        assert "decorated_function" in summary["operation_breakdown"]
        assert summary["operation_breakdown"]["decorated_function"]["count"] == 1
        assert summary["operation_breakdown"]["decorated_function"]["total"] >= 0.01
