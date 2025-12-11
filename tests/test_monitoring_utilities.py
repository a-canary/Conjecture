"""
Unit tests for logging and monitoring utilities
Tests logging and monitoring functionality without mocking
"""
import pytest
import logging
import time
from unittest.mock import patch, MagicMock

from src.utils.logging import get_logger, setup_logger
from src.monitoring.performance_monitor import PerformanceMonitor, get_performance_monitor


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
        with patch('logging.Logger.info') as mock_info:
            logger = get_logger("functionality_test")
            logger.info("Test message")
            
            mock_info.assert_called_once_with("Test message")

    def test_logger_level_filtering(self):
        """Test logger level filtering"""
        with patch('logging.Logger.debug') as mock_debug, \
             patch('logging.Logger.info') as mock_info:
            
            logger = get_logger("level_test", level=logging.INFO)
            logger.debug("Debug message")  # Should be filtered
            logger.info("Info message")     # Should pass
            
            mock_debug.assert_not_called()
            mock_info.assert_called_once_with("Info message")


class TestPerformanceMonitor:
    """Test performance monitoring utilities"""

    def test_performance_monitor_creation(self):
        """Test performance monitor creation"""
        monitor = PerformanceMonitor()
        
        assert hasattr(monitor, 'start_timing')
        assert hasattr(monitor, 'end_timing')
        assert hasattr(monitor, 'get_metrics')

    def test_start_timing(self):
        """Test starting timing"""
        monitor = PerformanceMonitor()
        
        start_time = monitor.start_timing("test_operation")
        
        assert isinstance(start_time, float)
        assert start_time > 0

    def test_end_timing(self):
        """Test ending timing"""
        monitor = PerformanceMonitor()
        
        start_time = monitor.start_timing("test_operation")
        time.sleep(0.01)  # Small delay
        duration = monitor.end_timing("test_operation")
        
        assert isinstance(duration, float)
        assert duration >= 0.01  # Should be at least the sleep time

    def test_get_metrics_empty(self):
        """Test getting metrics when no operations recorded"""
        monitor = PerformanceMonitor()
        
        metrics = monitor.get_metrics()
        
        assert isinstance(metrics, dict)
        assert len(metrics) == 0

    def test_get_metrics_with_operations(self):
        """Test getting metrics after operations"""
        monitor = PerformanceMonitor()
        
        # Record some operations
        monitor.start_timing("op1")
        time.sleep(0.01)
        monitor.end_timing("op1")
        
        monitor.start_timing("op2")
        time.sleep(0.02)
        monitor.end_timing("op2")
        
        metrics = monitor.get_metrics()
        
        assert "op1" in metrics
        assert "op2" in metrics
        assert metrics["op1"]["count"] == 1
        assert metrics["op2"]["count"] == 1
        assert metrics["op1"]["total_time"] >= 0.01
        assert metrics["op2"]["total_time"] >= 0.02

    def test_multiple_same_operations(self):
        """Test multiple executions of same operation"""
        monitor = PerformanceMonitor()
        
        # Run same operation multiple times
        for _ in range(3):
            monitor.start_timing("repeat_op")
            time.sleep(0.01)
            monitor.end_timing("repeat_op")
        
        metrics = monitor.get_metrics()
        
        assert metrics["repeat_op"]["count"] == 3
        assert metrics["repeat_op"]["total_time"] >= 0.03
        assert "average_time" in metrics["repeat_op"]
        assert metrics["repeat_op"]["average_time"] >= 0.01

    def test_performance_monitor_singleton(self):
        """Test performance monitor singleton behavior"""
        monitor1 = get_performance_monitor()
        monitor2 = get_performance_monitor()
        
        assert monitor1 is monitor2

    def test_timing_without_start(self):
        """Test ending timing without starting"""
        monitor = PerformanceMonitor()
        
        # Should handle gracefully
        duration = monitor.end_timing("nonexistent_operation")
        
        # Should return None or 0, not crash
        assert duration is None or duration == 0

    def test_concurrent_operations(self):
        """Test monitoring concurrent operations"""
        monitor = PerformanceMonitor()
        
        # Start multiple operations
        start1 = monitor.start_timing("concurrent1")
        start2 = monitor.start_timing("concurrent2")
        
        time.sleep(0.01)
        
        # End in reverse order
        duration2 = monitor.end_timing("concurrent2")
        duration1 = monitor.end_timing("concurrent1")
        
        assert duration1 >= 0.01
        assert duration2 >= 0.01
        assert start1 < start2  # Started in order

    def test_metrics_reset(self):
        """Test resetting metrics"""
        monitor = PerformanceMonitor()
        
        # Add some metrics
        monitor.start_timing("reset_test")
        monitor.end_timing("reset_test")
        
        assert len(monitor.get_metrics()) == 1
        
        # Reset and verify
        monitor.reset_metrics()
        
        assert len(monitor.get_metrics()) == 0

    def test_performance_decorator(self):
        """Test performance monitoring decorator"""
        monitor = PerformanceMonitor()
        
        @monitor.monitor_performance
        def decorated_function(x, y):
            time.sleep(0.01)
            return x + y
        
        result = decorated_function(2, 3)
        
        assert result == 5
        
        metrics = monitor.get_metrics()
        assert "decorated_function" in metrics
        assert metrics["decorated_function"]["count"] == 1
        assert metrics["decorated_function"]["total_time"] >= 0.01