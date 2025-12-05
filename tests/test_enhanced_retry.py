"""
Comprehensive tests for enhanced retry mechanism
Tests exponential backoff retry logic with 10s to 10min range
"""

import asyncio
import time
import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Any, Dict

# Import the enhanced retry utilities
from src.utils.retry_utils import (
    EnhancedRetryHandler, 
    EnhancedRetryConfig, 
    RetryErrorType,
    with_enhanced_retry,
    with_llm_retry,
    with_network_retry
)


class TestEnhancedRetryConfig:
    """Test enhanced retry configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = EnhancedRetryConfig()
        
        assert config.max_attempts == 5
        assert config.base_delay == 10.0  # 10 seconds minimum
        assert config.max_delay == 600.0  # 10 minutes maximum
        assert config.exponential_base == 2.0
        assert config.jitter == True
        assert config.rate_limit_multiplier == 3.0
        assert config.network_multiplier == 2.0
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = EnhancedRetryConfig(
            max_attempts=3,
            base_delay=5.0,
            max_delay=300.0,
            rate_limit_multiplier=2.5
        )
        
        assert config.max_attempts == 3
        assert config.base_delay == 5.0
        assert config.max_delay == 300.0
        assert config.rate_limit_multiplier == 2.5


class TestEnhancedRetryHandler:
    """Test enhanced retry handler functionality"""
    
    def test_error_classification(self):
        """Test error classification logic"""
        handler = EnhancedRetryHandler()
        
        # Test various error types
        assert handler.classify_error(Exception("timeout occurred")) == RetryErrorType.TIMEOUT_ERROR
        assert handler.classify_error(Exception("connection refused")) == RetryErrorType.CONNECTION_ERROR
        assert handler.classify_error(Exception("rate limit exceeded")) == RetryErrorType.RATE_LIMIT_ERROR
        assert handler.classify_error(Exception("network unreachable")) == RetryErrorType.NETWORK_ERROR
        assert handler.classify_error(Exception("unauthorized access")) == RetryErrorType.AUTHENTICATION_ERROR
        assert handler.classify_error(Exception("validation failed")) == RetryErrorType.VALIDATION_ERROR
        assert handler.classify_error(Exception("http 500 error")) == RetryErrorType.HTTP_ERROR
        assert handler.classify_error(Exception("unknown error")) == RetryErrorType.UNKNOWN_ERROR
    
    def test_delay_calculation(self):
        """Test delay calculation with exponential backoff"""
        config = EnhancedRetryConfig(
            base_delay=10.0,
            max_delay=600.0,
            exponential_base=2.0,
            jitter=False  # Disable jitter for predictable testing
        )
        handler = EnhancedRetryHandler(config)
        
        # Test exponential backoff: 10s, 40s, 80s, 160s, 320s
        assert handler.calculate_delay(0, RetryErrorType.NETWORK_ERROR) == 10.0
        assert handler.calculate_delay(1, RetryErrorType.NETWORK_ERROR) == 40.0  # 20s * 2.0
        assert handler.calculate_delay(2, RetryErrorType.NETWORK_ERROR) == 80.0
        assert handler.calculate_delay(3, RetryErrorType.NETWORK_ERROR) == 160.0
        assert handler.calculate_delay(4, RetryErrorType.NETWORK_ERROR) == 320.0
        
        # Test rate limit multiplier (3.0x) - only applied after attempt 0
        rate_limit_delay = handler.calculate_delay(1, RetryErrorType.RATE_LIMIT_ERROR)
        assert rate_limit_delay == 60.0  # 20s * 3.0
        
        # Test max delay capping
        large_attempt_delay = handler.calculate_delay(10, RetryErrorType.NETWORK_ERROR)
        assert large_attempt_delay <= 600.0  # Should be capped at 10 minutes
    
    def test_should_retry_logic(self):
        """Test retry decision logic"""
        config = EnhancedRetryConfig(max_attempts=3)
        handler = EnhancedRetryHandler(config)
        
        # Should retry on retryable errors within attempt limit
        assert handler.should_retry(RetryErrorType.NETWORK_ERROR, 0) == True
        assert handler.should_retry(RetryErrorType.NETWORK_ERROR, 1) == True
        assert handler.should_retry(RetryErrorType.NETWORK_ERROR, 2) == False  # Last attempt
        
        # Should not retry on non-retryable errors
        assert handler.should_retry(RetryErrorType.AUTHENTICATION_ERROR, 0) == False
        assert handler.should_retry(RetryErrorType.VALIDATION_ERROR, 0) == False
    
    def test_successful_execution_sync(self):
        """Test successful synchronous execution"""
        handler = EnhancedRetryHandler()
        mock_func = Mock(return_value="success")
        
        result = handler.execute_with_retry(mock_func, "arg1", kwarg1="value1")
        
        assert result == "success"
        mock_func.assert_called_once_with("arg1", kwarg1="value1")
        
        stats = handler.get_stats()
        assert stats["successful_attempts"] == 1
        assert stats["failed_attempts"] == 0
        assert stats["retry_count"] == 0
    
    def test_retry_execution_sync(self):
        """Test retry logic on synchronous execution"""
        handler = EnhancedRetryHandler(EnhancedRetryConfig(
            max_attempts=3,
            base_delay=0.1,  # Fast for testing
            jitter=False
        ))
        
        mock_func = Mock(side_effect=[
            Exception("network error"),
            Exception("network error"),
            "success"
        ])
        
        start_time = time.time()
        result = handler.execute_with_retry(mock_func)
        end_time = time.time()
        
        assert result == "success"
        assert mock_func.call_count == 3
        
        # Should have waited approximately 0.1s + 0.2s = 0.3s
        assert end_time - start_time >= 0.25  # Allow some tolerance
        
        stats = handler.get_stats()
        assert stats["successful_attempts"] == 1
        assert stats["failed_attempts"] == 2
        assert stats["retry_count"] == 2
    
    def test_max_attempts_exceeded_sync(self):
        """Test behavior when max attempts are exceeded"""
        handler = EnhancedRetryHandler(EnhancedRetryConfig(
            max_attempts=2,
            base_delay=0.1,
            jitter=False
        ))
        
        mock_func = Mock(side_effect=Exception("persistent error"))
        
        with pytest.raises(Exception, match="persistent error"):
            handler.execute_with_retry(mock_func)
        
        assert mock_func.call_count == 2
        
        stats = handler.get_stats()
        assert stats["successful_attempts"] == 0
        assert stats["failed_attempts"] == 2
        assert stats["retry_count"] == 1
    
    @pytest.mark.asyncio
    async def test_successful_execution_async(self):
        """Test successful asynchronous execution"""
        handler = EnhancedRetryHandler()
        mock_func = AsyncMock(return_value="success")
        
        result = await handler.execute_with_retry_async(mock_func, "arg1", kwarg1="value1")
        
        assert result == "success"
        mock_func.assert_called_once_with("arg1", kwarg1="value1")
        
        stats = handler.get_stats()
        assert stats["successful_attempts"] == 1
        assert stats["failed_attempts"] == 0
    
    @pytest.mark.asyncio
    async def test_retry_execution_async(self):
        """Test retry logic on asynchronous execution"""
        handler = EnhancedRetryHandler(EnhancedRetryConfig(
            max_attempts=3,
            base_delay=0.1,  # Fast for testing
            jitter=False
        ))
        
        mock_func = AsyncMock(side_effect=[
            Exception("network error"),
            Exception("network error"),
            "success"
        ])
        
        start_time = time.time()
        result = await handler.execute_with_retry_async(mock_func)
        end_time = time.time()
        
        assert result == "success"
        assert mock_func.call_count == 3
        
        # Should have waited approximately 0.1s + 0.2s = 0.3s
        assert end_time - start_time >= 0.25
        
        stats = handler.get_stats()
        assert stats["successful_attempts"] == 1
        assert stats["failed_attempts"] == 2
        assert stats["retry_count"] == 2


class TestRetryDecorators:
    """Test retry decorators"""
    
    def test_with_llm_retry_decorator_sync(self):
        """Test LLM retry decorator on synchronous function"""
        call_count = 0
        
        @with_llm_retry(max_attempts=3, base_delay=0.1, max_delay=1.0)
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("temporary failure")
            return "success"
        
        result = test_func()
        
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_with_llm_retry_decorator_async(self):
        """Test LLM retry decorator on asynchronous function"""
        call_count = 0
        
        @with_llm_retry(max_attempts=3, base_delay=0.1, max_delay=1.0)
        async def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("temporary failure")
            return "success"
        
        result = await test_func()
        
        assert result == "success"
        assert call_count == 3
    
    def test_with_network_retry_decorator(self):
        """Test network retry decorator"""
        call_count = 0
        
        @with_network_retry(max_attempts=2, base_delay=0.1)
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("connection refused")
            return "success"
        
        result = test_func()
        
        assert result == "success"
        assert call_count == 2


class TestRetryStats:
    """Test retry statistics tracking"""
    
    def test_stats_tracking(self):
        """Test statistics recording and summary"""
        handler = EnhancedRetryHandler(EnhancedRetryConfig(
            max_attempts=3,
            base_delay=0.1,
            jitter=False
        ))
        
        mock_func = Mock(side_effect=[
            Exception("network error"),
            Exception("rate limit"),
            "success"
        ])
        
        handler.execute_with_retry(mock_func)
        
        stats = handler.get_stats()
        
        assert stats["total_attempts"] == 3
        assert stats["successful_attempts"] == 1
        assert stats["failed_attempts"] == 2
        assert stats["success_rate"] == 1.0 / 3.0
        assert stats["retry_count"] == 2
        assert stats["total_delay_time"] >= 0.29  # Should have at least 0.1s + 0.2s = 0.3s (with tolerance)
        assert stats["average_delay"] > 0
        assert stats["error_counts"][RetryErrorType.NETWORK_ERROR.value] == 1
        assert stats["error_counts"][RetryErrorType.RATE_LIMIT_ERROR.value] == 1


class TestIntegrationWithLLMComponents:
    """Test integration with LLM components"""
    
    def test_retry_config_ranges(self):
        """Test that retry configurations meet 10s-10min requirements"""
        # Test default configuration
        config = EnhancedRetryConfig()
        assert config.base_delay >= 10.0, "Base delay should be at least 10 seconds"
        assert config.max_delay <= 600.0, "Max delay should be at most 10 minutes"
        
        # Test exponential backoff stays within bounds
        handler = EnhancedRetryHandler(config)
        for attempt in range(10):  # Test many attempts
            delay = handler.calculate_delay(attempt, RetryErrorType.NETWORK_ERROR)
            assert delay >= 10.0, f"Delay {delay} should be at least 10 seconds"
            assert delay <= 600.0, f"Delay {delay} should be at most 10 minutes"
    
    def test_error_specific_multipliers(self):
        """Test that different error types get appropriate multipliers"""
        config = EnhancedRetryConfig(
            base_delay=10.0,
            rate_limit_multiplier=3.0,
            network_multiplier=2.0,
            timeout_multiplier=1.5,
            jitter=False
        )
        handler = EnhancedRetryHandler(config)
        
        # Test different error types at same attempt level
        network_delay = handler.calculate_delay(1, RetryErrorType.NETWORK_ERROR)
        rate_limit_delay = handler.calculate_delay(1, RetryErrorType.RATE_LIMIT_ERROR)
        timeout_delay = handler.calculate_delay(1, RetryErrorType.TIMEOUT_ERROR)
        
        # Base delay for attempt 1: 10 * 2^1 = 20s
        assert network_delay == 40.0  # 20s * 2.0
        assert rate_limit_delay == 60.0  # 20s * 3.0
        assert timeout_delay == 30.0  # 20s * 1.5


if __name__ == "__main__":
    # Run basic tests to verify functionality
    print("Running enhanced retry tests...")
    
    # Test configuration
    test_config = TestEnhancedRetryConfig()
    test_config.test_default_config()
    print("+ Configuration tests passed")
    
    # Test handler
    test_handler = TestEnhancedRetryHandler()
    test_handler.test_error_classification()
    test_handler.test_delay_calculation()
    test_handler.test_should_retry_logic()
    print("+ Handler tests passed")
    
    # Test execution
    test_handler.test_successful_execution_sync()
    test_handler.test_retry_execution_sync()
    print("+ Execution tests passed")
    
    # Test stats
    test_stats = TestRetryStats()
    test_stats.test_stats_tracking()
    print("+ Statistics tests passed")
    
    # Test integration
    test_integration = TestIntegrationWithLLMComponents()
    test_integration.test_retry_config_ranges()
    test_integration.test_error_specific_multipliers()
    print("+ Integration tests passed")
    
    print("All enhanced retry tests completed successfully!")