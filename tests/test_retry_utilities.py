"""
Unit tests for retry utilities and timeout handling
Tests utility functions without mocking
"""
import pytest
import time
from unittest.mock import patch
from typing import Callable, Any

from src.utils.retry_utils import with_llm_retry, EnhancedRetryConfig


class TestRetryUtilities:
    """Test retry utilities and timeout handling"""

    def test_enhanced_retry_config_defaults(self):
        """Test EnhancedRetryConfig default values"""
        config = EnhancedRetryConfig()
        
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True

    def test_enhanced_retry_config_custom(self):
        """Test EnhancedRetryConfig with custom values"""
        config = EnhancedRetryConfig(
            max_attempts=5,
            base_delay=0.5,
            max_delay=30.0,
            exponential_base=3.0,
            jitter=False
        )
        
        assert config.max_attempts == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 3.0
        assert config.jitter is False

    def test_with_llm_retry_success_first_attempt(self):
        """Test retry decorator with success on first attempt"""
        call_count = 0
        
        @with_llm_retry()
        def successful_function():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = successful_function()
        
        assert result == "success"
        assert call_count == 1

    def test_with_llm_retry_success_after_retries(self):
        """Test retry decorator with success after retries"""
        call_count = 0
        
        @with_llm_retry(max_attempts=3, base_delay=0.01)
        def sometimes_successful():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        result = sometimes_successful()
        
        assert result == "success"
        assert call_count == 3

    def test_with_llm_retry_max_attempts_exceeded(self):
        """Test retry decorator when max attempts exceeded"""
        call_count = 0
        
        @with_llm_retry(max_attempts=3, base_delay=0.01)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise Exception("Permanent failure")
        
        with pytest.raises(Exception, match="Permanent failure"):
            always_fails()
        
        assert call_count == 3

    def test_retry_delay_calculation(self):
        """Test retry delay calculation"""
        delays = []
        
        @with_llm_retry(max_attempts=4, base_delay=0.1, exponential_base=2.0, jitter=False)
        def record_delays():
            delays.append(time.time())
            if len(delays) < 4:
                raise Exception("Fail to trigger retry")
        
        with pytest.raises(Exception):
            record_delays()
        
        # Should have 4 attempts (3 retries)
        assert len(delays) == 4
        
        # Check delays between attempts
        if len(delays) >= 3:
            delay1 = delays[1] - delays[0]  # Should be ~0.1
            delay2 = delays[2] - delays[1]  # Should be ~0.2
            delay3 = delays[3] - delays[2]  # Should be ~0.4
            
            assert 0.05 <= delay1 <= 0.15  # Allow some tolerance
            assert 0.15 <= delay2 <= 0.25
            assert 0.35 <= delay3 <= 0.45

    def test_retry_with_jitter(self):
        """Test retry with jitter enabled"""
        delays = []
        
        @with_llm_retry(max_attempts=3, base_delay=0.1, jitter=True)
        def record_delays():
            delays.append(time.time())
            if len(delays) < 3:
                raise Exception("Fail")
        
        with pytest.raises(Exception):
            record_delays()
        
        # With jitter, delays should vary
        if len(delays) >= 3:
            delay1 = delays[1] - delays[0]
            delay2 = delays[2] - delays[1]
            # Delays should not be exactly the same due to jitter
            assert abs(delay1 - delay2) > 0.001

    def test_retry_max_delay_limit(self):
        """Test retry respects max delay limit"""
        start_time = time.time()
        
        @with_llm_retry(max_attempts=3, base_delay=10.0, max_delay=0.5, exponential_base=2.0)
        def slow_retry():
            raise Exception("Always fails")
        
        with pytest.raises(Exception):
            slow_retry()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete quickly due to max_delay limit
        assert total_time < 2.0  # Much less than 10+20+40 seconds

    def test_retry_specific_exception_types(self):
        """Test retry with specific exception types"""
        call_count = 0
        
        @with_llm_retry(max_attempts=3, retry_on=(ValueError,))
        def function_with_different_exceptions():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Retry this")
            elif call_count == 2:
                raise TypeError("Don't retry this")
            else:
                return "success"
        
        # Should raise TypeError without retrying on it
        with pytest.raises(TypeError):
            function_with_different_exceptions()
        
        # Should have only retried ValueError once
        assert call_count == 2

    def test_retry_config_validation(self):
        """Test retry configuration validation"""
        # Invalid max_attempts
        with pytest.raises(ValueError):
            EnhancedRetryConfig(max_attempts=0)
        
        with pytest.raises(ValueError):
            EnhancedRetryConfig(max_attempts=-1)
        
        # Invalid delays
        with pytest.raises(ValueError):
            EnhancedRetryConfig(base_delay=-1.0)
        
        with pytest.raises(ValueError):
            EnhancedRetryConfig(max_delay=0.0)
        
        # Invalid exponential_base
        with pytest.raises(ValueError):
            EnhancedRetryConfig(exponential_base=1.0)

    def test_retry_function_preservation(self):
        """Test that retry decorator preserves function metadata"""
        @with_llm_retry()
        def decorated_function(arg1, arg2, kwarg1=None):
            """Test function docstring"""
            return f"{arg1}-{arg2}-{kwarg1}"
        
        # Function should be callable with original signature
        result = decorated_function("a", "b", kwarg1="c")
        assert result == "a-b-c"
        
        # Should preserve name and docstring
        assert decorated_function.__name__ == "decorated_function"
        assert "Test function docstring" in decorated_function.__doc__