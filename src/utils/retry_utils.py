"""
Enhanced Retry Utilities for LLM Operations
Provides comprehensive exponential backoff retry logic with configurable parameters (10s to 10min range)
Designed for use across all LLM components including LLMLocalRouter
"""

import time
import random
import logging
import asyncio
from typing import Any, Callable, Dict, List, Optional, Union, Type, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

# Configure logging
logger = logging.getLogger(__name__)

class RetryErrorType(Enum):
    """Types of errors that can occur during LLM processing"""
    NETWORK_ERROR = "network_error"
    API_ERROR = "api_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    AUTHENTICATION_ERROR = "authentication_error"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT_ERROR = "timeout_error"
    CONNECTION_ERROR = "connection_error"
    HTTP_ERROR = "http_error"
    UNKNOWN_ERROR = "unknown_error"

@dataclass
class EnhancedRetryConfig:
    """Enhanced configuration for retry logic with 10s to 10min range"""
    max_attempts: int = 5
    base_delay: float = 10.0  # Base delay in seconds (10s minimum)
    max_delay: float = 600.0  # Maximum delay in seconds (10min maximum)
    exponential_base: float = 2.0  # Exponential backoff base
    jitter: bool = True  # Add random jitter to prevent thundering herd
    jitter_percentage: float = 0.25  # Jitter percentage (±25%)
    
    # Error-specific multipliers for different backoff strategies
    rate_limit_multiplier: float = 3.0  # Higher multiplier for rate limits
    network_multiplier: float = 2.0  # Multiplier for network errors
    timeout_multiplier: float = 1.5  # Multiplier for timeout errors
    connection_multiplier: float = 2.5  # Multiplier for connection errors
    http_error_multiplier: float = 1.0  # Multiplier for HTTP errors
    
    # Error types to retry on
    retry_on: List[RetryErrorType] = field(default_factory=lambda: [
        RetryErrorType.NETWORK_ERROR,
        RetryErrorType.API_ERROR,
        RetryErrorType.RATE_LIMIT_ERROR,
        RetryErrorType.TIMEOUT_ERROR,
        RetryErrorType.CONNECTION_ERROR,
        RetryErrorType.HTTP_ERROR,
        RetryErrorType.UNKNOWN_ERROR
    ])
    
    # Errors that should not be retried
    non_retryable: List[RetryErrorType] = field(default_factory=lambda: [
        RetryErrorType.AUTHENTICATION_ERROR,
        RetryErrorType.VALIDATION_ERROR
    ])

class RetryStats:
    """Statistics tracking for retry operations"""
    
    def __init__(self):
        self.total_attempts = 0
        self.successful_attempts = 0
        self.failed_attempts = 0
        self.retry_count = 0
        self.total_delay_time = 0.0
        self.error_counts = {error_type: 0 for error_type in RetryErrorType}
    
    def record_attempt(self, success: bool, error_type: Optional[RetryErrorType] = None, delay: float = 0.0):
        """Record an attempt attempt"""
        self.total_attempts += 1
        if success:
            self.successful_attempts += 1
        else:
            self.failed_attempts += 1
            if error_type:
                self.error_counts[error_type] += 1
        
        if delay > 0:
            self.total_delay_time += delay
    
    def record_retry(self):
        """Record a retry attempt"""
        self.retry_count += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        success_rate = self.successful_attempts / self.total_attempts if self.total_attempts > 0 else 0.0
        return {
            "total_attempts": self.total_attempts,
            "successful_attempts": self.successful_attempts,
            "failed_attempts": self.failed_attempts,
            "success_rate": success_rate,
            "retry_count": self.retry_count,
            "total_delay_time": self.total_delay_time,
            "average_delay": self.total_delay_time / self.retry_count if self.retry_count > 0 else 0.0,
            "error_counts": dict(self.error_counts)
        }

class EnhancedRetryHandler:
    """Enhanced retry handler with comprehensive exponential backoff"""
    
    def __init__(self, config: Optional[EnhancedRetryConfig] = None):
        self.config = config or EnhancedRetryConfig()
        self.stats = RetryStats()
    
    def classify_error(self, error: Exception) -> RetryErrorType:
        """Classify error type for retry decision"""
        error_message = str(error).lower()
        error_class_name = error.__class__.__name__.lower()
        
        # Check for specific error patterns
        if "timeout" in error_message or "timed out" in error_message:
            return RetryErrorType.TIMEOUT_ERROR
        elif "connection" in error_message or "connect" in error_message:
            return RetryErrorType.CONNECTION_ERROR
        elif "network" in error_message or "dns" in error_message:
            return RetryErrorType.NETWORK_ERROR
        elif "rate limit" in error_message or "too many requests" in error_message or "429" in error_message:
            return RetryErrorType.RATE_LIMIT_ERROR
        elif "unauthorized" in error_message or "authentication" in error_message or "401" in error_message:
            return RetryErrorType.AUTHENTICATION_ERROR
        elif "validation" in error_message or "invalid" in error_message or "400" in error_message:
            return RetryErrorType.VALIDATION_ERROR
        elif "http" in error_message or "httperror" in error_class_name or "5" in error_message[:1]:  # 5xx errors
            return RetryErrorType.HTTP_ERROR
        elif "api" in error_message or "request" in error_message:
            return RetryErrorType.API_ERROR
        else:
            return RetryErrorType.UNKNOWN_ERROR
    
    def calculate_delay(self, attempt: int, error_type: RetryErrorType) -> float:
        """Calculate delay with exponential backoff and error-specific multipliers"""
        # Base exponential backoff: base_delay * (exponential_base ^ attempt)
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        
        # Apply error-specific multipliers
        multiplier_map = {
            RetryErrorType.RATE_LIMIT_ERROR: self.config.rate_limit_multiplier,
            RetryErrorType.NETWORK_ERROR: self.config.network_multiplier,
            RetryErrorType.TIMEOUT_ERROR: self.config.timeout_multiplier,
            RetryErrorType.CONNECTION_ERROR: self.config.connection_multiplier,
            RetryErrorType.HTTP_ERROR: self.config.http_error_multiplier,
        }
        
        if error_type in multiplier_map:
            delay *= multiplier_map[error_type]
        
        # Ensure we don't exceed max_delay
        delay = min(delay, self.config.max_delay)
        
        if self.config.jitter:
            # Add random jitter to prevent thundering herd
            jitter_range = delay * self.config.jitter_percentage
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)  # Ensure non-negative
    
    def should_retry(self, error_type: RetryErrorType, attempt: int) -> bool:
        """Determine if operation should be retried"""
        # Don't retry if we've reached max attempts
        if attempt >= self.config.max_attempts - 1:
            return False
        
        # Don't retry on non-retryable errors
        if error_type in self.config.non_retryable:
            return False
        
        # Only retry on configured error types
        return error_type in self.config.retry_on
    
    def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with enhanced retry logic (synchronous)"""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                result = func(*args, **kwargs)
                self.stats.record_attempt(success=True)
                return result
                
            except Exception as e:
                last_exception = e
                error_type = self.classify_error(e)
                
                # Record the failed attempt
                self.stats.record_attempt(success=False, error_type=error_type)
                
                # Check if we should retry
                if not self.should_retry(error_type, attempt):
                    logger.error(f"Non-retryable error ({error_type}): {e}")
                    break
                
                # Calculate delay for next attempt
                delay = self.calculate_delay(attempt, error_type)
                self.stats.record_retry()
                
                logger.warning(
                    f"Attempt {attempt + 1} failed ({error_type}): {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                time.sleep(delay)
        
        # All attempts failed
        logger.error(f"All {self.config.max_attempts} attempts failed. Last error: {last_exception}")
        raise last_exception
    
    async def execute_with_retry_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with enhanced retry logic (asynchronous)"""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                self.stats.record_attempt(success=True)
                return result
                
            except Exception as e:
                last_exception = e
                error_type = self.classify_error(e)
                
                # Record the failed attempt
                self.stats.record_attempt(success=False, error_type=error_type)
                
                # Check if we should retry
                if not self.should_retry(error_type, attempt):
                    logger.error(f"Non-retryable error ({error_type}): {e}")
                    break
                
                # Calculate delay for next attempt
                delay = self.calculate_delay(attempt, error_type)
                self.stats.record_retry()
                
                logger.warning(
                    f"Attempt {attempt + 1} failed ({error_type}): {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                await asyncio.sleep(delay)
        
        # All attempts failed
        logger.error(f"All {self.config.max_attempts} attempts failed. Last error: {last_exception}")
        raise last_exception
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retry statistics"""
        return self.stats.get_summary()
    
    def reset_stats(self):
        """Reset retry statistics"""
        self.stats = RetryStats()

# Global retry handler instance
default_retry_handler = EnhancedRetryHandler()

def with_enhanced_retry(config: Optional[EnhancedRetryConfig] = None, 
                       handler: Optional[EnhancedRetryHandler] = None):
    """Decorator to apply enhanced retry logic to functions"""
    retry_handler = handler or default_retry_handler
    
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await retry_handler.execute_with_retry_async(func, *args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return retry_handler.execute_with_retry(func, *args, **kwargs)
            return sync_wrapper
    
    return decorator

# Convenience decorators
def with_llm_retry(max_attempts: int = 5, base_delay: float = 10.0, max_delay: float = 600.0):
    """Convenience decorator for LLM operations with standard 10s-10min range"""
    config = EnhancedRetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay
    )
    return with_enhanced_retry(config)

def with_network_retry(max_attempts: int = 3, base_delay: float = 5.0):
    """Convenience decorator for network operations"""
    config = EnhancedRetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=60.0,
        network_multiplier=2.0
    )
    return with_enhanced_retry(config)

def with_rate_limit_retry(max_attempts: int = 5, base_delay: float = 30.0):
    """Convenience decorator for rate-limited operations"""
    config = EnhancedRetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=600.0,
        rate_limit_multiplier=3.0
    )
    return with_enhanced_retry(config)