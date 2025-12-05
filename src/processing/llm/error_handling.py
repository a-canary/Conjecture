"""
Enhanced Error Handling and Retry Logic for LLM Processors
Provides robust error handling with exponential backoff and circuit breaker pattern
"""

import time
import random
import logging
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of errors that can occur during LLM processing"""
    NETWORK_ERROR = "network_error"
    API_ERROR = "api_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    AUTHENTICATION_ERROR = "authentication_error"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class RetryConfig:
    """Configuration for retry logic with enhanced exponential backoff (10s to 10min range)"""
    max_attempts: int = 5  # Increased attempts for better resilience
    base_delay: float = 10.0  # Base delay in seconds (increased to 10s)
    max_delay: float = 600.0  # Maximum delay in seconds (10 minutes)
    exponential_base: float = 2.0  # Exponential backoff base
    jitter: bool = True  # Add random jitter to prevent thundering herd
    retry_on: List[ErrorType] = field(default_factory=lambda: [
        ErrorType.NETWORK_ERROR,
        ErrorType.API_ERROR,
        ErrorType.RATE_LIMIT_ERROR,  # Added rate limit retry
        ErrorType.TIMEOUT_ERROR,
        ErrorType.UNKNOWN_ERROR
    ])
    # Enhanced configuration for different error types
    rate_limit_multiplier: float = 2.0  # Multiplier for rate limit errors
    network_multiplier: float = 1.5  # Multiplier for network errors
    timeout_multiplier: float = 1.0  # Multiplier for timeout errors


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern"""
    failure_threshold: int = 5  # Number of failures before opening circuit
    recovery_timeout: float = 60.0  # Seconds to wait before trying again
    expected_exception: Type[Exception] = Exception


class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker to function"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.config.expected_exception as e:
                self._on_failure()
                raise
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should be reset to HALF_OPEN"""
        return (time.time() - self.last_failure_time) >= self.config.recovery_timeout
    
    def _on_success(self):
        """Handle successful operation"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = "OPEN"


class RetryHandler:
    """Enhanced retry handler with exponential backoff"""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply retry logic to function"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_type = self._classify_error(e)
                    
                    # Don't retry on certain error types
                    if error_type not in self.config.retry_on:
                        logger.error(f"Non-retryable error ({error_type}): {e}")
                        raise
                    
                    # Don't retry on last attempt
                    if attempt == self.config.max_attempts - 1:
                        logger.error(f"Max retry attempts ({self.config.max_attempts}) reached")
                        raise
                    
                    # Calculate delay with error type consideration
                    delay = self._calculate_delay(attempt, error_type)
                    logger.warning(f"Attempt {attempt + 1} failed ({error_type}): {e}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
            
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
        
        return wrapper
    
    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify error type for retry decision"""
        error_message = str(error).lower()
        
        if "timeout" in error_message or "timed out" in error_message:
            return ErrorType.TIMEOUT_ERROR
        elif "connection" in error_message or "network" in error_message:
            return ErrorType.NETWORK_ERROR
        elif "rate limit" in error_message or "too many requests" in error_message:
            return ErrorType.RATE_LIMIT_ERROR
        elif "unauthorized" in error_message or "authentication" in error_message:
            return ErrorType.AUTHENTICATION_ERROR
        elif "validation" in error_message or "invalid" in error_message:
            return ErrorType.VALIDATION_ERROR
        elif "api" in error_message or "request" in error_message:
            return ErrorType.API_ERROR
        else:
            return ErrorType.UNKNOWN_ERROR
    
    def _calculate_delay(self, attempt: int, error_type: ErrorType) -> float:
        """Calculate delay with exponential backoff, error-specific multipliers, and jitter"""
        # Base exponential backoff
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        
        # Apply error-specific multipliers
        if error_type == ErrorType.RATE_LIMIT_ERROR:
            delay *= self.config.rate_limit_multiplier
        elif error_type == ErrorType.NETWORK_ERROR:
            delay *= self.config.network_multiplier
        elif error_type == ErrorType.TIMEOUT_ERROR:
            delay *= self.config.timeout_multiplier
        
        # Ensure we don't exceed max_delay
        delay = min(delay, self.config.max_delay)
        
        if self.config.jitter:
            # Add random jitter (±25% of delay) to prevent thundering herd
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)  # Ensure non-negative


class LLMErrorHandler:
    """Enhanced error handler for LLM operations"""
    
    def __init__(self, 
                 retry_config: Optional[RetryConfig] = None,
                 circuit_breaker_config: Optional[CircuitBreakerConfig] = None):
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60.0
        )
        
        # Create circuit breaker for each operation type
        self.circuit_breakers = {
            "generation": CircuitBreaker(self.circuit_breaker_config),
            "processing": CircuitBreaker(self.circuit_breaker_config),
            "analysis": CircuitBreaker(self.circuit_breaker_config)
        }
        
        self.stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "retries_attempted": 0,
            "circuit_breaker_activations": 0
        }
    
    def handle_generation(self, func: Callable) -> Callable:
        """Handle generation operations with retry and circuit breaker"""
        return self._create_handler("generation", func)
    
    def handle_processing(self, func: Callable) -> Callable:
        """Handle processing operations with retry and circuit breaker"""
        return self._create_handler("processing", func)
    
    def handle_analysis(self, func: Callable) -> Callable:
        """Handle analysis operations with retry and circuit breaker"""
        return self._create_handler("analysis", func)
    
    def _create_handler(self, operation_type: str, func: Callable) -> Callable:
        """Create enhanced handler for specific operation type"""
        circuit_breaker = self.circuit_breakers[operation_type]
        retry_handler = RetryHandler(self.retry_config)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.stats["total_operations"] += 1
            
            try:
                # Apply circuit breaker and retry logic
                result = circuit_breaker(retry_handler(func))(*args, **kwargs)
                self.stats["successful_operations"] += 1
                return result
                
            except Exception as e:
                self.stats["failed_operations"] += 1
                
                # Check if circuit breaker is open
                if circuit_breaker.state == "OPEN":
                    self.stats["circuit_breaker_activations"] += 1
                    logger.error(f"Circuit breaker OPEN for {operation_type}: {e}")
                
                # Log the error with context
                logger.error(f"LLM operation failed ({operation_type}): {e}")
                
                # Re-raise the exception
                raise
        
        return wrapper
    
    def get_stats(self) -> Dict[str, Any]:
        """Get error handling statistics"""
        stats = self.stats.copy()
        
        if stats["total_operations"] > 0:
            stats["success_rate"] = stats["successful_operations"] / stats["total_operations"]
            stats["failure_rate"] = stats["failed_operations"] / stats["total_operations"]
        else:
            stats["success_rate"] = 0.0
            stats["failure_rate"] = 0.0
        
        # Add circuit breaker status
        stats["circuit_breakers"] = {}
        for name, cb in self.circuit_breakers.items():
            stats["circuit_breakers"][name] = {
                "state": cb.state,
                "failure_count": cb.failure_count,
                "last_failure_time": cb.last_failure_time
            }
        
        return stats
    
    def reset_stats(self):
        """Reset error handling statistics"""
        self.stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "retries_attempted": 0,
            "circuit_breaker_activations": 0
        }
        
        # Reset circuit breakers
        for cb in self.circuit_breakers.values():
            cb.failure_count = 0
            cb.state = "CLOSED"
            cb.last_failure_time = None


# Global error handler instance
default_error_handler = LLMErrorHandler()


def with_error_handling(operation_type: str = "generation", 
                      error_handler: Optional[LLMErrorHandler] = None):
    """Decorator to apply error handling to LLM operations"""
    handler = error_handler or default_error_handler
    
    def decorator(func: Callable) -> Callable:
        if operation_type == "generation":
            return handler.handle_generation(func)
        elif operation_type == "processing":
            return handler.handle_processing(func)
        elif operation_type == "analysis":
            return handler.handle_analysis(func)
        else:
            # Default to generation
            return handler.handle_generation(func)
    
    return decorator


# Convenience decorators
@with_error_handling("generation")
def handle_generation_errors(func: Callable) -> Callable:
    """Convenience decorator for generation operations"""
    return func


@with_error_handling("processing")
def handle_processing_errors(func: Callable) -> Callable:
    """Convenience decorator for processing operations"""
    return func


@with_error_handling("analysis")
def handle_analysis_errors(func: Callable) -> Callable:
    """Convenience decorator for analysis operations"""
    return func