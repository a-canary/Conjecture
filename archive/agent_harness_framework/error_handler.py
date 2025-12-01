"""
Error Handler for Agent Harness
Comprehensive error management, recovery, and fallback strategies
"""

import asyncio
import traceback
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
import logging

from .models import ErrorEntry, ErrorSeverity, ErrorResult, FallbackSolution
from ..utils.id_generator import generate_error_id


logger = logging.getLogger(__name__)


class ErrorCategory(str, Enum):
    """Error categories"""
    COMPONENT = "component"
    INTEGRATION = "integration"
    DATA = "data"
    WORKFLOW = "workflow"
    USER = "user"
    SYSTEM = "system"


class RecoveryStrategy(str, Enum):
    """Recovery strategies"""
    RETRY = "retry"
    FALLBACK = "fallback"
    DEGRADE = "degrade"
    ESCALATE = "escalate"
    IGNORE = "ignore"


class RetryPolicy:
    """Retry policy configuration"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, backoff_factor: float = 2.0,
                 jitter: bool = True):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt"""
        delay = min(self.base_delay * (self.backoff_factor ** attempt), self.max_delay)
        
        if self.jitter:
            # Add jitter to avoid thundering herd
            import random
            delay *= (0.5 + random.random() * 0.5)
        
        return delay


class ErrorHandler:
    """
    Comprehensive error handling with recovery, fallback, and escalation
    """

    def __init__(self, max_error_history: int = 1000):
        self.max_error_history = max_error_history
        self.error_history: List[ErrorEntry] = []
        
        # Error handling strategies
        self.error_handlers: Dict[str, Callable] = {}
        self.fallback_solutions: Dict[str, Callable] = {}
        self.retry_policies: Dict[str, RetryPolicy] = {}
        
        # Error patterns and their handlers
        self.error_patterns: List[Dict[str, Any]] = []
        
        # Circuit breaker state
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Monitoring
        self.error_counts: Dict[str, int] = {}
        self.error_rates: Dict[str, float] = {}

    def register_error_handler(self, error_type: str, handler: Callable) -> None:
        """
        Register a handler for specific error types
        
        Args:
            error_type: Error type or pattern
            handler: Handler function
        """
        self.error_handlers[error_type] = handler
        logger.info(f"Registered error handler for: {error_type}")

    def register_fallback_solution(self, operation: str, solution_generator: Callable) -> None:
        """
        Register a fallback solution generator for an operation
        
        Args:
            operation: Operation name
            solution_generator: Function that generates fallback solutions
        """
        self.fallback_solutions[operation] = solution_generator
        logger.info(f"Registered fallback solution for: {operation}")

    def register_retry_policy(self, operation: str, policy: RetryPolicy) -> None:
        """
        Register a retry policy for an operation
        
        Args:
            operation: Operation name
            policy: Retry policy
        """
        self.retry_policies[operation] = policy
        logger.info(f"Registered retry policy for: {operation}")

    def register_error_pattern(self, pattern: str, error_type: str, 
                             severity: ErrorSeverity, 
                             recovery_strategy: RecoveryStrategy) -> None:
        """
        Register an error pattern with associated handling strategy
        
        Args:
            pattern: Error message pattern (regex or substring)
            error_type: Categorized error type
            severity: Error severity
            recovery_strategy: Default recovery strategy
        """
        self.error_patterns.append({
            'pattern': pattern,
            'error_type': error_type,
            'severity': severity,
            'recovery_strategy': recovery_strategy
        })

    async def handle_error(self, error: Exception, context: Dict[str, Any],
                          session_id: Optional[str] = None,
                          workflow_execution_id: Optional[str] = None) -> ErrorResult:
        """
        Handle an error with appropriate recovery strategy
        
        Args:
            error: The exception that occurred
            context: Error context information
            session_id: Optional session ID
            workflow_execution_id: Optional workflow execution ID
            
        Returns:
            Error handling result
        """
        error_id = generate_error_id()
        error_message = str(error)
        error_type = type(error).__name__
        
        try:
            # Categorize and classify error
            error_category = self._classify_error(error_message, context)
            error_severity = self._determine_severity(error, error_category, context)
            
            # Create error entry
            error_entry = ErrorEntry(
                id=error_id,
                session_id=session_id,
                workflow_execution_id=workflow_execution_id,
                component=context.get('component', 'unknown'),
                error_type=error_type,
                message=error_message,
                severity=error_severity,
                stack_trace=traceback.format_exc(),
                context=context
            )
            
            # Store in history
            self._store_error(error_entry)
            
            # Update error statistics
            self._update_error_stats(error_entry)
            
            # Determine recovery strategy
            recovery_strategy = self._determine_recovery_strategy(
                error_entry, context
            )
            
            # Execute recovery strategy
            recovery_result = await self._execute_recovery_strategy(
                recovery_strategy, error_entry, context
            )
            
            # Generate user message
            user_message = self._generate_user_message(error_entry, recovery_result)
            
            # Check circuit breakers
            should_block = self._check_circuit_breakers(error_entry.component)
            
            error_result = ErrorResult(
                error_handled=recovery_result['handled'],
                recovery_method=recovery_result['method'],
                fallback_solution=recovery_result.get('fallback_solution'),
                user_message=user_message,
                technical_details=f"{error_type}: {error_message}",
                should_retry=recovery_result.get('should_retry', False)
            )
            
            # If error wasn't handled and should be blocked
            if not recovery_result['handled'] and should_block:
                error_result.user_message = (
                    f"Operation temporarily unavailable due to repeated failures. "
                    f"Please try again later. ({error_message})"
                )
            
            logger.info(f"Handled error {error_id}: {error_type} - {recovery_result['method']}")
            return error_result

        except Exception as e:
            # Emergency error handling - create minimal error result
            logger.error(f"Error handling failed: {e}")
            return ErrorResult(
                error_handled=False,
                user_message="An unexpected error occurred. Please try again.",
                technical_details=f"Critical error: {error_message}",
                should_retry=False
            )

    async def retry_operation(self, operation: Callable, operation_name: str,
                            context: Optional[Dict[str, Any]] = None,
                            *args, **kwargs) -> Any:
        """
        Execute an operation with retry logic
        
        Args:
            operation: Operation function to execute
            operation_name: Name of the operation for policy lookup
            context: Optional context
            *args, **kwargs: Arguments for the operation
            
        Returns:
            Operation result
            
        Raises:
            Last exception if all retries failed
        """
        context = context or {}
        retry_policy = self.retry_policies.get(operation_name, RetryPolicy())
        
        last_exception = None
        
        for attempt in range(retry_policy.max_retries + 1):
            try:
                if attempt > 0:
                    delay = retry_policy.get_delay(attempt - 1)
                    logger.info(f"Retrying {operation_name} attempt {attempt + 1} after {delay:.2f}s")
                    await asyncio.sleep(delay)
                
                # Execute operation
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)
                
                # If successful and this was a retry, log it
                if attempt > 0:
                    logger.info(f"{operation_name} succeeded on attempt {attempt + 1}")
                
                return result

            except Exception as e:
                last_exception = e
                
                # Handle the error but continue retrying
                error_context = {
                    'operation': operation_name,
                    'attempt': attempt + 1,
                    'max_attempts': retry_policy.max_retries + 1,
                    **context
                }
                
                await self.handle_error(e, error_context)
                
                # Check circuit breaker
                if self._check_circuit_breakers(operation_name):
                    logger.warning(f"Circuit breaker activated for {operation_name}, stopping retries")
                    break
        
        # All retries failed
        logger.error(f"Operation {operation_name} failed after {attempt + 1} attempts")
        raise last_exception

    async def get_fallback_solution(self, original_operation: str, 
                                   error_context: Dict[str, Any]) -> Optional[FallbackSolution]:
        """
        Get a fallback solution for a failed operation
        
        Args:
            original_operation: Name of the original operation
            error_context: Error context information
            
        Returns:
            Fallback solution or None if not available
        """
        try:
            solution_generator = self.fallback_solutions.get(original_operation)
            if not solution_generator:
                return None
            
            if asyncio.iscoroutinefunction(solution_generator):
                solution = await solution_generator(error_context)
            else:
                solution = solution_generator(error_context)
            
            if isinstance(solution, FallbackSolution):
                return solution
            elif isinstance(solution, dict):
                return FallbackSolution(
                    operation=original_operation,
                    **solution
                )
            else:
                logger.warning(f"Invalid fallback solution type for {original_operation}")
                return None

        except Exception as e:
            logger.error(f"Failed to generate fallback solution for {original_operation}: {e}")
            return None

    async def log_error(self, entry: ErrorEntry) -> str:
        """
        Log an error entry
        
        Args:
            entry: Error entry to log
            
        Returns:
            Error entry ID
        """
        self._store_error(entry)
        self._update_error_stats(entry)
        logger.error(f"Logged error: {entry.error_type} - {entry.message}")
        return entry.id

    async def resolve_error(self, error_id: str, resolution: str) -> bool:
        """
        Mark an error as resolved
        
        Args:
            error_id: Error entry ID
            resolution: Resolution description
            
        Returns:
            True if error was found and resolved
        """
        for entry in self.error_history:
            if entry.id == error_id:
                entry.resolve(resolution)
                logger.info(f"Resolved error {error_id}: {resolution}")
                return True
        return False

    async def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get error handling statistics
        
        Returns:
            Error statistics
        """
        try:
            total_errors = len(self.error_history)
            
            # Error counts by severity
            severity_counts = {}
            for entry in self.error_history:
                severity = entry.severity.value
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Error counts by component
            component_counts = {}
            for entry in self.error_history:
                component = entry.component
                component_counts[component] = component_counts.get(component, 0) + 1
            
            # Recent errors (last hour)
            recent_errors = [
                entry for entry in self.error_history
                if (datetime.utcnow() - entry.timestamp).total_seconds() < 3600
            ]
            
            # Resolution rate
            resolved_errors = [entry for entry in self.error_history if entry.resolved]
            resolution_rate = len(resolved_errors) / total_errors if total_errors > 0 else 0
            
            return {
                'total_errors': total_errors,
                'errors_last_hour': len(recent_errors),
                'severity_distribution': severity_counts,
                'component_distribution': component_counts,
                'resolution_rate': resolution_rate,
                'registered_handlers': len(self.error_handlers),
                'registered_fallback_solutions': len(self.fallback_solutions),
                'registered_retry_policies': len(self.retry_policies),
                'active_circuit_breakers': len(self._get_active_circuit_breakers()),
                'error_history_size': len(self.error_history),
                'max_error_history': self.max_error_history
            }

        except Exception as e:
            logger.error(f"Failed to get error statistics: {e}")
            return {}

    def _classify_error(self, error_message: str, context: Dict[str, Any]) -> ErrorCategory:
        """Classify error into category"""
        component = context.get('component', '').lower()
        
        # Check error patterns
        for pattern_info in self.error_patterns:
            pattern = pattern_info['pattern']
            if pattern in error_message.lower():
                # Could return specific category based on pattern
                pass
        
        # Classify based on component
        if 'component' in context:
            return ErrorCategory.COMPONENT
        elif 'integration' in context or 'api' in context or 'external' in context:
            return ErrorCategory.INTEGRATION
        elif 'workflow' in context:
            return ErrorCategory.WORKFLOW
        elif 'data' in context or 'database' in context or 'storage' in context:
            return ErrorCategory.DATA
        elif 'user' in context or 'input' in context:
            return ErrorCategory.USER
        else:
            return ErrorCategory.SYSTEM

    def _determine_severity(self, error: Exception, category: ErrorCategory, 
                          context: Dict[str, Any]) -> ErrorSeverity:
        """Determine error severity"""
        # Critical system errors
        if isinstance(error, (MemoryError, SystemExit, KeyboardInterrupt)):
            return ErrorSeverity.CRITICAL
        
        # Network and integration errors as high severity for now
        if category == ErrorCategory.INTEGRATION:
            return ErrorSeverity.HIGH
        
        # Workflow failures
        if category == ErrorCategory.WORKFLOW:
            return ErrorSeverity.MEDIUM
        
        # Data issues
        if category == ErrorCategory.DATA:
            return ErrorSeverity.MEDIUM
        
        # User input issues
        if category == ErrorCategory.USER:
            return ErrorSeverity.LOW
        
        # Default to medium
        return ErrorSeverity.MEDIUM

    def _determine_recovery_strategy(self, error_entry: ErrorEntry, 
                                   context: Dict[str, Any]) -> RecoveryStrategy:
        """Determine the best recovery strategy"""
        # Check for specific handlers
        if error_entry.error_type in self.error_handlers:
            return RecoveryStrategy.RETRY
        
        # Check retry policy
        operation = context.get('operation')
        if operation and operation in self.retry_policies:
            return RecoveryStrategy.RETRY
        
        # Check fallback solutions
        if operation and operation in self.fallback_solutions:
            return RecoveryStrategy.FALLBACK
        
        # High severity errors escalate
        if error_entry.severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.ESCALATE
        
        # User errors can usually be ignored or escalated
        if error_entry.component == 'user':
            return RecoveryStrategy.IGNORE
        
        # Default to retry
        return RecoveryStrategy.RETRY

    async def _execute_recovery_strategy(self, strategy: RecoveryStrategy, 
                                       error_entry: ErrorEntry, 
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a recovery strategy"""
        try:
            if strategy == RecoveryStrategy.RETRY:
                return {
                    'handled': True,
                    'method': 'retry',
                    'should_retry': True
                }
            
            elif strategy == RecoveryStrategy.FALLBACK:
                operation = context.get('operation', 'unknown')
                fallback_solution = await self.get_fallback_solution(operation, context)
                return {
                    'handled': True,
                    'method': 'fallback',
                    'fallback_solution': fallback_solution,
                    'should_retry': False
                }
            
            elif strategy == RecoveryStrategy.DEGRADE:
                return {
                    'handled': True,
                    'method': 'degrade',
                    'should_retry': False
                }
            
            elif strategy == RecoveryStrategy.ESCALATE:
                return {
                    'handled': False,
                    'method': 'escalate',
                    'should_retry': False
                }
            
            elif strategy == RecoveryStrategy.IGNORE:
                return {
                    'handled': True,
                    'method': 'ignore',
                    'should_retry': False
                }
            
            else:
                return {
                    'handled': False,
                    'method': 'unknown',
                    'should_retry': False
                }

        except Exception as e:
            logger.error(f"Recovery strategy execution failed: {e}")
            return {
                'handled': False,
                'method': 'failed',
                'error': str(e),
                'should_retry': False
            }

    def _generate_user_message(self, error_entry: ErrorEntry, 
                             recovery_result: Dict[str, Any]) -> str:
        """Generate appropriate user-facing error message"""
        if recovery_result['should_retry']:
            return f"An error occurred, but we can try again. {error_entry.message}"
        
        if recovery_result.get('fallback_solution'):
            return f"Operation completed with alternative approach. {error_entry.message}"
        
        if error_entry.severity == ErrorSeverity.LOW:
            return "A minor issue occurred, but the operation can continue."
        
        elif error_entry.severity == ErrorSeverity.MEDIUM:
            return f"An issue occurred: {error_entry.message}"
        
        elif error_entry.severity == ErrorSeverity.HIGH:
            return f"A serious error occurred: {error_entry.message}. The operation was stopped."
        
        else:  # CRITICAL
            return f"A critical system error occurred. Please contact support."

    def _store_error(self, error_entry: ErrorEntry) -> None:
        """Store error entry in history"""
        self.error_history.append(error_entry)
        
        # Trim history if needed
        if len(self.error_history) > self.max_error_history:
            self.error_history = self.error_history[-self.max_error_history:]

    def _update_error_stats(self, error_entry: ErrorEntry) -> None:
        """Update error statistics"""
        component = error_entry.component
        self.error_counts[component] = self.error_counts.get(component, 0) + 1
        
        # Simple error rate calculation (errors per hour)
        recent_errors = [
            entry for entry in self.error_history
            if (datetime.utcnow() - entry.timestamp).total_seconds() < 3600
            and entry.component == component
        ]
        self.error_rates[component] = len(recent_errors) / 3600  # errors per second

    def _check_circuit_breakers(self, component: str) -> bool:
        """Check if circuit breaker is activated for a component"""
        if component not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[component]
        if not breaker['active']:
            return False
        
        # Check if timeout has passed
        if datetime.utcnow() > breaker['timeout']:
            breaker['active'] = False
            return False
        
        return True

    def _get_active_circuit_breakers(self) -> List[str]:
        """Get list of components with active circuit breakers"""
        active_breakers = []
        for component, breaker in self.circuit_breakers.items():
            if breaker.get('active', False):
                active_breakers.append(component)
        return active_breakers