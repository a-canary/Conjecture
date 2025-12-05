#!/usr/bin/env python3
"""
Comprehensive Retry Statistics and Error Tracking System for Conjecture
Tracks retry patterns, error rates, and provides detailed analytics
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import threading
import uuid

logger = logging.getLogger(__name__)


@dataclass
class RetryEvent:
    """Individual retry event data"""
    event_id: str
    operation_type: str
    operation_id: str
    timestamp: datetime
    attempt_number: int
    max_attempts: int
    error_type: str
    error_message: str
    error_category: str  # "network", "timeout", "rate_limit", "model_error", "system"
    retry_delay_ms: int
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time_ms: Optional[int] = None


@dataclass
class ErrorPattern:
    """Pattern analysis for errors"""
    error_type: str
    error_category: str
    frequency: int
    first_occurrence: datetime
    last_occurrence: datetime
    avg_resolution_time_ms: float
    success_rate_after_retry: float
    common_contexts: List[Dict[str, Any]]
    recommendations: List[str]


@dataclass
class RetryStatistics:
    """Comprehensive retry statistics"""
    operation_type: str
    total_operations: int
    successful_without_retry: int
    successful_with_retry: int
    failed_operations: int
    total_retry_attempts: int
    average_retries_per_operation: float
    max_retries_for_single_operation: int
    retry_success_rate: float
    overall_success_rate: float
    error_breakdown: Dict[str, int]
    retry_patterns: Dict[str, Any]
    time_to_resolution: Dict[str, float]


class RetryTracker:
    """Advanced retry tracking and error analysis system"""
    
    def __init__(self, 
                 max_events: int = 10000,
                 analysis_interval_minutes: int = 5,
                 save_interval_minutes: int = 30):
        
        self.max_events = max_events
        self.analysis_interval = timedelta(minutes=analysis_interval_minutes)
        self.save_interval = timedelta(minutes=save_interval_minutes)
        
        # Data storage
        self.retry_events: deque = deque(maxlen=max_events)
        self.active_operations: Dict[str, Dict[str, Any]] = {}
        self.error_patterns: Dict[str, ErrorPattern] = {}
        self.retry_statistics: Dict[str, RetryStatistics] = {}
        
        # Analysis cache
        self._analysis_cache: Dict[str, Any] = {}
        self._last_analysis_time = datetime.utcnow()
        
        # Background tasks
        self._analysis_active = False
        self._analysis_task: Optional[asyncio.Task] = None
        self._save_task: Optional[asyncio.Task] = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Configuration
        self.error_categories = {
            'network': [
                'ConnectionError', 'TimeoutError', 'NetworkError', 
                'HTTPError', 'RequestException'
            ],
            'timeout': [
                'TimeoutError', 'AsyncTimeoutError', 'ReadTimeout',
                'ConnectTimeout', 'RequestTimeout'
            ],
            'rate_limit': [
                'RateLimitError', 'TooManyRequests', 'QuotaExceeded',
                'ThrottlingException', 'RateLimitException'
            ],
            'model_error': [
                'ModelError', 'InvalidResponse', 'ParseError',
                'ValidationError', 'ModelUnavailable'
            ],
            'system': [
                'MemoryError', 'CPUError', 'DiskError',
                'SystemError', 'ResourceError'
            ],
            'unknown': []  # Catch-all for uncategorized errors
        }
        
        # Output directory
        self.output_dir = Path("research/metrics/retry_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def start_operation(self, 
                     operation_type: str, 
                     operation_id: Optional[str] = None,
                     max_attempts: int = 3,
                     context: Optional[Dict[str, Any]] = None) -> str:
        """Start tracking a new operation"""
        
        if operation_id is None:
            operation_id = str(uuid.uuid4())
        
        with self._lock:
            self.active_operations[operation_id] = {
                'operation_type': operation_type,
                'start_time': datetime.utcnow(),
                'max_attempts': max_attempts,
                'attempt_count': 1,
                'context': context or {},
                'events': []
            }
        
        logger.debug(f"Started tracking operation {operation_id} of type {operation_type}")
        return operation_id
    
    def record_attempt(self, 
                    operation_id: str, 
                    success: bool = False,
                    error_type: Optional[str] = None,
                    error_message: Optional[str] = None,
                    retry_delay_ms: int = 0) -> bool:
        """Record an attempt for an operation"""
        
        with self._lock:
            if operation_id not in self.active_operations:
                logger.warning(f"Operation {operation_id} not found in active operations")
                return False
            
            operation = self.active_operations[operation_id]
            attempt_number = operation['attempt_count']
            
            if success:
                # Operation completed successfully
                end_time = datetime.utcnow()
                resolution_time_ms = int((end_time - operation['start_time']).total_seconds() * 1000)
                
                # Create retry event for final successful attempt
                retry_event = RetryEvent(
                    event_id=str(uuid.uuid4()),
                    operation_type=operation['operation_type'],
                    operation_id=operation_id,
                    timestamp=end_time,
                    attempt_number=attempt_number,
                    max_attempts=operation['max_attempts'],
                    error_type="",
                    error_message="",
                    error_category="",
                    retry_delay_ms=retry_delay_ms,
                    context=operation['context'],
                    resolved=True,
                    resolution_time_ms=resolution_time_ms
                )
                
                self.retry_events.append(retry_event)
                operation['events'].append(retry_event)
                
                # Remove from active operations
                del self.active_operations[operation_id]
                
                logger.debug(f"Operation {operation_id} completed successfully on attempt {attempt_number}")
                return True
            
            else:
                # Operation failed, record error
                error_category = self._categorize_error(error_type or "UnknownError")
                
                retry_event = RetryEvent(
                    event_id=str(uuid.uuid4()),
                    operation_type=operation['operation_type'],
                    operation_id=operation_id,
                    timestamp=datetime.utcnow(),
                    attempt_number=attempt_number,
                    max_attempts=operation['max_attempts'],
                    error_type=error_type or "UnknownError",
                    error_message=error_message or "Unknown error occurred",
                    error_category=error_category,
                    retry_delay_ms=retry_delay_ms,
                    context=operation['context'],
                    resolved=False
                )
                
                self.retry_events.append(retry_event)
                operation['events'].append(retry_event)
                
                # Check if we should retry
                operation['attempt_count'] += 1
                
                if attempt_number >= operation['max_attempts']:
                    # Max attempts reached, operation failed
                    end_time = datetime.utcnow()
                    
                    # Create final retry event for failed operation
                    final_event = RetryEvent(
                        event_id=str(uuid.uuid4()),
                        operation_type=operation['operation_type'],
                        operation_id=operation_id,
                        timestamp=end_time,
                        attempt_number=attempt_number,
                        max_attempts=operation['max_attempts'],
                        error_type=error_type or "UnknownError",
                        error_message=error_message or "Operation failed after all retries",
                        error_category=error_category,
                        retry_delay_ms=0,
                        context=operation['context'],
                        resolved=False
                    )
                    
                    self.retry_events.append(final_event)
                    
                    # Remove from active operations
                    del self.active_operations[operation_id]
                    
                    logger.warning(f"Operation {operation_id} failed after {attempt_number} attempts")
                    return False
                else:
                    logger.debug(f"Operation {operation_id} failed on attempt {attempt_number}, will retry")
                    return True
    
    def _categorize_error(self, error_type: str) -> str:
        """Categorize error type"""
        error_lower = error_type.lower()
        
        for category, error_list in self.error_categories.items():
            for error_pattern in error_list:
                if error_pattern.lower() in error_lower:
                    return category
        
        return 'unknown'
    
    def get_retry_statistics(self, 
                           operation_type: Optional[str] = None,
                           time_window_minutes: int = 60) -> Dict[str, RetryStatistics]:
        """Get retry statistics for operations"""
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)
        
        # Filter recent events
        recent_events = [
            event for event in self.retry_events 
            if event.timestamp >= cutoff_time
        ]
        
        # Filter by operation type if specified
        if operation_type:
            recent_events = [
                event for event in recent_events 
                if event.operation_type == operation_type
            ]
        
        # Group by operation type
        operation_groups = defaultdict(list)
        for event in recent_events:
            operation_groups[event.operation_type].append(event)
        
        # Calculate statistics for each operation type
        stats = {}
        
        for op_type, events in operation_groups.items():
            # Basic counts
            total_operations = len(set(event.operation_id for event in events))
            successful_operations = len(set(
                event.operation_id for event in events 
                if event.resolved
            ))
            failed_operations = total_operations - successful_operations
            
            # Retry analysis
            operation_retry_counts = defaultdict(int)
            for event in events:
                operation_retry_counts[event.operation_id] += 1
            
            total_retries = sum(count - 1 for count in operation_retry_counts.values() if count > 1)
            successful_with_retry = len([
                op_id for op_id, count in operation_retry_counts.items() 
                if count > 1 and any(e.resolved for e in events if e.operation_id == op_id)
            ])
            
            successful_without_retry = successful_operations - successful_with_retry
            
            # Calculate averages
            avg_retries = total_retries / total_operations if total_operations > 0 else 0
            max_retries = max(operation_retry_counts.values()) if operation_retry_counts else 0
            
            retry_success_rate = successful_with_retry / (successful_with_retry + failed_operations) if (successful_with_retry + failed_operations) > 0 else 0
            overall_success_rate = successful_operations / total_operations if total_operations > 0 else 0
            
            # Error breakdown
            error_breakdown = defaultdict(int)
            for event in events:
                if not event.resolved:
                    error_breakdown[event.error_type] += 1
            
            # Time to resolution analysis
            resolution_times = [
                event.resolution_time_ms for event in events 
                if event.resolved and event.resolution_time_ms
            ]
            
            avg_resolution_time = sum(resolution_times) / len(resolution_times) if resolution_times else 0
            median_resolution_time = sorted(resolution_times)[len(resolution_times) // 2] if resolution_times else 0
            
            # Retry patterns
            retry_patterns = {
                'average_retries': avg_retries,
                'max_retries': max_retries,
                'retry_distribution': dict(operation_retry_counts)
            }
            
            stats[op_type] = RetryStatistics(
                operation_type=op_type,
                total_operations=total_operations,
                successful_without_retry=successful_without_retry,
                successful_with_retry=successful_with_retry,
                failed_operations=failed_operations,
                total_retry_attempts=total_retries,
                average_retries_per_operation=avg_retries,
                max_retries_for_single_operation=max_retries,
                retry_success_rate=retry_success_rate,
                overall_success_rate=overall_success_rate,
                error_breakdown=dict(error_breakdown),
                retry_patterns=retry_patterns,
                time_to_resolution={
                    'average_ms': avg_resolution_time,
                    'median_ms': median_resolution_time
                }
            )
        
        return stats
    
    def analyze_error_patterns(self, 
                           time_window_minutes: int = 60) -> Dict[str, ErrorPattern]:
        """Analyze error patterns to identify issues"""
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)
        
        # Filter recent error events
        error_events = [
            event for event in self.retry_events 
            if event.timestamp >= cutoff_time and not event.resolved
        ]
        
        # Group by error type
        error_groups = defaultdict(list)
        for event in error_events:
            error_groups[event.error_type].append(event)
        
        # Analyze patterns for each error type
        patterns = {}
        
        for error_type, events in error_groups.items():
            if not events:
                continue
            
            # Basic frequency analysis
            frequency = len(events)
            first_occurrence = min(event.timestamp for event in events)
            last_occurrence = max(event.timestamp for event in events)
            
            # Resolution analysis
            related_resolutions = [
                event for event in self.retry_events 
                if event.error_type == error_type and event.resolved
            ]
            
            if related_resolutions:
                avg_resolution_time = sum(
                    event.resolution_time_ms for event in related_resolutions
                ) / len(related_resolutions)
                
                success_after_retry = len(related_resolutions)
                total_related_attempts = len(set(event.operation_id for event in related_resolutions))
                success_rate = success_after_retry / total_related_attempts if total_related_attempts > 0 else 0
            else:
                avg_resolution_time = 0
                success_rate = 0
            
            # Context analysis
            context_patterns = defaultdict(int)
            for event in events:
                # Extract key context patterns
                for key, value in event.context.items():
                    if isinstance(value, str):
                        context_patterns[f"{key}:{value}"] += 1
                    elif isinstance(value, (int, float, bool)):
                        context_patterns[f"{key}:{value}"] += 1
            
            # Get most common contexts
            common_contexts = [
                {'pattern': pattern, 'count': count}
                for pattern, count in sorted(context_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
            ]
            
            # Generate recommendations
            recommendations = self._generate_error_recommendations(
                error_type, events, success_rate
            )
            
            # Determine error category
            error_category = self._categorize_error(error_type)
            
            patterns[error_type] = ErrorPattern(
                error_type=error_type,
                error_category=error_category,
                frequency=frequency,
                first_occurrence=first_occurrence,
                last_occurrence=last_occurrence,
                avg_resolution_time_ms=avg_resolution_time,
                success_rate_after_retry=success_rate,
                common_contexts=common_contexts,
                recommendations=recommendations
            )
        
        return patterns
    
    def _generate_error_recommendations(self, 
                                   error_type: str, 
                                   events: List[RetryEvent], 
                                   success_rate: float) -> List[str]:
        """Generate recommendations based on error patterns"""
        
        recommendations = []
        
        # Low success rate recommendations
        if success_rate < 0.3:
            recommendations.append(f"Very low success rate ({success_rate:.1%}) for {error_type} - consider alternative approaches")
        elif success_rate < 0.6:
            recommendations.append(f"Low success rate ({success_rate:.1%}) for {error_type} - investigate root causes")
        
        # Category-specific recommendations
        error_category = self._categorize_error(error_type)
        
        if error_category == 'network':
            recommendations.extend([
                "Check network connectivity and stability",
                "Consider implementing circuit breaker pattern",
                "Add exponential backoff for retries"
            ])
        elif error_category == 'timeout':
            recommendations.extend([
                "Increase timeout values for slow operations",
                "Implement timeout monitoring and alerts",
                "Consider breaking down large operations"
            ])
        elif error_category == 'rate_limit':
            recommendations.extend([
                "Implement rate limiting on client side",
                "Add request queuing for burst handling",
                "Monitor API quota usage"
            ])
        elif error_category == 'model_error':
            recommendations.extend([
                "Validate input data before sending to model",
                "Add model response parsing error handling",
                "Consider fallback models for reliability"
            ])
        elif error_category == 'system':
            recommendations.extend([
                "Monitor system resources (memory, CPU)",
                "Implement resource cleanup procedures",
                "Consider scaling system resources"
            ])
        
        # Frequency-based recommendations
        if len(events) > 10:  # High frequency error
            recommendations.append(f"High frequency error ({len(events)} occurrences) - prioritize fixing this issue")
        
        # Time-based recommendations
        recent_events = [e for e in events if e.timestamp > datetime.utcnow() - timedelta(hours=1)]
        if len(recent_events) > len(events) * 0.5:  # More than half in last hour
            recommendations.append("Error frequency increasing - urgent attention needed")
        
        return recommendations
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time retry and error metrics"""
        
        current_time = datetime.utcnow()
        last_5_minutes = current_time - timedelta(minutes=5)
        last_hour = current_time - timedelta(hours=1)
        
        # Filter events by time windows
        recent_5min = [e for e in self.retry_events if e.timestamp >= last_5_minutes]
        recent_hour = [e for e in self.retry_events if e.timestamp >= last_hour]
        
        # Active operations
        active_count = len(self.active_operations)
        
        # Error rates
        errors_5min = [e for e in recent_5min if not e.resolved]
        errors_hour = [e for e in recent_hour if not e.resolved]
        
        error_rate_5min = len(errors_5min) / max(len(recent_5min), 1)
        error_rate_hour = len(errors_hour) / max(len(recent_hour), 1)
        
        # Most common errors
        error_counts_5min = defaultdict(int)
        for error in errors_5min:
            error_counts_5min[error.error_type] += 1
        
        top_errors_5min = sorted(
            error_counts_5min.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        return {
            'timestamp': current_time.isoformat(),
            'active_operations': active_count,
            'error_rates': {
                'last_5_minutes': error_rate_5min,
                'last_hour': error_rate_hour
            },
            'top_errors': {
                'last_5_minutes': [
                    {'error_type': error, 'count': count}
                    for error, count in top_errors_5min
                ]
            },
            'total_events_tracked': len(self.retry_events),
            'analysis_status': 'active' if self._analysis_active else 'idle'
        }
    
    def start_background_analysis(self):
        """Start background analysis and saving tasks"""
        if self._analysis_active:
            return
        
        self._analysis_active = True
        
        # Start analysis task
        self._analysis_task = asyncio.create_task(self._analysis_loop())
        
        # Start save task
        self._save_task = asyncio.create_task(self._save_loop())
        
        logger.info("Started background retry analysis and saving")
    
    def stop_background_analysis(self):
        """Stop background analysis and saving tasks"""
        self._analysis_active = False
        
        if self._analysis_task:
            self._analysis_task.cancel()
        
        if self._save_task:
            self._save_task.cancel()
        
        logger.info("Stopped background retry analysis and saving")
    
    async def _analysis_loop(self):
        """Background loop for periodic analysis"""
        while self._analysis_active:
            try:
                await asyncio.sleep(self.analysis_interval.total_seconds())
                
                # Perform analysis
                await self._perform_periodic_analysis()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
    
    async def _save_loop(self):
        """Background loop for periodic saving"""
        while self._analysis_active:
            try:
                await asyncio.sleep(self.save_interval.total_seconds())
                
                # Save data
                await self._save_retry_data()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in save loop: {e}")
    
    async def _perform_periodic_analysis(self):
        """Perform periodic analysis of retry data"""
        
        # Update retry statistics
        all_stats = self.get_retry_statistics()
        self.retry_statistics.update(all_stats)
        
        # Update error patterns
        error_patterns = self.analyze_error_patterns()
        self.error_patterns.update(error_patterns)
        
        self._last_analysis_time = datetime.utcnow()
        
        logger.debug(f"Updated retry analysis for {len(all_stats)} operation types")
    
    async def _save_retry_data(self):
        """Save retry data to files"""
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Save retry events
        events_file = self.output_dir / f"retry_events_{timestamp}.json"
        events_data = []
        
        for event in self.retry_events:
            event_dict = {
                'event_id': event.event_id,
                'operation_type': event.operation_type,
                'operation_id': event.operation_id,
                'timestamp': event.timestamp.isoformat(),
                'attempt_number': event.attempt_number,
                'max_attempts': event.max_attempts,
                'error_type': event.error_type,
                'error_message': event.error_message,
                'error_category': event.error_category,
                'retry_delay_ms': event.retry_delay_ms,
                'context': event.context,
                'resolved': event.resolved,
                'resolution_time_ms': event.resolution_time_ms
            }
            events_data.append(event_dict)
        
        with open(events_file, 'w', encoding='utf-8') as f:
            json.dump(events_data, f, indent=2)
        
        # Save statistics summary
        stats_file = self.output_dir / f"retry_statistics_{timestamp}.json"
        stats_data = {
            'timestamp': timestamp,
            'retry_statistics': {
                op_type: {
                    'total_operations': stats.total_operations,
                    'successful_without_retry': stats.successful_without_retry,
                    'successful_with_retry': stats.successful_with_retry,
                    'failed_operations': stats.failed_operations,
                    'total_retry_attempts': stats.total_retry_attempts,
                    'average_retries_per_operation': stats.average_retries_per_operation,
                    'max_retries_for_single_operation': stats.max_retries_for_single_operation,
                    'retry_success_rate': stats.retry_success_rate,
                    'overall_success_rate': stats.overall_success_rate,
                    'error_breakdown': stats.error_breakdown,
                    'time_to_resolution': stats.time_to_resolution
                }
                for op_type, stats in self.retry_statistics.items()
            },
            'error_patterns': {
                error_type: {
                    'error_type': pattern.error_type,
                    'error_category': pattern.error_category,
                    'frequency': pattern.frequency,
                    'first_occurrence': pattern.first_occurrence.isoformat(),
                    'last_occurrence': pattern.last_occurrence.isoformat(),
                    'avg_resolution_time_ms': pattern.avg_resolution_time_ms,
                    'success_rate_after_retry': pattern.success_rate_after_retry,
                    'common_contexts': pattern.common_contexts,
                    'recommendations': pattern.recommendations
                }
                for error_type, pattern in self.error_patterns.items()
            }
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2)
        
        logger.debug(f"Saved retry data to {events_file} and {stats_file}")
    
    def generate_retry_report(self) -> str:
        """Generate comprehensive retry analysis report"""
        
        real_time_metrics = self.get_real_time_metrics()
        all_stats = self.get_retry_statistics(time_window_minutes=60)
        error_patterns = self.analyze_error_patterns(time_window_minutes=60)
        
        report_lines = [
            "# Retry Analysis Report",
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Events Tracked: {len(self.retry_events)}",
            f"Active Operations: {real_time_metrics['active_operations']}",
            "",
            "## Real-Time Metrics",
            "",
            f"- **Error Rate (last 5 min)**: {real_time_metrics['error_rates']['last_5_minutes']:.2%}",
            f"- **Error Rate (last hour)**: {real_time_metrics['error_rates']['last_hour']:.2%}",
            "",
            "## Top Recent Errors",
            ""
        ]
        
        for error_info in real_time_metrics['top_errors']['last_5_minutes']:
            report_lines.extend([
                f"- **{error_info['error_type']}**: {error_info['count']} occurrences",
                ""
            ])
        
        report_lines.extend([
            "## Operation Type Statistics",
            "",
            "| Operation Type | Total | Success Rate | Avg Retries | Error Rate |",
            "|---------------|-------|-------------|-------------|-----------|"
        ])
        
        for op_type, stats in all_stats.items():
            report_lines.append(
                f"| {op_type} | {stats.total_operations} | {stats.overall_success_rate:.1%} | "
                f"{stats.average_retries_per_operation:.1f} | {stats.failed_operations/stats.total_operations:.1%} |"
            )
        
        report_lines.extend([
            "",
            "## Error Pattern Analysis",
            ""
        ])
        
        for error_type, pattern in error_patterns.items():
            report_lines.extend([
                f"### {error_type}",
                f"- **Category**: {pattern.error_category}",
                f"- **Frequency**: {pattern.frequency} occurrences",
                f"- **Success Rate After Retry**: {pattern.success_rate_after_retry:.1%}",
                f"- **Recommendations**: {', '.join(pattern.recommendations)}",
                ""
            ])
        
        return "\n".join(report_lines)


# Utility functions for integration
def create_retry_tracker(max_events: int = 10000) -> RetryTracker:
    """Create and initialize a retry tracker"""
    return RetryTracker(max_events=max_events)


# Context manager for retry tracking
class RetryContext:
    """Context manager for automatic retry tracking"""
    
    def __init__(self, 
                 retry_tracker: RetryTracker,
                 operation_type: str,
                 max_attempts: int = 3,
                 context: Optional[Dict[str, Any]] = None):
        self.retry_tracker = retry_tracker
        self.operation_type = operation_type
        self.max_attempts = max_attempts
        self.context = context or {}
        self.operation_id = None
        self.success = False
    
    def __enter__(self):
        self.operation_id = self.retry_tracker.start_operation(
            self.operation_type, 
            max_attempts=self.max_attempts,
            context=self.context
        )
        return self
    
    def record_success(self):
        """Record successful operation"""
        self.success = True
        self.retry_tracker.record_attempt(self.operation_id, success=True)
    
    def record_failure(self, error_type: str, error_message: str = None):
        """Record failed operation attempt"""
        self.success = False
        self.retry_tracker.record_attempt(
            self.operation_id, 
            success=False, 
            error_type=error_type, 
            error_message=error_message
        )
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Exception occurred, record as failure
            self.retry_tracker.record_attempt(
                self.operation_id,
                success=False,
                error_type=exc_type.__name__,
                error_message=str(exc_val)
            )
        elif not self.success:
            # No exception but no success recorded, record as failure
            self.retry_tracker.record_attempt(
                self.operation_id,
                success=False,
                error_type="UnknownError",
                error_message="Operation failed without explicit success"
            )
        
        return False  # Don't suppress exceptions