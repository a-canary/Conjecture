"""
OPTIMIZED: Comprehensive Performance Monitoring System for Conjecture
Tracks timing, cache performance, resource usage, and system health
"""

import asyncio
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import logging
import functools

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric data point"""
    name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceSnapshot:
    """Snapshot of performance metrics at a point in time"""
    timestamp: datetime
    metrics: Dict[str, PerformanceMetric] = field(default_factory=dict)
    system_info: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """
    OPTIMIZED: Comprehensive performance monitoring with real-time tracking
    """
    
    def __init__(
        self,
        max_history_size: int = 10000,
        snapshot_interval: int = 60,  # seconds
        enable_system_monitoring: bool = True
    ):
        self.max_history_size = max_history_size
        self.snapshot_interval = snapshot_interval
        self.enable_system_monitoring = enable_system_monitoring
        
        # Performance data storage
        self._metrics_history: deque = deque(maxlen=max_history_size)
        self._snapshots: deque = deque(maxlen=1000)  # Keep last 1000 snapshots
        self._active_timers: Dict[str, float] = {}
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = defaultdict(float)
        
        # Performance tracking
        self._performance_stats = {
            "total_operations": 0,
            "total_time": 0.0,
            "average_time": 0.0,
            "min_time": float('inf'),
            "max_time": 0.0,
            "operations_per_second": 0.0,
        }
        
        # Background monitoring
        self._monitoring_active = False
        self._snapshot_task: Optional[asyncio.Task] = None
        self._system_monitor_thread: Optional[threading.Thread] = None
        
        # Callbacks for custom monitoring
        self._metric_callbacks: List[Callable[[PerformanceMetric], None]] = []
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
    def start_monitoring(self):
        """Start performance monitoring"""
        if self._monitoring_active:
            return
            
        self._monitoring_active = True
        
        # Start background snapshot task only if there's an event loop
        try:
            self._snapshot_task = asyncio.create_task(self._snapshot_loop())
        except RuntimeError as e:
            if "no running event loop" in str(e):
                # No event loop available, skip snapshot task
                logger.warning("No event loop available, skipping background snapshot task")
                self._snapshot_task = None
            else:
                raise
        
        # Start system monitoring thread
        if self.enable_system_monitoring:
            self._system_monitor_thread = threading.Thread(
                target=self._system_monitor_loop,
                daemon=True
            )
            self._system_monitor_thread.start()
            
        logger.info("Performance monitoring started")
        
    def stop_monitoring(self):
        """Stop performance monitoring"""
        if not self._monitoring_active:
            return
            
        self._monitoring_active = False
        
        # Stop background tasks
        if self._snapshot_task:
            self._snapshot_task.cancel()
            
        if self._system_monitor_thread and self._system_monitor_thread.is_alive():
            # Thread will exit when _monitoring_active is False
            self._system_monitor_thread.join(timeout=5)
            
        logger.info("Performance monitoring stopped")
        
    def timer(self, operation_name: str, tags: Optional[Dict[str, str]] = None):
        """Decorator for timing operations"""
        def decorator(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    execution_time = time.time() - start_time
                    self.record_timing(operation_name, execution_time, tags or {})
                    
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    execution_time = time.time() - start_time
                    self.record_timing(operation_name, execution_time, tags or {})
                    
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        return decorator
        
    def start_timer(self, operation_name: str) -> str:
        """Start timing an operation and return timer ID"""
        timer_id = f"{operation_name}_{int(time.time() * 1000000)}"
        with self._lock:
            self._active_timers[timer_id] = time.time()
        return timer_id
        
    def end_timer(self, timer_id: str, tags: Optional[Dict[str, str]] = None):
        """End a timer and record the duration"""
        end_time = time.time()
        with self._lock:
            if timer_id not in self._active_timers:
                logger.warning(f"Timer {timer_id} not found")
                return
                
            start_time = self._active_timers.pop(timer_id)
            duration = end_time - start_time
            
        # Extract operation name from timer ID
        operation_name = timer_id.rsplit('_', 1)[0]
        self.record_timing(operation_name, duration, tags or {})
        
    def record_timing(self, operation_name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record a timing metric"""
        metric = PerformanceMetric(
            name=f"{operation_name}_duration",
            value=duration,
            unit="seconds",
            tags=tags or {}
        )
        
        self._add_metric(metric)
        self._update_performance_stats(duration)
        
        # Notify callbacks
        for callback in self._metric_callbacks:
            try:
                callback(metric)
            except Exception as e:
                logger.error(f"Error in metric callback: {e}")
                
    def increment_counter(self, counter_name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        with self._lock:
            self._counters[counter_name] += value
            
        metric = PerformanceMetric(
            name=counter_name,
            value=self._counters[counter_name],
            unit="count",
            tags=tags or {}
        )
        self._add_metric(metric)
        
    def set_gauge(self, gauge_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric"""
        with self._lock:
            self._gauges[gauge_name] = value
            
        metric = PerformanceMetric(
            name=gauge_name,
            value=value,
            unit="gauge",
            tags=tags or {}
        )
        self._add_metric(metric)
        
    def record_cache_hit(self, cache_name: str, hit_type: str = "hit"):
        """Record a cache hit"""
        self.increment_counter(
            f"{cache_name}_cache_{hit_type}",
            tags={"cache": cache_name, "type": hit_type}
        )
        
    def record_cache_miss(self, cache_name: str, miss_type: str = "miss"):
        """Record a cache miss"""
        self.increment_counter(
            f"{cache_name}_cache_{miss_type}",
            tags={"cache": cache_name, "type": miss_type}
        )
        
    def record_database_operation(self, operation: str, duration: float, affected_rows: int = 0):
        """Record a database operation"""
        self.record_timing(f"db_{operation}", duration, {"affected_rows": str(affected_rows)})
        
    def record_llm_operation(self, operation: str, duration: float, tokens_used: int = 0):
        """Record an LLM operation"""
        self.record_timing(
            f"llm_{operation}", 
            duration, 
            {"tokens_used": str(tokens_used)}
        )
        
    def _add_metric(self, metric: PerformanceMetric):
        """Add a metric to history"""
        with self._lock:
            self._metrics_history.append(metric)
            
    def _update_performance_stats(self, duration: float):
        """Update overall performance statistics"""
        with self._lock:
            self._performance_stats["total_operations"] += 1
            self._performance_stats["total_time"] += duration
            self._performance_stats["average_time"] = (
                self._performance_stats["total_time"] / 
                self._performance_stats["total_operations"]
            )
            self._performance_stats["min_time"] = min(
                self._performance_stats["min_time"], duration
            )
            self._performance_stats["max_time"] = max(
                self._performance_stats["max_time"], duration
            )
            
    async def _snapshot_loop(self):
        """Background loop for taking performance snapshots"""
        while self._monitoring_active:
            try:
                await asyncio.sleep(self.snapshot_interval)
                await self._take_snapshot()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in snapshot loop: {e}")
                
    async def _take_snapshot(self):
        """Take a performance snapshot"""
        try:
            timestamp = datetime.utcnow()
            
            # Get current system info
            system_info = {}
            if self.enable_system_monitoring:
                system_info = self._get_system_info()
                
            # Get current gauge values
            current_gauges = dict(self._gauges)
            
            # Create snapshot
            snapshot = PerformanceSnapshot(
                timestamp=timestamp,
                metrics={name: PerformanceMetric(name=name, value=value, unit="gauge") 
                         for name, value in current_gauges.items()},
                system_info=system_info
            )
            
            with self._lock:
                self._snapshots.append(snapshot)
                
        except Exception as e:
            logger.error(f"Error taking snapshot: {e}")
            
    def _system_monitor_loop(self):
        """Background thread for system monitoring"""
        while self._monitoring_active:
            try:
                time.sleep(30)  # Check every 30 seconds
                if not self._monitoring_active:
                    break
                    
                # Record system metrics
                system_info = self._get_system_info()
                
                # CPU usage
                self.set_gauge("system_cpu_percent", system_info["cpu_percent"])
                
                # Memory usage
                self.set_gauge("system_memory_percent", system_info["memory_percent"])
                self.set_gauge("system_memory_used_mb", system_info["memory_used_mb"])
                
                # Disk usage
                self.set_gauge("system_disk_percent", system_info["disk_percent"])
                
            except Exception as e:
                logger.error(f"Error in system monitor loop: {e}")
                
    def _get_system_info(self) -> Dict[str, Any]:
        """Get current system information"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_used_mb": memory_used_mb,
                "memory_total_mb": memory.total / (1024 * 1024),
                "disk_percent": disk_percent,
                "disk_used_gb": disk.used / (1024 * 1024 * 1024),
                "disk_total_gb": disk.total / (1024 * 1024 * 1024),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {}
            
    def get_performance_summary(self, time_window_minutes: int = 5) -> Dict[str, Any]:
        """Get performance summary for a time window"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)
        
        with self._lock:
            # Filter metrics within time window
            recent_metrics = [
                metric for metric in self._metrics_history 
                if metric.timestamp >= cutoff_time
            ]
            
            # Calculate summary statistics
            summary = {
                "time_window_minutes": time_window_minutes,
                "total_metrics": len(recent_metrics),
                "performance_stats": dict(self._performance_stats),
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "recent_snapshots": len([
                    snapshot for snapshot in self._snapshots 
                    if snapshot.timestamp >= cutoff_time
                ])
            }
            
            # Add operation breakdown
            operation_times = defaultdict(list)
            for metric in recent_metrics:
                if metric.name.endswith('_duration'):
                    operation_name = metric.name[:-9]  # Remove '_duration'
                    operation_times[operation_name].append(metric.value)
                    
            summary["operation_breakdown"] = {}
            for operation, times in operation_times.items():
                if times:
                    summary["operation_breakdown"][operation] = {
                        "count": len(times),
                        "average": sum(times) / len(times),
                        "min": min(times),
                        "max": max(times),
                        "total": sum(times)
                    }
                    
            return summary
            
    def get_cache_performance(self, cache_name: str) -> Dict[str, Any]:
        """Get cache performance statistics"""
        with self._lock:
            hits = self._counters.get(f"{cache_name}_cache_hit", 0)
            misses = self._counters.get(f"{cache_name}_cache_miss", 0)
            total = hits + misses
            
            return {
                "cache_name": cache_name,
                "hits": hits,
                "misses": misses,
                "total": total,
                "hit_rate": (hits / total * 100) if total > 0 else 0.0
            }
            
    def add_metric_callback(self, callback: Callable[[PerformanceMetric], None]):
        """Add a callback for metric notifications"""
        self._metric_callbacks.append(callback)
        
    def export_metrics(self, filename: str, time_window_hours: int = 1):
        """Export metrics to JSON file"""
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        
        with self._lock:
            # Filter metrics within time window
            export_metrics = [
                {
                    "name": metric.name,
                    "value": metric.value,
                    "unit": metric.unit,
                    "timestamp": metric.timestamp.isoformat(),
                    "tags": metric.tags,
                    "metadata": metric.metadata
                }
                for metric in self._metrics_history 
                if metric.timestamp >= cutoff_time
            ]
            
            export_data = {
                "export_time": datetime.utcnow().isoformat(),
                "time_window_hours": time_window_hours,
                "total_metrics": len(export_metrics),
                "metrics": export_metrics,
                "performance_summary": self.get_performance_summary(time_window_hours * 60)
            }
            
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            logger.info(f"Exported {len(export_metrics)} metrics to {filename}")
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")


# Global performance monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
        _global_monitor.start_monitoring()
    return _global_monitor


def monitor_performance(operation_name: str, tags: Optional[Dict[str, str]] = None):
    """Decorator for automatic performance monitoring"""
    monitor = get_performance_monitor()
    return monitor.timer(operation_name, tags)