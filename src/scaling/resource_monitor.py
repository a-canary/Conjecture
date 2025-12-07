"""
Resource Monitoring and Load Balancing for Scaling Analysis - Phase 2
Comprehensive system monitoring with adaptive load balancing capabilities
"""

import asyncio
import time
import psutil
import threading
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
import weakref
import queue
import math

from ..monitoring import get_performance_monitor

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Resource types for monitoring"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    DATABASE_CONNECTIONS = "database_connections"
    ACTIVE_EVALUATIONS = "active_evaluations"
    LLM_REQUESTS = "llm_requests"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESOURCE_BASED = "resource_based"
    ADAPTIVE = "adaptive"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class ResourceMetric:
    """Individual resource metric data point"""
    resource_type: ResourceType
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None


@dataclass
class LoadBalancingTarget:
    """Target for load balancing"""
    name: str
    weight: float = 1.0
    current_connections: int = 0
    max_connections: int = 10
    response_time: float = 0.0
    success_rate: float = 1.0
    last_used: datetime = field(default_factory=datetime.utcnow)
    health_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """System alert"""
    alert_id: str
    level: AlertLevel
    resource_type: ResourceType
    message: str
    value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class ResourceMonitor:
    """
    Comprehensive resource monitoring with real-time tracking and alerting
    """

    def __init__(
        self,
        monitoring_interval: float = 1.0,
        history_size: int = 1000,
        enable_alerts: bool = True,
        thresholds: Optional[Dict[ResourceType, Dict[str, float]]] = None
    ):
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.enable_alerts = enable_alerts

        # Default thresholds
        self.thresholds = thresholds or {
            ResourceType.CPU: {"warning": 70.0, "critical": 90.0},
            ResourceType.MEMORY: {"warning": 80.0, "critical": 95.0},
            ResourceType.DISK: {"warning": 85.0, "critical": 95.0},
            ResourceType.DATABASE_CONNECTIONS: {"warning": 15, "critical": 25},
            ResourceType.ACTIVE_EVALUATIONS: {"warning": 8, "critical": 15}
        }

        # Monitoring state
        self._monitoring_active = False
        self._monitor_task: Optional[asyncio.Task] = None

        # Data storage
        self._metrics_history: Dict[ResourceType, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self._current_metrics: Dict[ResourceType, ResourceMetric] = {}
        self._alerts: deque = deque(maxlen=1000)

        # Statistics
        self._statistics = {
            "total_samples": 0,
            "alerts_generated": 0,
            "alerts_resolved": 0,
            "resource_utilization": defaultdict(list)
        }

        # Alert callbacks
        self._alert_callbacks: List[Callable[[Alert], None]] = []

        # System info
        self._system_info = self._collect_system_info()

        logger.info("ResourceMonitor initialized")

    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect static system information"""
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total": psutil.virtual_memory().total,
            "disk_total": psutil.disk_usage('/').total if os.name != 'nt' else psutil.disk_usage('C:\\').total,
            "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
            "platform": os.name
        }

    async def start_monitoring(self):
        """Start resource monitoring"""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Resource monitoring started")

    async def stop_monitoring(self):
        """Stop resource monitoring"""
        self._monitoring_active = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("Resource monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._monitoring_active:
            try:
                # Collect all metrics
                await self._collect_all_metrics()

                # Check for alerts
                if self.enable_alerts:
                    self._check_alerts()

                # Update statistics
                self._update_statistics()

                # Sleep until next interval
                await asyncio.sleep(self.monitoring_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1)

    async def _collect_all_metrics(self):
        """Collect metrics for all resource types"""
        timestamp = datetime.utcnow()

        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_metric = ResourceMetric(
            resource_type=ResourceType.CPU,
            value=cpu_percent,
            unit="percent",
            timestamp=timestamp,
            threshold_warning=self.thresholds[ResourceType.CPU].get("warning"),
            threshold_critical=self.thresholds[ResourceType.CPU].get("critical")
        )
        self._current_metrics[ResourceType.CPU] = cpu_metric
        self._metrics_history[ResourceType.CPU].append(cpu_metric)

        # Memory metrics
        memory = psutil.virtual_memory()
        memory_metric = ResourceMetric(
            resource_type=ResourceType.MEMORY,
            value=memory.percent,
            unit="percent",
            timestamp=timestamp,
            threshold_warning=self.thresholds[ResourceType.MEMORY].get("warning"),
            threshold_critical=self.thresholds[ResourceType.MEMORY].get("critical")
        )
        self._current_metrics[ResourceType.MEMORY] = memory_metric
        self._metrics_history[ResourceType.MEMORY].append(memory_metric)

        # Disk metrics
        disk_usage = psutil.disk_usage('/') if os.name != 'nt' else psutil.disk_usage('C:\\')
        disk_percent = (disk_usage.used / disk_usage.total) * 100
        disk_metric = ResourceMetric(
            resource_type=ResourceType.DISK,
            value=disk_percent,
            unit="percent",
            timestamp=timestamp,
            threshold_warning=self.thresholds[ResourceType.DISK].get("warning"),
            threshold_critical=self.thresholds[ResourceType.DISK].get("critical")
        )
        self._current_metrics[ResourceType.DISK] = disk_metric
        self._metrics_history[ResourceType.DISK].append(disk_metric)

        self._statistics["total_samples"] += 1

    def _check_alerts(self):
        """Check if any metrics exceed thresholds and generate alerts"""
        for resource_type, metric in self._current_metrics.items():
            if resource_type not in self.thresholds:
                continue

            thresholds = self.thresholds[resource_type]
            value = metric.value

            # Check critical threshold
            if thresholds.get("critical") and value >= thresholds["critical"]:
                self._generate_alert(
                    AlertLevel.CRITICAL,
                    resource_type,
                    f"Critical resource utilization: {value:.1f}{metric.unit}",
                    value,
                    thresholds["critical"]
                )

            # Check warning threshold
            elif thresholds.get("warning") and value >= thresholds["warning"]:
                self._generate_alert(
                    AlertLevel.WARNING,
                    resource_type,
                    f"High resource utilization: {value:.1f}{metric.unit}",
                    value,
                    thresholds["warning"]
                )

    def _generate_alert(
        self,
        level: AlertLevel,
        resource_type: ResourceType,
        message: str,
        value: float,
        threshold: float
    ):
        """Generate a system alert"""
        alert = Alert(
            alert_id=f"alert_{int(time.time() * 1000)}",
            level=level,
            resource_type=resource_type,
            message=message,
            value=value,
            threshold=threshold
        )

        self._alerts.append(alert)
        self._statistics["alerts_generated"] += 1

        logger.warning(f"ALERT [{level.value.upper()}] {message}")

        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    def _update_statistics(self):
        """Update monitoring statistics"""
        for resource_type, history in self._metrics_history.items():
            if history:
                recent_values = [metric.value for metric in list(history)[-60:]]  # Last 60 samples
                self._statistics["resource_utilization"][resource_type].extend(recent_values)

                # Keep only recent values
                if len(self._statistics["resource_utilization"][resource_type]) > 300:
                    self._statistics["resource_utilization"][resource_type] = \
                        self._statistics["resource_utilization"][resource_type][-300:]

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback for alert notifications"""
        self._alert_callbacks.append(callback)

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current resource metrics"""
        return {
            resource_type.value: {
                "value": metric.value,
                "unit": metric.unit,
                "timestamp": metric.timestamp.isoformat(),
                "status": self._get_metric_status(metric)
            }
            for resource_type, metric in self._current_metrics.items()
        }

    def _get_metric_status(self, metric: ResourceMetric) -> str:
        """Get status of a metric based on thresholds"""
        if not metric.threshold_critical and not metric.threshold_warning:
            return "normal"

        if metric.threshold_critical and metric.value >= metric.threshold_critical:
            return "critical"
        elif metric.threshold_warning and metric.value >= metric.threshold_warning:
            return "warning"
        return "normal"

    def get_utilization_history(
        self,
        resource_type: ResourceType,
        minutes: int = 10
    ) -> List[Dict[str, Any]]:
        """Get utilization history for a resource type"""
        if resource_type not in self._metrics_history:
            return []

        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        history = self._metrics_history[resource_type]

        return [
            {
                "timestamp": metric.timestamp.isoformat(),
                "value": metric.value,
                "unit": metric.unit
            }
            for metric in history
            if metric.timestamp >= cutoff_time
        ]

    def get_alerts(
        self,
        level: Optional[AlertLevel] = None,
        resolved: Optional[bool] = None,
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get system alerts"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        filtered_alerts = []
        for alert in self._alerts:
            if alert.timestamp < cutoff_time:
                continue

            if level and alert.level != level:
                continue

            if resolved is not None and alert.resolved != resolved:
                continue

            filtered_alerts.append({
                "alert_id": alert.alert_id,
                "level": alert.level.value,
                "resource_type": alert.resource_type.value,
                "message": alert.message,
                "value": alert.value,
                "threshold": alert.threshold,
                "timestamp": alert.timestamp.isoformat(),
                "resolved": alert.resolved,
                "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None
            })

        return filtered_alerts

    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive system summary"""
        current_metrics = self.get_current_metrics()

        # Calculate averages
        averages = {}
        for resource_type, values in self._statistics["resource_utilization"].items():
            if values:
                averages[resource_type.value] = sum(values) / len(values)

        recent_alerts = self.get_alerts(hours=1)

        return {
            "system_info": self._system_info,
            "current_metrics": current_metrics,
            "average_utilization": averages,
            "recent_alerts_count": len(recent_alerts),
            "critical_alerts_count": len([a for a in recent_alerts if a["level"] == "critical"]),
            "monitoring_statistics": {
                "total_samples": self._statistics["total_samples"],
                "alerts_generated": self._statistics["alerts_generated"],
                "alerts_resolved": self._statistics["alerts_resolved"],
                "monitoring_active": self._monitoring_active
            }
        }


class LoadBalancer:
    """
    Intelligent load balancer with multiple strategies and adaptive behavior
    """

    def __init__(
        self,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE,
        health_check_interval: float = 30.0,
        enable_adaptive_weights: bool = True
    ):
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        self.enable_adaptive_weights = enable_adaptive_weights

        # Target management
        self._targets: Dict[str, LoadBalancingTarget] = {}
        self._target_index = 0  # For round-robin

        # Health monitoring
        self._health_check_active = False
        self._health_check_task: Optional[asyncio.Task] = None

        # Statistics
        self._statistics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "target_utilization": defaultdict(int)
        }

        # Performance tracking
        self._performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        logger.info(f"LoadBalancer initialized with strategy: {strategy.value}")

    def add_target(self, target: LoadBalancingTarget):
        """Add a load balancing target"""
        self._targets[target.name] = target
        logger.info(f"Added load balancing target: {target.name}")

    def remove_target(self, target_name: str):
        """Remove a load balancing target"""
        if target_name in self._targets:
            del self._targets[target_name]
            if target_name in self._performance_history:
                del self._performance_history[target_name]
            logger.info(f"Removed load balancing target: {target_name}")

    async def start_health_checks(self):
        """Start periodic health checks"""
        if self._health_check_active:
            return

        self._health_check_active = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Load balancer health checks started")

    async def stop_health_checks(self):
        """Stop health checks"""
        self._health_check_active = False

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        logger.info("Load balancer health checks stopped")

    async def _health_check_loop(self):
        """Periodic health check loop"""
        while self._health_check_active:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(5)

    async def _perform_health_checks(self):
        """Perform health checks on all targets"""
        for target_name, target in self._targets.items():
            try:
                # Simulate health check (in real implementation, would ping/service check)
                health_score = await self._check_target_health(target)
                target.health_score = health_score

                # Update adaptive weights if enabled
                if self.enable_adaptive_weights:
                    self._update_adaptive_weight(target)

            except Exception as e:
                logger.error(f"Health check failed for {target_name}: {e}")
                target.health_score = 0.0

    async def _check_target_health(self, target: LoadBalancingTarget) -> float:
        """Check health of a specific target"""
        # Simplified health check based on performance metrics
        base_health = 1.0

        # Factor in response time (lower is better)
        if target.response_time > 0:
            time_factor = max(0.1, 1.0 - (target.response_time / 10.0))  # 10s as worst case
            base_health *= time_factor

        # Factor in success rate
        base_health *= target.success_rate

        # Factor in connection utilization
        connection_factor = 1.0 - (target.current_connections / target.max_connections)
        base_health *= max(0.1, connection_factor)

        return min(1.0, max(0.0, base_health))

    def _update_adaptive_weight(self, target: LoadBalancingTarget):
        """Update target weight based on performance"""
        # Base weight on health score and response time
        if target.response_time > 0:
            # Lower response time = higher weight
            performance_weight = 1.0 / (1.0 + target.response_time)
        else:
            performance_weight = 1.0

        # Combine with health score
        target.weight = target.health_score * performance_weight

        # Ensure minimum weight
        target.weight = max(0.1, target.weight)

    def select_target(self, request_context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Select the best target for a request"""
        if not self._targets:
            return None

        # Filter healthy targets
        healthy_targets = [
            (name, target) for name, target in self._targets.items()
            if target.health_score > 0.1 and target.current_connections < target.max_connections
        ]

        if not healthy_targets:
            logger.warning("No healthy targets available")
            return None

        # Apply load balancing strategy
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            selected_name = self._select_round_robin(healthy_targets)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            selected_name = self._select_least_connections(healthy_targets)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            selected_name = self._select_weighted_round_robin(healthy_targets)
        elif self.strategy == LoadBalancingStrategy.RESOURCE_BASED:
            selected_name = self._select_resource_based(healthy_targets, request_context)
        elif self.strategy == LoadBalancingStrategy.ADAPTIVE:
            selected_name = self._select_adaptive(healthy_targets, request_context)
        else:
            selected_name = healthy_targets[0][0]

        if selected_name:
            self._targets[selected_name].current_connections += 1
            self._targets[selected_name].last_used = datetime.utcnow()
            self._statistics["target_utilization"][selected_name] += 1

        return selected_name

    def _select_round_robin(self, targets: List[Tuple[str, LoadBalancingTarget]]) -> str:
        """Round-robin selection"""
        if self._target_index >= len(targets):
            self._target_index = 0

        selected_name = targets[self._target_index][0]
        self._target_index += 1
        return selected_name

    def _select_least_connections(self, targets: List[Tuple[str, LoadBalancingTarget]]) -> str:
        """Select target with fewest connections"""
        return min(targets, key=lambda t: t[1].current_connections)[0]

    def _select_weighted_round_robin(self, targets: List[Tuple[str, LoadBalancingTarget]]) -> str:
        """Weighted round-robin selection"""
        total_weight = sum(target.weight for _, target in targets)
        if total_weight == 0:
            return targets[0][0]

        # Simple weighted selection
        import random
        r = random.uniform(0, total_weight)
        current_weight = 0

        for name, target in targets:
            current_weight += target.weight
            if r <= current_weight:
                return name

        return targets[-1][0]

    def _select_resource_based(
        self,
        targets: List[Tuple[str, LoadBalancingTarget]],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Resource-based selection"""
        # Simple implementation: choose target with best health score
        return max(targets, key=lambda t: t[1].health_score)[0]

    def _select_adaptive(
        self,
        targets: List[Tuple[str, LoadBalancingTarget]],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Adaptive selection combining multiple factors"""
        # Calculate composite score for each target
        best_target = None
        best_score = -1

        for name, target in targets:
            # Composite score combining health, response time, and load
            health_factor = target.health_score
            load_factor = 1.0 - (target.current_connections / target.max_connections)
            time_factor = 1.0 / (1.0 + target.response_time) if target.response_time > 0 else 1.0

            # Weighted combination
            composite_score = (health_factor * 0.4 + load_factor * 0.3 + time_factor * 0.3)

            if composite_score > best_score:
                best_score = composite_score
                best_target = name

        return best_target or targets[0][0]

    def release_target(self, target_name: str, success: bool = True, response_time: float = 0.0):
        """Release a target after request completion"""
        if target_name not in self._targets:
            return

        target = self._targets[target_name]
        target.current_connections = max(0, target.current_connections - 1)

        # Update statistics
        self._statistics["total_requests"] += 1
        if success:
            self._statistics["successful_requests"] += 1
        else:
            self._statistics["failed_requests"] += 1

        # Update target performance metrics
        if response_time > 0:
            # Update exponential moving average
            if target.response_time == 0:
                target.response_time = response_time
            else:
                target.response_time = target.response_time * 0.8 + response_time * 0.2

        # Update success rate
        total_requests = self._statistics["target_utilization"][target_name]
        if total_requests > 0:
            target.success_rate = self._statistics["successful_requests"] / self._statistics["total_requests"]

        # Record performance history
        self._performance_history[target_name].append({
            "timestamp": datetime.utcnow(),
            "response_time": response_time,
            "success": success
        })

    def get_target_statistics(self) -> Dict[str, Any]:
        """Get statistics for all targets"""
        return {
            "targets": {
                name: {
                    "weight": target.weight,
                    "current_connections": target.current_connections,
                    "max_connections": target.max_connections,
                    "health_score": target.health_score,
                    "response_time": target.response_time,
                    "success_rate": target.success_rate,
                    "last_used": target.last_used.isoformat(),
                    "utilization": self._statistics["target_utilization"][name]
                }
                for name, target in self._targets.items()
            },
            "overall_statistics": {
                **self._statistics,
                "success_rate": (
                    self._statistics["successful_requests"] /
                    max(1, self._statistics["total_requests"])
                ),
                "total_targets": len(self._targets),
                "healthy_targets": len([
                    t for t in self._targets.values() if t.health_score > 0.1
                ])
            }
        }


class ScalingOrchestrator:
    """
    Orchestrates resource monitoring and load balancing for optimal scaling
    """

    def __init__(self):
        self.resource_monitor = ResourceMonitor()
        self.load_balancer = LoadBalancer(strategy=LoadBalancingStrategy.ADAPTIVE)

        # Adaptive scaling parameters
        self.scaling_rules = {
            "cpu_scale_up_threshold": 80.0,
            "cpu_scale_down_threshold": 30.0,
            "memory_scale_up_threshold": 85.0,
            "response_time_threshold": 5.0,
            "queue_depth_threshold": 10
        }

        # Scaling state
        self._scaling_active = False
        self._scaling_task: Optional[asyncio.Task] = None

        logger.info("ScalingOrchestrator initialized")

    async def start(self):
        """Start the orchestrator"""
        await self.resource_monitor.start_monitoring()
        await self.load_balancer.start_health_checks()

        self._scaling_active = True
        self._scaling_task = asyncio.create_task(self._scaling_loop())

        logger.info("ScalingOrchestrator started")

    async def stop(self):
        """Stop the orchestrator"""
        self._scaling_active = False

        if self._scaling_task:
            self._scaling_task.cancel()
            try:
                await self._scaling_task
            except asyncio.CancelledError:
                pass

        await self.resource_monitor.stop_monitoring()
        await self.load_balancer.stop_health_checks()

        logger.info("ScalingOrchestrator stopped")

    async def _scaling_loop(self):
        """Main adaptive scaling loop"""
        while self._scaling_active:
            try:
                # Analyze current load and make scaling decisions
                await self._analyze_and_scale()
                await asyncio.sleep(10)  # Check every 10 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                await asyncio.sleep(5)

    async def _analyze_and_scale(self):
        """Analyze system state and make scaling decisions"""
        current_metrics = self.resource_monitor.get_current_metrics()

        # Get CPU and memory utilization
        cpu_util = current_metrics.get("cpu", {}).get("value", 0)
        memory_util = current_metrics.get("memory", {}).get("value", 0)

        # Check if scaling is needed
        scale_up_needed = (
            cpu_util > self.scaling_rules.get("cpu_scale_up_threshold", 80) or
            memory_util > self.scaling_rules.get("memory_scale_up_threshold", 85)
        )

        scale_down_needed = (
            cpu_util < self.scaling_rules.get("cpu_scale_down_threshold", 30) and
            memory_util < self.scaling_rules.get("memory_scale_down_threshold", 30)
        )

        if scale_up_needed:
            await self._trigger_scale_up()
        elif scale_down_needed:
            await self._trigger_scale_down()

    async def _trigger_scale_up(self):
        """Trigger scale-up actions"""
        logger.info("Scale-up triggered based on resource utilization")
        # In a real implementation, would add more targets, increase resources, etc.

    async def _trigger_scale_down(self):
        """Trigger scale-down actions"""
        logger.info("Scale-down triggered based on resource utilization")
        # In a real implementation, would remove targets, reduce resources, etc.

    def add_service_target(self, name: str, max_connections: int = 10, **kwargs):
        """Add a service target to the load balancer"""
        target = LoadBalancingTarget(
            name=name,
            max_connections=max_connections,
            **kwargs
        )
        self.load_balancer.add_target(target)

    async def execute_with_load_balancing(
        self,
        operation: Callable,
        operation_context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute an operation with load balancing"""
        # Select best target
        target_name = self.load_balancer.select_target(operation_context)

        if not target_name:
            raise RuntimeError("No available targets for load balancing")

        start_time = time.time()
        success = False
        result = None

        try:
            # Execute operation
            result = await operation(target_name)
            success = True
        except Exception as e:
            logger.error(f"Operation failed on target {target_name}: {e}")
            raise
        finally:
            # Release target
            response_time = time.time() - start_time
            self.load_balancer.release_target(target_name, success, response_time)

        return result

    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "resource_monitoring": self.resource_monitor.get_system_summary(),
            "load_balancing": self.load_balancer.get_target_statistics(),
            "scaling_state": {
                "active": self._scaling_active,
                "rules": self.scaling_rules
            },
            "overall_health": self._calculate_overall_health()
        }

    def _calculate_overall_health(self) -> Dict[str, Any]:
        """Calculate overall system health score"""
        current_metrics = self.resource_monitor.get_current_metrics()

        # Get critical metrics
        cpu_status = current_metrics.get("cpu", {}).get("status", "normal")
        memory_status = current_metrics.get("memory", {}).get("status", "normal")

        # Get recent alerts
        recent_alerts = self.resource_monitor.get_alerts(hours=1)
        critical_alerts = len([a for a in recent_alerts if a["level"] == "critical"])

        # Calculate health score
        health_score = 1.0
        if cpu_status == "critical":
            health_score -= 0.4
        elif cpu_status == "warning":
            health_score -= 0.2

        if memory_status == "critical":
            health_score -= 0.4
        elif memory_status == "warning":
            health_score -= 0.2

        health_score -= min(0.3, critical_alerts * 0.1)
        health_score = max(0.0, health_score)

        return {
            "score": health_score,
            "status": "healthy" if health_score > 0.7 else "degraded" if health_score > 0.4 else "critical",
            "cpu_status": cpu_status,
            "memory_status": memory_status,
            "critical_alerts": critical_alerts
        }