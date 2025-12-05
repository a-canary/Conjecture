#!/usr/bin/env python3
"""
Performance Monitoring and Metrics Collection System for Conjecture Hypothesis Validation
Comprehensive tracking of execution metrics, resource usage, and performance indicators
"""

import asyncio
import json
import time
import psutil
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
import statistics
import sys
import os
import uuid
from collections import defaultdict, deque

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single execution"""
    
    execution_id: str
    test_id: str
    approach: str
    model: str
    category: str
    
    # Timing metrics
    start_time: datetime
    end_time: datetime
    execution_time: float
    llm_response_time: float
    preprocessing_time: float
    postprocessing_time: float
    
    # Resource metrics
    cpu_usage_percent: float
    memory_usage_mb: float
    peak_memory_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_io_bytes: int
    
    # LLM-specific metrics
    token_usage_input: int
    token_usage_output: int
    token_usage_total: int
    tokens_per_second: float
    cost_estimate: float
    
    # Quality metrics
    evaluation_score: float
    success: bool
    error_type: Optional[str]
    error_message: Optional[str]
    
    # Metadata
    timestamp: datetime


@dataclass
class SystemMetrics:
    """System-wide performance metrics"""
    
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_processes: int
    load_average: Optional[Tuple[float, float, float]]


@dataclass
class PerformanceSummary:
    """Summary statistics for a performance category"""
    
    category: str
    approach: str
    metric_name: str
    
    # Basic statistics
    count: int
    mean: float
    median: float
    std: float
    min: float
    max: float
    
    # Percentiles
    p25: float
    p75: float
    p90: float
    p95: float
    p99: float
    
    # Quality indicators
    success_rate: float
    error_rate: float
    outlier_count: int


class PerformanceMonitoringSystem:
    """Comprehensive performance monitoring system"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        
        # Directory setup
        self.metrics_dir = Path("tests/metrics/performance")
        self.reports_dir = Path("tests/reports/performance")
        self.plots_dir = Path("tests/plots/performance")
        
        for dir_path in [self.metrics_dir, self.reports_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.execution_metrics: List[PerformanceMetrics] = []
        self.system_metrics: deque = deque(maxlen=1000)  # Rolling window
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        
        # System monitoring
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Baseline measurements
        self.baseline_metrics: Dict[str, float] = {}
        
        # Logging
        self.logger = self._setup_logging()
        
        # Cost tracking
        self.model_costs = {
            "ibm/granite-4-h-tiny": 0.0001,  # $ per 1K tokens
            "glm-z1-9b-0414": 0.0002,
            "zai-org/GLM-4.6": 0.001
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for performance monitoring"""
        logger = logging.getLogger("performance_monitoring")
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.metrics_dir / "performance_monitoring.log")
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def start_monitoring(self):
        """Start system monitoring"""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Performance monitoring stopped")
    
    def _monitor_system(self):
        """Monitor system metrics in background thread"""
        
        process = psutil.Process()
        
        while self.monitoring_active:
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                network = psutil.net_io_counters()
                load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else None
                
                # Get process-specific metrics
                process_memory = process.memory_info()
                process_cpu = process.cpu_percent()
                
                system_metrics = SystemMetrics(
                    timestamp=datetime.utcnow(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_available_gb=memory.available / (1024**3),
                    disk_usage_percent=disk.percent,
                    network_bytes_sent=network.bytes_sent,
                    network_bytes_recv=network.bytes_recv,
                    active_processes=len(psutil.pids()),
                    load_average=load_avg
                )
                
                self.system_metrics.append(system_metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in system monitoring: {e}")
                time.sleep(self.monitoring_interval)
    
    def start_execution_tracking(
        self, 
        execution_id: str,
        test_id: str,
        approach: str,
        model: str,
        category: str
    ) -> Dict[str, Any]:
        """Start tracking a new execution"""
        
        start_time = datetime.utcnow()
        
        # Get initial system state
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        initial_disk_io = process.io_counters()
        
        tracking_data = {
            "execution_id": execution_id,
            "test_id": test_id,
            "approach": approach,
            "model": model,
            "category": category,
            "start_time": start_time,
            "initial_memory_mb": initial_memory,
            "initial_disk_read": initial_disk_io.read_bytes / (1024 * 1024),
            "initial_disk_write": initial_disk_io.write_bytes / (1024 * 1024),
            "initial_network_io": psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
        }
        
        self.active_executions[execution_id] = tracking_data
        
        self.logger.debug(f"Started tracking execution {execution_id}")
        
        return tracking_data
    
    def record_llm_response(
        self,
        execution_id: str,
        response_time: float,
        token_usage_input: int,
        token_usage_output: int
    ):
        """Record LLM response metrics"""
        
        if execution_id not in self.active_executions:
            self.logger.warning(f"Execution {execution_id} not found in active tracking")
            return
        
        tracking_data = self.active_executions[execution_id]
        tracking_data.update({
            "llm_response_time": response_time,
            "token_usage_input": token_usage_input,
            "token_usage_output": token_usage_output,
            "token_usage_total": token_usage_input + token_usage_output
        })
    
    def end_execution_tracking(
        self,
        execution_id: str,
        success: bool = True,
        evaluation_score: float = 0.0,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> Optional[PerformanceMetrics]:
        """End tracking and create performance metrics"""
        
        if execution_id not in self.active_executions:
            self.logger.warning(f"Execution {execution_id} not found in active tracking")
            return None
        
        tracking_data = self.active_executions[execution_id]
        end_time = datetime.utcnow()
        
        # Calculate final system state
        process = psutil.Process()
        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        final_disk_io = process.io_counters()
        
        # Calculate timing metrics
        start_time = tracking_data["start_time"]
        execution_time = (end_time - start_time).total_seconds()
        
        llm_response_time = tracking_data.get("llm_response_time", 0.0)
        preprocessing_time = 0.1  # Placeholder
        postprocessing_time = 0.1  # Placeholder
        
        # Calculate resource metrics
        memory_usage_mb = final_memory
        peak_memory_mb = final_memory  # Would need peak tracking during execution
        memory_delta = final_memory - tracking_data["initial_memory_mb"]
        
        disk_io_read_mb = (final_disk_io.read_bytes / (1024 * 1024)) - tracking_data["initial_disk_read"]
        disk_io_write_mb = (final_disk_io.write_bytes / (1024 * 1024)) - tracking_data["initial_disk_write"]
        
        final_network_io = psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
        network_io_bytes = final_network_io - tracking_data["initial_network_io"]
        
        # Calculate LLM-specific metrics
        token_usage_input = tracking_data.get("token_usage_input", 0)
        token_usage_output = tracking_data.get("token_usage_output", 0)
        token_usage_total = tracking_data.get("token_usage_total", 0)
        
        tokens_per_second = token_usage_total / llm_response_time if llm_response_time > 0 else 0.0
        
        model = tracking_data["model"]
        cost_estimate = (token_usage_total / 1000) * self.model_costs.get(model, 0.001)
        
        # Create performance metrics
        metrics = PerformanceMetrics(
            execution_id=execution_id,
            test_id=tracking_data["test_id"],
            approach=tracking_data["approach"],
            model=model,
            category=tracking_data["category"],
            start_time=start_time,
            end_time=end_time,
            execution_time=execution_time,
            llm_response_time=llm_response_time,
            preprocessing_time=preprocessing_time,
            postprocessing_time=postprocessing_time,
            cpu_usage_percent=psutil.cpu_percent(),
            memory_usage_mb=memory_usage_mb,
            peak_memory_mb=peak_memory_mb,
            disk_io_read_mb=disk_io_read_mb,
            disk_io_write_mb=disk_io_write_mb,
            network_io_bytes=network_io_bytes,
            token_usage_input=token_usage_input,
            token_usage_output=token_usage_output,
            token_usage_total=token_usage_total,
            tokens_per_second=tokens_per_second,
            cost_estimate=cost_estimate,
            evaluation_score=evaluation_score,
            success=success,
            error_type=error_type,
            error_message=error_message,
            timestamp=datetime.utcnow()
        )
        
        # Store metrics
        self.execution_metrics.append(metrics)
        
        # Remove from active tracking
        del self.active_executions[execution_id]
        
        self.logger.debug(f"Completed tracking execution {execution_id} in {execution_time:.2f}s")
        
        return metrics
    
    def get_performance_summary(
        self, 
        category: Optional[str] = None,
        approach: Optional[str] = None
    ) -> Dict[str, List[PerformanceSummary]]:
        """Get performance summary statistics"""
        
        # Filter metrics
        filtered_metrics = self.execution_metrics
        if category:
            filtered_metrics = [m for m in filtered_metrics if m.category == category]
        if approach:
            filtered_metrics = [m for m in filtered_metrics if m.approach == approach]
        
        if not filtered_metrics:
            return {}
        
        # Group by category and approach
        groups = defaultdict(list)
        for metric in filtered_metrics:
            key = f"{metric.category}_{metric.approach}"
            groups[key].append(metric)
        
        # Calculate summaries for each group
        summaries = {}
        for group_key, group_metrics in groups.items():
            category_name, approach_name = group_key.split('_', 1)
            
            if category_name not in summaries:
                summaries[category_name] = []
            
            # Calculate summaries for each metric type
            metric_types = [
                ("execution_time", lambda m: m.execution_time),
                ("llm_response_time", lambda m: m.llm_response_time),
                ("memory_usage_mb", lambda m: m.memory_usage_mb),
                ("token_usage_total", lambda m: m.token_usage_total),
                ("tokens_per_second", lambda m: m.tokens_per_second),
                ("cost_estimate", lambda m: m.cost_estimate),
                ("evaluation_score", lambda m: m.evaluation_score)
            ]
            
            for metric_name, extractor in metric_types:
                values = [extractor(m) for m in group_metrics]
                
                if values:
                    summary = self._calculate_summary_statistics(
                        category_name, approach_name, metric_name, values, group_metrics
                    )
                    summaries[category_name].append(summary)
        
        return summaries
    
    def _calculate_summary_statistics(
        self,
        category: str,
        approach: str,
        metric_name: str,
        values: List[float],
        metrics: List[PerformanceMetrics]
    ) -> PerformanceSummary:
        """Calculate summary statistics for a metric"""
        
        if not values:
            raise ValueError("No values provided for summary calculation")
        
        # Basic statistics
        count = len(values)
        mean_val = statistics.mean(values)
        median_val = statistics.median(values)
        std_val = statistics.stdev(values) if count > 1 else 0.0
        min_val = min(values)
        max_val = max(values)
        
        # Percentiles
        sorted_values = sorted(values)
        p25 = sorted_values[int(0.25 * count)]
        p75 = sorted_values[int(0.75 * count)]
        p90 = sorted_values[int(0.90 * count)]
        p95 = sorted_values[int(0.95 * count)]
        p99 = sorted_values[int(0.99 * count)]
        
        # Quality indicators
        success_count = sum(1 for m in metrics if m.success)
        success_rate = success_count / count
        error_rate = 1.0 - success_rate
        
        # Outlier detection (using IQR method)
        q1 = p25
        q3 = p75
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = sum(1 for v in values if v < lower_bound or v > upper_bound)
        
        return PerformanceSummary(
            category=category,
            approach=approach,
            metric_name=metric_name,
            count=count,
            mean=mean_val,
            median=median_val,
            std=std_val,
            min=min_val,
            max=max_val,
            p25=p25,
            p75=p75,
            p90=p90,
            p95=p95,
            p99=p99,
            success_rate=success_rate,
            error_rate=error_rate,
            outlier_count=outlier_count
        )
    
    def compare_approach_performance(
        self, 
        category: str,
        metric_name: str = "execution_time"
    ) -> Dict[str, Any]:
        """Compare performance between approaches for a specific metric"""
        
        # Get metrics for the category
        category_metrics = [m for m in self.execution_metrics if m.category == category]
        
        if not category_metrics:
            return {}
        
        # Group by approach
        approach_data = defaultdict(list)
        for metric in category_metrics:
            approach_data[metric.approach].append(metric)
        
        # Extract metric values
        comparison = {}
        for approach, metrics in approach_data.items():
            values = [getattr(m, metric_name) for m in metrics]
            if values:
                comparison[approach] = {
                    "values": values,
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
        
        # Calculate improvement percentages
        if "direct" in comparison and "conjecture" in comparison:
            direct_mean = comparison["direct"]["mean"]
            conjecture_mean = comparison["conjecture"]["mean"]
            
            if metric_name == "execution_time":
                # For time, lower is better
                improvement = ((direct_mean - conjecture_mean) / direct_mean) * 100
            else:
                # For scores, higher is better
                improvement = ((conjecture_mean - direct_mean) / direct_mean) * 100
            
            comparison["improvement_percentage"] = improvement
            comparison["better_approach"] = "conjecture" if improvement > 0 else "direct"
        
        return comparison
    
    def detect_performance_anomalies(self) -> List[Dict[str, Any]]:
        """Detect performance anomalies and outliers"""
        
        anomalies = []
        
        # Check execution time anomalies
        execution_times = [m.execution_time for m in self.execution_metrics]
        if len(execution_times) > 10:
            mean_time = statistics.mean(execution_times)
            std_time = statistics.stdev(execution_times)
            threshold = mean_time + 3 * std_time
            
            for metric in self.execution_metrics:
                if metric.execution_time > threshold:
                    anomalies.append({
                        "type": "slow_execution",
                        "execution_id": metric.execution_id,
                        "test_id": metric.test_id,
                        "approach": metric.approach,
                        "execution_time": metric.execution_time,
                        "threshold": threshold,
                        "severity": "high" if metric.execution_time > threshold * 2 else "medium"
                    })
        
        # Check memory usage anomalies
        memory_usages = [m.memory_usage_mb for m in self.execution_metrics]
        if len(memory_usages) > 10:
            mean_memory = statistics.mean(memory_usages)
            std_memory = statistics.stdev(memory_usages)
            memory_threshold = mean_memory + 2 * std_memory
            
            for metric in self.execution_metrics:
                if metric.memory_usage_mb > memory_threshold:
                    anomalies.append({
                        "type": "high_memory_usage",
                        "execution_id": metric.execution_id,
                        "test_id": metric.test_id,
                        "approach": metric.approach,
                        "memory_usage_mb": metric.memory_usage_mb,
                        "threshold": memory_threshold,
                        "severity": "medium"
                    })
        
        # Check error rate spikes
        recent_metrics = [m for m in self.execution_metrics 
                        if m.timestamp > datetime.utcnow() - timedelta(hours=1)]
        
        if len(recent_metrics) > 5:
            error_count = sum(1 for m in recent_metrics if not m.success)
            error_rate = error_count / len(recent_metrics)
            
            if error_rate > 0.2:  # 20% error rate threshold
                anomalies.append({
                    "type": "high_error_rate",
                    "error_rate": error_rate,
                    "error_count": error_count,
                    "total_count": len(recent_metrics),
                    "time_window": "1 hour",
                    "severity": "high" if error_rate > 0.5 else "medium"
                })
        
        return anomalies
    
    async def save_metrics(self):
        """Save all metrics to files"""
        
        # Save execution metrics
        execution_metrics_data = []
        for metric in self.execution_metrics:
            metric_dict = asdict(metric)
            metric_dict["start_time"] = metric.start_time.isoformat()
            metric_dict["end_time"] = metric.end_time.isoformat()
            metric_dict["timestamp"] = metric.timestamp.isoformat()
            execution_metrics_data.append(metric_dict)
        
        execution_file = self.metrics_dir / f"execution_metrics_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(execution_file, 'w', encoding='utf-8') as f:
            json.dump(execution_metrics_data, f, indent=2, ensure_ascii=False)
        
        # Save system metrics
        system_metrics_data = []
        for metric in self.system_metrics:
            metric_dict = asdict(metric)
            metric_dict["timestamp"] = metric.timestamp.isoformat()
            system_metrics_data.append(metric_dict)
        
        system_file = self.metrics_dir / f"system_metrics_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(system_file, 'w', encoding='utf-8') as f:
            json.dump(system_metrics_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(execution_metrics_data)} execution metrics and {len(system_metrics_data)} system metrics")
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        
        report_lines = [
            "# Performance Monitoring Report",
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            "",
            f"**Total Executions Tracked**: {len(self.execution_metrics)}",
            f"**System Metrics Collected**: {len(self.system_metrics)}",
            f"**Active Monitoring**: {'Yes' if self.monitoring_active else 'No'}",
            "",
            "## Performance Summaries",
            ""
        ]
        
        # Get performance summaries by category
        summaries = self.get_performance_summary()
        
        for category, category_summaries in summaries.items():
            report_lines.extend([
                f"### {category.replace('_', ' ').title()}",
                ""
            ])
            
            # Group summaries by approach
            approach_summaries = defaultdict(list)
            for summary in category_summaries:
                approach_summaries[summary.approach].append(summary)
            
            for approach, approach_summary_list in approach_summaries.items():
                report_lines.extend([
                    f"#### {approach.title()} Approach",
                    ""
                ])
                
                for summary in approach_summary_list:
                    report_lines.extend([
                        f"**{summary.metric_name.replace('_', ' ').title()}:**",
                        f"- Mean: {summary.mean:.3f}",
                        f"- Median: {summary.median:.3f}",
                        f"- Std Dev: {summary.std:.3f}",
                        f"- Range: {summary.min:.3f} - {summary.max:.3f}",
                        f"- Success Rate: {summary.success_rate:.1%}",
                        f"- Outliers: {summary.outlier_count}",
                        ""
                    ])
        
        # Add approach comparisons
        report_lines.extend([
            "## Approach Comparisons",
            ""
        ])
        
        categories = set(m.category for m in self.execution_metrics)
        for category in categories:
            comparison = self.compare_approach_performance(category, "execution_time")
            if comparison and "improvement_percentage" in comparison:
                report_lines.extend([
                    f"### {category.replace('_', ' ').title()}",
                    f"**Performance Improvement**: {comparison['improvement_percentage']:.1f}%",
                    f"**Better Approach**: {comparison['better_approach'].title()}",
                    ""
                ])
        
        # Add anomaly detection
        anomalies = self.detect_performance_anomalies()
        if anomalies:
            report_lines.extend([
                "## Performance Anomalies",
                ""
            ])
            
            for anomaly in anomalies:
                report_lines.extend([
                    f"### {anomaly['type'].replace('_', ' ').title()}",
                    f"**Severity**: {anomaly['severity'].title()}",
                    f"**Details**: {anomaly}",
                    ""
                ])
        
        # Add recommendations
        report_lines.extend([
            "## Recommendations",
            "",
            "### Performance Optimization",
            "1. Monitor execution times for approaches with high variability",
            "2. Investigate memory usage patterns for optimization opportunities",
            "3. Track token usage efficiency for cost optimization",
            "4. Implement alerting for performance anomalies",
            "",
            "### System Resource Management",
            "1. Monitor system resource utilization during peak loads",
            "2. Implement resource pooling for better efficiency",
            "3. Consider scaling strategies for high-demand scenarios",
            ""
        ])
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_file = self.reports_dir / f"performance_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return report_content


# Context manager for execution tracking
class ExecutionTracker:
    """Context manager for tracking execution performance"""
    
    def __init__(
        self, 
        monitoring_system: PerformanceMonitoringSystem,
        test_id: str,
        approach: str,
        model: str,
        category: str
    ):
        self.monitoring_system = monitoring_system
        self.test_id = test_id
        self.approach = approach
        self.model = model
        self.category = category
        self.execution_id = str(uuid.uuid4())
        self.tracking_data = None
        self.success = True
        self.evaluation_score = 0.0
        self.error_type = None
        self.error_message = None
    
    def __enter__(self):
        self.tracking_data = self.monitoring_system.start_execution_tracking(
            self.execution_id, self.test_id, self.approach, self.model, self.category
        )
        return self
    
    def record_llm_response(self, response_time: float, token_input: int, token_output: int):
        """Record LLM response metrics"""
        self.monitoring_system.record_llm_response(
            self.execution_id, response_time, token_input, token_output
        )
    
    def set_result(self, success: bool, evaluation_score: float = 0.0, 
                   error_type: Optional[str] = None, error_message: Optional[str] = None):
        """Set the execution result"""
        self.success = success
        self.evaluation_score = evaluation_score
        self.error_type = error_type
        self.error_message = error_message
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.success = False
            self.error_type = exc_type.__name__ if exc_type else None
            self.error_message = str(exc_val) if exc_val else None
        
        self.monitoring_system.end_execution_tracking(
            self.execution_id, self.success, self.evaluation_score,
            self.error_type, self.error_message
        )
        
        return False  # Don't suppress exceptions


async def main():
    """Main function to test the performance monitoring system"""
    
    # Initialize monitoring system
    monitor = PerformanceMonitoringSystem(monitoring_interval=0.5)
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        print("Testing performance monitoring system...")
        
        # Simulate some executions
        for i in range(5):
            with ExecutionTracker(
                monitor,
                test_id=f"test_{i:03d}",
                approach="conjecture" if i % 2 == 0 else "direct",
                model="ibm/granite-4-h-tiny",
                category="sample_category"
            ) as tracker:
                
                # Simulate LLM response
                await asyncio.sleep(0.1 + (i * 0.05))  # Simulate variable response times
                tracker.record_llm_response(
                    response_time=0.1 + (i * 0.05),
                    token_input=100 + (i * 10),
                    token_output=50 + (i * 5)
                )
                
                # Set result
                success = i < 4  # Simulate one failure
                evaluation_score = 0.7 + (i * 0.05) if success else 0.3
                tracker.set_result(success, evaluation_score)
                
                print(f"Completed execution {i+1}/5")
        
        # Wait a bit for system metrics collection
        await asyncio.sleep(2)
        
        # Generate performance summary
        summary = monitor.get_performance_summary()
        print(f"\nPerformance Summary:")
        for category, summaries in summary.items():
            print(f"  {category}: {len(summaries)} metric summaries")
        
        # Detect anomalies
        anomalies = monitor.detect_performance_anomalies()
        print(f"\nDetected {len(anomalies)} anomalies")
        
        # Generate report
        report = monitor.generate_performance_report()
        print(f"\n{report}")
        
        # Save metrics
        await monitor.save_metrics()
        
    finally:
        # Stop monitoring
        monitor.stop_monitoring()
        print(f"\nMetrics saved to: {monitor.metrics_dir}")
        print(f"Reports saved to: {monitor.reports_dir}")


if __name__ == "__main__":
    asyncio.run(main())