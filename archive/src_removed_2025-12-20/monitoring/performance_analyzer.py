"""
Performance Analysis and Logging System
Analyzes Conjecture performance metrics and provides insights
"""

import json
import time
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Single performance metric"""
    timestamp: str
    operation: str
    duration: float
    success: bool
    tokens_used: int
    provider: str
    model: str
    task_type: str
    error: Optional[str] = None

@dataclass
class PerformanceSummary:
    """Performance summary for analysis"""
    operation: str
    total_requests: int
    success_rate: float
    avg_duration: float
    p50_duration: float
    p95_duration: float
    p99_duration: float
    avg_tokens: int
    total_tokens: int
    provider_stats: Dict[str, Dict[str, Any]]

class PerformanceAnalyzer:
    """Analyzes Conjecture performance metrics and provides insights"""

    def __init__(self, log_file: str = "conjecture_performance.log"):
        self.log_file = log_file
        self.metrics: List[PerformanceMetric] = []
        self.performance_cache = {}

    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric"""
        self.metrics.append(metric)
        logger.info(f"PERF: {metric.operation} - {metric.duration:.3f}s - {metric.provider} - {'SUCCESS' if metric.success else 'FAILED'}")

    def analyze_performance(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze performance metrics for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        # Filter recent metrics
        recent_metrics = [
            m for m in self.metrics
            if datetime.fromisoformat(m.timestamp) > cutoff_time
        ]

        if not recent_metrics:
            return {"error": f"No metrics found in last {hours} hours"}

        analysis = {
            "analysis_period": f"Last {hours} hours",
            "total_metrics": len(recent_metrics),
            "time_range": {
                "start": min(m.timestamp for m in recent_metrics),
                "end": max(m.timestamp for m in recent_metrics)
            },
            "overall": self._analyze_overall(recent_metrics),
            "by_operation": self._analyze_by_operation(recent_metrics),
            "by_provider": self._analyze_by_provider(recent_metrics),
            "by_task_type": self._analyze_by_task_type(recent_metrics),
            "performance_trends": self._analyze_trends(recent_metrics),
            "recommendations": self._generate_recommendations(recent_metrics)
        }

        return analysis

    def _analyze_overall(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Analyze overall performance"""
        successful = [m for m in metrics if m.success]
        failed = [m for m in metrics if not m.success]

        durations = [m.duration for m in successful]
        tokens = [m.tokens_used for m in successful]

        return {
            "total_requests": len(metrics),
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "success_rate": len(successful) / len(metrics) if metrics else 0,
            "avg_duration": statistics.mean(durations) if durations else 0,
            "p50_duration": statistics.median(durations) if durations else 0,
            "p95_duration": self._percentile(durations, 0.95) if durations else 0,
            "p99_duration": self._percentile(durations, 0.99) if durations else 0,
            "avg_tokens": statistics.mean(tokens) if tokens else 0,
            "total_tokens": sum(tokens),
            "error_rate": len(failed) / len(metrics) if metrics else 0
        }

    def _analyze_by_operation(self, metrics: List[PerformanceMetric]) -> Dict[str, PerformanceSummary]:
        """Analyze performance by operation type"""
        operation_metrics = defaultdict(list)
        for metric in metrics:
            operation_metrics[metric.operation].append(metric)

        summaries = {}
        for operation, op_metrics in operation_metrics.items():
            successful = [m for m in op_metrics if m.success]
            durations = [m.duration for m in successful]
            tokens = [m.tokens_used for m in successful]

            # Provider stats
            provider_stats = defaultdict(list)
            for m in op_metrics:
                provider_stats[m.provider].append(m)

            provider_summary = {}
            for provider, p_metrics in provider_stats.items():
                p_successful = [m for m in p_metrics if m.success]
                provider_summary[provider] = {
                    "requests": len(p_metrics),
                    "success_rate": len(p_successful) / len(p_metrics),
                    "avg_duration": statistics.mean([m.duration for m in p_successful]) if p_successful else 0
                }

            summaries[operation] = PerformanceSummary(
                operation=operation,
                total_requests=len(op_metrics),
                success_rate=len(successful) / len(op_metrics) if op_metrics else 0,
                avg_duration=statistics.mean(durations) if durations else 0,
                p50_duration=statistics.median(durations) if durations else 0,
                p95_duration=self._percentile(durations, 0.95) if durations else 0,
                p99_duration=self._percentile(durations, 0.99) if durations else 0,
                avg_tokens=statistics.mean(tokens) if tokens else 0,
                total_tokens=sum(tokens),
                provider_stats=provider_summary
            )

        return summaries

    def _analyze_by_provider(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Analyze performance by provider"""
        provider_metrics = defaultdict(list)
        for metric in metrics:
            provider_metrics[metric.provider].append(metric)

        analysis = {}
        for provider, p_metrics in provider_metrics.items():
            successful = [m for m in p_metrics if m.success]
            durations = [m.duration for m in successful]
            tokens = [m.tokens_used for m in successful]

            analysis[provider] = {
                "total_requests": len(p_metrics),
                "success_rate": len(successful) / len(p_metrics) if p_metrics else 0,
                "avg_duration": statistics.mean(durations) if durations else 0,
                "p95_duration": self._percentile(durations, 0.95) if durations else 0,
                "avg_tokens": statistics.mean(tokens) if tokens else 0,
                "total_tokens": sum(tokens),
                "models_used": list(set(m.model for m in p_metrics))
            }

        return analysis

    def _analyze_by_task_type(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Analyze performance by task type"""
        task_metrics = defaultdict(list)
        for metric in metrics:
            task_metrics[metric.task_type].append(metric)

        analysis = {}
        for task_type, t_metrics in task_metrics.items():
            successful = [m for m in t_metrics if m.success]
            durations = [m.duration for m in successful]

            analysis[task_type] = {
                "total_requests": len(t_metrics),
                "success_rate": len(successful) / len(t_metrics) if t_metrics else 0,
                "avg_duration": statistics.mean(durations) if durations else 0,
                "p95_duration": self._percentile(durations, 0.95) if durations else 0
            }

        return analysis

    def _analyze_trends(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        if len(metrics) < 10:
            return {"error": "Insufficient data for trend analysis"}

        # Sort by timestamp
        sorted_metrics = sorted(metrics, key=lambda x: x.timestamp)

        # Split into two halves for comparison
        mid_point = len(sorted_metrics) // 2
        first_half = sorted_metrics[:mid_point]
        second_half = sorted_metrics[mid_point:]

        first_successful = [m for m in first_half if m.success]
        second_successful = [m for m in second_half if m.success]

        first_durations = [m.duration for m in first_successful]
        second_durations = [m.duration for m in second_successful]

        return {
            "period_1": {
                "requests": len(first_half),
                "success_rate": len(first_successful) / len(first_half) if first_half else 0,
                "avg_duration": statistics.mean(first_durations) if first_durations else 0
            },
            "period_2": {
                "requests": len(second_half),
                "success_rate": len(second_successful) / len(second_half) if second_half else 0,
                "avg_duration": statistics.mean(second_durations) if second_durations else 0
            },
            "trends": {
                "success_rate_change": (len(second_successful) / len(second_half) - len(first_successful) / len(first_half)) * 100 if first_half and second_half else 0,
                "duration_change": (statistics.mean(second_durations) - statistics.mean(first_durations)) if first_durations and second_durations else 0
            }
        }

    def _generate_recommendations(self, metrics: List[PerformanceMetric]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []

        # Overall performance
        successful = [m for m in metrics if m.success]
        if len(successful) / len(metrics) < 0.9:
            recommendations.append("⚠️  Success rate below 90%. Investigate error patterns.")

        durations = [m.duration for m in successful]
        if durations and statistics.mean(durations) > 5.0:
            recommendations.append("🐌 Average response time > 5s. Consider optimization.")

        # Provider analysis
        provider_metrics = defaultdict(list)
        for metric in metrics:
            provider_metrics[metric.provider].append(metric)

        best_provider = None
        best_performance = float('inf')
        for provider, p_metrics in provider_metrics.items():
            p_successful = [m for m in p_metrics if m.success]
            if p_successful:
                avg_duration = statistics.mean([m.duration for m in p_successful])
                if avg_duration < best_performance:
                    best_performance = avg_duration
                    best_provider = provider

        if best_provider:
            recommendations.append(f"🏆 {best_provider} shows best performance ({best_performance:.2f}s avg)")

        # Error analysis
        failed = [m for m in metrics if not m.success]
        if failed:
            error_types = defaultdict(int)
            for metric in failed:
                if metric.error:
                    error_types[metric.error.split(':')[0]] += 1

            most_common_error = max(error_types.items(), key=lambda x: x[1])[0] if error_types else None
            if most_common_error:
                recommendations.append(f"🔧 Most common error: {most_common_error}")

        return recommendations

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0
        data_sorted = sorted(data)
        index = int(len(data_sorted) * percentile)
        return data_sorted[min(index, len(data_sorted) - 1)]

    def save_analysis(self, analysis: Dict[str, Any], filename: Optional[str] = None):
        """Save analysis to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_analysis_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)

        logger.info(f"Performance analysis saved to {filename}")

    def generate_performance_report(self, hours: int = 24) -> str:
        """Generate human-readable performance report"""
        analysis = self.analyze_performance(hours)

        if "error" in analysis:
            return f"Analysis Error: {analysis['error']}"

        report = []
        report.append("# Conjecture Performance Analysis Report\n")
        report.append(f"**Period**: {analysis['analysis_period']}")
        report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Overall performance
        overall = analysis['overall']
        report.append("## 📊 Overall Performance\n")
        report.append(f"- **Total Requests**: {overall['total_requests']:,}")
        report.append(f"- **Success Rate**: {overall['success_rate']:.1%}")
        report.append(f"- **Average Duration**: {overall['avg_duration']:.3f}s")
        report.append(f"- **95th Percentile**: {overall['p95_duration']:.3f}s")
        report.append(f"- **Average Tokens**: {overall['avg_tokens']:,}")
        report.append(f"- **Total Tokens**: {overall['total_tokens']:,}\n")

        # Provider performance
        if analysis['by_provider']:
            report.append("## 🏢 Provider Performance\n")
            for provider, stats in analysis['by_provider'].items():
                report.append(f"### {provider}")
                report.append(f"- **Requests**: {stats['total_requests']}")
                report.append(f"- **Success Rate**: {stats['success_rate']:.1%}")
                report.append(f"- **Avg Duration**: {stats['avg_duration']:.3f}s")
                report.append(f"- **Models**: {', '.join(stats['models_used'])}\n")

        # Operation performance
        if analysis['by_operation']:
            report.append("## ⚙️ Operation Performance\n")
            for operation, summary in analysis['by_operation'].items():
                report.append(f"### {operation}")
                report.append(f"- **Requests**: {summary.total_requests}")
                report.append(f"- **Success Rate**: {summary.success_rate:.1%}")
                report.append(f"- **Avg Duration**: {summary.avg_duration:.3f}s")
                report.append(f"- **95th Percentile**: {summary.p95_duration:.3f}s\n")

        # Recommendations
        if analysis.get('recommendations'):
            report.append("## 💡 Recommendations\n")
            for rec in analysis['recommendations']:
                report.append(f"- {rec}")

        return '\n'.join(report)

# Global performance analyzer instance
performance_analyzer = PerformanceAnalyzer()