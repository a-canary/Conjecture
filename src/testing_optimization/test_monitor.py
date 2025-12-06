"""
Comprehensive Test Monitoring and Reporting System
Provides real-time monitoring, performance analysis, and detailed reporting.
"""
import asyncio
import json
import time
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
import psutil
import statistics
import traceback
from datetime import datetime, timedelta


@dataclass
class TestExecutionMetric:
    """Metrics for a single test execution."""
    test_name: str
    test_file: str
    start_time: float
    end_time: float
    duration: float
    status: str  # "passed", "failed", "skipped", "error"
    memory_usage_mb: float
    cpu_usage_percent: float
    error_message: Optional[str] = None
    traceback_info: Optional[str] = None
    markers: List[str] = field(default_factory=list)
    coverage_lines: int = 0
    coverage_percentage: float = 0.0


@dataclass
class TestSessionMetrics:
    """Metrics for a complete test session."""
    session_id: str
    start_time: float
    end_time: float
    total_duration: float
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    error_tests: int
    total_memory_usage_mb: float
    peak_memory_usage_mb: float
    average_cpu_usage: float
    peak_cpu_usage: float
    test_metrics: List[TestExecutionMetric] = field(default_factory=list)
    coverage_overall: float = 0.0
    performance_score: float = 0.0
    optimization_score: float = 0.0


class RealTimeMonitor:
    """Real-time monitoring of test execution."""

    def __init__(self, update_interval: float = 0.5):
        self.update_interval = update_interval
        self.monitoring = False
        self.metrics_history: deque = deque(maxlen=1000)
        self.callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def add_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for real-time updates."""
        self.callbacks.append(callback)

    def start_monitoring(self):
        """Start real-time monitoring."""
        if self.monitoring:
            return

        self.monitoring = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.start()

    def stop_monitoring(self):
        """Stop real-time monitoring."""
        if not self.monitoring:
            return

        self.monitoring = False
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join()

    def _monitor_loop(self):
        """Main monitoring loop."""
        process = psutil.Process()

        while self.monitoring and not self._stop_event.is_set():
            try:
                # Get system metrics
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024

                # Get system-wide metrics
                system_cpu = psutil.cpu_percent()
                system_memory = psutil.virtual_memory().percent

                metrics = {
                    "timestamp": time.time(),
                    "process_cpu": cpu_percent,
                    "process_memory_mb": memory_mb,
                    "system_cpu": system_cpu,
                    "system_memory_percent": system_memory,
                    "threads": process.num_threads(),
                    "open_files": len(process.open_files())
                }

                self.metrics_history.append(metrics)

                # Notify callbacks
                for callback in self.callbacks:
                    try:
                        callback(metrics)
                    except Exception:
                        pass

            except Exception:
                pass

            time.sleep(self.update_interval)

    def get_current_metrics(self) -> Optional[Dict[str, Any]]:
        """Get most recent metrics."""
        return self.metrics_history[-1] if self.metrics_history else None

    def get_average_metrics(self, duration_seconds: float = 60.0) -> Dict[str, float]:
        """Get average metrics over specified duration."""
        if not self.metrics_history:
            return {}

        cutoff_time = time.time() - duration_seconds
        recent_metrics = [
            m for m in self.metrics_history
            if m["timestamp"] > cutoff_time
        ]

        if not recent_metrics:
            return {}

        return {
            "avg_process_cpu": statistics.mean(m["process_cpu"] for m in recent_metrics),
            "avg_process_memory_mb": statistics.mean(m["process_memory_mb"] for m in recent_metrics),
            "avg_system_cpu": statistics.mean(m["system_cpu"] for m in recent_metrics),
            "avg_system_memory_percent": statistics.mean(m["system_memory_percent"] for m in recent_metrics),
            "peak_process_memory_mb": max(m["process_memory_mb"] for m in recent_metrics),
            "peak_process_cpu": max(m["process_cpu"] for m in recent_metrics)
        }


class TestPerformanceAnalyzer:
    """Analyzes test performance and identifies optimization opportunities."""

    def __init__(self):
        self.performance_history: List[TestSessionMetrics] = []

    def analyze_test_session(self, session_metrics: TestSessionMetrics) -> Dict[str, Any]:
        """Analyze a test session for performance insights."""
        analysis = {
            "session_overview": {
                "total_tests": session_metrics.total_tests,
                "success_rate": session_metrics.passed_tests / max(session_metrics.total_tests, 1),
                "average_test_time": session_metrics.total_duration / max(session_metrics.total_tests, 1),
                "total_duration": session_metrics.total_duration
            },
            "performance_analysis": {},
            "optimization_suggestions": [],
            "problematic_tests": []
        }

        # Analyze test performance
        test_times = [m.duration for m in session_metrics.test_metrics]
        memory_usage = [m.memory_usage_mb for m in session_metrics.test_metrics]

        if test_times:
            analysis["performance_analysis"] = {
                "fastest_test": min(test_times),
                "slowest_test": max(test_times),
                "average_test_time": statistics.mean(test_times),
                "median_test_time": statistics.median(test_times),
                "test_time_std": statistics.stdev(test_times) if len(test_times) > 1 else 0,
                "memory_stats": {
                    "average_mb": statistics.mean(memory_usage),
                    "peak_mb": max(memory_usage),
                    "memory_std": statistics.stdev(memory_usage) if len(memory_usage) > 1 else 0
                }
            }

        # Identify slow tests
        slow_threshold = statistics.mean(test_times) + statistics.stdev(test_times) if len(test_times) > 1 else test_times[0] if test_times else 0
        slow_tests = [m for m in session_metrics.test_metrics if m.duration > slow_threshold]
        analysis["problematic_tests"] = [
            {
                "name": m.test_name,
                "duration": m.duration,
                "memory_mb": m.memory_usage_mb,
                "issue": "slow_execution"
            }
            for m in slow_tests
        ]

        # Identify memory-intensive tests
        memory_threshold = statistics.mean(memory_usage) + statistics.stdev(memory_usage) if len(memory_usage) > 1 else memory_usage[0] if memory_usage else 0
        memory_intensive_tests = [m for m in session_metrics.test_metrics if m.memory_usage_mb > memory_threshold]
        analysis["problematic_tests"].extend([
            {
                "name": m.test_name,
                "duration": m.duration,
                "memory_mb": m.memory_usage_mb,
                "issue": "memory_intensive"
            }
            for m in memory_intensive_tests if m not in slow_tests
        ])

        # Generate optimization suggestions
        if slow_tests:
            analysis["optimization_suggestions"].append(
                f"Consider optimizing {len(slow_tests)} slow tests (> {slow_threshold:.2f}s)"
            )

        if memory_intensive_tests:
            analysis["optimization_suggestions"].append(
                f"Review memory usage in {len(memory_intensive_tests)} tests (> {memory_threshold:.2f}MB)"
            )

        if session_metrics.failed_tests > 0:
            analysis["optimization_suggestions"].append(
                f"Fix {session_metrics.failed_tests} failing tests for better reliability"
            )

        # Calculate performance score
        performance_score = self._calculate_performance_score(session_metrics)
        analysis["performance_score"] = performance_score

        return analysis

    def _calculate_performance_score(self, session_metrics: TestSessionMetrics) -> float:
        """Calculate overall performance score (0-100)."""
        if session_metrics.total_tests == 0:
            return 0.0

        # Success rate (40% weight)
        success_rate = session_metrics.passed_tests / session_metrics.total_tests

        # Speed score (30% weight) - tests should complete quickly
        avg_time = session_metrics.total_duration / session_metrics.total_tests
        speed_score = max(0, 1 - (avg_time / 10.0))  # Normalize to 10 seconds as ideal

        # Memory efficiency (20% weight)
        memory_efficiency = max(0, 1 - (session_metrics.peak_memory_usage_mb / 500.0))  # 500MB as high water mark

        # Reliability score (10% weight) - no errors or crashes
        reliability_score = (session_metrics.passed_tests + session_metrics.skipped_tests) / session_metrics.total_tests

        performance_score = (
            success_rate * 40 +
            speed_score * 30 +
            memory_efficiency * 20 +
            reliability_score * 10
        )

        return min(100, performance_score)

    def compare_sessions(self, session1: TestSessionMetrics, session2: TestSessionMetrics) -> Dict[str, Any]:
        """Compare two test sessions and identify improvements or regressions."""
        comparison = {
            "summary": {},
            "improvements": [],
            "regressions": [],
            "unchanged": []
        }

        # Compare success rates
        success_rate_1 = session1.passed_tests / max(session1.total_tests, 1)
        success_rate_2 = session2.passed_tests / max(session2.total_tests, 1)
        success_diff = success_rate_2 - success_rate_1

        if abs(success_diff) > 0.05:  # 5% threshold
            if success_diff > 0:
                comparison["improvements"].append(f"Success rate improved by {success_diff:.1%}")
            else:
                comparison["regressions"].append(f"Success rate decreased by {-success_diff:.1%}")
        else:
            comparison["unchanged"].append("Success rate stable")

        # Compare execution times
        time_per_test_1 = session1.total_duration / max(session1.total_tests, 1)
        time_per_test_2 = session2.total_duration / max(session2.total_tests, 1)
        time_diff_pct = (time_per_test_2 - time_per_test_1) / time_per_test_1

        if abs(time_diff_pct) > 0.10:  # 10% threshold
            if time_diff_pct < 0:
                comparison["improvements"].append(f"Test speed improved by {-time_diff_pct:.1%}")
            else:
                comparison["regressions"].append(f"Test speed decreased by {time_diff_pct:.1%}")
        else:
            comparison["unchanged"].append("Test speed stable")

        # Compare memory usage
        memory_diff = session2.peak_memory_usage_mb - session1.peak_memory_usage_mb
        if abs(memory_diff) > 50:  # 50MB threshold
            if memory_diff < 0:
                comparison["improvements"].append(f"Memory usage reduced by {-memory_diff:.1f}MB")
            else:
                comparison["regressions"].append(f"Memory usage increased by {memory_diff:.1f}MB")
        else:
            comparison["unchanged"].append("Memory usage stable")

        comparison["summary"] = {
            "success_rate_change": success_diff,
            "speed_change_percent": time_diff_pct,
            "memory_change_mb": memory_diff,
            "overall_improvement": len(comparison["improvements"]) > len(comparison["regressions"])
        }

        return comparison


class TestReportGenerator:
    """Generates comprehensive test reports."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def generate_session_report(self, session_metrics: TestSessionMetrics,
                              performance_analysis: Dict[str, Any]) -> Path:
        """Generate detailed HTML report for a test session."""
        report_file = self.output_dir / f"test_report_{session_metrics.session_id}.html"

        html_content = self._generate_html_report(session_metrics, performance_analysis)

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return report_file

    def _generate_html_report(self, session_metrics: TestSessionMetrics,
                            performance_analysis: Dict[str, Any]) -> str:
        """Generate HTML content for test report."""
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Report - {session_metrics.session_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .metric-card {{ background-color: #f8f9fa; padding: 15px; border-radius: 6px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #007bff; }}
        .metric-label {{ color: #666; margin-top: 5px; }}
        .section {{ margin-bottom: 30px; }}
        .section h2 {{ border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        .success {{ color: #28a745; }}
        .failure {{ color: #dc3545; }}
        .warning {{ color: #ffc107; }}
        .info {{ color: #17a2b8; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ text-align: left; padding: 10px; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; }}
        .test-passed {{ background-color: #d4edda; }}
        .test-failed {{ background-color: #f8d7da; }}
        .test-skipped {{ background-color: #fff3cd; }}
        .progress-bar {{ width: 100%; background-color: #e9ecef; border-radius: 4px; overflow: hidden; }}
        .progress-fill {{ height: 20px; background-color: #007bff; transition: width 0.3s ease; }}
        .optimization-suggestion {{ background-color: #e7f3ff; border-left: 4px solid #007bff; padding: 10px; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Test Execution Report</h1>
            <p>Session ID: {session_metrics.session_id}</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value success">{session_metrics.passed_tests}</div>
                <div class="metric-label">Passed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value failure">{session_metrics.failed_tests}</div>
                <div class="metric-label">Failed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value warning">{session_metrics.skipped_tests}</div>
                <div class="metric-label">Skipped</div>
            </div>
            <div class="metric-card">
                <div class="metric-value info">{session_metrics.total_tests}</div>
                <div class="metric-label">Total</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{session_metrics.total_duration:.2f}s</div>
                <div class="metric-label">Duration</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{session_metrics.coverage_overall:.1%}</div>
                <div class="metric-label">Coverage</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{performance_analysis.get('performance_score', 0):.1f}</div>
                <div class="metric-label">Performance Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{session_metrics.peak_memory_usage_mb:.1f}MB</div>
                <div class="metric-label">Peak Memory</div>
            </div>
        </div>

        <div class="section">
            <h2>Success Rate</h2>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {session_metrics.passed_tests / max(session_metrics.total_tests, 1) * 100}%"></div>
            </div>
            <p>{session_metrics.passed_tests / max(session_metrics.total_tests, 1) * 100:.1f}% of tests passed</p>
        </div>

        <div class="section">
            <h2>Performance Analysis</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{performance_analysis.get('performance_analysis', {}).get('average_test_time', 0):.3f}s</div>
                    <div class="metric-label">Avg Test Time</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{performance_analysis.get('performance_analysis', {}).get('slowest_test', 0):.3f}s</div>
                    <div class="metric-label">Slowest Test</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{performance_analysis.get('performance_analysis', {}).get('memory_stats', {}).get('average_mb', 0):.1f}MB</div>
                    <div class="metric-label">Avg Memory</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Optimization Suggestions</h2>
            {"".join([f'<div class="optimization-suggestion">{suggestion}</div>' for suggestion in performance_analysis.get('optimization_suggestions', [])])}
        </div>

        <div class="section">
            <h2>Test Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Test Name</th>
                        <th>Duration (s)</th>
                        <th>Memory (MB)</th>
                        <th>Status</th>
                        <th>Coverage</th>
                    </tr>
                </thead>
                <tbody>
"""

        for test_metric in session_metrics.test_metrics:
            status_class = {
                "passed": "test-passed",
                "failed": "test-failed",
                "skipped": "test-skipped",
                "error": "test-failed"
            }.get(test_metric.status, "")

            html += f"""
                    <tr class="{status_class}">
                        <td>{test_metric.test_name}</td>
                        <td>{test_metric.duration:.3f}</td>
                        <td>{test_metric.memory_usage_mb:.1f}</td>
                        <td>{test_metric.status}</td>
                        <td>{test_metric.coverage_percentage:.1%}</td>
                    </tr>
"""

        html += """
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>Problematic Tests</h2>
            <table>
                <thead>
                    <tr>
                        <th>Test Name</th>
                        <th>Issue Type</th>
                        <th>Duration (s)</th>
                        <th>Memory (MB)</th>
                    </tr>
                </thead>
                <tbody>
"""

        for problematic_test in performance_analysis.get('problematic_tests', []):
            html += f"""
                    <tr>
                        <td>{problematic_test['name']}</td>
                        <td>{problematic_test['issue'].replace('_', ' ').title()}</td>
                        <td>{problematic_test['duration']:.3f}</td>
                        <td>{problematic_test['memory_mb']:.1f}</td>
                    </tr>
"""

        html += """
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""

        return html

    def generate_json_report(self, session_metrics: TestSessionMetrics,
                           performance_analysis: Dict[str, Any]) -> Path:
        """Generate JSON report for programmatic consumption."""
        report_file = self.output_dir / f"test_report_{session_metrics.session_id}.json"

        report_data = {
            "session": asdict(session_metrics),
            "analysis": performance_analysis,
            "generated_at": datetime.now().isoformat()
        }

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)

        return report_file


class ComprehensiveTestMonitor:
    """Main monitoring system that coordinates all components."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.real_time_monitor = RealTimeMonitor()
        self.performance_analyzer = TestPerformanceAnalyzer()
        self.report_generator = TestReportGenerator(output_dir)
        self.current_session: Optional[TestSessionMetrics] = None
        self.session_metrics: List[TestExecutionMetric] = []

    def start_session(self, session_id: str) -> str:
        """Start monitoring a new test session."""
        self.current_session = TestSessionMetrics(
            session_id=session_id,
            start_time=time.time(),
            end_time=0.0,
            total_duration=0.0,
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            skipped_tests=0,
            error_tests=0,
            total_memory_usage_mb=0.0,
            peak_memory_usage_mb=0.0,
            average_cpu_usage=0.0,
            peak_cpu_usage=0.0
        )

        self.session_metrics.clear()
        self.real_time_monitor.start_monitoring()

        return session_id

    def record_test_execution(self, test_metric: TestExecutionMetric):
        """Record execution of a single test."""
        if self.current_session:
            self.session_metrics.append(test_metric)

            # Update session counters
            self.current_session.total_tests += 1
            if test_metric.status == "passed":
                self.current_session.passed_tests += 1
            elif test_metric.status == "failed":
                self.current_session.failed_tests += 1
            elif test_metric.status == "skipped":
                self.current_session.skipped_tests += 1
            elif test_metric.status == "error":
                self.current_session.error_tests += 1

            # Update memory tracking
            self.current_session.total_memory_usage_mb += test_metric.memory_usage_mb
            self.current_session.peak_memory_usage_mb = max(
                self.current_session.peak_memory_usage_mb,
                test_metric.memory_usage_mb
            )

            # Update CPU tracking
            self.current_session.peak_cpu_usage = max(
                self.current_session.peak_cpu_usage,
                test_metric.cpu_usage_percent
            )

    def end_session(self) -> Dict[str, Any]:
        """End the current monitoring session and generate reports."""
        if not self.current_session:
            return {}

        self.current_session.end_time = time.time()
        self.current_session.total_duration = (
            self.current_session.end_time - self.current_session.start_time
        )
        self.current_session.test_metrics = self.session_metrics.copy()

        # Calculate average CPU usage
        if self.session_metrics:
            self.current_session.average_cpu_usage = statistics.mean(
                m.cpu_usage_percent for m in self.session_metrics
            )

        # Stop real-time monitoring
        self.real_time_monitor.stop_monitoring()

        # Generate performance analysis
        performance_analysis = self.performance_analyzer.analyze_test_session(self.current_session)

        # Calculate optimization score
        self.current_session.optimization_score = self._calculate_optimization_score(performance_analysis)

        # Generate reports
        html_report = self.report_generator.generate_session_report(self.current_session, performance_analysis)
        json_report = self.report_generator.generate_json_report(self.current_session, performance_analysis)

        # Store for historical comparison
        self.performance_analyzer.performance_history.append(self.current_session)

        return {
            "session_id": self.current_session.session_id,
            "html_report": str(html_report),
            "json_report": str(json_report),
            "performance_analysis": performance_analysis,
            "session_metrics": asdict(self.current_session)
        }

    def _calculate_optimization_score(self, performance_analysis: Dict[str, Any]) -> float:
        """Calculate optimization score based on performance analysis."""
        base_score = performance_analysis.get('performance_score', 0)

        # Bonus points for having optimization suggestions addressed
        optimization_count = len(performance_analysis.get('optimization_suggestions', []))
        suggestion_penalty = min(20, optimization_count * 5)

        return max(0, base_score - suggestion_penalty)

    def get_real_time_metrics(self) -> Optional[Dict[str, Any]]:
        """Get current real-time metrics."""
        return self.real_time_monitor.get_current_metrics()

    def add_real_time_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for real-time monitoring updates."""
        self.real_time_monitor.add_callback(callback)


# Global monitor instance
_global_monitor: Optional[ComprehensiveTestMonitor] = None


def get_test_monitor(output_dir: Optional[Path] = None) -> ComprehensiveTestMonitor:
    """Get global test monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        output_dir = output_dir or Path("test_reports")
        _global_monitor = ComprehensiveTestMonitor(output_dir)
    return _global_monitor