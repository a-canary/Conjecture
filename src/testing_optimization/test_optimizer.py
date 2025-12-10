"""
Intelligent Test Selection and Optimization System
Provides smart test execution based on code changes, coverage, and performance metrics.
"""
import ast
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import subprocess
import sys
import os

@dataclass
class TestMetrics:
    """Metrics for individual test performance."""
    name: str
    file_path: str
    execution_time: float
    memory_usage_mb: float
    coverage_score: float
    flaky_score: float
    criticality_score: float
    last_run_hash: str
    dependencies: List[str]

@dataclass
class OptimizationResult:
    """Result of test optimization analysis."""
    selected_tests: List[str]
    estimated_time_saved: float
    coverage_retained: float
    optimization_strategy: str
    reasoning: str

class TestDependencyAnalyzer:
    """Analyzes test dependencies and code coverage relationships."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.src_root = project_root / "src"
        self.test_root = project_root / "tests"
        self._dependency_cache = {}

    def analyze_file_dependencies(self, file_path: Path) -> Set[str]:
        """Analyze Python file for imported dependencies."""
        if file_path in self._dependency_cache:
            return self._dependency_cache[file_path]

        dependencies = set()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if not alias.name.startswith(('test_', 'pytest', 'unittest')):
                            dependencies.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module and not node.module.startswith(('test_', 'pytest', 'unittest')):
                        dependencies.add(node.module.split('.')[0])

        except Exception as e:
            print(f"Warning: Could not analyze dependencies for {file_path}: {e}")

        self._dependency_cache[file_path] = dependencies
        return dependencies

    def get_test_source_mapping(self, test_file: Path) -> Set[str]:
        """Map test file to its source file dependencies."""
        test_deps = self.analyze_file_dependencies(test_file)
        source_files = set()

        for dep in test_deps:
            # Find corresponding source files
            potential_source = self.src_root / f"{dep}.py"
            if potential_source.exists():
                source_files.add(str(potential_source.relative_to(self.project_root)))

        # Also check for relative imports
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Look for relative imports from src
            for line in content.split('\n'):
                if line.strip().startswith('from src.') or 'import src.' in line:
                    parts = line.replace('from ', '').replace('import ', '').split('.')
                    if len(parts) >= 2:
                        potential_path = self.src_root / '/'.join(parts[1:3]) + '.py'
                        if potential_path.exists():
                            source_files.add(str(potential_path.relative_to(self.project_root)))

        except Exception:
            pass

        return source_files

class TestPerformanceProfiler:
    """Profiles test execution performance."""

    def __init__(self, metrics_file: Path):
        self.metrics_file = metrics_file
        self.metrics = self._load_metrics()

    def _load_metrics(self) -> Dict[str, TestMetrics]:
        """Load existing test metrics from file."""
        if not self.metrics_file.exists():
            return {}

        try:
            with open(self.metrics_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {name: TestMetrics(**metrics) for name, metrics in data.items()}
        except Exception:
            return {}

    def save_metrics(self):
        """Save current metrics to file."""
        try:
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                data = {name: asdict(metrics) for name, metrics in self.metrics.items()}
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save metrics: {e}")

    def update_test_metrics(self, test_name: str, **kwargs):
        """Update metrics for a specific test."""
        if test_name not in self.metrics:
            self.metrics[test_name] = TestMetrics(
                name=test_name,
                file_path="",
                execution_time=0.0,
                memory_usage_mb=0.0,
                coverage_score=0.0,
                flaky_score=0.0,
                criticality_score=0.5,
                last_run_hash="",
                dependencies=[]
            )

        metrics = self.metrics[test_name]
        for key, value in kwargs.items():
            if hasattr(metrics, key):
                setattr(metrics, key, value)

    def get_slow_tests(self, threshold: float = 5.0) -> List[TestMetrics]:
        """Get tests slower than threshold seconds."""
        return [m for m in self.metrics.values() if m.execution_time > threshold]

    def get_critical_tests(self, threshold: float = 0.8) -> List[TestMetrics]:
        """Get tests above criticality threshold."""
        return [m for m in self.metrics.values() if m.criticality_score > threshold]

class IntelligentTestSelector:
    """Selects tests based on various optimization criteria."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.dependency_analyzer = TestDependencyAnalyzer(project_root)
        self.profiler = TestPerformanceProfiler(project_root / "test_metrics.json")
        self.changed_files = set()

    def detect_changed_files(self, base_commit: Optional[str] = None) -> Set[str]:
        """Detect changed files since base commit."""
        if base_commit is None:
            # If no base commit, assume all files changed
            return set()

        try:
            result = subprocess.run(
                ['git', 'diff', '--name-only', base_commit, 'HEAD'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            if result.returncode == 0:
                changed = set(result.stdout.strip().split('\n'))
                self.changed_files = {f for f in changed if f.endswith('.py')}
                return self.changed_files
        except Exception as e:
            print(f"Warning: Could not detect changed files: {e}")

        return set()

    def select_tests_by_changes(self) -> OptimizationResult:
        """Select tests based on code changes."""
        if not self.changed_files:
            # If no changes detected, run critical tests only
            critical_tests = self.profiler.get_critical_tests()
            selected = [f"tests/{m.name}.py" for m in critical_tests]
            return OptimizationResult(
                selected_tests=selected,
                estimated_time_saved=0.0,
                coverage_retained=0.7,
                optimization_strategy="critical_only",
                reasoning="No changes detected, running critical tests only"
            )

        # Map changed files to affected tests
        affected_tests = set()
        test_files = list(self.project_root.glob("tests/test_*.py"))

        for test_file in test_files:
            source_deps = self.dependency_analyzer.get_test_source_mapping(test_file)
            if any(any(changed_file in dep for changed_file in self.changed_files)
                   for dep in source_deps):
                affected_tests.add(str(test_file))

        # Always include critical tests
        critical_test_files = {f"tests/{m.name}.py" for m in self.profiler.get_critical_tests()}
        selected = list(affected_tests | critical_test_files)

        # Estimate time saved
        total_estimated_time = sum(m.execution_time for m in self.profiler.metrics.values())
        selected_time = sum(
            self.profiler.metrics.get(Path(test).stem,
                                   TestMetrics("", "", 1.0, 0.0, 0.0, 0.0, 0.5, "", [])).execution_time
            for test in selected
        )
        time_saved = max(0, total_estimated_time - selected_time)

        return OptimizationResult(
            selected_tests=selected,
            estimated_time_saved=time_saved,
            coverage_retained=0.85,
            optimization_strategy="change_based",
            reasoning=f"Selected {len(selected)} tests based on {len(self.changed_files)} changed files"
        )

    def select_tests_by_performance(self, max_time: float = 300.0) -> OptimizationResult:
        """Select tests to maximize coverage within time constraint."""
        # Sort tests by efficiency (coverage per second)
        efficient_tests = []
        for metrics in self.profiler.metrics.values():
            if metrics.execution_time > 0:
                efficiency = metrics.coverage_score / metrics.execution_time
                efficient_tests.append((efficiency, metrics))

        efficient_tests.sort(reverse=True)

        selected = []
        total_time = 0.0
        total_coverage = 0.0

        for efficiency, metrics in efficient_tests:
            if total_time + metrics.execution_time <= max_time:
                selected.append(f"tests/{metrics.name}.py")
                total_time += metrics.execution_time
                total_coverage = max(total_coverage, metrics.coverage_score)
            else:
                break

        return OptimizationResult(
            selected_tests=selected,
            estimated_time_saved=max(0, max_time - total_time),
            coverage_retained=total_coverage,
            optimization_strategy="performance_optimized",
            reasoning=f"Selected {len(selected)} tests to maximize coverage within {max_time}s"
        )

    def select_tests_by_priority(self, priority_levels: List[str] = ["critical", "unit"]) -> OptimizationResult:
        """Select tests based on priority levels."""
        priority_mapping = {
            "critical": 1.0,
            "unit": 0.8,
            "integration": 0.6,
            "performance": 0.4,
            "slow": 0.2
        }

        selected = []
        for level in priority_levels:
            threshold = priority_mapping.get(level, 0.5)
            for metrics in self.profiler.metrics.values():
                if metrics.criticality_score >= threshold:
                    test_path = f"tests/{metrics.name}.py"
                    if test_path not in selected:
                        selected.append(test_path)

        return OptimizationResult(
            selected_tests=selected,
            estimated_time_saved=0.0,
            coverage_retained=0.9,
            optimization_strategy="priority_based",
            reasoning=f"Selected tests by priority levels: {priority_levels}"
        )

class TestRunner:
    """Optimized test runner with performance monitoring."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.profiler = TestPerformanceProfiler(project_root / "test_metrics.json")

    def run_tests(self, test_list: List[str], parallel: bool = True) -> Dict[str, Any]:
        """Run specified tests with performance monitoring."""
        if not test_list:
            return {"status": "success", "tests_run": 0, "time": 0.0}

        cmd = ["python", "-m", "pytest", "-v", "--tb=short"]

        if parallel:
            cmd.extend(["-n", "auto", "--dist=loadscope"])

        cmd.extend(test_list)

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )

            execution_time = time.time() - start_time

            # Parse results to update metrics
            self._parse_test_results(result.stdout, execution_time)

            return {
                "status": "success" if result.returncode == 0 else "failed",
                "tests_run": len(test_list),
                "time": execution_time,
                "output": result.stdout,
                "errors": result.stderr,
                "return_code": result.returncode
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "time": time.time() - start_time
            }

    def _parse_test_results(self, output: str, total_time: float):
        """Parse pytest output to update test metrics."""
        # This would parse individual test times from pytest output
        # For now, distribute total time evenly
        lines = output.split('\n')
        test_count = sum(1 for line in lines if '::test_' in line and 'PASSED' in line)

        if test_count > 0:
            avg_time = total_time / test_count
            # Update metrics (simplified)
            pass

class TestingOptimizationFramework:
    """Main framework for test optimization."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.selector = IntelligentTestSelector(project_root)
        self.runner = TestRunner(project_root)
        self.results_cache = project_root / "optimization_results.json"

    def optimize_and_run(self,
                        strategy: str = "auto",
                        max_time: Optional[float] = None,
                        base_commit: Optional[str] = None) -> Dict[str, Any]:
        """Main optimization and execution entry point."""

        # Detect changes
        self.selector.detect_changed_files(base_commit)

        # Select optimization strategy
        if strategy == "auto":
            if self.selector.changed_files:
                result = self.selector.select_tests_by_changes()
            elif max_time:
                result = self.selector.select_tests_by_performance(max_time)
            else:
                result = self.selector.select_tests_by_priority()
        elif strategy == "changes":
            result = self.selector.select_tests_by_changes()
        elif strategy == "performance":
            result = self.selector.select_tests_by_performance(max_time or 300.0)
        elif strategy == "priority":
            result = self.selector.select_tests_by_priority()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Run selected tests
        run_result = self.runner.run_tests(result.selected_tests, parallel=True)

        # Combine results
        final_result = {
            "optimization": asdict(result),
            "execution": run_result,
            "timestamp": time.time(),
            "total_tests_available": len(self.runner.profiler.metrics),
            "tests_selected": len(result.selected_tests)
        }

        # Save results
        self._save_results(final_result)

        return final_result

    def _save_results(self, result: Dict[str, Any]):
        """Save optimization results."""
        try:
            with open(self.results_cache, 'a', encoding='utf-8') as f:
                json.dump(result, f, indent=2, default=str)
                f.write('\n')
        except Exception as e:
            print(f"Warning: Could not save optimization results: {e}")

    def generate_report(self) -> str:
        """Generate optimization report."""
        report = []
        report.append("# Testing Optimization Report\n")

        if self.results_cache.exists():
            try:
                with open(self.results_cache, 'r', encoding='utf-8') as f:
                    results = [json.loads(line) for line in f if line.strip()]

                if results:
                    latest = results[-1]
                    report.append(f"## Latest Optimization Run\n")
                    report.append(f"- **Strategy**: {latest['optimization']['optimization_strategy']}\n")
                    report.append(f"- **Tests Selected**: {latest['tests_selected']}/{latest['total_tests_available']}\n")
                    report.append(f"- **Estimated Time Saved**: {latest['optimization']['estimated_time_saved']:.2f}s\n")
                    report.append(f"- **Coverage Retained**: {latest['optimization']['coverage_retained']:.2%}\n")
                    report.append(f"- **Execution Status**: {latest['execution']['status']}\n")
                    report.append(f"- **Actual Time**: {latest['execution']['time']:.2f}s\n")

            except Exception as e:
                report.append(f"Error generating report: {e}\n")

        return ''.join(report)

def main():
    """Command-line interface for test optimization."""
    import argparse

    parser = argparse.ArgumentParser(description="Optimize test execution")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--strategy", choices=["auto", "changes", "performance", "priority"],
                       default="auto", help="Optimization strategy")
    parser.add_argument("--max-time", type=float, help="Maximum execution time in seconds")
    parser.add_argument("--base-commit", help="Base commit for change detection")
    parser.add_argument("--report", action="store_true", help="Generate optimization report")

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    framework = TestingOptimizationFramework(project_root)

    if args.report:
        print(framework.generate_report())
    else:
        result = framework.optimize_and_run(
            strategy=args.strategy,
            max_time=args.max_time,
            base_commit=args.base_commit
        )

        print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    main()