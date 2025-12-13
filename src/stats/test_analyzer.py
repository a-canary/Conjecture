#!/usr/bin/env python3
"""
Test Results Analyzer
Analyzes test results, coverage, and performance metrics
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

class TestAnalyzer:
    """Analyzes test results and coverage"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_dir = project_root / "tests"

    def analyze_tests(self) -> Dict[str, Any]:
        """Perform comprehensive test analysis"""
        analysis = {
            "analysis_timestamp": time.time(),
            "test_collection": self.analyze_test_collection(),
            "test_execution": self.analyze_test_execution(),
            "test_categories": self.analyze_test_categories(),
            "performance_metrics": self.analyze_test_performance()
        }

        return analysis

    def analyze_test_collection(self) -> Dict[str, Any]:
        """Analyze test collection with improved parsing"""
        collection = {
            "status": "unknown",
            "total_tests": 0,
            "collection_errors": [],
            "test_modules": [],
            "test_classes": [],
            "collection_success_rate": 0
        }

        try:
            # Run pytest collection with improved output parsing
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                "--collect-only",
                "--quiet",
                "-v"
            ], capture_output=True, text=True, cwd=self.project_root, timeout=60)

            collection["returncode"] = result.returncode
            collection["stdout_sample"] = result.stdout[:500] if result.stdout else ""
            collection["stderr_sample"] = result.stderr[:500] if result.stderr else ""

            # Consider collection successful if returncode is 0 or 1 (some pytest versions return 1 for collection warnings)
            if result.returncode in [0, 1]:
                collection["status"] = "success"
                output = result.stdout + result.stderr

                # Enhanced test count extraction
                import re
                patterns = [
                    r'(\d+)\s+items?\s+collected',
                    r'collected\s+(\d+)\s+items?',
                    r'(\d+)\s+test(s)?\s+found',
                    r'Test\s+session\s+starts.*?(\d+)\s+items?'
                ]

                for pattern in patterns:
                    match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
                    if match:
                        try:
                            collection["total_tests"] = int(match.group(1))
                            break
                        except:
                            pass

                # Parse test modules and classes
                module_pattern = r'<Module\s+(.*?\.py)>'
                class_pattern = r'<Class\s+(.*?)>'

                modules = re.findall(module_pattern, output)
                classes = re.findall(class_pattern, output)

                collection["test_modules"] = list(set(modules))
                collection["test_classes"] = list(set(classes))

                # Calculate collection success rate based on modules found
                if collection["test_modules"]:
                    collection["collection_success_rate"] = 100
                else:
                    collection["collection_success_rate"] = 0

            else:
                collection["status"] = "error"
                collection["collection_errors"] = result.stderr.split('\n')[:10]
                collection["collection_success_rate"] = 0

        except subprocess.TimeoutExpired:
            collection["status"] = "timeout"
            collection["collection_errors"].append("Test collection timed out after 60 seconds")
            collection["collection_success_rate"] = 0
        except Exception as e:
            collection["status"] = "error"
            collection["collection_errors"].append(f"Unexpected error: {str(e)}")
            collection["collection_success_rate"] = 0

        return collection

    def analyze_test_execution(self) -> Dict[str, Any]:
        """Analyze test execution for core tests with improved parsing"""
        execution = {
            "status": "unknown",
            "tests_run": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "execution_time": 0,
            "execution_errors": [],
            "pass_rate": 0.0,
            "tests_attempted": []
        }

        # Select core tests for quick analysis
        core_tests = [
            "tests/test_claim_models.py",
            "tests/test_id_utilities.py",
            "tests/test_claim_processing.py"
        ]

        # Check which tests exist
        existing_tests = []
        for test_file in core_tests:
            test_path = self.project_root / test_file
            if test_path.exists():
                existing_tests.append(test_file)

        if not existing_tests:
            execution["status"] = "no_tests_found"
            execution["execution_errors"].append("No core test files found")
            return execution

        execution["tests_attempted"] = existing_tests

        try:
            start_time = time.time()
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                *existing_tests,
                "-v",
                "--tb=line",
                "--no-header"
            ], capture_output=True, text=True, cwd=self.project_root, timeout=30)

            execution["execution_time"] = round(time.time() - start_time, 2)
            execution["returncode"] = result.returncode
            execution["stdout_sample"] = result.stdout[:300] if result.stdout else ""
            execution["stderr_sample"] = result.stderr[:300] if result.stderr else ""

            # Parse pytest output with enhanced patterns
            output = result.stdout + result.stderr
            import re

            # Try multiple patterns for test results
            patterns = [
                r'(\d+)\s+passed(?:,\s*(\d+)\s+failed)?(?:,\s*(\d+)\s+skipped)?(?:,\s*(\d+)\s+error(?:s)?)?',
                r'passed\s*=\s*(\d+).*?failed\s*=\s*(\d+).*?skipped\s*=\s*(\d+).*?error\s*=\s*(\d+)',
                r'PASSED\s*(\d+).*?FAILED\s*(\d+).*?SKIPPED\s*(\d+).*?ERROR\s*(\d+)',
                r'(\d+)\s+passed\s+in\s+[\d.]+s',
                r'test session starts.*?(?:\d+\s+passed|\d+\s+failed)'
            ]

            for pattern in patterns:
                match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
                if match:
                    try:
                        groups = match.groups()
                        if len(groups) >= 1:
                            execution["passed"] = int(groups[0])
                        if len(groups) >= 2 and groups[1]:
                            execution["failed"] = int(groups[1])
                        if len(groups) >= 3 and groups[2]:
                            execution["skipped"] = int(groups[2])
                        if len(groups) >= 4 and groups[3]:
                            execution["errors"] = int(groups[3])
                        break
                    except (ValueError, IndexError):
                        continue

            # Alternative parsing: count individual test result lines
            if execution["passed"] == 0 and execution["failed"] == 0:
                passed_count = len(re.findall(r'::.*PASSED', output))
                failed_count = len(re.findall(r'::.*FAILED', output))
                skipped_count = len(re.findall(r'::.*SKIPPED', output))
                error_count = len(re.findall(r'::.*ERROR', output))

                execution["passed"] = passed_count
                execution["failed"] = failed_count
                execution["skipped"] = skipped_count
                execution["errors"] = error_count

            execution["tests_run"] = execution["passed"] + execution["failed"] + execution["skipped"] + execution["errors"]

            # Calculate pass rate
            if execution["tests_run"] > 0:
                execution["pass_rate"] = round((execution["passed"] / execution["tests_run"]) * 100, 1)

            # Determine status
            if result.returncode == 0:
                execution["status"] = "success"
            elif execution["tests_run"] > 0:
                execution["status"] = "partial_success"
            else:
                execution["status"] = "failed"

        except subprocess.TimeoutExpired:
            execution["status"] = "timeout"
            execution["execution_errors"].append("Test execution timed out after 30 seconds")
        except Exception as e:
            execution["status"] = "error"
            execution["execution_errors"].append(f"Unexpected error: {str(e)}")

        return execution

    def analyze_test_categories(self) -> Dict[str, Any]:
        """Analyze test categories by examining test files"""
        categories = {
            "unit_tests": {"files": [], "count": 0},
            "integration_tests": {"files": [], "count": 0},
            "e2e_tests": {"files": [], "count": 0},
            "performance_tests": {"files": [], "count": 0},
            "other_tests": {"files": [], "count": 0}
        }

        if not self.test_dir.exists():
            return categories

        try:
            test_files = list(self.test_dir.glob("test_*.py"))

            for test_file in test_files:
                file_name = test_file.name.lower()

                if any(keyword in file_name for keyword in ['unit', 'model', 'util']):
                    categories["unit_tests"]["files"].append(test_file.name)
                elif any(keyword in file_name for keyword in ['integration', 'repo', 'database']):
                    categories["integration_tests"]["files"].append(test_file.name)
                elif any(keyword in file_name for keyword in ['e2e', 'lifecycle', 'workflow']):
                    categories["e2e_tests"]["files"].append(test_file.name)
                elif any(keyword in file_name for keyword in ['performance', 'benchmark', 'speed']):
                    categories["performance_tests"]["files"].append(test_file.name)
                else:
                    categories["other_tests"]["files"].append(test_file.name)

            # Count files in each category
            for category in categories:
                categories[category]["count"] = len(categories[category]["files"])

        except Exception as e:
            categories["error"] = str(e)

        return categories

    def analyze_test_performance(self) -> Dict[str, Any]:
        """Analyze test performance metrics"""
        performance = {
            "fast_test_threshold": 1.0,  # seconds
            "slow_test_threshold": 10.0,  # seconds
            "slow_tests": [],
            "test_speed_distribution": {"fast": 0, "medium": 0, "slow": 0}
        }

        # Try to run tests with timing (if available)
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                "tests/test_claim_models.py",
                "--tb=no",
                "--durations=0"
            ], capture_output=True, text=True, cwd=self.project_root, timeout=30)

            if result.returncode == 0:
                output = result.stdout
                lines = output.split('\n')

                # Parse durations
                for line in lines:
                    if 'test_' in line and 'sec' in line.lower():
                        try:
                            # Extract test name and duration
                            parts = line.split()
                            if len(parts) >= 3 and parts[-1].lower().endswith('sec'):
                                duration = float(parts[-2])
                                test_name = ' '.join(parts[:-2])

                                if duration > performance["slow_test_threshold"]:
                                    performance["slow_tests"].append({
                                        "name": test_name,
                                        "duration": duration
                                    })

                                # Categorize speed
                                if duration <= performance["fast_test_threshold"]:
                                    performance["test_speed_distribution"]["fast"] += 1
                                elif duration <= performance["slow_test_threshold"]:
                                    performance["test_speed_distribution"]["medium"] += 1
                                else:
                                    performance["test_speed_distribution"]["slow"] += 1

                        except (ValueError, IndexError):
                            continue

        except Exception as e:
            performance["analysis_error"] = str(e)

        return performance

    def get_test_health_score(self) -> Dict[str, Any]:
        """Calculate overall test health score"""
        health_score = {
            "overall_score": 0,
            "collection_health": 0,
            "execution_health": 0,
            "coverage_health": 0,
            "factors": {}
        }

        # Collection health
        collection = self.analyze_test_collection()
        if collection["status"] == "success":
            collection_health = 100
        elif collection["total_tests"] > 0:
            collection_health = 50  # Some tests collected but with warnings
        else:
            collection_health = 0

        health_score["collection_health"] = collection_health
        health_score["factors"]["collection"] = collection_health

        # Execution health
        execution = self.analyze_test_execution()
        if execution["status"] == "success":
            if execution["tests_run"] > 0:
                pass_rate = execution["passed"] / execution["tests_run"] * 100
                execution_health = pass_rate
            else:
                execution_health = 100  # No tests to fail
        else:
            execution_health = 0

        health_score["execution_health"] = execution_health
        health_score["factors"]["execution"] = execution_health

        # Coverage health (estimated from test collection)
        coverage_health = min(100, collection["total_tests"] * 2)  # Rough estimate
        health_score["coverage_health"] = coverage_health
        health_score["factors"]["coverage"] = coverage_health

        # Overall score (weighted average)
        weights = {"collection": 0.2, "execution": 0.5, "coverage": 0.3}
        overall = (weights["collection"] * collection_health +
                  weights["execution"] * execution_health +
                  weights["coverage"] * coverage_health)

        health_score["overall_score"] = round(overall, 1)

        return health_score