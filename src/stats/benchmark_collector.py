#!/usr/bin/env python3
"""
Benchmark Scores Collector
Extracts benchmark scores from all test models and configurations
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

class BenchmarkCollector:
    """Collects benchmark scores from various sources"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.benchmark_dir = project_root / "src" / "benchmarking"

    def collect_all_benchmarks(self) -> Dict[str, Any]:
        """Collect all benchmark scores"""
        benchmarks = {
            "collection_timestamp": datetime.now().isoformat(),
            "results": {}
        }

        # Collect from cycle results
        benchmarks["results"]["cycle_results"] = self.collect_cycle_results()

        # Collect from evaluation results
        benchmarks["results"]["evaluation_results"] = self.collect_evaluation_results()

        # Collect from improved claim system
        benchmarks["results"]["claim_system"] = self.collect_claim_system_results()

        # Collect from real API tests
        benchmarks["results"]["real_api_tests"] = self.collect_real_api_results()

        return benchmarks

    def collect_cycle_results(self) -> Dict[str, Any]:
        """Collect results from cycle result files"""
        cycle_results = {}
        cycle_dir = self.benchmark_dir / "cycle_results"

        if not cycle_dir.exists():
            return {"error": "Cycle results directory not found"}

        try:
            result_files = sorted(cycle_dir.glob("cycle_*_results.json"),
                                key=lambda x: x.stat().st_mtime, reverse=True)

            for result_file in result_files:
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    cycle_name = result_file.stem
                    cycle_data = {
                        "file": result_file.name,
                        "timestamp": datetime.fromtimestamp(result_file.stat().st_mtime).isoformat(),
                        "title": data.get("title", "Unknown"),
                        "success": data.get("success", False),
                        "overall_score": data.get("overall_score", "?")
                    }

                    # Extract detailed scores
                    if "scores" in data:
                        cycle_data["benchmarks"] = {}
                        for benchmark, score_data in data["scores"].items():
                            if isinstance(score_data, dict) and "overall_score" in score_data:
                                cycle_data["benchmarks"][benchmark] = score_data["overall_score"]
                            elif isinstance(score_data, (int, float)):
                                cycle_data["benchmarks"][benchmark] = score_data

                    # Extract benchmark list
                    if "benchmarks_run" in data:
                        cycle_data["benchmarks_run"] = data["benchmarks_run"]

                    cycle_results[cycle_name] = cycle_data

                except Exception as e:
                    cycle_results[result_file.name] = {"error": str(e)}

        except Exception as e:
            return {"error": f"Could not read cycle results: {e}"}

        return cycle_results

    def collect_evaluation_results(self) -> Dict[str, Any]:
        """Collect evaluation-specific results"""
        eval_results = {}

        # Look for various evaluation files
        eval_patterns = [
            "*evaluation*.json",
            "detailed_evaluation*.json",
            "real_api_*.json"
        ]

        for pattern in eval_patterns:
            try:
                files = list(self.benchmark_dir.glob(pattern))
                for file_path in files:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        file_name = file_path.name
                        eval_data = {
                            "file": file_name,
                            "timestamp": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                        }

                        # Extract performance metrics
                        if "performance_metrics" in data:
                            eval_data["performance_metrics"] = data["performance_metrics"]

                        # Extract scores
                        if "overall_score" in data:
                            eval_data["overall_score"] = data["overall_score"]

                        # Extract test problem results
                        if "test_problems" in data:
                            eval_data["problems_evaluated"] = len(data["test_problems"])

                        eval_results[file_name] = eval_data

                    except Exception as e:
                        eval_results[file_path.name] = {"error": str(e)}

            except Exception as e:
                eval_results[f"pattern_{pattern}"] = {"error": str(e)}

        return eval_results

    def collect_claim_system_results(self) -> Dict[str, Any]:
        """Collect improved claim system results"""
        claim_results = {}

        claim_files = [
            "improved_claim_system_results.json",
            "detailed_evaluation_demo_results.json"
        ]

        for claim_file in claim_files:
            file_path = self.benchmark_dir / claim_file
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    claim_data = {
                        "file": claim_file,
                        "timestamp": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    }

                    # Extract performance metrics
                    if "performance_metrics" in data:
                        perf = data["performance_metrics"]
                        claim_data["performance"] = {
                            "direct_accuracy": perf.get("direct_accuracy", "?"),
                            "conjecture_accuracy": perf.get("conjecture_accuracy", "?"),
                            "improvement": perf.get("improvement", "?"),
                            "direct_vs_conjecture_ratio": perf.get("direct_vs_conjecture_ratio", "?")
                        }

                    # Extract problem count
                    if "test_problems" in data:
                        claim_data["problems_evaluated"] = len(data["test_problems"])

                    # Extract claims created
                    if "claims_created" in data:
                        claim_data["claims_created"] = len(data["claims_created"])

                    claim_results[claim_file.replace('.json', '')] = claim_data

                except Exception as e:
                    claim_results[claim_file.replace('.json', '')] = {"error": str(e)}

        return claim_results

    def collect_real_api_results(self) -> Dict[str, Any]:
        """Collect real API test results"""
        api_results = {}

        real_api_file = self.benchmark_dir / "real_api_claim_system.py"
        if real_api_file.exists():
            try:
                with open(real_api_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                api_data = {
                    "file": "real_api_claim_system.py",
                    "file_size_bytes": len(content),
                    "timestamp": datetime.fromtimestamp(real_api_file.stat().st_mtime).isoformat()
                }

                # Check for real API implementation
                real_indicators = [
                    "requests.post",
                    "https://api.z.ai",
                    "https://openrouter.ai",
                    "timeout=",
                    "headers=",
                    "Authorization:"
                ]

                simulation_indicators = [
                    "MOCK_RESPONSE",
                    "hardcoded_response",
                    "placeholder_response",
                    "# This is a simulation"
                ]

                real_count = sum(content.count(indicator) for indicator in real_indicators)
                simulation_count = sum(content.count(indicator) for indicator in simulation_indicators)

                api_data["implementation"] = {
                    "real_api_indicators": real_count,
                    "simulation_indicators": simulation_count,
                    "uses_real_apis": real_count > simulation_count,
                    "api_implementation_quality": "PRODUCTION" if real_count > 5 else "DEVELOPMENT"
                }

                # Check for specific API providers
                if "glm-4.5-air" in content.lower():
                    api_data["glm_45_air_integration"] = True
                if "gpt-oss-20b" in content.lower():
                    api_data["gpt_oss_20b_integration"] = True

                api_results["real_api_system"] = api_data

            except Exception as e:
                api_results["real_api_system"] = {"error": str(e)}

        return api_results

    def get_benchmark_summary(self) -> Dict[str, Any]:
        """Generate benchmark summary statistics"""
        benchmarks = self.collect_all_benchmarks()

        summary = {
            "total_cycle_files": 0,
            "successful_cycles": 0,
            "latest_overall_score": "?",
            "benchmarks_covered": set(),
            "api_implementation_status": "UNKNOWN"
        }

        # Analyze cycle results
        if "cycle_results" in benchmarks["results"]:
            cycle_data = benchmarks["results"]["cycle_results"]
            summary["total_cycle_files"] = len([c for c in cycle_data.values() if "error" not in c])
            summary["successful_cycles"] = len([c for c in cycle_data.values()
                                             if c.get("success", False)])

            # Find latest score
            scores = [c.get("overall_score", "?") for c in cycle_data.values()
                     if isinstance(c.get("overall_score"), (int, float))]
            if scores:
                summary["latest_overall_score"] = max(scores)

            # Collect benchmark coverage
            for cycle in cycle_data.values():
                if "benchmarks_run" in cycle:
                    summary["benchmarks_covered"].update(cycle["benchmarks_run"])

        # Check API implementation status
        if "real_api_tests" in benchmarks["results"]:
            api_data = benchmarks["results"]["real_api_tests"]
            if "real_api_system" in api_data:
                api_impl = api_data["real_api_system"].get("implementation", {})
                if api_impl.get("uses_real_apis", False):
                    summary["api_implementation_status"] = "REAL_APIS"
                else:
                    summary["api_implementation_status"] = "SIMULATED"

        summary["benchmarks_covered"] = list(summary["benchmarks_covered"])

        return summary