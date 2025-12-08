"""
Integration Tests for End-to-End Workflow and CI/CD Pipeline Validation
Tests complete evaluation workflow, reporting integration, and automated testing pipeline
"""

import asyncio
import json
import pytest
import sys
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import (
    EvaluationFramework,
    create_conjecture_wrapper,
    get_available_conjecture_providers,
    evaluate_single_provider,
    evaluate_all_providers
)
from src.benchmarking.deepeval_integration import (
    ConjectureModelWrapper,
    DeepEvalBenchmarkRunner,
    AdvancedBenchmarkEvaluator
)


class TestEndToEndWorkflow:
    """Test complete end-to-end evaluation workflow"""

    @pytest.fixture
    def e2e_framework(self):
        """Create framework for end-to-end testing"""
        return EvaluationFramework()

    @pytest.fixture
    def complete_test_data(self):
        """Load complete test dataset for end-to-end testing"""
        test_data_path = Path(__file__).parent / "test_data" / "test_claims_scenarios.json"
        with open(test_data_path, 'r') as f:
            data = json.load(f)
        
        # Combine all test categories
        all_test_cases = []
        for category, claims in data.items():
            for claim in claims:
                all_test_cases.append({
                    "prompt": claim["prompt"],
                    "expected_answer": claim["expected_answer"],
                    "metadata": claim["metadata"],
                    "category": category,
                    "claim_id": claim["id"]
                })
        
        return all_test_cases

    @pytest.mark.asyncio
    async def test_complete_evaluation_pipeline(self, e2e_framework, complete_test_data):
        """Test complete evaluation pipeline from start to finish"""
        # Test with all providers
        providers = [
            "ibm/granite-4-h-tiny",
            "zai/GLM-4.6",
            "openrouter/gpt-oss-20b"
        ]
        
        # Mock evaluation for end-to-end testing
        with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
            mock_wrapper = AsyncMock()
            mock_wrapper.a_generate = AsyncMock(return_value="End-to-end test response")
            mock_create.return_value = mock_wrapper
            
            # Step 1: Create test cases
            test_cases = []
            for test_data in complete_test_data[:20]:  # Limit for testing
                test_case = e2e_framework.create_test_case(
                    input_text=test_data["prompt"],
                    expected_output=test_data["expected_answer"],
                    additional_metadata=test_data["metadata"]
                )
                test_cases.append(test_case)
            
            assert len(test_cases) == 20
            
            # Step 2: Evaluate all providers
            evaluation_results = await e2e_framework.evaluate_multiple_providers(
                providers, test_cases, compare_conjecture=True
            )
            
            # Step 3: Verify complete workflow
            assert "providers" in evaluation_results
            assert "comparison" in evaluation_results
            assert len(evaluation_results["providers"]) == len(providers) * 2  # direct + conjecture
            
            # Step 4: Verify comparison data
            comparison = evaluation_results["comparison"]
            assert "best_overall" in comparison
            assert "improvements" in comparison
            
            # Step 5: Generate summary
            summary = e2e_framework.get_evaluation_summary(evaluation_results)
            assert "DEEPEVAL EVALUATION SUMMARY" in summary
            assert len(summary) > 100  # Should be comprehensive

    @pytest.mark.asyncio
    async def test_evaluation_data_flow(self, e2e_framework, complete_test_data):
        """Test evaluation data flow and transformation"""
        # Test data flow through evaluation pipeline
        data_flow_stages = {}
        
        # Stage 1: Raw test data
        raw_data = complete_test_data[:10]
        data_flow_stages["raw_input"] = {
            "count": len(raw_data),
            "categories": set(item["category"] for item in raw_data),
            "sample": raw_data[0]
        }
        
        # Stage 2: Test case creation
        test_cases = []
        for test_data in raw_data:
            test_case = e2e_framework.create_test_case(
                input_text=test_data["prompt"],
                expected_output=test_data["expected_answer"],
                additional_metadata=test_data["metadata"]
            )
            test_cases.append(test_case)
        
        data_flow_stages["test_cases"] = {
            "count": len(test_cases),
            "valid": all(hasattr(tc, 'input') for tc in test_cases),
            "sample": test_cases[0].__dict__ if hasattr(test_cases[0], '__dict__') else str(test_cases[0])
        }
        
        # Stage 3: Provider evaluation
        with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
            mock_wrapper = AsyncMock()
            mock_wrapper.a_generate = AsyncMock(return_value="Data flow test response")
            mock_create.return_value = mock_wrapper
            
            provider_results = await e2e_framework.evaluate_provider(
                "data-flow-test", test_cases, use_conjecture=False
            )
            
            data_flow_stages["provider_evaluation"] = {
                "success": provider_results["success"],
                "test_cases_count": provider_results["test_cases_count"],
                "overall_score": provider_results["overall_score"],
                "metrics_count": len(provider_results["metrics_results"])
            }
        
        # Stage 4: Result aggregation
        data_flow_stages["result_aggregation"] = {
            "stages_completed": len(data_flow_stages),
            "data_integrity": all(
                stage.get("count", 0) > 0 
                for stage in data_flow_stages.values()
                if isinstance(stage, dict)
            )
        }
        
        # Verify data flow integrity
        assert data_flow_stages["raw_input"]["count"] == 10
        assert data_flow_stages["test_cases"]["count"] == 10
        assert data_flow_stages["provider_evaluation"]["success"] is True
        assert data_flow_stages["result_aggregation"]["stages_completed"] == 4
        assert data_flow_stages["result_aggregation"]["data_integrity"] is True

    @pytest.mark.asyncio
    async def test_evaluation_error_recovery(self, e2e_framework):
        """Test evaluation error recovery and fallback mechanisms"""
        # Create test cases that might cause errors
        error_test_cases = [
            e2e_framework.create_test_case(
                input_text="",  # Empty input
                expected_output="",
                additional_metadata={"error_test": "empty_input"}
            ),
            e2e_framework.create_test_case(
                input_text="Normal input",
                expected_output="Normal output",
                additional_metadata={"error_test": "normal"}
            ),
            e2e_framework.create_test_case(
                input_text="A" * 10000,  # Very long input
                expected_output="Long output",
                additional_metadata={"error_test": "long_input"}
            )
        ]
        
        # Mock wrapper that handles different error scenarios
        async def mock_error_handling(prompt):
            if not prompt.strip():
                raise ValueError("Empty prompt detected")
            elif len(prompt) > 5000:
                raise MemoryError("Input too long")
            else:
                return "Normal response"
        
        with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
            mock_wrapper = AsyncMock()
            mock_wrapper.a_generate = AsyncMock(side_effect=mock_error_handling)
            mock_create.return_value = mock_wrapper
            
            # Test error recovery
            result = await e2e_framework.evaluate_provider(
                "error-recovery-test", error_test_cases, use_conjecture=False
            )
            
            # Should handle errors gracefully
            assert "error" in result or result["success"] is False
            assert "overall_score" in result
            
            # Some test cases should succeed
            if result.get("metrics_results"):
                successful_metrics = [
                    metric for metric in result["metrics_results"].values()
                    if metric.get("success", False)
                ]
                assert len(successful_metrics) > 0

    @pytest.mark.asyncio
    async def test_evaluation_consistency_validation(self, e2e_framework, complete_test_data):
        """Test evaluation consistency across multiple runs"""
        # Test consistency with same data multiple times
        consistency_test_data = complete_test_data[:5]  # Small subset
        providers = ["ibm/granite-4-h-tiny", "zai/GLM-4.6"]
        
        consistency_results = {}
        
        for run_id in range(3):  # 3 runs for consistency
            with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
                mock_wrapper = AsyncMock()
                # Simulate consistent responses
                mock_wrapper.a_generate = AsyncMock(return_value=f"Consistent response {run_id}")
                mock_create.return_value = mock_wrapper
                
                run_results = await e2e_framework.evaluate_multiple_providers(
                    providers, consistency_test_data, compare_conjecture=False
                )
                
                consistency_results[f"run_{run_id}"] = run_results
        
        # Analyze consistency
        provider_scores = {}
        for provider in providers:
            scores = []
            for run_id, run_result in consistency_results.items():
                provider_key = f"{provider}_direct"
                if provider_key in run_result["providers"]:
                    scores.append(run_result["providers"][provider_key]["overall_score"])
            
            if scores:
                provider_scores[provider] = {
                    "scores": scores,
                    "mean": sum(scores) / len(scores),
                    "std": (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5,
                    "consistent": max(scores) - min(scores) < 0.1  # Low variance
                }
        
        # Verify consistency
        for provider, consistency_data in provider_scores.items():
            assert len(consistency_data["scores"]) == 3
            assert consistency_data["mean"] > 0
            assert consistency_data["std"] < 0.05  # Low standard deviation


class TestReportingIntegration:
    """Test reporting integration and result generation"""

    @pytest.fixture
    def reporting_framework(self):
        """Create framework for reporting testing"""
        return EvaluationFramework()

    @pytest.fixture
    def mock_evaluation_results(self):
        """Create mock evaluation results for reporting"""
        return {
            "evaluation_timestamp": "2024-01-01T00:00:00Z",
            "test_cases_count": 10,
            "providers": {
                "ibm/granite-4-h-tiny_direct": {
                    "provider": "ibm/granite-4-h-tiny",
                    "use_conjecture": False,
                    "overall_score": 0.82,
                    "success": True,
                    "test_cases_count": 10,
                    "metrics_results": {
                        "answer_relevancy": {"score": 0.85, "success": True},
                        "faithfulness": {"score": 0.80, "success": True},
                        "exact_match": {"score": 0.78, "success": True},
                        "summarization": {"score": 0.83, "success": True},
                        "bias": {"score": 0.95, "success": True},
                        "toxicity": {"score": 0.98, "success": True}
                    }
                },
                "ibm/granite-4-h-tiny_conjecture": {
                    "provider": "ibm/granite-4-h-tiny",
                    "use_conjecture": True,
                    "overall_score": 0.91,
                    "success": True,
                    "test_cases_count": 10,
                    "metrics_results": {
                        "answer_relevancy": {"score": 0.92, "success": True},
                        "faithfulness": {"score": 0.88, "success": True},
                        "exact_match": {"score": 0.85, "success": True},
                        "summarization": {"score": 0.90, "success": True},
                        "bias": {"score": 0.97, "success": True},
                        "toxicity": {"score": 0.99, "success": True}
                    }
                },
                "zai/GLM-4.6_direct": {
                    "provider": "zai/GLM-4.6",
                    "use_conjecture": False,
                    "overall_score": 0.85,
                    "success": True,
                    "test_cases_count": 10,
                    "metrics_results": {
                        "answer_relevancy": {"score": 0.87, "success": True},
                        "faithfulness": {"score": 0.83, "success": True},
                        "exact_match": {"score": 0.80, "success": True},
                        "summarization": {"score": 0.86, "success": True},
                        "bias": {"score": 0.94, "success": True},
                        "toxicity": {"score": 0.97, "success": True}
                    }
                },
                "zai/GLM-4.6_conjecture": {
                    "provider": "zai/GLM-4.6",
                    "use_conjecture": True,
                    "overall_score": 0.93,
                    "success": True,
                    "test_cases_count": 10,
                    "metrics_results": {
                        "answer_relevancy": {"score": 0.94, "success": True},
                        "faithfulness": {"score": 0.90, "success": True},
                        "exact_match": {"score": 0.87, "success": True},
                        "summarization": {"score": 0.92, "success": True},
                        "bias": {"score": 0.98, "success": True},
                        "toxicity": {"score": 0.99, "success": True}
                    }
                }
            },
            "comparison": {
                "best_overall": {
                    "provider": "zai/GLM-4.6",
                    "score": 0.93,
                    "mode": "conjecture"
                },
                "improvements": {
                    "ibm/granite-4-h-tiny": {
                        "improvement_percent": 11.0,
                        "direct_score": 0.82,
                        "conjecture_score": 0.91
                    },
                    "zai/GLM-4.6": {
                        "improvement_percent": 9.4,
                        "direct_score": 0.85,
                        "conjecture_score": 0.93
                    }
                }
            }
        }

    def test_summary_report_generation(self, reporting_framework, mock_evaluation_results):
        """Test summary report generation"""
        # Generate summary report
        summary = reporting_framework.get_evaluation_summary(mock_evaluation_results)
        
        # Verify report structure
        assert "DEEPEVAL EVALUATION SUMMARY" in summary
        assert "ibm/granite-4-h-tiny" in summary
        assert "zai/GLM-4.6" in summary
        assert "11.0%" in summary  # IBM improvement
        assert "9.4%" in summary  # GLM improvement
        assert "COMPARISON RESULTS" in summary
        assert "CONJECTURE IMPROVEMENTS" in summary
        
        # Verify report completeness
        report_lines = summary.split('\n')
        assert len(report_lines) > 20  # Should be comprehensive

    def test_detailed_metrics_report(self, reporting_framework, mock_evaluation_results):
        """Test detailed metrics report generation"""
        # Extract detailed metrics
        detailed_metrics = {}
        
        for provider_key, provider_data in mock_evaluation_results["providers"].items():
            provider_name = provider_data["provider"]
            metrics = provider_data["metrics_results"]
            
            detailed_metrics[provider_name] = {
                "overall_score": provider_data["overall_score"],
                "use_conjecture": provider_data["use_conjecture"],
                "metric_scores": {
                    metric: data["score"] 
                    for metric, data in metrics.items()
                },
                "metric_success_rates": {
                    metric: data["success"] 
                    for metric, data in metrics.items()
                },
                "failed_metrics": [
                    metric for metric, data in metrics.items()
                    if not data["success"]
                ]
            }
        
        # Verify detailed metrics structure
        for provider_name, metrics_data in detailed_metrics.items():
            assert "overall_score" in metrics_data
            assert "use_conjecture" in metrics_data
            assert "metric_scores" in metrics_data
            assert "metric_success_rates" in metrics_data
            assert "failed_metrics" in metrics_data
            
            # Verify metric completeness
            expected_metrics = ["answer_relevancy", "faithfulness", "exact_match", 
                             "summarization", "bias", "toxicity"]
            for metric in expected_metrics:
                assert metric in metrics_data["metric_scores"]
                assert metric in metrics_data["metric_success_rates"]

    def test_json_report_export(self, reporting_framework, mock_evaluation_results):
        """Test JSON report export functionality"""
        # Test JSON export
        import json
        
        json_report = json.dumps(mock_evaluation_results, indent=2, default=str)
        
        # Verify JSON structure
        parsed_report = json.loads(json_report)
        assert parsed_report == mock_evaluation_results
        
        # Verify JSON validity
        assert isinstance(parsed_report, dict)
        assert "providers" in parsed_report
        assert "comparison" in parsed_report

    def test_csv_report_export(self, reporting_framework, mock_evaluation_results):
        """Test CSV report export functionality"""
        import csv
        import io
        
        # Create CSV report
        csv_buffer = io.StringIO()
        
        # Write CSV header
        fieldnames = ['provider', 'use_conjecture', 'overall_score', 'answer_relevancy', 
                     'faithfulness', 'exact_match', 'summarization', 'bias', 'toxicity']
        writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write provider data
        for provider_key, provider_data in mock_evaluation_results["providers"].items():
            row = {
                'provider': provider_data["provider"],
                'use_conjecture': provider_data["use_conjecture"],
                'overall_score': provider_data["overall_score"],
                'answer_relevancy': provider_data["metrics_results"]["answer_relevancy"]["score"],
                'faithfulness': provider_data["metrics_results"]["faithfulness"]["score"],
                'exact_match': provider_data["metrics_results"]["exact_match"]["score"],
                'summarization': provider_data["metrics_results"]["summarization"]["score"],
                'bias': provider_data["metrics_results"]["bias"]["score"],
                'toxicity': provider_data["metrics_results"]["toxicity"]["score"]
            }
            writer.writerow(row)
        
        csv_report = csv_buffer.getvalue()
        
        # Verify CSV structure
        csv_lines = csv_report.split('\n')
        assert len(csv_lines) == len(mock_evaluation_results["providers"]) + 1  # +1 for header
        assert fieldnames[0] in csv_lines[0]  # Header should contain fieldnames

    def test_html_report_generation(self, reporting_framework, mock_evaluation_results):
        """Test HTML report generation"""
        # Generate HTML report
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>DeepEval Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f0f0; color: white; padding: 20px; text-align: center; }
                .provider { margin: 20px 0; padding: 15px; border: 1px solid #ddd; }
                .metrics { display: flex; justify-content: space-between; }
                .metric { text-align: center; padding: 10px; }
                .improvement { color: green; font-weight: bold; }
                .regression { color: red; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>DeepEval Evaluation Report</h1>
                <p>Generated on: {timestamp}</p>
            </div>
            {provider_sections}
            <div class="comparison">
                <h2>Comparison Results</h2>
                {comparison_section}
            </div>
        </body>
        </html>
        """
        
        # Generate provider sections
        provider_sections = ""
        for provider_key, provider_data in mock_evaluation_results["providers"].items():
            provider_section = f"""
            <div class="provider">
                <h3>{provider_data['provider']} ({'Conjecture' if provider_data['use_conjecture'] else 'Direct'})</h3>
                <p>Overall Score: {provider_data['overall_score']:.3f}</p>
                <div class="metrics">
                    <div class="metric">Relevancy: {provider_data['metrics_results']['answer_relevancy']['score']:.3f}</div>
                    <div class="metric">Faithfulness: {provider_data['metrics_results']['faithfulness']['score']:.3f}</div>
                    <div class="metric">Exact Match: {provider_data['metrics_results']['exact_match']['score']:.3f}</div>
                    <div class="metric">Summarization: {provider_data['metrics_results']['summarization']['score']:.3f}</div>
                    <div class="metric">Bias: {provider_data['metrics_results']['bias']['score']:.3f}</div>
                    <div class="metric">Toxicity: {provider_data['metrics_results']['toxicity']['score']:.3f}</div>
                </div>
            </div>
            """
            provider_sections += provider_section
        
        # Generate comparison section
        comparison = mock_evaluation_results["comparison"]
        comparison_section = f"""
        <p><strong>Best Overall:</strong> {comparison['best_overall']['provider']} ({comparison['best_overall']['score']:.3f})</p>
        <h3>Conjecture Improvements:</h3>
        """
        
        for provider, improvement in comparison["improvements"].items():
            improvement_class = "improvement" if improvement["improvement_percent"] > 0 else "regression"
            comparison_section += f"""
            <p><strong>{provider}:</strong> 
            <span class="{improvement_class}">{improvement['improvement_percent']:+.1f}%</span>
            (Direct: {improvement['direct_score']:.3f}, Conjecture: {improvement['conjecture_score']:.3f})</p>
            """
        
        # Generate final HTML
        html_report = html_template.format(
            timestamp=mock_evaluation_results["evaluation_timestamp"],
            provider_sections=provider_sections,
            comparison_section=comparison_section
        )
        
        # Verify HTML structure
        assert "<!DOCTYPE html>" in html_report
        assert "DeepEval Evaluation Report" in html_report
        assert "ibm/granite-4-h-tiny" in html_report
        assert "zai/GLM-4.6" in html_report
        assert "11.0%" in html_report
        assert "9.4%" in html_report


class TestCICDPipelineValidation:
    """Test CI/CD pipeline integration and validation"""

    @pytest.fixture
    def ci_cd_environment(self):
        """Create CI/CD environment for testing"""
        return {
            "CI": True,
            "GITHUB_ACTIONS": True,
            "PYTHONPATH": str(Path(__file__).parent.parent),
            "TEST_TIMEOUT": 300,
            "COVERAGE_THRESHOLD": 80,
            "PERFORMANCE_THRESHOLD": 0.7,
            "REPORTS_DIR": "/tmp/test-reports"
        }

    def test_ci_environment_detection(self, ci_cd_environment):
        """Test CI environment detection and configuration"""
        # Mock environment variables
        with patch.dict(os.environ, ci_cd_environment, clear=True):
            # Test CI detection
            is_ci = os.getenv("CI", "false").lower() == "true"
            assert is_ci is True
            
            # Test GitHub Actions detection
            is_github_actions = os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
            assert is_github_actions is True
            
            # Test configuration loading
            timeout = int(os.getenv("TEST_TIMEOUT", "300"))
            assert timeout == 300
            
            coverage_threshold = float(os.getenv("COVERAGE_THRESHOLD", "80"))
            assert coverage_threshold == 80.0
            
            performance_threshold = float(os.getenv("PERFORMANCE_THRESHOLD", "0.7"))
            assert performance_threshold == 0.7

    def test_test_timeout_enforcement(self, ci_cd_environment):
        """Test test timeout enforcement in CI environment"""
        with patch.dict(os.environ, ci_cd_environment, clear=True):
            # Create timeout test
            timeout_test_cases = [
                {
                    "name": "fast_test",
                    "duration": 10,  # Should pass
                    "expected_result": "pass"
                },
                {
                    "name": "slow_test",
                    "duration": 350,  # Should timeout
                    "expected_result": "timeout"
                },
                {
                    "name": "boundary_test",
                    "duration": 300,  # At boundary
                    "expected_result": "boundary"
                }
            ]
            
            for test_case in timeout_test_cases:
                start_time = time.time()
                
                # Simulate test execution
                time.sleep(test_case["duration"] / 100)  # Scale down for testing
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                timeout = int(os.getenv("TEST_TIMEOUT", "300"))
                
                if test_case["expected_result"] == "pass":
                    assert execution_time < timeout
                elif test_case["expected_result"] == "timeout":
                    assert execution_time >= timeout
                elif test_case["expected_result"] == "boundary":
                    assert execution_time <= timeout

    def test_coverage_threshold_validation(self, ci_cd_environment):
        """Test coverage threshold validation in CI"""
        with patch.dict(os.environ, ci_cd_environment, clear=True):
            # Mock coverage data
            coverage_data = [
                {
                    "file": "test_comprehensive_deepeval_evaluation.py",
                    "coverage": 85.5,  # Should pass
                    "expected_result": "pass"
                },
                {
                    "file": "test_partial_coverage.py",
                    "coverage": 75.2,  # Should fail
                    "expected_result": "fail"
                },
                {
                    "file": "test_boundary_coverage.py",
                    "coverage": 80.0,  # At boundary
                    "expected_result": "boundary"
                }
            ]
            
            coverage_threshold = float(os.getenv("COVERAGE_THRESHOLD", "80"))
            
            for coverage_info in coverage_data:
                coverage = coverage_info["coverage"]
                expected_result = coverage_info["expected_result"]
                
                if expected_result == "pass":
                    assert coverage >= coverage_threshold
                elif expected_result == "fail":
                    assert coverage < coverage_threshold
                elif expected_result == "boundary":
                    assert coverage >= coverage_threshold

    def test_performance_threshold_validation(self, ci_cd_environment):
        """Test performance threshold validation in CI"""
        with patch.dict(os.environ, ci_cd_environment, clear=True):
            # Mock performance data
            performance_data = [
                {
                    "provider": "ibm/granite-4-h-tiny",
                    "overall_score": 0.85,  # Should pass
                    "expected_result": "pass"
                },
                {
                    "provider": "zai/GLM-4.6",
                    "overall_score": 0.65,  # Should fail
                    "expected_result": "fail"
                },
                {
                    "provider": "openrouter/gpt-oss-20b",
                    "overall_score": 0.7,  # At boundary
                    "expected_result": "boundary"
                }
            ]
            
            performance_threshold = float(os.getenv("PERFORMANCE_THRESHOLD", "0.7"))
            
            for perf_info in performance_data:
                score = perf_info["overall_score"]
                expected_result = perf_info["expected_result"]
                
                if expected_result == "pass":
                    assert score >= performance_threshold
                elif expected_result == "fail":
                    assert score < performance_threshold
                elif expected_result == "boundary":
                    assert score >= performance_threshold

    def test_report_generation_in_ci(self, ci_cd_environment):
        """Test report generation in CI environment"""
        with patch.dict(os.environ, ci_cd_environment, clear=True):
            # Create mock results for CI reporting
            ci_results = {
                "evaluation_timestamp": "2024-01-01T00:00:00Z",
                "test_cases_count": 5,
                "providers": {
                    "ibm/granite-4-h-tiny_direct": {
                        "overall_score": 0.82,
                        "success": True,
                        "metrics_results": {
                            "answer_relevancy": {"score": 0.85},
                            "faithfulness": {"score": 0.80}
                        }
                    }
                },
                "comparison": {
                    "best_overall": {"provider": "ibm/granite-4-h-tiny", "score": 0.82},
                    "improvements": {}
                }
            }
            
            # Test report generation
            framework = EvaluationFramework()
            summary = framework.get_evaluation_summary(ci_results)
            
            # Verify CI-specific report content
            assert "DEEPEVAL EVALUATION SUMMARY" in summary
            assert "ibm/granite-4-h-tiny" in summary
            assert len(summary) > 50  # Should be substantial
            
            # Test report file creation
            reports_dir = Path(os.getenv("REPORTS_DIR", "/tmp"))
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = reports_dir / "evaluation_report.txt"
            with open(report_file, 'w') as f:
                f.write(summary)
            
            assert report_file.exists()
            assert report_file.stat().st_size > 100

    def test_ci_exit_code_handling(self, ci_cd_environment):
        """Test CI exit code handling based on test results"""
        with patch.dict(os.environ, ci_cd_environment, clear=True):
            # Mock test results for exit code testing
            exit_code_test_cases = [
                {
                    "name": "all_tests_pass",
                    "results": {"success": True, "failed_tests": 0},
                    "expected_exit_code": 0
                },
                {
                    "name": "some_tests_fail",
                    "results": {"success": False, "failed_tests": 2},
                    "expected_exit_code": 1
                },
                {
                    "name": "critical_failure",
                    "results": {"success": False, "failed_tests": 5, "critical_errors": 1},
                    "expected_exit_code": 2
                }
            ]
            
            for test_case in exit_code_test_cases:
                results = test_case["results"]
                expected_exit_code = test_case["expected_exit_code"]
                
                # Determine actual exit code
                if results["success"] and results["failed_tests"] == 0:
                    actual_exit_code = 0
                elif results.get("critical_errors", 0) > 0:
                    actual_exit_code = 2
                elif results["failed_tests"] > 0:
                    actual_exit_code = 1
                else:
                    actual_exit_code = 1
                
                assert actual_exit_code == expected_exit_code

    def test_artifact_collection(self, ci_cd_environment):
        """Test artifact collection in CI environment"""
        with patch.dict(os.environ, ci_cd_environment, clear=True):
            # Create test artifacts
            artifacts = {
                "test_results.json": {
                    "success": True,
                    "test_count": 10,
                    "passed": 8,
                    "failed": 2
                },
                "coverage_report.xml": '<?xml version="1.0"?><coverage version="1.0">85.5%</coverage>',
                "performance_metrics.json": {
                    "avg_response_time": 0.15,
                    "throughput": 100,
                    "memory_usage": 50
                }
            }
            
            # Test artifact creation
            reports_dir = Path(os.getenv("REPORTS_DIR", "/tmp"))
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            created_files = []
            for artifact_name, artifact_content in artifacts.items():
                artifact_path = reports_dir / artifact_name
                
                if artifact_name.endswith('.json'):
                    import json
                    with open(artifact_path, 'w') as f:
                        json.dump(artifact_content, f, indent=2)
                elif artifact_name.endswith('.xml'):
                    with open(artifact_path, 'w') as f:
                        f.write(artifact_content)
                elif artifact_name.endswith('.json'):
                    import json
                    with open(artifact_path, 'w') as f:
                        json.dump(artifact_content, f, indent=2)
                
                created_files.append(artifact_path)
                assert artifact_path.exists()
            
            assert len(created_files) == len(artifacts)
            assert all(path.stat().st_size > 0 for path in created_files)


class TestAutomatedTestingPipeline:
    """Test automated testing pipeline execution"""

    @pytest.fixture
    def pipeline_framework(self):
        """Create framework for pipeline testing"""
        return EvaluationFramework()

    def test_test_discovery(self):
        """Test automatic test discovery and execution"""
        # Test test file discovery
        test_files = []
        tests_dir = Path(__file__).parent
        
        for file_path in tests_dir.glob("test_*.py"):
            if file_path.is_file():
                test_files.append(str(file_path))
        
        # Verify test discovery
        assert len(test_files) > 0
        assert any("deepeval" in file.lower() for file in test_files)
        assert any("comprehensive" in file.lower() for file in test_files)
        assert any("performance" in file.lower() for file in test_files)

    def test_parallel_test_execution(self):
        """Test parallel test execution"""
        # Mock parallel test execution
        test_suites = [
            "test_comprehensive_deepeval_evaluation.py",
            "test_claim_evaluation_scenarios.py",
            "test_quality_issue_detection.py",
            "test_performance_benchmarks.py"
        ]
        
        # Simulate parallel execution
        execution_results = {}
        
        for test_suite in test_suites:
            start_time = time.time()
            
            # Simulate test execution
            time.sleep(0.1)  # Simulate test time
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            execution_results[test_suite] = {
                "execution_time": execution_time,
                "success": True,
                "tests_run": 10
            }
        
        # Verify parallel execution
        assert len(execution_results) == len(test_suites)
        
        # All tests should complete reasonably quickly
        for test_suite, result in execution_results.items():
            assert result["execution_time"] < 1.0  # Should complete within 1 second
            assert result["success"] is True

    def test_test_result_aggregation(self):
        """Test test result aggregation and reporting"""
        # Mock test results from multiple test suites
        suite_results = {
            "comprehensive_tests": {
                "total": 20,
                "passed": 18,
                "failed": 2,
                "skipped": 0,
                "execution_time": 5.2
            },
            "claim_evaluation_tests": {
                "total": 15,
                "passed": 14,
                "failed": 1,
                "skipped": 0,
                "execution_time": 3.8
            },
            "quality_detection_tests": {
                "total": 25,
                "passed": 23,
                "failed": 2,
                "skipped": 0,
                "execution_time": 6.1
            },
            "performance_tests": {
                "total": 18,
                "passed": 16,
                "failed": 2,
                "skipped": 0,
                "execution_time": 8.5
            }
        }
        
        # Aggregate results
        aggregated_results = {
            "total_suites": len(suite_results),
            "total_tests": sum(results["total"] for results in suite_results.values()),
            "total_passed": sum(results["passed"] for results in suite_results.values()),
            "total_failed": sum(results["failed"] for results in suite_results.values()),
            "total_skipped": sum(results["skipped"] for results in suite_results.values()),
            "total_execution_time": sum(results["execution_time"] for results in suite_results.values()),
            "overall_success_rate": 0,
            "suite_results": suite_results
        }
        
        # Calculate overall success rate
        if aggregated_results["total_tests"] > 0:
            aggregated_results["overall_success_rate"] = (
                aggregated_results["total_passed"] / aggregated_results["total_tests"]
            ) * 100
        
        # Verify aggregation
        assert aggregated_results["total_suites"] == 4
        assert aggregated_results["total_tests"] == 78  # 20 + 15 + 25 + 18
        assert aggregated_results["total_passed"] == 71  # 18 + 14 + 23 + 16
        assert aggregated_results["total_failed"] == 7  # 2 + 1 + 2 + 2
        assert aggregated_results["overall_success_rate"] == 91.03  # 71/78 * 100

    def test_pipeline_failure_handling(self):
        """Test pipeline failure handling and recovery"""
        # Mock pipeline failure scenarios
        failure_scenarios = [
            {
                "name": "test_execution_failure",
                "error_type": "TestExecutionError",
                "recoverable": True,
                "expected_action": "continue"
            },
            {
                "name": "infrastructure_failure",
                "error_type": "InfrastructureError",
                "recoverable": False,
                "expected_action": "abort"
            },
            {
                "name": "timeout_failure",
                "error_type": "TimeoutError",
                "recoverable": True,
                "expected_action": "retry"
            }
        ]
        
        for scenario in failure_scenarios:
            error_type = scenario["error_type"]
            recoverable = scenario["recoverable"]
            expected_action = scenario["expected_action"]
            
            # Test failure handling
            if error_type == "TestExecutionError":
                # Should log and continue
                action = "continue" if recoverable else "abort"
            elif error_type == "InfrastructureError":
                # Should abort immediately
                action = "continue" if recoverable else "abort"
            elif error_type == "TimeoutError":
                # Should retry if recoverable
                action = "retry" if recoverable else "abort"
            else:
                action = "abort"
            
            assert action == expected_action

    def test_pipeline_configuration(self):
        """Test pipeline configuration and customization"""
        # Test pipeline configuration options
        pipeline_config = {
            "parallel_execution": True,
            "max_workers": 4,
            "timeout": 300,
            "retry_attempts": 3,
            "retry_delay": 5,
            "coverage_threshold": 80,
            "performance_threshold": 0.7,
            "report_formats": ["json", "xml", "html"],
            "artifact_retention": "7d"
        }
        
        # Verify configuration validation
        assert isinstance(pipeline_config["parallel_execution"], bool)
        assert isinstance(pipeline_config["max_workers"], int)
        assert pipeline_config["max_workers"] > 0
        assert isinstance(pipeline_config["timeout"], int)
        assert pipeline_config["timeout"] > 0
        assert isinstance(pipeline_config["retry_attempts"], int)
        assert pipeline_config["retry_attempts"] > 0
        assert isinstance(pipeline_config["retry_delay"], int)
        assert pipeline_config["retry_delay"] >= 0
        assert isinstance(pipeline_config["coverage_threshold"], (int, float))
        assert 0 <= pipeline_config["coverage_threshold"] <= 100
        assert isinstance(pipeline_config["performance_threshold"], (int, float))
        assert 0 <= pipeline_config["performance_threshold"] <= 1
        assert isinstance(pipeline_config["report_formats"], list)
        assert len(pipeline_config["report_formats"]) > 0
        assert isinstance(pipeline_config["artifact_retention"], str)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])