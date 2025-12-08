"""
DeepEval Integration for Advanced Benchmark Evaluation
Provides sophisticated evaluation using DeepEval's metrics
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ExactMatchMetric,
    SummarizationMetric,
    BiasMetric,
    ToxicityMetric
)
from deepeval.dataset import EvaluationDataset
from deepeval.models import DeepEvalBaseModel
from deepeval.test_case import LLMTestCase

# Configure logging
logger = logging.getLogger(__name__)

# Import the new evaluation framework
try:
    from ..evaluation.conjecture_llm_wrapper import ConjectureLLMWrapper
    from ..evaluation.evaluation_framework import EvaluationFramework
    USE_NEW_FRAMEWORK = True
except ImportError:
    logger.warning("New evaluation framework not available, using legacy implementation")
    USE_NEW_FRAMEWORK = False


class ConjectureModelWrapper(DeepEvalBaseModel):
    """Wrapper for Conjecture models to work with DeepEval"""

    def __init__(self, model_name: str, use_conjecture: bool = False):
        self.model_name = model_name
        self.use_conjecture = use_conjecture
        
        # Use new framework if available
        if USE_NEW_FRAMEWORK:
            self._wrapper = ConjectureLLMWrapper(model_name, use_conjecture=use_conjecture)
        else:
            # Fallback to legacy implementation
            self._model_integration = None
            logger.warning("Using legacy ConjectureModelWrapper implementation")
        
        super().__init__()

    def _get_model_integration(self):
        """Lazy load model integration (legacy)"""
        if not hasattr(self, '_model_integration') or self._model_integration is None:
            from .model_integration import ModelIntegration
            self._model_integration = ModelIntegration()
        return self._model_integration

    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        """Generate response using Conjecture model"""
        try:
            if USE_NEW_FRAMEWORK and hasattr(self, '_wrapper'):
                # Use new framework
                return await self._wrapper.a_generate(prompt)
            else:
                # Legacy implementation
                integration = self._get_model_integration()
                
                # Map model names to provider identifiers
                model_mapping = {
                    "granite-tiny": "granite-tiny",
                    "gpt-oss-20b": "gpt-oss-20b",
                    "glm-4.6": "glm-4.6",
                    "ibm/granite-4-h-tiny": "granite-tiny",
                    "openrouter/gpt-oss-20b": "gpt-oss-20b",
                    "zai/glm-4.6": "glm-4.6"
                }

                mapped_model = model_mapping.get(self.model_name.lower(), self.model_name.lower())

                if self.use_conjecture:
                    return await integration.get_conjecture_enhanced_response(mapped_model, prompt)
                else:
                    return await integration.get_model_response(mapped_model, prompt)

        except Exception as e:
            logger.error(f"Error generating response with {self.model_name}: {e}")
            # Return fallback response for evaluation continuity
            return f"Evaluation fallback response for {self.model_name}: {prompt[:100]}..."

    def get_model_name(self) -> str:
        return f"Conjecture-{self.model_name}+Conjecture" if self.use_conjecture else f"Conjecture-{self.model_name}"

class DeepEvalBenchmarkRunner:
    """Advanced benchmark runner using DeepEval metrics"""

    def __init__(self, config=None):
        self.config = config
        # Initialize metrics with proper configuration
        self.metrics = {
            "relevancy": AnswerRelevancyMetric(threshold=0.5, model="gpt-3.5-turbo"),
            "faithfulness": FaithfulnessMetric(threshold=0.5, model="gpt-3.5-turbo"),
            "exact_match": ExactMatchMetric(threshold=0.5),
            "summarization": SummarizationMetric(threshold=0.5, model="gpt-3.5-turbo"),
            "bias": BiasMetric(threshold=0.5, model="gpt-3.5-turbo"),
            "toxicity": ToxicityMetric(threshold=0.5, model="gpt-3.5-turbo")
        }
        logger.info("DeepEval benchmark runner initialized with 6 metrics")

    async def evaluate_with_deepeval(self,
                                      tasks: List[Dict],
                                      model_name: str,
                                      use_conjecture: bool = False) -> Dict[str, Any]:
        """Evaluate tasks using DeepEval metrics"""

        try:
            logger.info(f"Starting DeepEval evaluation for {model_name} (conjecture={use_conjecture})")
            
            # Create Conjecture model wrapper
            model = ConjectureModelWrapper(model_name, use_conjecture)

            # Create test cases from tasks
            test_cases = []
            for task in tasks:
                test_case = LLMTestCase(
                    input=task.get('prompt', ''),
                    expected_output=task.get('expected_answer', ''),
                    actual_output=task.get('model_response', ''),
                    additional_metadata=task.get('metadata', {})
                )
                test_cases.append(test_case)

            if not test_cases:
                logger.warning("No test cases provided for evaluation")
                return {}

            # Evaluate with all metrics
            results = {}

            for metric_name, metric in self.metrics.items():
                try:
                    logger.info(f"Evaluating with {metric_name} metric...")
                    
                    # Create dataset for this metric
                    dataset = EvaluationDataset(test_cases=test_cases)

                    # Run evaluation
                    eval_result = evaluate(
                        dataset=dataset,
                        metrics=[metric],
                        model=model,
                        run_async=True
                    )

                    # Process results
                    if eval_result and len(eval_result) > 0:
                        results[metric_name] = {
                            "score": eval_result[0].score,
                            "reason": eval_result[0].reason,
                            "success": eval_result[0].score >= metric.threshold
                        }
                    else:
                        results[metric_name] = {
                            "score": 0.0,
                            "reason": "No evaluation result available",
                            "success": False
                        }

                except Exception as e:
                    logger.error(f"Error evaluating {metric_name}: {e}")
                    results[metric_name] = {
                        "score": 0.0,
                        "reason": f"Evaluation error: {str(e)}",
                        "success": False
                    }

            logger.info(f"DeepEval evaluation completed for {model_name}")
            return results

        except Exception as e:
            logger.error(f"Critical error in DeepEval evaluation: {e}")
            return {"error": f"Evaluation failed: {str(e)}"}

class AdvancedBenchmarkEvaluator:
    """Advanced evaluator combining DeepEval with custom metrics"""

    def __init__(self, config=None):
        # Try to use new evaluation framework
        try:
            from ..evaluation.evaluation_framework import EvaluationFramework
            self.use_new_framework = True
            self.evaluation_framework = EvaluationFramework(config)
            logger.info("Advanced benchmark evaluator initialized with new framework")
        except ImportError:
            self.use_new_framework = False
            self.deepeval_runner = DeepEvalBenchmarkRunner(config)
            logger.info("Advanced benchmark evaluator initialized with legacy framework")

    async def evaluate_benchmark_response(self,
                                          task: Dict,
                                          model_response: str,
                                          model_name: str,
                                          use_conjecture: bool = False) -> Dict[str, Any]:
        """Evaluate a single benchmark response with advanced metrics"""

        try:
            logger.info(f"Evaluating benchmark response for {model_name}")
            
            # Use new framework if available
            if self.use_new_framework and hasattr(self, 'evaluation_framework'):
                # Create test case for new framework
                test_case = self.evaluation_framework.create_test_case(
                    input_text=task.get('prompt', ''),
                    expected_output=task.get('expected_answer', ''),
                    actual_output=model_response,
                    additional_metadata=task.get('metadata', {})
                )
                
                # Evaluate using new framework
                result = await self.evaluation_framework.evaluate_provider(
                    model_name, [test_case], use_conjecture
                )
                
                # Add custom evaluation
                custom_score = self._custom_evaluation(task, model_response)
                if result.get("success", False):
                    # Combine with custom score
                    metric_scores = [r.get("score", 0.0) for r in result.get("metrics_results", {}).values()]
                    metric_scores.append(custom_score)
                    overall_score = sum(metric_scores) / len(metric_scores) if metric_scores else custom_score
                    
                    return {
                        "overall_score": overall_score,
                        "deepeval_metrics": result.get("metrics_results", {}),
                        "custom_score": custom_score,
                        "individual_scores": metric_scores,
                        "success": True
                    }
                else:
                    # Fallback to custom evaluation
                    return {
                        "overall_score": custom_score,
                        "deepeval_metrics": result.get("metrics_results", {}),
                        "custom_score": custom_score,
                        "individual_scores": [custom_score],
                        "success": False,
                        "error": result.get("error", "Framework evaluation failed")
                    }
            else:
                # Legacy implementation
                # Prepare task for DeepEval
                task_with_response = {
                    **task,
                    "model_response": model_response
                }

                # Get DeepEval results
                deepeval_results = await self.deepeval_runner.evaluate_with_deepeval(
                    [task_with_response], model_name, use_conjecture
                )

                # Combine with custom evaluation
                custom_score = self._custom_evaluation(task, model_response)

                # Calculate overall score
                if deepeval_results and "error" not in deepeval_results:
                    scores = [result.get("score", 0.0) for result in deepeval_results.values()]
                    scores.append(custom_score)
                    
                    overall_score = sum(scores) / len(scores) if scores else 0.0
                    
                    return {
                        "overall_score": overall_score,
                        "deepeval_metrics": deepeval_results,
                        "custom_score": custom_score,
                        "individual_scores": scores,
                        "success": True
                    }
                else:
                    # Fallback to custom evaluation only
                    logger.warning(f"DeepEval failed for {model_name}, using custom evaluation only")
                    return {
                        "overall_score": custom_score,
                        "deepeval_metrics": deepeval_results,
                        "custom_score": custom_score,
                        "individual_scores": [custom_score],
                        "success": False,
                        "error": deepeval_results.get("error", "Unknown DeepEval error")
                    }

        except Exception as e:
            logger.error(f"Error in evaluate_benchmark_response: {e}")
            return {
                "overall_score": 0.0,
                "deepeval_metrics": {},
                "custom_score": 0.0,
                "individual_scores": [0.0],
                "success": False,
                "error": str(e)
            }

    def _custom_evaluation(self, task: Dict, response: str) -> float:
        """Custom evaluation logic for specific benchmarks"""

        if not response or len(response.strip()) < 10:
            return 0.0

        score = 0.0

        # Length and completeness (30%)
        if len(response) > 200:
            score += 0.3
        elif len(response) > 100:
            score += 0.2
        elif len(response) > 50:
            score += 0.1

        # Domain-specific evaluation (50%)
        domain = task.get("metadata", {}).get("category", "general")

        if domain == "mathematics":
            if any(word in response.lower() for word in ["proof", "therefore", "contradiction", "assume"]):
                score += 0.25
            if "√" in response or "proof" in response.lower():
                score += 0.25

        elif domain == "algorithms":
            if "def " in response or "function" in response.lower():
                score += 0.25
            if "time" in response.lower() or "complexity" in response.lower():
                score += 0.25

        elif domain == "debugging":
            if "fix" in response.lower() or "error" in response.lower():
                score += 0.25
            if "correct" in response.lower() or "solution" in response.lower():
                score += 0.25

        elif domain == "creativity":
            if len(response.split()) > 100:  # Substantial content
                score += 0.25
            if "twist" in response.lower() or "unexpected" in response.lower():
                score += 0.25

        elif domain == "analysis":
            if "mean" in response.lower() or "median" in response.lower():
                score += 0.25
            if "result" in response.lower() or "conclusion" in response.lower():
                score += 0.25

        # Expected answer matching (20%)
        if task.get("expected_answer") and task["expected_answer"] in response:
            score += 0.2

        return min(1.0, score)

    async def benchmark_models_comparison(self,
                                          tasks: List[Dict],
                                          models: List[str]) -> Dict[str, Dict[str, Any]]:
        """Compare all models with DeepEval metrics"""

        try:
            logger.info(f"Starting benchmark comparison for models: {models}")
            comparison_results = {}

            # Use new framework if available
            if self.use_new_framework and hasattr(self, 'evaluation_framework'):
                # Convert tasks to test cases
                test_cases = self.evaluation_framework.create_test_cases_from_tasks(tasks)
                
                # Use new framework for multi-provider comparison
                comparison_result = await self.evaluation_framework.evaluate_multiple_providers(
                    models, test_cases, compare_conjecture=True
                )
                
                # Convert to legacy format for compatibility
                for model_name in models:
                    direct_key = f"{model_name}_direct"
                    conjecture_key = f"{model_name}_conjecture"
                    
                    direct_result = comparison_result["providers"].get(direct_key, {})
                    conjecture_result = comparison_result["providers"].get(conjecture_key, {})
                    
                    # Extract scores for legacy format
                    direct_score = direct_result.get("overall_score", 0.0)
                    conjecture_score = conjecture_result.get("overall_score", 0.0)
                    
                    # Calculate improvement
                    improvement = ((conjecture_score - direct_score) / direct_score * 100) if direct_score > 0 else 0
                    
                    comparison_results[model_name] = {
                        "direct_avg": direct_score,
                        "conjecture_avg": conjecture_score,
                        "improvement": improvement,
                        "all_results": {
                            "direct": [direct_result],
                            "conjecture": [conjecture_result]
                        },
                        "tasks_evaluated": len(tasks),
                        "successful_evaluations": 1 if direct_result.get("success", False) else 0
                    }
                
                # Add comparison data
                if "comparison" in comparison_result:
                    comparison_results["comparison"] = comparison_result["comparison"]
                
                logger.info("Benchmark comparison completed with new framework")
                return comparison_results
            else:
                # Legacy implementation
                for model_name in models:
                    logger.info(f"Evaluating {model_name}...")

                    model_results = {
                        "direct": [],
                        "conjecture": []
                    }

                    # Get model integration for this model
                    integration = self.deepeval_runner._get_model_integration()

                    for task in tasks:
                        try:
                            # Map model name to internal identifier
                            model_mapping = {
                                "granite-tiny": "granite-tiny",
                                "gpt-oss-20b": "gpt-oss-20b",
                                "glm-4.6": "glm-4.6",
                                "ibm/granite-4-h-tiny": "granite-tiny",
                                "openrouter/gpt-oss-20b": "gpt-oss-20b",
                                "zai/glm-4.6": "glm-4.6"
                            }
                            
                            mapped_model = model_mapping.get(model_name.lower(), model_name.lower())

                            # Get direct model response
                            direct_response = await integration.get_model_response(mapped_model, task.get('prompt', ''))
                            
                            # Direct evaluation
                            direct_score = await self.evaluate_benchmark_response(
                                task, direct_response, model_name, use_conjecture=False
                            )
                            model_results["direct"].append(direct_score)

                            # Conjecture-enhanced evaluation
                            conjecture_response = await integration.get_conjecture_enhanced_response(mapped_model, task.get('prompt', ''))
                            conjecture_score = await self.evaluate_benchmark_response(
                                task, conjecture_response, model_name, use_conjecture=True
                            )
                            model_results["conjecture"].append(conjecture_score)

                        except Exception as e:
                            logger.error(f"Error evaluating task with {model_name}: {e}")
                            # Add fallback results
                            fallback_result = {
                                "overall_score": 0.0,
                                "success": False,
                                "error": str(e)
                            }
                            model_results["direct"].append(fallback_result)
                            model_results["conjecture"].append(fallback_result)

                    # Calculate averages
                    if model_results["direct"]:
                        direct_scores = [r.get("overall_score", 0.0) for r in model_results["direct"] if r.get("success", False)]
                        direct_avg = sum(direct_scores) / len(direct_scores) if direct_scores else 0.0
                    else:
                        direct_avg = 0.0

                    if model_results["conjecture"]:
                        conjecture_scores = [r.get("overall_score", 0.0) for r in model_results["conjecture"] if r.get("success", False)]
                        conjecture_avg = sum(conjecture_scores) / len(conjecture_scores) if conjecture_scores else 0.0
                    else:
                        conjecture_avg = 0.0

                    improvement = ((conjecture_avg - direct_avg) / direct_avg * 100) if direct_avg > 0 else 0

                    comparison_results[model_name] = {
                        "direct_avg": direct_avg,
                        "conjecture_avg": conjecture_avg,
                        "improvement": improvement,
                        "all_results": model_results,
                        "tasks_evaluated": len(tasks),
                        "successful_evaluations": len(direct_scores)
                    }

                logger.info("Benchmark comparison completed with legacy framework")
                return comparison_results

        except Exception as e:
            logger.error(f"Critical error in benchmark_models_comparison: {e}")
            return {"error": f"Benchmark comparison failed: {str(e)}"}

    def generate_advanced_chart(self, results: Dict[str, Dict[str, Any]]) -> str:
        """Generate advanced comparison chart"""

        chart_lines = []
        chart_lines.append("\nDEEPVAL BENCHMARK RESULTS CHART")
        chart_lines.append("=" * 90)
        chart_lines.append(f"{'Model':<20} {'Direct Score':<15} {'Conjecture Score':<18} {'Improvement':<12}")
        chart_lines.append("-" * 90)

        for model_name, model_data in results.items():
            direct_str = f"{model_data['direct_avg']:.3f}"
            conjecture_str = f"{model_data['conjecture_avg']:.3f}"
            improvement_str = f"{model_data['improvement']:+.1f}%"

            chart_lines.append(
                f"{model_name:<20} {direct_str:<15} {conjecture_str:<18} {improvement_str:<12}"
            )

        # DeepEval metrics breakdown
        chart_lines.append("\n" + "=" * 90)
        chart_lines.append("DEEPVAL METRICS BREAKDOWN")
        chart_lines.append("=" * 90)

        # Show average metrics across all models
        all_metrics = ["relevancy", "faithfulness", "exact_match", "summarization"]
        for metric in all_metrics:
            scores = []
            for model_data in results.values():
                if model_data["all_results"]["conjecture"]:
                    latest_result = model_data["all_results"]["conjecture"][0]
                    if "deepeval_metrics" in latest_result and metric in latest_result["deepeval_metrics"]:
                        scores.append(latest_result["deepeval_metrics"][metric]["score"])

            if scores:
                avg_score = sum(scores) / len(scores)
                chart_lines.append(f"{metric:<15}: {avg_score:.3f} average score")

        return "\n".join(chart_lines)