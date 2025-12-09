"""
Basic Evaluation Framework with Core DeepEval Metrics
Provides systematic evaluation using DeepEval's sophisticated metrics
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ExactMatchMetric,
    SummarizationMetric,
    BiasMetric,
    ToxicityMetric,
    GEval
)
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase

from .conjecture_llm_wrapper import (
    ConjectureLLMWrapper,
    create_conjecture_wrapper,
    get_available_conjecture_providers
)

# Configure logging
logger = logging.getLogger(__name__)


class EvaluationFramework:
    """
    Core evaluation framework using DeepEval metrics
    Supports the 3 target Conjecture providers with comprehensive evaluation
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize evaluation framework
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._metrics_cache = {}
        
        # Initialize core DeepEval metrics
        self._initialize_metrics()
        
        logger.info("EvaluationFramework initialized with DeepEval metrics")

    def _initialize_metrics(self):
        """Initialize DeepEval metrics with appropriate configuration"""
        try:
            # Use GPT-3.5-turbo as judge model for consistency
            judge_model = "gpt-3.5-turbo"
            
            self.metrics = {
                "answer_relevancy": AnswerRelevancyMetric(
                    threshold=0.5, 
                    model=judge_model
                ),
                "faithfulness": FaithfulnessMetric(
                    threshold=0.5, 
                    model=judge_model
                ),
                "exact_match": ExactMatchMetric(threshold=0.5),
                "summarization": SummarizationMetric(
                    threshold=0.5, 
                    model=judge_model
                ),
                "bias": BiasMetric(
                    threshold=0.5, 
                    model=judge_model
                ),
                "toxicity": ToxicityMetric(
                    threshold=0.5, 
                    model=judge_model
                ),
                "geval": GEval(
                    threshold=0.5,
                    model=judge_model,
                    criteria="coherence"
                )
            }
            
            logger.info(f"Initialized {len(self.metrics)} DeepEval metrics")
            
        except Exception as e:
            logger.error(f"Error initializing metrics: {e}")
            # Fallback to basic metrics
            self.metrics = {
                "exact_match": ExactMatchMetric(threshold=0.5)
            }

    def create_test_case(self, 
                      input_text: str,
                      expected_output: str = "",
                      actual_output: str = "",
                      additional_metadata: Dict[str, Any] = None) -> LLMTestCase:
        """
        Create a DeepEval test case
        
        Args:
            input_text: The input prompt or question
            expected_output: Expected answer (optional)
            actual_output: Actual model response
            additional_metadata: Additional metadata for evaluation
            
        Returns:
            DeepEval LLMTestCase instance
        """
        return LLMTestCase(
            input=input_text,
            expected_output=expected_output,
            actual_output=actual_output,
            additional_metadata=additional_metadata or {}
        )

    def create_test_cases_from_tasks(self, 
                                  tasks: List[Dict[str, Any]]) -> List[LLMTestCase]:
        """
        Convert task dictionaries to DeepEval test cases
        
        Args:
            tasks: List of task dictionaries with prompt, expected_answer, etc.
            
        Returns:
            List of DeepEval test cases
        """
        test_cases = []
        
        for i, task in enumerate(tasks):
            try:
                test_case = self.create_test_case(
                    input_text=task.get('prompt', ''),
                    expected_output=task.get('expected_answer', ''),
                    actual_output=task.get('model_response', ''),
                    additional_metadata={
                        'task_id': i,
                        'category': task.get('category', 'general'),
                        'difficulty': task.get('difficulty', 'medium'),
                        **task.get('metadata', {})
                    }
                )
                test_cases.append(test_case)
                
            except Exception as e:
                logger.error(f"Error creating test case {i}: {e}")
                continue
        
        logger.info(f"Created {len(test_cases)} test cases from {len(tasks)} tasks")
        return test_cases

    async def evaluate_provider(self,
                           provider_name: str,
                           test_cases: List[LLMTestCase],
                           use_conjecture: bool = False,
                           metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate a provider using DeepEval metrics
        
        Args:
            provider_name: Name of the provider to evaluate
            test_cases: List of test cases to evaluate
            use_conjecture: Whether to use Conjecture enhancement
            metrics: Optional list of specific metrics to use
            
        Returns:
            Evaluation results dictionary
        """
        try:
            logger.info(f"Evaluating provider {provider_name} (conjecture={use_conjecture})")
            
            # Create Conjecture wrapper
            wrapper = create_conjecture_wrapper(provider_name, use_conjecture)
            
            # Determine which metrics to use
            metrics_to_use = metrics or list(self.metrics.keys())
            
            # Prepare evaluation results
            evaluation_results = {
                "provider": provider_name,
                "use_conjecture": use_conjecture,
                "test_cases_count": len(test_cases),
                "evaluation_timestamp": datetime.utcnow().isoformat(),
                "metrics_results": {},
                "overall_score": 0.0,
                "success": True
            }
            
            # Evaluate each metric
            metric_scores = []
            
            for metric_name in metrics_to_use:
                if metric_name not in self.metrics:
                    logger.warning(f"Metric {metric_name} not available, skipping")
                    continue
                
                try:
                    logger.info(f"Evaluating with {metric_name} metric...")
                    metric = self.metrics[metric_name]
                    
                    # Create dataset for this metric
                    dataset = EvaluationDataset(goldens=test_cases)
                    
                    # Run evaluation
                    eval_results = evaluate(
                        dataset=dataset,
                        metrics=[metric],
                        model=wrapper,
                        run_async=True
                    )
                    
                    # Process results
                    if eval_results and len(eval_results) > 0:
                        result = eval_results[0]
                        metric_score = result.score if hasattr(result, 'score') else 0.0
                        metric_success = result.score >= metric.threshold if hasattr(result, 'score') else False
                        
                        evaluation_results["metrics_results"][metric_name] = {
                            "score": metric_score,
                            "threshold": metric.threshold,
                            "success": metric_success,
                            "reason": getattr(result, 'reason', 'No reason provided')
                        }
                        
                        metric_scores.append(metric_score)
                        logger.info(f"Metric {metric_name}: {metric_score:.3f} (success: {metric_success})")
                    else:
                        evaluation_results["metrics_results"][metric_name] = {
                            "score": 0.0,
                            "threshold": metric.threshold,
                            "success": False,
                            "reason": "No evaluation result"
                        }
                        logger.warning(f"No result for metric {metric_name}")
                        
                except Exception as e:
                    logger.error(f"Error evaluating metric {metric_name}: {e}")
                    evaluation_results["metrics_results"][metric_name] = {
                        "score": 0.0,
                        "threshold": 0.5,
                        "success": False,
                        "reason": f"Evaluation error: {str(e)}"
                    }
            
            # Calculate overall score
            if metric_scores:
                evaluation_results["overall_score"] = sum(metric_scores) / len(metric_scores)
            else:
                evaluation_results["overall_score"] = 0.0
                evaluation_results["success"] = False
            
            logger.info(f"Provider {provider_name} evaluation completed: {evaluation_results['overall_score']:.3f}")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Critical error evaluating provider {provider_name}: {e}")
            return {
                "provider": provider_name,
                "use_conjecture": use_conjecture,
                "error": str(e),
                "success": False,
                "overall_score": 0.0
            }

    async def evaluate_multiple_providers(self,
                                     provider_names: List[str],
                                     test_cases: List[LLMTestCase],
                                     compare_conjecture: bool = True) -> Dict[str, Any]:
        """
        Evaluate multiple providers and compare results
        
        Args:
            provider_names: List of provider names to evaluate
            test_cases: Test cases to use for evaluation
            compare_conjecture: Whether to compare with Conjecture enhancement
            
        Returns:
            Comparison results dictionary
        """
        try:
            logger.info(f"Starting multi-provider evaluation for: {provider_names}")
            
            comparison_results = {
                "evaluation_timestamp": datetime.utcnow().isoformat(),
                "test_cases_count": len(test_cases),
                "providers": {},
                "comparison": {}
            }
            
            # Evaluate each provider
            for provider_name in provider_names:
                # Direct evaluation
                direct_result = await self.evaluate_provider(
                    provider_name, test_cases, use_conjecture=False
                )
                comparison_results["providers"][f"{provider_name}_direct"] = direct_result
                
                # Conjecture-enhanced evaluation (if requested)
                if compare_conjecture:
                    conjecture_result = await self.evaluate_provider(
                        provider_name, test_cases, use_conjecture=True
                    )
                    comparison_results["providers"][f"{provider_name}_conjecture"] = conjecture_result
            
            # Generate comparison metrics
            comparison_results["comparison"] = self._generate_comparison_metrics(
                comparison_results["providers"], provider_names
            )
            
            logger.info("Multi-provider evaluation completed")
            return comparison_results
            
        except Exception as e:
            logger.error(f"Error in multi-provider evaluation: {e}")
            return {
                "error": str(e),
                "success": False,
                "evaluation_timestamp": datetime.utcnow().isoformat()
            }

    def _generate_comparison_metrics(self, 
                                 provider_results: Dict[str, Any],
                                 provider_names: List[str]) -> Dict[str, Any]:
        """Generate comparison metrics from provider results"""
        comparison = {
            "best_overall": {"provider": "", "score": 0.0},
            "best_direct": {"provider": "", "score": 0.0},
            "best_conjecture": {"provider": "", "score": 0.0},
            "improvements": {},
            "metric_rankings": {}
        }
        
        # Track best scores
        best_overall_score = 0.0
        best_direct_score = 0.0
        best_conjecture_score = 0.0
        
        # Analyze each provider
        for provider_name in provider_names:
            direct_key = f"{provider_name}_direct"
            conjecture_key = f"{provider_name}_conjecture"
            
            # Get direct and conjecture results
            direct_result = provider_results.get(direct_key, {})
            conjecture_result = provider_results.get(conjecture_key, {})
            
            direct_score = direct_result.get("overall_score", 0.0)
            conjecture_score = conjecture_result.get("overall_score", 0.0)
            
            # Track best performers
            if direct_score > best_direct_score:
                best_direct_score = direct_score
                comparison["best_direct"]["provider"] = provider_name
                comparison["best_direct"]["score"] = direct_score
            
            if conjecture_score > best_conjecture_score:
                best_conjecture_score = conjecture_score
                comparison["best_conjecture"]["provider"] = provider_name
                comparison["best_conjecture"]["score"] = conjecture_score
            
            # Calculate improvement
            if direct_score > 0:
                improvement = ((conjecture_score - direct_score) / direct_score) * 100
                comparison["improvements"][provider_name] = {
                    "direct_score": direct_score,
                    "conjecture_score": conjecture_score,
                    "improvement_percent": improvement
                }
            
            # Track best overall (prefer conjecture-enhanced)
            overall_best_score = max(direct_score, conjecture_score)
            if overall_best_score > best_overall_score:
                best_overall_score = overall_best_score
                comparison["best_overall"]["provider"] = provider_name
                comparison["best_overall"]["score"] = overall_best_score
                comparison["best_overall"]["mode"] = "conjecture" if conjecture_score > direct_score else "direct"
        
        return comparison

    def get_evaluation_summary(self, results: Dict[str, Any]) -> str:
        """
        Generate human-readable summary of evaluation results
        
        Args:
            results: Evaluation results dictionary
            
        Returns:
            Formatted summary string
        """
        try:
            if "error" in results:
                return f"Evaluation Error: {results['error']}"
            
            summary_lines = []
            summary_lines.append("=" * 80)
            summary_lines.append("DEEPEVAL EVALUATION SUMMARY")
            summary_lines.append("=" * 80)
            
            # Provider information
            if "provider" in results:
                provider = results["provider"]
                summary_lines.append(f"Provider: {provider}")
                summary_lines.append(f"Conjecture Enhanced: {results.get('use_conjecture', False)}")
                summary_lines.append(f"Test Cases: {results.get('test_cases_count', 0)}")
                summary_lines.append(f"Overall Score: {results.get('overall_score', 0.0):.3f}")
                summary_lines.append("")
            
            # Metrics breakdown
            if "metrics_results" in results:
                summary_lines.append("METRICS BREAKDOWN:")
                summary_lines.append("-" * 40)
                
                for metric_name, metric_result in results["metrics_results"].items():
                    score = metric_result.get("score", 0.0)
                    success = metric_result.get("success", False)
                    threshold = metric_result.get("threshold", 0.5)
                    status = "✓ PASS" if success else "✗ FAIL"
                    
                    summary_lines.append(f"{metric_name:<20}: {score:.3f} (threshold: {threshold}) [{status}]")
            
            # Comparison information
            if "comparison" in results:
                comparison = results["comparison"]
                summary_lines.append("")
                summary_lines.append("COMPARISON RESULTS:")
                summary_lines.append("-" * 40)
                
                if "best_overall" in comparison:
                    best = comparison["best_overall"]
                    summary_lines.append(f"Best Overall: {best.get('provider', 'N/A')} ({best.get('score', 0.0):.3f})")
                    summary_lines.append(f"Best Mode: {best.get('mode', 'N/A')}")
                
                if "improvements" in comparison:
                    summary_lines.append("")
                    summary_lines.append("CONJECTURE IMPROVEMENTS:")
                    summary_lines.append("-" * 30)
                    
                    for provider, improvement in comparison["improvements"].items():
                        direct = improvement.get("direct_score", 0.0)
                        conjecture = improvement.get("conjecture_score", 0.0)
                        improvement_pct = improvement.get("improvement_percent", 0.0)
                        
                        summary_lines.append(f"{provider}:")
                        summary_lines.append(f"  Direct: {direct:.3f}")
                        summary_lines.append(f"  Conjecture: {conjecture:.3f}")
                        summary_lines.append(f"  Improvement: {improvement_pct:+.1f}%")
            
            summary_lines.append("=" * 80)
            return "\n".join(summary_lines)
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Error generating summary: {str(e)}"

    def get_available_metrics(self) -> List[str]:
        """Get list of available metrics"""
        return list(self.metrics.keys())

    def get_metric_info(self, metric_name: str) -> Dict[str, Any]:
        """Get information about a specific metric"""
        if metric_name not in self.metrics:
            return {"error": f"Metric {metric_name} not available"}
        
        metric = self.metrics[metric_name]
        return {
            "name": metric_name,
            "threshold": metric.threshold,
            "type": type(metric).__name__,
            "description": getattr(metric, '__doc__', 'No description available')
        }


# Convenience functions for common evaluation scenarios
async def evaluate_single_provider(provider_name: str, 
                               test_prompt: str,
                               expected_answer: str = "",
                               use_conjecture: bool = False) -> Dict[str, Any]:
    """
    Convenience function to evaluate a single provider with one test case
    
    Args:
        provider_name: Provider to evaluate
        test_prompt: Test prompt
        expected_answer: Expected answer (optional)
        use_conjecture: Whether to use Conjecture enhancement
        
    Returns:
        Evaluation result
    """
    framework = EvaluationFramework()
    
    test_case = framework.create_test_case(
        input_text=test_prompt,
        expected_output=expected_answer
    )
    
    return await framework.evaluate_provider(
        provider_name, [test_case], use_conjecture
    )


async def evaluate_all_providers(test_cases: List[Dict[str, Any]],
                             compare_conjecture: bool = True) -> Dict[str, Any]:
    """
    Convenience function to evaluate all available Conjecture providers
    
    Args:
        test_cases: List of test case dictionaries
        compare_conjecture: Whether to compare with Conjecture enhancement
        
    Returns:
        Multi-provider comparison results
    """
    framework = EvaluationFramework()
    
    # Get available providers
    providers = get_available_conjecture_providers()
    
    # Convert to DeepEval test cases
    deepeval_cases = framework.create_test_cases_from_tasks(test_cases)
    
    # Evaluate all providers
    return await framework.evaluate_multiple_providers(
        providers, deepeval_cases, compare_conjecture
    )