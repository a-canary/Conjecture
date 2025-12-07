"""
Comprehensive Testing Suite for Tiny LLM Prompt Optimizer

Tests the prompt engineering optimizer with various scenarios and validation metrics.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import pytest

from src.processing.llm_prompts.tiny_llm_optimizer import (
    TinyLLMPromptOptimizer,
    AdaptiveTemplateGenerator,
    TinyModelCapabilityProfiler,
    PromptPerformanceAnalyzer,
    TaskDescription,
    TaskComplexity,
    OptimizationStrategy
)
from src.processing.llm_prompts.template_evolution import (
    TemplateEvolution,
    EvolutionConfig,
    TemplateGenome
)
from src.processing.llm_prompts.models import (
    PromptTemplate,
    LLMResponse,
    PromptTemplateType
)


@dataclass
class TestScenario:
    """Test scenario for prompt optimization"""
    name: str
    task_type: str
    complexity: TaskComplexity
    input_text: str
    context_items: List[str]
    expected_output_format: str
    model_name: str = "granite-tiny"


class TinyLLMOptimizerTestSuite:
    """Comprehensive test suite for tiny LLM prompt optimizer"""

    def __init__(self):
        """Initialize test suite"""
        self.optimizer = TinyLLMPromptOptimizer()
        self.test_results = []
        self.performance_benchmarks = {}

    def create_test_scenarios(self) -> List[TestScenario]:
        """Create comprehensive test scenarios"""

        scenarios = [
            # Simple scenarios
            TestScenario(
                name="simple_extraction",
                task_type="extraction",
                complexity=TaskComplexity.SIMPLE,
                input_text="The temperature is 25 degrees Celsius.",
                context_items=["Weather report", "Temperature measurements"],
                expected_output_format="structured",
                model_name="granite-tiny"
            ),
            TestScenario(
                name="simple_classification",
                task_type="classification",
                complexity=TaskComplexity.SIMPLE,
                input_text="This product is excellent quality.",
                context_items=["Customer reviews", "Product categories"],
                expected_output_format="category",
                model_name="llama-3.2-1b"
            ),

            # Moderate complexity scenarios
            TestScenario(
                name="moderate_analysis",
                task_type="analysis",
                complexity=TaskComplexity.MODERATE,
                input_text="Sales data shows Q1 growth of 15% and Q2 decline of 5%.",
                context_items=["Sales reports", "Market trends", "Competitor analysis"],
                expected_output_format="analysis",
                model_name="phi-3-mini"
            ),
            TestScenario(
                name="moderate_coding",
                task_type="coding",
                complexity=TaskComplexity.MODERATE,
                input_text="Create a function that sorts a list of numbers.",
                context_items=["Programming examples", "Algorithm references"],
                expected_output_format="code",
                model_name="llama-3.2-1b"
            ),

            # Complex scenarios
            TestScenario(
                name="complex_research",
                task_type="research",
                complexity=TaskComplexity.COMPLEX,
                input_text="Analyze the impact of climate change on coastal ecosystems.",
                context_items=[
                    "Climate research papers",
                    "Ecosystem studies",
                    "Environmental impact reports",
                    "Sea level data",
                    "Biodiversity assessments"
                ],
                expected_output_format="comprehensive_report",
                model_name="granite-tiny"
            ),
            TestScenario(
                name="very_complex_synthesis",
                task_type="synthesis",
                complexity=TaskComplexity.VERY_COMPLEX,
                input_text="Integrate findings from multiple research areas to propose a new framework.",
                context_items=[
                    "Cross-disciplinary research",
                    "Framework designs",
                    "Case studies",
                    "Best practices",
                    "Implementation guidelines",
                    "Evaluation metrics",
                    "Stakeholder feedback"
                ],
                expected_output_format="framework_proposal",
                model_name="phi-3-mini"
            )
        ]

        return scenarios

    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite"""

        scenarios = self.create_test_scenarios()
        test_results = []

        print(f"Running {len(scenarios)} test scenarios...")

        for scenario in scenarios:
            print(f"\nTesting scenario: {scenario.name}")

            # Create task description
            task = TaskDescription(
                task_type=scenario.task_type,
                complexity=scenario.complexity,
                required_inputs=[scenario.input_text],
                expected_output_format=scenario.expected_output_format,
                context_requirements=scenario.context_items,
                token_budget=500 if scenario.complexity == TaskComplexity.SIMPLE else 1000
            )

            # Test optimization
            optimization_result = await self.test_prompt_optimization(
                task, scenario.context_items, scenario.model_name, scenario
            )

            # Test template generation
            template_result = await self.test_template_generation(
                task, scenario.model_name, scenario
            )

            # Test performance analysis
            performance_result = await self.test_performance_analysis(
                scenario, optimization_result
            )

            scenario_result = {
                "scenario": scenario.name,
                "optimization": optimization_result,
                "template_generation": template_result,
                "performance_analysis": performance_result,
                "overall_score": self._calculate_scenario_score(
                    optimization_result, template_result, performance_result
                )
            }

            test_results.append(scenario_result)

        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(test_results)

        test_summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_scenarios": len(scenarios),
            "test_results": test_results,
            "overall_metrics": overall_metrics,
            "recommendations": self._generate_recommendations(test_results)
        }

        # Save results
        await self._save_test_results(test_summary)

        return test_summary

    async def test_prompt_optimization(
        self,
        task: TaskDescription,
        context_items: List[str],
        model_name: str,
        scenario: TestScenario
    ) -> Dict[str, Any]:
        """Test prompt optimization for scenario"""

        start_time = time.time()

        try:
            # Optimize prompt
            optimized_prompt = await self.optimizer.optimize_prompt(
                task=task,
                context_items=context_items,
                model_name=model_name,
                optimization_strategy=None  # Let optimizer decide
            )

            optimization_time = (time.time() - start_time) * 1000

            # Validate optimized prompt
            validation_results = self._validate_optimized_prompt(
                optimized_prompt, task, model_name
            )

            # Simulate LLM response (for testing)
            mock_response = self._create_mock_response(
                optimized_prompt.optimized_prompt, scenario
            )

            # Analyze performance
            performance_metrics = await self.optimizer.analyze_performance(
                prompt=optimized_prompt,
                response=mock_response,
                task_success=True,  # Assume success for testing
                quality_metrics={"relevance": 0.8, "accuracy": 0.7}
            )

            return {
                "success": True,
                "optimization_time_ms": optimization_time,
                "optimized_prompt": optimized_prompt.optimized_prompt,
                "optimization_strategy": optimized_prompt.optimization_strategy,
                "validation_results": validation_results,
                "performance_metrics": performance_metrics.__dict__ if performance_metrics else None,
                "token_count": len(optimized_prompt.optimized_prompt.split()),
                "score": validation_results["score"]
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "optimization_time_ms": (time.time() - start_time) * 1000,
                "score": 0.0
            }

    async def test_template_generation(
        self,
        task: TaskDescription,
        model_name: str,
        scenario: TestScenario
    ) -> Dict[str, Any]:
        """Test adaptive template generation"""

        start_time = time.time()

        try:
            # Get model capabilities
            profiler = TinyModelCapabilityProfiler()
            model_capabilities = profiler.get_model_capabilities(model_name)

            if not model_capabilities:
                model_capabilities = profiler.analyze_model_characteristics(model_name, [])

            # Generate template
            template_generator = AdaptiveTemplateGenerator()
            template = await template_generator.generate_template(
                task=task,
                model_capabilities=model_capabilities,
                optimization_strategy=OptimizationStrategy.ADAPTIVE
            )

            generation_time = (time.time() - start_time) * 1000

            # Validate template
            validation_results = self._validate_template(template, task, model_capabilities)

            # Test template rendering
            render_test = self._test_template_rendering(template, task)

            return {
                "success": True,
                "generation_time_ms": generation_time,
                "template_id": template.id,
                "template_type": template.template_type.value,
                "validation_results": validation_results,
                "render_test": render_test,
                "variable_count": len(template.variables),
                "score": validation_results["score"]
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "generation_time_ms": (time.time() - start_time) * 1000,
                "score": 0.0
            }

    async def test_performance_analysis(
        self,
        scenario: TestScenario,
        optimization_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Test performance analysis capabilities"""

        start_time = time.time()

        try:
            # Create test response data
            test_responses = [
                {
                    "success": True,
                    "response_time": 2000,
                    "token_usage": {"total_tokens": 300, "prompt_tokens": 150, "completion_tokens": 150},
                    "quality_score": 0.8
                },
                {
                    "success": True,
                    "response_time": 1800,
                    "token_usage": {"total_tokens": 280, "prompt_tokens": 140, "completion_tokens": 140},
                    "quality_score": 0.7
                },
                {
                    "success": False,
                    "response_time": 5000,
                    "token_usage": {"total_tokens": 400, "prompt_tokens": 200, "completion_tokens": 200},
                    "quality_score": 0.3
                }
            ]

            # Analyze performance trends
            analyzer = PromptPerformanceAnalyzer()
            trends = []

            for i, response_data in enumerate(test_responses):
                # Create mock LLM response
                mock_response = LLMResponse(
                    content=f"Mock response {i}",
                    model=scenario.model_name,
                    token_usage=response_data["token_usage"],
                    response_time_ms=response_data["response_time"]
                )

                # Analyze performance
                metrics = await analyzer.analyze_performance(
                    prompt=PromptTemplate(
                        id="test",
                        name="Test",
                        description="Test template",
                        template_type=PromptTemplateType.GENERAL,
                        template_content="Test content",
                        variables=[]
                    ),
                    response=mock_response,
                    task_success=response_data["success"],
                    quality_metrics={"quality": response_data["quality_score"]}
                )

                trends.append(metrics.__dict__ if metrics else None)

            analysis_time = (time.time() - start_time) * 1000

            # Calculate performance statistics
            success_rate = sum(1 for r in test_responses if r["success"]) / len(test_responses)
            avg_response_time = sum(r["response_time"] for r in test_responses) / len(test_responses)
            avg_tokens = sum(r["token_usage"]["total_tokens"] for r in test_responses) / len(test_responses)

            return {
                "success": True,
                "analysis_time_ms": analysis_time,
                "performance_trends": trends,
                "statistics": {
                    "success_rate": success_rate,
                    "avg_response_time_ms": avg_response_time,
                    "avg_token_usage": avg_tokens,
                    "performance_score": (success_rate + (avg_tokens / 1000)) / 2
                },
                "score": success_rate * 100
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "analysis_time_ms": (time.time() - start_time) * 1000,
                "score": 0.0
            }

    def _validate_optimized_prompt(
        self,
        optimized_prompt,
        task: TaskDescription,
        model_name: str
    ) -> Dict[str, Any]:
        """Validate optimized prompt quality"""

        prompt_text = optimized_prompt.optimized_prompt
        score = 0.0
        issues = []

        # Check length constraints
        word_count = len(prompt_text.split())
        if word_count <= 500:  # Suitable for tiny models
            score += 30
        else:
            issues.append("Prompt too long for tiny model")

        # Check structure
        if task.complexity == TaskComplexity.SIMPLE:
            if word_count <= 100:
                score += 20
            else:
                issues.append("Simple task prompt too verbose")
        elif task.complexity in [TaskComplexity.COMPLEX, TaskComplexity.VERY_COMPLEX]:
            if word_count >= 200:
                score += 20
            else:
                issues.append("Complex task prompt too brief")

        # Check for required elements
        required_elements = ["task", "input", "format"]
        for element in required_elements:
            if element in prompt_text.lower():
                score += 10
            else:
                issues.append(f"Missing {element} in prompt")

        # Check clarity
        if "EXAMPLE:" in prompt_text or "example" in prompt_text.lower():
            score += 10

        # Check model-specific optimizations
        if "xml" in prompt_text.lower() and model_name == "granite-tiny":
            score += 10
        elif "json" in prompt_text.lower() and model_name == "llama-3.2-1b":
            score += 10

        return {
            "score": min(100, score),
            "word_count": word_count,
            "issues": issues,
            "passes_constraints": score >= 70
        }

    def _validate_template(
        self,
        template: PromptTemplate,
        task: TaskDescription,
        model_capabilities
    ) -> Dict[str, Any]:
        """Validate generated template"""

        score = 0.0
        issues = []

        # Check template completeness
        if template.name and template.description:
            score += 20
        else:
            issues.append("Missing template metadata")

        # Check variables
        if template.variables:
            score += 10
        else:
            issues.append("No template variables defined")

        # Check template type appropriateness
        expected_type = {
            "research": PromptTemplateType.RESEARCH,
            "analysis": PromptTemplateType.ANALYSIS,
            "coding": PromptTemplateType.CODING,
            "extraction": PromptTemplateType.VALIDATION
        }.get(task.task_type, PromptTemplateType.GENERAL)

        if template.template_type == expected_type:
            score += 20
        else:
            issues.append(f"Template type mismatch for {task.task_type}")

        # Check content quality
        content = template.template_content
        if len(content) > 50:  # Minimum content length
            score += 15
        else:
            issues.append("Template content too short")

        # Check model compatibility
        if model_capabilities:
            if model_capabilities.preferred_structure in content.lower():
                score += 15
            else:
                issues.append("Template not optimized for model preferences")

        # Check adaptability
        if "{{user_input}}" in content or "{{input}}" in content:
            score += 10
        else:
            issues.append("Template lacks input placeholders")

        # Check task complexity handling
        if task.complexity == TaskComplexity.SIMPLE:
            if len(content.split()) < 100:
                score += 10
        else:
            if "context" in content.lower():
                score += 10

        return {
            "score": min(100, score),
            "content_length": len(content),
            "variable_count": len(template.variables),
            "issues": issues,
            "passes_validation": score >= 70
        }

    def _test_template_rendering(self, template: PromptTemplate, task: TaskDescription) -> Dict[str, Any]:
        """Test template rendering functionality"""

        try:
            # Create test variables
            test_variables = {
                "user_input": task.required_inputs[0] if task.required_inputs else "Test input",
                "context": "Test context information"
            }

            # Render template
            rendered = template.render(test_variables)

            # Check rendering quality
            missing_placeholders = []
            for var in template.variables:
                if var.required and var.name not in test_variables and var.default_value is None:
                    missing_placeholders.append(var.name)

            return {
                "success": True,
                "rendered_length": len(rendered),
                "missing_placeholders": missing_placeholders,
                "rendering_quality": len(missing_placeholders) == 0
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "rendering_quality": False
            }

    def _create_mock_response(self, prompt: str, scenario: TestScenario) -> LLMResponse:
        """Create mock LLM response for testing"""

        # Generate appropriate mock response based on task type
        responses = {
            "extraction": f"Extracted information from: {scenario.input_text[:50]}...",
            "analysis": f"Analysis of {scenario.input_text[:30]} shows relevant patterns.",
            "coding": "def example_function():\n    return 'example'",
            "research": f"Research findings on {scenario.input_text[:30]} indicate...",
            "synthesis": f"Synthesis of available information regarding {scenario.input_text[:30]}..."
        }

        mock_content = responses.get(scenario.task_type, "Generated response")

        return LLMResponse(
            content=mock_content,
            model=scenario.model_name,
            token_usage={
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(mock_content.split()),
                "total_tokens": len(prompt.split()) + len(mock_content.split())
            },
            response_time_ms=1500
        )

    def _calculate_scenario_score(
        self,
        optimization_result: Dict[str, Any],
        template_result: Dict[str, Any],
        performance_result: Dict[str, Any]
    ) -> float:
        """Calculate overall score for scenario"""

        weights = {
            "optimization": 0.4,
            "template": 0.3,
            "performance": 0.3
        }

        scores = [
            optimization_result.get("score", 0) * weights["optimization"],
            template_result.get("score", 0) * weights["template"],
            performance_result.get("score", 0) * weights["performance"]
        ]

        return sum(scores)

    def _calculate_overall_metrics(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall test metrics"""

        if not test_results:
            return {}

        scores = [result["overall_score"] for result in test_results]
        optimization_scores = [result["optimization"]["score"] for result in test_results if result["optimization"].get("success", False)]
        template_scores = [result["template_generation"]["score"] for result in test_results if result["template_generation"].get("success", False)]
        performance_scores = [result["performance_analysis"]["score"] for result in test_results if result["performance_analysis"].get("success", False)]

        return {
            "average_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "success_rate": sum(1 for result in test_results if result["overall_score"] >= 70) / len(test_results),
            "optimization_avg": sum(optimization_scores) / len(optimization_scores) if optimization_scores else 0,
            "template_avg": sum(template_scores) / len(template_scores) if template_scores else 0,
            "performance_avg": sum(performance_scores) / len(performance_scores) if performance_scores else 0,
            "total_tests": len(test_results)
        }

    def _generate_recommendations(self, test_results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on test results"""

        recommendations = []

        overall_metrics = self._calculate_overall_metrics(test_results)

        if overall_metrics.get("success_rate", 0) < 0.8:
            recommendations.append("Overall success rate needs improvement - investigate failure patterns")

        if overall_metrics.get("optimization_avg", 0) < 70:
            recommendations.append("Prompt optimization scoring below threshold - review optimization algorithms")

        if overall_metrics.get("template_avg", 0) < 70:
            recommendations.append("Template generation needs improvement - review template patterns")

        if overall_metrics.get("performance_avg", 0) < 70:
            recommendations.append("Performance analysis not meeting standards - review metrics calculation")

        # Analyze specific failures
        for result in test_results:
            if result["optimization"].get("success") == False:
                recommendations.append(f"Fix optimization errors in {result['scenario']}")

            if result["template_generation"].get("success") == False:
                recommendations.append(f"Fix template generation errors in {result['scenario']}")

            if result["performance_analysis"].get("success") == False:
                recommendations.append(f"Fix performance analysis errors in {result['scenario']}")

        return list(set(recommendations))  # Remove duplicates

    async def _save_test_results(self, test_summary: Dict[str, Any]) -> None:
        """Save test results to file"""

        filename = f"tiny_llm_optimizer_test_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"

        try:
            with open(filename, 'w') as f:
                json.dump(test_summary, f, indent=2, default=str)
            print(f"\nTest results saved to: {filename}")
        except Exception as e:
            print(f"Failed to save test results: {e}")


async def run_tiny_llm_optimizer_tests():
    """Run the complete test suite"""

    print("=" * 60)
    print("Tiny LLM Prompt Optimizer Test Suite")
    print("=" * 60)

    test_suite = TinyLLMOptimizerTestSuite()

    # Run comprehensive tests
    results = await test_suite.run_comprehensive_tests()

    # Display summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    overall_metrics = results.get("overall_metrics", {})
    print(f"Total Scenarios: {overall_metrics.get('total_tests', 0)}")
    print(f"Overall Success Rate: {overall_metrics.get('success_rate', 0):.2%}")
    print(f"Average Score: {overall_metrics.get('average_score', 0):.1f}/100")
    print(f"Best Score: {overall_metrics.get('max_score', 0):.1f}/100")
    print(f"Worst Score: {overall_metrics.get('min_score', 0):.1f}/100")

    print(f"\nComponent Averages:")
    print(f"  Optimization: {overall_metrics.get('optimization_avg', 0):.1f}/100")
    print(f"  Template Generation: {overall_metrics.get('template_avg', 0):.1f}/100")
    print(f"  Performance Analysis: {overall_metrics.get('performance_avg', 0):.1f}/100")

    # Display recommendations
    recommendations = results.get("recommendations", [])
    if recommendations:
        print(f"\nRecommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")

    # Display scenario details
    print(f"\nScenario Results:")
    for result in results.get("test_results", []):
        scenario_name = result.get("scenario", "Unknown")
        overall_score = result.get("overall_score", 0)
        optimization_success = result.get("optimization", {}).get("success", False)
        template_success = result.get("template_generation", {}).get("success", False)
        performance_success = result.get("performance_analysis", {}).get("success", False)

        print(f"  {scenario_name}: {overall_score:.1f}/100 "
              f"(Opt: {optimization_success}, Temp: {template_success}, Perf: {performance_success})")

    return results


if __name__ == "__main__":
    # Run the test suite
    asyncio.run(run_tiny_llm_optimizer_tests())