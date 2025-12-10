#!/usr/bin/env python3
"""
Baseline Comparison Experiments
Compares Conjecture's claims-based approach against direct prompting baselines
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

# Add src to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from processing.llm.llm_manager import LLMManager
from config.common import ProviderConfig
from .experiment_framework import ExperimentConfig, TestResult, EvaluationResult, ExperimentRun
from .llm_judge import LLMJudge, EvaluationCriterion
from ..analysis.statistical_analyzer import StatisticalAnalyzer, ABTestResult

class BaselineType(str, Enum):
    """Types of baseline approaches to compare against"""
    DIRECT_PROMPT = "direct_prompt"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    ZERO_SHOT_COT = "zero_shot_cot"
    TEMPLATE_BASED = "template_based"

@dataclass
class BaselineConfig:
    """Configuration for baseline comparison"""
    baseline_type: BaselineType
    name: str
    description: str
    prompt_template: str
    parameters: Dict[str, Any]
    expected_advantages: List[str]

@dataclass
class ComparisonResult:
    """Result from comparing Conjecture vs baseline"""
    test_case_id: str
    model_name: str
    conjecture_result: TestResult
    baseline_result: TestResult
    conjecture_evaluations: Dict[EvaluationCriterion, Any]
    baseline_evaluations: Dict[EvaluationCriterion, Any]
    performance_comparison: Dict[str, float]
    winner: Optional[str]  # "conjecture", "baseline", or "tie"
    confidence_in_winner: float
    analysis: str

@dataclass
class ABRunConfig:
    """Configuration for A/B testing runs"""
    run_id: str
    experiment_config: ExperimentConfig
    baseline_configs: List[BaselineConfig]
    models_to_test: List[str]
    test_cases: List[Dict[str, Any]]
    sample_size_per_condition: int
    randomize_order: bool = True
    statistical_threshold: float = 0.05

class BaselineEngine:
    """Engine for executing baseline approaches"""

    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.baseline_configs = self._initialize_baseline_configs()

    def _initialize_baseline_configs(self) -> Dict[BaselineType, BaselineConfig]:
        """Initialize baseline configurations"""
        configs = {}

        # Direct prompt baseline
        configs[BaselineType.DIRECT_PROMPT] = BaselineConfig(
            baseline_type=BaselineType.DIRECT_PROMPT,
            name="Direct Prompting",
            description="Simple direct prompting without any specialized techniques",
            prompt_template="{question}",
            parameters={},
            expected_advantages=["simplicity", "speed", "low token usage"]
        )

        # Few-shot baseline
        configs[BaselineType.FEW_SHOT] = BaselineConfig(
            baseline_type=BaselineType.FEW_SHOT,
            name="Few-Shot Learning",
            description="Few-shot prompting with examples",
            prompt_template="""Here are some examples:

{examples}

Now, please answer: {question}""",
            parameters={"examples": []},
            expected_advantages=["context learning", "improved accuracy"]
        )

        # Chain of thought baseline
        configs[BaselineType.CHAIN_OF_THOUGHT] = BaselineConfig(
            baseline_type=BaselineType.CHAIN_OF_THOUGHT,
            name="Chain of Thought",
            description="Step-by-step reasoning approach",
            prompt_template="""Think step by step to answer this question:

{question}

Please show your reasoning process and then provide your final answer.""",
            parameters={},
            expected_advantages=["structured reasoning", "better accuracy"]
        )

        # Zero-shot CoT baseline
        configs[BaselineType.ZERO_SHOT_COT] = BaselineConfig(
            baseline_type=BaselineType.ZERO_SHOT_COT,
            name="Zero-Shot Chain of Thought",
            description="Zero-shot chain of thought prompting",
            prompt_template="""{question}

Let's think step by step.""",
            parameters={},
            expected_advantages=["no examples needed", "structured reasoning"]
        )

        # Template-based baseline
        configs[BaselineType.TEMPLATE_BASED] = BaselineConfig(
            baseline_type=BaselineType.TEMPLATE_BASED,
            name="Template-based",
            description="Structured template-based prompting",
            prompt_template="""Task: {task}

Context: {context}

Question: {question}

Instructions:
1. Analyze the given information
2. Consider the context carefully
3. Provide a comprehensive answer
4. Justify your reasoning

Answer:""",
            parameters={"task": "Answer the following question", "context": ""},
            expected_advantages=["structured approach", "comprehensive responses"]
        )

        return configs

    async def execute_baseline(self,
                             baseline_config: BaselineConfig,
                             question: str,
                             model_name: str,
                             context: Optional[str] = None,
                             examples: Optional[List[str]] = None) -> TestResult:
        """Execute a baseline approach and return results"""
        
        # Prepare prompt based on baseline type
        if baseline_config.baseline_type == BaselineType.FEW_SHOT and examples:
            examples_text = "\n\n".join(examples)
            prompt = baseline_config.prompt_template.format(
                examples=examples_text,
                question=question
            )
        elif baseline_config.baseline_type == BaselineType.TEMPLATE_BASED:
            prompt = baseline_config.prompt_template.format(
                task=baseline_config.parameters.get("task", "Answer the following question"),
                context=context or "",
                question=question
            )
        else:
            prompt = baseline_config.prompt_template.format(question=question)

        # Time the execution
        start_time = time.time()

        # Generate response
        try:
            response = await self.llm_manager.generate_response(
                prompt=prompt,
                model=model_name,
                max_tokens=2000,
                temperature=0.7
            )
            error = None
        except Exception as e:
            response = ""
            error = str(e)

        execution_time = time.time() - start_time

        # Create test result
        result = TestResult(
            test_case_id=f"baseline_{uuid.uuid4().hex[:8]}",
            model_name=model_name,
            prompt=prompt,
            response=response,
            execution_time_seconds=execution_time,
            metadata={
                "baseline_type": baseline_config.baseline_type.value,
                "baseline_name": baseline_config.name,
                "has_context": context is not None,
                "has_examples": examples is not None,
                "error": error is not None
            }
        )

        if error:
            result.error = error

        return result

class BaselineComparisonSuite:
    """Suite for running baseline comparison experiments"""

    def __init__(self, experiment_framework, judge_model: str = "chutes:zai-org/GLM-4.6-FP8"):
        self.framework = experiment_framework
        self.baseline_engine = BaselineEngine(experiment_framework.llm_manager)
        self.judge = LLMJudge(experiment_framework.llm_manager, judge_model)
        self.statistical_analyzer = StatisticalAnalyzer()
        self.comparison_results: List[ComparisonResult] = []
        self.statistical_analyses: Dict[str, ABTestResult] = {}

    async def run_ab_test(self,
                         experiment_config: ExperimentConfig,
                         baseline_types: List[BaselineType],
                         test_cases: List[Dict[str, Any]],
                         models_to_test: List[str]) -> List[ComparisonResult]:
        """Run A/B test comparing Conjecture vs baselines"""

        print(f"🆚 Running A/B test: Conjecture vs {len(baseline_types)} baselines")
        print(f"Models: {', '.join(models_to_test)}")
        print(f"Test cases: {len(test_cases)}")

        comparison_results = []

        for model_name in models_to_test:
            print(f"\n🔬 Testing model: {model_name}")

            for i, test_case in enumerate(test_cases):
                print(f"  📝 Test case {i+1}/{len(test_cases)}")

                question = test_case.get('question', '')
                context = test_case.get('context', '')
                ground_truth = test_case.get('ground_truth', '')
                examples = test_case.get('examples', [])

                # Run Conjecture approach
                print(f"    🎯 Running Conjecture approach...")
                conjecture_result = await self._run_conjecture_approach(
                    test_case, model_name
                )

                # Evaluate Conjecture result
                conjecture_evaluations = await self.judge.evaluate_response(
                    question=question,
                    response=conjecture_result.response,
                    ground_truth=ground_truth,
                    context=context,
                    criteria=[EvaluationCriterion.CORRECTNESS, EvaluationCriterion.COMPLETENESS, 
                             EvaluationCriterion.COHERENCE, EvaluationCriterion.REASONING_QUALITY,
                             EvaluationCriterion.EFFICIENCY, EvaluationCriterion.CLARITY]
                )

                # Test each baseline
                for baseline_type in baseline_types:
                    print(f"    📊 Running baseline: {baseline_type.value}")
                    
                    baseline_config = self.baseline_engine.baseline_configs[baseline_type]
                    
                    # Prepare baseline-specific parameters
                    baseline_context = context
                    baseline_examples = examples if baseline_type == BaselineType.FEW_SHOT else None

                    # Run baseline
                    baseline_result = await self.baseline_engine.execute_baseline(
                        baseline_config=baseline_config,
                        question=question,
                        model_name=model_name,
                        context=baseline_context,
                        examples=baseline_examples
                    )

                    # Evaluate baseline result
                    baseline_evaluations = await self.judge.evaluate_response(
                        question=question,
                        response=baseline_result.response,
                        ground_truth=ground_truth,
                        context=context,
                        criteria=[EvaluationCriterion.CORRECTNESS, EvaluationCriterion.COMPLETENESS,
                                 EvaluationCriterion.COHERENCE, EvaluationCriterion.REASONING_QUALITY,
                                 EvaluationCriterion.EFFICIENCY, EvaluationCriterion.CLARITY]
                    )

                    # Compare results
                    comparison = await self._compare_results(
                        test_case_id=test_case.get('id', f'test_{i}'),
                        model_name=model_name,
                        conjecture_result=conjecture_result,
                        baseline_result=baseline_result,
                        conjecture_evaluations=conjecture_evaluations,
                        baseline_evaluations=baseline_evaluations,
                        baseline_type=baseline_type
                    )

                    comparison_results.append(comparison)

                    # Print immediate comparison summary
                    print(f"      🏆 Winner: {comparison.winner} (confidence: {comparison.confidence_in_winner:.2f})")

        self.comparison_results.extend(comparison_results)

        # Perform statistical analysis
        print("\n📊 Running statistical analysis...")
        self.statistical_analyses = self.statistical_analyzer.analyze_baseline_comparisons(
            comparison_results
        )

        return comparison_results

    async def _run_conjecture_approach(self, test_case: Dict[str, Any], model_name: str) -> TestResult:
        """Run Conjecture's claims-based approach"""
        from processing.llm_prompts.context_integrator import ContextIntegrator
        from processing.support_systems.context_builder import ContextBuilder
        
        question = test_case.get('question', '')
        context = test_case.get('context', '')

        start_time = time.time()

        try:
            # Use Conjecture's approach
            context_builder = ContextBuilder()
            context_integrator = ContextIntegrator()

            # Build context using Conjecture's context system
            built_context = await context_builder.build_context(question, context)
            integrated_prompt = await context_integrator.integrate_context(question, built_context)

            # Generate response using Conjecture's approach
            response = await self.framework.llm_manager.generate_response(
                prompt=integrated_prompt,
                model=model_name,
                max_tokens=2000,
                temperature=0.7
            )

            error = None

        except Exception as e:
            response = ""
            error = str(e)

        execution_time = time.time() - start_time

        result = TestResult(
            test_case_id=f"conjecture_{uuid.uuid4().hex[:8]}",
            model_name=model_name,
            prompt=integrated_prompt if not error else question,
            response=response,
            execution_time_seconds=execution_time,
            metadata={
                "approach": "conjecture",
                "has_context": context is not None,
                "error": error is not None
            }
        )

        if error:
            result.error = error

        return result

    async def _compare_results(self,
                             test_case_id: str,
                             model_name: str,
                             conjecture_result: TestResult,
                             baseline_result: TestResult,
                             conjecture_evaluations: Dict[EvaluationCriterion, Any],
                             baseline_evaluations: Dict[EvaluationCriterion, Any],
                             baseline_type: BaselineType) -> ComparisonResult:
        """Compare Conjecture and baseline results"""

        # Calculate performance comparison
        performance_comparison = {}
        
        for criterion in EvaluationCriterion:
            if criterion in conjecture_evaluations and criterion in baseline_evaluations:
                conj_score = self._extract_score(conjecture_evaluations[criterion])
                base_score = self._extract_score(baseline_evaluations[criterion])
                
                performance_comparison[criterion.value] = conj_score - base_score

        # Determine winner
        overall_conj_score = sum(self._extract_score(conjecture_evaluations.get(c, 0.5)) 
                                for c in EvaluationCriterion) / len(EvaluationCriterion)
        overall_base_score = sum(self._extract_score(baseline_evaluations.get(c, 0.5)) 
                                for c in EvaluationCriterion) / len(EvaluationCriterion)
        
        score_difference = overall_conj_score - overall_base_score
        
        if abs(score_difference) < 0.05:
            winner = "tie"
            confidence = 1.0 - abs(score_difference) * 20  # Lower confidence for ties
        elif score_difference > 0:
            winner = "conjecture"
            confidence = min(1.0, abs(score_difference) * 2)
        else:
            winner = "baseline"
            confidence = min(1.0, abs(score_difference) * 2)

        # Generate analysis
        analysis = await self._generate_comparison_analysis(
            conjecture_evaluations, baseline_evaluations, baseline_type
        )

        return ComparisonResult(
            test_case_id=test_case_id,
            model_name=model_name,
            conjecture_result=conjecture_result,
            baseline_result=baseline_result,
            conjecture_evaluations=conjecture_evaluations,
            baseline_evaluations=baseline_evaluations,
            performance_comparison=performance_comparison,
            winner=winner,
            confidence_in_winner=confidence,
            analysis=analysis
        )

    def _extract_score(self, evaluation_result) -> float:
        """Extract score from evaluation result"""
        if hasattr(evaluation_result, 'score'):
            return evaluation_result.score
        elif hasattr(evaluation_result, 'final_score'):
            return evaluation_result.final_score
        elif isinstance(evaluation_result, dict):
            return evaluation_result.get('score', evaluation_result.get('final_score', 0.5))
        else:
            return 0.5

    async def _generate_comparison_analysis(self,
                                          conj_evals: Dict[EvaluationCriterion, Any],
                                          base_evals: Dict[EvaluationCriterion, Any],
                                          baseline_type: BaselineType) -> str:
        """Generate detailed analysis of the comparison"""
        
        advantages = []
        disadvantages = []
        
        for criterion in EvaluationCriterion:
            if criterion in conj_evals and criterion in base_evals:
                conj_score = self._extract_score(conj_evals[criterion])
                base_score = self._extract_score(base_evals[criterion])
                
                if conj_score > base_score + 0.1:
                    advantages.append(f"Conjecture outperforms in {criterion.value} ({conj_score:.2f} vs {base_score:.2f})")
                elif base_score > conj_score + 0.1:
                    disadvantages.append(f"Baseline outperforms in {criterion.value} ({base_score:.2f} vs {conj_score:.2f})")

        analysis = f"Comparison against {baseline_type.value}:\n"
        
        if advantages:
            analysis += "Advantages: " + "; ".join(advantages) + ".\n"
        if disadvantages:
            analysis += "Disadvantages: " + "; ".join(disadvantages) + "."
        
        if not advantages and not disadvantages:
            analysis += "Performance is very similar across all criteria."

        return analysis

    async def generate_comparative_report(self, comparison_results: List[ComparisonResult]) -> str:
        """Generate comprehensive comparative analysis report"""
        
        if not comparison_results:
            return "No comparison results to analyze."

        # Calculate overall statistics
        total_comparisons = len(comparison_results)
        conjecture_wins = sum(1 for r in comparison_results if r.winner == "conjecture")
        baseline_wins = sum(1 for r in comparison_results if r.winner == "baseline")
        ties = sum(1 for r in comparison_results if r.winner == "tie")

        # Group by baseline type
        baseline_stats = {}
        for result in comparison_results:
            baseline_type = result.baseline_result.metadata.get('baseline_type', 'unknown')
            if baseline_type not in baseline_stats:
                baseline_stats[baseline_type] = {"conjecture": 0, "baseline": 0, "tie": 0}
            baseline_stats[baseline_type][result.winner] += 1

        # Group by model
        model_stats = {}
        for result in comparison_results:
            model = result.model_name
            if model not in model_stats:
                model_stats[model] = {"conjecture": 0, "baseline": 0, "tie": 0}
            model_stats[model][result.winner] += 1

        # Performance by criterion
        criterion_performance = {}
        for result in comparison_results:
            for criterion, diff in result.performance_comparison.items():
                if criterion not in criterion_performance:
                    criterion_performance[criterion] = []
                criterion_performance[criterion].append(diff)

        # Calculate average performance differences
        avg_performance = {}
        for criterion, diffs in criterion_performance.items():
            avg_performance[criterion] = sum(diffs) / len(diffs) if diffs else 0

        report = f"""# Conjecture vs Baseline Comparison Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

- **Total Comparisons**: {total_comparisons}
- **Conjecture Wins**: {conjecture_wins} ({conjecture_wins/total_comparisons*100:.1f}%)
- **Baseline Wins**: {baseline_wins} ({baseline_wins/total_comparisons*100:.1f}%)
- **Ties**: {ties} ({ties/total_comparisons*100:.1f}%)

## Performance by Baseline Type
"""

        for baseline_type, stats in baseline_stats.items():
            total = sum(stats.values())
            conj_pct = stats['conjecture'] / total * 100 if total > 0 else 0
            base_pct = stats['baseline'] / total * 100 if total > 0 else 0
            tie_pct = stats['tie'] / total * 100 if total > 0 else 0
            
            report += f"""
### {baseline_type.replace('_', ' ').title()}
- Conjecture: {stats['conjecture']}/{total} ({conj_pct:.1f}%)
- Baseline: {stats['baseline']}/{total} ({base_pct:.1f}%)
- Ties: {stats['tie']}/{total} ({tie_pct:.1f}%)
"""

        report += f"""
## Performance by Model
"""

        for model, stats in model_stats.items():
            total = sum(stats.values())
            conj_pct = stats['conjecture'] / total * 100 if total > 0 else 0
            base_pct = stats['baseline'] / total * 100 if total > 0 else 0
            
            report += f"""
### {model}
- Conjecture: {stats['conjecture']}/{total} ({conj_pct:.1f}%)
- Baseline: {stats['baseline']}/{total} ({base_pct:.1f}%)
- Ties: {stats['tie']}/{total} ({ties/total*100:.1f}%)
"""

        report += f"""
## Performance by Criterion
"""

        for criterion, avg_diff in avg_performance.items():
            if avg_diff > 0.05:
                report += f"- **{criterion.replace('_', ' ').title()}**: Conjecture +{avg_diff:.3f}\n"
            elif avg_diff < -0.05:
                report += f"- **{criterion.replace('_', ' ').title()}**: Baseline {avg_diff:.3f}\n"
            else:
                report += f"- **{criterion.replace('_', ' ').title()}**: Similar ({avg_diff:.3f})\n"

        report += f"""
## Key Findings

1. **Overall Effectiveness**: Conjecture {'outperforms' if conjecture_wins > baseline_wins else 'is matched by'} baselines in {max(conjecture_wins, baseline_wins)}/{total_comparisons} comparisons

2. **Strongest Baseline Competitors**: {max(baseline_stats.keys(), key=lambda k: baseline_stats[k]['baseline']) if baseline_stats else 'None'}

3. **Consistent Advantages**: {[c for c, diff in avg_performance.items() if diff > 0.1]}

4. **Areas for Improvement**: {[c for c, diff in avg_performance.items() if diff < -0.1]}

## Statistical Analysis

"""
        if self.statistical_analyses:
            # Add statistical test results
            for comparison, analysis in self.statistical_analyses.items():
                report += f"### {comparison.replace('_', ' ').title()}\n"
                report += f"- {analysis.recommendation}\n"
                report += f"- Mean difference: {analysis.mean_difference:+.3f}\n"

                significant_tests = [test for test in analysis.statistical_tests.values() if test.is_significant]
                if significant_tests:
                    report += f"- Significant tests: {', '.join(significant_tests)}\n"
                else:
                    report += "- No statistically significant differences found\n"

                report += "\n"

            # Add statistical summary
            statistical_summary = self.statistical_analyzer.generate_statistical_summary(self.statistical_analyses)
            report += statistical_summary
        else:
            report += "Statistical analysis not available or insufficient data.\n"

        report += f"""
## Recommendations

Based on this analysis:

"""
        # Check statistical significance in the analyses
        significant_evidence = False
        if self.statistical_analyses:
            for analysis in self.statistical_analyses.values():
                if any(test.is_significant for test in analysis.statistical_tests.values()):
                    if analysis.mean_difference > 0:
                        significant_evidence = True
                        break

        if significant_evidence:
            report += "- ✅ **Statistically significant evidence** supports Conjecture's effectiveness\n"
            report += "- 🚀 **Data-driven recommendation to use Conjecture** over standard approaches\n"
        elif conjecture_wins > baseline_wins * 1.2:
            report += "- ✅ **Conjecture shows clear advantage** over standard prompting approaches\n"
            report += "- 🚀 **Recommended for production use** based on superior performance\n"
        elif conjecture_wins > baseline_wins:
            report += "- ✅ **Conjecture shows modest advantage** over baselines\n"
            report += "- 🎯 **Consider use case specificity** when choosing approach\n"
        else:
            report += "- ⚠️ **Conjecture needs improvement** to compete with baselines\n"
            report += "- 🔧 **Focus on weak areas** identified in criterion analysis\n"

        return report

    def save_comparison_results(self, filepath: str):
        """Save comparison results to file"""
        results_data = []
        
        for result in self.comparison_results:
            result_dict = asdict(result)
            # Convert complex objects to serializable format
            result_dict['conjecture_evaluations'] = {
                k.value: asdict(v) if hasattr(v, '__dict__') else str(v) 
                for k, v in result.conjecture_evaluations.items()
            }
            result_dict['baseline_evaluations'] = {
                k.value: asdict(v) if hasattr(v, '__dict__') else str(v) 
                for k, v in result.baseline_evaluations.items()
            }
            results_data.append(result_dict)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

async def main():
    """Example usage of baseline comparison suite"""
    print("Basline comparison example")

if __name__ == "__main__":
    asyncio.run(main())