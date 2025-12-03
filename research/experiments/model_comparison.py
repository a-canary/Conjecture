#!/usr/bin/env python3
"""
Model Comparison Experiments for Conjecture
Compares performance across different models with and without Conjecture's approach
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from experiment_framework import ExperimentFramework, ExperimentConfig, ExperimentType, EvaluationMetric


@dataclass
class ModelCapability:
    """Defines model capabilities for comparison"""
    name: str
    provider: str
    size_category: str  # "tiny", "small", "medium", "large"
    expected_strengths: List[str]
    expected_weaknesses: List[str]
    context_window: int
    cost_per_token: float


class ModelComparisonSuite:
    """Comprehensive model comparison experiments"""
    
    def __init__(self, framework: ExperimentFramework):
        self.framework = framework
        self.models = self._define_models()
    
    def _define_models(self) -> Dict[str, ModelCapability]:
        """Define the models to compare"""
        return {
            "lmstudio:ibm/granite-4-h-tiny": ModelCapability(
                name="IBM Granite-4-H-Tiny",
                provider="LM Studio",
                size_category="tiny",
                expected_strengths=["efficiency", "structured reasoning"],
                expected_weaknesses=["complex reasoning", "nuance"],
                context_window=2048,
                cost_per_token=0.0
            ),
            "lmstudio:GLM-Z1-9B-0414": ModelCapability(
                name="GLM-Z1-9B",
                provider="LM Studio", 
                size_category="small",
                expected_strengths=["reasoning", "multilingual"],
                expected_weaknesses["consistency", "depth"],
                context_window=4096,
                cost_per_token=0.0
            ),
            "chutes:GLM-4.5-Air": ModelCapability(
                name="GLM-4.5-Air",
                provider="Chutes",
                size_category="medium",
                expected_strengths=["reasoning", "following instructions"],
                expected_weaknesses["creativity", "complex synthesis"],
                context_window=8192,
                cost_per_token=0.0001
            ),
            "chutes:GLM-4.6": ModelCapability(
                name="GLM-4.6",
                provider="Chutes",
                size_category="medium",
                expected_strengths=["reasoning", "accuracy", "consistency"],
                expected_weaknesses["speed", "cost"],
                context_window=8192,
                cost_per_token=0.0002
            )
        }
    
    def create_baseline_vs_conjecture_experiment(self) -> ExperimentConfig:
        """
        Compare standard prompting vs Conjecture's claims-based approach
        across all models
        """
        return ExperimentConfig(
            experiment_id="baseline_vs_conjecture_001",
            experiment_type=ExperimentType.MODEL_COMPARISON,
            name="Baseline vs Conjecture Prompting",
            description="Compares standard prompting against Conjecture's claims-based approach",
            hypothesis="Conjecture prompting will improve small model performance by 20%+ while maintaining large model performance",
            models_to_test=list(self.models.keys()),
            test_cases=[
                "logical_deduction_001",
                "mathematical_reasoning_001", 
                "causal_analysis_001",
                "ethical_reasoning_001",
                "creative_problem_solving_001",
                "technical_troubleshooting_001",
                "research_synthesis_001",
                "argument_evaluation_001"
            ],
            metrics=[
                EvaluationMetric.CORRECTNESS,
                EvaluationMetric.COMPLETENESS,
                EvaluationMetric.COHERENCE,
                EvaluationMetric.CONFIDENCE_CALIBRATION
            ],
            parameters={
                "max_tokens": 2000,
                "temperature": 0.3,
                "compare_approaches": ["standard", "conjecture"],
                "randomize_order": True
            }
        )
    
    def create_size_vs_approach_experiment(self) -> ExperimentConfig:
        """
        Test if approach matters more than model size
        """
        return ExperimentConfig(
            experiment_id="size_vs_approach_001",
            experiment_type=ExperimentType.MODEL_COMPARISON,
            name="Model Size vs Prompting Approach",
            description="Tests if prompting approach can overcome model size limitations",
            hypothesis="Small models with Conjecture prompting will outperform large models with standard prompting",
            models_to_test=list(self.models.keys()),
            test_cases=[
                "complex_reasoning_001",
                "multi_step_analysis_001",
                "abstract_reasoning_001",
                "system_thinking_001",
                "strategic_planning_001"
            ],
            metrics=[
                EvaluationMetric.CORRECTNESS,
                EvaluationMetric.COHERENCE,
                EvaluationMetric.EFFICIENCY
            ],
            parameters={
                "max_tokens": 2500,
                "temperature": 0.2,
                "focus_on_reasoning_quality": True,
                "track_decomposition_quality": True
            }
        )
    
    def create_context_efficiency_experiment(self) -> ExperimentConfig:
        """
        Test how efficiently different models use context
        """
        return ExperimentConfig(
            experiment_id="context_efficiency_001",
            experiment_type=ExperimentType.CONTEXT_COMPRESSION,
            name="Context Usage Efficiency",
            description="Tests how efficiently models use compressed vs full context",
            hypothesis="Smaller models will benefit more from context compression than larger models",
            models_to_test=list(self.models.keys()),
            test_cases=[
                "long_document_qa_001",
                "multi_source_synthesis_001",
                "research_paper_analysis_001",
                "case_study_reasoning_001",
                "historical_analysis_001"
            ],
            metrics=[
                EvaluationMetric.CORRECTNESS,
                EvaluationMetric.EFFICIENCY,
                EvaluationMetric.COMPLETENESS
            ],
            parameters={
                "max_tokens": 2000,
                "context_sizes": ["full", "compressed_75", "compressed_50", "compressed_25"],
                "measure_token_efficiency": True,
                "track_compression_impact": True
            }
        )
    
    def create_reasoning_trajectory_experiment(self) -> ExperimentConfig:
        """
        Analyze how different models approach reasoning problems
        """
        return ExperimentConfig(
            experiment_id="reasoning_trajectory_001",
            experiment_type=ExperimentType.CLAIMS_REASONING,
            name="Reasoning Trajectory Analysis",
            description="Analyzes step-by-step reasoning patterns across models",
            hypothesis="Models will show distinct reasoning patterns that correlate with their capabilities",
            models_to_test=list(self.models.keys()),
            test_cases=[
                "step_by_step_deduction_001",
                "hypothesis_testing_001",
                "evidence_evaluation_001",
                "counterfactual_reasoning_001",
                "analogical_transfer_001"
            ],
            metrics=[
                EvaluationMetric.CORRECTNESS,
                EvaluationMetric.COHERENCE,
                EvaluationMetric.CONFIDENCE_CALIBRATION
            ],
            parameters={
                "max_tokens": 3000,
                "require_step_by_step": True,
                "track_confidence_evolution": True,
                "analyze_reasoning_patterns": True
            }
        )
    
    async def run_comprehensive_comparison(self) -> Dict[str, Any]:
        """Run all model comparison experiments"""
        experiments = [
            self.create_baseline_vs_conjecture_experiment(),
            self.create_size_vs_approach_experiment(),
            self.create_context_efficiency_experiment(),
            self.create_reasoning_trajectory_experiment()
        ]
        
        results = {}
        
        for experiment in experiments:
            print(f"\n🔬 Running model comparison: {experiment.name}")
            print(f"Models: {len(experiment.models_to_test)}")
            print(f"Test cases: {len(experiment.test_cases)}")
            
            try:
                run = await self.framework.run_experiment(experiment)
                results[experiment.experiment_id] = {
                    'status': 'completed',
                    'run_id': run.run_id,
                    'summary': run.summary,
                    'detailed_analysis': await self._analyze_model_performance(run)
                }
                print(f"✅ Comparison completed: {run.run_id}")
                
            except Exception as e:
                results[experiment.experiment_id] = {
                    'status': 'failed',
                    'error': str(e)
                }
                print(f"❌ Comparison failed: {e}")
        
        return results
    
    async def _analyze_model_performance(self, run) -> Dict[str, Any]:
        """Analyze detailed model performance from experiment run"""
        analysis = {
            'model_rankings': {},
            'approach_effectiveness': {},
            'size_vs_performance': {},
            'capability_insights': {}
        }
        
        # Group results by model
        model_results = {}
        for result in run.test_results:
            if result.model_name not in model_results:
                model_results[result.model_name] = []
            model_results[result.model_name].append(result)
        
        # Calculate model rankings
        for model_name, results in model_results.items():
            # Get evaluation scores for this model
            model_evaluations = [e for e in run.evaluation_results 
                               if e.test_result_id.startswith(f"_{model_name}")]
            
            if model_evaluations:
                avg_scores = {}
                for metric in EvaluationMetric:
                    metric_scores = [e.score for e in model_evaluations if e.metric == metric]
                    if metric_scores:
                        avg_scores[metric.value] = sum(metric_scores) / len(metric_scores)
                
                analysis['model_rankings'][model_name] = {
                    'average_score': sum(avg_scores.values()) / len(avg_scores),
                    'metric_breakdown': avg_scores,
                    'model_info': self.models.get(model_name).__dict__ if model_name in self.models else {}
                }
        
        # Analyze approach effectiveness (if applicable)
        if 'standard' in str(run.experiment_config.parameters):
            standard_results = [r for r in run.test_results if 'standard' in str(r.metadata)]
            conjecture_results = [r for r in run.test_results if 'conjecture' in str(r.metadata)]
            
            if standard_results and conjecture_results:
                analysis['approach_effectiveness'] = {
                    'standard_avg': self._calculate_avg_score(standard_results, run.evaluation_results),
                    'conjecture_avg': self._calculate_avg_score(conjecture_results, run.evaluation_results),
                    'improvement_percentage': self._calculate_improvement(
                        standard_results, conjecture_results, run.evaluation_results
                    )
                }
        
        # Size vs performance analysis
        size_performance = {}
        for model_name, ranking in analysis['model_rankings'].items():
            if model_name in self.models:
                size_cat = self.models[model_name].size_category
                if size_cat not in size_performance:
                    size_performance[size_cat] = []
                size_performance[size_cat].append(ranking['average_score'])
        
        for size_cat, scores in size_performance.items():
            analysis['size_vs_performance'][size_cat] = {
                'average_score': sum(scores) / len(scores),
                'model_count': len(scores),
                'score_range': [min(scores), max(scores)]
            }
        
        return analysis
    
    def _calculate_avg_score(self, test_results: List, evaluation_results: List) -> float:
        """Calculate average score for a set of test results"""
        scores = []
        for test_result in test_results:
            test_evaluations = [e for e in evaluation_results 
                              if e.test_result_id == f"{test_result.test_case_id}_{test_result.model_name}"]
            if test_evaluations:
                avg_score = sum(e.score for e in test_evaluations) / len(test_evaluations)
                scores.append(avg_score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _calculate_improvement(self, baseline_results: List, improved_results: List, 
                             evaluation_results: List) -> float:
        """Calculate percentage improvement between baseline and improved results"""
        baseline_avg = self._calculate_avg_score(baseline_results, evaluation_results)
        improved_avg = self._calculate_avg_score(improved_results, evaluation_results)
        
        if baseline_avg == 0:
            return 0.0
        
        return ((improved_avg - baseline_avg) / baseline_avg) * 100
    
    async def generate_model_comparison_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive model comparison report"""
        report = []
        report.append("# Conjecture Model Comparison Report")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive summary
        report.append("## Executive Summary")
        successful_experiments = [r for r in results.values() if r['status'] == 'completed']
        report.append(f"- Experiments completed: {len(successful_experiments)}/{len(results)}")
        
        if successful_experiments:
            # Find best performing model overall
            all_rankings = {}
            for exp_result in successful_experiments:
                if 'detailed_analysis' in exp_result:
                    rankings = exp_result['detailed_analysis'].get('model_rankings', {})
                    for model, ranking in rankings.items():
                        if model not in all_rankings:
                            all_rankings[model] = []
                        all_rankings[model].append(ranking['average_score'])
            
            if all_rankings:
                best_model = max(all_rankings.keys(), 
                               key=lambda m: sum(all_rankings[m]) / len(all_rankings[m]))
                best_score = sum(all_rankings[best_model]) / len(all_rankings[best_model])
                report.append(f"- Best performing model: {best_model} ({best_score:.3f} avg)")
        
        report.append("")
        
        # Model capabilities overview
        report.append("## Model Capabilities Overview")
        for model_id, capability in self.models.items():
            report.append(f"### {capability.name}")
            report.append(f"- Size Category: {capability.size_category}")
            report.append(f"- Provider: {capability.provider}")
            report.append(f"- Context Window: {capability.context_window}")
            report.append(f"- Strengths: {', '.join(capability.expected_strengths)}")
            report.append(f"- Weaknesses: {', '.join(capability.expected_weaknesses)}")
            report.append("")
        
        # Individual experiment results
        report.append("## Experiment Results")
        for exp_id, result in results.items():
            report.append(f"### {exp_id}")
            report.append(f"Status: {result['status']}")
            
            if result['status'] == 'completed':
                summary = result['summary']
                detailed = result.get('detailed_analysis', {})
                
                # Model rankings
                if 'model_rankings' in detailed:
                    report.append("#### Model Performance Rankings:")
                    sorted_models = sorted(detailed['model_rankings'].items(), 
                                          key=lambda x: x[1]['average_score'], reverse=True)
                    for i, (model, ranking) in enumerate(sorted_models, 1):
                        report.append(f"{i}. {model}: {ranking['average_score']:.3f}")
                
                # Approach effectiveness
                if 'approach_effectiveness' in detailed:
                    effectiveness = detailed['approach_effectiveness']
                    report.append("#### Approach Effectiveness:")
                    report.append(f"- Standard: {effectiveness['standard_avg']:.3f}")
                    report.append(f"- Conjecture: {effectiveness['conjecture_avg']:.3f}")
                    report.append(f"- Improvement: {effectiveness['improvement_percentage']:.1f}%")
                
                # Size vs performance
                if 'size_vs_performance' in detailed:
                    report.append("#### Size vs Performance:")
                    for size_cat, perf in detailed['size_vs_performance'].items():
                        report.append(f"- {size_cat}: {perf['average_score']:.3f} avg ({perf['model_count']} models)")
            
            report.append("")
        
        # Key insights
        report.append("## Key Insights")
        
        if successful_experiments:
            insights = []
            
            # Analyze approach effectiveness
            approach_improvements = []
            for exp_result in successful_experiments:
                detailed = exp_result.get('detailed_analysis', {})
                if 'approach_effectiveness' in detailed:
                    improvement = detailed['approach_effectiveness']['improvement_percentage']
                    approach_improvements.append(improvement)
            
            if approach_improvements:
                avg_improvement = sum(approach_improvements) / len(approach_improvements)
                insights.append(f"Conjecture approach shows {avg_improvement:.1f}% average improvement")
            
            # Analyze size vs performance
            size_performance = {}
            for exp_result in successful_experiments:
                detailed = exp_result.get('detailed_analysis', {})
                if 'size_vs_performance' in detailed:
                    for size_cat, perf in detailed['size_vs_performance'].items():
                        if size_cat not in size_performance:
                            size_performance[size_cat] = []
                        size_performance[size_cat].append(perf['average_score'])
            
            if size_performance:
                best_size = max(size_performance.keys(), 
                              key=lambda s: sum(size_performance[s]) / len(size_performance[s]))
                insights.append(f"Best size category: {best_size}")
            
            for insight in insights:
                report.append(f"- {insight}")
        
        report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("1. **Model Selection**: Choose models based on specific task requirements")
        report.append("2. **Approach Adoption**: Use Conjecture prompting for complex reasoning tasks")
        report.append("3. **Context Optimization**: Implement context compression for smaller models")
        report.append("4. **Cost Efficiency**: Balance model size with prompting approach for optimal cost/performance")
        
        return "\n".join(report)


async def main():
    """Main function to run model comparison experiments"""
    from config.common import ProviderConfig
    
    # Setup providers
    providers = [
        ProviderConfig(
            url="http://localhost:1234",
            api_key="",
            model="ibm/granite-4-h-tiny"
        ),
        ProviderConfig(
            url="http://localhost:1234", 
            api_key="",
            model="GLM-Z1-9B-0414"
        ),
        ProviderConfig(
            url="https://llm.chutes.ai/v1",
            api_key="your-api-key",
            model="GLM-4.5-Air"
        ),
        ProviderConfig(
            url="https://llm.chutes.ai/v1",
            api_key="your-api-key",
            model="GLM-4.6"
        )
    ]
    
    # Initialize framework
    framework = ExperimentFramework()
    await framework.initialize(providers)
    
    # Run comparisons
    comparison_suite = ModelComparisonSuite(framework)
    results = await comparison_suite.run_comprehensive_comparison()
    
    # Generate report
    report = await comparison_suite.generate_model_comparison_report(results)
    
    # Save report
    with open("research/analysis/model_comparison_report.md", "w") as f:
        f.write(report)
    
    print("\n" + "="*50)
    print("MODEL COMPARISON COMPLETE")
    print("="*50)
    print(report)


if __name__ == "__main__":
    asyncio.run(main())