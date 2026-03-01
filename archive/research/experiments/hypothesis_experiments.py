#!/usr/bin/env python3
"""
Specific experiments to validate Conjecture's core hypotheses
"""

import asyncio
from typing import List, Dict, Any
from experiment_framework import (
    ExperimentFramework, ExperimentConfig, ExperimentType, EvaluationMetric
)

class HypothesisExperiments:
    """Collection of experiments to validate Conjecture's core hypotheses"""
    
    def __init__(self, framework: ExperimentFramework):
        self.framework = framework
    
    def create_task_decomposition_experiment(self) -> ExperimentConfig:
        """
        Hypothesis: By decomposing complex tasks into smaller claims, 
        small LLMs can achieve better reasoning performance.
        """
        return ExperimentConfig(
            experiment_id="task_decomp_001",
            experiment_type=ExperimentType.TASK_DECOMPOSITION,
            name="Task Decomposition Effectiveness",
            description="Tests if breaking down complex problems improves small LLM performance",
            hypothesis="Small LLMs will show 20%+ improvement in correctness when using task decomposition vs direct approach",
            models_to_test=[
                "lmstudio:ibm/granite-4-h-tiny",
                "lmstudio:GLM-Z1-9B-0414", 
                "chutes:GLM-4.5-Air",
                "chutes:GLM-4.6"
            ],
            test_cases=[
                "complex_reasoning_001",
                "planning_task_001", 
                "analysis_problem_001",
                "synthesis_task_001",
                "debugging_challenge_001"
            ],
            metrics=[
                EvaluationMetric.CORRECTNESS,
                EvaluationMetric.COMPLETENESS,
                EvaluationMetric.COHERENCE
            ],
            parameters={
                "max_tokens": 2000,
                "temperature": 0.7,
                "enable_decomposition": True,
                "comparison_baseline": True
            }
        )
    
    def create_context_compression_experiment(self) -> ExperimentConfig:
        """
        Hypothesis: By compressing context using claims-based format,
        models can maintain performance with significantly reduced context.
        """
        return ExperimentConfig(
            experiment_id="context_comp_001",
            experiment_type=ExperimentType.CONTEXT_COMPRESSION,
            name="Context Compression Efficiency",
            description="Tests if claims-based context compression maintains performance while reducing tokens",
            hypothesis="Models will maintain 90%+ performance with 50%+ context reduction using claims format",
            models_to_test=[
                "lmstudio:ibm/granite-4-h-tiny",
                "lmstudio:GLM-Z1-9B-0414",
                "chutes:GLM-4.5-Air", 
                "chutes:GLM-4.6"
            ],
            test_cases=[
                "long_context_qa_001",
                "document_analysis_001",
                "research_synthesis_001",
                "case_study_reasoning_001",
                "multi_source_analysis_001"
            ],
            metrics=[
                EvaluationMetric.CORRECTNESS,
                EvaluationMetric.EFFICIENCY,
                EvaluationMetric.COMPLETENESS
            ],
            parameters={
                "max_tokens": 1500,
                "compression_ratio_target": 0.5,
                "preserve_key_claims": True,
                "baseline_full_context": True
            }
        )
    
    def create_model_comparison_experiment(self) -> ExperimentConfig:
        """
        Hypothesis: Small models with Conjecture's prompting approach
        can compete with larger models using standard prompting.
        """
        return ExperimentConfig(
            experiment_id="model_comp_001",
            experiment_type=ExperimentType.MODEL_COMPARISON,
            name="Size vs Prompting Strategy",
            description="Compares small models with Conjecture prompting vs larger models with standard prompting",
            hypothesis="Small models (3-9B) with Conjecture prompting will match/exceed larger models (30B+) on reasoning tasks",
            models_to_test=[
                "lmstudio:ibm/granite-4-h-tiny",  # ~3B
                "lmstudio:GLM-Z1-9B-0414",       # ~9B
                "chutes:GLM-4.5-Air",            # ~10B
                "chutes:GLM-4.6"                 # ~10B
            ],
            test_cases=[
                "logical_reasoning_001",
                "mathematical_problem_001",
                "causal_inference_001",
                "analogical_reasoning_001",
                "ethical_reasoning_001"
            ],
            metrics=[
                EvaluationMetric.CORRECTNESS,
                EvaluationMetric.COHERENCE,
                EvaluationMetric.CONFIDENCE_CALIBRATION
            ],
            parameters={
                "max_tokens": 2000,
                "temperature": 0.3,
                "conjecture_prompting": True,
                "standard_prompting_baseline": True
            }
        )
    
    def create_claims_reasoning_experiment(self) -> ExperimentConfig:
        """
        Hypothesis: Claims-based reasoning with confidence scores
        improves reasoning transparency and accuracy.
        """
        return ExperimentConfig(
            experiment_id="claims_reason_001",
            experiment_type=ExperimentType.CLAIMS_REASONING,
            name="Claims-Based Reasoning",
            description="Tests if explicit claim representation with confidence scores improves reasoning",
            hypothesis="Claims-based reasoning will show 15%+ improvement in correctness and confidence calibration",
            models_to_test=[
                "lmstudio:ibm/granite-4-h-tiny",
                "lmstudio:GLM-Z1-9B-0414",
                "chutes:GLM-4.5-Air",
                "chutes:GLM-4.6"
            ],
            test_cases=[
                "evidence_evaluation_001",
                "argument_analysis_001",
                "hypothesis_testing_001",
                "decision_making_001",
                "scientific_reasoning_001"
            ],
            metrics=[
                EvaluationMetric.CORRECTNESS,
                EvaluationMetric.CONFIDENCE_CALIBRATION,
                EvaluationMetric.COHERENCE
            ],
            parameters={
                "max_tokens": 2000,
                "explicit_confidence": True,
                "claim_relationships": True,
                "baseline_standard_reasoning": True
            }
        )
    
    def create_end_to_end_experiment(self) -> ExperimentConfig:
        """
        Hypothesis: Full Conjecture pipeline (decomposition + compression + claims reasoning)
        provides the best overall performance for small models.
        """
        return ExperimentConfig(
            experiment_id="end_to_end_001",
            experiment_type=ExperimentType.END_TO_END,
            name="Full Conjecture Pipeline",
            description="Tests the complete Conjecture approach against baseline methods",
            hypothesis="Full pipeline will show 25%+ improvement over baseline for small models on complex tasks",
            models_to_test=[
                "lmstudio:ibm/granite-4-h-tiny",
                "lmstudio:GLM-Z1-9B-0414",
                "chutes:GLM-4.5-Air",
                "chutes:GLM-4.6"
            ],
            test_cases=[
                "complex_research_task_001",
                "multi_step_analysis_001",
                "comprehensive_evaluation_001",
                "strategic_planning_001",
                "integrated_problem_solving_001"
            ],
            metrics=[
                EvaluationMetric.CORRECTNESS,
                EvaluationMetric.COMPLETENESS,
                EvaluationMetric.COHERENCE,
                EvaluationMetric.CONFIDENCE_CALIBRATION
            ],
            parameters={
                "max_tokens": 3000,
                "full_pipeline": True,
                "baseline_comparison": True,
                "intermediate_steps_analysis": True
            }
        )
    
    async def run_all_hypothesis_tests(self) -> Dict[str, Any]:
        """Run all hypothesis validation experiments"""
        experiments = [
            self.create_task_decomposition_experiment(),
            self.create_context_compression_experiment(),
            self.create_model_comparison_experiment(),
            self.create_claims_reasoning_experiment(),
            self.create_end_to_end_experiment()
        ]
        
        results = {}
        
        for experiment in experiments:
            print(f"\n🧪 Running experiment: {experiment.name}")
            print(f"Hypothesis: {experiment.hypothesis}")
            
            try:
                run = await self.framework.run_experiment(experiment)
                results[experiment.experiment_id] = {
                    'status': 'completed',
                    'run_id': run.run_id,
                    'summary': run.summary
                }
                print(f"✅ Experiment completed: {run.run_id}")
                
            except Exception as e:
                results[experiment.experiment_id] = {
                    'status': 'failed',
                    'error': str(e)
                }
                print(f"❌ Experiment failed: {e}")
        
        return results
    
    async def generate_hypothesis_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive report on hypothesis validation"""
        report = []
        report.append("# Conjecture Hypothesis Validation Report")
        report.append(f"Generated: {asyncio.get_event_loop().time()}")
        report.append("")
        
        total_experiments = len(results)
        successful_experiments = len([r for r in results.values() if r['status'] == 'completed'])
        
        report.append(f"## Summary")
        report.append(f"- Total Experiments: {total_experiments}")
        report.append(f"- Successful: {successful_experiments}")
        report.append(f"- Failed: {total_experiments - successful_experiments}")
        report.append("")
        
        # Individual experiment results
        report.append("## Individual Experiment Results")
        report.append("")
        
        for exp_id, result in results.items():
            report.append(f"### {exp_id}")
            report.append(f"Status: {result['status']}")
            
            if result['status'] == 'completed':
                summary = result['summary']
                report.append(f"Hypothesis: {summary.get('hypothesis', 'N/A')}")
                
                # Performance metrics
                if 'evaluation_metrics' in summary:
                    report.append("#### Evaluation Metrics:")
                    for metric, values in summary['evaluation_metrics'].items():
                        report.append(f"- {metric}: {values['avg_score']:.3f} avg score")
                
                # Model performance
                if 'model_performance' in summary:
                    report.append("#### Model Performance:")
                    for model, stats in summary['model_performance'].items():
                        report.append(f"- {model}: {stats['success_rate']:.2%} success rate")
                
            else:
                report.append(f"Error: {result.get('error', 'Unknown error')}")
            
            report.append("")
        
        # Overall conclusions
        report.append("## Overall Conclusions")
        
        if successful_experiments > 0:
            report.append("### Key Findings:")
            
            # Analyze which hypotheses were supported
            for exp_id, result in results.items():
                if result['status'] == 'completed':
                    summary = result['summary']
                    # Simple heuristic: if avg score > 0.7, consider hypothesis supported
                    if 'evaluation_metrics' in summary:
                        avg_scores = [v['avg_score'] for v in summary['evaluation_metrics'].values()]
                        overall_avg = sum(avg_scores) / len(avg_scores)
                        
                        if overall_avg > 0.7:
                            report.append(f"- ✅ {exp_id}: Hypothesis appears supported (avg score: {overall_avg:.3f})")
                        else:
                            report.append(f"- ⚠️ {exp_id}: Hypothesis needs refinement (avg score: {overall_avg:.3f})")
        
        report.append("")
        report.append("### Recommendations:")
        report.append("1. Focus on the most successful approaches for further development")
        report.append("2. Refine experiments that showed mixed results")
        report.append("3. Expand test cases for validated hypotheses")
        report.append("4. Investigate failure modes in unsuccessful experiments")
        
        return "\n".join(report)

async def main():
    """Main function to run hypothesis validation experiments"""
    from config.common import ProviderConfig
    
    # Setup provider configurations
    providers = [
        ProviderConfig(
            url="http://localhost:1234",  # LM Studio
            api_key="",
            model="ibm/granite-4-h-tiny"
        ),
        ProviderConfig(
            url="http://localhost:1234",  # LM Studio
            api_key="",
            model="GLM-Z1-9B-0414"
        ),
        ProviderConfig(
            url="https://llm.chutes.ai/v1",  # Chutes
            api_key="your-api-key",
            model="GLM-4.5-Air"
        ),
        ProviderConfig(
            url="https://llm.chutes.ai/v1",  # Chutes
            api_key="your-api-key", 
            model="GLM-4.6"
        )
    ]
    
    # Initialize framework
    framework = ExperimentFramework()
    await framework.initialize(providers)
    
    # Run experiments
    hypothesis_experiments = HypothesisExperiments(framework)
    results = await hypothesis_experiments.run_all_hypothesis_tests()
    
    # Generate report
    report = await hypothesis_experiments.generate_hypothesis_report(results)
    
    # Save report
    with open("research/analysis/hypothesis_validation_report.md", "w") as f:
        f.write(report)
    
    print("\n" + "="*50)
    print("HYPOTHESIS VALIDATION COMPLETE")
    print("="*50)
    print(report)

if __name__ == "__main__":
    asyncio.run(main())