#!/usr/bin/env python3
"""
End-to-End Pipeline Experiment Runner
Tests the hypothesis: "Full pipeline shows 25%+ improvement over baseline for tiny models on complex tasks"

This is the fifth and final critical experiment for validating the core hypothesis that:
"By decomposing tasks and concepts, and providing relevant context through claims-based representations 
that include in-context learning examples of task breakdown strategies, research-plan-work-validate phases, 
scientific method, critical thinking, and fact-checking best practices, small LLMs can achieve 
performance comparable to larger models on complex reasoning tasks."

This experiment compares:
- Approach A (Baseline): Direct prompting without any Conjecture methods
- Approach B (Full Pipeline): Complete Conjecture pipeline with all optimizations

Uses LLM-as-a-Judge with GLM-4.6 for standardized evaluation across multiple criteria.
"""

import asyncio
import json
import time
import uuid
import statistics
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
import sys
import os
from scipy import stats
import pandas as pd

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.models import Claim, ClaimState, ClaimType
from processing.llm.llm_manager import LLMManager
from config.common import ProviderConfig
from conjecture import Conjecture

# Add research to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "research"))
from statistical_analyzer import ConjectureStatisticalAnalyzer

# Add tests to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from test_llm_judge import LLMJudgeSystem, JudgeConfiguration, EvaluationResult


@dataclass
class ExperimentConfig:
    """Configuration for end-to-end pipeline experiment"""
    
    # Test parameters
    sample_size: int = 75  # Target 50-100 test cases
    target_improvement: float = 0.25  # 25% improvement target
    alpha_level: float = 0.05  # Statistical significance
    power_target: float = 0.8  # Statistical power
    
    # Model configurations
    tiny_model: str = "ibm/granite-4-h-tiny"
    judge_model: str = "zai-org/GLM-4.6"
    
    # Pipeline configurations
    baseline_approach: str = "direct"  # Direct prompting
    full_pipeline_approach: str = "conjecture"  # Complete pipeline
    
    # Evaluation criteria
    evaluation_criteria: List[str] = None
    criterion_weights: Dict[str, float] = None
    
    # Output settings
    output_dir: str = "experiments/results"
    save_intermediate_results: bool = True
    
    def __post_init__(self):
        if self.evaluation_criteria is None:
            self.evaluation_criteria = [
                "correctness", "completeness", "reasoning_quality", 
                "coherence", "efficiency", "hallucination_reduction"
            ]
        
        if self.criterion_weights is None:
            self.criterion_weights = {
                "correctness": 0.25,
                "completeness": 0.20,
                "reasoning_quality": 0.20,
                "coherence": 0.15,
                "efficiency": 0.10,
                "hallucination_reduction": 0.10
            }


@dataclass
class PipelineStageMetrics:
    """Metrics for individual pipeline stages"""
    
    stage_name: str
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    tokens_processed: int = 0
    cache_hits: int = 0
    quality_score: float = 0.0


@dataclass
class ExperimentResult:
    """Results from a single test case execution"""
    
    test_case_id: str
    approach: str  # "baseline" or "full_pipeline"
    response: str
    execution_time: float
    token_usage: int
    pipeline_stages: List[PipelineStageMetrics]
    evaluation_result: Optional[EvaluationResult] = None
    error_message: Optional[str] = None


@dataclass
class ExperimentSummary:
    """Summary statistics for the experiment"""
    
    experiment_id: str
    hypothesis: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_test_cases: int = 0
    successful_baseline: int = 0
    successful_full_pipeline: int = 0
    baseline_results: List[ExperimentResult] = None
    full_pipeline_results: List[ExperimentResult] = None
    statistical_analysis: Optional[Dict[str, Any]] = None
    hypothesis_validated: bool = False
    improvement_percentage: float = 0.0
    
    def __post_init__(self):
        if self.baseline_results is None:
            self.baseline_results = []
        if self.full_pipeline_results is None:
            self.full_pipeline_results = []


class EndToEndPipelineExperiment:
    """Main experiment class for end-to-end pipeline validation"""
    
    def __init__(self, config: ExperimentConfig = None):
        self.config = config or ExperimentConfig()
        self.experiment_id = f"end_to_end_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger = logging.getLogger(__name__)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Initialize components
        self.llm_manager = None
        self.conjecture = None
        self.judge_system = None
        self.statistical_analyzer = None
        
        # Data storage
        self.test_cases = []
        self.results = ExperimentSummary(
            experiment_id=self.experiment_id,
            hypothesis="Full pipeline shows 25%+ improvement over baseline for tiny models on complex tasks",
            start_time=datetime.now()
        )
        
        # Directory setup
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_cases_dir = Path("research/test_cases")
        
    async def initialize(self) -> bool:
        """Initialize experiment components"""
        try:
            self.logger.info(f"Initializing End-to-End Pipeline Experiment: {self.experiment_id}")
            
            # Setup LLM manager
            await self._setup_llm_manager()
            
            # Setup Conjecture system
            await self._setup_conjecture()
            
            # Setup judge system
            await self._setup_judge_system()
            
            # Setup statistical analyzer
            self.statistical_analyzer = ConjectureStatisticalAnalyzer()
            
            # Load test cases
            await self._load_test_cases()
            
            self.logger.info("End-to-End Pipeline Experiment initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize experiment: {e}")
            return False
    
    async def _setup_llm_manager(self):
        """Setup LLM manager with provider configurations"""
        self.llm_manager = LLMManager()
        
        # Setup tiny model provider
        tiny_provider = ProviderConfig(
            url="http://localhost:11434",  # Ollama for local model
            api="",
            model=self.config.tiny_model,
            name="ollama_tiny"
        )
        
        # Setup judge model provider
        judge_provider = ProviderConfig(
            url="https://llm.chutes.ai/v1",
            api=os.getenv("CHUTES_API_KEY", ""),
            model=self.config.judge_model,
            name="chutes_judge"
        )
        
        # Add providers
        await self.llm_manager.add_provider(tiny_provider)
        await self.llm_manager.add_provider(judge_provider)
        
        # Test connections
        tiny_test = await self.llm_manager.test_connection(tiny_provider)
        judge_test = await self.llm_manager.test_connection(judge_provider)
        
        if not tiny_test.success:
            self.logger.error(f"Failed to connect to tiny model: {tiny_test.error}")
            raise ConnectionError(f"Tiny model connection failed: {tiny_test.error}")
        
        if not judge_test.success:
            self.logger.error(f"Failed to connect to judge model: {judge_test.error}")
            raise ConnectionError(f"Judge model connection failed: {judge_test.error}")
        
        self.logger.info("LLM manager setup complete")
    
    async def _setup_conjecture(self):
        """Setup Conjecture system for full pipeline approach"""
        from config.config import Config
        
        # Create config with tiny model
        config = Config()
        config.providers = [
            {
                "url": "http://localhost:11434",
                "api": "",
                "model": self.config.tiny_model,
                "name": "ollama_tiny"
            }
        ]
        
        self.conjecture = Conjecture(config)
        await self.conjecture.start_services()
        
        self.logger.info("Conjecture system setup complete")
    
    async def _setup_judge_system(self):
        """Setup LLM-as-a-Judge evaluation system"""
        judge_config = JudgeConfiguration(
            judge_model=self.config.judge_model,
            evaluation_criteria=self.config.evaluation_criteria,
            criterion_weights=self.config.criterion_weights
        )
        
        # Create judge system with its own LLM manager
        judge_llm_manager = LLMManager()
        judge_provider = ProviderConfig(
            url="https://llm.chutes.ai/v1",
            api=os.getenv("CHUTES_API_KEY", ""),
            model=self.config.judge_model,
            name="chutes_judge"
        )
        await judge_llm_manager.add_provider(judge_provider)
        
        self.judge_system = LLMJudgeSystem(judge_config, judge_llm_manager)
        
        self.logger.info("Judge system setup complete")
    
    async def _load_test_cases(self):
        """Load comprehensive test cases from research/test_cases/ directory"""
        self.logger.info("Loading comprehensive test cases for end-to-end pipeline experiment")
        
        # Load existing test cases
        test_case_files = list(self.test_cases_dir.glob("*.json"))
        
        for file_path in test_case_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    test_case = json.load(f)
                    self.test_cases.append(test_case)
            except Exception as e:
                self.logger.error(f"Failed to load test case {file_path}: {e}")
        
        # Generate additional test cases if needed to reach target
        if len(self.test_cases) < self.config.sample_size:
            await self._generate_additional_test_cases(self.config.sample_size - len(self.test_cases))
        
        # Limit sample size if specified
        if len(self.test_cases) > self.config.sample_size:
            import random
            random.shuffle(self.test_cases)
            self.test_cases = self.test_cases[:self.config.sample_size]
        
        self.results.total_test_cases = len(self.test_cases)
        self.logger.info(f"Loaded {len(self.test_cases)} test cases for experiment")
    
    async def _generate_additional_test_cases(self, count: int):
        """Generate additional test cases to reach target sample size"""
        self.logger.info(f"Generating {count} additional test cases")
        
        from research.test_cases.test_case_generator import TestCaseGenerator
        generator = TestCaseGenerator()
        
        # Generate diverse test cases across all categories
        categories = [
            "complex_reasoning", "mathematical_reasoning", "evidence_evaluation",
            "task_decomposition", "context_compression", "claims_reasoning",
            "research_synthesis", "policy_analysis", "system_analysis"
        ]
        
        for i in range(count):
            category = categories[i % len(categories)]
            
            if category == "complex_reasoning":
                test_case = generator._generate_seating_puzzle()
            elif category == "mathematical_reasoning":
                test_case = generator._generate_algebra_problem()
            elif category == "evidence_evaluation":
                test_case = generator._generate_evidence_scenario()
            else:
                test_case = generator._generate_generic_case(category)
            
            self.test_cases.append(test_case)
        
        self.logger.info(f"Generated {count} additional test cases")
    
    async def run_experiment(self) -> bool:
        """Run the complete end-to-end pipeline experiment"""
        self.logger.info(f"Starting End-to-End Pipeline Experiment: {self.experiment_id}")
        self.logger.info(f"Hypothesis: {self.results.hypothesis}")
        
        try:
            # Initialize experiment
            if not await self.initialize():
                return False
            
            # Run both approaches for all test cases
            for i, test_case in enumerate(self.test_cases):
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"Processing test case {i+1}/{len(self.test_cases)}: {test_case['id']}")
                self.logger.info(f"{'='*60}")
                
                # Run baseline approach
                baseline_result = await self._run_baseline_approach(test_case)
                if baseline_result:
                    self.results.baseline_results.append(baseline_result)
                    self.results.successful_baseline += 1
                
                # Run full pipeline approach
                pipeline_result = await self._run_full_pipeline_approach(test_case)
                if pipeline_result:
                    self.results.full_pipeline_results.append(pipeline_result)
                    self.results.successful_full_pipeline += 1
                
                # Save intermediate results if enabled
                if self.config.save_intermediate_results:
                    await self._save_intermediate_results(i, test_case, baseline_result, pipeline_result)
            
            # Perform statistical analysis
            await self._perform_statistical_analysis()
            
            # Generate final report
            await self._generate_final_report()
            
            self.results.end_time = datetime.now()
            self.logger.info("End-to-End Pipeline Experiment completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            self.results.end_time = datetime.now()
            return False
    
    async def _run_baseline_approach(self, test_case: Dict[str, Any]) -> Optional[ExperimentResult]:
        """Run baseline approach (direct prompting)"""
        try:
            start_time = time.time()
            
            # Create direct prompt
            question = test_case.get('question') or test_case.get('task', '')
            prompt = f"""Please answer the following question directly and comprehensively:

{question}

Provide a complete, well-reasoned answer with clear explanations."""
            
            # Execute with tiny model
            response = await self.llm_manager.generate_response(
                prompt=prompt,
                model=self.config.tiny_model,
                temperature=0.1,
                max_tokens=1500
            )
            
            execution_time = time.time() - start_time
            
            # Create result with minimal pipeline stages
            result = ExperimentResult(
                test_case_id=test_case['id'],
                approach="baseline",
                response=response.content,
                execution_time=execution_time,
                token_usage=response.usage.total_tokens if response.usage else 0,
                pipeline_stages=[
                    PipelineStageMetrics(
                        stage_name="direct_prompting",
                        execution_time=execution_time,
                        success=True,
                        tokens_processed=response.usage.total_tokens if response.usage else 0
                    )
                ]
            )
            
            # Evaluate response
            result.evaluation_result = await self._evaluate_response(test_case, result)
            
            self.logger.info(f"Baseline approach completed for {test_case['id']} in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Baseline approach failed for {test_case['id']}: {e}")
            return ExperimentResult(
                test_case_id=test_case['id'],
                approach="baseline",
                response="",
                execution_time=0,
                token_usage=0,
                pipeline_stages=[],
                error_message=str(e)
            )
    
    async def _run_full_pipeline_approach(self, test_case: Dict[str, Any]) -> Optional[ExperimentResult]:
        """Run full Conjecture pipeline approach"""
        try:
            start_time = time.time()
            pipeline_stages = []
            
            # Stage 1: Task Decomposition
            stage_start = time.time()
            question = test_case.get('question') or test_case.get('task', '')
            
            # Create initial claim for task decomposition
            initial_claim = Claim(
                content=f"Analyze and answer: {question}",
                claim_type=ClaimType.TASK,
                scope="user-workspace"
            )
            
            # Process through Conjecture pipeline
            decomposition_result = await self.conjecture.async_evaluation.evaluate_claim_async(initial_claim)
            stage_time = time.time() - stage_start
            
            pipeline_stages.append(PipelineStageMetrics(
                stage_name="task_decomposition",
                execution_time=stage_time,
                success=True,
                quality_score=decomposition_result.get('confidence', 0.0) if decomposition_result else 0.0
            ))
            
            # Stage 2: Context Collection and Compression
            stage_start = time.time()
            context = await self.conjecture.context_collector.collect_context(initial_claim)
            stage_time = time.time() - stage_start
            
            pipeline_stages.append(PipelineStageMetrics(
                stage_name="context_collection",
                execution_time=stage_time,
                success=True,
                tokens_processed=len(str(context))
            ))
            
            # Stage 3: Claims Generation and Evaluation
            stage_start = time.time()
            
            # Generate sub-claims based on decomposition
            sub_claims = []
            if decomposition_result and 'sub_tasks' in decomposition_result:
                for sub_task in decomposition_result['sub_tasks']:
                    sub_claim = Claim(
                        content=sub_task,
                        claim_type=ClaimType.ASSERTION,
                        scope="user-workspace"
                    )
                    sub_claims.append(sub_claim)
            
            # Evaluate sub-claims
            evaluated_claims = []
            for claim in sub_claims:
                claim_result = await self.conjecture.async_evaluation.evaluate_claim_async(claim)
                evaluated_claims.append({
                    'claim': claim,
                    'evaluation': claim_result
                })
            
            stage_time = time.time() - stage_start
            
            pipeline_stages.append(PipelineStageMetrics(
                stage_name="claims_evaluation",
                execution_time=stage_time,
                success=True,
                quality_score=statistics.mean([e.get('confidence', 0.0) for e in evaluated_claims]) if evaluated_claims else 0.0
            ))
            
            # Stage 4: Final Synthesis
            stage_start = time.time()
            
            # Create synthesis prompt with all claims and context
            synthesis_prompt = f"""Based on the following analysis and claims, provide a comprehensive answer to the original question:

Original Question: {question}

Task Decomposition: {decomposition_result.get('analysis', '') if decomposition_result else ''}

Context: {context}

Evaluated Claims:
{json.dumps(evaluated_claims, indent=2, default=str)}

Synthesize this information into a complete, well-reasoned answer."""
            
            final_response = await self.llm_manager.generate_response(
                prompt=synthesis_prompt,
                model=self.config.tiny_model,
                temperature=0.1,
                max_tokens=1500
            )
            
            stage_time = time.time() - stage_start
            
            pipeline_stages.append(PipelineStageMetrics(
                stage_name="final_synthesis",
                execution_time=stage_time,
                success=True,
                tokens_processed=final_response.usage.total_tokens if final_response.usage else 0
            ))
            
            total_execution_time = time.time() - start_time
            
            # Create result
            result = ExperimentResult(
                test_case_id=test_case['id'],
                approach="full_pipeline",
                response=final_response.content,
                execution_time=total_execution_time,
                token_usage=final_response.usage.total_tokens if final_response.usage else 0,
                pipeline_stages=pipeline_stages
            )
            
            # Evaluate response
            result.evaluation_result = await self._evaluate_response(test_case, result)
            
            self.logger.info(f"Full pipeline approach completed for {test_case['id']} in {total_execution_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Full pipeline approach failed for {test_case['id']}: {e}")
            return ExperimentResult(
                test_case_id=test_case['id'],
                approach="full_pipeline",
                response="",
                execution_time=0,
                token_usage=0,
                pipeline_stages=[],
                error_message=str(e)
            )
    
    async def _evaluate_response(self, test_case: Dict[str, Any], result: ExperimentResult) -> EvaluationResult:
        """Evaluate response using LLM-as-a-Judge"""
        try:
            ground_truth = test_case.get('ground_truth', '')
            question = test_case.get('question') or test_case.get('task', '')
            
            evaluation = await self.judge_system.evaluate_response(
                question=question,
                response=result.response,
                ground_truth=ground_truth,
                test_case_metadata=test_case
            )
            
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Evaluation failed for {test_case['id']}: {e}")
            return EvaluationResult(
                overall_score=0.0,
                criterion_scores={criterion: 0.0 for criterion in self.config.evaluation_criteria},
                confidence=0.0,
                justification=f"Evaluation failed: {str(e)}"
            )
    
    async def _save_intermediate_results(self, index: int, test_case: Dict[str, Any], 
                                        baseline_result: Optional[ExperimentResult], 
                                        pipeline_result: Optional[ExperimentResult]):
        """Save intermediate results for debugging and analysis"""
        intermediate_data = {
            "test_case": test_case,
            "baseline_result": asdict(baseline_result) if baseline_result else None,
            "pipeline_result": asdict(pipeline_result) if pipeline_result else None,
            "timestamp": datetime.now().isoformat()
        }
        
        filename = f"intermediate_{index:03d}_{test_case['id']}.json"
        filepath = self.output_dir / "intermediate" / filename
        filepath.parent.mkdir(exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(intermediate_data, f, indent=2, default=str)
    
    async def _perform_statistical_analysis(self):
        """Perform statistical analysis of results"""
        self.logger.info("Performing statistical analysis")
        
        # Extract scores for analysis
        baseline_scores = [r.evaluation_result.overall_score for r in self.results.baseline_results 
                          if r.evaluation_result]
        pipeline_scores = [r.evaluation_result.overall_score for r in self.results.full_pipeline_results 
                          if r.evaluation_result]
        
        if not baseline_scores or not pipeline_scores:
            self.logger.warning("Insufficient data for statistical analysis")
            return
        
        # Perform statistical tests
        statistical_results = await self.statistical_analyzer.compare_approaches(
            baseline_scores=baseline_scores,
            treatment_scores=pipeline_scores,
            alpha=self.config.alpha_level,
            effect_size_threshold=self.config.target_improvement
        )
        
        self.results.statistical_analysis = statistical_results
        
        # Calculate improvement percentage
        baseline_mean = statistics.mean(baseline_scores)
        pipeline_mean = statistics.mean(pipeline_scores)
        improvement = (pipeline_mean - baseline_mean) / baseline_mean if baseline_mean > 0 else 0
        self.results.improvement_percentage = improvement
        
        # Check if hypothesis is validated
        self.results.hypothesis_validated = (
            improvement >= self.config.target_improvement and
            statistical_results.get('significant', False)
        )
        
        self.logger.info(f"Statistical analysis complete:")
        self.logger.info(f"  Baseline mean: {baseline_mean:.3f}")
        self.logger.info(f"  Pipeline mean: {pipeline_mean:.3f}")
        self.logger.info(f"  Improvement: {improvement:.1%}")
        self.logger.info(f"  Hypothesis validated: {self.results.hypothesis_validated}")
    
    async def _generate_final_report(self):
        """Generate comprehensive final report"""
        self.logger.info("Generating final report")
        
        report = {
            "experiment_metadata": {
                "experiment_id": self.experiment_id,
                "hypothesis": self.results.hypothesis,
                "start_time": self.results.start_time.isoformat(),
                "end_time": self.results.end_time.isoformat() if self.results.end_time else None,
                "config": asdict(self.config)
            },
            "test_cases_summary": {
                "total_test_cases": self.results.total_test_cases,
                "successful_baseline": self.results.successful_baseline,
                "successful_full_pipeline": self.results.successful_full_pipeline
            },
            "performance_metrics": {
                "baseline_results": [asdict(r) for r in self.results.baseline_results],
                "full_pipeline_results": [asdict(r) for r in self.results.full_pipeline_results]
            },
            "statistical_analysis": self.results.statistical_analysis,
            "conclusion": {
                "improvement_percentage": self.results.improvement_percentage,
                "target_improvement": self.config.target_improvement,
                "hypothesis_validated": self.results.hypothesis_validated,
                "statistical_significance": self.results.statistical_analysis.get('significant', False) if self.results.statistical_analysis else False
            }
        }
        
        # Save detailed results
        results_file = self.output_dir / f"end_to_end_results_{self.experiment_id}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate markdown report
        await self._generate_markdown_report(report)
        
        self.logger.info(f"Final report saved to {results_file}")
    
    async def _generate_markdown_report(self, report: Dict[str, Any]):
        """Generate markdown report for human reading"""
        md_content = f"""# End-to-End Pipeline Experiment Report

**Experiment ID**: {self.experiment_id}  
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  

## Hypothesis

{self.results.hypothesis}

## Configuration

- **Target Improvement**: {self.config.target_improvement:.1%}
- **Sample Size**: {self.results.total_test_cases} test cases
- **Tiny Model**: {self.config.tiny_model}
- **Judge Model**: {self.config.judge_model}
- **Significance Level**: {self.config.alpha_level}

## Results Summary

### Test Cases
- **Total Test Cases**: {self.results.total_test_cases}
- **Successful Baseline**: {self.results.successful_baseline}
- **Successful Full Pipeline**: {self.results.successful_full_pipeline}

### Performance Improvement
- **Baseline Mean Score**: {statistics.mean([r.evaluation_result.overall_score for r in self.results.baseline_results if r.evaluation_result]):.3f}
- **Pipeline Mean Score**: {statistics.mean([r.evaluation_result.overall_score for r in self.results.full_pipeline_results if r.evaluation_result]):.3f}
- **Improvement**: {self.results.improvement_percentage:.1%}
- **Target Achieved**: {"✅ Yes" if self.results.improvement_percentage >= self.config.target_improvement else "❌ No"}

### Statistical Analysis
"""
        
        if self.results.statistical_analysis:
            stats = self.results.statistical_analysis
            md_content += f"""- **Statistical Significance**: {"✅ Significant" if stats.get('significant', False) else "❌ Not Significant"}
- **P-value**: {stats.get('p_value', 'N/A')}
- **Effect Size**: {stats.get('effect_size', 'N/A')}
- **Confidence Interval**: {stats.get('confidence_interval', 'N/A')}
"""
        
        md_content += f"""
## Conclusion

**Hypothesis Validated**: {"✅ YES" if self.results.hypothesis_validated else "❌ NO"}

The end-to-end pipeline experiment {'successfully validated' if self.results.hypothesis_validated else 'failed to validate'} the core hypothesis that the full Conjecture pipeline shows 25%+ improvement over baseline for tiny models on complex tasks.

### Key Findings

1. **Performance Improvement**: {self.results.improvement_percentage:.1%} improvement over baseline
2. **Statistical Significance**: {self.results.statistical_analysis.get('significant', False) if self.results.statistical_analysis else 'Unknown'}
3. **Target Achievement**: {'Met' if self.results.improvement_percentage >= self.config.target_improvement else 'Not Met'}

### Recommendations

"""
        
        if self.results.hypothesis_validated:
            md_content += """- ✅ The full Conjecture pipeline is ready for production use
- ✅ Tiny models can achieve significant performance improvements with the complete pipeline
- ✅ All pipeline stages are functioning effectively
- ✅ The approach validates the core Conjecture hypothesis
"""
        else:
            md_content += """- 🔧 Further optimization of pipeline stages may be needed
- 🔧 Consider adjusting pipeline configuration for better performance
- 🔧 Additional research into specific bottleneck areas recommended
- 🔧 Review individual pipeline stage effectiveness
"""
        
        # Save markdown report
        md_file = self.output_dir / f"end_to_end_report_{self.experiment_id}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_content)


async def main():
    """Main function to run the end-to-end pipeline experiment"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run End-to-End Pipeline Experiment")
    parser.add_argument("--sample-size", type=int, default=75, help="Number of test cases to use")
    parser.add_argument("--target-improvement", type=float, default=0.25, help="Target improvement percentage")
    parser.add_argument("--output-dir", type=str, default="experiments/results", help="Output directory")
    
    args = parser.parse_args()
    
    # Create experiment configuration
    config = ExperimentConfig(
        sample_size=args.sample_size,
        target_improvement=args.target_improvement,
        output_dir=args.output_dir
    )
    
    # Run experiment
    experiment = EndToEndPipelineExperiment(config)
    success = await experiment.run_experiment()
    
    if success:
        print(f"\n✅ End-to-End Pipeline Experiment completed successfully!")
        print(f"📊 Results saved to: {config.output_dir}")
        print(f"📈 Improvement: {experiment.results.improvement_percentage:.1%}")
        print(f"🎯 Hypothesis Validated: {experiment.results.hypothesis_validated}")
    else:
        print(f"\n❌ Experiment failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())