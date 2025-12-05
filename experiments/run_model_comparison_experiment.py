#!/usr/bin/env python3
"""
Model Comparison Experiment Runner
Tests the hypothesis: "Small models (3-9B) with Conjecture match/exceed larger models (30B+) without Conjecture"

This experiment compares:
- Model A (Small+Conjecture): IBM Granite Tiny with Conjecture methods
- Model B (Large without Conjecture): GLM-4.6 with direct prompting  
- Model C (Large+Conjecture): GLM-4.6 with Conjecture (optional comparison)

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

# Add research to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "research"))
from statistical_analyzer import ConjectureStatisticalAnalyzer

# Add tests to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from test_llm_judge import LLMJudgeSystem, JudgeConfiguration, EvaluationResult


@dataclass
class ExperimentConfig:
    """Configuration for model comparison experiment"""
    
    # Test parameters
    sample_size: int = 75  # Target 50-100 test cases
    alpha_level: float = 0.05  # Statistical significance
    power_target: float = 0.8  # Statistical power
    
    # Model configurations
    small_model: str = "ibm/granite-4-h-tiny"  # 3B parameter model
    large_model: str = "zai-org/GLM-4.6"       # 30B+ parameter model
    judge_model: str = "zai-org/GLM-4.6"       # Judge model
    
    # Provider configurations
    small_provider: Dict[str, Any] = None
    large_provider: Dict[str, Any] = None
    judge_provider: Dict[str, Any] = None
    
    # Evaluation criteria
    evaluation_criteria: List[str] = None
    criterion_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.evaluation_criteria is None:
            self.evaluation_criteria = [
                "correctness", "completeness", "coherence", 
                "reasoning_quality", "confidence_calibration", 
                "efficiency", "hallucination_reduction"
            ]
        
        if self.criterion_weights is None:
            self.criterion_weights = {
                "correctness": 1.5,
                "reasoning_quality": 1.2,
                "completeness": 1.0,
                "coherence": 1.0,
                "confidence_calibration": 1.0,
                "efficiency": 0.5,
                "hallucination_reduction": 1.3
            }


@dataclass
class TestResult:
    """Result from a single test case execution"""
    
    test_id: str
    test_category: str
    approach: str  # small_conjecture, large_direct, large_conjecture
    model: str
    prompt: str
    
    # Response data
    response: str
    response_time: float
    response_length: int
    status: str  # success, error, timeout
    
    # Conjecture-specific data
    claims_generated: List[Dict[str, Any]]
    has_claim_format: bool
    reasoning_steps: int
    
    # Evaluation data
    evaluation_result: Optional[EvaluationResult] = None
    
    # Metadata
    timestamp: datetime
    error_message: Optional[str] = None


class ModelComparisonExperiment:
    """Main experiment runner for model comparison"""
    
    def __init__(self, config: ExperimentConfig = None):
        self.config = config or ExperimentConfig()
        
        # Directory setup
        self.experiments_dir = Path("experiments")
        self.results_dir = Path("experiments/results")
        self.test_cases_dir = Path("research/test_cases")
        self.reports_dir = Path("experiments/reports")
        
        for dir_path in [self.results_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.llm_manager = None
        self.judge_system = None
        self.statistical_analyzer = ConjectureStatisticalAnalyzer(str(self.results_dir))
        
        # Test data
        self.test_cases: List[Dict[str, Any]] = []
        self.results: List[TestResult] = []
        
        # Experiment metadata
        self.experiment_id = f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time = None
        self.end_time = None
        
        # Logging
        self.logger = self._setup_logging()
        
        # Hypothesis definition
        self.hypothesis = "Small models (3-9B) with Conjecture match/exceed larger models (30B+) without Conjecture"
        self.null_hypothesis = "Small models with Conjecture perform worse than large models without Conjecture"
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the experiment"""
        logger = logging.getLogger(f"model_comparison_{self.experiment_id}")
        logger.setLevel(logging.INFO)
        
        # File handler
        log_file = self.results_dir / f"{self.experiment_id}.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    async def initialize(self) -> bool:
        """Initialize the experiment with all required components"""
        try:
            self.logger.info("Initializing Model Comparison Experiment...")
            
            # Setup provider configurations
            await self._setup_providers()
            
            # Initialize LLM manager
            provider_configs = [
                ProviderConfig(**self.config.small_provider),
                ProviderConfig(**self.config.large_provider),
                ProviderConfig(**self.config.judge_provider)
            ]
            
            self.llm_manager = LLMManager(provider_configs)
            
            # Initialize judge system
            judge_config = JudgeConfiguration(
                judge_model=self.config.judge_model,
                evaluation_criteria=self.config.evaluation_criteria,
                criterion_weights=self.config.criterion_weights
            )
            self.judge_system = LLMJudgeSystem(judge_config)
            
            # Test connections
            for i, provider in enumerate(provider_configs):
                test_result = await self.llm_manager.test_connection(provider)
                if not test_result.success:
                    self.logger.error(f"Failed to connect to {provider.model}: {test_result.error}")
                    return False
                else:
                    self.logger.info(f"Successfully connected to {provider.model}")
            
            # Load test cases
            await self._load_test_cases()
            
            self.logger.info("Model comparison experiment initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize experiment: {e}")
            return False
    
    async def _setup_providers(self):
        """Setup provider configurations for the experiment"""
        # Default provider configurations - these should be overridden based on actual setup
        self.config.small_provider = {
            "url": "http://localhost:11434",  # Ollama for local small model
            "api": "",
            "model": self.config.small_model,
            "name": "ollama_small"
        }
        
        self.config.large_provider = {
            "url": "https://llm.chutes.ai/v1",
            "api": os.getenv("CHUTES_API_KEY", ""),
            "model": self.config.large_model,
            "name": "chutes_large"
        }
        
        self.config.judge_provider = {
            "url": "https://llm.chutes.ai/v1",
            "api": os.getenv("CHUTES_API_KEY", ""),
            "model": self.config.judge_model,
            "name": "chutes_judge"
        }
    
    async def _load_test_cases(self):
        """Load test cases from research/test_cases/ directory"""
        self.logger.info("Loading test cases from research/test_cases/")
        
        test_case_files = list(self.test_cases_dir.glob("*.json"))
        
        if not test_case_files:
            self.logger.warning("No test case files found, generating sample test cases")
            await self._generate_sample_test_cases()
            test_case_files = list(self.test_cases_dir.glob("*.json"))
        
        for file_path in test_case_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    test_case = json.load(f)
                    self.test_cases.append(test_case)
            except Exception as e:
                self.logger.error(f"Failed to load test case {file_path}: {e}")
        
        # Limit sample size if specified
        if self.config.sample_size and len(self.test_cases) > self.config.sample_size:
            import random
            random.shuffle(self.test_cases)
            self.test_cases = self.test_cases[:self.config.sample_size]
        
        self.logger.info(f"Loaded {len(self.test_cases)} test cases")
    
    async def _generate_sample_test_cases(self):
        """Generate sample test cases if none exist"""
        from research.test_cases.test_case_generator import TestCaseGenerator
        
        generator = TestCaseGenerator()
        generator.generate_test_suite(count_per_type=3)
        self.logger.info("Generated sample test cases")
    
    async def run_experiment(self) -> bool:
        """Run the complete model comparison experiment"""
        self.start_time = datetime.now()
        self.logger.info(f"Starting Model Comparison Experiment: {self.experiment_id}")
        self.logger.info(f"Hypothesis: {self.hypothesis}")
        
        try:
            # Initialize experiment
            if not await self.initialize():
                return False
            
            # Run all three approaches
            approaches = ["small_conjecture", "large_direct", "large_conjecture"]
            
            for approach in approaches:
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"Running approach: {approach}")
                self.logger.info(f"{'='*60}")
                
                await self._run_approach(approach)
            
            # Evaluate all results
            self.logger.info(f"\n{'='*60}")
            self.logger.info("Evaluating results with LLM-as-a-Judge...")
            self.logger.info(f"{'='*60}")
            
            await self._evaluate_all_results()
            
            # Perform statistical analysis
            self.logger.info(f"\n{'='*60}")
            self.logger.info("Performing statistical analysis...")
            self.logger.info(f"{'='*60}")
            
            await self._perform_statistical_analysis()
            
            # Generate comprehensive report
            self.logger.info(f"\n{'='*60}")
            self.logger.info("Generating comprehensive report...")
            self.logger.info(f"{'='*60}")
            
            await self._generate_comprehensive_report()
            
            self.end_time = datetime.now()
            duration = (self.end_time - self.start_time).total_seconds()
            
            self.logger.info(f"\nExperiment completed successfully in {duration:.1f} seconds")
            return True
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            return False
    
    async def _run_approach(self, approach: str):
        """Run all test cases with a specific approach"""
        for i, test_case in enumerate(self.test_cases):
            self.logger.info(f"[{approach}] Running test case {i+1}/{len(self.test_cases)}: {test_case.get('id', 'unknown')}")
            
            result = await self._execute_single_test(test_case, approach)
            if result:
                self.results.append(result)
                self.logger.info(f"  ✓ Completed in {result.response_time:.1f}s")
            else:
                self.logger.error(f"  ✗ Failed to execute test case")
    
    async def _execute_single_test(self, test_case: Dict[str, Any], approach: str) -> Optional[TestResult]:
        """Execute a single test case with the specified approach"""
        try:
            # Determine model and prompt based on approach
            if approach == "small_conjecture":
                model = self.config.small_model
                prompt = self._format_conjecture_prompt(test_case)
            elif approach == "large_direct":
                model = self.config.large_model
                prompt = self._format_direct_prompt(test_case)
            elif approach == "large_conjecture":
                model = self.config.large_model
                prompt = self._format_conjecture_prompt(test_case)
            else:
                raise ValueError(f"Unknown approach: {approach}")
            
            # Execute the request
            start_time = time.time()
            
            # Create provider config for this approach
            if approach.startswith("small"):
                provider = ProviderConfig(**self.config.small_provider)
            else:
                provider = ProviderConfig(**self.config.large_provider)
            
            # Make LLM call
            response = await self.llm_manager.generate_response(
                provider=provider,
                prompt=prompt,
                max_tokens=2000,
                temperature=0.7
            )
            
            response_time = time.time() - start_time
            
            if not response.success:
                return TestResult(
                    test_id=test_case.get('id', 'unknown'),
                    test_category=test_case.get('category', 'unknown'),
                    approach=approach,
                    model=model,
                    prompt=prompt,
                    response="",
                    response_time=response_time,
                    response_length=0,
                    status="error",
                    claims_generated=[],
                    has_claim_format=False,
                    reasoning_steps=0,
                    timestamp=datetime.now(),
                    error_message=response.error
                )
            
            # Parse response for claims and reasoning steps
            claims_generated = self._parse_claims(response.content) if "conjecture" in approach else []
            has_claim_format = len(claims_generated) > 0
            reasoning_steps = self._count_reasoning_steps(response.content)
            
            return TestResult(
                test_id=test_case.get('id', 'unknown'),
                test_category=test_case.get('category', 'unknown'),
                approach=approach,
                model=model,
                prompt=prompt,
                response=response.content,
                response_time=response_time,
                response_length=len(response.content),
                status="success",
                claims_generated=claims_generated,
                has_claim_format=has_claim_format,
                reasoning_steps=reasoning_steps,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error executing test case: {e}")
            return None
    
    def _format_conjecture_prompt(self, test_case: Dict[str, Any]) -> str:
        """Format prompt for Conjecture approach"""
        question = test_case.get('question', test_case.get('task', ''))
        context = test_case.get('context', '')
        
        prompt = f"""You are an expert knowledge explorer using the Conjecture system. Analyze the following task systematically and generate structured claims.

TASK: {question}

{f'CONTEXT: {context}' if context else ''}

Generate comprehensive claims about this task. Focus on:
1. Factual accuracy and verifiable information
2. Key concepts and relationships
3. Step-by-step reasoning process
4. Final conclusion or solution

For each claim, provide:
- Clear, specific statement
- Confidence score (0.0-1.0)
- Appropriate claim type
- Relevant reasoning

RESPONSE FORMAT:
Claim: "Specific factual statement" Confidence: 0.85 Type: concept Reasoning: [brief explanation]

Analyze the task thoroughly and provide your final answer in the claims format above."""
        
        return prompt
    
    def _format_direct_prompt(self, test_case: Dict[str, Any]) -> str:
        """Format prompt for direct approach (no Conjecture)"""
        question = test_case.get('question', test_case.get('task', ''))
        context = test_case.get('context', '')
        
        prompt = f"""Please analyze and answer the following question directly.

QUESTION: {question}

{f'CONTEXT: {context}' if context else ''}

Provide a clear, comprehensive answer with step-by-step reasoning. Focus on accuracy and completeness."""
        
        return prompt
    
    def _parse_claims(self, response: str) -> List[Dict[str, Any]]:
        """Parse claims from Conjecture-formatted response"""
        claims = []
        
        # Pattern to match claim format
        claim_pattern = r'Claim:\s*"([^"]+)"\s+Confidence:\s*([\d.]+)\s+Type:\s*(\w+)(?:\s+Reasoning:\s*"([^"]*)")?'
        
        matches = re.findall(claim_pattern, response)
        
        for match in matches:
            claim = {
                "statement": match[0],
                "confidence": float(match[1]),
                "type": match[2],
                "reasoning": match[3] if len(match) > 3 else ""
            }
            claims.append(claim)
        
        return claims
    
    def _count_reasoning_steps(self, response: str) -> int:
        """Count reasoning steps in response"""
        # Simple heuristic: count numbered lists, step indicators, etc.
        step_patterns = [
            r'\d+\.\s+',  # Numbered lists
            r'Step\s+\d+',  # "Step 1", "Step 2", etc.
            r'First,?[^,]*',  # "First," "Firstly," etc.
            r'Second,?[^,]*',  # "Second," "Secondly," etc.
            r'Third,?[^,]*',  # "Third," "Thirdly," etc.
            r'Finally,?[^,]*',  # "Finally," etc.
        ]
        
        total_steps = 0
        for pattern in step_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            total_steps += len(matches)
        
        return max(1, total_steps)  # At least 1 step if there's content
    
    async def _evaluate_all_results(self):
        """Evaluate all results using LLM-as-a-Judge"""
        # Initialize judge system if not already done
        if not self.judge_system.llm_manager:
            judge_provider = ProviderConfig(**self.config.judge_provider)
            self.judge_system.llm_manager = LLMManager([judge_provider])
        
        for result in self.results:
            if result.status == "success":
                self.logger.info(f"Evaluating result for {result.test_id} ({result.approach})")
                
                try:
                    evaluation = await self.judge_system.evaluate_response(
                        question=result.prompt,
                        response=result.response,
                        ground_truth=result.test_id,  # Would need actual ground truth
                        category=result.test_category,
                        approach=result.approach
                    )
                    
                    result.evaluation_result = evaluation
                    self.logger.info(f"  ✓ Overall score: {evaluation.overall_score:.3f}")
                    
                except Exception as e:
                    self.logger.error(f"  ✗ Evaluation failed: {e}")
    
    async def _perform_statistical_analysis(self):
        """Perform statistical analysis on the results"""
        # Convert results to format expected by statistical analyzer
        analysis_results = []
        
        for result in self.results:
            if result.evaluation_result:
                analysis_data = {
                    "approach": result.approach,
                    "model": result.model,
                    "test_category": result.test_category,
                    "correctness": result.evaluation_result.criterion_scores.get("correctness", 0.0),
                    "completeness": result.evaluation_result.criterion_scores.get("completeness", 0.0),
                    "coherence": result.evaluation_result.criterion_scores.get("coherence", 0.0),
                    "reasoning_quality": result.evaluation_result.criterion_scores.get("reasoning_quality", 0.0),
                    "confidence_calibration": result.evaluation_result.criterion_scores.get("confidence_calibration", 0.0),
                    "efficiency": result.evaluation_result.criterion_scores.get("efficiency", 0.0),
                    "hallucination_reduction": result.evaluation_result.criterion_scores.get("hallucination_reduction", 0.0),
                    "overall_score": result.evaluation_result.overall_score,
                    "response_time": result.response_time,
                    "response_length": result.response_length,
                    "reasoning_steps": result.reasoning_steps,
                    "has_claim_format": result.has_claim_format,
                    "claims_count": len(result.claims_generated)
                }
                analysis_results.append(analysis_data)
        
        # Define success criteria for hypothesis testing
        success_criteria = {
            "primary_metric": "overall_score",
            "metrics": ["correctness", "reasoning_quality", "overall_score"],
            "threshold": 0.7,  # Minimum acceptable performance
            "improvement_threshold": 0.05,  # 5% improvement threshold
            "primary_comparison": ("small_conjecture", "large_direct")
        }
        
        # Perform statistical analysis
        self.statistical_analysis = self.statistical_analyzer.analyze_hypothesis_results(
            hypothesis_id=self.experiment_id,
            test_results=analysis_results,
            success_criteria=success_criteria
        )
        
        # Save statistical analysis
        analysis_file = self.results_dir / f"{self.experiment_id}_statistical_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.statistical_analysis), f, indent=2, default=str)
        
        self.logger.info(f"Statistical analysis saved to {analysis_file}")
    
    async def _generate_comprehensive_report(self):
        """Generate comprehensive experiment report"""
        report_lines = [
            "# Model Comparison Experiment Report",
            f"Experiment ID: {self.experiment_id}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Duration: {(self.end_time - self.start_time).total_seconds():.1f} seconds",
            "",
            "## Executive Summary",
            "",
            f"**Hypothesis**: {self.hypothesis}",
            f"**Null Hypothesis**: {self.null_hypothesis}",
            "",
            f"**Test Cases**: {len(self.test_cases)}",
            f"**Total Executions**: {len(self.results)}",
            f"**Approaches Tested**: 3 (small_conjecture, large_direct, large_conjecture)",
            "",
            "## Models Compared",
            "",
            f"- **Small Model**: {self.config.small_model} (3B parameters)",
            f"- **Large Model**: {self.config.large_model} (30B+ parameters)",
            f"- **Judge Model**: {self.config.judge_model}",
            "",
            "## Key Findings",
            ""
        ]
        
        # Add hypothesis validation results
        if hasattr(self, 'statistical_analysis'):
            statistical_tests = self.statistical_analysis.get('statistical_tests', {})
            
            # Find the primary comparison test
            primary_test = None
            for test_name, test_result in statistical_tests.items():
                if 'overall_score' in test_name and 'paired' in test_name:
                    primary_test = test_result
                    break
            
            if primary_test:
                report_lines.extend([
                    f"### Primary Hypothesis Test",
                    f"- **Test Statistic**: {primary_test.statistic:.4f}",
                    f"- **P-value**: {primary_test.p_value:.4f}",
                    f"- **Effect Size**: {primary_test.effect_size:.4f}",
                    f"- **Interpretation**: {primary_test.interpretation}",
                    "",
                ])
                
                # Determine hypothesis validation
                if primary_test.p_value < self.config.alpha_level:
                    if primary_test.effect_size > 0:
                        report_lines.append("**HYPOTHESIS VALIDATED**: Small models with Conjecture significantly outperform large models without Conjecture.")
                    else:
                        report_lines.append("**HYPOTHESIS REJECTED**: Small models with Conjecture perform significantly worse than large models without Conjecture.")
                else:
                    report_lines.append("**HYPOTHESIS NOT VALIDATED**: No significant difference found between small models with Conjecture and large models without Conjecture.")
                report_lines.append("")
        
        # Add performance summary by approach
        approach_performance = {}
        for result in self.results:
            if result.evaluation_result:
                approach = result.approach
                if approach not in approach_performance:
                    approach_performance[approach] = []
                approach_performance[approach].append(result.evaluation_result.overall_score)
        
        report_lines.extend([
            "## Performance Summary by Approach",
            ""
        ])
        
        for approach, scores in approach_performance.items():
            if scores:
                report_lines.extend([
                    f"### {approach.replace('_', ' ').title()}",
                    f"- **Mean Score**: {statistics.mean(scores):.3f}",
                    f"- **Std Dev**: {statistics.stdev(scores) if len(scores) > 1 else 0:.3f}",
                    f"- **Min Score**: {min(scores):.3f}",
                    f"- **Max Score**: {max(scores):.3f}",
                    f"- **Sample Size**: {len(scores)}",
                    ""
                ])
        
        # Add detailed statistical analysis
        if hasattr(self, 'statistical_analysis'):
            statistical_report = self.statistical_analyzer.generate_comprehensive_report(
                self.statistical_analysis
            )
            report_lines.extend([
                "## Detailed Statistical Analysis",
                "",
                statistical_report,
                ""
            ])
        
        # Add recommendations
        report_lines.extend([
            "## Recommendations",
            "",
            "Based on the experimental results, the following recommendations are made:",
            ""
        ])
        
        # Generate recommendations based on findings
        if hasattr(self, 'statistical_analysis'):
            recommendations = self.statistical_analysis.get('recommendations', [])
            for i, rec in enumerate(recommendations, 1):
                report_lines.append(f"{i}. {rec}")
        else:
            report_lines.extend([
                "1. Increase sample size for better statistical power",
                "2. Test with additional model sizes for finer-grained comparison",
                "3. Include more diverse test case categories",
                "4. Consider real-world performance metrics beyond accuracy"
            ])
        
        # Add technical details
        report_lines.extend([
            "",
            "## Technical Details",
            "",
            f"### Evaluation Criteria",
            ""
        ])
        
        for criterion in self.config.evaluation_criteria:
            weight = self.config.criterion_weights.get(criterion, 1.0)
            report_lines.append(f"- **{criterion}**: weight = {weight}")
        
        report_lines.extend([
            "",
            f"### Statistical Parameters",
            f"- **Alpha Level**: {self.config.alpha_level}",
            f"- **Power Target**: {self.config.power_target}",
            f"- **Sample Size**: {len(self.test_cases)}",
            "",
            "## Data Files",
            "",
            f"- **Results**: `experiments/results/{self.experiment_id}.json`",
            f"- **Statistical Analysis**: `experiments/results/{self.experiment_id}_statistical_analysis.json`",
            f"- **Log**: `experiments/results/{self.experiment_id}.log`",
            "",
            "---",
            f"*Report generated by Model Comparison Experiment Runner*",
            f"*Conjecture AI-Powered Evidence-Based Reasoning System*"
        ])
        
        # Save report
        report_content = "\n".join(report_lines)
        report_file = self.reports_dir / f"{self.experiment_id}_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"Comprehensive report saved to {report_file}")
        
        # Save raw results
        results_file = self.results_dir / f"{self.experiment_id}.json"
        results_data = {
            "experiment_id": self.experiment_id,
            "hypothesis": self.hypothesis,
            "config": asdict(self.config),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "test_cases": self.test_cases,
            "results": [asdict(result) for result in self.results]
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        self.logger.info(f"Raw results saved to {results_file}")


async def main():
    """Main entry point for the model comparison experiment"""
    print("=" * 80)
    print("MODEL COMPARISON EXPERIMENT")
    print("Testing: Small models with Conjecture vs Large models without Conjecture")
    print("=" * 80)
    
    # Create experiment configuration
    config = ExperimentConfig(
        sample_size=50,  # Adjust based on available test cases
        alpha_level=0.05,
        power_target=0.8
    )
    
    # Create and run experiment
    experiment = ModelComparisonExperiment(config)
    
    try:
        success = await experiment.run_experiment()
        
        if success:
            print("\n" + "=" * 80)
            print("EXPERIMENT COMPLETED SUCCESSFULLY")
            print(f"Results saved to: experiments/results/")
            print(f"Reports saved to: experiments/reports/")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("EXPERIMENT FAILED")
            print("Check the log file for details")
            print("=" * 80)
            return 1
            
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        return 1
    except Exception as e:
        print(f"\nExperiment failed with error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))