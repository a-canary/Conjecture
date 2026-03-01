#!/usr/bin/env python3
"""
Conjecture Research Experiment Framework
Core framework for validating Conjecture's hypotheses through controlled experiments
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import logging

# Add src to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from core.models import Claim, ClaimState, ClaimType
from processing.llm_prompts.context_integrator import ContextIntegrator
from processing.support_systems.context_builder import ContextBuilder
from processing.llm.llm_manager import LLMManager
from config.common import ProviderConfig

class ExperimentType(str, Enum):
    """Types of experiments to run"""
    TASK_DECOMPOSITION = "task_decomposition"
    CONTEXT_COMPRESSION = "context_compression"
    MODEL_COMPARISON = "model_comparison"
    CLAIMS_REASONING = "claims_reasoning"
    END_TO_END = "end_to_end"

class EvaluationMetric(str, Enum):
    """Evaluation metrics for experiments"""
    CORRECTNESS = "correctness"
    COMPLETENESS = "completeness"
    EFFICIENCY = "efficiency"
    COHERENCE = "coherence"
    CONFIDENCE_CALIBRATION = "confidence_calibration"

@dataclass
class ExperimentConfig:
    """Configuration for an experiment"""
    experiment_id: str
    experiment_type: ExperimentType
    name: str
    description: str
    hypothesis: str
    models_to_test: List[str]
    test_cases: List[str]
    metrics: List[EvaluationMetric]
    parameters: Dict[str, Any]
    max_runtime_minutes: int = 30

@dataclass
class TestResult:
    """Result from a single test case"""
    test_case_id: str
    model_name: str
    prompt: str
    response: str
    execution_time_seconds: float
    token_usage: Optional[Dict[str, int]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class EvaluationResult:
    """Result from LLM-as-a-Judge evaluation"""
    test_result_id: str
    judge_model: str
    metric: EvaluationMetric
    score: float  # 0.0 to 1.0
    reasoning: str
    confidence: float

@dataclass
class ExperimentRun:
    """A complete run of an experiment"""
    run_id: str
    experiment_config: ExperimentConfig
    start_time: datetime
    end_time: Optional[datetime] = None
    test_results: List[TestResult] = None
    evaluation_results: List[EvaluationResult] = None
    summary: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.test_results is None:
            self.test_results = []
        if self.evaluation_results is None:
            self.evaluation_results = []

class ExperimentFramework:
    """Main experiment framework for Conjecture research"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "research/config.json"
        self.results_dir = Path("research/results")
        self.test_cases_dir = Path("research/test_cases")
        self.analysis_dir = Path("research/analysis")
        
        # Ensure directories exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.test_cases_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.llm_manager = None
        self.context_builder = None
        self.context_integrator = None
        
        # Experiment tracking
        self.active_runs: Dict[str, ExperimentRun] = {}
        self.completed_runs: List[ExperimentRun] = []
        
        # Logging
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the experiment framework"""
        logger = logging.getLogger("experiment_framework")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        fh = logging.FileHandler(self.results_dir / "experiments.log")
        fh.setLevel(logging.INFO)
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    async def initialize(self, provider_configs: List[ProviderConfig]):
        """Initialize LLM manager and other components"""
        try:
            self.llm_manager = LLMManager(provider_configs)
            self.context_builder = ContextBuilder()
            self.context_integrator = ContextIntegrator(self.context_builder)
            
            self.logger.info("Experiment framework initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize experiment framework: {e}")
            return False
    
    async def run_experiment(self, config: ExperimentConfig) -> ExperimentRun:
        """Run a complete experiment"""
        run_id = str(uuid.uuid4())[:8]
        run = ExperimentRun(
            run_id=run_id,
            experiment_config=config,
            start_time=datetime.utcnow()
        )
        
        self.active_runs[run_id] = run
        self.logger.info(f"Starting experiment {config.name} (run {run_id})")
        
        try:
            # Run test cases for each model
            for model_name in config.models_to_test:
                for test_case_id in config.test_cases:
                    await self._run_single_test(run, model_name, test_case_id, config)
            
            # Evaluate results using LLM-as-a-Judge
            await self._evaluate_results(run, config)
            
            # Generate summary
            run.summary = self._generate_summary(run, config)
            run.end_time = datetime.utcnow()
            
            # Save results
            await self._save_run_results(run)
            
            # Move to completed
            self.completed_runs.append(run)
            del self.active_runs[run_id]
            
            self.logger.info(f"Experiment {config.name} completed successfully")
            return run
            
        except Exception as e:
            self.logger.error(f"Experiment {config.name} failed: {e}")
            run.end_time = datetime.utcnow()
            run.summary = {"error": str(e)}
            await self._save_run_results(run)
            raise
    
    async def _run_single_test(self, run: ExperimentRun, model_name: str, 
                              test_case_id: str, config: ExperimentConfig):
        """Run a single test case"""
        try:
            # Load test case
            test_case = await self._load_test_case(test_case_id)
            if not test_case:
                self.logger.error(f"Test case {test_case_id} not found")
                return
            
            # Generate prompt based on experiment type
            prompt = await self._generate_prompt(test_case, config, model_name)
            
            # Execute prompt
            start_time = time.time()
            response = await self.llm_manager.generate_response(
                prompt=prompt,
                model=model_name,
                max_tokens=config.parameters.get("max_tokens", 2000)
            )
            execution_time = time.time() - start_time
            
            # Store result
            result = TestResult(
                test_case_id=test_case_id,
                model_name=model_name,
                prompt=prompt,
                response=response,
                execution_time_seconds=execution_time,
                metadata={
                    "experiment_type": config.experiment_type.value,
                    "parameters": config.parameters
                }
            )
            
            run.test_results.append(result)
            self.logger.debug(f"Test completed: {model_name} on {test_case_id}")
            
        except Exception as e:
            self.logger.error(f"Test failed: {model_name} on {test_case_id}: {e}")
            result = TestResult(
                test_case_id=test_case_id,
                model_name=model_name,
                prompt="",
                response="",
                execution_time_seconds=0,
                error=str(e)
            )
            run.test_results.append(result)
    
    async def _evaluate_results(self, run: ExperimentRun, config: ExperimentConfig):
        """Evaluate test results using LLM-as-a-Judge"""
        judge_model = "chutes:GLM-4.6"  # Always use GLM-4.6 as judge
        
        for test_result in run.test_results:
            if test_result.error:
                continue
                
            for metric in config.metrics:
                try:
                    evaluation = await self._judge_response(
                        test_result, metric, judge_model
                    )
                    run.evaluation_results.append(evaluation)
                    
                except Exception as e:
                    self.logger.error(f"Evaluation failed: {e}")
    
    async def _judge_response(self, test_result: TestResult, 
                            metric: EvaluationMetric, judge_model: str) -> EvaluationResult:
        """Use LLM-as-a-Judge to evaluate a response"""
        # Load test case for ground truth
        test_case = await self._load_test_case(test_result.test_case_id)
        
        # Create evaluation prompt
        eval_prompt = self._create_evaluation_prompt(
            test_case, test_result, metric
        )
        
        # Get evaluation from judge model
        evaluation_response = await self.llm_manager.generate_response(
            prompt=eval_prompt,
            model=judge_model,
            max_tokens=1000
        )
        
        # Parse evaluation response
        score, reasoning, confidence = self._parse_evaluation(evaluation_response)
        
        return EvaluationResult(
            test_result_id=f"{test_result.test_case_id}_{test_result.model_name}",
            judge_model=judge_model,
            metric=metric,
            score=score,
            reasoning=reasoning,
            confidence=confidence
        )
    
    def _create_evaluation_prompt(self, test_case: Dict[str, Any], 
                                test_result: TestResult, 
                                metric: EvaluationMetric) -> str:
        """Create prompt for LLM-as-a-Judge evaluation"""
        metric_instructions = {
            EvaluationMetric.CORRECTNESS: """
            Evaluate the factual correctness of the response compared to the expected answer.
            Score 0.0-1.0 where 1.0 = completely correct, 0.0 = completely incorrect.
            """,
            EvaluationMetric.COMPLETENESS: """
            Evaluate how completely the response addresses all aspects of the question.
            Score 0.0-1.0 where 1.0 = fully complete, 0.0 = missing major points.
            """,
            EvaluationMetric.EFFICIENCY: """
            Evaluate the efficiency and conciseness of the response.
            Score 0.0-1.0 where 1.0 = highly efficient, 0.0 = verbose or inefficient.
            """,
            EvaluationMetric.COHERENCE: """
            Evaluate the logical coherence and flow of the response.
            Score 0.0-1.0 where 1.0 = perfectly coherent, 0.0 = incoherent.
            """,
            EvaluationMetric.CONFIDENCE_CALIBRATION: """
            Evaluate how well the model's confidence matches its actual accuracy.
            Score 0.0-1.0 where 1.0 = perfectly calibrated, 0.0 = poorly calibrated.
            """
        }
        
        prompt = f"""
You are an expert evaluator for AI model responses. Your task is to evaluate a response based on the metric: {metric.value}.

{metric_instructions.get(metric, "")}

**Question/Task:**
{test_case.get('question', test_case.get('task', ''))}

**Expected Answer (Ground Truth):**
{test_case.get('expected_answer', test_case.get('ground_truth', ''))}

**Model Response:**
{test_result.response}

**Context:** This response was generated by model: {test_result.model_name}

Please evaluate the response and provide your assessment in the following format:

SCORE: [0.0-1.0]
REASONING: [Detailed explanation of your evaluation]
CONFIDENCE: [0.0-1.0 for your confidence in this evaluation]

Be objective, thorough, and fair in your evaluation.
"""
        return prompt
    
    def _parse_evaluation(self, evaluation_response: str) -> Tuple[float, str, float]:
        """Parse evaluation response into score, reasoning, and confidence"""
        try:
            # Extract score
            score_match = evaluation_response.split("SCORE:")[1].split("\n")[0].strip()
            score = float(score_match)
            
            # Extract reasoning
            reasoning_start = evaluation_response.find("REASONING:")
            reasoning_end = evaluation_response.find("CONFIDENCE:")
            reasoning = evaluation_response[reasoning_start:reasoning_end].replace("REASONING:", "").strip()
            
            # Extract confidence
            confidence_match = evaluation_response.split("CONFIDENCE:")[1].strip()
            confidence = float(confidence_match)
            
            return score, reasoning, confidence
            
        except Exception as e:
            self.logger.warning(f"Failed to parse evaluation: {e}")
            return 0.5, f"Parsing error: {evaluation_response}", 0.3
    
    async def _load_test_case(self, test_case_id: str) -> Optional[Dict[str, Any]]:
        """Load a test case from file"""
        test_case_file = self.test_cases_dir / f"{test_case_id}.json"
        try:
            with open(test_case_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Test case file not found: {test_case_file}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to load test case {test_case_id}: {e}")
            return None
    
    async def _generate_prompt(self, test_case: Dict[str, Any], 
                             config: ExperimentConfig, model_name: str) -> str:
        """Generate prompt based on experiment type"""
        if config.experiment_type == ExperimentType.TASK_DECOMPOSITION:
            return self._generate_task_decomposition_prompt(test_case, config)
        elif config.experiment_type == ExperimentType.CONTEXT_COMPRESSION:
            return self._generate_context_compression_prompt(test_case, config)
        elif config.experiment_type == ExperimentType.MODEL_COMPARISON:
            return self._generate_model_comparison_prompt(test_case, config)
        elif config.experiment_type == ExperimentType.CLAIMS_REASONING:
            return self._generate_claims_reasoning_prompt(test_case, config)
        else:
            return test_case.get('question', test_case.get('task', ''))
    
    def _generate_task_decomposition_prompt(self, test_case: Dict[str, Any], 
                                          config: ExperimentConfig) -> str:
        """Generate prompt for task decomposition experiment"""
        return f"""
You are tasked with solving a complex problem. Use Conjecture's approach of breaking down the problem into smaller, manageable claims or subtasks.

**Problem:**
{test_case.get('task', '')}

**Instructions:**
1. Decompose the problem into 3-5 key claims or subtasks
2. For each claim/subtask, provide a confidence score (0.0-1.0)
3. Show how the claims relate to each other
4. Provide a final solution based on the claims

Format your response using Conjecture's claim format:
[c1 | claim content | / confidence]
[c2 | supporting claim | / confidence]
etc.

Then provide your final solution.
"""
    
    def _generate_context_compression_prompt(self, test_case: Dict[str, Any], 
                                           config: ExperimentConfig) -> str:
        """Generate prompt for context compression experiment"""
        context = test_case.get('context', '')
        question = test_case.get('question', '')
        
        return f"""
You are given a large context and a question. Use Conjecture's approach to compress and optimize the context while preserving essential information.

**Context:**
{context}

**Question:**
{question}

**Instructions:**
1. Extract the most relevant claims/facts from the context
2. Organize them by relevance and confidence
3. Compress the context using Conjecture's claim format
4. Answer the question based on the compressed context

Format your response using:
[c1 | key fact 1 | / confidence]
[c2 | key fact 2 | / confidence]
etc.

Then provide your answer to the question.
"""
    
    def _generate_model_comparison_prompt(self, test_case: Dict[str, Any], 
                                        config: ExperimentConfig) -> str:
        """Generate prompt for model comparison experiment"""
        return f"""
Answer the following question to the best of your ability. This is part of a model comparison study.

**Question:**
{test_case.get('question', '')}

**Context (if provided):**
{test_case.get('context', 'No context provided.')}

Provide a clear, accurate, and complete answer.
"""
    
    def _generate_claims_reasoning_prompt(self, test_case: Dict[str, Any], 
                                        config: ExperimentConfig) -> str:
        """Generate prompt for claims-based reasoning experiment"""
        claims = test_case.get('claims', [])
        question = test_case.get('question', '')
        
        claims_text = '\n'.join([f"[{claim['id']}] {claim['content']} (confidence: {claim['confidence']})" 
                                for claim in claims])
        
        return f"""
You are given a set of claims with confidence scores. Use Conjecture's reasoning approach to answer the question based on these claims.

**Claims:**
{claims_text}

**Question:**
{question}

**Instructions:**
1. Analyze the given claims and their confidence scores
2. Identify supporting and contradictory relationships
3. Reason step-by-step using the claims as evidence
4. Provide your final answer with an overall confidence assessment

Format your response to show your reasoning process clearly.
"""
    
    def _generate_summary(self, run: ExperimentRun, config: ExperimentConfig) -> Dict[str, Any]:
        """Generate summary statistics for the experiment run"""
        if not run.test_results:
            return {"error": "No test results to summarize"}
        
        # Basic statistics
        total_tests = len(run.test_results)
        successful_tests = len([r for r in run.test_results if not r.error])
        failed_tests = total_tests - successful_tests
        
        # Performance by model
        model_stats = {}
        for result in run.test_results:
            if result.model_name not in model_stats:
                model_stats[result.model_name] = {
                    'total': 0,
                    'successful': 0,
                    'avg_time': 0,
                    'total_time': 0
                }
            
            stats = model_stats[result.model_name]
            stats['total'] += 1
            stats['total_time'] += result.execution_time_seconds
            if not result.error:
                stats['successful'] += 1
        
        # Calculate averages
        for model_name, stats in model_stats.items():
            if stats['total'] > 0:
                stats['avg_time'] = stats['total_time'] / stats['total']
                stats['success_rate'] = stats['successful'] / stats['total']
        
        # Evaluation metrics
        evaluation_stats = {}
        if run.evaluation_results:
            for metric in config.metrics:
                metric_results = [e for e in run.evaluation_results if e.metric == metric]
                if metric_results:
                    scores = [e.score for e in metric_results]
                    evaluation_stats[metric.value] = {
                        'avg_score': sum(scores) / len(scores),
                        'min_score': min(scores),
                        'max_score': max(scores),
                        'count': len(scores)
                    }
        
        return {
            'experiment_id': run.run_id,
            'experiment_name': config.name,
            'experiment_type': config.experiment_type.value,
            'duration_seconds': (run.end_time - run.start_time).total_seconds() if run.end_time else None,
            'test_statistics': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'failed_tests': failed_tests,
                'success_rate': successful_tests / total_tests if total_tests > 0 else 0
            },
            'model_performance': model_stats,
            'evaluation_metrics': evaluation_stats,
            'hypothesis': config.hypothesis
        }
    
    async def _save_run_results(self, run: ExperimentRun):
        """Save experiment run results to file"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{run.experiment_config.experiment_type.value}_{run.run_id}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Convert to serializable format
        run_data = {
            'run_id': run.run_id,
            'experiment_config': asdict(run.experiment_config),
            'start_time': run.start_time.isoformat(),
            'end_time': run.end_time.isoformat() if run.end_time else None,
            'test_results': [asdict(r) for r in run.test_results],
            'evaluation_results': [asdict(e) for e in run.evaluation_results],
            'summary': run.summary
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(run_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
    
    async def get_experiment_results(self, run_id: str) -> Optional[ExperimentRun]:
        """Load experiment results from file"""
        # Check active runs first
        if run_id in self.active_runs:
            return self.active_runs[run_id]
        
        # Check completed runs
        for run in self.completed_runs:
            if run.run_id == run_id:
                return run
        
        # Try to load from file
        results_files = list(self.results_dir.glob(f"*_{run_id}_*.json"))
        if results_files:
            try:
                with open(results_files[0], 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Reconstruct ExperimentRun
                config = ExperimentConfig(**data['experiment_config'])
                run = ExperimentRun(
                    run_id=data['run_id'],
                    experiment_config=config,
                    start_time=datetime.fromisoformat(data['start_time']),
                    end_time=datetime.fromisoformat(data['end_time']) if data['end_time'] else None,
                    summary=data['summary']
                )
                
                # Reconstruct test results
                for tr_data in data['test_results']:
                    run.test_results.append(TestResult(**tr_data))
                
                # Reconstruct evaluation results
                for er_data in data['evaluation_results']:
                    run.evaluation_results.append(EvaluationResult(**er_data))
                
                return run
                
            except Exception as e:
                self.logger.error(f"Failed to load experiment results: {e}")
        
        return None
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all available experiments"""
        experiments = []
        
        # Add active experiments
        for run_id, run in self.active_runs.items():
            experiments.append({
                'run_id': run_id,
                'name': run.experiment_config.name,
                'type': run.experiment_config.experiment_type.value,
                'status': 'running',
                'start_time': run.start_time.isoformat()
            })
        
        # Add completed experiments
        for run in self.completed_runs:
            experiments.append({
                'run_id': run.run_id,
                'name': run.experiment_config.name,
                'type': run.experiment_config.experiment_type.value,
                'status': 'completed',
                'start_time': run.start_time.isoformat(),
                'end_time': run.end_time.isoformat() if run.end_time else None
            })
        
        return experiments