#!/usr/bin/env python3
"""
Simple Task Decomposition Experiment
Tests if Conjecture methods provide 20%+ improvement in correctness when using task decomposition vs direct approach.

This version works directly with the existing Conjecture codebase without complex dependencies.
"""

import asyncio
import json
import time
import uuid
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
import sys
import os
import requests
from scipy import stats

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@dataclass
class TestResult:
    """Result from a single test case execution"""
    
    test_id: str
    approach: str
    question: str
    generated_answer: str
    execution_time: float
    
    # Evaluation metrics
    correctness: float
    completeness: float
    coherence: float
    reasoning_quality: float
    
    # Metadata
    timestamp: datetime
    difficulty: str


@dataclass
class ExperimentResults:
    """Complete results from task decomposition experiment"""
    
    experiment_id: str
    start_time: datetime
    end_time: Optional[datetime]
    
    # Test results
    direct_results: List[TestResult]
    conjecture_results: List[TestResult]
    
    # Statistical analysis
    improvement_percentages: Dict[str, float]
    statistical_significance: Dict[str, float]
    effect_sizes: Dict[str, float]
    
    # Overall results
    hypothesis_validated: bool
    target_achieved: bool
    confidence_in_results: float


class SimpleTaskDecompositionExperiment:
    """Simplified experiment runner for task decomposition hypothesis validation"""
    
    def __init__(self, sample_size: int = 25):
        self.sample_size = sample_size
        self.target_improvement = 0.20  # 20% improvement target
        self.alpha_level = 0.05  # Statistical significance
        
        # Directory setup
        self.experiments_dir = Path("experiments")
        self.results_dir = Path("experiments/results")
        self.test_cases_dir = Path("experiments/test_cases")
        self.reports_dir = Path("experiments/reports")
        
        for dir_path in [self.experiments_dir, self.results_dir, self.test_cases_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results: ExperimentResults = None
        
        # Logging
        self.logger = self._setup_logging()
        
        # API configuration
        self.api_url = "https://api.z.ai/api/coding/paas/v4"
        self.api_key = "70e6e12e4d7c46e2a4d0b85503d51f38.LQHl8d98kDJChttb"
        self.tiny_model_url = "http://localhost:1234"  # LM Studio for Granite Tiny
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger("simple_task_decomposition_experiment")
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.results_dir / "simple_experiment.log")
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
    
    async def call_api(self, prompt: str, model: str, use_tiny_model: bool = False) -> str:
        """Make API call to get LLM response"""
        
        if use_tiny_model:
            # Call local LM Studio for Granite Tiny
            try:
                response = requests.post(
                    f"{self.tiny_model_url}/v1/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": "ibm/granite-4-h-tiny",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 2000,
                        "temperature": 0.7
                    },
                    timeout=60
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                self.logger.error(f"Tiny model API call failed: {e}")
                return f"Error: Failed to get response from tiny model - {str(e)}"
        
        else:
            # Call Z.AI API for GLM-4.6
            try:
                response = requests.post(
                    f"{self.api_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 2000,
                        "temperature": 0.7
                    },
                    timeout=60
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                self.logger.error(f"API call failed: {e}")
                return f"Error: Failed to get response - {str(e)}"
    
    def generate_test_cases(self) -> List[Dict[str, Any]]:
        """Generate task decomposition test cases"""
        
        self.logger.info(f"Generating {self.sample_size} task decomposition test cases...")
        
        test_cases = []
        
        # Project planning scenarios
        planning_scenarios = [
            "Launch a new mobile app for food delivery in a competitive market",
            "Organize an international conference with 1000+ attendees",
            "Develop a comprehensive employee training program for a Fortune 500 company",
            "Implement a city-wide recycling program to reduce waste by 40%",
            "Design and build a community garden for urban food security",
            "Create a disaster response plan for coastal flooding",
            "Establish a startup incubator for tech entrepreneurs",
            "Plan a digital transformation initiative for a traditional retail company"
        ]
        
        # Multi-step problem solving
        problem_solving_scenarios = [
            "Design a system to reduce customer churn by 25% in 6 months",
            "Optimize supply chain operations to reduce costs by 15% while maintaining quality",
            "Develop a strategy to enter a new international market",
            "Create a process to improve product quality scores by 30%",
            "Design an employee retention program to reduce turnover by 40%",
            "Develop a digital transformation roadmap for a manufacturing company",
            "Create a crisis communication plan for a data breach incident"
        ]
        
        # Strategic planning scenarios
        strategic_scenarios = [
            "Develop a 5-year digital transformation strategy for a traditional bank",
            "Create a market expansion strategy for a SaaS company entering Asia",
            "Design an innovation strategy for a consumer goods company",
            "Develop a sustainability strategy for a manufacturing conglomerate",
            "Create a competitive differentiation strategy for a retail chain"
        ]
        
        all_scenarios = planning_scenarios + problem_solving_scenarios + strategic_scenarios
        
        for i, scenario in enumerate(all_scenarios[:self.sample_size]):
            difficulty = "medium" if i % 3 == 0 else "hard"
            
            case = {
                "id": f"task_decomp_{i+1:03d}",
                "category": "task_decomposition",
                "difficulty": difficulty,
                "task": f"You are tasked with: {scenario}. Break this down into manageable phases and provide a comprehensive solution plan.",
                "scenario": scenario
            }
            test_cases.append(case)
        
        # Save test cases
        test_cases_file = self.test_cases_dir / f"simple_task_decomposition_cases_{self.sample_size}.json"
        with open(test_cases_file, 'w', encoding='utf-8') as f:
            json.dump(test_cases, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Generated {len(test_cases)} task decomposition test cases")
        return test_cases
    
    async def run_experiment(self) -> ExperimentResults:
        """Run the complete task decomposition experiment"""
        
        experiment_id = str(uuid.uuid4())[:8]
        start_time = datetime.utcnow()
        
        self.logger.info(f"Starting Simple Task Decomposition Experiment: {experiment_id}")
        
        # Initialize results
        self.results = ExperimentResults(
            experiment_id=experiment_id,
            start_time=start_time,
            end_time=None,
            direct_results=[],
            conjecture_results=[],
            improvement_percentages={},
            statistical_significance={},
            effect_sizes={},
            hypothesis_validated=False,
            target_achieved=False,
            confidence_in_results=0.0
        )
        
        try:
            # Generate test cases
            test_cases = self.generate_test_cases()
            self.logger.info(f"Generated {len(test_cases)} test cases")
            
            # Run direct approach tests
            self.logger.info("Running direct approach tests...")
            for i, test_case in enumerate(test_cases):
                self.logger.info(f"Direct test {i+1}/{len(test_cases)}: {test_case['id']}")
                result = await self._run_direct_test(test_case)
                if result:
                    self.results.direct_results.append(result)
            
            # Run Conjecture approach tests
            self.logger.info("Running Conjecture approach tests...")
            for i, test_case in enumerate(test_cases):
                self.logger.info(f"Conjecture test {i+1}/{len(test_cases)}: {test_case['id']}")
                result = await self._run_conjecture_test(test_case)
                if result:
                    self.results.conjecture_results.append(result)
            
            # Evaluate results using GLM-4.6 as judge
            self.logger.info("Evaluating results with GLM-4.6 judge...")
            await self._evaluate_results()
            
            # Perform statistical analysis
            self.logger.info("Performing statistical analysis...")
            self._perform_statistical_analysis()
            
            # Determine hypothesis validation
            self._determine_hypothesis_validation()
            
            # Save results
            self.results.end_time = datetime.utcnow()
            await self._save_results()
            
            # Generate report
            await self._generate_report()
            
            self.logger.info(f"Experiment {experiment_id} completed successfully")
            return self.results
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            self.results.end_time = datetime.utcnow()
            await self._save_results()
            raise
    
    async def _run_direct_test(self, test_case: Dict[str, Any]) -> Optional[TestResult]:
        """Run direct approach test"""
        try:
            # Generate direct prompt
            prompt = f"""
Please provide a comprehensive solution to the following task:

{test_case['task']}

Provide a detailed, well-structured response that addresses all aspects of the task. Be thorough and practical in your approach.
"""
            
            # Execute with tiny model
            start_time = time.time()
            response = await self.call_api(prompt, "ibm/granite-4-h-tiny", use_tiny_model=True)
            execution_time = time.time() - start_time
            
            # Create result
            result = TestResult(
                test_id=test_case["id"],
                approach="direct",
                question=test_case["task"],
                generated_answer=response,
                execution_time=execution_time,
                correctness=0.0,  # Will be filled by evaluation
                completeness=0.0,
                coherence=0.0,
                reasoning_quality=0.0,
                timestamp=datetime.utcnow(),
                difficulty=test_case["difficulty"]
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Direct test failed for {test_case['id']}: {e}")
            return None
    
    async def _run_conjecture_test(self, test_case: Dict[str, Any]) -> Optional[TestResult]:
        """Run Conjecture approach test"""
        try:
            # Generate Conjecture prompt with task decomposition
            prompt = f"""
You are using Conjecture's task decomposition approach to solve a complex problem. Break down the task into smaller, manageable claims or subtasks, then provide a comprehensive solution.

**Task:**
{test_case['task']}

**Instructions:**
1. First, decompose the problem into 3-7 key claims or subtasks
2. For each claim/subtask, provide a confidence score (0.0-1.0)
3. Show how the claims relate to each other
4. Provide a final solution based on the claims

Format your response using Conjecture's claim format:
[c1 | claim content | / confidence]
[c2 | supporting claim | / confidence]
[c3 | subtask claim | / confidence]
etc.

Then provide your final comprehensive solution based on these claims.
"""
            
            # Execute with tiny model
            start_time = time.time()
            response = await self.call_api(prompt, "ibm/granite-4-h-tiny", use_tiny_model=True)
            execution_time = time.time() - start_time
            
            # Create result
            result = TestResult(
                test_id=test_case["id"],
                approach="conjecture",
                question=test_case["task"],
                generated_answer=response,
                execution_time=execution_time,
                correctness=0.0,  # Will be filled by evaluation
                completeness=0.0,
                coherence=0.0,
                reasoning_quality=0.0,
                timestamp=datetime.utcnow(),
                difficulty=test_case["difficulty"]
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Conjecture test failed for {test_case['id']}: {e}")
            return None
    
    async def _evaluate_results(self):
        """Evaluate all results using GLM-4.6 as judge"""
        
        all_results = self.results.direct_results + self.results.conjecture_results
        
        for result in all_results:
            try:
                # Create evaluation prompt
                eval_prompt = f"""
You are an expert evaluator assessing AI model responses on complex task decomposition problems.

**Task:**
{result.question}

**Model Response:**
{result.generated_answer}

**Approach Used:** {result.approach}

**Evaluation Instructions:**
Evaluate the response on the following metrics (score 0.0-1.0):

1. **Correctness**: Factual accuracy and correctness of the solution
2. **Completeness**: How thoroughly the response addresses all aspects of the task
3. **Coherence**: Logical flow, consistency, and structural coherence
4. **Reasoning Quality**: Quality of logical reasoning and problem-solving approach

Provide your evaluation in this format:

CORRECTNESS: [0.0-1.0]
COMPLETENESS: [0.0-1.0]
COHERENCE: [0.0-1.0]
REASONING_QUALITY: [0.0-1.0]

Be objective and thorough in your evaluation.
"""
                
                # Get evaluation from GLM-4.6
                evaluation_response = await self.call_api(eval_prompt, "glm-4.6", use_tiny_model=False)
                
                # Parse evaluation
                scores = self._parse_evaluation(evaluation_response)
                
                # Update result with scores
                result.correctness = scores.get('correctness', 0.5)
                result.completeness = scores.get('completeness', 0.5)
                result.coherence = scores.get('coherence', 0.5)
                result.reasoning_quality = scores.get('reasoning_quality', 0.5)
                
            except Exception as e:
                self.logger.error(f"Evaluation failed for {result.test_id}: {e}")
                # Use default scores
                result.correctness = 0.5
                result.completeness = 0.5
                result.coherence = 0.5
                result.reasoning_quality = 0.5
    
    def _parse_evaluation(self, evaluation_response: str) -> Dict[str, float]:
        """Parse evaluation response into scores"""
        scores = {}
        metrics = ['correctness', 'completeness', 'coherence', 'reasoning_quality']
        
        for metric in metrics:
            try:
                # Look for metric name in response
                metric_upper = metric.upper()
                if metric_upper in evaluation_response:
                    # Extract score after the metric name
                    start_idx = evaluation_response.find(metric_upper) + len(metric_upper) + 1
                    end_idx = evaluation_response.find('\n', start_idx)
                    if end_idx == -1:
                        end_idx = len(evaluation_response)
                    
                    score_str = evaluation_response[start_idx:end_idx].strip()
                    scores[metric] = float(score_str)
                else:
                    scores[metric] = 0.5  # Default if not found
            except:
                scores[metric] = 0.5  # Default if parsing fails
        
        return scores
    
    def _perform_statistical_analysis(self):
        """Perform statistical analysis on results"""
        
        # Extract scores for each approach
        direct_scores = {
            'correctness': [r.correctness for r in self.results.direct_results],
            'completeness': [r.completeness for r in self.results.direct_results],
            'coherence': [r.coherence for r in self.results.direct_results],
            'reasoning_quality': [r.reasoning_quality for r in self.results.direct_results]
        }
        
        conjecture_scores = {
            'correctness': [r.correctness for r in self.results.conjecture_results],
            'completeness': [r.completeness for r in self.results.conjecture_results],
            'coherence': [r.coherence for r in self.results.conjecture_results],
            'reasoning_quality': [r.reasoning_quality for r in self.results.conjecture_results]
        }
        
        # Calculate improvements and statistical tests
        for metric in direct_scores.keys():
            direct_mean = statistics.mean(direct_scores[metric]) if direct_scores[metric] else 0
            conjecture_mean = statistics.mean(conjecture_scores[metric]) if conjecture_scores[metric] else 0
            
            # Calculate improvement percentage
            if direct_mean > 0:
                improvement = (conjecture_mean - direct_mean) / direct_mean
                self.results.improvement_percentages[metric] = improvement
            else:
                self.results.improvement_percentages[metric] = 0
            
            # Perform paired t-test if we have paired samples
            if len(direct_scores[metric]) >= 2 and len(conjecture_scores[metric]) >= 2:
                try:
                    # Paired t-test
                    t_stat, p_value = stats.ttest_rel(conjecture_scores[metric], direct_scores[metric])
                    self.results.statistical_significance[metric] = p_value
                    
                    # Calculate effect size (Cohen's d for paired samples)
                    diff_mean = statistics.mean([c - d for c, d in zip(conjecture_scores[metric], direct_scores[metric])])
                    diff_std = statistics.stdev([c - d for c, d in zip(conjecture_scores[metric], direct_scores[metric])]) if len(conjecture_scores[metric]) > 1 else 1
                    effect_size = diff_mean / (diff_std + 0.001)  # Add small constant to avoid division by zero
                    self.results.effect_sizes[metric] = effect_size
                    
                except Exception as e:
                    self.logger.error(f"Statistical analysis failed for {metric}: {e}")
                    self.results.statistical_significance[metric] = 1.0
                    self.results.effect_sizes[metric] = 0.0
            else:
                self.results.statistical_significance[metric] = 1.0
                self.results.effect_sizes[metric] = 0.0
    
    def _determine_hypothesis_validation(self):
        """Determine if the hypothesis is validated"""
        
        # Primary metric is correctness
        correctness_improvement = self.results.improvement_percentages.get('correctness', 0.0)
        correctness_significance = self.results.statistical_significance.get('correctness', 1.0)
        correctness_effect_size = abs(self.results.effect_sizes.get('correctness', 0.0))
        
        # Check if target achieved
        target_achieved = (
            correctness_improvement >= self.target_improvement and
            correctness_significance < self.alpha_level and
            correctness_effect_size >= 0.5
        )
        
        self.results.target_achieved = target_achieved
        
        # Overall hypothesis validation
        hypothesis_validated = target_achieved
        self.results.hypothesis_validated = hypothesis_validated
        
        # Calculate confidence in results
        successful_tests = len(self.results.direct_results) + len(self.results.conjecture_results)
        total_tests = self.sample_size * 2  # Both approaches
        completion_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        # Confidence based on completion rate and statistical significance
        avg_significance = statistics.mean(list(self.results.statistical_significance.values())) if self.results.statistical_significance else 1.0
        self.results.confidence_in_results = completion_rate * (1.0 - avg_significance)
    
    async def _save_results(self):
        """Save experiment results to file"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"simple_task_decomposition_experiment_{self.results.experiment_id}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Convert to serializable format
        results_data = asdict(self.results)
        
        # Convert datetime objects to strings
        results_data['start_time'] = self.results.start_time.isoformat()
        results_data['end_time'] = self.results.end_time.isoformat() if self.results.end_time else None
        
        # Convert test results to dicts
        results_data['direct_results'] = [asdict(r) for r in self.results.direct_results]
        results_data['conjecture_results'] = [asdict(r) for r in self.results.conjecture_results]
        
        # Convert timestamps in test results
        for result_list in [results_data['direct_results'], results_data['conjecture_results']]:
            for result in result_list:
                result['timestamp'] = result['timestamp']
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
    
    async def _generate_report(self):
        """Generate comprehensive experiment report"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"simple_task_decomposition_report_{self.results.experiment_id}_{timestamp}.md"
        filepath = self.reports_dir / filename
        
        report_lines = [
            "# Simple Task Decomposition Experiment Report",
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Experiment ID: {self.results.experiment_id}",
            "",
            "## Executive Summary",
            "",
            f"**Hypothesis**: Small LLMs show 20%+ improvement in correctness when using task decomposition vs direct approach",
            f"**Target Improvement**: {self.target_improvement * 100:.0f}%",
            f"**Sample Size**: {len(self.results.direct_results)} direct + {len(self.results.conjecture_results)} conjecture tests",
            f"**Model Tested**: ibm/granite-4-h-tiny (local)",
            f"**Judge Model**: glm-4.6 (Z.AI API)",
            "",
            "## Results Summary",
            "",
            f"**Hypothesis Validated**: {'✅ YES' if self.results.hypothesis_validated else '❌ NO'}",
            f"**Target Achieved**: {'✅ YES' if self.results.target_achieved else '❌ NO'}",
            f"**Confidence in Results**: {self.results.confidence_in_results:.2%}",
            "",
            "## Performance Metrics",
            "",
            "| Metric | Direct Mean | Conjecture Mean | Improvement | P-value | Effect Size | Significant |",
            "|--------|-------------|----------------|------------|----------|-------------|------------|"
        ]
        
        # Add metric comparisons
        for metric in ['correctness', 'completeness', 'coherence', 'reasoning_quality']:
            
            direct_mean = statistics.mean([getattr(r, metric) for r in self.results.direct_results]) if self.results.direct_results else 0
            conjecture_mean = statistics.mean([getattr(r, metric) for r in self.results.conjecture_results]) if self.results.conjecture_results else 0
            improvement = self.results.improvement_percentages.get(metric, 0)
            p_value = self.results.statistical_significance.get(metric, 1.0)
            effect_size = self.results.effect_sizes.get(metric, 0)
            
            report_lines.append(
                f"| {metric} | {direct_mean:.3f} | {conjecture_mean:.3f} | {improvement:+.1%} | {p_value:.3f} | {effect_size:.3f} | {'✅' if p_value < 0.05 else '❌'} |"
            )
        
        report_lines.extend([
            "",
            "## Statistical Analysis",
            "",
            f"**Primary Metric (Correctness)**:",
            f"- Improvement: {self.results.improvement_percentages.get('correctness', 0):+.1%}",
            f"- Statistical Significance: p = {self.results.statistical_significance.get('correctness', 1.0):.3f}",
            f"- Effect Size (Cohen's d): {self.results.effect_sizes.get('correctness', 0):.3f}",
            "",
            "## Conclusions",
            ""
        ])
        
        if self.results.hypothesis_validated:
            report_lines.extend([
                "✅ **HYPOTHESIS VALIDATED**: The task decomposition approach provides statistically significant improvements.",
                "",
                "### Key Findings:",
                f"- Task decomposition achieved {self.results.improvement_percentages.get('correctness', 0):+.1%} improvement in correctness",
                f"- Results are statistically significant (p < {self.alpha_level})",
                f"- Effect size indicates {'large' if abs(self.results.effect_sizes.get('correctness', 0)) >= 0.8 else 'medium' if abs(self.results.effect_sizes.get('correctness', 0)) >= 0.5 else 'small'} practical significance",
                "",
                "### Recommendations:",
                "- Implement task decomposition as a core feature in Conjecture",
                "- Further optimize the decomposition prompting strategy",
                "- Extend validation to additional model families"
            ])
        else:
            report_lines.extend([
                "❌ **HYPOTHESIS NOT VALIDATED**: Task decomposition did not achieve the target improvement.",
                "",
                "### Key Findings:",
                f"- Task decomposition achieved {self.results.improvement_percentages.get('correctness', 0):+.1%} improvement in correctness",
                f"- Target was {self.target_improvement * 100:.0f}% improvement",
                "- Results did not meet statistical significance or practical significance thresholds",
                "",
                "### Recommendations:",
                "- Refine the task decomposition prompting approach",
                "- Investigate alternative decomposition strategies",
                "- Consider model-specific optimization"
            ])
        
        report_lines.extend([
            "",
            "## Technical Details",
            "",
            f"**Experiment Duration**: {(self.results.end_time - self.results.start_time).total_seconds():.1f} seconds",
            f"**Average Execution Time**: {statistics.mean([r.execution_time for r in self.results.direct_results + self.results.conjecture_results]):.2f} seconds",
            "",
            "## Data Files",
            "",
            f"- Raw results: `experiments/results/simple_task_decomposition_experiment_{self.results.experiment_id}_*.json`",
            f"- Test cases: `experiments/test_cases/simple_task_decomposition_cases_{self.sample_size}.json`",
            "",
            "---",
            f"**Experiment completed**: {self.results.end_time.strftime('%Y-%m-%d %H:%M:%S') if self.results.end_time else 'N/A'}"
        ])
        
        report_content = "\n".join(report_lines)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            self.logger.info(f"Report generated: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate report: {e}")


async def main():
    """Main function to run the simple task decomposition experiment"""
    
    # Initialize experiment with smaller sample size for testing
    sample_size = 10  # Reduced for initial testing
    experiment = SimpleTaskDecompositionExperiment(sample_size)
    
    print("Starting Simple Task Decomposition Experiment...")
    print(f"Target: {experiment.target_improvement * 100:.0f}% improvement in correctness")
    print(f"Sample size: {sample_size} test cases")
    print(f"Model: ibm/granite-4-h-tiny (local)")
    print(f"Judge: glm-4.6 (Z.AI API)")
    print("")
    
    try:
        results = await experiment.run_experiment()
        
        print("\n" + "="*60)
        print("EXPERIMENT RESULTS")
        print("="*60)
        print(f"Hypothesis Validated: {'YES' if results.hypothesis_validated else 'NO'}")
        print(f"Target Achieved: {'YES' if results.target_achieved else 'NO'}")
        print(f"Correctness Improvement: {results.improvement_percentages.get('correctness', 0):+.1%}")
        print(f"Statistical Significance: p = {results.statistical_significance.get('correctness', 1.0):.3f}")
        print(f"Effect Size: {results.effect_sizes.get('correctness', 0):.3f}")
        print(f"Confidence in Results: {results.confidence_in_results:.2%}")
        print("="*60)
        
        if results.hypothesis_validated:
            print("\nSUCCESS: Task decomposition hypothesis validated!")
            print("Conjecture methods provide statistically significant improvements.")
        else:
            print("\nTARGET NOT ACHIEVED: Hypothesis not validated")
            print("Further refinement of task decomposition approach needed.")
            
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)