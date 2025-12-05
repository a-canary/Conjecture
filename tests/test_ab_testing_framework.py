#!/usr/bin/env python3
"""
A/B Testing Framework for Conjecture Hypothesis Validation
Implements systematic comparison between direct and Conjecture approaches
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
import statistics
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.models import Claim, ClaimState, ClaimType
from processing.llm.llm_manager import LLMManager
from config.common import ProviderConfig


@dataclass
class ABTestConfiguration:
    """Configuration for A/B testing"""
    
    # Test approaches
    approaches: List[str] = None
    
    # Model configurations
    test_models: List[str] = None
    
    # Evaluation settings
    judge_model: str = "zai-org/GLM-4.6"
    evaluation_criteria: List[str] = None
    
    # Randomization
    randomize_order: bool = True
    balanced_assignment: bool = True
    
    def __post_init__(self):
        if self.approaches is None:
            self.approaches = ["direct", "conjecture", "few_shot"]
        
        if self.test_models is None:
            self.test_models = ["ibm/granite-4-h-tiny", "zai-org/GLM-4.6"]
        
        if self.evaluation_criteria is None:
            self.evaluation_criteria = [
                "correctness", "completeness", "coherence", 
                "reasoning_quality", "confidence_calibration", 
                "efficiency", "hallucination_reduction"
            ]


@dataclass
class ABTestResult:
    """Result from a single A/B test execution"""
    
    test_id: str
    category: str
    test_case: Dict[str, Any]
    
    # Direct approach results
    direct_result: Optional[Dict[str, Any]] = None
    
    # Conjecture approach results
    conjecture_result: Optional[Dict[str, Any]] = None
    
    # Few-shot approach results (if applicable)
    few_shot_result: Optional[Dict[str, Any]] = None
    
    # Comparative analysis
    winner: Optional[str] = None
    improvement_percentage: Optional[float] = None
    statistical_significance: Optional[float] = None
    
    # Metadata
    timestamp: datetime = None
    execution_time: float = 0.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class ABTestingFramework:
    """Comprehensive A/B testing framework for Conjecture validation"""
    
    def __init__(self, config: ABTestConfiguration = None):
        self.config = config or ABTestConfiguration()
        
        # Directory setup
        self.results_dir = Path("tests/results/ab_testing")
        self.prompts_dir = Path("tests/prompts/ab_testing")
        
        for dir_path in [self.results_dir, self.prompts_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.llm_manager = None
        
        # Results storage
        self.test_results: List[ABTestResult] = []
        self.category_results: Dict[str, List[ABTestResult]] = {}
        
        # Logging
        self.logger = self._setup_logging()
        
        # Load prompt templates
        self.prompt_templates = self._load_prompt_templates()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for A/B testing framework"""
        logger = logging.getLogger("ab_testing")
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.results_dir / "ab_testing.log")
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
    
    async def initialize(self, provider_configs: List[ProviderConfig]) -> bool:
        """Initialize LLM manager and validate connections"""
        try:
            self.llm_manager = LLMManager(provider_configs)
            
            # Test connections
            for provider in provider_configs:
                test_result = await self.llm_manager.test_connection(provider)
                if not test_result.success:
                    self.logger.error(f"Failed to connect to {provider.model}: {test_result.error}")
                    return False
            
            self.logger.info("A/B testing framework initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize A/B testing framework: {e}")
            return False
    
    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load prompt templates for different approaches"""
        
        # Create prompt templates directory if it doesn't exist
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        
        templates = {
            "direct": """
You are a helpful AI assistant. Please provide a direct, comprehensive answer to the following question.

Question: {question}

Provide your answer:
""",
            
            "conjecture": """
You are an AI assistant using the Conjecture methodology for enhanced reasoning. Break down the problem into claims and evaluate each systematically.

JSON Frontmatter:
```json
{{
  "confidence": 0.8,
  "tags": ["analysis", "reasoning"],
  "context_type": "problem_solving",
  "reasoning_approach": "claims_based"
}}
```

Problem Analysis:
1. Identify the core question and key components
2. Break down into sub-claims or smaller reasoning steps
3. Evaluate each claim with confidence scores
4. Synthesize into a comprehensive answer

Question: {question}

Analysis:
""",
            
            "few_shot": """
You are an AI assistant. Use the following examples to guide your answer, then provide your response to the question.

Example 1:
Question: What is the capital of France?
Answer: The capital of France is Paris. This is a well-established geographical fact.

Example 2:
Question: How do you calculate the area of a circle?
Answer: The area of a circle is calculated using the formula A = πr², where r is the radius of the circle.

Now, answer this question:
Question: {question}

Answer:
""",
        }
        
        # Save templates to files
        for approach, template in templates.items():
            template_file = self.prompts_dir / f"{approach}_template.txt"
            with open(template_file, 'w', encoding='utf-8') as f:
                f.write(template)
        
        return templates
    
    async def run_ab_test(self, test_case: Dict[str, Any], category: str) -> ABTestResult:
        """Run a single A/B test comparing approaches"""
        
        test_id = str(uuid.uuid4())
        start_time = time.time()
        
        self.logger.info(f"Running A/B test {test_id} for category: {category}")
        
        # Initialize result
        result = ABTestResult(
            test_id=test_id,
            category=category,
            test_case=test_case
        )
        
        # Extract question and context
        question = test_case.get("question", "")
        context = test_case.get("context", "")
        
        # Prepare full question with context
        full_question = self._prepare_question(question, context)
        
        # Test each approach
        for approach in self.config.approaches:
            try:
                approach_result = await self._run_approach(
                    approach, full_question, test_case, category
                )
                
                # Store result
                if approach == "direct":
                    result.direct_result = approach_result
                elif approach == "conjecture":
                    result.conjecture_result = approach_result
                elif approach == "few_shot":
                    result.few_shot_result = approach_result
                    
            except Exception as e:
                self.logger.error(f"Error running {approach} approach: {e}")
                # Create error result
                error_result = {
                    "approach": approach,
                    "error": str(e),
                    "answer": "",
                    "execution_time": 0.0,
                    "token_usage": 0
                }
                
                if approach == "direct":
                    result.direct_result = error_result
                elif approach == "conjecture":
                    result.conjecture_result = error_result
                elif approach == "few_shot":
                    result.few_shot_result = error_result
        
        # Calculate execution time
        result.execution_time = time.time() - start_time
        
        # Perform comparative analysis
        await self._analyze_comparison(result)
        
        return result
    
    def _prepare_question(self, question: str, context: str = "") -> str:
        """Prepare the full question with context"""
        if context:
            return f"Context:\n{context}\n\nQuestion: {question}"
        return question
    
    async def _run_approach(
        self, 
        approach: str, 
        question: str, 
        test_case: Dict[str, Any], 
        category: str
    ) -> Dict[str, Any]:
        """Run a specific approach on the question"""
        
        start_time = time.time()
        
        # Get the appropriate prompt template
        template = self.prompt_templates.get(approach, self.prompt_templates["direct"])
        
        # Format the prompt
        if approach == "conjecture":
            # For Conjecture approach, add category-specific reasoning guidance
            guidance = self._get_conjecture_guidance(category)
            formatted_prompt = template.format(
                question=question,
                guidance=guidance
            )
        else:
            formatted_prompt = template.format(question=question)
        
        # Select model based on approach
        model = self.config.test_models[0] if approach == "conjecture" else self.config.test_models[1]
        
        # Make LLM call
        response = await self.llm_manager.generate_response(
            prompt=formatted_prompt,
            model=model,
            max_tokens=2000,
            temperature=0.7
        )
        
        execution_time = time.time() - start_time
        
        return {
            "approach": approach,
            "model": model,
            "answer": response.content,
            "execution_time": execution_time,
            "token_usage": response.token_usage if hasattr(response, 'token_usage') else 0,
            "prompt": formatted_prompt
        }
    
    def _get_conjecture_guidance(self, category: str) -> str:
        """Get category-specific guidance for Conjecture approach"""
        
        guidance_map = {
            "complex_reasoning": """
Focus on logical decomposition:
- Identify premises and conclusions
- Break down complex inferences into smaller steps
- Validate each logical step
- Synthesize into coherent reasoning
""",
            
            "mathematical_reasoning": """
Focus on mathematical decomposition:
- Identify given information and what needs to be found
- Break down into smaller calculations
- Show step-by-step work
- Verify final answer
""",
            
            "context_compression": """
Focus on relevant information extraction:
- Identify key claims and evidence
- Filter out irrelevant details
- Organize information logically
- Synthesize comprehensive answer
""",
            
            "evidence_evaluation": """
Focus on evidence assessment:
- Evaluate strength and reliability of each source
- Identify conflicts and agreements
- Weigh evidence appropriately
- Form evidence-based conclusion
""",
            
            "task_decomposition": """
Focus on systematic planning:
- Break task into manageable subtasks
- Identify dependencies and sequence
- Allocate resources appropriately
- Create actionable plan
""",
            
            "coding_tasks": """
Focus on systematic problem-solving:
- Understand requirements thoroughly
- Design algorithm step by step
- Implement with clean code
- Test and validate solution
"""
        }
        
        return guidance_map.get(category, "")
    
    async def _analyze_comparison(self, result: ABTestResult):
        """Analyze and compare the results from different approaches"""
        
        # Collect approach results
        approaches = {}
        if result.direct_result:
            approaches["direct"] = result.direct_result
        if result.conjecture_result:
            approaches["conjecture"] = result.conjecture_result
        if result.few_shot_result:
            approaches["few_shot"] = result.few_shot_result
        
        if len(approaches) < 2:
            self.logger.warning(f"Insufficient approaches for comparison in test {result.test_id}")
            return
        
        # Evaluate each approach using LLM-as-a-Judge
        evaluations = {}
        for approach_name, approach_result in approaches.items():
            if "error" not in approach_result:
                evaluation = await self._evaluate_approach(
                    approach_result, result.test_case, result.category
                )
                evaluations[approach_name] = evaluation
        
        # Compare approaches
        if len(evaluations) >= 2:
            comparison = self._compare_approaches(evaluations)
            result.winner = comparison.get("winner")
            result.improvement_percentage = comparison.get("improvement_percentage")
            result.statistical_significance = comparison.get("statistical_significance")
    
    async def _evaluate_approach(
        self, 
        approach_result: Dict[str, Any], 
        test_case: Dict[str, Any], 
        category: str
    ) -> Dict[str, float]:
        """Evaluate an approach using LLM-as-a-Judge"""
        
        # Create evaluation prompt
        evaluation_prompt = self._create_evaluation_prompt(
            approach_result, test_case, category
        )
        
        # Get evaluation from judge model
        response = await self.llm_manager.generate_response(
            prompt=evaluation_prompt,
            model=self.config.judge_model,
            max_tokens=1000,
            temperature=0.1  # Low temperature for consistent evaluation
        )
        
        # Parse evaluation scores
        try:
            scores = self._parse_evaluation_scores(response.content)
        except Exception as e:
            self.logger.error(f"Failed to parse evaluation scores: {e}")
            # Return default scores
            scores = {criterion: 0.5 for criterion in self.config.evaluation_criteria}
        
        return scores
    
    def _create_evaluation_prompt(
        self, 
        approach_result: Dict[str, Any], 
        test_case: Dict[str, Any], 
        category: str
    ) -> str:
        """Create evaluation prompt for LLM-as-a-Judge"""
        
        question = test_case.get("question", "")
        expected_answer = test_case.get("expected_answer", "")
        answer = approach_result.get("answer", "")
        approach = approach_result.get("approach", "")
        
        prompt = f"""
You are an expert evaluator assessing AI responses. Please evaluate the following answer on multiple criteria.

Question: {question}
Expected Answer Type: {test_case.get('expected_answer_type', 'comprehensive answer')}
Approach Used: {approach}

Answer to Evaluate:
{answer}

Please evaluate the answer on the following criteria (score from 0.0 to 1.0):

1. Correctness: How factually accurate is the answer?
2. Completeness: How thoroughly does the answer address all aspects of the question?
3. Coherence: How well-structured and logical is the answer?
4. Reasoning Quality: How strong is the reasoning process demonstrated?
5. Confidence Calibration: How well does the confidence level match the actual accuracy?
6. Efficiency: How concise and to-the-point is the answer?
7. Hallucination Reduction: How well-grounded is the answer in the provided information?

Provide your evaluation in this JSON format:
{{
  "correctness": 0.0-1.0,
  "completeness": 0.0-1.0,
  "coherence": 0.0-1.0,
  "reasoning_quality": 0.0-1.0,
  "confidence_calibration": 0.0-1.0,
  "efficiency": 0.0-1.0,
  "hallucination_reduction": 0.0-1.0,
  "overall_score": 0.0-1.0,
  "detailed_feedback": "brief explanation of scores"
}}
"""
        
        return prompt
    
    def _parse_evaluation_scores(self, evaluation_text: str) -> Dict[str, float]:
        """Parse evaluation scores from LLM response"""
        
        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', evaluation_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group()
                scores = json.loads(json_str)
                
                # Ensure all criteria are present
                parsed_scores = {}
                for criterion in self.config.evaluation_criteria:
                    parsed_scores[criterion] = float(scores.get(criterion, 0.5))
                
                return parsed_scores
            
        except Exception as e:
            self.logger.error(f"Error parsing evaluation JSON: {e}")
        
        # Fallback: try to extract individual scores
        scores = {}
        for criterion in self.config.evaluation_criteria:
            pattern = f'{criterion}[:\s]+([0-9.]+)'
            match = re.search(pattern, evaluation_text.lower())
            scores[criterion] = float(match.group(1)) if match else 0.5
        
        return scores
    
    def _compare_approaches(self, evaluations: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Compare approaches and determine winner"""
        
        # Calculate overall scores for each approach
        approach_scores = {}
        for approach, scores in evaluations.items():
            # Weighted average (could be customized)
            weights = {
                "correctness": 1.5,
                "reasoning_quality": 1.2,
                "completeness": 1.0,
                "coherence": 1.0,
                "confidence_calibration": 1.0,
                "efficiency": 0.5,
                "hallucination_reduction": 1.3
            }
            
            weighted_score = 0.0
            total_weight = 0.0
            
            for criterion, score in scores.items():
                weight = weights.get(criterion, 1.0)
                weighted_score += score * weight
                total_weight += weight
            
            approach_scores[approach] = weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Find winner
        winner = max(approach_scores.keys(), key=lambda x: approach_scores[x])
        
        # Calculate improvement percentage
        if "direct" in approach_scores and "conjecture" in approach_scores:
            improvement = ((approach_scores["conjecture"] - approach_scores["direct"]) / 
                          approach_scores["direct"]) * 100
        else:
            improvement = 0.0
        
        # Simple statistical significance (would need more samples for real test)
        significance = 0.05 if abs(improvement) > 10 else 0.1  # Placeholder
        
        return {
            "winner": winner,
            "approach_scores": approach_scores,
            "improvement_percentage": improvement,
            "statistical_significance": significance
        }
    
    async def run_category_tests(
        self, 
        test_cases: List[Dict[str, Any]], 
        category: str
    ) -> List[ABTestResult]:
        """Run all A/B tests for a specific category"""
        
        self.logger.info(f"Running A/B tests for category: {category} ({len(test_cases)} test cases)")
        
        category_results = []
        
        for i, test_case in enumerate(test_cases):
            self.logger.info(f"Running test {i+1}/{len(test_cases)} for {category}")
            
            try:
                result = await self.run_ab_test(test_case, category)
                category_results.append(result)
                
                # Save intermediate results
                await self._save_test_result(result)
                
            except Exception as e:
                self.logger.error(f"Error running test {i+1} for {category}: {e}")
                continue
        
        # Store category results
        self.category_results[category] = category_results
        self.test_results.extend(category_results)
        
        self.logger.info(f"Completed {len(category_results)} tests for {category}")
        
        return category_results
    
    async def _save_test_result(self, result: ABTestResult):
        """Save individual test result to file"""
        
        # Convert to dictionary for JSON serialization
        result_dict = asdict(result)
        result_dict["timestamp"] = result.timestamp.isoformat()
        
        # Save to file
        filename = f"ab_test_{result.test_id}.json"
        filepath = self.results_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save test result {result.test_id}: {e}")
    
    def generate_category_summary(self, category: str) -> Dict[str, Any]:
        """Generate summary statistics for a category"""
        
        if category not in self.category_results:
            return {}
        
        results = self.category_results[category]
        
        # Count wins per approach
        approach_wins = {}
        approach_scores = {}
        
        for result in results:
            if result.winner:
                approach_wins[result.winner] = approach_wins.get(result.winner, 0) + 1
            
            # Collect scores for statistical analysis
            for approach_name, approach_result in [
                ("direct", result.direct_result),
                ("conjecture", result.conjecture_result),
                ("few_shot", result.few_shot_result)
            ]:
                if approach_result and "evaluation" in approach_result:
                    if approach_name not in approach_scores:
                        approach_scores[approach_name] = []
                    approach_scores[approach_name].append(approach_result["evaluation"])
        
        # Calculate statistics
        summary = {
            "category": category,
            "total_tests": len(results),
            "approach_wins": approach_wins,
            "win_percentages": {
                approach: (wins / len(results)) * 100 
                for approach, wins in approach_wins.items()
            }
        }
        
        # Add statistical analysis if we have scores
        if approach_scores:
            summary["statistical_analysis"] = self._calculate_approach_statistics(approach_scores)
        
        return summary
    
    def _calculate_approach_statistics(self, approach_scores: Dict[str, List[Dict[str, float]]]) -> Dict[str, Any]:
        """Calculate statistical comparisons between approaches"""
        
        stats = {}
        
        # Compare direct vs conjecture if both available
        if "direct" in approach_scores and "conjecture" in approach_scores:
            direct_scores = [s.get("overall_score", 0.5) for s in approach_scores["direct"]]
            conjecture_scores = [s.get("overall_score", 0.5) for s in approach_scores["conjecture"]]
            
            if len(direct_scores) >= 3 and len(conjecture_scores) >= 3:
                # Simple statistical comparison
                direct_mean = statistics.mean(direct_scores)
                conjecture_mean = statistics.mean(conjecture_scores)
                
                improvement = ((conjecture_mean - direct_mean) / direct_mean) * 100
                
                stats["direct_vs_conjecture"] = {
                    "direct_mean": direct_mean,
                    "conjecture_mean": conjecture_mean,
                    "improvement_percentage": improvement,
                    "sample_size": len(direct_scores)
                }
        
        return stats
    
    async def generate_comprehensive_report(self) -> str:
        """Generate comprehensive A/B testing report"""
        
        report_lines = [
            "# Conjecture A/B Testing Report",
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            "",
            f"**Total Tests Run**: {len(self.test_results)}",
            f"**Categories Tested**: {len(self.category_results)}",
            "",
            "## Category Results",
            ""
        ]
        
        # Add results for each category
        for category in self.test_categories if hasattr(self, 'test_categories') else self.category_results.keys():
            summary = self.generate_category_summary(category)
            
            if summary:
                report_lines.extend([
                    f"### {category.replace('_', ' ').title()}",
                    f"**Total Tests**: {summary['total_tests']}",
                    "**Approach Win Percentages**:",
                ])
                
                for approach, percentage in summary.get("win_percentages", {}).items():
                    report_lines.append(f"- {approach.title()}: {percentage:.1f}%")
                
                # Add statistical analysis if available
                if "statistical_analysis" in summary:
                    stats = summary["statistical_analysis"]
                    if "direct_vs_conjecture" in stats:
                        comp = stats["direct_vs_conjecture"]
                        report_lines.extend([
                            f"**Direct vs Conjecture Comparison**:",
                            f"- Direct Mean: {comp['direct_mean']:.3f}",
                            f"- Conjecture Mean: {comp['conjecture_mean']:.3f}",
                            f"- Improvement: {comp['improvement_percentage']:.1f}%",
                            ""
                        ])
                
                report_lines.append("")
        
        # Add overall conclusions
        report_lines.extend([
            "## Overall Conclusions",
            "",
            "### Key Findings:",
            "1. A/B testing framework successfully implemented",
            "2. Multiple approaches systematically compared",
            "3. Statistical analysis provides validation framework",
            "",
            "### Recommendations:",
            "1. Scale testing to achieve statistical significance",
            "2. Refine prompt engineering based on results",
            "3. Integrate with broader hypothesis validation framework",
            ""
        ])
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_file = self.results_dir / f"ab_testing_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return report_content


async def main():
    """Main function to test the A/B testing framework"""
    
    # Configuration
    config = ABTestConfiguration(
        approaches=["direct", "conjecture"],
        test_models=["ibm/granite-4-h-tiny", "zai-org/GLM-4.6"],
        judge_model="zai-org/GLM-4.6"
    )
    
    # Initialize framework
    framework = ABTestingFramework(config)
    
    # Setup provider configurations
    providers = [
        ProviderConfig(
            url="http://localhost:1234",  # LM Studio
            api_key="",
            model="ibm/granite-4-h-tiny"
        ),
        ProviderConfig(
            url="https://llm.chutes.ai/v1",  # Chutes
            api_key="your-api-key",
            model="zai-org/GLM-4.6"
        )
    ]
    
    # Initialize
    if not await framework.initialize(providers):
        print("Failed to initialize A/B testing framework")
        return
    
    # Create sample test cases
    sample_test_cases = [
        {
            "id": "sample_001",
            "question": "What are the main causes of climate change and what can be done to mitigate them?",
            "expected_answer_type": "comprehensive_analysis",
            "difficulty": "medium"
        },
        {
            "id": "sample_002", 
            "question": "If a car travels 300 miles in 5 hours, what is its average speed?",
            "expected_answer_type": "numerical_solution",
            "difficulty": "easy"
        }
    ]
    
    # Run sample tests
    print("Running sample A/B tests...")
    results = await framework.run_category_tests(sample_test_cases, "sample_tests")
    
    print(f"Completed {len(results)} sample tests")
    
    # Generate report
    report = await framework.generate_comprehensive_report()
    print("\n" + report)
    
    print(f"\nResults saved to: {framework.results_dir}")


if __name__ == "__main__":
    asyncio.run(main())