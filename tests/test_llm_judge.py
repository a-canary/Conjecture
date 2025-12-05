#!/usr/bin/env python3
"""
LLM-as-a-Judge Evaluation System for Conjecture Hypothesis Validation
Uses GLM-4.6 for consistent, high-quality evaluation of test results
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
import re

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.models import Claim, ClaimState, ClaimType
from processing.llm.llm_manager import LLMManager
from config.common import ProviderConfig


@dataclass
class JudgeConfiguration:
    """Configuration for LLM-as-a-Judge evaluation"""
    
    # Judge model settings
    judge_model: str = "zai-org/GLM-4.6"
    temperature: float = 0.1  # Low temperature for consistency
    max_tokens: int = 1500
    
    # Evaluation criteria
    evaluation_criteria: List[str] = None
    criterion_weights: Dict[str, float] = None
    
    # Quality control
    require_justification: bool = True
    min_confidence_threshold: float = 0.3
    max_confidence_threshold: float = 0.95
    
    # Calibration
    calibration_samples: int = 10
    inter_rater_reliability_target: float = 0.8
    
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
class EvaluationResult:
    """Result from LLM-as-a-Judge evaluation"""
    
    evaluation_id: str
    test_id: str
    approach: str
    category: str
    
    # Individual criterion scores
    criterion_scores: Dict[str, float]
    
    # Overall metrics
    overall_score: float
    confidence_level: float
    
    # Qualitative feedback
    detailed_feedback: str
    strengths: List[str]
    weaknesses: List[str]
    improvement_suggestions: List[str]
    
    # Metadata
    judge_model: str
    evaluation_time: float
    timestamp: datetime
    
    # Quality indicators
    evaluation_quality: float  # Self-assessment of evaluation quality
    calibration_score: float   # How well-calibrated the evaluation is


class LLMJudgeSystem:
    """Comprehensive LLM-as-a-Judge evaluation system"""
    
    def __init__(self, config: JudgeConfiguration = None):
        self.config = config or JudgeConfiguration()
        
        # Directory setup
        self.evaluations_dir = Path("tests/results/llm_judge")
        self.prompts_dir = Path("tests/prompts/llm_judge")
        self.calibration_dir = Path("tests/calibration/llm_judge")
        
        for dir_path in [self.evaluations_dir, self.prompts_dir, self.calibration_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.llm_manager = None
        
        # Results storage
        self.evaluations: List[EvaluationResult] = []
        self.calibration_data: Dict[str, List[float]] = {}
        
        # Logging
        self.logger = self._setup_logging()
        
        # Load evaluation prompts
        self.evaluation_prompts = self._load_evaluation_prompts()
        
        # Calibration state
        self.is_calibrated = False
        self.calibration_metrics = {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for LLM judge system"""
        logger = logging.getLogger("llm_judge")
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.evaluations_dir / "judge_evaluations.log")
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
            
            # Test judge model connection
            judge_config = next(
                (p for p in provider_configs if p.model == self.config.judge_model), 
                None
            )
            
            if not judge_config:
                self.logger.error(f"Judge model {self.config.judge_model} not found in provider configs")
                return False
            
            test_result = await self.llm_manager.test_connection(judge_config)
            if not test_result.success:
                self.logger.error(f"Failed to connect to judge model: {test_result.error}")
                return False
            
            self.logger.info("LLM Judge system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM Judge system: {e}")
            return False
    
    def _load_evaluation_prompts(self) -> Dict[str, str]:
        """Load evaluation prompt templates"""
        
        prompts = {
            "main_evaluation": """
You are an expert AI evaluator with deep expertise in assessing language model responses across multiple dimensions. Your task is to evaluate the given answer comprehensively and objectively.

**EVALUATION CONTEXT**
- Category: {category}
- Question: {question}
- Expected Answer Type: {expected_answer_type}
- Approach Used: {approach}
- Difficulty Level: {difficulty}

**ANSWER TO EVALUATE**
{answer}

**EVALUATION CRITERIA**

Please evaluate the answer on each criterion using a 0.0-1.0 scale:

1. **Correctness** (Weight: {correctness_weight})
   - Factual accuracy and correctness
   - Absence of factual errors or misconceptions
   - Alignment with established knowledge

2. **Completeness** (Weight: {completeness_weight})
   - Thoroughness in addressing all aspects of the question
   - Coverage of relevant subtopics and considerations
   - No important aspects omitted

3. **Coherence** (Weight: {coherence_weight})
   - Logical flow and structure
   - Consistency in reasoning and argumentation
   - Clear organization of ideas

4. **Reasoning Quality** (Weight: {reasoning_quality_weight})
   - Depth and rigor of logical reasoning
   - Quality of analytical thinking
   - Soundness of arguments and conclusions

5. **Confidence Calibration** (Weight: {confidence_calibration_weight})
   - Appropriate confidence level relative to answer accuracy
   - Avoidance of overconfidence when uncertain
   - Appropriate expression of uncertainty

6. **Efficiency** (Weight: {efficiency_weight})
   - Conciseness without sacrificing completeness
   - Avoidance of unnecessary verbosity
   - Direct and focused response

7. **Hallucination Reduction** (Weight: {hallucination_reduction_weight})
   - Grounding in provided information or established facts
   - Absence of fabricated or unsupported claims
   - Appropriate acknowledgment of limitations

**EVALUATION FORMAT**

Provide your evaluation in this JSON format:
```json
{{
  "criterion_scores": {{
    "correctness": 0.0-1.0,
    "completeness": 0.0-1.0,
    "coherence": 0.0-1.0,
    "reasoning_quality": 0.0-1.0,
    "confidence_calibration": 0.0-1.0,
    "efficiency": 0.0-1.0,
    "hallucination_reduction": 0.0-1.0
  }},
  "overall_score": 0.0-1.0,
  "confidence_level": 0.0-1.0,
  "detailed_feedback": "comprehensive explanation of the evaluation",
  "strengths": ["list of specific strengths"],
  "weaknesses": ["list of specific weaknesses"],
  "improvement_suggestions": ["list of actionable suggestions"],
  "evaluation_quality": 0.0-1.0,
  "calibration_score": 0.0-1.0
}}
```

Please ensure your evaluation is:
- Objective and unbiased
- Consistent with the evaluation criteria
- Supported by specific evidence from the answer
- Helpful for improving future responses
""",
            
            "calibration_evaluation": """
You are calibrating an AI evaluation system. Compare your evaluation with a reference evaluation and assess consistency.

**YOUR EVALUATION**
{your_evaluation}

**REFERENCE EVALUATION**
{reference_evaluation}

**CALIBRATION TASK**
1. Compare the criterion scores between your evaluation and the reference
2. Calculate the average absolute difference
3. Assess the consistency of your qualitative feedback
4. Identify any systematic biases in your evaluation

Provide calibration assessment in this JSON format:
```json
{{
  "score_differences": {{
    "correctness": difference,
    "completeness": difference,
    "coherence": difference,
    "reasoning_quality": difference,
    "confidence_calibration": difference,
    "efficiency": difference,
    "hallucination_reduction": difference
  }},
  "average_difference": 0.0-1.0,
  "consistency_score": 0.0-1.0,
  "identified_biases": ["list of potential biases"],
  "calibration_quality": 0.0-1.0,
  "improvement_notes": "notes for improving consistency"
}}
```
""",
            
            "quality_assessment": """
You are assessing the quality of an AI evaluation. Evaluate how well the evaluation meets quality standards.

**EVALUATION TO ASSESS**
{evaluation}

**QUALITY CRITERIA**
1. Objectivity and fairness
2. Consistency with criteria
3. Evidence-based reasoning
4. Constructive feedback
5. Appropriate confidence levels

Provide quality assessment in this JSON format:
```json
{{
  "objectivity_score": 0.0-1.0,
  "consistency_score": 0.0-1.0,
  "evidence_quality": 0.0-1.0,
  "feedback_constructiveness": 0.0-1.0,
  "confidence_appropriateness": 0.0-1.0,
  "overall_quality": 0.0-1.0,
  "quality_issues": ["list of identified issues"],
  "improvement_recommendations": ["list of recommendations"]
}}
```
"""
        }
        
        # Save prompts to files
        for prompt_name, prompt_content in prompts.items():
            prompt_file = self.prompts_dir / f"{prompt_name}.txt"
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(prompt_content)
        
        return prompts
    
    async def evaluate_response(
        self,
        test_id: str,
        approach: str,
        category: str,
        question: str,
        answer: str,
        test_case: Dict[str, Any]
    ) -> EvaluationResult:
        """Evaluate a single response using LLM-as-a-Judge"""
        
        evaluation_id = str(uuid.uuid4())
        start_time = time.time()
        
        self.logger.info(f"Evaluating response {evaluation_id} for test {test_id}")
        
        # Prepare evaluation prompt
        evaluation_prompt = self._prepare_evaluation_prompt(
            category, question, answer, approach, test_case
        )
        
        try:
            # Get evaluation from judge model
            response = await self.llm_manager.generate_response(
                prompt=evaluation_prompt,
                model=self.config.judge_model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            evaluation_time = time.time() - start_time
            
            # Parse evaluation response
            evaluation_data = self._parse_evaluation_response(response.content)
            
            # Create evaluation result
            result = EvaluationResult(
                evaluation_id=evaluation_id,
                test_id=test_id,
                approach=approach,
                category=category,
                criterion_scores=evaluation_data.get("criterion_scores", {}),
                overall_score=evaluation_data.get("overall_score", 0.5),
                confidence_level=evaluation_data.get("confidence_level", 0.5),
                detailed_feedback=evaluation_data.get("detailed_feedback", ""),
                strengths=evaluation_data.get("strengths", []),
                weaknesses=evaluation_data.get("weaknesses", []),
                improvement_suggestions=evaluation_data.get("improvement_suggestions", []),
                judge_model=self.config.judge_model,
                evaluation_time=evaluation_time,
                timestamp=datetime.utcnow(),
                evaluation_quality=evaluation_data.get("evaluation_quality", 0.5),
                calibration_score=evaluation_data.get("calibration_score", 0.5)
            )
            
            # Validate evaluation quality
            await self._validate_evaluation_quality(result)
            
            # Store evaluation
            self.evaluations.append(result)
            
            # Save to file
            await self._save_evaluation(result)
            
            self.logger.info(f"Completed evaluation {evaluation_id} with overall score: {result.overall_score:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating response {evaluation_id}: {e}")
            
            # Return default evaluation
            return self._create_default_evaluation(evaluation_id, test_id, approach, category, evaluation_time)
    
    def _prepare_evaluation_prompt(
        self,
        category: str,
        question: str,
        answer: str,
        approach: str,
        test_case: Dict[str, Any]
    ) -> str:
        """Prepare the evaluation prompt with all necessary context"""
        
        # Extract test case details
        expected_answer_type = test_case.get("expected_answer_type", "comprehensive answer")
        difficulty = test_case.get("difficulty", "medium")
        reasoning_requirements = test_case.get("reasoning_requirements", [])
        
        # Get criterion weights
        weights = self.config.criterion_weights
        
        # Format the main evaluation prompt
        prompt = self.evaluation_prompts["main_evaluation"].format(
            category=category,
            question=question,
            expected_answer_type=expected_answer_type,
            approach=approach,
            difficulty=difficulty,
            answer=answer,
            correctness_weight=weights.get("correctness", 1.0),
            completeness_weight=weights.get("completeness", 1.0),
            coherence_weight=weights.get("coherence", 1.0),
            reasoning_quality_weight=weights.get("reasoning_quality", 1.0),
            confidence_calibration_weight=weights.get("confidence_calibration", 1.0),
            efficiency_weight=weights.get("efficiency", 1.0),
            hallucination_reduction_weight=weights.get("hallucination_reduction", 1.0)
        )
        
        # Add category-specific guidance
        category_guidance = self._get_category_guidance(category)
        if category_guidance:
            prompt += f"\n\n**CATEGORY-SPECIFIC GUIDANCE**\n{category_guidance}"
        
        return prompt
    
    def _get_category_guidance(self, category: str) -> str:
        """Get category-specific evaluation guidance"""
        
        guidance_map = {
            "complex_reasoning": """
Focus on:
- Logical structure and validity of arguments
- Proper handling of complex multi-step reasoning
- Identification and resolution of logical fallacies
- Quality of inferential reasoning
""",
            
            "mathematical_reasoning": """
Focus on:
- Accuracy of mathematical calculations
- Clarity of problem-solving approach
- Proper use of mathematical concepts and formulas
- Validation of final answers
""",
            
            "context_compression": """
Focus on:
- Effective extraction of relevant information
- Proper prioritization of key details
- Ability to synthesize information from multiple sources
- Maintaining accuracy while being concise
""",
            
            "evidence_evaluation": """
Focus on:
- Proper weighing of evidence strength
- Identification of biases and limitations
- Logical synthesis of conflicting information
- Evidence-based conclusions
""",
            
            "task_decomposition": """
Focus on:
- Quality of problem breakdown
- Logical sequencing of steps
- Completeness of planning
- Feasibility and practicality of solutions
""",
            
            "coding_tasks": """
Focus on:
- Code correctness and efficiency
- Proper handling of edge cases
- Code readability and maintainability
- Algorithm design and optimization
"""
        }
        
        return guidance_map.get(category, "")
    
    def _parse_evaluation_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the evaluation response from the LLM"""
        
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                evaluation_data = json.loads(json_str)
                return evaluation_data
            else:
                # Try to find JSON without code blocks
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    evaluation_data = json.loads(json_str)
                    return evaluation_data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")
        except Exception as e:
            self.logger.error(f"Error parsing evaluation response: {e}")
        
        # Fallback: try to extract individual scores with regex
        scores = {}
        for criterion in self.config.evaluation_criteria:
            pattern = f'"{criterion}"\s*:\s*([0-9.]+)'
            match = re.search(pattern, response_text.lower())
            scores[criterion] = float(match.group(1)) if match else 0.5
        
        # Extract overall score
        overall_pattern = '"overall_score"\s*:\s*([0-9.]+)'
        overall_match = re.search(overall_pattern, response_text.lower())
        overall_score = float(overall_match.group(1)) if overall_match else 0.5
        
        return {
            "criterion_scores": scores,
            "overall_score": overall_score,
            "confidence_level": 0.5,
            "detailed_feedback": "Failed to parse structured evaluation",
            "strengths": [],
            "weaknesses": [],
            "improvement_suggestions": [],
            "evaluation_quality": 0.3,
            "calibration_score": 0.3
        }
    
    def _create_default_evaluation(
        self,
        evaluation_id: str,
        test_id: str,
        approach: str,
        category: str,
        evaluation_time: float
    ) -> EvaluationResult:
        """Create a default evaluation when parsing fails"""
        
        default_scores = {criterion: 0.5 for criterion in self.config.evaluation_criteria}
        
        return EvaluationResult(
            evaluation_id=evaluation_id,
            test_id=test_id,
            approach=approach,
            category=category,
            criterion_scores=default_scores,
            overall_score=0.5,
            confidence_level=0.5,
            detailed_feedback="Evaluation failed - using default scores",
            strengths=["N/A"],
            weaknesses=["Evaluation parsing failed"],
            improvement_suggestions=["Retry evaluation"],
            judge_model=self.config.judge_model,
            evaluation_time=evaluation_time,
            timestamp=datetime.utcnow(),
            evaluation_quality=0.1,
            calibration_score=0.1
        )
    
    async def _validate_evaluation_quality(self, result: EvaluationResult):
        """Validate the quality of an evaluation"""
        
        # Check for score consistency
        scores = list(result.criterion_scores.values())
        if scores:
            score_std = statistics.stdev(scores)
            if score_std > 0.4:  # High variance might indicate inconsistency
                self.logger.warning(f"High score variance in evaluation {result.evaluation_id}: {score_std:.3f}")
        
        # Check confidence calibration
        if abs(result.confidence_level - result.overall_score) > 0.3:
            self.logger.warning(f"Poor confidence calibration in evaluation {result.evaluation_id}")
        
        # Check for missing criteria
        missing_criteria = set(self.config.evaluation_criteria) - set(result.criterion_scores.keys())
        if missing_criteria:
            self.logger.warning(f"Missing evaluation criteria {missing_criteria} in {result.evaluation_id}")
    
    async def _save_evaluation(self, result: EvaluationResult):
        """Save evaluation result to file"""
        
        # Convert to dictionary for JSON serialization
        result_dict = asdict(result)
        result_dict["timestamp"] = result.timestamp.isoformat()
        
        # Save to file
        filename = f"evaluation_{result.evaluation_id}.json"
        filepath = self.evaluations_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save evaluation {result.evaluation_id}: {e}")
    
    async def calibrate_judge(self, calibration_samples: List[Dict[str, Any]]):
        """Calibrate the judge system using reference evaluations"""
        
        self.logger.info(f"Starting judge calibration with {len(calibration_samples)} samples")
        
        calibration_results = []
        
        for i, sample in enumerate(calibration_samples):
            self.logger.info(f"Processing calibration sample {i+1}/{len(calibration_samples)}")
            
            try:
                # Evaluate the sample
                evaluation = await self.evaluate_response(
                    test_id=sample.get("test_id", f"calib_{i}"),
                    approach=sample.get("approach", "unknown"),
                    category=sample.get("category", "unknown"),
                    question=sample.get("question", ""),
                    answer=sample.get("answer", ""),
                    test_case=sample.get("test_case", {})
                )
                
                # Compare with reference evaluation
                reference_evaluation = sample.get("reference_evaluation", {})
                comparison = await self._compare_with_reference(evaluation, reference_evaluation)
                
                calibration_results.append({
                    "evaluation": evaluation,
                    "reference": reference_evaluation,
                    "comparison": comparison
                })
                
            except Exception as e:
                self.logger.error(f"Error in calibration sample {i+1}: {e}")
                continue
        
        # Calculate calibration metrics
        self.calibration_metrics = self._calculate_calibration_metrics(calibration_results)
        self.is_calibrated = True
        
        # Save calibration results
        await self._save_calibration_results(calibration_results, self.calibration_metrics)
        
        self.logger.info(f"Calibration completed. Quality score: {self.calibration_metrics.get('overall_quality', 0):.3f}")
    
    async def _compare_with_reference(
        self, 
        evaluation: EvaluationResult, 
        reference: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare evaluation with reference evaluation"""
        
        comparison = {
            "score_differences": {},
            "average_difference": 0.0,
            "consistency_score": 0.0
        }
        
        # Calculate score differences
        total_diff = 0.0
        count = 0
        
        for criterion in self.config.evaluation_criteria:
            eval_score = evaluation.criterion_scores.get(criterion, 0.5)
            ref_score = reference.get("criterion_scores", {}).get(criterion, 0.5)
            
            diff = abs(eval_score - ref_score)
            comparison["score_differences"][criterion] = diff
            total_diff += diff
            count += 1
        
        # Calculate average difference
        comparison["average_difference"] = total_diff / count if count > 0 else 0.0
        
        # Calculate consistency score (inverse of average difference)
        comparison["consistency_score"] = max(0.0, 1.0 - comparison["average_difference"])
        
        return comparison
    
    def _calculate_calibration_metrics(
        self, 
        calibration_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate overall calibration metrics"""
        
        if not calibration_results:
            return {"overall_quality": 0.0, "sample_count": 0}
        
        # Collect consistency scores
        consistency_scores = [r["comparison"]["consistency_score"] for r in calibration_results]
        
        # Calculate metrics
        metrics = {
            "sample_count": len(calibration_results),
            "average_consistency": statistics.mean(consistency_scores) if consistency_scores else 0.0,
            "consistency_std": statistics.stdev(consistency_scores) if len(consistency_scores) > 1 else 0.0,
            "min_consistency": min(consistency_scores) if consistency_scores else 0.0,
            "max_consistency": max(consistency_scores) if consistency_scores else 0.0,
            "overall_quality": statistics.mean(consistency_scores) if consistency_scores else 0.0
        }
        
        # Assess if calibration meets target
        metrics["meets_target"] = metrics["average_consistency"] >= self.config.inter_rater_reliability_target
        
        return metrics
    
    async def _save_calibration_results(
        self, 
        calibration_results: List[Dict[str, Any]], 
        metrics: Dict[str, Any]
    ):
        """Save calibration results to file"""
        
        calibration_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "configuration": asdict(self.config),
            "metrics": metrics,
            "sample_results": calibration_results
        }
        
        # Save to file
        filename = f"judge_calibration_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.calibration_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(calibration_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save calibration results: {e}")
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get statistics about all evaluations"""
        
        if not self.evaluations:
            return {"total_evaluations": 0}
        
        # Collect scores by criterion
        criterion_scores = {criterion: [] for criterion in self.config.evaluation_criteria}
        overall_scores = []
        confidence_levels = []
        evaluation_qualities = []
        
        for eval_result in self.evaluations:
            overall_scores.append(eval_result.overall_score)
            confidence_levels.append(eval_result.confidence_level)
            evaluation_qualities.append(eval_result.evaluation_quality)
            
            for criterion, score in eval_result.criterion_scores.items():
                if criterion in criterion_scores:
                    criterion_scores[criterion].append(score)
        
        # Calculate statistics
        stats = {
            "total_evaluations": len(self.evaluations),
            "overall_statistics": self._calculate_score_statistics(overall_scores),
            "confidence_statistics": self._calculate_score_statistics(confidence_levels),
            "quality_statistics": self._calculate_score_statistics(evaluation_qualities),
            "criterion_statistics": {}
        }
        
        for criterion, scores in criterion_scores.items():
            stats["criterion_statistics"][criterion] = self._calculate_score_statistics(scores)
        
        # Add calibration status
        stats["calibration_status"] = {
            "is_calibrated": self.is_calibrated,
            "calibration_metrics": self.calibration_metrics
        }
        
        return stats
    
    def _calculate_score_statistics(self, scores: List[float]) -> Dict[str, float]:
        """Calculate basic statistics for a list of scores"""
        
        if not scores:
            return {}
        
        return {
            "count": len(scores),
            "mean": statistics.mean(scores),
            "median": statistics.median(scores),
            "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "min": min(scores),
            "max": max(scores),
            "range": max(scores) - min(scores)
        }


async def main():
    """Main function to test the LLM judge system"""
    
    # Configuration
    config = JudgeConfiguration(
        judge_model="zai-org/GLM-4.6",
        temperature=0.1
    )
    
    # Initialize judge system
    judge = LLMJudgeSystem(config)
    
    # Setup provider configurations
    providers = [
        ProviderConfig(
            url="https://llm.chutes.ai/v1",  # Chutes
            api_key="your-api-key",
            model="zai-org/GLM-4.6"
        )
    ]
    
    # Initialize
    if not await judge.initialize(providers):
        print("Failed to initialize LLM judge system")
        return
    
    # Test evaluation
    print("Testing LLM judge evaluation...")
    
    test_evaluation = await judge.evaluate_response(
        test_id="test_001",
        approach="conjecture",
        category="complex_reasoning",
        question="What are the main causes of climate change?",
        answer="Climate change is primarily caused by greenhouse gas emissions from human activities, including burning fossil fuels, deforestation, and industrial processes. These activities increase atmospheric concentrations of CO2, methane, and other greenhouse gases, leading to global warming.",
        test_case={
            "expected_answer_type": "comprehensive_analysis",
            "difficulty": "medium"
        }
    )
    
    print(f"Evaluation completed with overall score: {test_evaluation.overall_score:.3f}")
    print(f"Detailed feedback: {test_evaluation.detailed_feedback}")
    
    # Get statistics
    stats = judge.get_evaluation_statistics()
    print(f"\nEvaluation Statistics:")
    print(f"Total evaluations: {stats['total_evaluations']}")
    print(f"Overall mean score: {stats['overall_statistics'].get('mean', 0):.3f}")
    
    print(f"\nResults saved to: {judge.evaluations_dir}")


if __name__ == "__main__":
    asyncio.run(main())