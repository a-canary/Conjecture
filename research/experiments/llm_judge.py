#!/usr/bin/env python3
"""
LLM-as-a-Judge Evaluation System
Uses GLM-4.6 to evaluate model responses with structured rubrics
"""

import asyncio
import json
import re
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

# Add src to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from processing.llm.llm_manager import LLMManager
from config.common import ProviderConfig


class EvaluationCriterion(str, Enum):
    """Evaluation criteria for LLM-as-a-Judge"""
    CORRECTNESS = "correctness"
    COMPLETENESS = "completeness" 
    COHERENCE = "coherence"
    RELEVANCE = "relevance"
    DEPTH = "depth"
    CLARITY = "clarity"
    REASONING_QUALITY = "reasoning_quality"
    CONFIDENCE_CALIBRATION = "confidence_calibration"
    EFFICIENCY = "efficiency"
    CREATIVITY = "creativity"


class JudgeMode(str, Enum):
    """Judging modes"""
    SINGLE_JUDGE = "single_judge"
    MULTIPLE_JUDGE = "multiple_judge"
    CONSENSUS = "consensus"
    ADVERSARIAL = "adversarial"


@dataclass
class EvaluationRubric:
    """Rubric for evaluating responses"""
    criterion: EvaluationCriterion
    description: str
    score_levels: Dict[int, str]  # score -> description
    weight: float = 1.0
    evaluation_prompt: str = ""


@dataclass
class JudgeEvaluation:
    """Single evaluation from a judge"""
    judge_id: str
    criterion: EvaluationCriterion
    score: float  # 0.0 to 1.0
    reasoning: str
    confidence: float  # 0.0 to 1.0
    evaluation_time_seconds: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ConsensusEvaluation:
    """Consensus evaluation from multiple judges"""
    criterion: EvaluationCriterion
    final_score: float
    individual_evaluations: List[JudgeEvaluation]
    consensus_strength: float  # 0.0 to 1.0
    disagreement_analysis: str


class LLMJudge:
    """LLM-as-a-Judge evaluation system"""
    
    def __init__(self, llm_manager: LLMManager, judge_model: str = "chutes:zai-org/GLM-4.6"):
        self.llm_manager = llm_manager
        self.judge_model = judge_model
        self.rubrics = self._initialize_rubrics()
        self.evaluation_history: List[JudgeEvaluation] = []
        
    def _initialize_rubrics(self) -> Dict[EvaluationCriterion, EvaluationRubric]:
        """Initialize evaluation rubrics for all criteria"""
        rubrics = {}
        
        # Correctness rubric
        rubrics[EvaluationCriterion.CORRECTNESS] = EvaluationRubric(
            criterion=EvaluationCriterion.CORRECTNESS,
            description="Factual accuracy and correctness of the response",
            score_levels={
                0: "Completely incorrect or contains major factual errors",
                0.25: "Mostly incorrect with some accurate elements",
                0.5: "Partially correct, mixture of accurate and inaccurate information",
                0.75: "Mostly correct with minor inaccuracies",
                1.0: "Completely correct and factually accurate"
            },
            weight=1.5  # Higher weight for correctness
        )
        
        # Completeness rubric
        rubrics[EvaluationCriterion.COMPLETENESS] = EvaluationRubric(
            criterion=EvaluationCriterion.COMPLETENESS,
            description="How completely the response addresses all aspects of the question",
            score_levels={
                0: "Fails to address the question or major aspects missing",
                0.25: "Addresses only minor aspects, major components missing",
                0.5: "Addresses some key aspects but incomplete overall",
                0.75: "Addresses most aspects with minor omissions",
                1.0: "Completely addresses all aspects of the question"
            },
            weight=1.0
        )
        
        # Coherence rubric
        rubrics[EvaluationCriterion.COHERENCE] = EvaluationRubric(
            criterion=EvaluationCriterion.COHERENCE,
            description="Logical flow, consistency, and structural coherence",
            score_levels={
                0: "Incoherent, contradictory, or completely disorganized",
                0.25: "Poorly organized with significant logical gaps",
                0.5: "Somewhat coherent but with organizational issues",
                0.75: "Well-organized with minor logical issues",
                1.0: "Perfectly coherent, logical, and well-structured"
            },
            weight=1.0
        )
        
        # Reasoning Quality rubric
        rubrics[EvaluationCriterion.REASONING_QUALITY] = EvaluationRubric(
            criterion=EvaluationCriterion.REASONING_QUALITY,
            description="Quality of logical reasoning and argumentation",
            score_levels={
                0: "No reasoning or completely flawed logic",
                0.25: "Weak reasoning with major logical fallacies",
                0.5: "Adequate reasoning but with some gaps or weaknesses",
                0.75: "Strong reasoning with minor logical issues",
                1.0: "Excellent reasoning, rigorous and insightful"
            },
            weight=1.2
        )
        
        # Depth rubric
        rubrics[EvaluationCriterion.DEPTH] = EvaluationRubric(
            criterion=EvaluationCriterion.DEPTH,
            description="Depth of analysis and insight provided",
            score_levels={
                0: "Superficial or shallow response",
                0.25: "Minimal depth, only surface-level analysis",
                0.5: "Moderate depth with some insights",
                0.75: "Good depth with meaningful insights",
                1.0: "Exceptional depth with profound insights"
            },
            weight=0.8
        )
        
        # Clarity rubric
        rubrics[EvaluationCriterion.CLARITY] = EvaluationRubric(
            criterion=EvaluationCriterion.CLARITY,
            description="Clarity of expression and ease of understanding",
            score_levels={
                0: "Unclear, confusing, or incomprehensible",
                0.25: "Difficult to understand with clarity issues",
                0.5: "Generally understandable but some clarity problems",
                0.75: "Clear with minor expression issues",
                1.0: "Perfectly clear and easy to understand"
            },
            weight=0.6
        )
        
        # Confidence Calibration rubric
        rubrics[EvaluationCriterion.CONFIDENCE_CALIBRATION] = EvaluationRubric(
            criterion=EvaluationCriterion.CONFIDENCE_CALIBRATION,
            description="How well the model's confidence matches its actual accuracy",
            score_levels={
                0: "Completely misaligned confidence (overconfident when wrong, underconfident when right)",
                0.25: "Poorly calibrated confidence",
                0.5: "Somewhat calibrated confidence with notable misalignments",
                0.75: "Well-calibrated confidence with minor misalignments",
                1.0: "Perfectly calibrated confidence"
            },
            weight=1.0
        )
        
        # Efficiency rubric
        rubrics[EvaluationCriterion.EFFICIENCY] = EvaluationRubric(
            criterion=EvaluationCriterion.EFFICIENCY,
            description="Efficiency and conciseness of the response",
            score_levels={
                0: "Extremely verbose, inefficient, or incomplete due to brevity",
                0.25: "Inefficient with significant verbosity or important omissions",
                0.5: "Moderately efficient with some verbosity or minor omissions",
                0.75: "Efficient with minor verbosity issues",
                1.0: "Perfectly efficient, concise yet complete"
            },
            weight=0.5
        )
        
        return rubrics
    
    async def evaluate_response(self, 
                              question: str,
                              response: str,
                              ground_truth: Optional[str] = None,
                              context: Optional[str] = None,
                              criteria: List[EvaluationCriterion] = None,
                              mode: JudgeMode = JudgeMode.SINGLE_JUDGE) -> Dict[EvaluationCriterion, Any]:
        """
        Evaluate a model response using LLM-as-a-Judge
        
        Args:
            question: The original question/task
            response: The model's response to evaluate
            ground_truth: Optional ground truth answer
            context: Optional context provided to the model
            criteria: List of criteria to evaluate (default: all)
            mode: Judging mode to use
            
        Returns:
            Dictionary mapping criteria to evaluation results
        """
        if criteria is None:
            criteria = list(EvaluationCriterion)
        
        results = {}
        
        for criterion in criteria:
            if mode == JudgeMode.SINGLE_JUDGE:
                evaluation = await self._single_judge_evaluation(
                    question, response, ground_truth, context, criterion
                )
                results[criterion] = evaluation
                
            elif mode == JudgeMode.MULTIPLE_JUDGE:
                evaluations = await self._multiple_judge_evaluation(
                    question, response, ground_truth, context, criterion
                )
                results[criterion] = evaluations
                
            elif mode == JudgeMode.CONSENSUS:
                consensus = await self._consensus_evaluation(
                    question, response, ground_truth, context, criterion
                )
                results[criterion] = consensus
        
        return results
    
    async def _single_judge_evaluation(self,
                                     question: str,
                                     response: str,
                                     ground_truth: Optional[str],
                                     context: Optional[str],
                                     criterion: EvaluationCriterion) -> JudgeEvaluation:
        """Single judge evaluation"""
        rubric = self.rubrics[criterion]
        prompt = self._create_evaluation_prompt(
            question, response, ground_truth, context, rubric
        )
        
        start_time = time.time()
        judge_response = await self.llm_manager.generate_response(
            prompt=prompt,
            model=self.judge_model,
            max_tokens=1000,
            temperature=0.1  # Low temperature for consistent evaluation
        )
        evaluation_time = time.time() - start_time
        
        # Parse the judge response
        score, reasoning, confidence = self._parse_judge_response(judge_response)
        
        evaluation = JudgeEvaluation(
            judge_id=self.judge_model,
            criterion=criterion,
            score=score,
            reasoning=reasoning,
            confidence=confidence,
            evaluation_time_seconds=evaluation_time,
            metadata={
                "question_length": len(question),
                "response_length": len(response),
                "has_ground_truth": ground_truth is not None,
                "has_context": context is not None
            }
        )
        
        self.evaluation_history.append(evaluation)
        return evaluation
    
    async def _multiple_judge_evaluation(self,
                                       question: str,
                                       response: str,
                                       ground_truth: Optional[str],
                                       context: Optional[str],
                                       criterion: EvaluationCriterion,
                                       num_judges: int = 3) -> List[JudgeEvaluation]:
        """Multiple judge evaluation with different temperature settings"""
        evaluations = []
        
        # Use different temperature settings for diversity
        temperatures = [0.1, 0.3, 0.5]
        
        for i in range(min(num_judges, len(temperatures))):
            temp = temperatures[i]
            judge_id = f"{self.judge_model}_temp{temp}"
            
            rubric = self.rubrics[criterion]
            prompt = self._create_evaluation_prompt(
                question, response, ground_truth, context, rubric
            )
            
            start_time = time.time()
            judge_response = await self.llm_manager.generate_response(
                prompt=prompt,
                model=self.judge_model,
                max_tokens=1000,
                temperature=temp
            )
            evaluation_time = time.time() - start_time
            
            score, reasoning, confidence = self._parse_judge_response(judge_response)
            
            evaluation = JudgeEvaluation(
                judge_id=judge_id,
                criterion=criterion,
                score=score,
                reasoning=reasoning,
                confidence=confidence,
                evaluation_time_seconds=evaluation_time,
                metadata={"temperature": temp}
            )
            
            evaluations.append(evaluation)
            self.evaluation_history.append(evaluation)
        
        return evaluations
    
    async def _consensus_evaluation(self,
                                  question: str,
                                  response: str,
                                  ground_truth: Optional[str],
                                  context: Optional[str],
                                  criterion: EvaluationCriterion) -> ConsensusEvaluation:
        """Consensus evaluation from multiple judges"""
        # Get multiple evaluations
        individual_evals = await self._multiple_judge_evaluation(
            question, response, ground_truth, context, criterion
        )
        
        # Calculate consensus
        scores = [e.score for e in individual_evals]
        final_score = sum(scores) / len(scores)
        
        # Calculate consensus strength (inverse of variance)
        if len(scores) > 1:
            variance = sum((s - final_score) ** 2 for s in scores) / len(scores)
            consensus_strength = max(0.0, 1.0 - (variance * 4))  # Scale variance to 0-1
        else:
            consensus_strength = 1.0
        
        # Analyze disagreement
        disagreement_analysis = self._analyze_disagreement(individual_evals)
        
        consensus = ConsensusEvaluation(
            criterion=criterion,
            final_score=final_score,
            individual_evaluations=individual_evals,
            consensus_strength=consensus_strength,
            disagreement_analysis=disagreement_analysis
        )
        
        return consensus
    
    def _create_evaluation_prompt(self,
                                question: str,
                                response: str,
                                ground_truth: Optional[str],
                                context: Optional[str],
                                rubric: EvaluationRubric) -> str:
        """Create evaluation prompt for the judge"""
        
        # Build score level descriptions
        score_descriptions = []
        for score, description in rubric.score_levels.items():
            score_descriptions.append(f"Score {score}: {description}")
        
        prompt = f"""You are an expert evaluator assessing AI model responses. Your task is to evaluate a response based on the criterion: {rubric.criterion.value.upper()}.

**Criterion Description:**
{rubric.description}

**Scoring Rubric:**
{chr(10).join(score_descriptions)}

**Question/Task:**
{question}

**Context Provided to Model:**
{context if context else "No context provided."}

**Model Response:**
{response}

**Ground Truth (Reference Answer):**
{ground_truth if ground_truth else "No ground truth provided."}

**Evaluation Instructions:**
1. Carefully read and understand the question and model response
2. Compare the response to the ground truth (if provided)
3. Apply the scoring rubric above
4. Provide your evaluation in the following format:

SCORE: [0.0-1.0]
REASONING: [Detailed explanation of your evaluation, referencing specific aspects of the response]
CONFIDENCE: [0.0-1.0 for your confidence in this evaluation]

Be objective, thorough, and fair in your evaluation. Focus specifically on the {rubric.criterion.value} criterion."""
        
        return prompt
    
    def _parse_judge_response(self, response: str) -> Tuple[float, str, float]:
        """Parse judge response into score, reasoning, and confidence"""
        try:
            # Extract score
            score_match = re.search(r'SCORE:\s*([0-9]*\.?[0-9]+)', response, re.IGNORECASE)
            if score_match:
                score = float(score_match.group(1))
                score = max(0.0, min(1.0, score))  # Clamp to 0-1
            else:
                score = 0.5  # Default if not found
            
            # Extract reasoning
            reasoning_match = re.search(r'REASONING:\s*(.*?)(?=CONFIDENCE:|$)', response, re.DOTALL | re.IGNORECASE)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
            else:
                reasoning = "Reasoning not clearly structured in response."
            
            # Extract confidence
            confidence_match = re.search(r'CONFIDENCE:\s*([0-9]*\.?[0-9]+)', response, re.IGNORECASE)
            if confidence_match:
                confidence = float(confidence_match.group(1))
                confidence = max(0.0, min(1.0, confidence))  # Clamp to 0-1
            else:
                confidence = 0.5  # Default if not found
            
            return score, reasoning, confidence
            
        except Exception as e:
            # Fallback parsing
            return 0.5, f"Parsing error: {str(e)}. Original response: {response[:200]}...", 0.3
    
    def _analyze_disagreement(self, evaluations: List[JudgeEvaluation]) -> str:
        """Analyze disagreement between multiple judges"""
        if len(evaluations) <= 1:
            return "No disagreement analysis (single evaluation)."
        
        scores = [e.score for e in evaluations]
        score_range = max(scores) - min(scores)
        
        if score_range < 0.1:
            return "High agreement among judges (score range < 0.1)."
        elif score_range < 0.3:
            return "Moderate agreement among judges (score range < 0.3)."
        else:
            return "Significant disagreement among judges (score range >= 0.3)."
    
    async def batch_evaluate(self, 
                           test_cases: List[Dict[str, Any]],
                           criteria: List[EvaluationCriterion] = None,
                           mode: JudgeMode = JudgeMode.SINGLE_JUDGE) -> List[Dict[str, Any]]:
        """Evaluate multiple test cases in batch"""
        results = []
        
        for i, test_case in enumerate(test_cases):
            print(f"Evaluating test case {i+1}/{len(test_cases)}...")
            
            evaluation_result = await self.evaluate_response(
                question=test_case.get('question', ''),
                response=test_case.get('response', ''),
                ground_truth=test_case.get('ground_truth'),
                context=test_case.get('context'),
                criteria=criteria,
                mode=mode
            )
            
            results.append({
                'test_case_id': test_case.get('id', f'test_{i}'),
                'evaluations': evaluation_result,
                'test_case_metadata': test_case.get('metadata', {})
            })
        
        return results
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get statistics about evaluations performed"""
        if not self.evaluation_history:
            return {"total_evaluations": 0}
        
        stats = {
            "total_evaluations": len(self.evaluation_history),
            "criteria_evaluated": {},
            "average_scores": {},
            "average_confidence": 0.0,
            "average_evaluation_time": 0.0
        }
        
        # Group by criterion
        for criterion in EvaluationCriterion:
            criterion_evals = [e for e in self.evaluation_history if e.criterion == criterion]
            if criterion_evals:
                scores = [e.score for e in criterion_evals]
                confidences = [e.confidence for e in criterion_evals]
                times = [e.evaluation_time_seconds for e in criterion_evals]
                
                stats["criteria_evaluated"][criterion.value] = len(criterion_evals)
                stats["average_scores"][criterion.value] = sum(scores) / len(scores)
        
        # Overall averages
        stats["average_confidence"] = sum(e.confidence for e in self.evaluation_history) / len(self.evaluation_history)
        stats["average_evaluation_time"] = sum(e.evaluation_time_seconds for e in self.evaluation_history) / len(self.evaluation_history)
        
        return stats
    
    def save_evaluation_history(self, filepath: str):
        """Save evaluation history to file"""
        history_data = [asdict(eval) for eval in self.evaluation_history]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
    
    def load_evaluation_history(self, filepath: str):
        """Load evaluation history from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            history_data = json.load(f)
        
        self.evaluation_history = [JudgeEvaluation(**data) for data in history_data]


async def main():
    """Example usage of the LLM Judge system"""
    from config.common import ProviderConfig
    
    # Setup LLM manager
    providers = [
        ProviderConfig(
            url="https://llm.chutes.ai/v1",
            api_key="your-api-key",
            model="zai-org/GLM-4.6"
        )
    ]
    
    llm_manager = LLMManager(providers)
    judge = LLMJudge(llm_manager)
    
    # Example evaluation
    question = "What are the main causes of climate change?"
    response = "Climate change is primarily caused by human activities, particularly the emission of greenhouse gases like carbon dioxide from burning fossil fuels, deforestation, and industrial processes. These activities trap heat in the atmosphere, leading to global warming."
    ground_truth = "The main causes of climate change include greenhouse gas emissions from fossil fuel combustion, deforestation, industrial processes, agriculture, and transportation. Natural factors like volcanic activity and solar variations also play a role but are minor compared to human influences."
    
    # Evaluate the response
    results = await judge.evaluate_response(
        question=question,
        response=response,
        ground_truth=ground_truth,
        criteria=[EvaluationCriterion.CORRECTNESS, EvaluationCriterion.COMPLETENESS, EvaluationCriterion.COHERENCE],
        mode=JudgeMode.CONSENSUS
    )
    
    # Print results
    print("Evaluation Results:")
    for criterion, result in results.items():
        print(f"\n{criterion.value.upper()}:")
        if isinstance(result, ConsensusEvaluation):
            print(f"  Final Score: {result.final_score:.3f}")
            print(f"  Consensus Strength: {result.consensus_strength:.3f}")
            print(f"  Disagreement: {result.disagreement_analysis}")
        else:
            print(f"  Score: {result.score:.3f}")
            print(f"  Reasoning: {result.reasoning[:200]}...")
            print(f"  Confidence: {result.confidence:.3f}")
    
    # Save evaluation history
    judge.save_evaluation_history("research/results/evaluation_history.json")
    
    # Print statistics
    stats = judge.get_evaluation_statistics()
    print(f"\nEvaluation Statistics:")
    print(f"Total evaluations: {stats['total_evaluations']}")
    print(f"Average confidence: {stats['average_confidence']:.3f}")
    print(f"Average evaluation time: {stats['average_evaluation_time']:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())