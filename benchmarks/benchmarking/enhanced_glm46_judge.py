#!/usr/bin/env python3
"""
Enhanced GLM-4.6 Judge System
Optimized evaluation methodology with enhanced prompt system integration

Provides sophisticated LLM-as-judge capabilities with:
- Enhanced evaluation prompts
- Multiple evaluation criteria
- Confidence scoring
- Structured feedback
- Integration with restored prompt system enhancements
"""

import asyncio
import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import aiohttp
from datetime import datetime

# Add src to path for imports
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.agent.prompt_system import PromptSystem, ProblemType, Difficulty

logger = logging.getLogger(__name__)


@dataclass
class JudgeEvaluation:
    """Structured evaluation from GLM-4.6 judge"""

    is_correct: bool
    confidence: float
    reasoning_quality: str
    problem_type_match: bool
    enhancement_usage: str
    feedback: str
    detailed_scores: Dict[str, float]
    evaluation_time: float


class EnhancedGLM46Judge:
    """Enhanced GLM-4.6 judge with improved evaluation methodology"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize judge with configuration"""
        self.api_key = config.get("key", "")
        self.base_url = config.get("url", "")
        self.model = config.get("model", "glm-4.6")
        self.prompt_system = PromptSystem()
        self.evaluation_cache = {}

        if not all([self.api_key, self.base_url]):
            raise ValueError("GLM-4.6 judge configuration incomplete")

    async def evaluate_response(
        self,
        problem: str,
        response: str,
        expected: str,
        problem_type: Optional[str] = None,
    ) -> JudgeEvaluation:
        """Evaluate response using enhanced GLM-4.6 judge methodology"""

        # Create cache key
        cache_key = hash(f"{problem[:100]}_{response[:100]}_{expected[:50]}")
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]

        start_time = datetime.now()

        # Analyze problem with our enhanced prompt system
        if not problem_type:
            detected_type = self.prompt_system._detect_problem_type(problem)
            problem_type = detected_type.value

        difficulty = self.prompt_system._estimate_difficulty(problem)

        # Create enhanced judge prompt
        judge_prompt = self._create_enhanced_judge_prompt(
            problem, response, expected, problem_type, difficulty.value
        )

        try:
            # Call GLM-4.6 judge
            evaluation_result = await self._call_glm46_judge(judge_prompt)

            # Parse and structure the evaluation
            evaluation = self._parse_evaluation_response(
                evaluation_result, problem_type, start_time
            )

            # Cache the result
            self.evaluation_cache[cache_key] = evaluation

            return evaluation

        except asyncio.TimeoutError:
            logger.error("GLM-4.6 evaluation timed out - using conservative fallback")
            evaluation_time = (datetime.now() - start_time).total_seconds()
            return JudgeEvaluation(
                is_correct=False,
                confidence=30,
                reasoning_quality="fair",
                problem_type_match=True,
                enhancement_usage="none",
                feedback="Timeout - Unable to verify correctness",
                detailed_scores={
                    "correctness": 30,
                    "methodology": 30,
                    "clarity": 30,
                    "completeness": 30,
                    "enhancement_usage": 0,
                },
                evaluation_time=evaluation_time,
            )
        except Exception as e:
            logger.error(f"GLM-4.6 evaluation failed: {e}")
            # Fallback evaluation
            return self._fallback_evaluation(problem, response, expected, start_time)

    def _create_enhanced_judge_prompt(
        self,
        problem: str,
        response: str,
        expected: str,
        problem_type: str,
        difficulty: str,
    ) -> str:
        """Create enhanced evaluation prompt with domain-specific criteria"""

        # Domain-specific evaluation criteria
        domain_criteria = {
            "mathematical": """
MATHEMATICAL EVALUATION CRITERIA:
- Correct calculation and final answer
- Proper mathematical methodology
- Clear step-by-step reasoning
- Appropriate use of formulas and units
- Verification of results""",
            "logical": """
LOGICAL EVALUATION CRITERIA:
- Valid logical structure
- Sound reasoning process
- Proper use of logical principles
- Clear argumentation
- No logical fallacies""",
            "scientific": """
SCIENTIFIC EVALUATION CRITERIA:
- Scientific accuracy
- Proper methodology
- Evidence-based reasoning
- Appropriate terminology
- Consideration of limitations""",
            "sequential": """
SEQUENTIAL EVALUATION CRITERIA:
- Correct step ordering
- Complete coverage of required steps
- Proper dependencies
- Clear progression
- No missed steps""",
            "decomposition": """
DECOMPOSITION EVALUATION CRITERIA:
- Complete component analysis
- Proper component identification
- Logical integration
- Comprehensive coverage
- Systematic approach""",
            "general": """
GENERAL EVALUATION CRITERIA:
- Relevance to question
- Clarity of expression
- Logical coherence
- Completeness of answer
- Quality of reasoning""",
        }

        criteria = domain_criteria.get(problem_type, domain_criteria["general"])

        return f"""You are an expert evaluator for AI responses with deep knowledge of {problem_type} problem-solving.

EVALUATION CONTEXT:
- Problem Type: {problem_type.upper()}
- Difficulty Level: {difficulty.upper()}
- This is a {problem_type} problem requiring specialized reasoning approaches

{criteria}

PROBLEM TO SOLVE:
{problem}

EXPECTED ANSWER (for reference):
{expected}

RESPONSE TO EVALUATE:
{response}

ENHANCED EVALUATION TASK:
Assess this response across multiple dimensions and provide detailed, structured feedback.

EVALUATION DIMENSIONS:
1. **Correctness**: Is the answer factually correct?
2. **Methodology**: Does it use appropriate {problem_type} reasoning approaches?
3. **Clarity**: Is the reasoning clear and well-structured?
4. **Completeness**: Does it fully address all aspects of the problem?
5. **Enhancement Usage**: Does it leverage domain-specific strategies?

Return your evaluation as JSON:
{{
    "is_correct": true/false,
    "confidence": 0-100,
    "reasoning_quality": "poor/fair/good/excellent",
    "problem_type_match": true/false,
    "enhancement_usage": "none/minimal/moderate/extensive",
    "feedback": "Detailed explanation of strengths and areas for improvement",
    "detailed_scores": {{
        "correctness": 0-100,
        "methodology": 0-100,
        "clarity": 0-100,
        "completeness": 0-100,
        "enhancement_usage": 0-100
    }}
}}

Focus on providing constructive feedback that would help improve future responses."""

    async def _call_glm46_judge(self, judge_prompt: str) -> str:
        """Make API call to GLM-4.6 judge"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": judge_prompt}],
            "temperature": 0.1,  # Low temperature for consistent evaluation
            "max_tokens": 800,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    raise Exception(
                        f"GLM-4.6 API error: {response.status} - {await response.text()}"
                    )

    def _parse_evaluation_response(
        self, response: str, problem_type: str, start_time: datetime
    ) -> JudgeEvaluation:
        """Parse GLM-4.6 response into structured evaluation"""

        try:
            # Extract JSON from response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                eval_data = json.loads(json_match.group())
            else:
                # Fallback parsing
                eval_data = self._parse_text_evaluation(response)

            evaluation_time = (datetime.now() - start_time).total_seconds()

            return JudgeEvaluation(
                is_correct=eval_data.get("is_correct", False),
                confidence=float(eval_data.get("confidence", 50)),
                reasoning_quality=eval_data.get("reasoning_quality", "fair"),
                problem_type_match=eval_data.get("problem_type_match", True),
                enhancement_usage=eval_data.get("enhancement_usage", "minimal"),
                feedback=eval_data.get("feedback", "No detailed feedback provided"),
                detailed_scores=eval_data.get("detailed_scores", {}),
                evaluation_time=evaluation_time,
            )

        except Exception as e:
            logger.warning(f"Failed to parse GLM-4.6 evaluation: {e}")
            return self._fallback_evaluation("", "", "", start_time)

    def _parse_text_evaluation(self, response: str) -> Dict[str, Any]:
        """Parse non-JSON evaluation response"""
        # Simple fallback parsing
        is_correct = any(
            word in response.lower() for word in ["correct", "accurate", "right"]
        )
        confidence = 70 if is_correct else 30

        return {
            "is_correct": is_correct,
            "confidence": confidence,
            "reasoning_quality": "fair",
            "problem_type_match": True,
            "enhancement_usage": "minimal",
            "feedback": response[:200],
            "detailed_scores": {
                "correctness": confidence,
                "methodology": 60,
                "clarity": 60,
                "completeness": 60,
                "enhancement_usage": 40,
            },
        }

    def _fallback_evaluation(
        self, problem: str, response: str, expected: str, start_time: datetime
    ) -> JudgeEvaluation:
        """Fallback evaluation when GLM-4.6 is unavailable"""

        # Simple text-based evaluation
        response_lower = response.lower()
        expected_lower = expected.lower()

        # Basic correctness check
        is_correct = expected_lower in response_lower or any(
            word in response_lower for word in expected_lower.split() if len(word) > 2
        )

        evaluation_time = (datetime.now() - start_time).total_seconds()

        return JudgeEvaluation(
            is_correct=is_correct,
            confidence=60 if is_correct else 40,
            reasoning_quality="fair",
            problem_type_match=True,
            enhancement_usage="minimal",
            feedback="Fallback evaluation - GLM-4.6 unavailable",
            detailed_scores={
                "correctness": 60 if is_correct else 40,
                "methodology": 50,
                "clarity": 50,
                "completeness": 50,
                "enhancement_usage": 30,
            },
            evaluation_time=evaluation_time,
        )

    def get_evaluation_summary(
        self, evaluations: List[JudgeEvaluation]
    ) -> Dict[str, Any]:
        """Get summary statistics for multiple evaluations"""

        if not evaluations:
            return {"error": "No evaluations provided"}

        correct_count = sum(1 for e in evaluations if e.is_correct)
        total_count = len(evaluations)

        avg_confidence = sum(e.confidence for e in evaluations) / total_count
        avg_evaluation_time = sum(e.evaluation_time for e in evaluations) / total_count

        # Aggregate detailed scores
        score_categories = [
            "correctness",
            "methodology",
            "clarity",
            "completeness",
            "enhancement_usage",
        ]
        avg_scores = {}

        for category in score_categories:
            scores = [
                e.detailed_scores.get(category, 0)
                for e in evaluations
                if e.detailed_scores
            ]
            avg_scores[category] = sum(scores) / len(scores) if scores else 0

        return {
            "total_evaluations": total_count,
            "correct_count": correct_count,
            "accuracy": correct_count / total_count,
            "average_confidence": avg_confidence,
            "average_evaluation_time": avg_evaluation_time,
            "average_scores": avg_scores,
            "reasoning_quality_distribution": {
                quality: sum(1 for e in evaluations if e.reasoning_quality == quality)
                for quality in ["poor", "fair", "good", "excellent"]
            },
            "enhancement_usage_distribution": {
                usage: sum(1 for e in evaluations if e.enhancement_usage == usage)
                for usage in ["none", "minimal", "moderate", "extensive"]
            },
        }

    async def benchmark_enhanced_vs_standard(
        self, test_problems: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Benchmark enhanced evaluation vs standard evaluation"""

        results = {
            "enhanced_evaluations": [],
            "standard_evaluations": [],
            "comparison": {},
        }

        for problem in test_problems:
            # Enhanced evaluation
            enhanced_eval = await self.evaluate_response(
                problem["problem"],
                problem["response"],
                problem["expected"],
                problem.get("type", "general"),
            )
            results["enhanced_evaluations"].append(enhanced_eval)

            # Standard evaluation (simplified)
            standard_eval = await self._standard_evaluation(
                problem["problem"], problem["response"], problem["expected"]
            )
            results["standard_evaluations"].append(standard_eval)

        # Compare results
        enhanced_correct = sum(
            1 for e in results["enhanced_evaluations"] if e.is_correct
        )
        standard_correct = sum(
            1 for e in results["standard_evaluations"] if e.is_correct
        )

        results["comparison"] = {
            "enhanced_accuracy": enhanced_correct / len(test_problems),
            "standard_accuracy": standard_correct / len(test_problems),
            "improvement": (enhanced_correct - standard_correct) / len(test_problems),
            "enhanced_avg_confidence": sum(
                e.confidence for e in results["enhanced_evaluations"]
            )
            / len(test_problems),
            "standard_avg_confidence": sum(
                e.confidence for e in results["standard_evaluations"]
            )
            / len(test_problems),
        }

        return results

    async def _standard_evaluation(
        self, problem: str, response: str, expected: str
    ) -> JudgeEvaluation:
        """Standard evaluation for comparison"""

        # Simplified judge prompt
        simple_prompt = f"""Is this response correct?

Problem: {problem}
Response: {response}
Expected: {expected}

Answer with only: CORRECT or INCORRECT"""

        try:
            judge_response = await self._call_glm46_judge(simple_prompt)
            is_correct = "CORRECT" in judge_response.upper()

            return JudgeEvaluation(
                is_correct=is_correct,
                confidence=80 if is_correct else 20,
                reasoning_quality="fair",
                problem_type_match=True,
                enhancement_usage="none",
                feedback="Standard evaluation",
                detailed_scores={"correctness": 80 if is_correct else 20},
                evaluation_time=1.0,
            )
        except Exception:
            # Fallback
            return JudgeEvaluation(
                is_correct=False,
                confidence=50,
                reasoning_quality="fair",
                problem_type_match=True,
                enhancement_usage="none",
                feedback="Evaluation failed",
                detailed_scores={},
                evaluation_time=1.0,
            )
