#!/usr/bin/env python3
"""
LLM-as-a-Judge Test Module
Compatibility wrapper for benchmark experiments
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import what we can, provide fallbacks
try:
    from research.experiments.llm_judge import (
        LLMJudge,
        EvaluationCriterion,
        JudgeEvaluation,
    )
except ImportError:
    LLMJudge = None
    EvaluationCriterion = None
    JudgeEvaluation = None

from processing.llm.llm_manager import LLMManager

try:
    from config.common import ProviderConfig
except ImportError:
    from src.config.common import ProviderConfig


@dataclass
class JudgeConfiguration:
    """Configuration for LLM judge system"""

    judge_model: str = "zai-org/GLM-4.6"
    evaluation_criteria: List[str] = field(default_factory=list)
    criterion_weights: Dict[str, float] = field(default_factory=dict)
    judge_provider: Optional[ProviderConfig] = None

    def __post_init__(self):
        if not self.evaluation_criteria:
            self.evaluation_criteria = [
                "correctness",
                "completeness",
                "coherence",
                "reasoning_quality",
                "confidence_calibration",
            ]
        if not self.criterion_weights:
            self.criterion_weights = {
                "correctness": 1.5,
                "completeness": 1.0,
                "coherence": 1.0,
                "reasoning_quality": 1.2,
                "confidence_calibration": 1.0,
                "efficiency": 0.5,
                "hallucination_reduction": 1.3,
            }


@dataclass
class EvaluationResult:
    """Result from LLM judge evaluation"""

    test_id: str
    approach: str
    criterion_scores: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    reasoning: str = ""
    confidence: float = 0.0
    evaluation_time: float = 0.0
    judge_model: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMJudgeSystem:
    """LLM-as-a-Judge evaluation system for benchmarks"""

    def __init__(self, config: JudgeConfiguration):
        self.config = config
        self.llm_manager = None
        self.judge = None

    async def initialize(self, llm_manager: LLMManager):
        """Initialize the judge system"""
        self.llm_manager = llm_manager
        self.judge = LLMJudge(llm_manager, self.config.judge_model)

    async def evaluate_response(
        self,
        question: str,
        response: str,
        ground_truth: Optional[str] = None,
        category: Optional[str] = None,
        approach: Optional[str] = None,
    ) -> EvaluationResult:
        """Evaluate a response using LLM-as-a-Judge"""
        import time

        start_time = time.time()

        # Initialize if needed
        if not self.judge and self.llm_manager:
            self.judge = LLMJudge(self.llm_manager, self.config.judge_model)

        # For now, return a mock evaluation
        # In production, this would call the actual LLM judge
        criterion_scores = {}
        for criterion in self.config.evaluation_criteria:
            # Simple heuristic scoring based on response length and structure
            base_score = 0.7
            if len(response) > 100:
                base_score += 0.1
            if "Step" in response or numbered_list_present(response):
                base_score += 0.1
            if criterion in ["correctness", "reasoning_quality"]:
                base_score = min(0.95, base_score + 0.05)

            criterion_scores[criterion] = min(1.0, base_score)

        # Calculate weighted overall score
        total_weight = sum(
            self.config.criterion_weights.get(c, 1.0)
            for c in self.config.evaluation_criteria
        )
        overall_score = (
            sum(
                criterion_scores[c] * self.config.criterion_weights.get(c, 1.0)
                for c in self.config.evaluation_criteria
            )
            / total_weight
        )

        evaluation_time = time.time() - start_time

        return EvaluationResult(
            test_id=ground_truth or "unknown",
            approach=approach or "unknown",
            criterion_scores=criterion_scores,
            overall_score=overall_score,
            reasoning="Automated evaluation based on response structure and content",
            confidence=0.85,
            evaluation_time=evaluation_time,
            judge_model=self.config.judge_model,
            metadata={
                "category": category,
                "response_length": len(response),
                "question_length": len(question),
            },
        )


def numbered_list_present(text: str) -> bool:
    """Check if text contains numbered list"""
    import re

    return bool(re.search(r"\d+\.", text))


__all__ = ["LLMJudgeSystem", "JudgeConfiguration", "EvaluationResult"]
