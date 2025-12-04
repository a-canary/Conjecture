#!/usr/bin/env python3
"""
Conjecture Hypothesis Testing Rubric & Iteration Loop
Comprehensive framework for systematically testing and proving Conjecture hypotheses
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
import statistics
from scipy import stats
import pandas as pd

# Add src to path for imports
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from core.models import Claim, ClaimState, ClaimType
from processing.llm.llm_manager import LLMManager
from config.common import ProviderConfig


class HypothesisTier(str, Enum):
    """Hypothesis testing priority tiers"""

    CORE_TECHNICAL = "core_technical"  # Tier 1: Highest priority
    ARCHITECTURE = "architecture"  # Tier 2: Design goals
    USER_EXPERIENCE = "user_experience"  # Tier 3: UX objectives
    RESEARCH_VALIDATION = "research_validation"  # Tier 4: Research goals
    PERFORMANCE_EFFICIENCY = "performance_efficiency"  # Tier 4: Efficiency targets


class HypothesisStatus(str, Enum):
    """Hypothesis testing status"""

    UNTESTED = "untested"
    IN_PROGRESS = "in_progress"
    PARTIALLY_PROVEN = "partially_proven"
    PROVEN = "proven"
    DISPROVEN = "disproven"


class EvidenceStrength(str, Enum):
    """Evidence strength levels"""

    NONE = "none"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    CONCLUSIVE = "conclusive"


@dataclass
class EvaluationRubric:
    """Enhanced evaluation rubric for hypothesis testing"""

    metric: str
    weight: float
    success_threshold: float
    description: str
    score_levels: Dict[int, str]


@dataclass
class HypothesisConfig:
    """Configuration for testing a specific hypothesis"""

    hypothesis_id: str
    tier: HypothesisTier
    statement: str
    success_criteria: str
    test_categories: List[str]
    models_to_compare: List[str]
    approaches_to_test: List[str]
    sample_size_min: int = 20
    sample_size_preferred: int = 50
    statistical_power_target: float = 0.8
    alpha_level: float = 0.05


@dataclass
class HypothesisResult:
    """Results from testing a hypothesis"""

    hypothesis_id: str
    status: HypothesisStatus
    confidence: float  # 0.0 to 1.0
    evidence_strength: EvidenceStrength
    iterations_completed: int
    last_updated: datetime
    primary_metric_score: float
    statistical_significance: float
    effect_size: float
    blocking_issues: List[str]
    detailed_results: Dict[str, Any]


@dataclass
class IterationResult:
    """Results from a single testing iteration"""

    iteration_number: int
    hypothesis_id: str
    test_results: List[Dict[str, Any]]
    statistical_analysis: Dict[str, float]
    improvements_made: List[str]
    next_actions: List[str]
    timestamp: datetime


class ConjectureTestingFramework:
    """Comprehensive framework for testing Conjecture hypotheses"""

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
        self.hypotheses = self._initialize_hypotheses()
        self.evaluation_rubrics = self._initialize_evaluation_rubrics()

        # Testing state
        self.active_hypotheses: Dict[str, HypothesisResult] = {}
        self.completed_hypotheses: Dict[str, HypothesisResult] = {}
        self.iteration_history: Dict[str, List[IterationResult]] = {}

        # Logging
        self.logger = self._setup_logging()

    def _initialize_hypotheses(self) -> Dict[str, HypothesisConfig]:
        """Initialize all 20 Conjecture hypotheses"""
        hypotheses = {}

        # Core Technical Hypotheses (Tier 1)
        hypotheses["central_hypothesis"] = HypothesisConfig(
            hypothesis_id="central_hypothesis",
            tier=HypothesisTier.CORE_TECHNICAL,
            statement="By decomposing tasks and concepts, and providing relevant context through claims-based representations that include in-context learning examples of task breakdown strategies, research-plan-work-validate phases, scientific method, critical thinking, and fact-checking best practices, small LLMs can achieve performance comparable to larger models on complex reasoning tasks.",
            success_criteria="≥20% improvement in correctness for tiny models vs direct approach, with statistical significance (p<0.05)",
            test_categories=[
                "task_decomposition",
                "complex_reasoning",
                "research_tasks",
            ],
            models_to_compare=[
                "ibm/granite-4-h-tiny",
                "glm-z1-9b-0414",
                "zai-org/GLM-4.6",
            ],
            approaches_to_test=["direct", "conjecture", "few_shot"],
        )

        hypotheses["task_decomposition"] = HypothesisConfig(
            hypothesis_id="task_decomposition",
            tier=HypothesisTier.CORE_TECHNICAL,
            statement="Small LLMs will show 20%+ improvement in correctness when using task decomposition versus direct approach.",
            success_criteria="≥20% improvement in correctness, p<0.05, effect size >0.5",
            test_categories=[
                "complex_reasoning",
                "planning_tasks",
                "analysis_problems",
            ],
            models_to_compare=["ibm/granite-4-h-tiny", "glm-z1-9b-0414"],
            approaches_to_test=["direct", "task_decomposition"],
        )

        hypotheses["cost_efficiency"] = HypothesisConfig(
            hypothesis_id="cost_efficiency",
            tier=HypothesisTier.CORE_TECHNICAL,
            statement="The multi-claim evaluation process may require more tokens, turns, and time, but can be executed with smaller models and smaller context windows, reducing overall computational cost while maintaining nearly equivalent accuracy. Over multiple sessions, claim persistence functions like caching, decreasing cumulative LLM token usage.",
            success_criteria="≥15% reduction in computational cost, ≥90% accuracy maintained",
            test_categories=[
                "token_efficiency",
                "session_persistence",
                "context_window_usage",
            ],
            models_to_compare=["ibm/granite-4-h-tiny", "glm-z1-9b-0414"],
            approaches_to_test=["single_turn", "multi_claim"],
        )

        hypotheses["model_parity"] = HypothesisConfig(
            hypothesis_id="model_parity",
            tier=HypothesisTier.CORE_TECHNICAL,
            statement="Small models (3-9B parameters) with Conjecture prompting will match or exceed larger models (30B+) on reasoning tasks.",
            success_criteria="≤10% performance gap between small+Conjecture vs large models",
            test_categories=[
                "reasoning_tasks",
                "mathematical_problems",
                "logical_puzzles",
            ],
            models_to_compare=[
                "ibm/granite-4-h-tiny",
                "glm-z1-9b-0414",
                "zai-org/GLM-4.6",
            ],
            approaches_to_test=["conjecture_small", "standard_large"],
        )

        hypotheses["claims_reasoning"] = HypothesisConfig(
            hypothesis_id="claims_reasoning",
            tier=HypothesisTier.CORE_TECHNICAL,
            statement="Claims-based reasoning will show 15%+ improvement in correctness and confidence calibration.",
            success_criteria="≥15% improvement in correctness and confidence calibration",
            test_categories=[
                "evidence_evaluation",
                "argument_analysis",
                "confidence_scoring",
            ],
            models_to_compare=["ibm/granite-4-h-tiny", "glm-z1-9b-0414"],
            approaches_to_test=["standard_reasoning", "claims_based_reasoning"],
        )

        hypotheses["end_to_end_pipeline"] = HypothesisConfig(
            hypothesis_id="end_to_end_pipeline",
            tier=HypothesisTier.CORE_TECHNICAL,
            statement="Full Conjecture pipeline will show 25%+ improvement over baseline for small models on complex tasks.",
            success_criteria="≥25% improvement over baseline on complex multi-step tasks",
            test_categories=[
                "integrated_research",
                "multi_step_analysis",
                "comprehensive_evaluation",
            ],
            models_to_compare=["ibm/granite-4-h-tiny", "glm-z1-9b-0414"],
            approaches_to_test=["baseline", "full_conjecture_pipeline"],
        )

        # Architecture Goals (Tier 2)
        hypotheses["three_layer_architecture"] = HypothesisConfig(
            hypothesis_id="three_layer_architecture",
            tier=HypothesisTier.ARCHITECTURE,
            statement="Implement clean separation between Data Layer (Claims + Tools), Process Layer (Context building, LLM orchestration), and Presentation Layer (CLI, TUI, GUI).",
            success_criteria="Clean separation achieved with <5% cross-layer dependencies",
            test_categories=[
                "component_interaction",
                "interface_consistency",
                "separation_validation",
            ],
            models_to_compare=["system_architecture"],
            approaches_to_test=["current_implementation"],
        )

        hypotheses["claim_centric_system"] = HypothesisConfig(
            hypothesis_id="claim_centric_system",
            tier=HypothesisTier.ARCHITECTURE,
            statement="All knowledge, including methodologies and skills, represented as claims with confidence scores and evidence linking.",
            success_criteria="≥90% of knowledge successfully represented as claims",
            test_categories=[
                "knowledge_representation",
                "claim_relationships",
                "confidence_scoring",
            ],
            models_to_compare=["knowledge_system"],
            approaches_to_test=["claims_representation"],
        )

        # Add remaining hypotheses (simplified for brevity)
        # ... (would include all 20 hypotheses in full implementation)

        return hypotheses

    def _initialize_evaluation_rubrics(self) -> Dict[str, EvaluationRubric]:
        """Initialize comprehensive evaluation rubrics"""
        rubrics = {}

        rubrics["correctness"] = EvaluationRubric(
            metric="correctness",
            weight=1.5,
            success_threshold=0.70,
            description="Factual accuracy and correctness of response",
            score_levels={
                0: "Completely incorrect or contains major factual errors",
                0.25: "Mostly incorrect with some accurate elements",
                0.5: "Partially correct, mixture of accurate and inaccurate information",
                0.75: "Mostly correct with minor inaccuracies",
                1.0: "Completely correct and factually accurate",
            },
        )

        rubrics["reasoning_quality"] = EvaluationRubric(
            metric="reasoning_quality",
            weight=1.2,
            success_threshold=0.65,
            description="Quality of logical reasoning and argumentation",
            score_levels={
                0: "No reasoning or completely flawed logic",
                0.25: "Weak reasoning with major logical fallacies",
                0.5: "Adequate reasoning but with some gaps or weaknesses",
                0.75: "Strong reasoning with minor logical issues",
                1.0: "Excellent reasoning, rigorous and insightful",
            },
        )

        rubrics["completeness"] = EvaluationRubric(
            metric="completeness",
            weight=1.0,
            success_threshold=0.75,
            description="How completely response addresses all aspects of question",
            score_levels={
                0: "Fails to address question or major aspects missing",
                0.25: "Addresses only minor aspects, major components missing",
                0.5: "Addresses some key aspects but incomplete overall",
                0.75: "Addresses most aspects with minor omissions",
                1.0: "Completely addresses all aspects of question",
            },
        )

        rubrics["coherence"] = EvaluationRubric(
            metric="coherence",
            weight=1.0,
            success_threshold=0.70,
            description="Logical flow, consistency, and structural coherence",
            score_levels={
                0: "Incoherent, contradictory, or completely disorganized",
                0.25: "Poorly organized with significant logical gaps",
                0.5: "Somewhat coherent but with organizational issues",
                0.75: "Well-organized with minor logical issues",
                1.0: "Perfectly coherent, logical, and well-structured",
            },
        )

        rubrics["confidence_calibration"] = EvaluationRubric(
            metric="confidence_calibration",
            weight=1.0,
            success_threshold=0.60,
            description="How well model's confidence matches its actual accuracy",
            score_levels={
                0: "Completely misaligned confidence (overconfident when wrong, underconfident when right)",
                0.25: "Poorly calibrated confidence",
                0.5: "Somewhat calibrated confidence with notable misalignments",
                0.75: "Well-calibrated confidence with minor misalignments",
                1.0: "Perfectly calibrated confidence",
            },
        )

        rubrics["efficiency"] = EvaluationRubric(
            metric="efficiency",
            weight=0.5,
            success_threshold=0.60,
            description="Efficiency and conciseness of response",
            score_levels={
                0: "Extremely verbose, inefficient, or incomplete due to brevity",
                0.25: "Inefficient with significant verbosity or important omissions",
                0.5: "Moderately efficient with some verbosity or minor omissions",
                0.75: "Efficient with minor verbosity issues",
                1.0: "Perfectly efficient, concise yet complete",
            },
        )

        rubrics["hallucination_reduction"] = EvaluationRubric(
            metric="hallucination_reduction",
            weight=1.3,
            success_threshold=0.80,
            description="Reduction in hallucinations and factual grounding",
            score_levels={
                0: "No reduction, many hallucinations",
                0.25: "Minimal reduction, still significant hallucinations",
                0.5: "Moderate reduction, some hallucinations remain",
                0.75: "Good reduction, few minor hallucinations",
                1.0: "Excellent reduction, no detectable hallucinations",
            },
        )

        return rubrics

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for hypothesis testing framework"""
        logger = logging.getLogger("hypothesis_testing")
        logger.setLevel(logging.INFO)

        # Create file handler
        fh = logging.FileHandler(self.results_dir / "hypothesis_testing.log")
        fh.setLevel(logging.INFO)

        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    async def initialize(self, provider_configs: List[ProviderConfig]):
        """Initialize LLM manager and testing framework"""
        try:
            self.llm_manager = LLMManager(provider_configs)
            self.logger.info("Hypothesis testing framework initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize hypothesis testing framework: {e}")
            return False

    async def run_hypothesis_testing_cycle(
        self, hypothesis_ids: List[str] = None, max_iterations: int = 5
    ) -> Dict[str, HypothesisResult]:
        """
        Run complete hypothesis testing cycle with iteration loop

        Args:
            hypothesis_ids: List of hypothesis IDs to test (None = all)
            max_iterations: Maximum iterations per hypothesis

        Returns:
            Dictionary of hypothesis results
        """
        if hypothesis_ids is None:
            hypothesis_ids = list(self.hypotheses.keys())

        results = {}

        for hypothesis_id in hypothesis_ids:
            self.logger.info(f"Starting hypothesis testing cycle for: {hypothesis_id}")

            hypothesis_config = self.hypotheses[hypothesis_id]
            hypothesis_result = await self._test_hypothesis_with_iterations(
                hypothesis_config, max_iterations
            )

            results[hypothesis_id] = hypothesis_result
            self.completed_hypotheses[hypothesis_id] = hypothesis_result

            # Save intermediate results
            await self._save_hypothesis_results(hypothesis_id, hypothesis_result)

        # Generate comprehensive report
        await self._generate_testing_report(results)

        return results

    async def _test_hypothesis_with_iterations(
        self, hypothesis_config: HypothesisConfig, max_iterations: int
    ) -> HypothesisResult:
        """Test a single hypothesis with iteration loop"""
        hypothesis_id = hypothesis_config.hypothesis_id

        # Initialize hypothesis result
        hypothesis_result = HypothesisResult(
            hypothesis_id=hypothesis_id,
            status=HypothesisStatus.UNTESTED,
            confidence=0.0,
            evidence_strength=EvidenceStrength.NONE,
            iterations_completed=0,
            last_updated=datetime.utcnow(),
            primary_metric_score=0.0,
            statistical_significance=1.0,
            effect_size=0.0,
            blocking_issues=[],
            detailed_results={},
        )

        self.active_hypotheses[hypothesis_id] = hypothesis_result

        for iteration in range(max_iterations):
            self.logger.info(
                f"Testing {hypothesis_id} - Iteration {iteration + 1}/{max_iterations}"
            )

            # Run single iteration
            iteration_result = await self._run_single_iteration(
                hypothesis_config, iteration + 1
            )

            # Store iteration result
            if hypothesis_id not in self.iteration_history:
                self.iteration_history[hypothesis_id] = []
            self.iteration_history[hypothesis_id].append(iteration_result)

            # Update hypothesis result based on iteration
            hypothesis_result = await self._update_hypothesis_result(
                hypothesis_result, iteration_result, hypothesis_config
            )

            # Check if we should stop early
            if hypothesis_result.status in [
                HypothesisStatus.PROVEN,
                HypothesisStatus.DISPROVEN,
            ]:
                self.logger.info(
                    f"Early stopping for {hypothesis_id}: {hypothesis_result.status}"
                )
                break

            # Save iteration results
            await self._save_iteration_results(hypothesis_id, iteration_result)

        hypothesis_result.iterations_completed = len(
            self.iteration_history.get(hypothesis_id, [])
        )
        hypothesis_result.last_updated = datetime.utcnow()

        return hypothesis_result

    async def _run_single_iteration(
        self, hypothesis_config: HypothesisConfig, iteration_number: int
    ) -> IterationResult:
        """Run a single testing iteration"""
        self.logger.info(
            f"Running iteration {iteration_number} for {hypothesis_config.hypothesis_id}"
        )

        # Execute real LLM testing - NO SIMULATION
        raise NotImplementedError(
            "Real LLM testing implementation required. "
            "Use simple_experiment.py or comprehensive_scientific_research.py for actual model calls."
        )

        # Calculate statistical analysis
        statistical_analysis = self._calculate_statistical_significance(
            test_results, hypothesis_config
        )

        # Determine improvements and next actions
        improvements_made = self._identify_improvements(test_results, iteration_number)
        next_actions = self._plan_next_actions(
            test_results, hypothesis_config, iteration_number
        )

        return IterationResult(
            iteration_number=iteration_number,
            hypothesis_id=hypothesis_config.hypothesis_id,
            test_results=test_results,
            statistical_analysis=statistical_analysis,
            improvements_made=improvements_made,
            next_actions=next_actions,
            timestamp=datetime.utcnow(),
        )

    def _calculate_statistical_significance(
        self, test_results: List[Dict[str, Any]], hypothesis_config: HypothesisConfig
    ) -> Dict[str, float]:
        """Calculate statistical significance of test results"""
        if len(test_results) < 2:
            return {"p_value": 1.0, "effect_size": 0.0, "power": 0.0}

        # Extract primary metric scores
        scores = [result.get("correctness", 0.0) for result in test_results]

        # Simple statistical analysis (would be more sophisticated in real implementation)
        if len(scores) >= 2:
            # Paired t-test or similar
            mean_score = statistics.mean(scores)
            std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0

            # Real statistical calculation using scipy
            from scipy import stats
            import numpy as np

            # Calculate p-value using t-test
            t_stat, p_value = stats.ttest_1samp(
                scores, 0.5
            )  # Test against null hypothesis of 0.5

            # Effect size (Cohen's d)
            effect_size = mean_score / (std_score + 0.01)

            # Statistical power calculation
            from scipy.stats import norm

            alpha = 0.05
            z_alpha = norm.ppf(1 - alpha / 2)
            z_beta = (effect_size * np.sqrt(len(scores))) - z_alpha
            power = norm.cdf(z_beta)

            return {
                "p_value": p_value,
                "effect_size": effect_size,
                "power": power,
                "mean_score": mean_score,
                "std_score": std_score,
            }

        return {"p_value": 1.0, "effect_size": 0.0, "power": 0.0}

    def _identify_improvements(
        self, test_results: List[Dict[str, Any]], iteration_number: int
    ) -> List[str]:
        """Identify improvements made in this iteration"""
        improvements = []

        if iteration_number == 1:
            improvements.append("Established baseline measurements")
        else:
            improvements.append(f"Iteration {iteration_number} refinements applied")

        # Check for metric improvements
        if len(test_results) > 1:
            conj_score = test_results[0].get("correctness", 0.0)
            direct_score = test_results[1].get("correctness", 0.0)

            if conj_score > direct_score * 1.1:  # 10% improvement
                improvements.append("Conjecture approach shows significant improvement")

            if conj_score > 0.7:  # Above threshold
                improvements.append("Achieved success threshold for correctness")

        return improvements

    def _plan_next_actions(
        self,
        test_results: List[Dict[str, Any]],
        hypothesis_config: HypothesisConfig,
        iteration_number: int,
    ) -> List[str]:
        """Plan next actions based on current results"""
        actions = []

        # Analyze current performance
        avg_score = statistics.mean([r.get("correctness", 0.0) for r in test_results])

        if avg_score < hypothesis_config.success_criteria.split("≥")[1].split("%")[
            0
        ].replace(" ", ""):
            actions.append("Refine Conjecture implementation for better performance")

        if avg_score >= 0.7:
            actions.append("Prepare for final validation")

        if iteration_number >= 3:
            actions.append("Consider hypothesis proven or disproven based on trend")

        return actions

    async def _update_hypothesis_result(
        self,
        hypothesis_result: HypothesisResult,
        iteration_result: IterationResult,
        hypothesis_config: HypothesisConfig,
    ) -> HypothesisResult:
        """Update hypothesis result based on iteration results"""
        # Extract key metrics
        test_results = iteration_result.test_results
        statistical_analysis = iteration_result.statistical_analysis

        # Calculate primary metric score (weighted average)
        primary_score = 0.0
        total_weight = 0.0

        for result in test_results:
            for metric, rubric in self.evaluation_rubrics.items():
                if metric in result:
                    primary_score += result[metric] * rubric.weight
                    total_weight += rubric.weight

        if total_weight > 0:
            primary_score = primary_score / total_weight

        # Update hypothesis result
        hypothesis_result.primary_metric_score = primary_score
        hypothesis_result.statistical_significance = statistical_analysis.get(
            "p_value", 1.0
        )
        hypothesis_result.effect_size = statistical_analysis.get("effect_size", 0.0)
        hypothesis_result.iterations_completed = iteration_result.iteration_number

        # Determine status and confidence
        p_value = statistical_analysis.get("p_value", 1.0)
        effect_size = statistical_analysis.get("effect_size", 0.0)

        if primary_score >= 0.8 and p_value < 0.05 and effect_size > 0.5:
            hypothesis_result.status = HypothesisStatus.PROVEN
            hypothesis_result.confidence = 0.9
            hypothesis_result.evidence_strength = EvidenceStrength.STRONG
        elif primary_score >= 0.6 and p_value < 0.1 and effect_size > 0.3:
            hypothesis_result.status = HypothesisStatus.PARTIALLY_PROVEN
            hypothesis_result.confidence = 0.6
            hypothesis_result.evidence_strength = EvidenceStrength.MODERATE
        elif primary_score < 0.4 or p_value >= 0.2:
            hypothesis_result.status = HypothesisStatus.DISPROVEN
            hypothesis_result.confidence = 0.2
            hypothesis_result.evidence_strength = EvidenceStrength.WEAK
        else:
            hypothesis_result.status = HypothesisStatus.IN_PROGRESS
            hypothesis_result.confidence = 0.4
            hypothesis_result.evidence_strength = EvidenceStrength.MODERATE

        return hypothesis_result

    async def _save_hypothesis_results(
        self, hypothesis_id: str, result: HypothesisResult
    ):
        """Save hypothesis results to file"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"hypothesis_{hypothesis_id}_{timestamp}.json"
        filepath = self.results_dir / filename

        result_data = asdict(result)

        # Convert datetime objects to strings
        result_data["last_updated"] = result.last_updated.isoformat()

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Hypothesis results saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save hypothesis results: {e}")

    async def _save_iteration_results(
        self, hypothesis_id: str, result: IterationResult
    ):
        """Save iteration results to file"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"iteration_{hypothesis_id}_{result.iteration_number}_{timestamp}.json"
        )
        filepath = self.results_dir / filename

        result_data = asdict(result)
        result_data["timestamp"] = result.timestamp.isoformat()

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Iteration results saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save iteration results: {e}")

    async def _generate_testing_report(self, results: Dict[str, HypothesisResult]):
        """Generate comprehensive testing report"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"hypothesis_testing_report_{timestamp}.md"
        filepath = self.analysis_dir / filename

        report_lines = [
            "# Conjecture Hypothesis Testing Report",
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            "",
        ]

        # Calculate summary statistics
        total_hypotheses = len(results)
        proven_count = len(
            [r for r in results.values() if r.status == HypothesisStatus.PROVEN]
        )
        partially_proven_count = len(
            [
                r
                for r in results.values()
                if r.status == HypothesisStatus.PARTIALLY_PROVEN
            ]
        )
        disproven_count = len(
            [r for r in results.values() if r.status == HypothesisStatus.DISPROVEN]
        )

        report_lines.extend(
            [
                f"- **Total Hypotheses Tested**: {total_hypotheses}",
                f"- **Proven**: {proven_count} ({proven_count / total_hypotheses * 100:.1f}%)",
                f"- **Partially Proven**: {partially_proven_count} ({partially_proven_count / total_hypotheses * 100:.1f}%)",
                f"- **Disproven**: {disproven_count} ({disproven_count / total_hypotheses * 100:.1f}%)",
                "",
                "## Detailed Results",
                "",
            ]
        )

        # Group by tier
        tier_results = {}
        for hypothesis_id, result in results.items():
            tier = self.hypotheses[hypothesis_id].tier.value
            if tier not in tier_results:
                tier_results[tier] = []
            tier_results[tier].append((hypothesis_id, result))

        for tier, tier_hypotheses in tier_results.items():
            report_lines.extend([f"### {tier.replace('_', ' ').title()}", ""])

            for hypothesis_id, result in tier_hypotheses:
                status_emoji = {
                    HypothesisStatus.PROVEN: "✅",
                    HypothesisStatus.PARTIALLY_PROVEN: "🟡",
                    HypothesisStatus.DISPROVEN: "❌",
                    HypothesisStatus.IN_PROGRESS: "🔄",
                }.get(result.status, "❓")

                report_lines.extend(
                    [
                        f"#### {status_emoji} {hypothesis_id.replace('_', ' ').title()}",
                        f"**Status**: {result.status.value}",
                        f"**Confidence**: {result.confidence:.2f}",
                        f"**Evidence Strength**: {result.evidence_strength.value}",
                        f"**Primary Score**: {result.primary_metric_score:.3f}",
                        f"**Statistical Significance**: p={result.statistical_significance:.3f}",
                        f"**Effect Size**: {result.effect_size:.3f}",
                        f"**Iterations**: {result.iterations_completed}",
                        "",
                    ]
                )

        # Add recommendations
        report_lines.extend(
            [
                "## Recommendations",
                "",
                "### For Proven Hypotheses",
                "- Integrate into production Conjecture implementation",
                "- Document best practices and patterns",
                "- Use as foundation for further enhancements",
                "",
                "### For Partially Proven Hypotheses",
                "- Conduct additional iterations with refined approaches",
                "- Investigate edge cases and failure modes",
                "- Consider hybrid approaches",
                "",
                "### For Disproven Hypotheses",
                "- Re-evaluate underlying assumptions",
                "- Consider alternative approaches",
                "- Document lessons learned",
                "",
            ]
        )

        report_content = "\n".join(report_lines)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(report_content)

            self.logger.info(f"Testing report generated: {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to generate testing report: {e}")

    def get_progress_dashboard(self) -> Dict[str, Any]:
        """Get current progress across all hypotheses"""
        total_hypotheses = len(self.hypotheses)
        completed_count = len(self.completed_hypotheses)

        # Count by status
        status_counts = {}
        for result in self.completed_hypotheses.values():
            status_counts[result.status.value] = (
                status_counts.get(result.status.value, 0) + 1
            )

        # Count by tier
        tier_counts = {}
        for hypothesis_id, config in self.hypotheses.items():
            tier = config.tier.value
            if hypothesis_id in self.completed_hypotheses:
                tier_counts[tier] = tier_counts.get(tier, 0) + 1

        return {
            "total_hypotheses": total_hypotheses,
            "completed_hypotheses": completed_count,
            "completion_rate": completed_count / total_hypotheses
            if total_hypotheses > 0
            else 0.0,
            "status_breakdown": status_counts,
            "tier_breakdown": tier_counts,
            "active_hypotheses": list(self.active_hypotheses.keys()),
            "last_updated": datetime.utcnow().isoformat(),
        }


async def main():
    """Main function to run hypothesis testing framework"""
    from config.common import ProviderConfig

    # Setup provider configurations
    providers = [
        ProviderConfig(
            url="http://localhost:1234",  # LM Studio
            api_key="",
            model="ibm/granite-4-h-tiny",
        ),
        ProviderConfig(
            url="http://localhost:1234",  # LM Studio
            api_key="",
            model="glm-z1-9b-0414",
        ),
        ProviderConfig(
            url="https://llm.chutes.ai/v1",  # Chutes
            api_key="your-api-key",
            model="zai-org/GLM-4.6",
        ),
    ]

    # Initialize framework
    framework = ConjectureTestingFramework()
    await framework.initialize(providers)

    # Run hypothesis testing cycle
    # Test core technical hypotheses first
    core_hypotheses = [
        "central_hypothesis",
        "task_decomposition",
        "cost_efficiency",
        "model_parity",
        "claims_reasoning",
        "end_to_end_pipeline",
    ]

    results = await framework.run_hypothesis_testing_cycle(
        hypothesis_ids=core_hypotheses, max_iterations=3
    )

    # Print progress dashboard
    dashboard = framework.get_progress_dashboard()
    print("\n" + "=" * 60)
    print("HYPOTHESIS TESTING PROGRESS DASHBOARD")
    print("=" * 60)
    print(f"Total Hypotheses: {dashboard['total_hypotheses']}")
    print(
        f"Completed: {dashboard['completed_hypotheses']} ({dashboard['completion_rate']:.1%})"
    )
    print(f"Status Breakdown: {dashboard['status_breakdown']}")
    print(f"Tier Breakdown: {dashboard['tier_breakdown']}")
    print("=" * 60)

    # Print individual results
    for hypothesis_id, result in results.items():
        status_emoji = {
            HypothesisStatus.PROVEN: "✅",
            HypothesisStatus.PARTIALLY_PROVEN: "🟡",
            HypothesisStatus.DISPROVEN: "❌",
            HypothesisStatus.IN_PROGRESS: "🔄",
        }.get(result.status, "❓")

        print(
            f"{status_emoji} {hypothesis_id}: {result.status.value} (confidence: {result.confidence:.2f})"
        )


if __name__ == "__main__":
    asyncio.run(main())
