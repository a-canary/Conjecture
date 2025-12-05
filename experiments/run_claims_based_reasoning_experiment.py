#!/usr/bin/env python3
"""
Claims-Based Reasoning Experiment Runner
Tests if Conjecture's claims-based reasoning shows 15%+ improvement in correctness and confidence calibration.

This is the fourth critical experiment for validating the core hypothesis that:
"Claims-based reasoning will show 15%+ improvement in correctness and confidence calibration"
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


@dataclass
class ExperimentConfig:
    """Configuration for claims-based reasoning experiment"""
    
    # Test parameters
    sample_size: int = 75  # Target 50-100 test cases
    target_improvement: float = 0.15  # 15% improvement target
    alpha_level: float = 0.05  # Statistical significance
    power_target: float = 0.8  # Statistical power
    
    # Model configurations
    tiny_model: str = "ibm/granite-4-h-tiny"
    judge_model: str = "zai-org/GLM-4.6"
    
    # Testing approaches
    approaches: List[str] = None
    
    def __post_init__(self):
        if self.approaches is None:
            self.approaches = ["direct", "claims_based"]


@dataclass
class TestResult:
    """Result from a single test case execution"""
    
    test_id: str
    approach: str
    model: str
    question: str
    expected_answer: Optional[str]
    generated_answer: str
    execution_time: float
    token_usage: int
    
    # Evaluation metrics
    correctness: float
    completeness: float
    coherence: float
    reasoning_quality: float
    confidence_calibration: float
    efficiency: float
    hallucination_reduction: float
    
    # Claims-specific metrics
    claims_extracted: int
    confidence_scores: List[float]
    claim_consistency: float
    confidence_calibration_accuracy: float
    
    # Metadata
    timestamp: datetime
    difficulty: str
    reasoning_requirements: List[str]


@dataclass
class ExperimentResults:
    """Complete results from claims-based reasoning experiment"""
    
    experiment_id: str
    config: ExperimentConfig
    start_time: datetime
    end_time: Optional[datetime]
    
    # Test results
    direct_results: List[TestResult]
    claims_based_results: List[TestResult]
    
    # Statistical analysis
    statistical_significance: Dict[str, float]
    effect_sizes: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    
    # Performance metrics
    improvement_percentages: Dict[str, float]
    practical_significance: Dict[str, bool]
    
    # Claims-specific analysis
    confidence_calibration_analysis: Dict[str, Any]
    claims_consistency_analysis: Dict[str, Any]
    
    # Overall results
    hypothesis_validated: bool
    target_achieved: bool
    confidence_in_results: float


class ClaimsBasedReasoningExperiment:
    """Main experiment runner for claims-based reasoning hypothesis validation"""
    
    def __init__(self, config: ExperimentConfig = None):
        self.config = config or ExperimentConfig()
        
        # Directory setup
        self.experiments_dir = Path("experiments")
        self.results_dir = Path("experiments/results")
        self.test_cases_dir = Path("experiments/test_cases")
        self.reports_dir = Path("experiments/reports")
        
        for dir_path in [self.experiments_dir, self.results_dir, self.test_cases_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.llm_manager = None
        self.statistical_analyzer = None
        
        # Results storage
        self.results: ExperimentResults = None
        
        # Logging
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger("claims_based_reasoning_experiment")
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.results_dir / "claims_based_reasoning_experiment.log")
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
            self.statistical_analyzer = ConjectureStatisticalAnalyzer(str(self.results_dir))
            
            # Test connections
            for provider in provider_configs:
                test_result = await self.llm_manager.test_connection(provider)
                if not test_result.success:
                    self.logger.error(f"Failed to connect to {provider.model}: {test_result.error}")
                    return False
            
            self.logger.info("Claims-based reasoning experiment initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize experiment: {e}")
            return False
    
    def generate_claims_based_reasoning_test_cases(self) -> List[Dict[str, Any]]:
        """Generate 50-100 claims-based reasoning test cases for statistical significance"""
        
        self.logger.info(f"Generating {self.config.sample_size} claims-based reasoning test cases...")
    def _generate_evidence_evaluation_cases(self, count: int) -> List[Dict[str, Any]]:
        """Generate evidence evaluation test cases"""
        cases = []
        
        scenarios = [
            {
                "scenario": "A new study claims that meditation reduces stress by 40%",
                "evidence": [
                    {"source": "Study A", "finding": "Meditation reduces cortisol levels by 25%", "confidence": 0.8, "sample_size": 100},
                    {"source": "Study B", "finding": "No significant stress reduction from meditation", "confidence": 0.6, "sample_size": 50},
                    {"source": "Study C", "finding": "Meditation reduces self-reported stress by 35%", "confidence": 0.7, "sample_size": 200},
                    {"source": "Meta-analysis", "finding": "Average stress reduction of 20% across studies", "confidence": 0.9, "sample_size": 1000}
                ],
                "question": "Evaluate claim about meditation and stress reduction using provided evidence. Provide a structured analysis with confidence scores for each aspect."
            },
            {
                "scenario": "A company claims their new battery lasts 50% longer than competitors",
                "evidence": [
                    {"source": "Independent Lab Test", "finding": "45% longer battery life in controlled tests", "confidence": 0.85, "sample_size": 20},
                    {"source": "User Reviews", "finding": "30% longer battery life reported", "confidence": 0.6, "sample_size": 500},
                    {"source": "Competitor Analysis", "finding": "Only 15% improvement over market leader", "confidence": 0.7, "sample_size": 10},
                    {"source": "Company Internal Tests", "finding": "52% longer battery life", "confidence": 0.5, "sample_size": 100}
                ],
                "question": "Assess the company's battery claim using conflicting evidence. Provide claims with confidence scores."
            },
            {
                "scenario": "Researchers claim that artificial sweeteners cause weight gain",
                "evidence": [
                    {"source": "Epidemiological Study", "finding": "Correlation between artificial sweetener use and weight gain", "confidence": 0.7, "sample_size": 5000},
                    {"source": "Randomized Trial", "finding": "No significant weight difference between groups", "confidence": 0.8, "sample_size": 300},
                    {"source": "Animal Study", "finding": "Artificial sweeteners lead to increased appetite", "confidence": 0.6, "sample_size": 50},
                    {"source": "Meta-analysis", "finding": "No conclusive evidence for weight gain", "confidence": 0.9, "sample_size": 15000}
                ],
                "question": "Evaluate the claim about artificial sweeteners and weight gain. Analyze evidence quality and provide structured conclusions."
            }
        ]
        
        for i in range(min(count, len(scenarios))):
            scenario = scenarios[i % len(scenarios)]
            difficulty = "medium" if i % 3 == 0 else "hard"
            
            case = {
                "id": f"evidence_eval_{i+1:03d}",
                "category": "claims_based_reasoning",
                "difficulty": difficulty,
                "description": f"Evidence evaluation: {scenario['scenario'][:50]}...",
                "scenario": scenario["scenario"],
                "evidence": scenario["evidence"],
                "task": scenario["question"],
                "expected_approach": "evidence_synthesis",
                "key_reasoning_requirements": [
                    "evidence_quality_assessment",
                    "source_credibility_evaluation",
                    "confidence_calibration",
                    "conflict_resolution",
                    "structured_claims_formatting"
                ],
                "claims_benefits": [
                    "Improves evidence synthesis accuracy",
                    "Enhances confidence calibration", 
                    "Reduces confirmation bias",
                    "Provides transparent reasoning",
                    "Enables systematic evaluation"
                ],
                "metadata": {
                    "type": "evidence_evaluation",
                    "complexity_level": difficulty,
                    "estimated_time_minutes": 20,
                    "claims_based_approach_beneficial": True,
                    "confidence_critical": True
                }
            }
            cases.append(case)
        
        return cases
    
    def _generate_argument_analysis_cases(self, count: int) -> List[Dict[str, Any]]:
        """Generate argument analysis test cases"""
        cases = []
        
        scenarios = [
            {
                "argument": "Universal Basic Income (UBI) will eliminate poverty and boost economic growth",
                "premises": [
                    "UBI provides financial security to all citizens",
                    "Financial security reduces stress and improves health outcomes",
                    "People with basic needs met are more productive",
                    "Increased productivity leads to economic growth",
                    "Poverty rates will decrease with UBI implementation"
                ],
                "counterarguments": [
                    "UBI is too expensive to implement",
                    "It may reduce work incentives",
                    "Inflation could offset the benefits",
                    "Targeted programs may be more effective"
                ],
                "question": "Analyze the UBI argument using claims-based reasoning. Evaluate premises, counterarguments, and provide a structured assessment with confidence scores."
            },
            {
                "argument": "Remote work is more productive than office work for most knowledge workers",
                "premises": [
                    "Fewer office distractions increase focus",
                    "Flexible schedules improve work-life balance",
                    "Reduced commute time increases productivity",
                    "Access to preferred work environment",
                    "Technology enables effective collaboration remotely"
                ],
                "counterarguments": [
                    "Spontaneous collaboration decreases",
                    "Team cohesion may suffer",
                    "Home environment distractions increase",
                    "Career advancement opportunities may be limited"
                ],
                "question": "Evaluate the remote work productivity argument. Use structured claims to analyze premises and counterarguments with confidence scores."
            },
            {
                "argument": "Artificial intelligence will create more jobs than it eliminates",
                "premises": [
                    "AI creates new industries and roles",
                    "Automation increases economic efficiency",
                    "Historical precedents show job creation from technology",
                    "AI enables human workers to focus on higher-value tasks",
                    "New service industries will emerge around AI"
                ],
                "counterarguments": [
                    "AI automation is faster than job creation",
                    "Many jobs will be permanently eliminated",
                    "Skills gap may prevent workforce transition",
                    "Economic disruption could be severe"
                ],
                "question": "Analyze the AI job creation argument using claims-based reasoning. Assess both positive and negative claims with confidence scores."
            }
        ]
        
        for i in range(min(count, len(scenarios))):
            scenario = scenarios[i % len(scenarios)]
            difficulty = "medium" if i % 2 == 0 else "hard"
            
            case = {
                "id": f"argument_analysis_{i+1:03d}",
                "category": "claims_based_reasoning",
                "difficulty": difficulty,
                "description": f"Argument analysis: {scenario['argument'][:50]}...",
                "argument": scenario["argument"],
                "premises": scenario["premises"],
                "counterarguments": scenario["counterarguments"],
                "task": scenario["question"],
                "expected_approach": "structured_argument_analysis",
                "key_reasoning_requirements": [
                    "premise_evaluation",
                    "counterargument_analysis",
                    "logical_structure_assessment",
                    "confidence_scoring",
                    "claims_relationship_mapping"
                ],
                "claims_benefits": [
                    "Improves argument structure analysis",
                    "Enhances logical consistency checking",
                    "Provides transparent evaluation",
                    "Reduces bias in argument assessment",
                    "Enables systematic reasoning"
                ],
                "metadata": {
                    "type": "argument_analysis",
                    "complexity_level": difficulty,
                    "estimated_time_minutes": 25,
                    "claims_based_approach_beneficial": True,
                    "structured_reasoning_critical": True
                }
            }
            cases.append(case)
        
        return cases
    
    def _generate_scientific_claim_cases(self, count: int) -> List[Dict[str, Any]]:
        """Generate scientific claim evaluation test cases"""
        cases = []
        
        scenarios = [
            {
                "claim": "Regular consumption of blueberries improves cognitive function in adults",
                "research_data": {
                    "study_design": "Randomized controlled trial",
                    "duration": "12 weeks",
                    "sample_size": 200,
                    "participants": "Adults aged 25-65",
                    "intervention": "Daily blueberry consumption (150g)",
                    "control": "Placebo with similar appearance/taste",
                    "outcomes": {
                        "treatment_group": {"improvement": 15, "std_dev": 5, "p_value": 0.02},
                        "control_group": {"improvement": 5, "std_dev": 4, "p_value": 0.15}
                    },
                    "effect_size": 0.8,
                    "confidence_interval": [0.1, 0.3]
                },
                "question": "Evaluate scientific claim about blueberries and cognitive function. Analyze research methodology, statistical significance, and provide structured claims with confidence scores."
            },
            {
                "claim": "Intermittent fasting extends lifespan by reducing cellular damage",
                "research_data": {
                    "study_design": "Animal study with human observational data",
                    "duration": "2 years (animals), 5 years (human observation)",
                    "sample_size": {"animals": 100, "humans": 500},
                    "participants": "Mice and healthy adults",
                    "intervention": "16:8 fasting schedule",
                    "control": "Ad libitum feeding",
                    "outcomes": {
                        "animals": {"lifespan_extension": 15, "std_dev": 3, "p_value": 0.001},
                        "humans": {"biomarker_improvement": 20, "std_dev": 8, "p_value": 0.08}
                    },
                    "effect_size": 0.6,
                    "confidence_interval": [0.05, 0.25]
                },
                "question": "Assess the intermittent fasting lifespan claim. Evaluate evidence quality, generalize from animal to human data, and provide structured claims with confidence scores."
            },
            {
                "claim": "Meditation reduces stress hormones and improves immune function",
                "research_data": {
                    "study_design": "Meta-analysis of 50 studies",
                    "total_participants": 3000,
                    "study_types": ["RCT", "observational", "longitudinal"],
                    "intervention": "Various meditation practices",
                    "outcomes": {
                        "cortisol_reduction": {"mean": 18, "std_dev": 7, "p_value": 0.01},
                        "immune_markers": {"improvement": 12, "std_dev": 5, "p_value": 0.03},
                        "self_reported_stress": {"reduction": 25, "std_dev": 10, "p_value": 0.001}
                    },
                    "effect_size": 0.7,
                    "confidence_interval": [0.15, 0.35]
                },
                "question": "Evaluate the meditation stress reduction claim. Analyze meta-analysis methodology, effect sizes, and provide structured claims with confidence scores."
            }
        ]
        
        for i in range(min(count, len(scenarios))):
            scenario = scenarios[i % len(scenarios)]
            difficulty = "hard"  # Scientific claim evaluation is inherently complex
            
            case = {
                "id": f"scientific_claim_{i+1:03d}",
                "category": "claims_based_reasoning",
                "difficulty": difficulty,
                "description": f"Scientific claim evaluation: {scenario['claim'][:50]}...",
                "claim": scenario["claim"],
                "research_data": scenario["research_data"],
                "task": scenario["question"],
                "expected_approach": "scientific_evaluation",
                "key_reasoning_requirements": [
                    "methodology_assessment",
                    "statistical_analysis",
                    "evidence_quality_evaluation",
                    "confidence_calibration",
                    "scientific_reasoning"
                ],
                "claims_benefits": [
                    "Improves scientific claim evaluation",
                    "Enhances statistical reasoning",
                    "Provides transparent analysis",
                    "Reduces misinterpretation of data",
                    "Enables systematic evidence assessment"
                ],
                "metadata": {
                    "type": "scientific_evaluation",
                    "complexity_level": difficulty,
                    "estimated_time_minutes": 30,
                    "claims_based_approach_beneficial": True,
                    "scientific_accuracy_critical": True
                }
            }
            cases.append(case)
        
        return cases
        
        test_cases = []
        
        # Evidence evaluation scenarios (35 cases)
        evidence_scenarios = self._generate_evidence_evaluation_cases(35)
        test_cases.extend(evidence_scenarios)
        
        # Argument analysis scenarios (35 cases)
        argument_scenarios = self._generate_argument_analysis_cases(35)
        test_cases.extend(argument_scenarios)
        
        # Scientific claim evaluation scenarios (30 cases)
        scientific_scenarios = self._generate_scientific_claim_cases(30)
        test_cases.extend(scientific_scenarios)
        
        # Shuffle and limit to sample size
        import random
        random.shuffle(test_cases)
        test_cases = test_cases[:self.config.sample_size]
        
        # Save test cases
        test_cases_file = self.test_cases_dir / f"claims_based_reasoning_cases_{self.config.sample_size}.json"
        with open(test_cases_file, 'w', encoding='utf-8') as f:
            json.dump(test_cases, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Generated {len(test_cases)} claims-based reasoning test cases")
        return test_cases
    async def run_experiment(self) -> ExperimentResults:
        """Run complete claims-based reasoning experiment"""
        
        experiment_id = str(uuid.uuid4())[:8]
        start_time = datetime.utcnow()
        
        self.logger.info(f"Starting Claims-Based Reasoning Experiment: {experiment_id}")
        
        # Initialize results
        self.results = ExperimentResults(
            experiment_id=experiment_id,
            config=self.config,
            start_time=start_time,
            end_time=None,
            direct_results=[],
            claims_based_results=[],
            statistical_significance={},
            effect_sizes={},
            confidence_intervals={},
            improvement_percentages={},
            practical_significance={},
            confidence_calibration_analysis={},
            claims_consistency_analysis={},
            hypothesis_validated=False,
            target_achieved=False,
            confidence_in_results=0.0
        )
        
        try:
            # Generate test cases
            test_cases = self.generate_claims_based_reasoning_test_cases()
            self.logger.info(f"Generated {len(test_cases)} test cases")
            
            # Run direct approach tests
            self.logger.info("Running direct approach tests...")
            for i, test_case in enumerate(test_cases):
                self.logger.info(f"Direct test {i+1}/{len(test_cases)}: {test_case['id']}")
                result = await self._run_direct_test(test_case)
                if result:
                    self.results.direct_results.append(result)
            
            # Run claims-based approach tests
            self.logger.info("Running claims-based approach tests...")
            for i, test_case in enumerate(test_cases):
                self.logger.info(f"Claims-based test {i+1}/{len(test_cases)}: {test_case['id']}")
                result = await self._run_claims_based_test(test_case)
                if result:
                    self.results.claims_based_results.append(result)
            
            # Evaluate results using LLM-as-a-Judge
            self.logger.info("Evaluating results with LLM-as-a-Judge...")
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
            prompt = self._generate_direct_prompt(test_case)
            
            # Execute with tiny model
            start_time = time.time()
            response = await self.llm_manager.generate_response(
                prompt=prompt,
                model=self.config.tiny_model,
                max_tokens=2000,
                temperature=0.7
            )
            execution_time = time.time() - start_time
            
            # Create result
            result = TestResult(
                test_id=test_case["id"],
                approach="direct",
                model=self.config.tiny_model,
                question=test_case["task"],
                expected_answer=test_case.get("ground_truth", ""),
                generated_answer=response,
                execution_time=execution_time,
                token_usage=len(response.split()),  # Approximate
                correctness=0.0,  # Will be filled by evaluation
                completeness=0.0,
                coherence=0.0,
                reasoning_quality=0.0,
                confidence_calibration=0.0,
                efficiency=0.0,
                hallucination_reduction=0.0,
                claims_extracted=0,  # Direct approach doesn't use claims
                confidence_scores=[],
                claim_consistency=0.0,
                confidence_calibration_accuracy=0.0,
                timestamp=datetime.utcnow(),
                difficulty=test_case["difficulty"],
                reasoning_requirements=test_case.get("key_reasoning_requirements", [])
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Direct test failed for {test_case['id']}: {e}")
            return None
    
    async def _run_claims_based_test(self, test_case: Dict[str, Any]) -> Optional[TestResult]:
        """Run claims-based approach test"""
        try:
            # Generate claims-based prompt
            prompt = self._generate_claims_based_prompt(test_case)
            
            # Execute with tiny model
            start_time = time.time()
            response = await self.llm_manager.generate_response(
                prompt=prompt,
                model=self.config.tiny_model,
                max_tokens=2000,
                temperature=0.7
            )
            execution_time = time.time() - start_time
            
            # Extract claims and confidence scores
            claims_data = self._extract_claims_from_response(response)
            
            # Create result
            result = TestResult(
                test_id=test_case["id"],
                approach="claims_based",
                model=self.config.tiny_model,
                question=test_case["task"],
                expected_answer=test_case.get("ground_truth", ""),
                generated_answer=response,
                execution_time=execution_time,
                token_usage=len(response.split()),  # Approximate
                correctness=0.0,  # Will be filled by evaluation
                completeness=0.0,
                coherence=0.0,
                reasoning_quality=0.0,
                confidence_calibration=0.0,
                efficiency=0.0,
                hallucination_reduction=0.0,
                claims_extracted=claims_data["count"],
                confidence_scores=claims_data["confidence_scores"],
                claim_consistency=claims_data["consistency_score"],
                confidence_calibration_accuracy=0.0,  # Will be calculated after evaluation
                timestamp=datetime.utcnow(),
                difficulty=test_case["difficulty"],
                reasoning_requirements=test_case.get("key_reasoning_requirements", [])
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Claims-based test failed for {test_case['id']}: {e}")
            return None
    
    def _generate_direct_prompt(self, test_case: Dict[str, Any]) -> str:
        """Generate direct approach prompt"""
        return f"""
Please provide a comprehensive solution to the following task:

{test_case['task']}

Provide a detailed, well-structured response that addresses all aspects of the task. Be thorough and analytical in your approach.
"""
    
    def _generate_claims_based_prompt(self, test_case: Dict[str, Any]) -> str:
        """Generate claims-based approach prompt"""
        return f"""
You are using Conjecture's claims-based reasoning approach to solve a complex problem. Structure your reasoning as explicit claims with confidence scores.

**Task:**
{test_case['task']}

**Instructions:**
1. Break down the problem into 3-7 key claims
2. For each claim, provide a confidence score (0.0-1.0)
3. Show how claims relate to each other
4. Provide a final solution based on the claims

**Format your response using Conjecture's claim format:**
[c1 | claim content | / confidence]
[c2 | supporting claim | / confidence]
[c3 | subtask claim | / confidence]
etc.

Then provide your final comprehensive solution based on these claims.

**Key areas to address:**
{', '.join(test_case.get('key_reasoning_requirements', []))}
"""
    
    def _extract_claims_from_response(self, response: str) -> Dict[str, Any]:
        """Extract claims and confidence scores from response"""
        claims_pattern = r'\[c(\d+)\s*\|\s*([^|]+)\s*\|\s*/\s*([0-9.]+)\s*\]'
        matches = re.findall(claims_pattern, response)
        
        confidence_scores = []
        for match in matches:
            try:
                confidence = float(match[2])
                confidence_scores.append(confidence)
            except ValueError:
                confidence_scores.append(0.5)  # Default if parsing fails
        
        # Calculate consistency score based on confidence distribution
        consistency_score = 0.0
        if confidence_scores:
            # Check if confidence scores are reasonable (not all 0.0 or 1.0)
            unique_scores = set(confidence_scores)
            if len(unique_scores) > 1:
                consistency_score = 0.8  # Good variation
            elif confidence_scores[0] > 0.3 and confidence_scores[0] < 0.8:
                consistency_score = 0.6  # Moderate but reasonable
            else:
                consistency_score = 0.3  # Poor calibration
        
        return {
            "count": len(matches),
            "confidence_scores": confidence_scores,
            "consistency_score": consistency_score
        }
        
        return test_cases
    
    async def _evaluate_results(self):
        """Evaluate results using LLM-as-a-Judge"""
        judge_model = self.config.judge_model
        
        # Evaluate all results
        all_results = self.results.direct_results + self.results.claims_based_results
        
        for result in all_results:
            try:
                # Evaluate correctness
                correctness_prompt = self._create_evaluation_prompt(result, "correctness")
                correctness_response = await self.llm_manager.generate_response(
                    prompt=correctness_prompt,
                    model=judge_model,
                    max_tokens=500
                )
                correctness_score = self._parse_evaluation_score(correctness_response)
                
                # Evaluate confidence calibration
                calibration_prompt = self._create_evaluation_prompt(result, "confidence_calibration")
                calibration_response = await self.llm_manager.generate_response(
                    prompt=calibration_prompt,
                    model=judge_model,
                    max_tokens=500
                )
                calibration_score = self._parse_evaluation_score(calibration_response)
                
                # Evaluate other metrics
                completeness_prompt = self._create_evaluation_prompt(result, "completeness")
                completeness_response = await self.llm_manager.generate_response(
                    prompt=completeness_prompt,
                    model=judge_model,
                    max_tokens=500
                )
                completeness_score = self._parse_evaluation_score(completeness_response)
                
                coherence_prompt = self._create_evaluation_prompt(result, "coherence")
                coherence_response = await self.llm_manager.generate_response(
                    prompt=coherence_prompt,
                    model=judge_model,
                    max_tokens=500
                )
                coherence_score = self._parse_evaluation_score(coherence_response)
                
                reasoning_prompt = self._create_evaluation_prompt(result, "reasoning_quality")
                reasoning_response = await self.llm_manager.generate_response(
                    prompt=reasoning_prompt,
                    model=judge_model,
                    max_tokens=500
                )
                reasoning_score = self._parse_evaluation_score(reasoning_response)
                
                # Update result with evaluation scores
                result.correctness = correctness_score
                result.confidence_calibration = calibration_score
                result.completeness = completeness_score
                result.coherence = coherence_score
                result.reasoning_quality = reasoning_score
                
                # Calculate efficiency and hallucination reduction
                result.efficiency = min(1.0, 1000 / (result.token_usage + 1))  # Token efficiency
                result.hallucination_reduction = 1.0 - (result.execution_time / 60.0)  # Time efficiency
                
            except Exception as e:
                self.logger.error(f"Evaluation failed for {result.test_id}: {e}")
                # Set default scores
                result.correctness = 0.5
                result.confidence_calibration = 0.5
                result.completeness = 0.5
                result.coherence = 0.5
                result.reasoning_quality = 0.5
                result.efficiency = 0.5
                result.hallucination_reduction = 0.5
    
    def _create_evaluation_prompt(self, result: 'TestResult', metric: str) -> str:
        """Create evaluation prompt for LLM-as-a-Judge"""
        metric_instructions = {
            "correctness": "Evaluate factual accuracy and correctness of the answer",
            "confidence_calibration": "Evaluate how well confidence scores match actual accuracy",
            "completeness": "Evaluate how completely the answer addresses all aspects of the question",
            "coherence": "Evaluate logical flow and consistency of the reasoning",
            "reasoning_quality": "Evaluate depth, soundness, and quality of reasoning"
        }
        
        return f"""
You are an expert evaluator assessing AI model responses.

**Task:** {result.question}

**Model Response:**
{result.generated_answer}

**Approach:** {result.approach}

**Evaluation Metric:** {metric}
**Instructions:** {metric_instructions.get(metric, "")}

Provide a score from 0.0 to 1.0 where:
- 1.0 = Excellent performance on this metric
- 0.5 = Average performance
- 0.0 = Poor performance

Score: """
    
    def _parse_evaluation_score(self, response: str) -> float:
        """Parse evaluation score from LLM response"""
        try:
            # Look for numeric score in response
            import re
            score_match = re.search(r'([0-9]*\.?[0-9]+)', response)
            if score_match:
                score = float(score_match.group(1))
                # Ensure score is in valid range
                return max(0.0, min(1.0, score))
            return 0.5  # Default if no score found
        except Exception:
            return 0.5  # Default on error
    
    def _perform_statistical_analysis(self):
        """Perform statistical analysis on results"""
        if not self.results.direct_results or not self.results.claims_based_results:
            self.logger.warning("Insufficient data for statistical analysis")
            return
        
        # Extract metrics for comparison
        direct_metrics = {
            'correctness': [r.correctness for r in self.results.direct_results],
            'confidence_calibration': [r.confidence_calibration for r in self.results.direct_results],
            'completeness': [r.completeness for r in self.results.direct_results],
            'coherence': [r.coherence for r in self.results.direct_results],
            'reasoning_quality': [r.reasoning_quality for r in self.results.direct_results]
        }
        
        claims_metrics = {
            'correctness': [r.correctness for r in self.results.claims_based_results],
            'confidence_calibration': [r.confidence_calibration for r in self.results.claims_based_results],
            'completeness': [r.completeness for r in self.results.claims_based_results],
            'coherence': [r.coherence for r in self.results.claims_based_results],
            'reasoning_quality': [r.reasoning_quality for r in self.results.claims_based_results]
        }
        
        # Perform statistical tests for each metric
        for metric in direct_metrics.keys():
            if len(direct_metrics[metric]) >= 3 and len(claims_metrics[metric]) >= 3:
                # Paired t-test
                try:
                    t_stat, p_value = stats.ttest_rel(claims_metrics[metric], direct_metrics[metric])
                    
                    # Calculate effect size (Cohen's d)
                    mean_diff = statistics.mean(claims_metrics[metric]) - statistics.mean(direct_metrics[metric])
                    pooled_std = statistics.stdev(claims_metrics[metric] + direct_metrics[metric])
                    effect_size = mean_diff / pooled_std if pooled_std > 0 else 0.0
                    
                    # Calculate improvement percentage
                    direct_mean = statistics.mean(direct_metrics[metric])
                    claims_mean = statistics.mean(claims_metrics[metric])
                    improvement_pct = ((claims_mean - direct_mean) / direct_mean * 100) if direct_mean > 0 else 0.0
                    
                    # Store results
                    self.results.statistical_significance[metric] = p_value
                    self.results.effect_sizes[metric] = effect_size
                    self.results.improvement_percentages[metric] = improvement_pct
                    
                    # Confidence interval
                    se = pooled_std / (len(claims_metrics[metric]) ** 0.5)
                    margin = stats.t.ppf(1 - self.config.alpha_level / 2, len(claims_metrics[metric]) - 1) * se
                    ci_lower = mean_diff - margin
                    ci_upper = mean_diff + margin
                    self.results.confidence_intervals[metric] = (ci_lower, ci_upper)
                    
                    # Practical significance
                    self.results.practical_significance[metric] = improvement_pct >= (self.config.target_improvement * 100)
                    
                except Exception as e:
                    self.logger.error(f"Statistical analysis failed for {metric}: {e}")
        
        # Claims-specific analysis
        self._analyze_claims_specific_metrics()
    
    def _analyze_claims_specific_metrics(self):
        """Analyze claims-specific metrics"""
        claims_results = self.results.claims_based_results
        
        if not claims_results:
            return
        
        # Confidence calibration analysis
        all_confidence_scores = []
        for result in claims_results:
            all_confidence_scores.extend(result.confidence_scores)
        
        if all_confidence_scores:
            # Calculate calibration metrics
            mean_confidence = statistics.mean(all_confidence_scores)
            confidence_variance = statistics.stdev(all_confidence_scores) if len(all_confidence_scores) > 1 else 0.0
            
            self.results.confidence_calibration_analysis = {
                'mean_confidence': mean_confidence,
                'confidence_variance': confidence_variance,
                'confidence_range': (min(all_confidence_scores), max(all_confidence_scores)),
                'well_calibrated': 0.3 <= mean_confidence <= 0.8  # Reasonable confidence range
            }
        
        # Claims consistency analysis
        consistency_scores = [r.claim_consistency for r in claims_results if r.claim_consistency > 0]
        if consistency_scores:
            self.results.claims_consistency_analysis = {
                'mean_consistency': statistics.mean(consistency_scores),
                'consistency_variance': statistics.stdev(consistency_scores) if len(consistency_scores) > 1 else 0.0,
                'high_consistency_rate': len([s for s in consistency_scores if s >= 0.7]) / len(consistency_scores)
            }
    
    def _determine_hypothesis_validation(self):
        """Determine if hypothesis is validated"""
        # Primary metrics: correctness and confidence calibration
        primary_metrics = ['correctness', 'confidence_calibration']
        
        # Check if both primary metrics meet targets
        primary_improvements = [
            self.results.improvement_percentages.get(metric, 0) >= (self.config.target_improvement * 100)
            for metric in primary_metrics
        ]
        
        # Check statistical significance
        primary_significant = [
            self.results.statistical_significance.get(metric, 1.0) < self.config.alpha_level
            for metric in primary_metrics
        ]
        
        # Overall hypothesis validation
        self.results.hypothesis_validated = all(primary_improvements) and all(primary_significant)
        
        # Target achievement (15%+ improvement)
        self.results.target_achieved = all(primary_improvements)
        
        # Confidence in results
        significant_metrics = len([m for m in primary_metrics if self.results.statistical_significance.get(m, 1.0) < self.config.alpha_level])
        self.results.confidence_in_results = significant_metrics / len(primary_metrics)
    
    async def _save_results(self):
        """Save experiment results to file"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"claims_based_reasoning_results_{self.results.experiment_id}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Convert results to serializable format
        results_data = {
            'experiment_id': self.results.experiment_id,
            'config': asdict(self.results.config),
            'start_time': self.results.start_time.isoformat(),
            'end_time': self.results.end_time.isoformat() if self.results.end_time else None,
            'direct_results': [asdict(r) for r in self.results.direct_results],
            'claims_based_results': [asdict(r) for r in self.results.claims_based_results],
            'statistical_significance': self.results.statistical_significance,
            'effect_sizes': self.results.effect_sizes,
            'confidence_intervals': {k: (v[0], v[1]) for k, v in self.results.confidence_intervals.items()},
            'improvement_percentages': self.results.improvement_percentages,
            'practical_significance': self.results.practical_significance,
            'confidence_calibration_analysis': self.results.confidence_calibration_analysis,
            'claims_consistency_analysis': self.results.claims_consistency_analysis,
            'hypothesis_validated': self.results.hypothesis_validated,
            'target_achieved': self.results.target_achieved,
            'confidence_in_results': self.results.confidence_in_results
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
    
    async def _generate_report(self):
        """Generate comprehensive experiment report"""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        
        report_lines = [
            "# Claims-Based Reasoning Experiment Report",
            f"Generated: {timestamp}",
            f"Experiment ID: {self.results.experiment_id}",
            "",
            "## Executive Summary",
            "",
            f"**Hypothesis:** Claims-based reasoning will show 15%+ improvement in correctness and confidence calibration",
            f"**Target Improvement:** {self.config.target_improvement * 100}%",
            f"**Sample Size:** {len(self.results.direct_results)} test cases per approach",
            f"**Model:** {self.config.tiny_model}",
            f"**Judge Model:** {self.config.judge_model}",
            "",
            f"**Hypothesis Validated:** {'✅ YES' if self.results.hypothesis_validated else '❌ NO'}",
            f"**Target Achieved:** {'✅ YES' if self.results.target_achieved else '❌ NO'}",
            f"**Confidence in Results:** {self.results.confidence_in_results:.2%}",
            "",
            "## Performance Improvements",
            ""
        ]
        
        # Add improvement results
        for metric in ['correctness', 'confidence_calibration', 'completeness', 'coherence', 'reasoning_quality']:
            if metric in self.results.improvement_percentages:
                improvement = self.results.improvement_percentages[metric]
                p_value = self.results.statistical_significance.get(metric, None)
                effect_size = self.results.effect_sizes.get(metric, None)
                significant = "✅ Significant" if p_value and p_value < self.config.alpha_level else "❌ Not significant"
                target_met = "✅ Yes" if improvement >= (self.config.target_improvement * 100) else "❌ No"
                
                report_lines.extend([
                    f"### {metric.replace('_', ' ').title()}",
                    f"- **Improvement:** {improvement:.1f}%",
                    f"- **Statistical Significance:** {significant} (p={p_value:.4f})" if p_value else "- **Statistical Significance:** Unable to calculate",
                    f"- **Effect Size:** {effect_size:.3f}" if effect_size else "- **Effect Size:** Unable to calculate",
                    f"- **Target Met (15%+):** {target_met}",
                    ""
                ])
        
        # Add claims-specific analysis
        if self.results.confidence_calibration_analysis:
            report_lines.extend([
                "## Confidence Calibration Analysis",
                f"- **Mean Confidence:** {self.results.confidence_calibration_analysis['mean_confidence']:.3f}",
                f"- **Confidence Variance:** {self.results.confidence_calibration_analysis['confidence_variance']:.3f}",
                f"- **Well Calibrated:** {'✅ Yes' if self.results.confidence_calibration_analysis['well_calibrated'] else '❌ No'}",
                ""
            ])
        
        if self.results.claims_consistency_analysis:
            report_lines.extend([
                "## Claims Consistency Analysis",
                f"- **Mean Consistency:** {self.results.claims_consistency_analysis['mean_consistency']:.3f}",
                f"- **High Consistency Rate:** {self.results.claims_consistency_analysis['high_consistency_rate']:.1%}",
                ""
            ])
        
        # Add conclusions
        report_lines.extend([
            "## Conclusions",
            ""
        ])
        
        if self.results.hypothesis_validated:
            report_lines.extend([
                "✅ **HYPOTHESIS VALIDATED**: Claims-based reasoning shows statistically significant improvement of 15%+ in both correctness and confidence calibration.",
                "",
                "The results support the core hypothesis that structured claims-based reasoning enhances tiny LLM performance.",
                "This validates Conjecture's approach as an effective method for improving AI reasoning quality."
            ])
        else:
            report_lines.extend([
                "❌ **HYPOTHESIS NOT VALIDATED**: Claims-based reasoning did not achieve the target 15%+ improvement in both primary metrics.",
                "",
                "While some improvements may be observed, they do not meet the statistical significance or practical significance thresholds.",
                "Further refinement of the claims-based approach may be needed."
            ])
        
        # Add recommendations
        report_lines.extend([
            "",
            "## Recommendations",
            ""
        ])
        
        if self.results.hypothesis_validated:
            report_lines.extend([
                "1. **Deploy claims-based reasoning** in production for improved tiny model performance",
                "2. **Expand to additional models** to test generalizability",
                "3. **Optimize claim format** for even better performance",
                "4. **Investigate specific domains** where benefits are largest"
            ])
        else:
            report_lines.extend([
                "1. **Refine claim format** and confidence scoring mechanism",
                "2. **Increase sample size** for better statistical power",
                "3. **Test on different model families**",
                "4. **Investigate hybrid approaches** combining claims with other methods"
            ])
        
        # Save report
        report_content = "\n".join(report_lines)
        report_filename = f"claims_based_reasoning_report_{self.results.experiment_id}_{timestamp.replace(':', '-')}.md"
        report_filepath = self.reports_dir / report_filename
        
        try:
            with open(report_filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            self.logger.info(f"Report saved to {report_filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")


async def main():
    """Main function to run the claims-based reasoning experiment"""
    
    # Load configuration
    config = ExperimentConfig(
        sample_size=75,  # Target 50-100 test cases
        target_improvement=0.15,  # 15% improvement target
        alpha_level=0.05,  # Statistical significance
        power_target=0.8,  # Statistical power
        tiny_model="lms/granite-4-h-tiny",  # From config
        judge_model="zai/GLM-4.6"  # From config
    )
    
    # Initialize experiment
    experiment = ClaimsBasedReasoningExperiment(config)
    
    # Load provider configurations
    provider_configs = [
        ProviderConfig(
            url="http://localhost:1234",
            api_key="",
            model="lms/granite-4-h-tiny"
        ),
        ProviderConfig(
            url="https://api.z.ai/api/coding/paas/v4",
            api_key="70e6e12e4d7c46e2a4d0b85503d51f38.LQHl8d98kDJChttb",
            model="zai/GLM-4.6"
        )
    ]
    
    try:
        # Initialize experiment
        if not await experiment.initialize(provider_configs):
            print("Failed to initialize experiment")
            return
        
        # Run experiment
        results = await experiment.run_experiment()
        
        # Print summary
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETED")
        print("="*60)
        print(f"Hypothesis Validated: {'✅ YES' if results.hypothesis_validated else '❌ NO'}")
        print(f"Target Achieved: {'✅ YES' if results.target_achieved else '❌ NO'}")
        print(f"Confidence in Results: {results.confidence_in_results:.2%}")
        
        if results.improvement_percentages:
            print("\nKey Improvements:")
            for metric, improvement in results.improvement_percentages.items():
                if metric in ['correctness', 'confidence_calibration']:
                    target_met = "✅" if improvement >= 15 else "❌"
                    print(f"  {metric}: {improvement:.1f}% {target_met}")
        
        print(f"\nDetailed report saved to: experiments/reports/")
        print(f"Raw data saved to: experiments/results/")
        
    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())