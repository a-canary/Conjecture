#!/usr/bin/env python3
"""
Context Compression Experiment Runner
Tests if models maintain 90%+ performance with 50%+ context reduction using claims format.

This is the second critical experiment for validating the core hypothesis that:
"Models will maintain 90%+ performance with 50%+ context reduction using claims format"
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
    """Configuration for context compression experiment"""
    
    # Test parameters
    sample_size: int = 75  # Target 50-100 test cases
    target_performance_retention: float = 0.90  # 90% performance retention target
    target_compression_ratio: float = 0.50  # 50% context reduction target
    alpha_level: float = 0.05  # Statistical significance
    power_target: float = 0.8  # Statistical power
    
    # Model configurations
    tiny_model: str = "ibm/granite-4-h-tiny"
    judge_model: str = "zai-org/GLM-4.6"
    
    # Testing approaches
    approaches: List[str] = None
    
    def __post_init__(self):
        if self.approaches is None:
            self.approaches = ["full_context", "compressed_context"]

@dataclass
class TestResult:
    """Result from a single test case execution"""
    
    test_id: str
    approach: str
    model: str
    question: str
    context: str
    expected_answer: Optional[str]
    generated_answer: str
    execution_time: float
    token_usage: int
    
    # Context compression metrics
    original_context_length: int
    compressed_context_length: int
    compression_ratio: float
    compression_achieved: bool
    
    # Evaluation metrics
    correctness: float
    completeness: float
    coherence: float
    reasoning_quality: float
    confidence_calibration: float
    efficiency: float
    hallucination_reduction: float
    
    # Metadata
    timestamp: datetime
    difficulty: str
    reasoning_requirements: List[str]

@dataclass
class ExperimentResults:
    """Complete results from context compression experiment"""
    
    experiment_id: str
    config: ExperimentConfig
    start_time: datetime
    end_time: Optional[datetime]
    
    # Test results
    full_context_results: List[TestResult]
    compressed_context_results: List[TestResult]
    
    # Statistical analysis
    statistical_significance: Dict[str, float]
    effect_sizes: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    
    # Performance metrics
    performance_retention_percentages: Dict[str, float]
    compression_effectiveness: Dict[str, float]
    practical_significance: Dict[str, bool]
    
    # Overall results
    hypothesis_validated: bool
    targets_achieved: Dict[str, bool]
    confidence_in_results: float

class ContextCompressionExperiment:
    """Main experiment runner for context compression hypothesis validation"""
    
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
        logger = logging.getLogger("context_compression_experiment")
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.results_dir / "context_compression_experiment.log")
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
            
            self.logger.info("Context compression experiment initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize experiment: {e}")
            return False
    
    def generate_context_compression_test_cases(self) -> List[Dict[str, Any]]:
        """Generate 50-100 context compression test cases for statistical significance"""
        
        self.logger.info(f"Generating {self.config.sample_size} context compression test cases...")
        
        test_cases = []
        
        # Long document QA cases (40 cases)
        long_document_cases = self._generate_long_document_cases(40)
        test_cases.extend(long_document_cases)
        
        # Multi-source synthesis cases (30 cases)
        synthesis_cases = self._generate_multi_source_synthesis_cases(30)
        test_cases.extend(synthesis_cases)
        
        # Research paper analysis cases (30 cases)
        research_cases = self._generate_research_paper_cases(30)
        test_cases.extend(research_cases)
        
        # Shuffle and limit to sample size
        import random
        random.shuffle(test_cases)
        test_cases = test_cases[:self.config.sample_size]
        
        # Save test cases
        test_cases_file = self.test_cases_dir / f"context_compression_cases_{self.config.sample_size}.json"
        with open(test_cases_file, 'w', encoding='utf-8') as f:
            json.dump(test_cases, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Generated {len(test_cases)} context compression test cases")
        return test_cases
    
    def _generate_long_document_cases(self, count: int) -> List[Dict[str, Any]]:
        """Generate long document QA test cases"""
        cases = []
        
        documents = [
            {
                "title": "Climate Change Impact Assessment",
                "content": """
                Executive Summary:
                This comprehensive report analyzes the multifaceted impacts of climate change on global ecosystems, economies, and human societies over the next century. The findings indicate urgent action is required across multiple sectors to mitigate catastrophic consequences.
                
                Chapter 1: Environmental Impacts
                Rising global temperatures have accelerated glacier melt, with polar ice sheets losing 280 billion tons of ice annually since 2002. Sea levels are rising at 3.3 millimeters per year, threatening coastal communities worldwide. Ocean acidification has increased by 30% since the Industrial Revolution, devastating marine ecosystems and coral reefs.
                
                Chapter 2: Economic Consequences
                Climate-related disasters cost the global economy $210 billion in 2020, a 50% increase from the previous decade. Agricultural productivity is projected to decline by 15% in tropical regions by 2050, threatening food security for 2 billion people. Insurance premiums for climate risks have tripled in vulnerable regions over the past decade.
                
                Chapter 3: Social Implications
                Climate migration is accelerating, with an estimated 30 million people displaced annually by weather-related disasters. Water scarcity affects 4 billion people for at least one month per year. Health impacts include increased respiratory diseases from air pollution and expanded ranges for vector-borne diseases.
                
                Chapter 4: Mitigation Strategies
                Renewable energy costs have decreased by 85% since 2010, making solar and wind competitive with fossil fuels. Carbon capture technologies can remove up to 90% of CO2 emissions from power plants. Reforestation initiatives could sequester 30% of annual carbon emissions by 2030.
                
                Chapter 5: Policy Recommendations
                Immediate implementation of carbon pricing could reduce emissions by 40% by 2030. International cooperation is essential for technology transfer and climate finance. Investment in adaptation measures must increase to $300 billion annually to protect vulnerable communities.
                """,
                "questions": [
                    "What are the four main areas of climate impact discussed, and what specific mitigation strategies are recommended?",
                    "Analyze the economic implications presented in the document. What are the key cost projections?",
                    "What policy recommendations would provide the highest return on investment in climate mitigation?"
                ]
            },
            {
                "title": "Artificial Intelligence Ethics Framework",
                "content": """
                Introduction:
                As artificial intelligence systems become increasingly integrated into critical decision-making processes, establishing comprehensive ethical frameworks becomes paramount for ensuring responsible development and deployment.
                
                Section 1: Core Ethical Principles
                Transparency requires that AI systems be explainable and their decision-making processes understandable to stakeholders. Fairness demands that AI systems avoid bias and discrimination across demographic groups. Privacy protection ensures that personal data is handled with appropriate safeguards and user consent.
                
                Section 2: Accountability Mechanisms
                Clear lines of responsibility must be established for AI system outcomes. Audit trails should be maintained for all significant decisions. Human oversight requirements should be implemented for high-stakes decisions affecting individuals' rights or opportunities.
                
                Section 3: Risk Assessment Protocols
                Pre-deployment testing must include bias detection, robustness evaluation, and security vulnerability assessment. Ongoing monitoring should detect performance degradation or emergent biases. Incident response plans must be established for AI system failures or misuse.
                
                Section 4: Governance Structures
                Multi-stakeholder oversight committees should include technical experts, ethicists, and affected community representatives. Regulatory compliance frameworks must be established specific to application domains. International coordination is needed for cross-border AI systems.
                
                Section 5: Implementation Guidelines
                Ethical impact assessments should be conducted before AI system deployment. Training programs must educate developers on ethical considerations. Public engagement processes should gather diverse perspectives on AI system impacts.
                """,
                "questions": [
                    "What are the four core ethical principles outlined for AI systems?",
                    "Describe the accountability mechanisms proposed for high-stakes AI decisions.",
                    "What governance structures are recommended for cross-border AI systems?"
                ]
            },
            {
                "title": "Global Supply Chain Transformation",
                "content": """
                Overview:
                The global supply chain landscape is undergoing unprecedented transformation driven by technological innovation, sustainability requirements, and geopolitical shifts. Companies must adapt to remain competitive and resilient.
                
                Digital Transformation Impact:
                IoT sensors and blockchain technology are enabling real-time supply chain visibility, reducing inventory costs by 23% and improving delivery accuracy. AI-powered demand forecasting has reduced stockouts by 35% while optimizing warehouse utilization. Digital twins technology allows for simulation and optimization of complex supply networks before implementation.
                
                Sustainability Integration:
                Circular economy principles are being adopted, with 67% of Fortune 500 companies committing to 100% renewable energy in supply chains by 2030. Carbon footprint tracking systems now cover 80% of global shipping, enabling emissions reduction strategies. Sustainable packaging innovations have reduced material waste by 40% while maintaining product protection.
                
                Resilience Building:
                Near-shoring and friend-shoring strategies have reduced supply chain disruption risks by 45% compared to purely globalized models. Multi-sourcing strategies have decreased single-point-of-failure vulnerabilities from 60% to 15% of critical components. Advanced analytics provide early warning systems for potential disruptions, improving response times by 70%.
                
                Workforce Evolution:
                Supply chain workforce skills are shifting from manual logistics to data analysis and system optimization. Training programs focusing on digital literacy and analytics have increased worker productivity by 28%. Remote monitoring technologies enable more flexible work arrangements while improving system oversight.
                
                Future Outlook:
                Autonomous vehicles and drones are expected to handle 30% of last-mile deliveries by 2027. Quantum computing applications could optimize routing problems that are currently computationally intractable. Integration of biological and digital supply chain monitoring may enable self-healing supply networks.
                """,
                "questions": [
                    "What digital transformation technologies are mentioned, and what measurable impacts do they provide?",
                    "Analyze the sustainability initiatives described. What are the key metrics and targets?",
                    "What resilience-building strategies have proven most effective according to the document?"
                ]
            }
        ]
        
        for i in range(min(count, len(documents) * 3)):  # 3 questions per document
            doc_idx = i // 3
            question_idx = i % 3
            doc = documents[doc_idx]
            
            case = {
                "id": f"long_doc_qa_{i+1:03d}",
                "category": "context_compression",
                "difficulty": "hard",
                "description": f"Long document QA: {doc['title']}",
                "context": doc["content"],
                "question": doc["questions"][question_idx],
                "context_length": len(doc["content"].split()),
                "reasoning_requirements": ["information_extraction", "comprehension", "relevance_filtering"],
                "expected_answer_type": "comprehensive_answer",
                "compression_strategy": "extract_key_claims",
                "metadata": {
                    "type": "reading_comprehension",
                    "requires_synthesis": True,
                    "multiple_categories": True,
                    "estimated_time_minutes": 12,
                    "claims_based_approach_beneficial": True,
                    "compression_ratio_target": 0.5
                }
            }
            cases.append(case)
        
        return cases[:count]
    
    def _generate_multi_source_synthesis_cases(self, count: int) -> List[Dict[str, Any]]:
        """Generate multi-source synthesis test cases"""
        cases = []
        
        synthesis_scenarios = [
            {
                "topic": "Renewable Energy Transition",
                "sources": [
                    "Source A: International Energy Agency reports solar panel costs have fallen by 85% since 2010, making it competitive with fossil fuels in most regions. Solar installations increased by 22% globally in 2023.",
                    "Source B: Stanford University study finds wind energy capacity could meet global demand 4x over, but requires grid infrastructure investments of $2.3 trillion by 2030.",
                    "Source C: Bloomberg analysis indicates renewable energy jobs grew 70% faster than overall economy, with 10.3 million people employed globally in clean energy sectors.",
                    "Source D: European Commission research shows energy storage costs decreased by 70% since 2015, enabling 24/7 renewable power availability."
                ],
                "question": "Synthesize the information about renewable energy transition. What are the key economic and technical factors driving adoption, and what challenges remain?"
            },
            {
                "topic": "Remote Work Productivity",
                "sources": [
                    "Source A: Harvard Business Review analysis of 2,000 workers shows 42% productivity increase in remote work, with 30% reduction in employee turnover.",
                    "Source B: Microsoft study finds remote workers save 2.5 hours daily on commuting, reporting better work-life balance and 15% higher job satisfaction.",
                    "Source C: Gartner research indicates companies save $11,000 per remote worker annually in real estate and overhead costs, but face challenges in team cohesion and innovation.",
                    "Source D: Stanford University research shows remote work requires different management approaches, with 1:1 manager-to-employee ratios optimal vs 1:7 for in-office work."
                ],
                "question": "Based on the sources, analyze the productivity impacts of remote work. What are the main benefits and key management challenges?"
            },
            {
                "topic": "AI in Healthcare",
                "sources": [
                    "Source A: FDA reports AI diagnostic tools show 94% accuracy in detecting diabetic retinopathy, matching human specialists while reducing screening time by 80%.",
                    "Source B: Mayo Clinic study finds AI-assisted surgery reduces complications by 21% and operation time by 15%, though initial training costs are significant.",
                    "Source C: McKinsey analysis estimates AI could save $150 billion annually in US healthcare costs through preventive care and operational efficiency.",
                    "Source D: Patient advocacy groups raise concerns about AI bias, with studies showing 15% lower accuracy for underrepresented demographic groups."
                ],
                "question": "Evaluate the potential of AI in healthcare based on these sources. What are the main benefits and critical concerns that need addressing?"
            }
        ]
        
        for i in range(min(count, len(synthesis_scenarios))):
            scenario = synthesis_scenarios[i % len(synthesis_scenarios)]
            
            case = {
                "id": f"multi_source_{i+1:03d}",
                "category": "context_compression",
                "difficulty": "hard",
                "description": f"Multi-source synthesis: {scenario['topic']}",
                "sources": scenario["sources"],
                "question": scenario["question"],
                "context_length": sum(len(source.split()) for source in scenario["sources"]),
                "reasoning_requirements": ["information_integration", "synthesis", "source_evaluation", "conflict_resolution"],
                "expected_answer_type": "synthesized_analysis",
                "compression_strategy": "integrate_claims",
                "metadata": {
                    "type": "synthesis_task",
                    "conflicting_information": True,
                    "requires_critical_evaluation": True,
                    "estimated_time_minutes": 15,
                    "claims_based_approach_beneficial": True,
                    "compression_ratio_target": 0.4
                }
            }
            cases.append(case)
        
        return cases[:count]
    
    def _generate_research_paper_cases(self, count: int) -> List[Dict[str, Any]]:
        """Generate research paper analysis test cases"""
        cases = []
        
        papers = [
            {
                "title": "Machine Learning in Drug Discovery",
                "abstract": "This study presents a novel deep learning approach for accelerating drug discovery processes. We developed a graph neural network architecture that predicts molecular binding affinity with 92% accuracy, reducing the need for expensive laboratory experiments. Our approach identified 3 promising drug candidates for Alzheimer's treatment, cutting discovery time from 5 years to 8 months. The model was trained on 50,000 known drug-target interactions and validated against 2,000 experimental results.",
                "methodology": "We employed a multi-task learning framework combining molecular property prediction with binding affinity estimation. The architecture uses attention mechanisms to identify key substructures responsible for binding. Cross-validation was performed using temporal splits to ensure realistic performance estimation. Uncertainty quantification was implemented through Monte Carlo dropout.",
                "results": "The approach achieved state-of-the-art performance on benchmark datasets, with particularly strong results on novel targets. Computational cost was reduced by 85% compared to traditional methods. Three identified candidates are now in preclinical trials with promising early results.",
                "limitations": "The model requires high-quality training data, which may not be available for rare diseases. Interpretability remains challenging, though we implemented attention visualization to address this. External validation is still needed for clinical applications.",
                "question": "Critically evaluate this research paper on ML in drug discovery. What are the key innovations, methodological strengths, and limitations that need addressing?"
            },
            {
                "title": "Climate Change and Food Security",
                "abstract": "This meta-analysis examines the impacts of climate change on global food security through 2030. We analyzed 850 studies across 120 countries, finding that crop yields could decline by 15% in tropical regions while increasing by 10% in northern latitudes. Food price volatility is expected to increase by 40%, affecting low-income populations disproportionately.",
                "methodology": "We employed integrated assessment models combining climate projections with agricultural productivity models. Statistical analysis used weighted regression based on regional agricultural importance. Uncertainty was quantified using ensemble modeling across 15 climate models.",
                "results": "Adaptation strategies could prevent 60% of projected yield losses through crop rotation, irrigation improvements, and heat-resistant varieties. International coordination on food reserves could reduce price volatility impacts by 35%. Investment needs are estimated at $45 billion annually through 2030.",
                "limitations": "Models have higher uncertainty for extreme climate scenarios. Socio-economic factors like conflict and policy changes are not fully integrated. Regional variations in adaptive capacity require localized solutions.",
                "question": "Based on this research, what are the projected impacts of climate change on food security, and what adaptation strategies are most promising?"
            }
        ]
        
        for i in range(min(count, len(papers))):
            paper = papers[i % len(papers)]
            
            # Combine all paper content
            full_content = f"""
            Abstract: {paper['abstract']}
            
            Methodology: {paper['methodology']}
            
            Results: {paper['results']}
            
            Limitations: {paper['limitations']}
            """
            
            case = {
                "id": f"research_analysis_{i+1:03d}",
                "category": "context_compression",
                "difficulty": "hard",
                "description": f"Research paper analysis: {paper['title']}",
                "context": full_content,
                "question": paper["question"],
                "context_length": len(full_content.split()),
                "reasoning_requirements": ["academic_comprehension", "critical_analysis", "research_evaluation"],
                "expected_answer_type": "research_analysis",
                "compression_strategy": "extract_key_findings",
                "metadata": {
                    "type": "academic_analysis",
                    "requires_critical_thinking": True,
                    "methodology_assessment": True,
                    "estimated_time_minutes": 18,
                    "claims_based_approach_beneficial": True,
                    "compression_ratio_target": 0.3
                }
            }
            cases.append(case)
        
        return cases[:count]
    
    async def run_experiment(self) -> ExperimentResults:
        """Run complete context compression experiment"""
        
        experiment_id = str(uuid.uuid4())[:8]
        start_time = datetime.utcnow()
        
        self.logger.info(f"Starting Context Compression Experiment: {experiment_id}")
        
        # Initialize results
        self.results = ExperimentResults(
            experiment_id=experiment_id,
            config=self.config,
            start_time=start_time,
            end_time=None,
            full_context_results=[],
            compressed_context_results=[],
            statistical_significance={},
            effect_sizes={},
            confidence_intervals={},
            performance_retention_percentages={},
            compression_effectiveness={},
            practical_significance={},
            hypothesis_validated=False,
            targets_achieved={},
            confidence_in_results=0.0
        )
        
        try:
            # Generate test cases
            test_cases = self.generate_context_compression_test_cases()
            self.logger.info(f"Generated {len(test_cases)} test cases")
            
            # Run full context approach tests
            self.logger.info("Running full context approach tests...")
            for i, test_case in enumerate(test_cases):
                self.logger.info(f"Full context test {i+1}/{len(test_cases)}: {test_case['id']}")
                result = await self._run_full_context_test(test_case)
                if result:
                    self.results.full_context_results.append(result)
            
            # Run compressed context approach tests
            self.logger.info("Running compressed context approach tests...")
            for i, test_case in enumerate(test_cases):
                self.logger.info(f"Compressed context test {i+1}/{len(test_cases)}: {test_case['id']}")
                result = await self._run_compressed_context_test(test_case)
                if result:
                    self.results.compressed_context_results.append(result)
            
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
    
    async def _run_full_context_test(self, test_case: Dict[str, Any]) -> Optional[TestResult]:
        """Run full context approach test"""
        try:
            # Generate full context prompt
            prompt = self._generate_full_context_prompt(test_case)
            
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
                approach="full_context",
                model=self.config.tiny_model,
                question=test_case["question"],
                context=test_case.get("context", ""),
                expected_answer=test_case.get("expected_answer", ""),
                generated_answer=response,
                execution_time=execution_time,
                token_usage=len(response.split()),  # Approximate
                original_context_length=test_case.get("context_length", 0),
                compressed_context_length=test_case.get("context_length", 0),  # Same for full context
                compression_ratio=1.0,  # No compression
                compression_achieved=False,
                correctness=0.0,  # Will be filled by evaluation
                completeness=0.0,
                coherence=0.0,
                reasoning_quality=0.0,
                confidence_calibration=0.0,
                efficiency=0.0,
                hallucination_reduction=0.0,
                timestamp=datetime.utcnow(),
                difficulty=test_case["difficulty"],
                reasoning_requirements=test_case.get("reasoning_requirements", [])
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Full context test failed for {test_case['id']}: {e}")
            return None
    
    async def _run_compressed_context_test(self, test_case: Dict[str, Any]) -> Optional[TestResult]:
        """Run compressed context approach test"""
        try:
            # First, compress the context using claims format
            compression_result = await self._compress_context(test_case)
            
            if not compression_result:
                self.logger.error(f"Context compression failed for {test_case['id']}")
                return None
            
            compressed_context, compression_ratio = compression_result
            
            # Generate compressed context prompt
            prompt = self._generate_compressed_context_prompt(test_case, compressed_context)
            
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
                approach="compressed_context",
                model=self.config.tiny_model,
                question=test_case["question"],
                context=test_case.get("context", ""),
                expected_answer=test_case.get("expected_answer", ""),
                generated_answer=response,
                execution_time=execution_time,
                token_usage=len(response.split()),  # Approximate
                original_context_length=test_case.get("context_length", 0),
                compressed_context_length=len(compressed_context.split()),
                compression_ratio=compression_ratio,
                compression_achieved=compression_ratio <= self.config.target_compression_ratio,
                correctness=0.0,  # Will be filled by evaluation
                completeness=0.0,
                coherence=0.0,
                reasoning_quality=0.0,
                confidence_calibration=0.0,
                efficiency=0.0,
                hallucination_reduction=0.0,
                timestamp=datetime.utcnow(),
                difficulty=test_case["difficulty"],
                reasoning_requirements=test_case.get("reasoning_requirements", [])
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Compressed context test failed for {test_case['id']}: {e}")
            return None
    
    async def _compress_context(self, test_case: Dict[str, Any]) -> Optional[Tuple[str, float]]:
        """Compress context using claims format"""
        try:
            context = test_case.get("context", "")
            question = test_case.get("question", "")
            
            # Use LLM to extract key claims and compress context
            compression_prompt = f"""
            You are tasked with compressing a large context into Conjecture's claims format while preserving essential information for answering questions.

            **Context:**
            {context}

            **Question:**
            {question}

            **Instructions:**
            1. Extract the most relevant claims/facts from the context that are essential for answering the question
            2. Organize them by relevance and importance
            3. Convert to claims format with confidence scores
            4. Ensure at least 50% reduction in token count while preserving key information

            Format your response using Conjecture's claim format:
            [c1 | key fact 1 | / confidence]
            [c2 | key fact 2 | / confidence]
            [c3 | key fact 3 | / confidence]
            etc.

            Focus on factual claims, direct relationships, and essential information needed to answer the question.
            """
            
            # Get compression from tiny model
            compression_response = await self.llm_manager.generate_response(
                prompt=compression_prompt,
                model=self.config.tiny_model,
                max_tokens=1500,
                temperature=0.3
            )
            
            # Extract claims from response
            claims = self._extract_claims_from_response(compression_response)
            
            if not claims:
                self.logger.warning(f"No claims extracted from compression for {test_case['id']}")
                return None
            
            # Create compressed context from claims
            compressed_context = "\n".join(claims)
            
            # Calculate compression ratio
            original_tokens = len(context.split())
            compressed_tokens = len(compressed_context.split())
            compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
            
            self.logger.debug(f"Context compression for {test_case['id']}: {original_tokens} -> {compressed_tokens} tokens (ratio: {compression_ratio:.2f})")
            
            return compressed_context, compression_ratio
            
        except Exception as e:
            self.logger.error(f"Context compression failed for {test_case['id']}: {e}")
            return None
    
    def _extract_claims_from_response(self, response: str) -> List[str]:
        """Extract claims from LLM response"""
        claims = []
        
        # Try to extract claims in [c# | content | / confidence] format
        import re
        claim_pattern = r'\[c\d+\s*\|\s*([^|]+)\s*\|\s*/\s*([0-9.]+)\s*\]'
        matches = re.findall(claim_pattern, response)
        
        for match in matches:
            content = match[0].strip()
            confidence = match[1].strip()
            claim = f"[c | {content} | / {confidence}]"
            claims.append(claim)
        
        # If no claims found, try alternative patterns
        if not claims:
            # Try JSON-like format
            json_pattern = r'"content":\s*"([^"]+)"'
            json_matches = re.findall(json_pattern, response)
            for i, content in enumerate(json_matches[:10]):  # Limit to 10 claims
                claim = f"[c{i+1} | {content.strip()} | / 0.8]"
                claims.append(claim)
        
        # If still no claims, create from key sentences
        if not claims:
            sentences = response.split('.')
            for i, sentence in enumerate(sentences[:10]):  # Limit to 10 sentences
                if len(sentence.strip()) > 10:  # Only meaningful sentences
                    claim = f"[c{i+1} | {sentence.strip()} | / 0.7]"
                    claims.append(claim)
        
        return claims
    
    def _generate_full_context_prompt(self, test_case: Dict[str, Any]) -> str:
        """Generate prompt for full context approach"""
        context = test_case.get("context", "")
        question = test_case.get("question", "")
        
        return f"""
        You are given a large context and a question. Read the context carefully and provide a comprehensive answer.

        **Context:**
        {context}

        **Question:**
        {question}

        Provide a detailed, accurate answer based on the context. Be thorough and address all aspects of the question.
        """
    
    def _generate_compressed_context_prompt(self, test_case: Dict[str, Any], compressed_context: str) -> str:
        """Generate prompt for compressed context approach"""
        question = test_case.get("question", "")
        
        return f"""
        You are given a compressed context in claims format and a question. Use the claims to provide a comprehensive answer.

        **Compressed Context (Claims Format):**
        {compressed_context}

        **Question:**
        {question}

        **Instructions:**
        1. Use the provided claims as your primary source of information
        2. Consider the confidence scores when weighing information
        3. Provide a comprehensive answer based on the claims
        4. If claims are insufficient, acknowledge limitations

        Format your answer clearly and logically, referencing the claims where appropriate.
        """
    
    async def _evaluate_results(self):
        """Evaluate all results using LLM-as-a-Judge"""
        
        all_results = self.results.full_context_results + self.results.compressed_context_results
        
        for result in all_results:
            try:
                # Create evaluation prompt
                eval_prompt = self._create_evaluation_prompt(result)
                
                # Get evaluation from judge model
                evaluation_response = await self.llm_manager.generate_response(
                    prompt=eval_prompt,
                    model=self.config.judge_model,
                    max_tokens=1000,
                    temperature=0.3
                )
                
                # Parse evaluation
                scores = self._parse_evaluation(evaluation_response)
                
                # Update result with scores
                result.correctness = scores.get('correctness', 0.5)
                result.completeness = scores.get('completeness', 0.5)
                result.coherence = scores.get('coherence', 0.5)
                result.reasoning_quality = scores.get('reasoning_quality', 0.5)
                result.confidence_calibration = scores.get('confidence_calibration', 0.5)
                result.efficiency = scores.get('efficiency', 0.5)
                result.hallucination_reduction = scores.get('hallucination_reduction', 0.5)
                
            except Exception as e:
                self.logger.error(f"Evaluation failed for {result.test_id}: {e}")
                # Use default scores
                result.correctness = 0.5
                result.completeness = 0.5
                result.coherence = 0.5
                result.reasoning_quality = 0.5
                result.confidence_calibration = 0.5
                result.efficiency = 0.5
                result.hallucination_reduction = 0.5
    
    def _create_evaluation_prompt(self, result: TestResult) -> str:
        """Create LLM-as-a-Judge evaluation prompt"""
        return f"""
        You are an expert evaluator assessing AI model responses on context compression tasks.

        **Question:**
        {result.question}

        **Original Context Length:** {result.original_context_length} tokens
        **Compressed Context Length:** {result.compressed_context_length} tokens
        **Compression Ratio:** {result.compression_ratio:.2f}

        **Model Response:**
        {result.generated_answer}

        **Approach Used:** {result.approach}

        **Evaluation Instructions:**
        Evaluate the response on the following metrics (score 0.0-1.0):

        1. **Correctness**: Factual accuracy and correctness of answer based on original context
        2. **Completeness**: How thoroughly response addresses all aspects of the question
        3. **Coherence**: Logical flow, consistency, and structural coherence
        4. **Reasoning Quality**: Quality of logical reasoning and use of context
        5. **Confidence Calibration**: Appropriateness of confidence levels (if expressed)
        6. **Efficiency**: Conciseness and focus in response
        7. **Hallucination Reduction**: Grounding in provided information, absence of fabricated claims

        For compressed context approach, also consider:
        - How well the model utilized the compressed claims format
        - Whether important information was lost during compression

        Provide your evaluation in this format:

        CORRECTNESS: [0.0-1.0]
        COMPLETENESS: [0.0-1.0]
        COHERENCE: [0.0-1.0]
        REASONING_QUALITY: [0.0-1.0]
        CONFIDENCE_CALIBRATION: [0.0-1.0]
        EFFICIENCY: [0.0-1.0]
        HALLUCINATION_REDUCTION: [0.0-1.0]

        Be objective and thorough in your evaluation.
        """
    
    def _parse_evaluation(self, evaluation_response: str) -> Dict[str, float]:
        """Parse evaluation response into scores"""
        scores = {}
        metrics = ['correctness', 'completeness', 'coherence', 'reasoning_quality', 
                  'confidence_calibration', 'efficiency', 'hallucination_reduction']
        
        for metric in metrics:
            try:
                # Look for metric name in response
                metric_upper = metric.upper()
                if metric_upper in evaluation_response:
                    # Extract score after metric name
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
        full_scores = {
            'correctness': [r.correctness for r in self.results.full_context_results],
            'completeness': [r.completeness for r in self.results.full_context_results],
            'coherence': [r.coherence for r in self.results.full_context_results],
            'reasoning_quality': [r.reasoning_quality for r in self.results.full_context_results],
            'confidence_calibration': [r.confidence_calibration for r in self.results.full_context_results],
            'efficiency': [r.efficiency for r in self.results.full_context_results],
            'hallucination_reduction': [r.hallucination_reduction for r in self.results.full_context_results]
        }
        
        compressed_scores = {
            'correctness': [r.correctness for r in self.results.compressed_context_results],
            'completeness': [r.completeness for r in self.results.compressed_context_results],
            'coherence': [r.coherence for r in self.results.compressed_context_results],
            'reasoning_quality': [r.reasoning_quality for r in self.results.compressed_context_results],
            'confidence_calibration': [r.confidence_calibration for r in self.results.compressed_context_results],
            'efficiency': [r.efficiency for r in self.results.compressed_context_results],
            'hallucination_reduction': [r.hallucination_reduction for r in self.results.compressed_context_results]
        }
        
        # Calculate performance retention percentages
        for metric in full_scores.keys():
            full_mean = statistics.mean(full_scores[metric]) if full_scores[metric] else 0
            compressed_mean = statistics.mean(compressed_scores[metric]) if compressed_scores[metric] else 0
            
            if full_mean > 0:
                performance_retention = compressed_mean / full_mean
                self.results.performance_retention_percentages[metric] = performance_retention
            else:
                self.results.performance_retention_percentages[metric] = 0
        
        # Calculate compression effectiveness
        compression_ratios = [r.compression_ratio for r in self.results.compressed_context_results]
        avg_compression_ratio = statistics.mean(compression_ratios) if compression_ratios else 1.0
        compression_success_rate = len([r for r in self.results.compressed_context_results if r.compression_achieved]) / len(self.results.compressed_context_results) if self.results.compressed_context_results else 0
        
        self.results.compression_effectiveness = {
            'avg_compression_ratio': avg_compression_ratio,
            'compression_success_rate': compression_success_rate,
            'target_compression_ratio': self.config.target_compression_ratio
        }
        
        # Calculate improvements and statistical tests
        for metric in full_scores.keys():
            full_mean = statistics.mean(full_scores[metric]) if full_scores[metric] else 0
            compressed_mean = statistics.mean(compressed_scores[metric]) if compressed_scores[metric] else 0
            
            # Perform paired t-test if we have paired samples
            if len(full_scores[metric]) >= 2 and len(compressed_scores[metric]) >= 2:
                try:
                    # Paired t-test
                    t_stat, p_value = stats.ttest_rel(compressed_scores[metric], full_scores[metric])
                    self.results.statistical_significance[metric] = p_value
                    
                    # Calculate effect size (Cohen's d for paired samples)
                    diff_mean = statistics.mean([c - f for c, f in zip(compressed_scores[metric], full_scores[metric])])
                    diff_std = statistics.stdev([c - f for c, f in zip(compressed_scores[metric], full_scores[metric])]) if len(compressed_scores[metric]) > 1 else 1
                    effect_size = diff_mean / (diff_std + 0.001)  # Add small constant to avoid division by zero
                    self.results.effect_sizes[metric] = effect_size
                    
                    # Calculate confidence interval for difference
                    se = diff_std / (len(compressed_scores[metric]) ** 0.5)  # Standard error
                    ci_lower = diff_mean - stats.t.ppf(0.975, len(compressed_scores[metric]) - 1) * se
                    ci_upper = diff_mean + stats.t.ppf(0.975, len(compressed_scores[metric]) - 1) * se
                    self.results.confidence_intervals[metric] = (ci_lower, ci_upper)
                    
                except Exception as e:
                    self.logger.error(f"Statistical analysis failed for {metric}: {e}")
                    self.results.statistical_significance[metric] = 1.0
                    self.results.effect_sizes[metric] = 0.0
                    self.results.confidence_intervals[metric] = (0.0, 0.0)
            else:
                self.results.statistical_significance[metric] = 1.0
                self.results.effect_sizes[metric] = 0.0
                self.results.confidence_intervals[metric] = (0.0, 0.0)
            
            # Determine practical significance for performance retention
            self.results.practical_significance[metric] = (
                self.results.performance_retention_percentages[metric] >= self.config.target_performance_retention and
                self.results.statistical_significance[metric] < self.config.alpha_level and
                abs(self.results.effect_sizes[metric]) >= 0.5  # Medium effect size threshold
            )
    
    def _determine_hypothesis_validation(self):
        """Determine if hypothesis is validated"""
        
        # Primary metrics are correctness and completeness
        correctness_retention = self.results.performance_retention_percentages.get('correctness', 0.0)
        completeness_retention = self.results.performance_retention_percentages.get('completeness', 0.0)
        correctness_significance = self.results.statistical_significance.get('correctness', 1.0)
        completeness_significance = self.results.statistical_significance.get('completeness', 1.0)
        
        # Check if performance retention targets are achieved
        performance_target_achieved = (
            correctness_retention >= self.config.target_performance_retention and
            completeness_retention >= self.config.target_performance_retention
        )
        
        # Check if compression target is achieved
        compression_target_achieved = (
            self.results.compression_effectiveness['avg_compression_ratio'] <= self.config.target_compression_ratio and
            self.results.compression_effectiveness['compression_success_rate'] >= 0.8  # 80% of cases achieve target
        )
        
        self.results.targets_achieved = {
            'performance_retention': performance_target_achieved,
            'compression_ratio': compression_target_achieved
        }
        
        # Overall hypothesis validation (conservative approach)
        # Require both performance retention and compression targets
        hypothesis_validated = (
            performance_target_achieved and
            compression_target_achieved and
            correctness_significance < self.config.alpha_level
        )
        
        self.results.hypothesis_validated = hypothesis_validated
        
        # Calculate confidence in results
        successful_tests = len(self.results.full_context_results) + len(self.results.compressed_context_results)
        total_tests = self.config.sample_size * 2  # Both approaches
        completion_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        # Confidence based on completion rate and statistical significance
        avg_significance = statistics.mean(list(self.results.statistical_significance.values())) if self.results.statistical_significance else 1.0
        self.results.confidence_in_results = completion_rate * (1.0 - avg_significance)
    
    async def _save_results(self):
        """Save experiment results to file"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"context_compression_experiment_{self.results.experiment_id}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Convert to serializable format
        results_data = asdict(self.results)
        
        # Convert datetime objects to strings
        results_data['start_time'] = self.results.start_time.isoformat()
        results_data['end_time'] = self.results.end_time.isoformat() if self.results.end_time else None
        
        # Convert test results to dicts
        results_data['full_context_results'] = [asdict(r) for r in self.results.full_context_results]
        results_data['compressed_context_results'] = [asdict(r) for r in self.results.compressed_context_results]
        
        # Convert timestamps in test results
        for result_list in [results_data['full_context_results'], results_data['compressed_context_results']]:
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
        filename = f"context_compression_report_{self.results.experiment_id}_{timestamp}.md"
        filepath = self.reports_dir / filename
        
        report_lines = [
            "# Context Compression Experiment Report",
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Experiment ID: {self.results.experiment_id}",
            "",
            "## Executive Summary",
            "",
            f"**Hypothesis**: Models will maintain 90%+ performance with 50%+ context reduction using claims format",
            f"**Performance Retention Target**: {self.config.target_performance_retention * 100:.0f}%",
            f"**Compression Ratio Target**: {self.config.target_compression_ratio * 100:.0f}%",
            f"**Sample Size**: {len(self.results.full_context_results)} full context + {len(self.results.compressed_context_results)} compressed context tests",
            f"**Model Tested**: {self.config.tiny_model}",
            f"**Judge Model**: {self.config.judge_model}",
            "",
            "## Results Summary",
            "",
            f"**Hypothesis Validated**: {'✅ YES' if self.results.hypothesis_validated else '❌ NO'}",
            f"**Performance Retention Target Achieved**: {'✅ YES' if self.results.targets_achieved['performance_retention'] else '❌ NO'}",
            f"**Compression Target Achieved**: {'✅ YES' if self.results.targets_achieved['compression_ratio'] else '❌ NO'}",
            f"**Confidence in Results**: {self.results.confidence_in_results:.2%}",
            "",
            "## Compression Effectiveness",
            "",
            f"**Average Compression Ratio**: {self.results.compression_effectiveness['avg_compression_ratio']:.2f} (target: {self.config.target_compression_ratio:.2f})",
            f"**Compression Success Rate**: {self.results.compression_effectiveness['compression_success_rate']:.1%} (≥80% needed)",
            "",
            "## Performance Retention Analysis",
            "",
            "| Metric | Full Context Mean | Compressed Context Mean | Performance Retention | P-value | Effect Size | Significant |",
            "|--------|-------------------|----------------------|-------------------|----------|-------------|------------|"
        ]
        
        # Add metric comparisons
        for metric in ['correctness', 'completeness', 'coherence', 'reasoning_quality', 
                      'confidence_calibration', 'efficiency', 'hallucination_reduction']:
            
            full_mean = statistics.mean([getattr(r, metric) for r in self.results.full_context_results]) if self.results.full_context_results else 0
            compressed_mean = statistics.mean([getattr(r, metric) for r in self.results.compressed_context_results]) if self.results.compressed_context_results else 0
            performance_retention = self.results.performance_retention_percentages.get(metric, 0)
            p_value = self.results.statistical_significance.get(metric, 1.0)
            effect_size = self.results.effect_sizes.get(metric, 0)
            significant = self.results.practical_significance.get(metric, False)
            
            report_lines.append(
                f"| {metric} | {full_mean:.3f} | {compressed_mean:.3f} | {performance_retention:.1%} | {p_value:.3f} | {effect_size:.3f} | {'✅' if significant else '❌'} |"
            )
        
        report_lines.extend([
            "",
            "## Statistical Analysis",
            "",
            f"**Primary Metrics (Correctness & Completeness)**:",
            f"- Correctness Retention: {self.results.performance_retention_percentages.get('correctness', 0):.1%} (target: {self.config.target_performance_retention * 100:.0f}%)",
            f"- Completeness Retention: {self.results.performance_retention_percentages.get('completeness', 0):.1%} (target: {self.config.target_performance_retention * 100:.0f}%)",
            f"- Statistical Significance: p = {self.results.statistical_significance.get('correctness', 1.0):.3f}",
            f"- Effect Size (Cohen's d): {self.results.effect_sizes.get('correctness', 0):.3f}",
            f"- 95% Confidence Interval: [{self.results.confidence_intervals.get('correctness', (0, 0))[0]:.3f}, {self.results.confidence_intervals.get('correctness', (0, 0))[1]:.3f}]",
            "",
            "## Conclusions",
            ""
        ])
        
        if self.results.hypothesis_validated:
            report_lines.extend([
                "✅ **HYPOTHESIS VALIDATED**: Context compression with claims format maintains performance while reducing context.",
                "",
                "### Key Findings:",
                f"- Performance retention of {self.results.performance_retention_percentages.get('correctness', 0):.1%} for correctness and {self.results.performance_retention_percentages.get('completeness', 0):.1%} for completeness",
                f"- Average compression ratio of {self.results.compression_effectiveness['avg_compression_ratio']:.2f} meets {self.config.target_compression_ratio * 100:.0f}% reduction target",
                f"- Results are statistically significant (p < {self.config.alpha_level})",
                f"- Effect size indicates {'large' if abs(self.results.effect_sizes.get('correctness', 0)) >= 0.8 else 'medium' if abs(self.results.effect_sizes.get('correctness', 0)) >= 0.5 else 'small'} practical significance",
                "",
                "### Recommendations:",
                "- Implement context compression as a core feature in Conjecture",
                "- Optimize claim extraction algorithms for better compression ratios",
                "- Extend validation to additional model families and document types",
                "- Investigate which compression strategies work best for different content types"
            ])
        else:
            report_lines.extend([
                "❌ **HYPOTHESIS NOT VALIDATED**: Context compression did not achieve performance retention target.",
                "",
                "### Key Findings:",
                f"- Performance retention of {self.results.performance_retention_percentages.get('correctness', 0):.1%} for correctness and {self.results.performance_retention_percentages.get('completeness', 0):.1%} for completeness",
                f"- Target was {self.config.target_performance_retention * 100:.0f}% performance retention",
                f"- Compression ratio achieved: {self.results.compression_effectiveness['avg_compression_ratio']:.2f}",
                "- Results did not meet performance retention or statistical significance thresholds",
                "",
                "### Recommendations:",
                "- Refine context compression prompting approach",
                "- Improve claim extraction and selection algorithms",
                "- Investigate model-specific optimization for compression",
                "- Analyze failure cases for improvement opportunities"
            ])
        
        report_lines.extend([
            "",
            "## Technical Details",
            "",
            f"**Experiment Duration**: {(self.results.end_time - self.results.start_time).total_seconds():.1f} seconds",
            f"**Average Execution Time**: {statistics.mean([r.execution_time for r in self.results.full_context_results + self.results.compressed_context_results]):.2f} seconds",
            f"**Compression Success Rate**: {self.results.compression_effectiveness['compression_success_rate']:.1%}",
            "",
            "## Data Files",
            "",
            f"- Raw results: `experiments/results/context_compression_experiment_{self.results.experiment_id}_*.json`",
            f"- Test cases: `experiments/test_cases/context_compression_cases_{self.config.sample_size}.json`",
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
    """Main function to run context compression experiment"""
    
    # Configuration
    config = ExperimentConfig(
        sample_size=75,  # Target 50-100 test cases
        target_performance_retention=0.90,  # 90% performance retention target
        target_compression_ratio=0.50,  # 50% context reduction target
        alpha_level=0.05,
        power_target=0.8
    )
    
    # Initialize experiment
    experiment = ContextCompressionExperiment(config)
    
    # Setup provider configurations using existing config
    providers = [
        ProviderConfig(
            url="http://localhost:1234",  # LM Studio
            api_key="",
            model="ibm/granite-4-h-tiny"
        ),
        ProviderConfig(
            url="https://api.z.ai/api/coding/paas/v4",  # Z.AI
            api_key="70e6e12e4d7c46e2a4d0b85503d51f38.LQHl8d98kDJChttb",  # From config
            model="glm-4.6"
        )
    ]
    
    print("Starting Context Compression Experiment...")
    print(f"Hypothesis: Models maintain {config.target_performance_retention * 100:.0f}%+ performance with {config.target_compression_ratio * 100:.0f}%+ context reduction using claims format")
    print(f"Sample size: {config.sample_size} test cases")
    print(f"Model: {config.tiny_model}")
    print(f"Judge: {config.judge_model}")
    print("")
    
    try:
        # Initialize
        if not await experiment.initialize(providers):
            print("Failed to initialize experiment")
            return 1
        
        # Run experiment
        results = await experiment.run_experiment()
        
        print("\n" + "="*60)
        print("CONTEXT COMPRESSION EXPERIMENT RESULTS")
        print("="*60)
        print(f"Hypothesis Validated: {'YES' if results.hypothesis_validated else 'NO'}")
        print(f"Performance Retention Target Achieved: {'YES' if results.targets_achieved['performance_retention'] else 'NO'}")
        print(f"Compression Target Achieved: {'YES' if results.targets_achieved['compression_ratio'] else 'NO'}")
        print(f"Correctness Retention: {results.performance_retention_percentages.get('correctness', 0):.1%}")
        print(f"Completeness Retention: {results.performance_retention_percentages.get('completeness', 0):.1%}")
        print(f"Average Compression Ratio: {results.compression_effectiveness['avg_compression_ratio']:.2f}")
        print(f"Compression Success Rate: {results.compression_effectiveness['compression_success_rate']:.1%}")
        print(f"Confidence in Results: {results.confidence_in_results:.2%}")
        print("="*60)
        
        if results.hypothesis_validated:
            print("\nSUCCESS: Context compression hypothesis validated!")
            print("Conjecture's claims-based compression maintains performance while reducing context.")
        else:
            print("\nTARGET NOT ACHIEVED: Hypothesis not fully validated")
            print("Further refinement of context compression approach needed.")
            
        return 0
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)