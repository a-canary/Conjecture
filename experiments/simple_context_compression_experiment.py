#!/usr/bin/env python3
"""
Simple Context Compression Experiment
Tests if models maintain 90%+ performance with 50%+ context reduction using claims format.

This version works directly with existing Conjecture codebase without complex dependencies.
"""

import asyncio
import json
import time
import uuid
import statistics
import re
import requests
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
import sys
import os
from scipy import stats

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@dataclass
class TestResult:
    """Result from a single test case execution"""
    
    test_id: str
    approach: str
    question: str
    context: str
    generated_answer: str
    execution_time: float
    
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
    
    # Metadata
    timestamp: datetime
    difficulty: str


@dataclass
class ExperimentResults:
    """Complete results from context compression experiment"""
    
    experiment_id: str
    start_time: datetime
    end_time: Optional[datetime]
    
    # Test results
    full_context_results: List[TestResult]
    compressed_context_results: List[TestResult]
    
    # Statistical analysis
    performance_retention_percentages: Dict[str, float]
    statistical_significance: Dict[str, float]
    effect_sizes: Dict[str, float]
    
    # Overall results
    hypothesis_validated: bool
    targets_achieved: Dict[str, bool]
    confidence_in_results: float


class SimpleContextCompressionExperiment:
    """Simplified experiment runner for context compression hypothesis validation"""
    
    def __init__(self, sample_size: int = 25):
        self.sample_size = sample_size
        self.target_performance_retention = 0.90  # 90% performance retention target
        self.target_compression_ratio = 0.50  # 50% context reduction target
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
        logger = logging.getLogger("simple_context_compression_experiment")
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.results_dir / "simple_context_compression_experiment.log")
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
        """Generate context compression test cases"""
        
        self.logger.info(f"Generating {self.sample_size} context compression test cases...")
        
        test_cases = []
        
        # Long document QA cases
        for i in range(min(self.sample_size, 15)):
            case = {
                "id": f"long_doc_qa_{i+1:03d}",
                "category": "context_compression",
                "difficulty": "hard",
                "context": """
                The Renaissance was a period of cultural, artistic, political and economic rebirth following the Middle Ages in Europe. Generally described as taking place from the 14th to the 17th century, the Renaissance promoted the rediscovery of classical philosophy, literature and art. Some of the greatest thinkers, authors, statesmen, scientists and artists in human history thrived during this era, while global exploration opened up new lands and cultures to European commerce.
                
                The Renaissance began in Florence, Italy, in the 14th century. Various theories have been proposed to account for its origins and characteristics, focusing on a variety of factors including the social and civic peculiarities of Florence at the time: its political structure, the patronage of its dominant family, the Medici, and the migration of Greek scholars and texts to Italy following the Fall of Constantinople to the Ottoman Turks.
                
                The Renaissance saw many changes in culture, science, and technology. The printing press was invented, allowing books to be mass-produced for the first time. This led to a dramatic increase in literacy and the spread of new ideas. Artists developed new techniques such as linear perspective and chiaroscuro, creating more realistic and emotionally powerful works. Scientists such as Copernicus and Galileo challenged traditional views of the universe, laying the groundwork for the Scientific Revolution.
                
                In art, the Renaissance is perhaps best known for its artistic developments and the contributions of Leonardo da Vinci and Michelangelo, who inspired the term Renaissance man. However, many other notable artists made significant contributions during this period, including Raphael, Donatello, Titian, and Dürer. Renaissance art is characterized by realism, expression of human emotion, and the use of classical themes and motifs.
                
                In science, the Renaissance challenged medieval views of the world. Nicolaus Copernicus formulated a heliocentric model of the universe that placed the Sun rather than Earth at its center. Galileo Galilei improved the telescope and used it to make observations that supported Copernicus's theory. Andreas Vesalius revolutionized the study of anatomy with his detailed drawings of the human body.
                """,
                "question": "Based on the text, what were the four main areas of achievement during the Renaissance, and how did the printing press specifically contribute to the spread of Renaissance ideas?",
                "context_length": 400,
                "reasoning_requirements": ["information_extraction", "comprehension", "relevance_filtering"],
                "expected_answer_type": "comprehensive_answer"
            }
            test_cases.append(case)
        
        # Multi-source synthesis cases
        for i in range(min(self.sample_size - 15, 10)):
            case = {
                "id": f"multi_source_{i+1:03d}",
                "category": "context_compression",
                "difficulty": "hard",
                "context": """
                Source A: International Energy Agency reports solar panel costs have fallen by 85% since 2010, making it competitive with fossil fuels in most regions. Solar installations increased by 22% globally in 2023.
                
                Source B: Stanford University study finds wind energy capacity could meet global demand 4x over, but requires grid infrastructure investments of $2.3 trillion by 2030.
                
                Source C: Bloomberg analysis indicates renewable energy jobs grew 70% faster than overall economy, with 10.3 million people employed globally in clean energy sectors.
                
                Source D: European Commission research shows energy storage costs decreased by 70% since 2015, enabling 24/7 renewable power availability.
                """,
                "question": "Synthesize the information about renewable energy transition. What are the key economic and technical factors driving adoption, and what challenges remain?",
                "context_length": 350,
                "reasoning_requirements": ["information_integration", "synthesis", "source_evaluation"],
                "expected_answer_type": "synthesized_analysis"
            }
            test_cases.append(case)
        
        # Save test cases
        test_cases_file = self.test_cases_dir / f"simple_context_compression_cases_{self.sample_size}.json"
        with open(test_cases_file, 'w', encoding='utf-8') as f:
            json.dump(test_cases, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Generated {len(test_cases)} context compression test cases")
        return test_cases
    
    async def run_experiment(self) -> ExperimentResults:
        """Run complete context compression experiment"""
        
        experiment_id = str(uuid.uuid4())[:8]
        start_time = datetime.utcnow()
        
        self.logger.info(f"Starting Simple Context Compression Experiment: {experiment_id}")
        
        # Initialize results
        self.results = ExperimentResults(
            experiment_id=experiment_id,
            start_time=start_time,
            end_time=None,
            full_context_results=[],
            compressed_context_results=[],
            performance_retention_percentages={},
            statistical_significance={},
            effect_sizes={},
            hypothesis_validated=False,
            targets_achieved={},
            confidence_in_results=0.0
        )
        
        try:
            # Generate test cases
            test_cases = self.generate_test_cases()
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
    
    async def _run_full_context_test(self, test_case: Dict[str, Any]) -> Optional[TestResult]:
        """Run full context approach test"""
        try:
            # Generate full context prompt
            prompt = f"""
            You are given a large context and a question. Read the context carefully and provide a comprehensive answer.

            **Context:**
            {test_case['context']}

            **Question:**
            {test_case['question']}

            Provide a detailed, accurate answer based on the context. Be thorough and address all aspects of the question.
            """
            
            # Execute with tiny model
            start_time = time.time()
            response = await self.call_api(prompt, "ibm/granite-4-h-tiny", use_tiny_model=True)
            execution_time = time.time() - start_time
            
            # Create result
            result = TestResult(
                test_id=test_case["id"],
                approach="full_context",
                question=test_case["question"],
                context=test_case["context"],
                generated_answer=response,
                execution_time=execution_time,
                original_context_length=test_case.get("context_length", 0),
                compressed_context_length=test_case.get("context_length", 0),  # Same for full context
                compression_ratio=1.0,  # No compression
                compression_achieved=False,
                correctness=0.0,  # Will be filled by evaluation
                completeness=0.0,
                coherence=0.0,
                reasoning_quality=0.0,
                timestamp=datetime.utcnow(),
                difficulty=test_case["difficulty"]
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
            prompt = f"""
            You are given a compressed context in claims format and a question. Use the claims to provide a comprehensive answer.

            **Compressed Context (Claims Format):**
            {compressed_context}

            **Question:**
            {test_case['question']}

            **Instructions:**
            1. Use the provided claims as your primary source of information
            2. Consider the confidence scores when weighing information
            3. Provide a comprehensive answer based on the claims
            4. If claims are insufficient, acknowledge limitations

            Format your answer clearly and logically, referencing the claims where appropriate.
            """
            
            # Execute with tiny model
            start_time = time.time()
            response = await self.call_api(prompt, "ibm/granite-4-h-tiny", use_tiny_model=True)
            execution_time = time.time() - start_time
            
            # Create result
            result = TestResult(
                test_id=test_case["id"],
                approach="compressed_context",
                question=test_case["question"],
                context=test_case["context"],
                generated_answer=response,
                execution_time=execution_time,
                original_context_length=test_case.get("context_length", 0),
                compressed_context_length=len(compressed_context.split()),
                compression_ratio=compression_ratio,
                compression_achieved=compression_ratio <= self.target_compression_ratio,
                correctness=0.0,  # Will be filled by evaluation
                completeness=0.0,
                coherence=0.0,
                reasoning_quality=0.0,
                timestamp=datetime.utcnow(),
                difficulty=test_case["difficulty"]
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
            compression_response = await self.call_api(compression_prompt, "ibm/granite-4-h-tiny", use_tiny_model=True)
            
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
    
    async def _evaluate_results(self):
        """Evaluate all results using GLM-4.6 as judge"""
        
        all_results = self.results.full_context_results + self.results.compressed_context_results
        
        for result in all_results:
            try:
                # Create evaluation prompt
                eval_prompt = f"""
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

                For compressed context approach, also consider:
                - How well the model utilized the compressed claims format
                - Whether important information was lost during compression

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
            'reasoning_quality': [r.reasoning_quality for r in self.results.full_context_results]
        }
        
        compressed_scores = {
            'correctness': [r.correctness for r in self.results.compressed_context_results],
            'completeness': [r.completeness for r in self.results.compressed_context_results],
            'coherence': [r.coherence for r in self.results.compressed_context_results],
            'reasoning_quality': [r.reasoning_quality for r in self.results.compressed_context_results]
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
                    
                except Exception as e:
                    self.logger.error(f"Statistical analysis failed for {metric}: {e}")
                    self.results.statistical_significance[metric] = 1.0
                    self.results.effect_sizes[metric] = 0.0
            else:
                self.results.statistical_significance[metric] = 1.0
                self.results.effect_sizes[metric] = 0.0
    
    def _determine_hypothesis_validation(self):
        """Determine if hypothesis is validated"""
        
        # Primary metrics are correctness and completeness
        correctness_retention = self.results.performance_retention_percentages.get('correctness', 0.0)
        completeness_retention = self.results.performance_retention_percentages.get('completeness', 0.0)
        correctness_significance = self.results.statistical_significance.get('correctness', 1.0)
        
        # Check if performance retention targets are achieved
        performance_target_achieved = (
            correctness_retention >= self.target_performance_retention and
            completeness_retention >= self.target_performance_retention
        )
        
        # Check if compression target is achieved
        compression_ratios = [r.compression_ratio for r in self.results.compressed_context_results]
        avg_compression_ratio = statistics.mean(compression_ratios) if compression_ratios else 1.0
        compression_target_achieved = avg_compression_ratio <= self.target_compression_ratio
        
        self.results.targets_achieved = {
            'performance_retention': performance_target_achieved,
            'compression_ratio': compression_target_achieved
        }
        
        # Overall hypothesis validation (conservative approach)
        # Require both performance retention and compression targets
        hypothesis_validated = (
            performance_target_achieved and
            compression_target_achieved and
            correctness_significance < self.alpha_level
        )
        
        self.results.hypothesis_validated = hypothesis_validated
        
        # Calculate confidence in results
        successful_tests = len(self.results.full_context_results) + len(self.results.compressed_context_results)
        total_tests = self.sample_size * 2  # Both approaches
        completion_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        # Confidence based on completion rate and statistical significance
        avg_significance = statistics.mean(list(self.results.statistical_significance.values())) if self.results.statistical_significance else 1.0
        self.results.confidence_in_results = completion_rate * (1.0 - avg_significance)
    
    async def _save_results(self):
        """Save experiment results to file"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"simple_context_compression_experiment_{self.results.experiment_id}_{timestamp}.json"
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
        filename = f"simple_context_compression_report_{self.results.experiment_id}_{timestamp}.md"
        filepath = self.reports_dir / filename
        
        report_lines = [
            "# Simple Context Compression Experiment Report",
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Experiment ID: {self.results.experiment_id}",
            "",
            "## Executive Summary",
            "",
            f"**Hypothesis**: Models will maintain 90%+ performance with 50%+ context reduction using claims format",
            f"**Performance Retention Target**: {self.target_performance_retention * 100:.0f}%",
            f"**Compression Ratio Target**: {self.target_compression_ratio * 100:.0f}%",
            f"**Sample Size**: {len(self.results.full_context_results)} full context + {len(self.results.compressed_context_results)} compressed context tests",
            f"**Model Tested**: ibm/granite-4-h-tiny (local)",
            f"**Judge Model**: glm-4.6 (Z.AI API)",
            "",
            "## Results Summary",
            "",
            f"**Hypothesis Validated**: {'✅ YES' if self.results.hypothesis_validated else '❌ NO'}",
            f"**Performance Retention Target Achieved**: {'✅ YES' if self.results.targets_achieved['performance_retention'] else '❌ NO'}",
            f"**Compression Target Achieved**: {'✅ YES' if self.results.targets_achieved['compression_ratio'] else '❌ NO'}",
            f"**Confidence in Results**: {self.results.confidence_in_results:.2%}",
            "",
            "## Performance Retention Analysis",
            "",
            "| Metric | Full Context Mean | Compressed Context Mean | Performance Retention | P-value | Effect Size | Significant |",
            "|--------|-------------------|----------------------|-------------------|----------|-------------|------------|"
        ]
        
        # Add metric comparisons
        for metric in ['correctness', 'completeness', 'coherence', 'reasoning_quality']:
            
            full_mean = statistics.mean([getattr(r, metric) for r in self.results.full_context_results]) if self.results.full_context_results else 0
            compressed_mean = statistics.mean([getattr(r, metric) for r in self.results.compressed_context_results]) if self.results.compressed_context_results else 0
            performance_retention = self.results.performance_retention_percentages.get(metric, 0)
            p_value = self.results.statistical_significance.get(metric, 1.0)
            effect_size = self.results.effect_sizes.get(metric, 0)
            significant = p_value < self.alpha_level and abs(effect_size) >= 0.5
            
            report_lines.append(
                f"| {metric} | {full_mean:.3f} | {compressed_mean:.3f} | {performance_retention:.1%} | {p_value:.3f} | {effect_size:.3f} | {'✅' if significant else '❌'} |"
            )
        
        # Calculate compression effectiveness
        compression_ratios = [r.compression_ratio for r in self.results.compressed_context_results]
        avg_compression_ratio = statistics.mean(compression_ratios) if compression_ratios else 1.0
        compression_success_rate = len([r for r in self.results.compressed_context_results if r.compression_achieved]) / len(self.results.compressed_context_results) if self.results.compressed_context_results else 0
        
        report_lines.extend([
            "",
            "## Compression Effectiveness",
            "",
            f"**Average Compression Ratio**: {avg_compression_ratio:.2f} (target: {self.target_compression_ratio:.2f})",
            f"**Compression Success Rate**: {compression_success_rate:.1%} (≥80% needed)",
            "",
            "## Statistical Analysis",
            "",
            f"**Primary Metrics (Correctness & Completeness)**:",
            f"- Correctness Retention: {self.results.performance_retention_percentages.get('correctness', 0):.1%} (target: {self.target_performance_retention * 100:.0f}%)",
            f"- Completeness Retention: {self.results.performance_retention_percentages.get('completeness', 0):.1%} (target: {self.target_performance_retention * 100:.0f}%)",
            f"- Statistical Significance: p = {self.results.statistical_significance.get('correctness', 1.0):.3f}",
            f"- Effect Size (Cohen's d): {self.results.effect_sizes.get('correctness', 0):.3f}",
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
                f"- Average compression ratio of {avg_compression_ratio:.2f} meets {self.target_compression_ratio * 100:.0f}% reduction target",
                f"- Results are statistically significant (p < {self.alpha_level})",
                "",
                "### Recommendations:",
                "- Implement context compression as a core feature in Conjecture",
                "- Optimize claim extraction algorithms for better compression ratios",
                "- Extend validation to additional model families and document types"
            ])
        else:
            report_lines.extend([
                "❌ **HYPOTHESIS NOT VALIDATED**: Context compression did not achieve performance retention target.",
                "",
                "### Key Findings:",
                f"- Performance retention of {self.results.performance_retention_percentages.get('correctness', 0):.1%} for correctness and {self.results.performance_retention_percentages.get('completeness', 0):.1%} for completeness",
                f"- Target was {self.target_performance_retention * 100:.0f}% performance retention",
                f"- Compression ratio achieved: {avg_compression_ratio:.2f}",
                "- Results did not meet performance retention or statistical significance thresholds",
                "",
                "### Recommendations:",
                "- Refine context compression prompting approach",
                "- Improve claim extraction and selection algorithms",
                "- Investigate model-specific optimization for compression"
            ])
        
        report_lines.extend([
            "",
            "## Technical Details",
            "",
            f"**Experiment Duration**: {(self.results.end_time - self.results.start_time).total_seconds():.1f} seconds",
            f"**Average Execution Time**: {statistics.mean([r.execution_time for r in self.results.full_context_results + self.results.compressed_context_results]):.2f} seconds",
            "",
            "## Data Files",
            "",
            f"- Raw results: `experiments/results/simple_context_compression_experiment_{self.results.experiment_id}_*.json`",
            f"- Test cases: `experiments/test_cases/simple_context_compression_cases_{self.sample_size}.json`",
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
    """Main function to run simple context compression experiment"""
    
    # Initialize experiment with smaller sample size for testing
    sample_size = 15  # Reduced for initial testing
    experiment = SimpleContextCompressionExperiment(sample_size)
    
    print("Starting Simple Context Compression Experiment...")
    print(f"Hypothesis: Models maintain {experiment.target_performance_retention * 100:.0f}%+ performance with {experiment.target_compression_ratio * 100:.0f}%+ context reduction using claims format")
    print(f"Sample size: {sample_size} test cases")
    print(f"Model: ibm/granite-4-h-tiny (local)")
    print(f"Judge: glm-4.6 (Z.AI API)")
    print("")
    
    try:
        results = await experiment.run_experiment()
        
        print("\n" + "="*60)
        print("CONTEXT COMPRESSION EXPERIMENT RESULTS")
        print("="*60)
        print(f"Hypothesis Validated: {'YES' if results.hypothesis_validated else 'NO'}")
        print(f"Performance Retention Target Achieved: {'YES' if results.targets_achieved['performance_retention'] else 'NO'}")
        print(f"Compression Target Achieved: {'YES' if results.targets_achieved['compression_ratio'] else 'NO'}")
        print(f"Correctness Retention: {results.performance_retention_percentages.get('correctness', 0):.1%}")
        print(f"Completeness Retention: {results.performance_retention_percentages.get('completeness', 0):.1%}")
        
        # Calculate compression effectiveness
        compression_ratios = [r.compression_ratio for r in results.compressed_context_results]
        avg_compression_ratio = statistics.mean(compression_ratios) if compression_ratios else 1.0
        compression_success_rate = len([r for r in results.compressed_context_results if r.compression_achieved]) / len(results.compressed_context_results) if results.compressed_context_results else 0
        
        print(f"Average Compression Ratio: {avg_compression_ratio:.2f}")
        print(f"Compression Success Rate: {compression_success_rate:.1%}")
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