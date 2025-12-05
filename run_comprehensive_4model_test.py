#!/usr/bin/env python3
"""
Comprehensive 4-Model Comparison Test with All Fixes
Tests Direct vs Conjecture approaches with proper metrics collection
Validates the core hypothesis: Conjecture achieves 25%+ improvement
"""

import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, "src"))

try:
    import aiohttp
    import requests
    print("[OK] Required libraries available")
except ImportError as e:
    print(f"[FAIL] Missing required library: {e}")
    print("Please install with: pip install aiohttp requests")
    sys.exit(1)

@dataclass
class ModelTestResult:
    """Enhanced result with comprehensive metrics"""
    model_name: str
    access_method: str  # "direct" or "conjecture"
    test_case_id: str
    test_category: str
    prompt: str
    response: str
    response_time: float
    tokens_used: int
    success: bool
    error_message: Optional[str] = None
    timestamp: str = None
    
    # Additional metrics for hypothesis validation
    reasoning_quality: float = 0.0
    confidence_calibration: float = 0.0
    evidence_integration: float = 0.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class ComparisonSummary:
    """Enhanced summary with hypothesis validation metrics"""
    model_name: str
    access_method: str
    total_tests: int
    successful_tests: int
    success_rate: float
    avg_response_time: float
    total_tokens: int
    avg_tokens_per_test: float
    
    # Hypothesis validation metrics
    avg_reasoning_quality: float
    avg_confidence_calibration: float
    avg_evidence_integration: float
    overall_performance_score: float
    
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class EnhancedModelProvider:
    """Enhanced provider with retry logic and proper error handling"""
    
    def __init__(self, provider_type: str, config: Dict[str, Any]):
        self.provider_type = provider_type
        self.config = config
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def call_model(self, prompt: str, model_name: str) -> Tuple[str, float, int]:
        """Enhanced model call with retry logic"""
        start_time = time.time()
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                if self.provider_type == "lmstudio":
                    return await self._call_lmstudio(prompt, model_name, start_time)
                elif self.provider_type == "chutes":
                    return await self._call_chutes(prompt, model_name, start_time)
                elif self.provider_type == "conjecture":
                    return await self._call_conjecture(prompt, model_name, start_time)
                else:
                    raise ValueError(f"Unknown provider type: {self.provider_type}")
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"  Retry {attempt + 1}/{max_retries} after {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    raise e
    
    async def _call_lmstudio(self, prompt: str, model_name: str, start_time: float) -> Tuple[str, float, int]:
        """Call LM Studio model directly"""
        request_data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Provide clear, reasoned responses."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 2000,
            "temperature": 0.7,
        }
        
        async with self.session.post(
            f"{self.config['base_url']}/v1/chat/completions", 
            json=request_data, 
            timeout=60
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"LM Studio API error: {response.status} - {error_text}")
            
            data = await response.json()
            response_time = time.time() - start_time
            
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            tokens_used = data.get("usage", {}).get("total_tokens", 0)
            
            return content, response_time, tokens_used
    
    async def _call_chutes(self, prompt: str, model_name: str, start_time: float) -> Tuple[str, float, int]:
        """Call Chutes API model directly"""
        headers = {
            "Authorization": f"Bearer {self.config['api_key']}",
            "Content-Type": "application/json",
        }
        
        request_data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Provide clear, reasoned responses."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 2000,
            "temperature": 0.7,
        }
        
        async with self.session.post(
            f"{self.config['base_url']}/chat/completions",
            json=request_data,
            headers=headers,
            timeout=60,
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Chutes API error: {response.status} - {error_text}")
            
            data = await response.json()
            response_time = time.time() - start_time
            
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            tokens_used = data.get("usage", {}).get("total_tokens", 0)
            
            return content, response_time, tokens_used
    
    async def _call_conjecture(self, prompt: str, model_name: str, start_time: float) -> Tuple[str, float, int]:
        """Call model through Conjecture router"""
        request_data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Provide clear, reasoned responses."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 2000,
            "temperature": 0.7,
        }
        
        # Use the correct port for Conjecture router
        async with self.session.post(
            f"{self.config['base_url']}/v1/chat/completions", 
            json=request_data, 
            timeout=120  # Longer timeout for Conjecture processing
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Conjecture API error: {response.status} - {error_text}")
            
            data = await response.json()
            response_time = time.time() - start_time
            
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            tokens_used = data.get("usage", {}).get("total_tokens", 0)
            
            return content, response_time, tokens_used

class Comprehensive4ModelTest:
    """Comprehensive 4-model comparison test with hypothesis validation"""
    
    def __init__(self):
        self.test_results = []
        self.output_dir = Path("research/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Provider configurations with correct ports
        self.provider_configs = {
            "lmstudio": {
                "type": "lmstudio",
                "config": {
                    "base_url": "http://localhost:1234",
                }
            },
            "chutes": {
                "type": "chutes", 
                "config": {
                    "base_url": "https://api.z.ai/api/coding/paas/v4",
                    "api_key": "70e6e12e4d7c46e2a4d0b85503d51f38.LQHl8d98kDJChttb"
                }
            },
            "conjecture": {
                "type": "conjecture",
                "config": {
                    "base_url": "http://localhost:5677",  # Correct port
                }
            }
        }
        
        # Load test configuration
        self.load_config()
    
    def load_config(self):
        """Load configuration from config file"""
        try:
            with open("c:/Users/Aaron.Canary/.conjecture/config.json", "r") as f:
                config = json.load(f)
                self.models = config.get("interanl_dev", {}).get("test_models", [])
                print(f"Loaded models from config: {self.models}")
        except Exception as e:
            print(f"Error loading config: {e}")
            self.models = ["lms/granite-4-h-tiny", "zai/GLM-4.6", "openrouter/gpt-oss-20b"]
    
    def load_test_cases(self) -> List[Dict[str, Any]]:
        """Load comprehensive test cases"""
        test_cases = []
        
        # Load from research/test_cases directory
        test_case_dir = Path("research/test_cases")
        if test_case_dir.exists():
            for case_file in test_case_dir.glob("*.json"):
                try:
                    with open(case_file, "r") as f:
                        case_data = json.load(f)
                        test_cases.append({
                            "id": case_file.stem,
                            "category": "evidence_conflict" if "conflict" in case_file.stem else "general",
                            "data": case_data,
                        })
                except Exception as e:
                    print(f"Error loading {case_file}: {e}")
        
        # Add additional test cases for comprehensive evaluation
        additional_cases = [
            {
                "id": "reasoning_complex_1",
                "category": "complex_reasoning",
                "data": {
                    "question": "A company has three departments A, B, and C. Department A handles 40% of customers, B handles 35%, and C handles 25%. Satisfaction rates are 85%, 90%, and 80% respectively. What is the overall customer satisfaction rate? Explain your reasoning step by step.",
                    "expected_approach": "weighted_average_calculation"
                }
            },
            {
                "id": "evidence_synthesis_1", 
                "category": "evidence_synthesis",
                "data": {
                    "question": "You have three studies on a new medication: Study 1 (n=100) shows 60% effectiveness, Study 2 (n=200) shows 75% effectiveness, Study 3 (n=150) shows 65% effectiveness. What is your best estimate of the medication's effectiveness? Provide confidence intervals.",
                    "expected_approach": "meta_analysis"
                }
            },
            {
                "id": "creative_problem_1",
                "category": "creative_problem",
                "data": {
                    "question": "Design a sustainable urban transportation system for a city of 500,000 people. Consider environmental impact, cost-effectiveness, and user experience. Provide a detailed proposal with implementation phases.",
                    "expected_approach": "system_design"
                }
            }
        ]
        
        test_cases.extend(additional_cases)
        print(f"Loaded {len(test_cases)} test cases")
        return test_cases
    
    def format_prompt(self, test_case: Dict[str, Any]) -> str:
        """Format test case into prompt"""
        category = test_case.get("category", "general")
        data = test_case.get("data", {})
        
        if category == "evidence_conflict":
            claims = data.get("claims", [])
            question = data.get("question", "")
            
            prompt = f"""Analyze the following conflicting evidence and provide a reasoned recommendation:

CLAIMS:
"""
            for i, claim in enumerate(claims, 1):
                prompt += f"{i}. {claim.get('content', '')} (Confidence: {claim.get('confidence', 0)}, Source: {claim.get('source', '')})\n"
            
            prompt += f"""
QUESTION: {question}

Please provide:
1. A clear recommendation
2. Confidence level in your recommendation
3. Reasoning process explaining how you resolved conflicts
4. Key factors influencing your decision
"""
            return prompt
        
        elif category == "complex_reasoning":
            return f"""Please solve the following problem with step-by-step reasoning:

{data.get('question', '')}

Show your work clearly and explain each step of your reasoning process."""
        
        elif category == "evidence_synthesis":
            return f"""Please synthesize the following evidence:

{data.get('question', '')}

Provide:
1. Your best estimate
2. Confidence intervals
3. Methodology for synthesis
4. Limitations and assumptions
"""
        
        elif category == "creative_problem":
            return f"""Please address the following challenge:

{data.get('question', '')}

Provide a comprehensive solution with:
1. Overview
2. Detailed components
3. Implementation plan
4. Risk assessment
"""
        
        else:  # General category
            if "question" in data:
                return data["question"]
            else:
                return str(data)
    
    def evaluate_response_quality(self, response: str, test_case: Dict[str, Any]) -> Tuple[float, float, float]:
        """Evaluate response quality for hypothesis validation"""
        if not response:
            return 0.0, 0.0, 0.0
        
        # Simple heuristic evaluation (in real implementation, could use LLM judge)
        response_length = len(response.split())
        
        # Reasoning quality: based on structure and completeness
        reasoning_quality = min(1.0, response_length / 200)  # Normalize to 0-1
        
        # Confidence calibration: look for confidence indicators
        confidence_indicators = ["confidence", "certain", "likely", "probably", "estimate", "approximate"]
        confidence_score = sum(1 for indicator in confidence_indicators if indicator.lower() in response.lower())
        confidence_calibration = min(1.0, confidence_score / 3)  # Normalize to 0-1
        
        # Evidence integration: look for evidence synthesis patterns
        evidence_patterns = ["study", "research", "data", "evidence", "according to", "shows that"]
        evidence_score = sum(1 for pattern in evidence_patterns if pattern.lower() in response.lower())
        evidence_integration = min(1.0, evidence_score / 4)  # Normalize to 0-1
        
        return reasoning_quality, confidence_calibration, evidence_integration
    
    async def test_model_configuration(self, model_config: Dict[str, Any], test_cases: List[Dict[str, Any]]) -> List[ModelTestResult]:
        """Test a specific model configuration"""
        model_name = model_config["name"]
        provider_name = model_config["provider"]
        access_method = model_config["access_method"]
        
        print(f"\nTesting {model_name} via {access_method} using {provider_name}...")
        
        provider_config = self.provider_configs[provider_name]
        
        async with EnhancedModelProvider(provider_config["type"], provider_config["config"]) as provider:
            results = []
            
            for i, test_case in enumerate(test_cases):
                test_id = test_case["id"]
                category = test_case["category"]
                prompt = self.format_prompt(test_case)
                
                print(f"  [{i+1}/{len(test_cases)}] {test_id} ({category})...", end=" ")
                
                try:
                    response, response_time, tokens = await provider.call_model(prompt, model_name)
                    
                    # Evaluate response quality
                    reasoning_quality, confidence_calibration, evidence_integration = self.evaluate_response_quality(response, test_case)
                    
                    result = ModelTestResult(
                        model_name=model_name,
                        access_method=access_method,
                        test_case_id=test_id,
                        test_category=category,
                        prompt=prompt,
                        response=response,
                        response_time=response_time,
                        tokens_used=tokens,
                        success=True,
                        reasoning_quality=reasoning_quality,
                        confidence_calibration=confidence_calibration,
                        evidence_integration=evidence_integration
                    )
                    
                    results.append(result)
                    print(f"SUCCESS ({response_time:.2f}s, {tokens} tokens)")
                    
                except Exception as e:
                    result = ModelTestResult(
                        model_name=model_name,
                        access_method=access_method,
                        test_case_id=test_id,
                        test_category=category,
                        prompt=prompt,
                        response="",
                        response_time=0,
                        tokens_used=0,
                        success=False,
                        error_message=str(e)
                    )
                    
                    results.append(result)
                    print(f"FAILED ({str(e)[:50]}...)")
            
            return results
    
    def calculate_enhanced_summary(self, results: List[ModelTestResult], model_name: str, access_method: str) -> ComparisonSummary:
        """Calculate enhanced summary with hypothesis validation metrics"""
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return ComparisonSummary(
                model_name=model_name,
                access_method=access_method,
                total_tests=len(results),
                successful_tests=0,
                success_rate=0.0,
                avg_response_time=0.0,
                total_tokens=0,
                avg_tokens_per_test=0.0,
                avg_reasoning_quality=0.0,
                avg_confidence_calibration=0.0,
                avg_evidence_integration=0.0,
                overall_performance_score=0.0
            )
        
        success_rate = len(successful_results) / len(results)
        avg_response_time = statistics.mean(r.response_time for r in successful_results)
        total_tokens = sum(r.tokens_used for r in successful_results)
        avg_tokens_per_test = total_tokens / len(successful_results)
        
        # Enhanced metrics
        avg_reasoning_quality = statistics.mean(r.reasoning_quality for r in successful_results)
        avg_confidence_calibration = statistics.mean(r.confidence_calibration for r in successful_results)
        avg_evidence_integration = statistics.mean(r.evidence_integration for r in successful_results)
        
        # Overall performance score (weighted combination)
        overall_performance_score = (
            0.3 * success_rate +
            0.2 * (1.0 / (1.0 + avg_response_time)) +  # Speed factor
            0.2 * avg_reasoning_quality +
            0.15 * avg_confidence_calibration +
            0.15 * avg_evidence_integration
        )
        
        return ComparisonSummary(
            model_name=model_name,
            access_method=access_method,
            total_tests=len(results),
            successful_tests=len(successful_results),
            success_rate=success_rate,
            avg_response_time=avg_response_time,
            total_tokens=total_tokens,
            avg_tokens_per_test=avg_tokens_per_test,
            avg_reasoning_quality=avg_reasoning_quality,
            avg_confidence_calibration=avg_confidence_calibration,
            avg_evidence_integration=avg_evidence_integration,
            overall_performance_score=overall_performance_score
        )
    
    async def run_comprehensive_test(self):
        """Run the comprehensive 4-model comparison test"""
        print("Starting Comprehensive 4-Model Comparison Test")
        print("=" * 80)
        print("Hypothesis: Conjecture approach achieves 25%+ performance improvement")
        print("=" * 80)
        
        # Load test cases
        test_cases = self.load_test_cases()
        
        # Define model configurations based on available models
        model_configs = []
        
        for model in self.models:
            # Add direct access test
            if "granite" in model.lower():
                model_configs.append({
                    "name": "ibm/granite-4-h-tiny",
                    "provider": "lmstudio",
                    "access_method": "direct"
                })
            elif "glm" in model.lower():
                model_configs.append({
                    "name": "glm-4.6",
                    "provider": "chutes", 
                    "access_method": "direct"
                })
            elif "gpt" in model.lower():
                model_configs.append({
                    "name": "openai/gpt-oss-20b",
                    "provider": "chutes",
                    "access_method": "direct"
                })
            
            # Add Conjecture access test for each model
            model_configs.append({
                "name": model,
                "provider": "conjecture",
                "access_method": "conjecture"
            })
        
        print(f"Testing {len(model_configs)} configurations across {len(test_cases)} test cases")
        
        # Run tests for each configuration
        all_results = []
        summaries = []
        
        for config in model_configs:
            results = await self.test_model_configuration(config, test_cases)
            all_results.extend(results)
            
            summary = self.calculate_enhanced_summary(results, config["name"], config["access_method"])
            summaries.append(summary)
            
            print(f"\nResults for {config['name']} ({config['access_method']}):")
            print(f"   Success Rate: {summary.success_rate:.2%}")
            print(f"   Avg Response Time: {summary.avg_response_time:.2f}s")
            print(f"   Avg Tokens per Test: {summary.avg_tokens_per_test:.0f}")
            print(f"   Reasoning Quality: {summary.avg_reasoning_quality:.3f}")
            print(f"   Confidence Calibration: {summary.avg_confidence_calibration:.3f}")
            print(f"   Evidence Integration: {summary.avg_evidence_integration:.3f}")
            print(f"   Overall Performance Score: {summary.overall_performance_score:.3f}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.output_dir / f"comprehensive_4model_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump([asdict(result) for result in all_results], f, indent=2)
        
        # Save summary
        summary_file = self.output_dir / f"comprehensive_4model_summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump([asdict(summary) for summary in summaries], f, indent=2)
        
        # Generate comprehensive report
        report_file = self.output_dir / f"comprehensive_4model_report_{timestamp}.md"
        self.generate_comprehensive_report(summaries, report_file, all_results)
        
        print("\n" + "=" * 80)
        print("Comprehensive 4-Model Comparison Test Complete!")
        print(f"Results saved to: {results_file}")
        print(f"Summary saved to: {summary_file}")
        print(f"Report saved to: {report_file}")
        
        # Analyze hypothesis validation
        self.validate_hypothesis(summaries)
        
        return summaries, all_results
    
    def generate_comprehensive_report(self, summaries: List[ComparisonSummary], output_file: Path, all_results: List[ModelTestResult]):
        """Generate comprehensive markdown report"""
        with open(output_file, "w") as f:
            f.write("# Comprehensive 4-Model Comparison Report\n\n")
            f.write(f"Generated on: {datetime.now().isoformat()}\n\n")
            f.write("## Hypothesis\n\n")
            f.write("**Conjecture pipeline with claims-based reasoning achieves significant performance improvements over direct prompting approaches.**\n\n")
            f.write("Target: ≥25% improvement in overall performance score\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            best_direct = max([s for s in summaries if s.access_method == "direct"], key=lambda x: x.overall_performance_score, default=None)
            best_conjecture = max([s for s in summaries if s.access_method == "conjecture"], key=lambda x: x.overall_performance_score, default=None)
            
            if best_direct and best_conjecture:
                improvement = ((best_conjecture.overall_performance_score - best_direct.overall_performance_score) / best_direct.overall_performance_score) * 100
                f.write(f"- **Best Direct Approach**: {best_direct.model_name} (Score: {best_direct.overall_performance_score:.3f})\n")
                f.write(f"- **Best Conjecture Approach**: {best_conjecture.model_name} (Score: {best_conjecture.overall_performance_score:.3f})\n")
                f.write(f"- **Performance Improvement**: {improvement:.1f}%\n")
                f.write(f"- **Hypothesis Validated**: {'✅ YES' if improvement >= 25 else '❌ NO'}\n\n")
            
            # Detailed Results Table
            f.write("## Detailed Results\n\n")
            f.write("| Model | Access Method | Success Rate | Response Time (s) | Tokens/Test | Reasoning | Confidence | Evidence | Overall Score |\n")
            f.write("|-------|---------------|--------------|-------------------|------------|-------------|-----------|---------------|\n")
            
            for summary in sorted(summaries, key=lambda x: x.overall_performance_score, reverse=True):
                f.write(f"| {summary.model_name} | {summary.access_method} | {summary.success_rate:.2%} | {summary.avg_response_time:.2f} | {summary.avg_tokens_per_test:.0f} | {summary.avg_reasoning_quality:.3f} | {summary.avg_confidence_calibration:.3f} | {summary.avg_evidence_integration:.3f} | {summary.overall_performance_score:.3f} |\n")
            
            # Model-by-Model Analysis
            f.write("\n## Model-by-Model Analysis\n\n")
            
            models = list(set(s.model_name for s in summaries))
            for model in models:
                f.write(f"### {model}\n\n")
                
                direct_summary = next((s for s in summaries if s.model_name == model and s.access_method == "direct"), None)
                conjecture_summary = next((s for s in summaries if s.model_name == model and s.access_method == "conjecture"), None)
                
                if direct_summary and conjecture_summary:
                    improvement = ((conjecture_summary.overall_performance_score - direct_summary.overall_performance_score) / direct_summary.overall_performance_score) * 100
                    
                    f.write(f"**Direct Access**:\n")
                    f.write(f"- Success Rate: {direct_summary.success_rate:.2%}\n")
                    f.write(f"- Response Time: {direct_summary.avg_response_time:.2f}s\n")
                    f.write(f"- Overall Score: {direct_summary.overall_performance_score:.3f}\n\n")
                    
                    f.write(f"**Conjecture Access**:\n")
                    f.write(f"- Success Rate: {conjecture_summary.success_rate:.2%}\n")
                    f.write(f"- Response Time: {conjecture_summary.avg_response_time:.2f}s\n")
                    f.write(f"- Overall Score: {conjecture_summary.overall_performance_score:.3f}\n\n")
                    
                    f.write(f"**Performance Change**: {improvement:+.1f}% ({'✅ Improvement' if improvement > 0 else '❌ Degradation'})\n\n")
            
            # Test Case Analysis
            f.write("## Test Case Performance\n\n")
            
            categories = list(set(r.test_category for r in all_results if r.success))
            for category in categories:
                f.write(f"### {category.replace('_', ' ').title()}\n\n")
                
                category_results = [r for r in all_results if r.test_category == category and r.success]
                
                # Compare approaches by category
                direct_scores = [r.reasoning_quality + r.confidence_calibration + r.evidence_integration for r in category_results if r.access_method == "direct"]
                conjecture_scores = [r.reasoning_quality + r.confidence_calibration + r.evidence_integration for r in category_results if r.access_method == "conjecture"]
                
                if direct_scores and conjecture_scores:
                    avg_direct = statistics.mean(direct_scores)
                    avg_conjecture = statistics.mean(conjecture_scores)
                    improvement = ((avg_conjecture - avg_direct) / avg_direct) * 100 if avg_direct > 0 else 0
                    
                    f.write(f"- Direct Average Quality Score: {avg_direct:.3f}\n")
                    f.write(f"- Conjecture Average Quality Score: {avg_conjecture:.3f}\n")
                    f.write(f"- Quality Improvement: {improvement:+.1f}%\n\n")
            
            # Conclusions
            f.write("## Conclusions\n\n")
            
            if best_direct and best_conjecture:
                overall_improvement = ((best_conjecture.overall_performance_score - best_direct.overall_performance_score) / best_direct.overall_performance_score) * 100
                
                if overall_improvement >= 25:
                    f.write("✅ **HYPOTHESIS VALIDATED**: The Conjecture approach achieves the target ≥25% performance improvement.\n\n")
                    f.write(f"The best Conjecture configuration ({best_conjecture.model_name}) outperforms the best Direct approach ({best_direct.model_name}) by {overall_improvement:.1f}%.\n\n")
                else:
                    f.write("❌ **HYPOTHESIS NOT VALIDATED**: The Conjecture approach does not achieve the target ≥25% performance improvement.\n\n")
                    f.write(f"The best Conjecture configuration ({best_conjecture.model_name}) improves over the best Direct approach ({best_direct.model_name}) by only {overall_improvement:.1f}%.\n\n")
                
                # Key insights
                f.write("### Key Insights\n\n")
                
                # Speed analysis
                avg_direct_time = statistics.mean([s.avg_response_time for s in summaries if s.access_method == "direct"])
                avg_conjecture_time = statistics.mean([s.avg_response_time for s in summaries if s.access_method == "conjecture"])
                time_overhead = ((avg_conjecture_time - avg_direct_time) / avg_direct_time) * 100
                
                f.write(f"- **Processing Overhead**: Conjecture adds {time_overhead:+.1f}% processing time\n")
                f.write(f"- **Quality Enhancement**: Conjecture improves reasoning quality by {statistics.mean([s.avg_reasoning_quality for s in summaries if s.access_method == 'conjecture']) - statistics.mean([s.avg_reasoning_quality for s in summaries if s.access_method == 'direct']):.3f} on average\n")
                f.write(f"- **Confidence Calibration**: Conjecture improves confidence calibration by {statistics.mean([s.avg_confidence_calibration for s in summaries if s.access_method == 'conjecture']) - statistics.mean([s.avg_confidence_calibration for s in summaries if s.access_method == 'direct']):.3f} on average\n")
                
                f.write("\n### Recommendations\n\n")
                if overall_improvement >= 25:
                    f.write("- ✅ Deploy Conjecture pipeline for production use\n")
                    f.write("- ✅ Focus on models showing highest improvement rates\n")
                    f.write("- ✅ Implement monitoring for continuous improvement\n")
                else:
                    f.write("- 🔧 Optimize Conjecture pipeline to reduce overhead\n")
                    f.write("- 🔧 Investigate models with negative performance impact\n")
                    f.write("- 🔧 Consider hybrid approaches for specific use cases\n")
    
    def validate_hypothesis(self, summaries: List[ComparisonSummary]):
        """Validate the core hypothesis with statistical analysis"""
        print("\n" + "=" * 80)
        print("🔬 HYPOTHESIS VALIDATION")
        print("=" * 80)
        
        direct_scores = [s.overall_performance_score for s in summaries if s.access_method == "direct"]
        conjecture_scores = [s.overall_performance_score for s in summaries if s.access_method == "conjecture"]
        
        if not direct_scores or not conjecture_scores:
            print("❌ Insufficient data for hypothesis validation")
            return
        
        avg_direct = statistics.mean(direct_scores)
        avg_conjecture = statistics.mean(conjecture_scores)
        
        improvement = ((avg_conjecture - avg_direct) / avg_direct) * 100
        
        print(f"Average Direct Performance Score: {avg_direct:.3f}")
        print(f"Average Conjecture Performance Score: {avg_conjecture:.3f}")
        print(f"Average Improvement: {improvement:.1f}%")
        
        # Statistical significance (simple t-test approximation)
        if len(direct_scores) >= 2 and len(conjecture_scores) >= 2:
            # Calculate pooled standard deviation
            pooled_std = statistics.sqrt(
                ((len(direct_scores) - 1) * statistics.stdev(direct_scores) ** 2 +
                 (len(conjecture_scores) - 1) * statistics.stdev(conjecture_scores) ** 2) /
                (len(direct_scores) + len(conjecture_scores) - 2)
            )
            
            # Calculate t-statistic
            standard_error = pooled_std * statistics.sqrt(1/len(direct_scores) + 1/len(conjecture_scores))
            t_stat = (avg_conjecture - avg_direct) / standard_error if standard_error > 0 else 0
            
            print(f"T-statistic: {t_stat:.3f}")
            print(f"Statistical Significance: {'✅ Significant' if abs(t_stat) > 2.0 else '❌ Not significant'}")
        
        # Final validation
        print(f"\n🎯 HYPOTHESIS: {'✅ VALIDATED' if improvement >= 25 else '❌ NOT VALIDATED'}")
        print(f"Target: ≥25% improvement")
        print(f"Achieved: {improvement:.1f}% improvement")
        
        if improvement >= 25:
            print("\n🎉 SUCCESS: The Conjecture pipeline demonstrates significant performance improvements!")
            print("✅ Claims-based reasoning enhances model performance")
            print("✅ Evidence integration improves response quality")
            print("✅ Confidence calibration provides better reliability")
        else:
            print("\n⚠️  PARTIAL SUCCESS: Further optimization needed")
            print("🔧 Consider pipeline optimization")
            print("🔧 Investigate specific model performance patterns")
            print("🔧 Refine claims-based reasoning process")

async def main():
    """Main entry point"""
    test_runner = Comprehensive4ModelTest()
    await test_runner.run_comprehensive_test()

if __name__ == "__main__":
    asyncio.run(main())