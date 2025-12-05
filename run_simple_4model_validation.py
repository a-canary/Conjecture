#!/usr/bin/env python3
"""
Simple 4-Model Validation Test
Focused test to validate hypothesis with shorter prompts
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
    """Simplified result for hypothesis validation"""
    model_name: str
    access_method: str
    test_case_id: str
    response: str
    response_time: float
    tokens_used: int
    success: bool
    error_message: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class ComparisonSummary:
    """Simplified summary for hypothesis validation"""
    model_name: str
    access_method: str
    total_tests: int
    successful_tests: int
    success_rate: float
    avg_response_time: float
    total_tokens: int
    avg_tokens_per_test: float
    overall_performance_score: float
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class SimpleModelProvider:
    """Simplified provider for testing"""
    
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
        """Simple model call"""
        start_time = time.time()
        
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
            raise e
    
    async def _call_lmstudio(self, prompt: str, model_name: str, start_time: float) -> Tuple[str, float, int]:
        """Call LM Studio model directly"""
        request_data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 500,  # Reduced tokens
            "temperature": 0.7,
        }
        
        async with self.session.post(
            f"{self.config['base_url']}/v1/chat/completions", 
            json=request_data, 
            timeout=30
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
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 500,  # Reduced tokens
            "temperature": 0.7,
        }
        
        async with self.session.post(
            f"{self.config['base_url']}/chat/completions",
            json=request_data,
            headers=headers,
            timeout=30,
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
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 500,  # Reduced tokens
            "temperature": 0.7,
        }
        
        async with self.session.post(
            f"{self.config['base_url']}/v1/chat/completions", 
            json=request_data, 
            timeout=60
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Conjecture API error: {response.status} - {error_text}")
            
            data = await response.json()
            response_time = time.time() - start_time
            
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            tokens_used = data.get("usage", {}).get("total_tokens", 0)
            
            return content, response_time, tokens_used

class Simple4ModelValidation:
    """Simple 4-model validation test"""
    
    def __init__(self):
        self.test_results = []
        self.output_dir = Path("research/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Provider configurations
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
                    "base_url": "http://localhost:5677",
                }
            }
        }
    
    def get_simple_test_cases(self) -> List[Dict[str, Any]]:
        """Get simple test cases for validation"""
        return [
            {
                "id": "math_basic",
                "category": "math",
                "question": "What is 15 + 27?"
            },
            {
                "id": "reasoning_simple",
                "category": "reasoning",
                "question": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?"
            },
            {
                "id": "knowledge_factual",
                "category": "knowledge",
                "question": "What is the capital of France?"
            },
            {
                "id": "creative_simple",
                "category": "creative",
                "question": "Name three benefits of regular exercise."
            }
        ]
    
    async def test_configuration(self, model_config: Dict[str, Any], test_cases: List[Dict[str, Any]]) -> List[ModelTestResult]:
        """Test a specific model configuration"""
        model_name = model_config["name"]
        provider_name = model_config["provider"]
        access_method = model_config["access_method"]
        
        print(f"\nTesting {model_name} via {access_method} using {provider_name}...")
        
        provider_config = self.provider_configs[provider_name]
        
        async with SimpleModelProvider(provider_config["type"], provider_config["config"]) as provider:
            results = []
            
            for i, test_case in enumerate(test_cases):
                test_id = test_case["id"]
                question = test_case["question"]
                
                print(f"  [{i+1}/{len(test_cases)}] {test_id}...", end=" ")
                
                try:
                    response, response_time, tokens = await provider.call_model(question, model_name)
                    
                    result = ModelTestResult(
                        model_name=model_name,
                        access_method=access_method,
                        test_case_id=test_id,
                        response=response,
                        response_time=response_time,
                        tokens_used=tokens,
                        success=True
                    )
                    
                    results.append(result)
                    print(f"SUCCESS ({response_time:.2f}s)")
                    
                except Exception as e:
                    result = ModelTestResult(
                        model_name=model_name,
                        access_method=access_method,
                        test_case_id=test_id,
                        response="",
                        response_time=0,
                        tokens_used=0,
                        success=False,
                        error_message=str(e)
                    )
                    
                    results.append(result)
                    print(f"FAILED ({str(e)[:30]}...)")
            
            return results
    
    def calculate_summary(self, results: List[ModelTestResult], model_name: str, access_method: str) -> ComparisonSummary:
        """Calculate summary statistics"""
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
                overall_performance_score=0.0
            )
        
        success_rate = len(successful_results) / len(results)
        avg_response_time = statistics.mean(r.response_time for r in successful_results)
        total_tokens = sum(r.tokens_used for r in successful_results)
        avg_tokens_per_test = total_tokens / len(successful_results)
        
        # Simple performance score (higher is better)
        # Combine success rate, speed, and efficiency
        speed_score = 1.0 / (1.0 + avg_response_time)  # Lower time = higher score
        efficiency_score = 1.0 / (1.0 + avg_tokens_per_test / 100)  # Lower tokens = higher score
        
        overall_performance_score = (
            0.5 * success_rate +
            0.3 * speed_score +
            0.2 * efficiency_score
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
            overall_performance_score=overall_performance_score
        )
    
    async def run_validation(self):
        """Run simple 4-model validation"""
        print("Starting Simple 4-Model Validation Test")
        print("=" * 60)
        print("Hypothesis: Conjecture approach achieves 25%+ performance improvement")
        print("=" * 60)
        
        # Get simple test cases
        test_cases = self.get_simple_test_cases()
        print(f"Using {len(test_cases)} simple test cases")
        
        # Define model configurations (simplified)
        model_configs = [
            {
                "name": "ibm/granite-4-h-tiny",
                "provider": "lmstudio",
                "access_method": "direct"
            },
            {
                "name": "lms/granite-4-h-tiny", 
                "provider": "conjecture",
                "access_method": "conjecture"
            },
            {
                "name": "glm-4.6",
                "provider": "chutes",
                "access_method": "direct"
            },
            {
                "name": "zai/GLM-4.6",
                "provider": "conjecture", 
                "access_method": "conjecture"
            }
        ]
        
        # Run tests
        all_results = []
        summaries = []
        
        for config in model_configs:
            results = await self.test_configuration(config, test_cases)
            all_results.extend(results)
            
            summary = self.calculate_summary(results, config["name"], config["access_method"])
            summaries.append(summary)
            
            print(f"\nResults for {config['name']} ({config['access_method']}):")
            print(f"  Success Rate: {summary.success_rate:.2%}")
            print(f"  Avg Response Time: {summary.avg_response_time:.2f}s")
            print(f"  Avg Tokens per Test: {summary.avg_tokens_per_test:.0f}")
            print(f"  Overall Performance Score: {summary.overall_performance_score:.3f}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results_file = self.output_dir / f"simple_4model_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump([asdict(result) for result in all_results], f, indent=2)
        
        summary_file = self.output_dir / f"simple_4model_summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump([asdict(summary) for summary in summaries], f, indent=2)
        
        # Generate report
        report_file = self.output_dir / f"simple_4model_report_{timestamp}.md"
        self.generate_report(summaries, report_file)
        
        print("\n" + "=" * 60)
        print("Simple 4-Model Validation Test Complete!")
        print(f"Results saved to: {results_file}")
        print(f"Report saved to: {report_file}")
        
        # Validate hypothesis
        self.validate_hypothesis(summaries)
        
        return summaries, all_results
    
    def generate_report(self, summaries: List[ComparisonSummary], output_file: Path):
        """Generate validation report"""
        with open(output_file, "w") as f:
            f.write("# Simple 4-Model Validation Report\n\n")
            f.write(f"Generated on: {datetime.now().isoformat()}\n\n")
            f.write("## Hypothesis\n\n")
            f.write("**Conjecture pipeline achieves 25%+ performance improvement over direct prompting.**\n\n")
            
            # Results table
            f.write("## Results Summary\n\n")
            f.write("| Model | Access Method | Success Rate | Response Time (s) | Performance Score |\n")
            f.write("|-------|---------------|--------------|-------------------|------------------|\n")
            
            for summary in sorted(summaries, key=lambda x: x.overall_performance_score, reverse=True):
                f.write(f"| {summary.model_name} | {summary.access_method} | {summary.success_rate:.2%} | {summary.avg_response_time:.2f} | {summary.overall_performance_score:.3f} |\n")
            
            # Analysis
            f.write("\n## Analysis\n\n")
            
            direct_summaries = [s for s in summaries if s.access_method == "direct"]
            conjecture_summaries = [s for s in summaries if s.access_method == "conjecture"]
            
            if direct_summaries and conjecture_summaries:
                avg_direct_score = statistics.mean([s.overall_performance_score for s in direct_summaries])
                avg_conjecture_score = statistics.mean([s.overall_performance_score for s in conjecture_summaries])
                
                improvement = ((avg_conjecture_score - avg_direct_score) / avg_direct_score) * 100
                
                f.write(f"**Average Direct Performance Score**: {avg_direct_score:.3f}\n")
                f.write(f"**Average Conjecture Performance Score**: {avg_conjecture_score:.3f}\n")
                f.write(f"**Performance Improvement**: {improvement:.1f}%\n")
                f.write(f"**Hypothesis Validated**: {'YES' if improvement >= 25 else 'NO'}\n\n")
                
                # Model-by-model comparison
                f.write("### Model-by-Model Comparison\n\n")
                
                for model in ["ibm/granite-4-h-tiny", "glm-4.6"]:
                    direct = next((s for s in direct_summaries if model in s.model_name), None)
                    conjecture = next((s for s in conjecture_summaries if model in s.model_name), None)
                    
                    if direct and conjecture:
                        model_improvement = ((conjecture.overall_performance_score - direct.overall_performance_score) / direct.overall_performance_score) * 100
                        
                        f.write(f"**{model}**:\n")
                        f.write(f"- Direct Score: {direct.overall_performance_score:.3f}\n")
                        f.write(f"- Conjecture Score: {conjecture.overall_performance_score:.3f}\n")
                        f.write(f"- Improvement: {model_improvement:+.1f}%\n\n")
            
            # Conclusions
            f.write("## Conclusions\n\n")
            if improvement >= 25:
                f.write("SUCCESS **HYPOTHESIS VALIDATED**: Conjecture achieves target >=25% improvement.\n")
                f.write("The claims-based reasoning pipeline provides significant performance benefits.\n")
            else:
                f.write("❌ **HYPOTHESIS NOT VALIDATED**: Conjecture does not achieve target ≥25% improvement.\n")
                f.write("Further optimization may be needed for the claims-based pipeline.\n")
    
    def validate_hypothesis(self, summaries: List[ComparisonSummary]):
        """Validate the core hypothesis"""
        print("\n" + "=" * 60)
        print("HYPOTHESIS VALIDATION")
        print("=" * 60)
        
        direct_scores = [s.overall_performance_score for s in summaries if s.access_method == "direct"]
        conjecture_scores = [s.overall_performance_score for s in summaries if s.access_method == "conjecture"]
        
        if not direct_scores or not conjecture_scores:
            print("Insufficient data for hypothesis validation")
            return
        
        avg_direct = statistics.mean(direct_scores)
        avg_conjecture = statistics.mean(conjecture_scores)
        
        improvement = ((avg_conjecture - avg_direct) / avg_direct) * 100
        
        print(f"Average Direct Performance Score: {avg_direct:.3f}")
        print(f"Average Conjecture Performance Score: {avg_conjecture:.3f}")
        print(f"Average Improvement: {improvement:.1f}%")
        
        print(f"\nHYPOTHESIS: {'VALIDATED' if improvement >= 25 else 'NOT VALIDATED'}")
        print(f"Target: >=25% improvement")
        print(f"Achieved: {improvement:.1f}% improvement")
        
        if improvement >= 25:
            print("\nSUCCESS: The Conjecture pipeline demonstrates significant performance improvements!")
            print("Claims-based reasoning enhances model performance")
        else:
            print("\nPARTIAL SUCCESS: Further optimization may be needed")
            print("Consider refining the claims-based reasoning process")

async def main():
    """Main entry point"""
    test_runner = Simple4ModelValidation()
    await test_runner.run_validation()

if __name__ == "__main__":
    asyncio.run(main())