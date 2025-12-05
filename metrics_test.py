#!/usr/bin/env python3
"""
Simple Metrics Test for LLM Models
Tests specific models and outputs JSON metrics
Usage: python metrics_test.py {model} {harness} {output_file_json}
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import httpx

class MetricsTest:
    """Simple metrics test for LLM models"""
    
    def __init__(self, model: str, harness: str = "router", output_file: str = "metrics_results.json"):
        self.model = model
        self.harness = harness
        self.output_file = output_file
        self.results = []
        
    async def test_model_response_time(self) -> Dict[str, Any]:
        """Test model response time and basic functionality"""
        start_time = time.time()
        
        try:
            if self.harness == "router":
                # Test through LLMLocalRouter
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        "http://localhost:5677/v1/chat/completions",
                        json={
                            "model": self.model,
                            "messages": [{"role": "user", "content": "Hello! Please respond with a brief greeting."}],
                            "max_tokens": 50
                        }
                    )
                    
                    if response.status_code == 200:
                        end_time = time.time()
                        response_data = response.json()
                        
                        return {
                            "success": True,
                            "response_time": end_time - start_time,
                            "model": self.model,
                            "tokens_used": response_data.get("usage", {}).get("total_tokens", 0),
                            "response_length": len(response_data.get("choices", [{}])[0].get("message", {}).get("content", "")),
                            "provider": response_data.get("provider", "unknown"),
                            "test_timestamp": datetime.now().isoformat()
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"HTTP {response.status_code}",
                            "response_time": time.time() - start_time,
                            "model": self.model
                        }
            else:
                return {"success": False, "error": f"Unsupported harness: {self.harness}"}
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response_time": time.time() - start_time,
                "model": self.model
            }
    
    async def test_model_reasoning(self) -> Dict[str, Any]:
        """Test model reasoning capabilities"""
        start_time = time.time()
        
        try:
            if self.harness == "router":
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        "http://localhost:5677/v1/chat/completions",
                        json={
                            "model": self.model,
                            "messages": [{"role": "user", "content": "What is 2+2? Please show your reasoning."}],
                            "max_tokens": 100
                        }
                    )
                    
                    if response.status_code == 200:
                        end_time = time.time()
                        response_data = response.json()
                        content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        
                        # Simple reasoning evaluation
                        has_reasoning = "reason" in content.lower() or "because" in content.lower() or "step" in content.lower()
                        has_correct_answer = "4" in content
                        
                        return {
                            "success": True,
                            "response_time": end_time - start_time,
                            "model": self.model,
                            "reasoning_detected": has_reasoning,
                            "correct_answer": has_correct_answer,
                            "tokens_used": response_data.get("usage", {}).get("total_tokens", 0),
                            "test_type": "reasoning",
                            "test_timestamp": datetime.now().isoformat()
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"HTTP {response.status_code}",
                            "test_type": "reasoning",
                            "model": self.model
                        }
            else:
                return {"success": False, "error": f"Unsupported harness: {self.harness}"}
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "test_type": "reasoning",
                "model": self.model
            }
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        print(f"TESTING model: {self.model}")
        print(f"Harness: {self.harness}")
        print(f"Output: {self.output_file}")
        
        # Run multiple test types
        tests = [
            ("basic_response", self.test_model_response_time),
            ("reasoning", self.test_model_reasoning)
        ]
        
        results = []
        for test_name, test_func in tests:
            print(f"  Running {test_name} test...")
            result = await test_func()
            results.append(result)
            
            if result.get("success", False):
                print(f"  SUCCESS {test_name}: ({result.get('response_time', 0):.2f}s)")
            else:
                print(f"  FAILED {test_name}: {result.get('error', 'Unknown error')}")
        
        # Calculate overall metrics
        successful_tests = [r for r in results if r.get("success", False)]
        total_tests = len(results)
        
        overall_metrics = {
            "model": self.model,
            "harness": self.harness,
            "test_timestamp": datetime.now().isoformat(),
            "total_tests": total_tests,
            "successful_tests": len(successful_tests),
            "success_rate": len(successful_tests) / total_tests if total_tests > 0 else 0.0,
            "average_response_time": sum(r.get("response_time", 0) for r in successful_tests) / len(successful_tests) if successful_tests else 0.0,
            "total_tokens_used": sum(r.get("tokens_used", 0) for r in successful_tests),
            "test_results": results,
            "status": "PASS" if len(successful_tests) == total_tests else "FAIL"
        }
        
        # Save results
        self.save_results(overall_metrics)
        
        return overall_metrics
    
    def save_results(self, metrics: Dict[str, Any]):
        """Save test results to JSON file"""
        output_path = Path(self.output_file)
        
        # Load existing results if file exists
        existing_results = []
        if output_path.exists():
            try:
                with open(output_path, 'r') as f:
                    existing_results = json.load(f)
            except:
                existing_results = []
        
        # Add new results
        existing_results.append(metrics)
        
        # Save all results
        with open(output_path, 'w') as f:
            json.dump(existing_results, f, indent=2)
        
        print(f"Results saved to {output_path}")
        
        # Print summary
        print(f"\nTEST SUMMARY for {self.model}:")
        print(f"   Status: {metrics['status']}")
        print(f"   Success Rate: {metrics['success_rate']:.1%}")
        print(f"   Avg Response Time: {metrics['average_response_time']:.2f}s")
        print(f"   Total Tokens: {metrics['total_tokens_used']}")


async def main():
    """Main function"""
    if len(sys.argv) != 4:
        print("Usage: python metrics_test.py {model} {harness} {output_file_json}")
        print("Example: python metrics_test.py openrouter/gpt-oss-20b:openai/gpt-oss-20b router gpt_oss_metrics.json")
        sys.exit(1)
    
    model = sys.argv[1]
    harness = sys.argv[2]
    output_file = sys.argv[3]
    
    # Create and run test
    test = MetricsTest(model, harness, output_file)
    await test.run_comprehensive_test()


if __name__ == "__main__":
    asyncio.run(main())