#!/usr/bin/env python3
"""
Correctness-Focused Metrics Test for LLM Models
Tests model accuracy and correctness of responses
Usage: python correctness_metrics_test.py {model} {harness} {output_file_json}
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import httpx

class CorrectnessMetricsTest:
    """Correctness-focused metrics test for LLM models"""
    
    def __init__(self, model: str, harness: str = "router", output_file: str = "correctness_metrics.json"):
        self.model = model
        self.harness = harness
        self.output_file = output_file
        self.results = []
        
    async def test_math_correctness(self) -> Dict[str, Any]:
        """Test basic math correctness"""
        start_time = time.time()
        
        try:
            if self.harness == "router":
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        "http://localhost:5677/v1/chat/completions",
                        json={
                            "model": self.model,
                            "messages": [{"role": "user", "content": "What is 15 + 27? Give only the number."}],
                            "max_tokens": 50
                        }
                    )
                    
                    if response.status_code == 200:
                        end_time = time.time()
                        response_data = response.json()
                        content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        
                        # Check for correct answer
                        is_correct = "42" in content and len(content.strip()) < 20
                        
                        return {
                            "success": True,
                            "response_time": end_time - start_time,
                            "model": self.model,
                            "tokens_used": response_data.get("usage", {}).get("total_tokens", 0),
                            "response_content": content,
                            "is_correct": is_correct,
                            "test_type": "math_basic",
                            "expected_answer": "42",
                            "test_timestamp": datetime.now().isoformat()
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"HTTP {response.status_code}",
                            "response_time": time.time() - start_time,
                            "model": self.model,
                            "test_type": "math_basic"
                        }
            else:
                return {"success": False, "error": f"Unsupported harness: {self.harness}"}
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response_time": time.time() - start_time,
                "model": self.model,
                "test_type": "math_basic"
            }
    
    async def test_reasoning_correctness(self) -> Dict[str, Any]:
        """Test logical reasoning correctness"""
        start_time = time.time()
        
        try:
            if self.harness == "router":
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        "http://localhost:5677/v1/chat/completions",
                        json={
                            "model": self.model,
                            "messages": [{"role": "user", "content": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Answer yes or no and explain why."}],
                            "max_tokens": 100
                        }
                    )
                    
                    if response.status_code == 200:
                        end_time = time.time()
                        response_data = response.json()
                        content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        
                        # Check for correct logical reasoning
                        content_lower = content.lower()
                        has_yes = "yes" in content_lower
                        has_correct_reasoning = any(phrase in content_lower for phrase in ["some", "not all", "subset"])
                        
                        # Correct answer: Yes, because if all roses are flowers and some flowers fade quickly, then some roses (being a subset of all roses) must fade quickly
                        is_correct = has_yes and has_correct_reasoning
                        
                        return {
                            "success": True,
                            "response_time": end_time - start_time,
                            "model": self.model,
                            "tokens_used": response_data.get("usage", {}).get("total_tokens", 0),
                            "response_content": content,
                            "is_correct": is_correct,
                            "test_type": "logical_reasoning",
                            "expected_answer": "Yes with explanation about subset logic",
                            "test_timestamp": datetime.now().isoformat()
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"HTTP {response.status_code}",
                            "response_time": time.time() - start_time,
                            "model": self.model,
                            "test_type": "logical_reasoning"
                        }
            else:
                return {"success": False, "error": f"Unsupported harness: {self.harness}"}
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response_time": time.time() - start_time,
                "model": self.model,
                "test_type": "logical_reasoning"
            }
    
    async def test_factual_correctness(self) -> Dict[str, Any]:
        """Test factual knowledge correctness"""
        start_time = time.time()
        
        try:
            if self.harness == "router":
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        "http://localhost:5677/v1/chat/completions",
                        json={
                            "model": self.model,
                            "messages": [{"role": "user", "content": "What is the capital of France? Give only the city name."}],
                            "max_tokens": 50
                        }
                    )
                    
                    if response.status_code == 200:
                        end_time = time.time()
                        response_data = response.json()
                        content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        
                        # Check for correct answer
                        is_correct = "paris" in content.lower() and len(content.strip()) < 20
                        
                        return {
                            "success": True,
                            "response_time": end_time - start_time,
                            "model": self.model,
                            "tokens_used": response_data.get("usage", {}).get("total_tokens", 0),
                            "response_content": content,
                            "is_correct": is_correct,
                            "test_type": "factual_knowledge",
                            "expected_answer": "Paris",
                            "test_timestamp": datetime.now().isoformat()
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"HTTP {response.status_code}",
                            "response_time": time.time() - start_time,
                            "model": self.model,
                            "test_type": "factual_knowledge"
                        }
            else:
                return {"success": False, "error": f"Unsupported harness: {self.harness}"}
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response_time": time.time() - start_time,
                "model": self.model,
                "test_type": "factual_knowledge"
            }
    
    async def run_correctness_tests(self) -> Dict[str, Any]:
        """Run correctness-focused test suite"""
        print(f"TESTING CORRECTNESS: {self.model}")
        print(f"Harness: {self.harness}")
        print(f"Output: {self.output_file}")
        
        # Run correctness tests
        tests = [
            ("math_basic", self.test_math_correctness),
            ("logical_reasoning", self.test_reasoning_correctness),
            ("factual_knowledge", self.test_factual_correctness)
        ]
        
        results = []
        correct_count = 0
        total_tests = 0
        
        for test_name, test_func in tests:
            print(f"  Running {test_name} test...")
            result = await test_func()
            results.append(result)
            
            if result.get("success", False):
                total_tests += 1
                if result.get("is_correct", False):
                    correct_count += 1
                    print(f"  CORRECT {test_name}: {result.get('response_time', 0):.2f}s")
                else:
                    print(f"  INCORRECT {test_name}: {result.get('response_time', 0):.2f}s")
                    print(f"    Expected: {result.get('expected_answer', 'N/A')}")
                    print(f"    Got: {result.get('response_content', 'N/A')[:100]}...")
            else:
                print(f"  FAILED {test_name}: {result.get('error', 'Unknown error')}")
        
        # Calculate correctness metrics
        correctness_rate = correct_count / total_tests if total_tests > 0 else 0.0
        successful_tests = [r for r in results if r.get("success", False)]
        
        overall_metrics = {
            "model": self.model,
            "harness": self.harness,
            "test_timestamp": datetime.now().isoformat(),
            "total_tests": total_tests,
            "correct_tests": correct_count,
            "incorrect_tests": total_tests - correct_count,
            "correctness_rate": correctness_rate,
            "success_rate": len(successful_tests) / total_tests if total_tests > 0 else 0.0,
            "average_response_time": sum(r.get("response_time", 0) for r in successful_tests) / len(successful_tests) if successful_tests else 0.0,
            "total_tokens_used": sum(r.get("tokens_used", 0) for r in successful_tests),
            "test_results": results,
            "status": "PASS" if correctness_rate >= 0.8 else "FAIL"  # 80% correctness threshold
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
        
        # Print correctness summary
        print(f"\nCORRECTNESS SUMMARY for {self.model}:")
        print(f"   Status: {metrics['status']}")
        print(f"   Correctness Rate: {metrics['correctness_rate']:.1%}")
        print(f"   Correct: {metrics['correct_tests']}/{metrics['total_tests']}")
        print(f"   Avg Response Time: {metrics['average_response_time']:.2f}s")
        print(f"   Total Tokens: {metrics['total_tokens_used']}")


async def main():
    """Main function"""
    if len(sys.argv) != 4:
        print("Usage: python correctness_metrics_test.py {model} {harness} {output_file_json}")
        print("Example: python correctness_metrics_test.py openrouter/gpt-oss-20b:openai/gpt-oss-20b router gpt_oss_correctness.json")
        sys.exit(1)
    
    model = sys.argv[1]
    harness = sys.argv[2]
    output_file = sys.argv[3]
    
    # Create and run test
    test = CorrectnessMetricsTest(model, harness, output_file)
    await test.run_correctness_tests()


if __name__ == "__main__":
    asyncio.run(main())