#!/usr/bin/env python3
"""
4-Model Comparison Research Script
Compares model performance when accessed directly vs through Conjecture provider
Uses huggingface datasets for scaling test samples
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
project_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_root, "src"))

# Try to import required libraries
try:
    import aiohttp
    from datasets import load_dataset

    print("[OK] Required libraries available")
except ImportError as e:
    print(f"[FAIL] Missing required library: {e}")
    print("Please install with: pip install aiohttp datasets")
    sys.exit(1)


@dataclass
class ModelTestResult:
    """Result of testing a model on a specific task"""

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

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ComparisonSummary:
    """Summary statistics for model comparison"""

    model_name: str
    access_method: str
    total_tests: int
    successful_tests: int
    success_rate: float
    avg_response_time: float
    total_tokens: int
    avg_tokens_per_test: float
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class ModelProvider:
    """Interface for different model providers"""

    async def call_model(self, prompt: str, model_name: str) -> Tuple[str, float, int]:
        """Make a call to the model and return (response, time, tokens)"""
        raise NotImplementedError


class LMStudioProvider(ModelProvider):
    """Provider for LM Studio local models"""

    def __init__(self, base_url: str = "http://localhost:1234"):
        self.base_url = base_url

    async def call_model(self, prompt: str, model_name: str) -> Tuple[str, float, int]:
        """Call LM Studio model directly"""
        start_time = time.time()

        request_data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 2000,
            "temperature": 0.7,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/chat/completions", json=request_data, timeout=60
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(
                        f"LM Studio API error: {response.status} - {error_text}"
                    )

                data = await response.json()
                response_time = time.time() - start_time

                if "choices" in data and len(data["choices"]) > 0:
                    content = data["choices"][0]["message"]["content"]
                else:
                    content = ""

                tokens_used = data.get("usage", {}).get("total_tokens", 0)
                return content, response_time, tokens_used


class ChutesProvider(ModelProvider):
    """Provider for ZAI API models"""

    def __init__(
        self, api_key: str, base_url: str = "https://api.z.ai/api/coding/paas/v4"
    ):


    async def call_model(self, prompt: str, model_name: str) -> Tuple[str, float, int]:
        """Call Chutes API model directly"""
        start_time = time.time()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        request_data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 2000,
            "temperature": 0.7,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=request_data,
                headers=headers,
                timeout=60,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(
                        f"Chutes API error: {response.status} - {error_text}"
                    )

                data = await response.json()
                response_time = time.time() - start_time

                if "choices" in data and len(data["choices"]) > 0:
                    content = data["choices"][0]["message"]["content"]
                else:
                    content = ""

                tokens_used = data.get("usage", {}).get("total_tokens", 0)
                return content, response_time, tokens_used


class ConjectureProvider(ModelProvider):
    """Provider for models through Conjecture"""

    def __init__(self, base_url: str = "http://127.0.0.1:5678"):
        self.base_url = base_url

    async def call_model(self, prompt: str, model_name: str) -> Tuple[str, float, int]:
        """Call model through Conjecture provider"""
        start_time = time.time()

        # Note: Conjecture doesn't actually use the model_name parameter
        # It's just passed for compatibility with the test framework
        request_data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 2000,
            "temperature": 0.7,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/chat/completions", json=request_data, timeout=60
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(
                        f"Conjecture API error: {response.status} - {error_text}"
                    )

                data = await response.json()
                response_time = time.time() - start_time

                if "choices" in data and len(data["choices"]) > 0:
                    content = data["choices"][0]["message"]["content"]
                else:
                    content = ""

                tokens_used = data.get("usage", {}).get("total_tokens", 0)
                return content, response_time, tokens_used


class ModelComparisonTest:
    """Main test class for 4-model comparison"""

    def __init__(self):
        self.providers = {
            "lmstudio": LMStudioProvider(),
            "chutes": ChutesProvider(os.getenv("ZAI_API_KEY", ""), os.getenv("ZAI_API_URL", "https://api.z.ai/api/coding/paas/v4")),
            "conjecture": ConjectureProvider(),
        }
        self.test_results = []

    def load_test_cases(self):
        """Load test cases from multiple sources"""
        test_cases = []

        # 1. Load from local test_cases directory
        test_case_dir = Path(project_root) / "research" / "test_cases"
        for case_file in test_case_dir.glob("*.json"):
            try:
                with open(case_file, "r") as f:
                    case_data = json.load(f)
                    test_cases.append(
                        {
                            "id": case_file.stem,
                            "category": case_file.stem.split("_")[0],
                            "data": case_data,
                        }
                    )
            except Exception as e:
                print(f"Error loading {case_file}: {e}")

        # 2. Load from huggingface datasets
        try:
            print("Loading huggingface datasets...")

            # Load MMLU dataset for academic reasoning
            mmlu = load_dataset("cais/mmlu", "all", split="test[:5%]")
            for i, item in enumerate(mmlu):
                test_cases.append(
                    {
                        "id": f"hugg_mmlu_{i}",
                        "category": "academic_reasoning",
                        "data": {
                            "question": item["question"],
                            "choices": item["choices"],
                            "answer": item["answer"],
                        },
                    }
                )
                if i >= 10:  # Limit for testing
                    break

            # Load TruthfulQA for hallucination testing
            truthful_qa = load_dataset(
                "truthful_qa", "generation", split="validation[:5%]"
            )
            for i, item in enumerate(truthful_qa):
                test_cases.append(
                    {
                        "id": f"hugg_truthful_{i}",
                        "category": "hallucination",
                        "data": {
                            "question": item["question"],
                            "correct_answers": item["correct_answers"],
                            "incorrect_answers": item["incorrect_answers"],
                        },
                    }
                )
                if i >= 10:  # Limit for testing
                    break

            # Load HumanEval for coding tasks
            humaneval = load_dataset("openai_humaneval", split="test[:5%]")
            for i, item in enumerate(humaneval):
                test_cases.append(
                    {
                        "id": f"hugg_code_{i}",
                        "category": "coding",
                        "data": {
                            "prompt": item["prompt"],
                            "canonical_solution": item["canonical_solution"],
                        },
                    }
                )
                if i >= 10:  # Limit for testing
                    break

            print(f"Loaded {len(test_cases)} test cases including huggingface datasets")

        except Exception as e:
            print(f"Error loading huggingface datasets: {e}")
            print("Continuing with local test cases only")

        # 3. Add custom hallucination and impossible question tests
        custom_tests = [
            {
                "id": "custom_hallucination_1",
                "category": "hallucination",
                "data": {
                    "question": "Explain the research of Dr. Alexandra Chen on quantum photosynthesis",
                    "context": "Dr. Alexandra Chen is a fictional scientist who doesn't exist",
                },
            },
            {
                "id": "custom_hallucination_2",
                "category": "hallucination",
                "data": {
                    "question": "Describe the technical specifications of the Quantum Computing Processor Model X-9000",
                    "context": "The X-9000 is a non-existent quantum computer model",
                },
            },
            {
                "id": "custom_impossible_1",
                "category": "impossible",
                "data": {
                    "question": "Using only a compass and straightedge, construct a perfect square with an area of 10",
                    "context": "This is mathematically impossible due to the transcendental nature of sqrt(10)",
                },
            },
            {
                "id": "custom_impossible_2",
                "category": "impossible",
                "data": {
                    "question": "Design a Turing machine that can solve the halting problem for all possible inputs",
                    "context": "This is a classic undecidable problem in computer science",
                },
            },
        ]

        test_cases.extend(custom_tests)
        return test_cases

    def format_prompt(self, test_case):
        """Format a test case into a prompt"""
        category = test_case.get("category", "general")
        data = test_case.get("data", {})

        if category == "hallucination":
            return f"""Please answer the following question:

{data.get("question", "")}

Note: If you are not familiar with this topic or if it seems to refer to non-existent information, please state that clearly."""

        elif category == "impossible":
            return f"""Please attempt to solve the following problem:

{data.get("question", "")}

If you believe this problem is impossible to solve, please explain why."""

        elif category == "coding":
            return f"""Please write Python code to solve the following problem:

{data.get("prompt", "")}

Provide your solution with clear comments explaining your approach."""

        elif category == "academic_reasoning":
            choices_text = "\n".join(
                [
                    f"({chr(65 + i)}) {choice}"
                    for i, choice in enumerate(data.get("choices", []))
                ]
            )
            return f"""Please answer the following multiple-choice question:

{data.get("question", "")}

{choices_text}

Provide the letter of your answer and explain your reasoning."""

        elif category == "evidence":
            return f"""Please analyze the following evidence and provide your evaluation:

{data.get("question", "")}

Provide a balanced assessment with confidence levels."""

        else:  # General category
            if "question" in data:
                return data["question"]
            elif "task" in data:
                return data["task"]
            else:
                return str(data)

    async def test_model(self, provider_name, model_name, test_cases, access_method):
        """Test a specific model with all test cases"""
        print(
            f"Testing {model_name} via {access_method} using {provider_name} provider..."
        )

        provider = self.providers[provider_name]
        results = []

        for i, test_case in enumerate(test_cases):
            test_id = test_case["id"]
            category = test_case["category"]
            prompt = self.format_prompt(test_case)

            try:
                response, response_time, tokens = await provider.call_model(
                    prompt, model_name
                )
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
                )
                print(
                    f"  [{i + 1}/{len(test_cases)}] {test_id} ({category}): ✓ ({response_time:.2f}s)"
                )
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
                    error_message=str(e),
                )
                print(
                    f"  [{i + 1}/{len(test_cases)}] {test_id} ({category}): ✗ ({str(e)})"
                )

            results.append(result)
            self.test_results.append(result)

        return results

    def calculate_summary(self, results, model_name, access_method):
        """Calculate summary statistics for a set of results"""
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
            )

        success_rate = len(successful_results) / len(results)
        avg_response_time = statistics.mean(r.response_time for r in successful_results)
        total_tokens = sum(r.tokens_used for r in successful_results)
        avg_tokens_per_test = total_tokens / len(successful_results)

        return ComparisonSummary(
            model_name=model_name,
            access_method=access_method,
            total_tests=len(results),
            successful_tests=len(successful_results),
            success_rate=success_rate,
            avg_response_time=avg_response_time,
            total_tokens=total_tokens,
            avg_tokens_per_test=avg_tokens_per_test,
        )

    async def run_comparison(self):
        """Run the full 4-model comparison"""
        print("🚀 Starting 4-Model Comparison Test")
        print("=" * 60)

        # Load test cases
        test_cases = self.load_test_cases()
        print(f"Loaded {len(test_cases)} test cases")

        # Define model configurations
        model_configs = [
            {
                "name": "ibm/granite-4-h-tiny",
                "provider": "lmstudio",
                "access_method": "direct",
            },
            {
                "name": "ibm/granite-4-h-tiny",
                "provider": "conjecture",
                "access_method": "conjecture",
            },
            {
                "name": "glm-4.6",
                "provider": "chutes",
                "access_method": "direct",
            },
            {
                "name": "glm-4.6",
                "provider": "conjecture",
                "access_method": "conjecture",
            },
        ]

        # Run tests for each model configuration
        summaries = []
        for config in model_configs:
            results = await self.test_model(
                provider_name=config["provider"],
                model_name=config["name"],
                test_cases=test_cases,
                access_method=config["access_method"],
            )
            summary = self.calculate_summary(
                results, config["name"], config["access_method"]
            )
            summaries.append(summary)

            print(f"\nResults for {config['name']} ({config['access_method']}):")
            print(f"  Success Rate: {summary.success_rate:.2%}")
            print(f"  Average Response Time: {summary.avg_response_time:.2f}s")
            print(f"  Average Tokens per Test: {summary.avg_tokens_per_test:.0f}")

        # Save results
        output_dir = Path(project_root) / "research" / "results"
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        results_file = output_dir / f"4model_comparison_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump([asdict(result) for result in self.test_results], f, indent=2)

        # Save summary
        summary_file = output_dir / f"4model_comparison_summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump([asdict(summary) for summary in summaries], f, indent=2)

        # Generate comparison report
        report_file = output_dir / f"4model_comparison_report_{timestamp}.md"
        self.generate_report(summaries, report_file)

        print("\n" + "=" * 60)
        print("🎉 4-Model Comparison Test Complete!")
        print(f"Results saved to: {results_file}")
        print(f"Report saved to: {report_file}")

        # Print final comparison
        self.print_comparison_table(summaries)

    def generate_report(self, summaries, output_file):
        """Generate a markdown report of the comparison"""
        with open(output_file, "w") as f:
            f.write("# 4-Model Comparison Report\n\n")
            f.write(f"Generated on: {datetime.now().isoformat()}\n\n")

            # Create comparison table
            f.write("## Summary Results\n\n")
            f.write(
                "| Model | Access Method | Success Rate | Avg Response Time (s) | Avg Tokens per Test |\n"
            )
            f.write(
                "|-------|---------------|--------------|----------------------|---------------------|\n"
            )

            for summary in summaries:
                f.write(
                    f"| {summary.model_name} | {summary.access_method} | {summary.success_rate:.2%} | {summary.avg_response_time:.2f} | {summary.avg_tokens_per_test:.0f} |\n"
                )

            # Add analysis
            f.write("\n## Analysis\n\n")

            # Group by model
            granite_direct = next(
                (
                    s
                    for s in summaries
                    if s.model_name == "ibm/granite-4-h-tiny"
                    and s.access_method == "direct"
                ),
                None,
            )
            granite_conj = next(
                (
                    s
                    for s in summaries
                    if s.model_name == "ibm/granite-4-h-tiny"
                    and s.access_method == "conjecture"
                ),
                None,
            )

            glm_direct = next(
                (
                    s
                    for s in summaries
                    if s.model_name == "zai-org/GLM-4.6-FP8"
                    and s.access_method == "direct"
                ),
                None,
            )
            glm_conj = next(
                (
                    s
                    for s in summaries
                    if s.model_name == "zai-org/GLM-4.6-FP8"
                    and s.access_method == "conjecture"
                ),
                None,
            )

            # Compare access methods
            f.write("### Access Method Comparison\n\n")

            if granite_direct and granite_conj:
                f.write(f"**IBM Granite-4-H-Tiny:**\n")
                f.write(f"- Direct success rate: {granite_direct.success_rate:.2%}\n")
                f.write(f"- Conjecture success rate: {granite_conj.success_rate:.2%}\n")
                f.write(
                    f"- Direct avg response time: {granite_direct.avg_response_time:.2f}s\n"
                )
                f.write(
                    f"- Conjecture avg response time: {granite_conj.avg_response_time:.2f}s\n\n"
                )

            if glm_direct and glm_conj:
                f.write(f"**GLM-4.6:**\n")
                f.write(f"- Direct success rate: {glm_direct.success_rate:.2%}\n")
                f.write(f"- Conjecture success rate: {glm_conj.success_rate:.2%}\n")
                f.write(
                    f"- Direct avg response time: {glm_direct.avg_response_time:.2f}s\n"
                )
                f.write(
                    f"- Conjecture avg response time: {glm_conj.avg_response_time:.2f}s\n\n"
                )

    def print_comparison_table(self, summaries):
        """Print a formatted comparison table to console"""
        print("\n4-Model Comparison Summary:")
        print("-" * 80)
        print(
            f"{'Model':<25} {'Method':<12} {'Success':<10} {'Time (s)':<10} {'Tokens':<10}"
        )
        print("-" * 80)

        for summary in summaries:
            print(
                f"{summary.model_name:<25} {summary.access_method:<12} {summary.success_rate:<10.2%} {summary.avg_response_time:<10.2f} {summary.avg_tokens_per_test:<10.0f}"
            )

        print("-" * 80)


async def main():
    """Main entry point"""
    # Check environment
    if not os.getenv("ZAI_API_KEY"):
        print("⚠️ Warning: ZAI_API_KEY not set. ZAI API tests may fail.")

    # Check if required services are running
    import socket

    def is_port_open(host, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.settimeout(1)
                result = s.connect_ex((host, port))
                return result == 0
            except:
                return False

    if not is_port_open("localhost", 1234):
        print("❌ LM Studio is not running on port 1234")
        print("Please start LM Studio and load the ibm/granite-4-h-tiny model")
        return

    if not is_port_open("127.0.0.1", 5678):
        print("❌ Conjecture provider is not running on port 5678")
        print(
            "Please start the Conjecture provider with: python scripts/start_conjecture_provider.py"
        )
        return

    # Run the comparison
    test_runner = ModelComparisonTest()
    await test_runner.run_comparison()


if __name__ == "__main__":
    asyncio.run(main())
