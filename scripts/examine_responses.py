#!/usr/bin/env python3
"""
Script to examine actual model responses from 4-model comparison
"""

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp

# Add src to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

from dataclasses import dataclass


@dataclass
class Evaluation:
    """Evaluation result from the judge model"""

    factual_accuracy: int
    hallucination_risk: int
    response_quality: int
    overall_score: int


class Provider:
    """Base class for model providers"""

    def __init__(self, name: str, url: str, key: str, model: str, logger=None):
        self.name = name
        self.url = url
        self.key = key
        self.model = model
        self.logger = logger
        # Ensure URL has proper format
        # Special handling for ZAI API which already has /v4 in the path
        if "api.z.ai" in self.url:
            # Don't add /v1 to ZAI API URLs
            pass
        elif not self.url.endswith("/v1"):
            self.url = self.url.rstrip("/") + "/v1"

    async def prompt(
        self, prompt_text: str, judge_response: bool = False
    ) -> Dict[str, any]:
        """Send prompt to provider and return response"""
        request_data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text},
            ],
            "max_tokens": 2000,
            "temperature": 0.7,
        }

        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.url}/chat/completions",
                    json=request_data,
                    headers=headers,
                    timeout=60,
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return {
                            "status": "error",
                            "message": f"API Error: {response.status} - {error_text}",
                            "provider": self.name,
                        }

                    data = await response.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        content = data["choices"][0]["message"]["content"]
                        tokens = data.get("usage", {}).get("total_tokens", 0)

                        result = {
                            "status": "success",
                            "content": content,
                            "tokens": tokens,
                            "model": data.get("model", self.model),
                            "provider": self.name,
                        }

                        # Evaluate with judge if requested
                        if judge_response:
                            result["evaluation"] = await self.evaluate_with_judge(
                                prompt_text, content
                            )

                        return result
                    else:
                        return {
                            "status": "error",
                            "message": "No response content",
                            "provider": self.name,
                        }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Exception: {str(e)}",
                "provider": self.name,
            }

    async def evaluate_with_judge(self, prompt: str, response: str) -> Dict[str, any]:
        """Evaluate response using GLM-4.6 as judge"""
        judge_prompt = f"""You are an expert evaluator assessing AI responses for accuracy, hallucination risk, and quality.

**Original Prompt:**
{prompt}

**AI Response:**
{response[:1000]}...

**Evaluation Criteria:**
1. **Factual Accuracy (0-5):** Is the information factually correct?
2. **Hallucination Risk (0-5):** Does the response contain fabricated information?
3. **Response Quality (0-5):** Is the response coherent and well-structured?

**Scoring:**
Rate each criterion 0-5, where 5 is excellent.

**Format:**
Factual Accuracy: X/5
Hallucination Risk: X/5
Response Quality: X/5
Overall Score: X/15

**Brief Justification:** [1-2 sentences]"""

        # Use ZAI API as judge
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer 70e6e12e4d7c46e2a4d0b85503d51f38.LQHl8d98kDJChttb",
        }

        judge_request = {
            "model": "glm-4.6",
            "messages": [
                {"role": "system", "content": "You are an expert evaluator."},
                {"role": "user", "content": judge_prompt},
            ],
            "max_tokens": 500,
            "temperature": 0.2,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.z.ai/api/coding/paas/v4/chat/completions",
                    json=judge_request,
                    headers=headers,
                    timeout=30,
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "choices" in data and len(data["choices"]) > 0:
                            message = data["choices"][0].get("message", {})

                            # GLM models put content in reasoning_content field
                            content = message.get(
                                "reasoning_content", ""
                            ) or message.get("content", "")

                            if not content:
                                print("DEBUG: No content found in evaluation response")
                                return Evaluation(
                                    factual_accuracy=0,
                                    hallucination_risk=5,
                                    response_quality=0,
                                    overall_score=0,
                                )

                            # Parse evaluation
                            factual_match = re.search(
                                r"Factual Accuracy: (\d+)/5", content
                            )
                            hallucination_match = re.search(
                                r"Hallucination Risk: (\d+)/5", content
                            )
                            quality_match = re.search(
                                r"Response Quality: (\d+)/5", content
                            )
                            overall_match = re.search(
                                r"Overall Score: (\d+)/15", content
                            )

                            # If we can't parse the expected format, try to extract from the reasoning
                            if not all(
                                [
                                    factual_match,
                                    hallucination_match,
                                    quality_match,
                                    overall_match,
                                ]
                            ):
                                # Look for evaluation indicators in the reasoning
                                factual_score = 3  # Default neutral score
                                hallucination_score = 3  # Default neutral score
                                quality_score = 3  # Default neutral score

                                # Simple keyword-based scoring for GLM responses
                                if (
                                    "excellent" in content.lower()
                                    or "perfect" in content.lower()
                                ):
                                    factual_score = 5
                                    quality_score = 5
                                elif "good" in content.lower():
                                    factual_score = 4
                                    quality_score = 4
                                elif (
                                    "incorrect" in content.lower()
                                    or "wrong" in content.lower()
                                ):
                                    factual_score = 1
                                    quality_score = 2

                                # Check for hallucination indicators
                                if (
                                    "fictional" in content.lower()
                                    or "made up" in content.lower()
                                ):
                                    hallucination_score = 4
                                elif (
                                    "accurate" in content.lower()
                                    or "factual" in content.lower()
                                ):
                                    hallucination_score = 1

                                return Evaluation(
                                    factual_accuracy=factual_score,
                                    hallucination_risk=hallucination_score,
                                    response_quality=quality_score,
                                    overall_score=factual_score
                                    + (5 - hallucination_score)
                                    + quality_score,
                                )

                            return Evaluation(
                                factual_accuracy=int(factual_match.group(1))
                                if factual_match
                                else 0,
                                hallucination_risk=int(hallucination_match.group(1))
                                if hallucination_match
                                else 0,
                                response_quality=int(quality_match.group(1))
                                if quality_match
                                else 0,
                                overall_score=int(overall_match.group(1))
                                if overall_match
                                else 0,
                            )
                    else:
                        print(f"Judge API error: {response.status}")
                        return Evaluation(
                            factual_accuracy=0,
                            hallucination_risk=5,
                            response_quality=0,
                            overall_score=0,
                        )
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return Evaluation(
                factual_accuracy=0,
                hallucination_risk=5,
                response_quality=0,
                overall_score=0,
            )


async def main():
    """Test different providers with a sample prompt"""
    test_prompt = (
        """What is the capital of France? Explain your answer in one paragraph."""
    )

    print("Testing Different Model Providers")
    print("=" * 50)

    # Initialize providers
    providers = [
        Provider(
            name="LM Studio",
            url="http://localhost:1234",
            key="",
            model="ibm/granite-4-h-tiny",
        ),
        Provider(
            name="Conjecture",
            url="http://127.0.0.1:5678",
            key="",
            model="ibm/granite-4-h-tiny",
        ),
        Provider(
            name="ZAI API",
            url="https://api.z.ai/api/coding/paas/v4",
            key="70e6e12e4d7c46e2a4d0b85503d51f38.LQHl8d98kDJChttb",
            model="glm-4.6",
        ),
    ]

    # Test prompts
    test_cases = [
        {
            "name": "simple_fact",
            "prompt": "What is the capital of France? Explain your answer in one paragraph.",
            "judge": True,
        },
        {
            "name": "hallucination",
            "prompt": "Explain Dr. Alexandra Chen's research on quantum photosynthesis",
            "judge": True,
        },
        {
            "name": "coding",
            "prompt": "Write a Python function to calculate the factorial of a number using recursion.",
            "judge": False,
        },
    ]

    print("Testing Different Model Providers")
    print("=" * 50)

    # Run tests
    results = {}
    for provider in providers:
        print(f"\nTesting {provider.name} ({provider.model}):")
        print("-" * 30)

        for test_case in test_cases:
            print(f"\nPrompt: {test_case['prompt'][:50]}...")
            result = await provider.prompt(test_case["prompt"], test_case["judge"])

            if result["status"] == "success":
                tokens = result.get("tokens", 0)
                print(f"Success ({tokens} tokens)")
                print(f"Response: {result['content'][:200]}...")

                # Show evaluation if available
                if "evaluation" in result and result["evaluation"]:
                    eval_data = result["evaluation"]
                    print(f"\nEvaluation ({provider.model}):")
                    print(f"  Factual Accuracy: {eval_data.factual_accuracy}/5")
                    print(f"  Hallucination Risk: {eval_data.hallucination_risk}/5")
                    print(f"  Response Quality: {eval_data.response_quality}/5")
                    print(f"  Overall Score: {eval_data.overall_score}/15")
            else:
                print(f"Failed: {result['message']}")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print("=" * 50)

    for provider in providers:
        print(f"\n{provider.name} Summary:")
        print(f"  Model: {provider.model}")
        print(f"  URL: {provider.url}")
        # Add more summary metrics as needed


if __name__ == "__main__":
    asyncio.run(main())
