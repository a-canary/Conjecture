#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Model Matrix runner to test provider configuration
"""

import asyncio
import json
import time
import os
import sys
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Enforce UTF-8 encoding globally
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Windows UTF-8 console handling
if sys.platform.startswith('win'):
    try:
        import ctypes
        import ctypes.wintypes

        kernel32 = ctypes.windll.kernel32
        STD_OUTPUT_HANDLE = -11
        mode = ctypes.wintypes.DWORD()
        handle = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
        kernel32.GetConsoleMode(handle, ctypes.byref(mode))
        mode.value |= 0x0004
        kernel32.SetConsoleMode(handle, mode)
        kernel32.SetConsoleOutputCP(65001)
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', errors='replace', buffering=1)
        sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', errors='replace', buffering=1)

from dataclasses import dataclass
import hashlib

@dataclass
class MatrixResult:
    model: str
    harness: str
    test_prompt: str
    response: str
    response_time: float
    response_length: int
    success: bool
    error: str = None
    relevance_score: float = 0.0
    coherence_score: float = 0.0
    accuracy_score: float = 0.0
    overall_score: float = 0.0

def load_config():
    """Load configuration from ~/.conjecture/config.json"""
    config_path = os.path.expanduser("~/.conjecture/config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"[*] Using config: {config_path}")
    return config

class SimpleModelMatrixRunner:
    """Simplified Model Matrix runner using direct API calls"""

    def __init__(self):
        self.config = load_config()
        self.results: List[MatrixResult] = []

        # Extract available models
        self.available_models = []
        providers = self.config.get("providers", {})
        for provider_name, provider_config in providers.items():
            model_name = provider_config.get("model", "")
            if model_name:
                clean_name = model_name.replace("ibm/", "").replace("zai-org/", "").replace("openai/", "")
                self.available_models.append({
                    "name": clean_name,
                    "original": model_name,
                    "url": provider_config.get("url", ""),
                    "name_field": provider_name,
                    "api_key": provider_config.get("key", "")
                })

        print(f"Found {len(self.available_models)} available models:")
        for model in self.available_models:
            print(f"  - {model['name']} ({model['original']}) - URL: {model['url']}")

        # Simple test prompts
        self.test_prompts = [
            "What is 2+2?",
            "What is the capital of France?",
            "Hello, how are you?"
        ]

    async def test_provider_directly(self, model_info: Dict, prompt: str) -> Dict:
        """Test provider with direct HTTP request"""
        import requests

        headers = {
            "Content-Type": "application/json",
        }

        if model_info["api_key"]:
            # GLM-4.6 uses direct API key, others use Bearer
            if model_info["name"] == "glm-4.6":
                headers["Authorization"] = model_info['api_key']
            else:
                headers["Authorization"] = f"Bearer {model_info['api_key']}"

        # Format for different providers
        if model_info["name"] == "granite-4-h-tiny":
            # LM Studio format
            data = {
                "model": model_info["original"],
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100
            }
            endpoint = f"{model_info['url']}/v1/chat/completions"
        else:
            # OpenAI-compatible format
            data = {
                "model": model_info["original"],
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100
            }
            endpoint = f"{model_info['url']}/chat/completions"

        try:
            response = requests.post(endpoint, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()

            # Extract response text
            if "choices" in result and result["choices"]:
                message = result["choices"][0]["message"]
                # GLM-4.6 uses reasoning_content field
                content = message.get("reasoning_content") or message.get("content", "")
            else:
                content = str(result)

            return {
                "success": True,
                "response": content,
                "response_time": 0.5,  # Mock timing
                "response_length": len(content)
            }

        except Exception as e:
            return {
                "success": False,
                "response": "",
                "response_time": 0,
                "response_length": 0,
                "error": str(e)
            }

    def calculate_quality_score(self, response: str) -> float:
        """Simple quality scoring based on response length and content"""
        if not response or len(response.strip()) < 10:
            return 20.0

        score = 50.0  # Base score

        # Length bonus
        if 10 <= len(response) <= 200:
            score += 20
        elif len(response) > 200:
            score += 10

        # Content bonus
        if any(word in response.lower() for word in ["answer", "solution", "paris", "hello"]):
            score += 15

        # Penalty for errors
        if "error" in response.lower() or "fail" in response.lower():
            score -= 30

        return min(100.0, max(0.0, score))

    async def run_simple_matrix(self):
        """Run simplified matrix test"""
        print("\n" + "="*80)
        print("SIMPLIFIED MODEL MATRIX TEST")
        print("="*80)

        for model_info in self.available_models:
            print(f"\n[*] Testing model: {model_info['name']}")

            for i, prompt in enumerate(self.test_prompts):
                print(f"  [+] Prompt {i+1}: {prompt[:30]}...")

                # Test both approaches
                for approach in ["Direct", "Conjecture"]:
                    test_prompt = prompt if approach == "Direct" else f"Analyze step by step: {prompt}"

                    start_time = time.time()
                    result = await self.test_provider_directly(model_info, test_prompt)
                    duration = time.time() - start_time

                    if result["success"]:
                        score = self.calculate_quality_score(result["response"])
                        print(f"    {approach}: Score {score:.1f} ({result['response'][:50]}...)")

                        matrix_result = MatrixResult(
                            model=model_info['name'],
                            harness=approach,
                            test_prompt=test_prompt,
                            response=result["response"],
                            response_time=duration,
                            response_length=result["response_length"],
                            success=True,
                            overall_score=score
                        )
                        self.results.append(matrix_result)
                    else:
                        print(f"    {approach}: FAILED - {result['error']}")

        print(f"\n[SUCCESS] Completed {len(self.results)} tests")
        return self.results

async def main():
    runner = SimpleModelMatrixRunner()
    await runner.run_simple_matrix()

if __name__ == "__main__":
    asyncio.run(main())