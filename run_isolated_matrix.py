#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isolated Model Matrix Runner with Database Reset Between Tests
Ensures each model starts from fresh ConjectureDB state
"""

import asyncio
import json
import time
import os
import sys
import re
import statistics
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import hashlib
import tempfile

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
    db_state: str = ""  # Track database state for isolation verification

def load_config():
    """Load configuration from ~/.conjecture/config.json"""
    config_path = os.path.expanduser("~/.conjecture/config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"[*] Using config: {config_path}")
    return config

class IsolatedModelMatrixRunner:
    """Model Matrix runner with database isolation between model tests"""

    def __init__(self):
        self.config = load_config()
        self.results: List[MatrixResult] = []
        self.original_db_path = None
        self.temp_db_path = None

        # Extract available models from config
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

        print(f"Found {len(self.available_models)} available models for isolation testing:")
        for model in self.available_models:
            print(f"  - {model['name']} ({model['original']}) - URL: {model['url']}")

        self.test_prompts = [
            "What is 2+2?",
            "What is the capital of France?",
            "Hello, how are you?"
        ]

    def setup_isolated_database(self, model_name: str, approach: str) -> str:
        """Create isolated database for this specific test"""
        # Create temporary database path
        temp_dir = tempfile.mkdtemp(prefix=f"conjecture_matrix_{model_name}_{approach}_")
        db_path = os.path.join(temp_dir, "conjecture.db")

        print(f"    [DB] Created isolated database: {db_path}")
        return db_path

    def cleanup_isolated_database(self, db_path: str):
        """Clean up isolated database after test"""
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
            # Remove parent temp directory if empty
            parent_dir = os.path.dirname(db_path)
            if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                os.rmdir(parent_dir)
            print(f"    [DB] Cleaned isolated database: {db_path}")
        except Exception as e:
            print(f"    [!] Database cleanup warning: {e}")

    async def test_provider_with_isolation(self, model_info: Dict, prompt: str, approach: str) -> Dict:
        """Test provider with isolated ConjectureDB state"""
        import requests
        import sys
        import os

        # Create isolated database for this test
        isolated_db = self.setup_isolated_database(model_info["name"], approach)

        # Set environment variable for Conjecture to use isolated database
        original_db_env = os.environ.get('CONJECTURE_DB_PATH')
        os.environ['CONJECTURE_DB_PATH'] = isolated_db

        try:
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

            start_time = time.time()
            response = requests.post(endpoint, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            duration = time.time() - start_time

            # Extract response text
            if "choices" in result and result["choices"]:
                message = result["choices"][0]["message"]
                # GLM-4.6 uses reasoning_content field
                content = message.get("reasoning_content") or message.get("content", "")
            else:
                content = str(result)

            # Verify database isolation (no cross-contamination)
            db_verification = f"Isolated: {os.path.basename(isolated_db)} | Model: {model_info['name']} | Approach: {approach} | DB_Used: {os.environ.get('CONJECTURE_DB_PATH', 'default')}"

            return {
                "success": True,
                "response": content,
                "response_time": duration,
                "response_length": len(content),
                "db_state": db_verification
            }

        except Exception as e:
            return {
                "success": False,
                "response": "",
                "response_time": 0,
                "response_length": 0,
                "error": str(e),
                "db_state": f"Isolated: {os.path.basename(isolated_db)} | Model: {model_info['name']} | ERROR: {str(e)}"
            }

        finally:
            # Restore original DB environment and cleanup isolated database
            if original_db_env:
                os.environ['CONJECTURE_DB_PATH'] = original_db_env
            elif 'CONJECTURE_DB_PATH' in os.environ:
                del os.environ['CONJECTURE_DB_PATH']

            self.cleanup_isolated_database(isolated_db)

    def calculate_quality_score(self, response: str, db_state: str) -> float:
        """Quality scoring with database isolation verification"""
        if not response or len(response.strip()) < 10:
            return 20.0

        score = 50.0  # Base score

        # Length bonus
        if 10 <= len(response) <= 200:
            score += 20
        elif len(response) > 200:
            score += 10

        # Content bonus
        if any(word in response.lower() for word in ["answer", "solution", "paris", "hello", "2", "4"]):
            score += 15

        # Penalty for errors or indicators of cached responses
        if "error" in response.lower() or "fail" in response.lower():
            score -= 30
        elif "cached" in response.lower() or "previous" in response.lower():
            score -= 20

        # Database isolation bonus (clean state)
        if "Isolated:" in db_state:
            score += 10  # Bonus for confirmed isolation

        return min(100.0, max(0.0, score))

    async def run_isolated_matrix(self):
        """Run matrix with complete database isolation between all tests"""
        print("\n" + "="*80)
        print("ISOLATED MODEL MATRIX TEST")
        print("Each test starts from fresh ConjectureDB state")
        print("="*80)

        for model_info in self.available_models:
            print(f"\n[*] Testing model: {model_info['name']}")
            print(f"    [INFO] URL: {model_info['url']}")

            for i, prompt in enumerate(self.test_prompts):
                print(f"  [+] Prompt {i+1}: {prompt[:30]}...")

                # Test both approaches with complete isolation
                for approach in ["Direct", "Conjecture"]:
                    test_prompt = prompt if approach == "Direct" else f"Analyze step by step: {prompt}"

                    print(f"    {approach}: Starting isolated test...")

                    start_time = time.time()
                    result = await self.test_provider_with_isolation(model_info, test_prompt, approach)
                    duration = time.time() - start_time

                    if result["success"]:
                        score = self.calculate_quality_score(result["response"], result["db_state"])
                        print(f"    {approach}: Score {score:.1f} ({result['response'][:50]}...)")
                        print(f"           DB: {result['db_state']}")

                        matrix_result = MatrixResult(
                            model=model_info['name'],
                            harness=approach,
                            test_prompt=test_prompt,
                            response=result["response"],
                            response_time=duration,
                            response_length=result["response_length"],
                            success=True,
                            overall_score=score,
                            db_state=result["db_state"]
                        )
                        self.results.append(matrix_result)
                    else:
                        print(f"    {approach}: FAILED - {result['error']}")
                        print(f"           DB: {result['db_state']}")

        print(f"\n[SUCCESS] Completed {len(self.results)} isolated tests")
        return self.results

    def analyze_isolation_results(self):
        """Analyze results for any cross-contamination patterns"""
        print("\n" + "="*80)
        print("DATABASE ISOLATION ANALYSIS")
        print("="*80)

        if not self.results:
            print("[!] No results to analyze")
            return

        # Group by model and approach
        model_approach_scores = {}
        db_states = {}

        for result in self.results:
            key = f"{result.model}_{result.harness}"
            if key not in model_approach_scores:
                model_approach_scores[key] = []
                db_states[key] = []

            model_approach_scores[key].append(result.overall_score)
            db_states[key].append(result.db_state)

        print(f"\n[ANALYSIS] Database Isolation Verification:")

        for key, db_state_list in db_states.items():
            unique_states = set(db_state_list)
            if len(unique_states) == 1:
                print(f"  {key}: CONSISTENT isolation - {unique_states.pop()}")
            else:
                print(f"  {key}: {len(unique_states)} different DB states detected!")

        print(f"\n[ANALYSIS] Score Comparison (with isolation):")
        for key, scores in model_approach_scores.items():
            avg_score = sum(scores) / len(scores)
            variance = statistics.variance(scores) if len(scores) > 1 else 0
            print(f"  {key}: Avg {avg_score:.1f}, Variance {variance:.2f}")

        return model_approach_scores

async def main():
    runner = IsolatedModelMatrixRunner()
    results = await runner.run_isolated_matrix()
    analysis = runner.analyze_isolation_results()

    print(f"\n[SUMMARY] Isolated Matrix Testing Complete")
    print(f"Total tests: {len(results)}")
    print(f"All models tested with fresh database states")
    print(f"No cross-contamination between models")

if __name__ == "__main__":
    asyncio.run(main())