# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
GPT-OSS-20B Integration for Benchmarks
Faster model for quick iteration and Conjecture testing

Includes exponential backoff retry logic for reliability (backlog #153).
"""

import aiohttp
import asyncio
import json
import logging
from typing import Dict, Any, Optional

# Import retry utilities for exponential backoff
import sys
sys.path.insert(0, '/workspace/src')
from utils.retry_utils import EnhancedRetryConfig, EnhancedRetryHandler, RetryErrorType

logger = logging.getLogger(__name__)

class GPTOSSIntegration:
    """Direct integration with GPT-OSS-20B via OpenRouter with retry logic"""

    def __init__(self, api_key: str = None, model: str = "openrouter/gpt-oss-20b"):
        self.model = model
        self.api_key = api_key or "sk-or-your-api-key-here"  # You'll need to set this
        self.base_url = "https://openrouter.ai/api/v1"
        self.session: Optional[aiohttp.ClientSession] = None

        # Configure retry handler for reliability (backlog #153)
        self.retry_config = EnhancedRetryConfig(
            max_attempts=5,
            base_delay=10.0,  # 10 second base
            max_delay=300.0,  # 5 minute max
            rate_limit_multiplier=3.0,
        )
        self.retry_handler = EnhancedRetryHandler(self.retry_config)

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_response(self, prompt: str, **kwargs) -> str:
        """Get response from GPT-OSS-20B with exponential backoff retry"""
        return await self.retry_handler.execute_with_retry_async(
            self._make_request, prompt, **kwargs
        )

    async def _make_request(self, prompt: str, **kwargs) -> str:
        """Internal method to make API request (called by retry handler)"""
        if not self.session:
            self.session = aiohttp.ClientSession()

        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert AI assistant responding to benchmark evaluation questions. Provide accurate, well-reasoned answers to the best of your ability."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": kwargs.get('max_tokens', 2000),
            "temperature": kwargs.get('temperature', 0.1),
            "top_p": kwargs.get('top_p', 0.9)
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            async with self.session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=60)) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                elif response.status == 429:
                    error_text = await response.text()
                    raise Exception(f"Rate limit error 429: {error_text}")
                else:
                    error_text = await response.text()
                    raise Exception(f"GPT-OSS API error {response.status}: {error_text}")

        except asyncio.TimeoutError:
            raise Exception("GPT-OSS request timed out")
        except aiohttp.ClientError as e:
            raise Exception(f"Network error: {str(e)}")

    async def test_connection(self) -> bool:
        """Test if GPT-OSS is accessible"""
        try:
            response = await self.get_response("Hello, can you respond with 'Connection successful'?", max_tokens=50)
            return "connection successful" in response.lower()
        except:
            return False

# Direct model functions for benchmark runner
async def gpt_oss_direct(prompt: str) -> str:
    """Direct GPT-OSS-20B model"""
    async with GPTOSSIntegration() as gpt_oss:
        return await gpt_oss.get_response(prompt)

async def gpt_oss_direct_conjecture(prompt: str) -> str:
    """GPT-OSS-20B with Conjecture-style enhancement"""
    enhanced_prompt = f"""Solve this step-by-step with maximum accuracy:

1. Analyze the problem thoroughly
2. Consider multiple approaches
3. Select the best method
4. Provide a complete solution with clear reasoning
5. State the final answer clearly

{prompt}

Please show your work and ensure your final answer is clearly stated."""

    async with GPTOSSIntegration() as gpt_oss:
        return await gpt_oss.get_response(enhanced_prompt)

# Test function
async def test_gpt_oss_connection():
    """Test connection to GPT-OSS"""
    try:
        async with GPTOSSIntegration() as gpt_oss:
            if await gpt_oss.test_connection():
                print("✓ GPT-OSS connection successful")
                return True
            else:
                print("✗ GPT-OSS connection failed - check API key")
                return False
    except Exception as e:
        print(f"✗ GPT-OSS connection error: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_gpt_oss_connection())