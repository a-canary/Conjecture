# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
OpenRouter Free Models - Pytest Fixtures
Verified working models for benchmarking without cost.

Models:
- openai/gpt-oss-20b:free - reasoning model, good for math
- nvidia/nemotron-3-nano-30b-a3b:free - reasoning model, 30B params

Requirements:
- OPENROUTER_API_KEY env var (trim whitespace)
- HTTP-Referer header required
- Free tier: 50 requests/day, 20 requests/min
- Privacy settings must allow free model publication
"""

import os
import pytest
import aiohttp
import asyncio
from typing import Optional, Dict, Any, Callable, Awaitable
from dataclasses import dataclass


@dataclass
class OpenRouterFreeConfig:
    """Configuration for OpenRouter free models"""
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    referer: str = "https://conjecture.dev"
    timeout: int = 120
    max_tokens: int = 500

    # Verified free models (2026-03-01)
    MODELS = {
        "gpt-oss-20b": "openai/gpt-oss-20b:free",
        "nemotron-30b": "nvidia/nemotron-3-nano-30b-a3b:free",
    }


class OpenRouterFreeClient:
    """Client for OpenRouter free-tier models"""

    def __init__(self, config: OpenRouterFreeConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            self.session = None

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.config.referer,
        }

    async def chat(
        self,
        prompt: str,
        model: str = "gpt-oss-20b",
        max_tokens: Optional[int] = None,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Send chat completion request to OpenRouter free model.

        Returns dict with:
        - content: response text
        - reasoning: chain-of-thought (if model supports)
        - model: model used
        - usage: token counts
        """
        if not self.session:
            self.session = aiohttp.ClientSession()

        model_id = self.config.MODELS.get(model, model)

        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature,
        }

        url = f"{self.config.base_url}/chat/completions"
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)

        async with self.session.post(
            url,
            json=payload,
            headers=self._get_headers(),
            timeout=timeout
        ) as response:
            data = await response.json()

            if response.status != 200:
                error = data.get("error", {})
                raise OpenRouterError(
                    code=response.status,
                    message=error.get("message", str(data)),
                    is_rate_limit=response.status == 429
                )

            choice = data["choices"][0]["message"]
            return {
                "content": choice.get("content", ""),
                "reasoning": choice.get("reasoning", ""),
                "model": data.get("model"),
                "usage": data.get("usage", {}),
            }

    async def test_connection(self, model: str = "gpt-oss-20b") -> bool:
        """Test if model is accessible"""
        try:
            result = await self.chat("What is 2+2?", model=model, max_tokens=50)
            return "4" in result["content"]
        except OpenRouterError as e:
            # Rate limit is expected for free tier - not a connection failure
            if e.is_rate_limit:
                return True  # Model exists but rate limited
            return False
        except Exception:
            return False


class OpenRouterError(Exception):
    """OpenRouter API error"""
    def __init__(self, code: int, message: str, is_rate_limit: bool = False):
        self.code = code
        self.message = message
        self.is_rate_limit = is_rate_limit
        super().__init__(f"OpenRouter error {code}: {message}")


# --- Pytest Fixtures ---

def get_openrouter_api_key() -> Optional[str]:
    """Get and clean API key from environment"""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    return key.strip() if key else None


@pytest.fixture(scope="session")
def openrouter_api_key():
    """Session-scoped API key fixture"""
    key = get_openrouter_api_key()
    if not key:
        pytest.skip("OPENROUTER_API_KEY not set")
    return key


@pytest.fixture(scope="session")
def openrouter_config(openrouter_api_key):
    """Session-scoped OpenRouter configuration"""
    return OpenRouterFreeConfig(api_key=openrouter_api_key)


@pytest.fixture(scope="function")
def openrouter_client(openrouter_config):
    """
    Function-scoped OpenRouter client factory.

    Usage in async tests:
        async with openrouter_client as client:
            result = await client.chat("prompt")
    """
    return OpenRouterFreeClient(openrouter_config)


@pytest.fixture(scope="session")
def free_models():
    """Available free models for benchmarking"""
    return list(OpenRouterFreeConfig.MODELS.keys())


# --- Model-specific fixtures ---

ModelCallable = Callable[[str], Awaitable[Dict[str, Any]]]


@pytest.fixture(scope="function")
def gpt_oss_20b(openrouter_config) -> ModelCallable:
    """
    GPT-OSS-20B model fixture.
    Returns an async callable that manages its own session.
    """
    async def call(prompt: str, **kwargs) -> Dict[str, Any]:
        async with OpenRouterFreeClient(openrouter_config) as client:
            return await client.chat(prompt, model="gpt-oss-20b", **kwargs)
    return call


@pytest.fixture(scope="function")
def nemotron_30b(openrouter_config) -> ModelCallable:
    """
    Nemotron-3-Nano-30B model fixture.
    Returns an async callable that manages its own session.
    """
    async def call(prompt: str, **kwargs) -> Dict[str, Any]:
        async with OpenRouterFreeClient(openrouter_config) as client:
            return await client.chat(prompt, model="nemotron-30b", **kwargs)
    return call


# --- Benchmark utilities ---

@pytest.fixture(scope="function")
def benchmark_prompt_factory():
    """Factory for creating benchmark prompts"""

    def create_math_prompt(question: str) -> str:
        return f"""Solve this math problem. Show your work and provide only the final numerical answer on the last line.

Question: {question}

Answer:"""

    def create_reasoning_prompt(question: str) -> str:
        return f"""Answer the following question with step-by-step reasoning.

Question: {question}

Reasoning and Answer:"""

    return {
        "math": create_math_prompt,
        "reasoning": create_reasoning_prompt,
    }
