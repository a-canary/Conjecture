# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Conjecture Model Wrapper for lm-evaluation-harness

This module provides a custom LM implementation that integrates Conjecture's
claim-based reasoning with the standard lm-eval benchmarking framework.

Usage:
    lm_eval run --model conjecture --model_args provider=cerebras,model=llama3.1-8b --tasks gsm8k
"""

import asyncio
import os
import re
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import httpx

from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model


@dataclass
class ClaimContext:
    """Tracks claims for accumulation"""
    claims: List[Dict] = None

    def __post_init__(self):
        if self.claims is None:
            self.claims = []

    def add_claim(self, content: str, confidence: float, is_correct: bool, domain: str = "math"):
        self.claims.append({
            "content": content,
            "confidence": confidence,
            "is_correct": is_correct,
            "domain": domain
        })

    def get_relevant(self, domain: str = "math", n: int = 3) -> List[Dict]:
        """Get top N relevant correct claims"""
        relevant = [c for c in self.claims if c["domain"] == domain and c["is_correct"]]
        return sorted(relevant, key=lambda x: x["confidence"], reverse=True)[:n]


class LLMProvider:
    """Abstract LLM provider interface"""

    async def generate(self, prompt: str, max_tokens: int = 256) -> str:
        raise NotImplementedError


class CerebrasProvider(LLMProvider):
    """Cerebras API provider"""

    def __init__(self, model: str = "llama3.1-8b"):
        self.model = model
        self.api_key = os.getenv("CEREBRAS_API_KEY")
        self.url = "https://api.cerebras.ai/v1/chat/completions"

    async def generate(self, prompt: str, max_tokens: int = 256) -> str:
        async with httpx.AsyncClient() as client:
            for attempt in range(3):
                try:
                    resp = await client.post(
                        self.url,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={
                            "model": self.model,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": max_tokens,
                            "temperature": 0.1
                        },
                        timeout=60.0
                    )
                    if resp.status_code == 429:
                        await asyncio.sleep(3 * (attempt + 1))
                        continue
                    if resp.status_code == 200:
                        return resp.json()["choices"][0]["message"]["content"]
                except Exception:
                    await asyncio.sleep(1)
        return ""


class ChutesProvider(LLMProvider):
    """Chutes API provider"""

    def __init__(self, model: str = "deepseek-ai/DeepSeek-V3"):
        self.model = model
        self.api_key = os.getenv("CHUTES_API_KEY")
        self.url = "https://llm.chutes.ai/v1/chat/completions"

    async def generate(self, prompt: str, max_tokens: int = 256) -> str:
        async with httpx.AsyncClient() as client:
            for attempt in range(3):
                try:
                    resp = await client.post(
                        self.url,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={
                            "model": self.model,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": max_tokens,
                            "temperature": 0.1
                        },
                        timeout=120.0
                    )
                    if resp.status_code == 429:
                        await asyncio.sleep(5 * (attempt + 1))
                        continue
                    if resp.status_code == 200:
                        return resp.json()["choices"][0]["message"]["content"]
                except Exception:
                    await asyncio.sleep(2)
        return ""


def get_provider(provider_name: str, model: str) -> LLMProvider:
    """Factory for LLM providers"""
    providers = {
        "cerebras": CerebrasProvider,
        "chutes": ChutesProvider,
    }
    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}")
    return providers[provider_name](model)


@register_model("conjecture", "conjecture_lm")
class ConjectureLM(LM):
    """
    Conjecture-enhanced Language Model for lm-evaluation-harness.

    Implements claim-based reasoning enhancement:
    1. Decompose problem into claims
    2. Evaluate claims with supporting evidence
    3. Synthesize final answer from validated claims
    """

    def __init__(
        self,
        provider: str = "cerebras",
        model: str = "llama3.1-8b",
        use_conjecture: bool = True,
        accumulate: bool = False,
        batch_size: int = 1,
        **kwargs
    ):
        super().__init__()
        self.provider_name = provider
        self.model_name = model
        self.provider = get_provider(provider, model)
        self.use_conjecture = use_conjecture
        self.accumulate = accumulate
        self._batch_size = batch_size
        self.claim_context = ClaimContext() if accumulate else None

    @property
    def eot_token_id(self):
        return None

    @property
    def max_length(self):
        return 4096

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return "cpu"

    def tok_encode(self, string: str, **kwargs) -> List[int]:
        """Approximate tokenization for length estimation"""
        return list(range(len(string.split())))

    def tok_decode(self, tokens: List[int], **kwargs) -> str:
        return ""

    def _generate_direct(self, prompt: str) -> str:
        """Direct generation without Conjecture enhancement"""
        return asyncio.get_event_loop().run_until_complete(
            self.provider.generate(prompt, max_tokens=256)
        )

    def _generate_conjecture(self, prompt: str) -> str:
        """Generate with Conjecture claim-based reasoning"""

        async def _conjecture_pipeline():
            # Step 1: Decompose - identify key claims needed
            decompose_prompt = f"""Analyze this problem and identify the key facts and steps needed to solve it.
Be specific about what information is given and what needs to be calculated.

Problem: {prompt}

Key facts and steps:"""

            decomposition = await self.provider.generate(decompose_prompt, max_tokens=200)

            # Step 2: Get accumulated hints if enabled
            hints = ""
            if self.accumulate and self.claim_context:
                relevant = self.claim_context.get_relevant("math", n=3)
                if relevant:
                    hints = "Patterns from similar problems:\n"
                    for c in relevant:
                        hints += f"- {c['content'][:100]}\n"
                    hints += "\n"

            # Step 3: Solve with decomposition context
            solve_prompt = f"""{hints}Analysis: {decomposition[:300]}

Problem: {prompt}

Solve step by step, then give the final answer after ####."""

            solution = await self.provider.generate(solve_prompt, max_tokens=300)

            return solution

        return asyncio.get_event_loop().run_until_complete(_conjecture_pipeline())

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False) -> List[str]:
        """Generate completions for a batch of requests"""
        results = []

        for request in requests:
            context = request.args[0]
            gen_kwargs = request.args[1] if len(request.args) > 1 else {}

            if self.use_conjecture:
                response = self._generate_conjecture(context)
            else:
                response = self._generate_direct(context)

            results.append(response)

        return results

    def loglikelihood(self, requests: List[Instance], disable_tqdm: bool = False) -> List[Tuple[float, bool]]:
        """
        Compute log-likelihood for completion tasks (e.g., MMLU multiple choice).

        For API-based models, we approximate by checking if the model generates
        the expected completion.
        """
        results = []

        for request in requests:
            context, continuation = request.args

            # Generate response and check if it matches
            prompt = f"{context}\nAnswer:"
            response = self._generate_direct(prompt)

            # Simple match check - in practice you'd want proper scoring
            continuation_clean = continuation.strip().lower()
            response_clean = response.strip().lower()[:len(continuation_clean)]

            is_match = continuation_clean in response_clean
            # Approximate log probability
            log_prob = 0.0 if is_match else -10.0

            results.append((log_prob, is_match))

        return results

    def loglikelihood_rolling(self, requests: List[Instance], disable_tqdm: bool = False) -> List[Tuple[float]]:
        """Compute rolling log-likelihood (perplexity)"""
        # Not implemented for API models
        return [(-1.0,) for _ in requests]


# Direct model registration for non-Conjecture baseline
@register_model("direct", "direct_lm")
class DirectLM(ConjectureLM):
    """Direct LLM without Conjecture enhancement (baseline)"""

    def __init__(self, **kwargs):
        kwargs["use_conjecture"] = False
        super().__init__(**kwargs)


if __name__ == "__main__":
    # Quick test
    import asyncio

    async def test():
        provider = CerebrasProvider("llama3.1-8b")
        response = await provider.generate("What is 2+2?", max_tokens=50)
        print(f"Test response: {response}")

    asyncio.run(test())
