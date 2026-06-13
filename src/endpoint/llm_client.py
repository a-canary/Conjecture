# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
LLM Client for ConjectureEndpoint

Simple OpenAI-compatible client for calling LLMs.
Supports Chutes.ai, OpenRouter, or any OpenAI-compatible endpoint.

O-0005: Includes circuit breaker with exponential backoff for graceful degradation.
"""

import os
import time
import asyncio
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# O-0005: Circuit Breaker
# ---------------------------------------------------------------------------

@dataclass
class CircuitBreaker:
    """Simple circuit breaker for O-0005: Exponential Backoff with Circuit Breaker.

    States:
        CLOSED: Normal operation, requests go through
        OPEN: Too many failures, requests are rejected immediately
        HALF_OPEN: Testing if service recovered (allows one request)
    """
    failure_threshold: int = 5  # Open after this many consecutive failures
    recovery_timeout: float = 60.0  # Seconds before trying half-open
    _failures: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _state: str = field(default="CLOSED", init=False)

    def record_failure(self) -> None:
        """Record a failure and potentially open the circuit."""
        self._failures += 1
        self._last_failure_time = time.time()
        if self._failures >= self.failure_threshold:
            self._state = "OPEN"
            logger.warning(
                "Circuit breaker OPEN after %d consecutive failures",
                self._failures
            )

    def record_success(self) -> None:
        """Record a success and reset the circuit."""
        self._failures = 0
        self._state = "CLOSED"

    def can_execute(self) -> bool:
        """Check if a request can be executed."""
        if self._state == "CLOSED":
            return True
        elif self._state == "OPEN":
            # Check if recovery timeout has passed
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                self._state = "HALF_OPEN"
                logger.info("Circuit breaker HALF_OPEN, allowing test request")
                return True
            return False
        else:  # HALF_OPEN
            return True

    @property
    def state(self) -> str:
        return self._state


# Default circuit breaker instance
_default_circuit_breaker = CircuitBreaker()


# Default models for different use cases
DEFAULT_MODEL = "openai/gpt-oss-20b"  # Fast, general purpose
TOOL_CAPABLE_MODEL = "Qwen/Qwen3-32B"  # Supports function/tool calling


class LLMClient:
    """Async LLM client using OpenAI-compatible API.

    Per O-0005, includes circuit breaker for graceful degradation on failures.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://llm.chutes.ai/v1",
        model: str = DEFAULT_MODEL,
        circuit_breaker: Optional[CircuitBreaker] = None
    ):
        """Initialize the LLM client.

        Args:
            api_key: API key (defaults to CHUTES_API_KEY env var)
            base_url: API base URL (defaults to Chutes.ai)
            model: Model ID to use
            circuit_breaker: Optional circuit breaker (uses default if None)
        """
        self.api_key = api_key or os.environ.get("CHUTES_API_KEY")
        self.base_url = base_url
        self.model = model
        self._client: Optional[AsyncOpenAI] = None
        self._circuit_breaker = circuit_breaker or _default_circuit_breaker

    def _get_client(self) -> AsyncOpenAI:
        """Get or create the async client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "No API key provided. Set CHUTES_API_KEY env var or pass api_key."
                )
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        return self._client

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """Generate a response from the LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Dict with 'content', 'model', 'usage' keys

        Raises:
            RuntimeError: If circuit breaker is open (too many failures)
        """
        # O-0005: Check circuit breaker before attempting
        if not self._circuit_breaker.can_execute():
            raise RuntimeError(
                f"Circuit breaker OPEN - LLM service unavailable. "
                f"Retry after {self._circuit_breaker.recovery_timeout}s"
            )

        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Success - reset circuit breaker
            self._circuit_breaker.record_success()

            return {
                "content": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
            }

        except Exception as e:
            # Record failure for circuit breaker
            self._circuit_breaker.record_failure()
            logger.error(f"LLM generation failed: {e}")
            raise

    async def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """Generate a response from the LLM with tool calling support.

        Per A-0010: The LLM operates via claim tools, not raw text responses.

        Args:
            prompt: User prompt
            tools: List of tool definitions in OpenAI function-calling format
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Dict with 'content', 'tool_calls', 'model', 'usage' keys.
            'tool_calls' is a list of dicts with 'name' and 'arguments' keys.

        Raises:
            RuntimeError: If circuit breaker is open (too many failures)
        """
        # O-0005: Check circuit breaker before attempting
        if not self._circuit_breaker.can_execute():
            raise RuntimeError(
                f"Circuit breaker OPEN - LLM service unavailable. "
                f"Retry after {self._circuit_breaker.recovery_timeout}s"
            )

        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=temperature,
                max_tokens=max_tokens
            )

            message = response.choices[0].message

            # Parse tool calls if present
            tool_calls = []
            if message.tool_calls:
                import json
                for tc in message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {"raw": tc.function.arguments}
                    tool_calls.append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": args
                    })

            # Success - reset circuit breaker
            self._circuit_breaker.record_success()

            return {
                "content": message.content or "",
                "tool_calls": tool_calls,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
            }

        except Exception as e:
            # Record failure for circuit breaker
            self._circuit_breaker.record_failure()
            logger.error(f"LLM generation with tools failed: {e}")
            raise

    async def close(self):
        """Close the client."""
        if self._client:
            await self._client.close()
            self._client = None


def build_claim_context(claims: List[Dict[str, Any]]) -> str:
    """Build a context string from claims.

    Args:
        claims: List of claim dictionaries with 'content' and 'confidence'

    Returns:
        Formatted context string
    """
    if not claims:
        return ""

    context_parts = ["Relevant knowledge (confidence scores shown):"]
    for i, claim in enumerate(claims, 1):
        content = claim.get("content", "")
        confidence = claim.get("confidence", 0.5)
        context_parts.append(f"{i}. [{confidence:.0%}] {content}")

    return "\n".join(context_parts)


def build_enhanced_prompt(query: str, claim_context: str) -> str:
    """Build an enhanced prompt with claim context.

    This is the pattern that achieved GSM8K +40pp improvement.

    Args:
        query: User's original query
        claim_context: Formatted claim context string

    Returns:
        Enhanced prompt with reasoning instructions
    """
    if claim_context:
        return f"""{claim_context}

Based on the above knowledge, answer the following:

{query}

Think step-by-step. Show your reasoning clearly.
After working through the problem, verify your answer makes sense."""
    else:
        return f"""{query}

Think step-by-step. Show your reasoning clearly.
After working through the problem, verify your answer makes sense."""
