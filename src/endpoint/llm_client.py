"""
LLM Client for ConjectureEndpoint

Simple OpenAI-compatible client for calling LLMs.
Supports Chutes.ai, OpenRouter, or any OpenAI-compatible endpoint.
"""

import os
import logging
from typing import Optional, List, Dict, Any

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    """Async LLM client using OpenAI-compatible API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://llm.chutes.ai/v1",
        model: str = "openai/gpt-oss-20b"
    ):
        """Initialize the LLM client.

        Args:
            api_key: API key (defaults to CHUTES_API_KEY env var)
            base_url: API base URL (defaults to Chutes.ai)
            model: Model ID to use
        """
        self.api_key = api_key or os.environ.get("CHUTES_API_KEY")
        self.base_url = base_url
        self.model = model
        self._client: Optional[AsyncOpenAI] = None

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
        """
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
        """
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
