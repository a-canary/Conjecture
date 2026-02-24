"""
Anthropic Integration - Claude Agent SDK Provider
Primary LLM provider using Anthropic's Claude models (T-0008)
Default model: claude-3-5-haiku-latest (cost-effective)
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

from .common import GenerationConfig, LLMProcessingResult

logger = logging.getLogger(__name__)

# Default model per CHOICES.md T-0008
DEFAULT_MODEL = "claude-3-5-haiku-latest"  # Haiku 4.5 equivalent

@dataclass
class AnthropicConfig:
    """Configuration for Anthropic provider"""
    api_key: str = ""  # Empty string means use environment variable
    model: str = DEFAULT_MODEL
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: float = 60.0

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "AnthropicConfig":
        return cls(
            api_key=config.get("api", config.get("api_key", "")),
            model=config.get("model", DEFAULT_MODEL),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.7),
            timeout=config.get("timeout", 60.0),
        )


class AnthropicProcessor:
    """
    Real Anthropic processor using Claude Agent SDK

    Per T-0008: Claude Agent SDK handles authentication and secrets.
    When running in Claude Code environment, API key is provided by runtime.
    """

    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], AnthropicConfig]] = None,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ):
        """
        Initialize Anthropic processor

        Args:
            config: Configuration dict or AnthropicConfig object
            api_key: Optional API key (overrides config)
            model: Model name (default: claude-3-5-haiku-latest)
        """
        if isinstance(config, dict):
            self.config = AnthropicConfig.from_dict(config)
        elif isinstance(config, AnthropicConfig):
            self.config = config
        else:
            self.config = AnthropicConfig(model=model)

        # Override with explicit api_key if provided
        if api_key:
            self.config.api_key = api_key

        self.name = "anthropic"
        self.model = self.config.model
        self._client = None

        # Stats tracking
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_processing_time": 0.0,
        }

        logger.info(f"AnthropicProcessor initialized with model: {self.model}")

    def _get_client(self):
        """Lazy-initialize the Anthropic client"""
        if self._client is None:
            try:
                import anthropic

                # Get API key from config, environment, or let SDK handle it
                api_key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")

                if api_key:
                    self._client = anthropic.Anthropic(api_key=api_key)
                else:
                    # Let SDK handle auth (e.g., when running in Claude Code)
                    self._client = anthropic.Anthropic()

            except ImportError:
                logger.error("anthropic package not installed. Run: pip install anthropic")
                raise ImportError("anthropic package required. Install with: pip install anthropic")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
                raise

        return self._client

    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate response using Anthropic Claude

        Args:
            prompt: User message
            system_prompt: Optional system message
            max_tokens: Override max tokens
            temperature: Override temperature

        Returns:
            Generated response text
        """
        start_time = time.time()
        self.stats["total_requests"] += 1

        try:
            client = self._get_client()

            messages = [{"role": "user", "content": prompt}]

            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens or self.config.max_tokens,
                temperature=temperature if temperature is not None else self.config.temperature,
                system=system_prompt or "",
                messages=messages,
            )

            # Extract response text
            response_text = ""
            if response.content:
                for block in response.content:
                    if hasattr(block, 'text'):
                        response_text += block.text

            # Update stats
            self.stats["successful_requests"] += 1
            if hasattr(response, 'usage'):
                self.stats["total_tokens"] += response.usage.input_tokens + response.usage.output_tokens

            processing_time = time.time() - start_time
            self.stats["total_processing_time"] += processing_time

            logger.debug(f"Anthropic response generated in {processing_time:.2f}s")
            return response_text

        except Exception as e:
            self.stats["failed_requests"] += 1
            logger.error(f"Anthropic generation failed: {e}")
            raise

    async def generate_response_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Async version of generate_response

        Uses the synchronous client in an async context.
        For true async, consider using anthropic.AsyncAnthropic.
        """
        # For now, use sync client (can upgrade to AsyncAnthropic later)
        return self.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

    def process_with_claims(
        self,
        prompt: str,
        claims_context: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMProcessingResult:
        """
        Process prompt with claim context

        Args:
            prompt: User request
            claims_context: Formatted claims for context
            system_prompt: Optional system instructions

        Returns:
            LLMProcessingResult with response and metadata
        """
        start_time = time.time()

        # Build full prompt with claims context
        full_prompt = f"""## Relevant Claims
{claims_context}

## User Request
{prompt}

## Instructions
Based on the claims above, respond to the user's request. Reference specific claims when relevant.
If you identify new claims or need to update confidence in existing claims, note them clearly."""

        try:
            response_text = self.generate_response(
                prompt=full_prompt,
                system_prompt=system_prompt,
                **kwargs
            )

            processing_time = time.time() - start_time

            return LLMProcessingResult(
                success=True,
                response_text=response_text,
                processing_time=processing_time,
                model_name=self.model,
                provider_name=self.name,
                metadata={
                    "claims_in_context": claims_context.count("claim_") if claims_context else 0,
                }
            )

        except Exception as e:
            processing_time = time.time() - start_time
            return LLMProcessingResult(
                success=False,
                error_message=str(e),
                processing_time=processing_time,
                model_name=self.model,
                provider_name=self.name,
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics"""
        return {
            **self.stats,
            "model": self.model,
            "provider": self.name,
            "avg_processing_time": (
                self.stats["total_processing_time"] / self.stats["total_requests"]
                if self.stats["total_requests"] > 0 else 0
            ),
            "success_rate": (
                self.stats["successful_requests"] / self.stats["total_requests"] * 100
                if self.stats["total_requests"] > 0 else 0
            ),
        }

    def health_check(self) -> bool:
        """Check if provider is healthy and can generate responses"""
        try:
            response = self.generate_response(
                prompt="Hello",
                max_tokens=10,
            )
            return bool(response)
        except Exception as e:
            logger.warning(f"Anthropic health check failed: {e}")
            return False


def create_anthropic_processor(
    config: Optional[Dict[str, Any]] = None,
    model: str = DEFAULT_MODEL,
) -> AnthropicProcessor:
    """
    Factory function to create Anthropic processor

    Args:
        config: Optional configuration dict
        model: Model name (default: claude-3-5-haiku-latest)

    Returns:
        Configured AnthropicProcessor
    """
    if config:
        return AnthropicProcessor(config=config)
    return AnthropicProcessor(model=model)


# Export for convenience
__all__ = [
    "AnthropicProcessor",
    "AnthropicConfig",
    "create_anthropic_processor",
    "DEFAULT_MODEL",
]
