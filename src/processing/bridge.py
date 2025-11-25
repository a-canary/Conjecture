"""
Modular LLM Bridge Interface
Provides clean abstraction between Conjecture API and LLM providers
Follows single responsibility principle with minimal complexity
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from ..core.models import Claim


@dataclass
class LLMRequest:
    """Standardized LLM request structure"""

    prompt: str
    context_claims: Optional[List[Claim]] = None
    max_tokens: int = 2048
    temperature: float = 0.7
    task_type: str = "general"  # explore, validate, analyze


@dataclass
class LLMResponse:
    """Standardized LLM response structure"""

    success: bool
    content: str
    generated_claims: List[Claim]
    metadata: Dict[str, Any]
    errors: List[str]
    processing_time: float
    tokens_used: int


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._initialize()

    @abstractmethod
    def _initialize(self):
        """Initialize provider-specific resources"""
        pass

    @abstractmethod
    def process_request(self, request: LLMRequest) -> LLMResponse:
        """Process standardized request and return standardized response"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available and configured"""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get provider-specific statistics"""
        return {"provider": self.__class__.__name__, "available": self.is_available()}


class LLMBridge:
    """
    Simple bridge between Conjecture API and LLM providers
    Provides clean interface with no over-engineering
    """

    def __init__(self, provider: Optional[LLMProvider] = None):
        self.provider = provider
        self.fallback_provider = None

    def process(self, request: LLMRequest) -> LLMResponse:
        """Process request using available provider"""
        if self.provider and hasattr(self.provider, "process_request"):
            # Handle simple provider interface
            result = self.provider.process_request(
                request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            return LLMResponse(
                success=result["success"],
                content=result["content"],
                generated_claims=[],
                metadata={},
                errors=[result.get("error", "")] if not result["success"] else [],
                processing_time=0.0,
                tokens_used=0,
            )
        elif self.provider:
            # Handle original LLMProvider interface
            return self.provider.process_request(request)
        else:
            return LLMResponse(
                success=False,
                content="",
                generated_claims=[],
                metadata={},
                errors=["No LLM provider available"],
                processing_time=0.0,
                tokens_used=0,
            )

    def set_provider(self, provider: LLMProvider):
        """Set primary LLM provider"""
        self.provider = provider

    def set_fallback(self, provider: LLMProvider):
        """Set fallback provider for resilience"""
        self.fallback_provider = provider

    def is_available(self) -> bool:
        """Check if any provider is available"""
        if self.provider and hasattr(self.provider, "is_available"):
            return self.provider.is_available()
        return self.provider is not None

    def process(self, request: LLMRequest) -> LLMResponse:
        """
        Process request using available provider
        Simple fallback logic with minimal complexity
        """
        if not self.provider or not self.provider.is_available():
            if self.fallback_provider and self.fallback_provider.is_available():
                return self.fallback_provider.process_request(request)
            else:
                return LLMResponse(
                    success=False,
                    content="",
                    generated_claims=[],
                    metadata={},
                    errors=["No LLM provider available"],
                    processing_time=0.0,
                    tokens_used=0,
                )

        try:
            return self.provider.process_request(request)
        except Exception as e:
            # Try fallback on error
            if self.fallback_provider and self.fallback_provider.is_available():
                return self.fallback_provider.process_request(request)
            else:
                return LLMResponse(
                    success=False,
                    content="",
                    generated_claims=[],
                    metadata={},
                    errors=[f"LLM processing failed: {e}"],
                    processing_time=0.0,
                    tokens_used=0,
                )

    def is_available(self) -> bool:
        """Check if any provider is available"""
        return (self.provider and self.provider.is_available()) or (
            self.fallback_provider and self.fallback_provider.is_available()
        )

    def get_status(self) -> Dict[str, Any]:
        """Get bridge status and provider information"""
        return {
            "primary_available": self.provider.is_available()
            if self.provider
            else False,
            "fallback_available": self.fallback_provider.is_available()
            if self.fallback_provider
            else False,
            "primary_stats": self.provider.get_stats() if self.provider else {},
            "fallback_stats": self.fallback_provider.get_stats()
            if self.fallback_provider
            else {},
        }
