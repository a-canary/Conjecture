"""
Local Providers Adapter for Conjecture
Provides unified interface for local LLM providers (Ollama, LM Studio, etc.)
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .common import GenerationConfig as BaseGenerationConfig, LLMProcessingResult


@dataclass
class GenerationConfig(BaseGenerationConfig):
    """Extended generation config for local providers"""
    # Local provider specific parameters can be added here
    pass


@dataclass
class LocalClaim:
    """Local provider claim representation"""
    id: str
    content: str
    confidence: float
    type: Any  # ClaimType enum
    state: Any  # ClaimState enum
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalProcessingResult:
    """Result from local provider processing"""
    success: bool
    processed_claims: List[LocalClaim]
    model_used: str
    tokens_used: int
    processing_time_ms: int
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class LocalProviderProcessor:
    """
    Unified processor for local LLM providers
    Supports Ollama, LM Studio, and other OpenAI-compatible local providers
    """

    def __init__(self, provider_type: str, base_url: str, model_name: str):
        self.provider_type = provider_type.lower()
        self.base_url = base_url
        self.model_name = model_name
        self._stats = {
            "requests_processed": 0,
            "successful_requests": 0,
            "total_tokens": 0,
            "total_time": 0.0,
        }

    def generate_response(self, prompt: str, config: GenerationConfig) -> LocalProcessingResult:
        """Generate response from local provider"""
        start_time = time.time()
        self._stats["requests_processed"] += 1

        try:
            # Mock implementation for now - in real scenario this would call local provider API
            # For testing purposes, we'll create a mock response
            
            mock_claim = LocalClaim(
                id=f"local_{self.provider_type}_001",
                content=f"Local {self.provider_type} response to: {prompt[:100]}...",
                confidence=0.75,
                type="fact",  # Would be ClaimType.FACT in real implementation
                state="validated",  # Would be ClaimState.VALIDATED in real implementation
                tags=["local", self.provider_type, "mock"],
                metadata={
                    "model": self.model_name,
                    "provider": self.provider_type,
                    "base_url": self.base_url
                }
            )

            processing_time = (time.time() - start_time) * 1000
            self._stats["successful_requests"] += 1
            self._stats["total_tokens"] += 120  # Mock token count
            self._stats["total_time"] += processing_time / 1000

            return LocalProcessingResult(
                success=True,
                processed_claims=[mock_claim],
                model_used=self.model_name,
                tokens_used=120,
                processing_time_ms=int(processing_time),
                metadata={
                    "provider": self.provider_type,
                    "base_url": self.base_url
                }
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self._stats["total_time"] += processing_time / 1000

            return LocalProcessingResult(
                success=False,
                processed_claims=[],
                model_used=self.model_name,
                tokens_used=0,
                processing_time_ms=int(processing_time),
                errors=[f"Local {self.provider_type} API error: {str(e)}"]
            )

    def process_claims(self, claims: List[Any], task: str, config: GenerationConfig) -> LocalProcessingResult:
        """Process existing claims through local provider"""
        start_time = time.time()
        self._stats["requests_processed"] += 1

        try:
            # Mock implementation for processing claims
            processed_claims = []
            
            for i, claim in enumerate(claims):
                # Create a processed version of the claim
                processed_claim = LocalClaim(
                    id=f"local_{self.provider_type}_processed_{claim.id if hasattr(claim, 'id') else i}",
                    content=f"Local {self.provider_type} processed: {claim.content if hasattr(claim, 'content') else str(claim)}",
                    confidence=claim.confidence if hasattr(claim, 'confidence') else 0.7,
                    type=claim.type if hasattr(claim, 'type') else "fact",
                    state=claim.state if hasattr(claim, 'state') else "validated",
                    tags=getattr(claim, 'tags', []) + ["processed", f"local_{self.provider_type}"],
                    metadata={
                        "task": task,
                        "original_id": getattr(claim, 'id', None),
                        "provider": self.provider_type,
                        "model": self.model_name
                    }
                )
                processed_claims.append(processed_claim)

            processing_time = (time.time() - start_time) * 1000
            self._stats["successful_requests"] += 1
            self._stats["total_tokens"] += len(claims) * 80  # Estimate tokens
            self._stats["total_time"] += processing_time / 1000

            return LocalProcessingResult(
                success=True,
                processed_claims=processed_claims,
                model_used=self.model_name,
                tokens_used=len(claims) * 80,
                processing_time_ms=int(processing_time),
                metadata={
                    "task": task,
                    "claims_processed": len(claims),
                    "provider": self.provider_type
                }
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self._stats["total_time"] += processing_time / 1000

            return LocalProcessingResult(
                success=False,
                processed_claims=[],
                model_used=self.model_name,
                tokens_used=0,
                processing_time_ms=int(processing_time),
                errors=[f"Local {self.provider_type} claim processing error: {str(e)}"]
            )

    def health_check(self) -> Dict[str, Any]:
        """Check health of the local provider"""
        try:
            # Mock health check - in real implementation would ping the provider
            return {
                "status": "healthy",
                "last_check": time.time(),
                "model": self.model_name,
                "provider": self.provider_type,
                "base_url": self.base_url,
                "response_time_ms": 50  # Mock response time
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "last_check": time.time(),
                "model": self.model_name,
                "provider": self.provider_type,
                "base_url": self.base_url,
                "error": str(e)
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics"""
        if self._stats["requests_processed"] > 0:
            success_rate = self._stats["successful_requests"] / self._stats["requests_processed"]
            avg_time = self._stats["total_time"] / self._stats["requests_processed"]
            avg_tokens = self._stats["total_tokens"] / self._stats["successful_requests"] if self._stats["successful_requests"] > 0 else 0
        else:
            success_rate = 0.0
            avg_time = 0.0
            avg_tokens = 0.0

        return {
            "requests_processed": self._stats["requests_processed"],
            "successful_requests": self._stats["successful_requests"],
            "success_rate": success_rate,
            "total_tokens": self._stats["total_tokens"],
            "average_tokens_per_request": avg_tokens,
            "average_processing_time": avg_time,
            "total_processing_time": self._stats["total_time"],
            "model": self.model_name,
            "provider": self.provider_type,
            "base_url": self.base_url
        }


def create_local_provider_processor(provider_type: str, base_url: str, model_name: str) -> LocalProviderProcessor:
    """Factory function to create LocalProviderProcessor"""
    return LocalProviderProcessor(provider_type=provider_type, base_url=base_url, model_name=model_name)