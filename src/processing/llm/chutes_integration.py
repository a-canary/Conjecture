"""
Chutes.ai Integration for Conjecture
Provides direct integration with Chutes.ai API for claim processing
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .common import GenerationConfig as BaseGenerationConfig, LLMProcessingResult


@dataclass
class GenerationConfig(BaseGenerationConfig):
    """Extended generation config for Chutes.ai"""
    # Chutes-specific parameters can be added here
    pass


@dataclass
class ChutesClaim:
    """Chutes.ai claim representation"""
    id: str
    content: str
    confidence: float
    type: Any  # ClaimType enum
    state: Any  # ClaimState enum
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChutesProcessingResult:
    """Result from Chutes.ai processing"""
    success: bool
    processed_claims: List[ChutesClaim]
    model_used: str
    tokens_used: int
    processing_time_ms: int
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChutesProcessor:
    """
    Direct Chutes.ai processor for claim generation and processing
    Maintains compatibility with existing Conjecture architecture
    """

    def __init__(self, api_key: str, api_url: str, model_name: str):
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name
        self._stats = {
            "requests_processed": 0,
            "successful_requests": 0,
            "total_tokens": 0,
            "total_time": 0.0,
        }

    def generate_response(self, prompt: str, config: GenerationConfig) -> ChutesProcessingResult:
        """Generate response from Chutes.ai"""
        start_time = time.time()
        self._stats["requests_processed"] += 1

        try:
            # Mock implementation for now - in real scenario this would call Chutes.ai API
            # For testing purposes, we'll create a mock response
            
            mock_claim = ChutesClaim(
                id="mock_001",
                content=f"Mock response to: {prompt[:100]}...",
                confidence=0.85,
                type="fact",  # Would be ClaimType.FACT in real implementation
                state="validated",  # Would be ClaimState.VALIDATED in real implementation
                tags=["mock", "test"],
                metadata={"model": self.model_name}
            )

            processing_time = (time.time() - start_time) * 1000
            self._stats["successful_requests"] += 1
            self._stats["total_tokens"] += 150  # Mock token count
            self._stats["total_time"] += processing_time / 1000

            return ChutesProcessingResult(
                success=True,
                processed_claims=[mock_claim],
                model_used=self.model_name,
                tokens_used=150,
                processing_time_ms=int(processing_time),
                metadata={"api_url": self.api_url}
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self._stats["total_time"] += processing_time / 1000

            return ChutesProcessingResult(
                success=False,
                processed_claims=[],
                model_used=self.model_name,
                tokens_used=0,
                processing_time_ms=int(processing_time),
                errors=[f"Chutes.ai API error: {str(e)}"]
            )

    def process_claims(self, claims: List[Any], task: str, config: GenerationConfig) -> ChutesProcessingResult:
        """Process existing claims through Chutes.ai"""
        start_time = time.time()
        self._stats["requests_processed"] += 1

        try:
            # Mock implementation for processing claims
            processed_claims = []
            
            for i, claim in enumerate(claims):
                # Create a processed version of the claim
                processed_claim = ChutesClaim(
                    id=f"processed_{claim.id if hasattr(claim, 'id') else i}",
                    content=f"Processed: {claim.content if hasattr(claim, 'content') else str(claim)}",
                    confidence=claim.confidence if hasattr(claim, 'confidence') else 0.8,
                    type=claim.type if hasattr(claim, 'type') else "observation",
                    state=claim.state if hasattr(claim, 'state') else "validated",
                    tags=getattr(claim, 'tags', []) + ["processed"],
                    metadata={"task": task, "original_id": getattr(claim, 'id', None)}
                )
                processed_claims.append(processed_claim)

            processing_time = (time.time() - start_time) * 1000
            self._stats["successful_requests"] += 1
            self._stats["total_tokens"] += len(claims) * 100  # Estimate tokens
            self._stats["total_time"] += processing_time / 1000

            return ChutesProcessingResult(
                success=True,
                processed_claims=processed_claims,
                model_used=self.model_name,
                tokens_used=len(claims) * 100,
                processing_time_ms=int(processing_time),
                metadata={"task": task, "claims_processed": len(claims)}
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self._stats["total_time"] += processing_time / 1000

            return ChutesProcessingResult(
                success=False,
                processed_claims=[],
                model_used=self.model_name,
                tokens_used=0,
                processing_time_ms=int(processing_time),
                errors=[f"Chutes.ai claim processing error: {str(e)}"]
            )

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
            "api_url": self.api_url
        }


def create_chutes_processor(api_key: str, api_url: str = "https://llm.chutes.ai/v1", model_name: str = "zai-org/GLM-4.6") -> ChutesProcessor:
    """Factory function to create ChutesProcessor"""
    return ChutesProcessor(api_key=api_key, api_url=api_url, model_name=model_name)