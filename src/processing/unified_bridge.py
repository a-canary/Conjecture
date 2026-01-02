"""
=============================================================================
GENERATED CODE - SC-FEAT-001 - TEST BRANCH
=============================================================================
Unified LLM Bridge for Conjecture
Provides a unified interface for LLM interactions across all layers
Wraps SimplifiedLLMManager for backward compatibility

Modified 2025-12-30:
  - Fixed GenerationConfig usage for OpenAICompatibleProcessor
  - Added proper config parameter handling
=============================================================================
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time

# SC-FEAT-001: Import GenerationConfig for proper LLM parameter handling
from src.processing.llm.common import GenerationConfig


@dataclass
class LLMRequest:
    """Request object for LLM generation"""

    prompt: str
    max_tokens: int = 2000
    temperature: float = 0.0
    task_type: str = "generation"
    provider: Optional[str] = None
    model: Optional[str] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Response object from LLM generation"""

    content: str
    success: bool = True
    errors: List[str] = field(default_factory=list)
    model: Optional[str] = None
    provider: Optional[str] = None
    tokens_used: int = 0
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedLLMBridge:
    """
    Unified LLM bridge providing consistent interface across all layers
    Wraps SimplifiedLLMManager for backward compatibility with benchmarks
    """

    def __init__(self, llm_manager=None):
        """
        Initialize LLM bridge

        Args:
            llm_manager: SimplifiedLLMManager instance. If None, creates one.
        """
        if llm_manager is None:
            from .simplified_llm_manager import get_simplified_llm_manager

            self.llm_manager = get_simplified_llm_manager()
        else:
            self.llm_manager = llm_manager

    def is_available(self) -> bool:
        """Check if any LLM providers are available"""
        return (
            self.llm_manager is not None
            and len(self.llm_manager.get_available_providers()) > 0
        )

    def process(self, request: LLMRequest) -> LLMResponse:
        """
        Process an LLM request

        SC-FEAT-001: Fixed to use GenerationConfig object instead of kwargs

        Args:
            request: LLMRequest object containing prompt and parameters

        Returns:
            LLMResponse object with generation results
        """
        start_time = time.time()

        try:
            # SC-FEAT-001: Build GenerationConfig object (required by OpenAICompatibleProcessor)
            config = GenerationConfig(
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=0.9,  # Default value
            )

            # Generate response with GenerationConfig
            response = self.llm_manager.generate_response(
                prompt=request.prompt,
                provider=request.provider,
                config=config,  # SC-FEAT-001: Pass config object, not kwargs
            )

            # Extract content from LLMProcessingResult
            response_content = ""
            if hasattr(response, "content") and response.content:
                # CRITICAL FIX: Check content is not just truthy, but actually has content
                content = response.content
                if isinstance(content, str) and content.strip():
                    response_content = content
            elif hasattr(response, "processed_claims") and response.processed_claims:
                response_content = str(response.processed_claims)
            else:
                response_content = str(response)

            # Extract model and provider info if available
            processor = self.llm_manager.get_processor(request.provider)
            provider_name = request.provider
            if not provider_name and processor:
                provider_name = self.llm_manager._get_provider_name(processor)

            model_name = None
            if processor:
                model_name = getattr(processor, "model_name", None)
            if not model_name and request.model:
                model_name = request.model

            tokens_used = 0
            if hasattr(response, "tokens_used"):
                tokens_used = response.tokens_used

            processing_time = time.time() - start_time

            return LLMResponse(
                content=response_content,
                success=getattr(response, "success", True),
                errors=getattr(response, "errors", []),
                model=model_name,
                provider=provider_name,
                tokens_used=tokens_used,
                processing_time=processing_time,
                metadata={"task_type": request.task_type},
            )

        except Exception as e:
            processing_time = time.time() - start_time

            # Get provider info even on error
            provider_name = request.provider
            try:
                processor = self.llm_manager.get_processor(request.provider)
                if processor:
                    provider_name = self.llm_manager._get_provider_name(processor)
            except:
                pass

            return LLMResponse(
                content="",
                success=False,
                errors=[str(e)],
                provider=provider_name,
                processing_time=processing_time,
                metadata={
                    "task_type": request.task_type,
                    "error_type": type(e).__name__,
                },
            )

    async def process_async(self, request: LLMRequest) -> LLMResponse:
        """
        Async version of process method

        Args:
            request: LLMRequest object containing prompt and parameters

        Returns:
            LLMResponse object with generation results
        """
        # For now, process synchronously
        # TODO: Add true async support when underlying processors support it
        return self.process(request)

    def batch_process(self, requests: List[LLMRequest]) -> List[LLMResponse]:
        """
        Process multiple LLM requests

        Args:
            requests: List of LLMRequest objects

        Returns:
            List of LLMResponse objects
        """
        return [self.process(req) for req in requests]

    def get_available_providers(self) -> List[str]:
        """Get list of available provider names"""
        return self.llm_manager.get_available_providers()

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on LLM bridge"""
        try:
            health = self.llm_manager.health_check()
            return {
                "status": health.get("overall_status", "unknown"),
                "available_providers": self.get_available_providers(),
                "primary_provider": self.llm_manager.primary_provider,
                "bridge_available": self.is_available(),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "bridge_available": False,
                "error": str(e),
                "available_providers": [],
                "primary_provider": None,
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from LLM bridge"""
        try:
            return self.llm_manager.get_combined_stats()
        except Exception as e:
            return {
                "error": str(e),
                "total_providers": 0,
                "available_providers": 0,
            }


# Global instance for easy access
_unified_llm_bridge = None


def get_unified_bridge(llm_manager=None) -> UnifiedLLMBridge:
    """
    Get global unified LLM bridge instance

    Args:
        llm_manager: Optional LLM manager to use

    Returns:
        UnifiedLLMBridge instance
    """
    global _unified_llm_bridge
    if _unified_llm_bridge is None:
        _unified_llm_bridge = UnifiedLLMBridge(llm_manager=llm_manager)
    return _unified_llm_bridge


def reset_unified_bridge():
    """Reset global unified LLM bridge instance"""
    global _unified_llm_bridge
    _unified_llm_bridge = None


# ============================================================================
# END OF GENERATED CODE - SC-FEAT-001 - TEST BRANCH
# ============================================================================
