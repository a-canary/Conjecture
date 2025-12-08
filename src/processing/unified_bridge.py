"""
Unified LLM Bridge Interface for Conjecture
Provides clean abstraction between Conjecture API and unified LLM manager
Follows single responsibility principle with minimal complexity
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import time
import logging

try:
    from ..core.models import Claim
    from .unified_llm_manager import UnifiedLLMManager, get_unified_llm_manager
    from ..utils.retry_utils import with_llm_retry, EnhancedRetryConfig
except ImportError:
    # Handle relative import issues for test compatibility
    import sys
    import os
    # Add src directory to path for absolute imports
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.core.models import Claim
    from src.processing.unified_llm_manager import UnifiedLLMManager, get_unified_llm_manager
    from src.utils.retry_utils import with_llm_retry, EnhancedRetryConfig

# Configure logging
logger = logging.getLogger(__name__)


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


class UnifiedLLMBridge:
    """
    Unified bridge between Conjecture API and LLM providers
    Provides clean interface with no over-engineering
    """

    def __init__(self, llm_manager: Optional[UnifiedLLMManager] = None, retry_config: Optional[EnhancedRetryConfig] = None):
        self.llm_manager = llm_manager or get_unified_llm_manager()
        self.retry_config = retry_config or EnhancedRetryConfig()

    @with_llm_retry(max_attempts=5, base_delay=10.0, max_delay=600.0)
    def process(self, request: LLMRequest) -> LLMResponse:
        """Process request using unified LLM manager with retry logic"""
        start_time = time.time()
        
        try:
            # Use unified LLM manager for processing
            result = self.llm_manager.generate_response(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                provider=None  # Let manager choose optimal provider
            )
            
            processing_time = time.time() - start_time
            logger.info(f"LLM request processed successfully in {processing_time:.2f}s using provider: {result.get('provider', 'unified')}")
            
            return LLMResponse(
                success=True,
                content=result.get("content", ""),
                generated_claims=[],
                metadata={
                    "provider": result.get("provider", "unified"),
                    "model": result.get("model", "unknown"),
                    "usage": result.get("usage", {}),
                    "task_type": request.task_type,
                    "retry_attempts": getattr(self, '_retry_attempts', 0)
                },
                errors=[],
                processing_time=processing_time,
                tokens_used=result.get("usage", {}).get("total_tokens", 0),
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            retry_attempts = getattr(self, '_retry_attempts', 0)
            logger.error(f"LLM request failed after {retry_attempts} attempts in {processing_time:.2f}s: {e}")
            
            return LLMResponse(
                success=False,
                content="",
                generated_claims=[],
                metadata={
                    "retry_attempts": retry_attempts,
                    "error_type": type(e).__name__
                },
                errors=[f"Unified LLM processing failed: {e}"],
                processing_time=processing_time,
                tokens_used=0,
            )

    @with_llm_retry(max_attempts=5, base_delay=10.0, max_delay=600.0)
    def process_claims(
        self,
        claims: List[Claim],
        task: str = "analyze",
        **kwargs
    ) -> LLMResponse:
        """Process claims using unified LLM manager with retry logic"""
        start_time = time.time()
        
        try:
            # Use unified LLM manager for claim processing
            result = self.llm_manager.process_claims(
                claims=claims,
                task=task,
                **kwargs
            )
            
            processing_time = time.time() - start_time
            logger.info(f"Claim processing completed successfully in {processing_time:.2f}s using provider: {result.get('provider', 'unified')}")
            
            return LLMResponse(
                success=True,
                content=result.get("content", ""),
                generated_claims=result.get("generated_claims", []),
                metadata={
                    "provider": result.get("provider", "unified"),
                    "model": result.get("model", "unknown"),
                    "usage": result.get("usage", {}),
                    "task": task,
                    "claims_processed": len(claims),
                    "retry_attempts": getattr(self, '_retry_attempts', 0)
                },
                errors=[],
                processing_time=processing_time,
                tokens_used=result.get("usage", {}).get("total_tokens", 0),
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            retry_attempts = getattr(self, '_retry_attempts', 0)
            logger.error(f"Claim processing failed after {retry_attempts} attempts in {processing_time:.2f}s: {e}")
            
            return LLMResponse(
                success=False,
                content="",
                generated_claims=[],
                metadata={
                    "retry_attempts": retry_attempts,
                    "error_type": type(e).__name__,
                    "claims_processed": len(claims)
                },
                errors=[f"Claim processing failed: {e}"],
                processing_time=processing_time,
                tokens_used=0,
            )

    def is_available(self) -> bool:
        """Check if unified LLM manager is available"""
        return self.llm_manager is not None and len(self.llm_manager.get_available_providers()) > 0

    def get_status(self) -> Dict[str, Any]:
        """Get bridge status and provider information"""
        if not self.llm_manager:
            return {
                "available": False,
                "error": "No LLM manager available",
                "bridge_type": "unified"
            }
        
        provider_info = self.llm_manager.get_provider_info()
        health_status = self.llm_manager.health_check()
        
        return {
            "available": self.is_available(),
            "bridge_type": "unified",
            "provider_info": provider_info,
            "health_status": health_status,
            "primary_provider": self.llm_manager.primary_provider
        }

    def switch_provider(self, provider_name: str) -> bool:
        """Switch to a specific provider"""
        if not self.llm_manager:
            return False
        
        return self.llm_manager.switch_provider(provider_name)

    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        if not self.llm_manager:
            return []
        
        return self.llm_manager.get_available_providers()

    def reset_failed_providers(self):
        """Reset failed providers list"""
        if self.llm_manager:
            self.llm_manager.failed_providers.clear()


# Global instance for easy access
_unified_bridge = None


def get_unified_bridge() -> UnifiedLLMBridge:
    """Get the global unified bridge instance"""
    global _unified_bridge
    if _unified_bridge is None:
        _unified_bridge = UnifiedLLMBridge()
    return _unified_bridge