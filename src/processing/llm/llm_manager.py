"""
Unified LLM Manager for Conjecture
Handles multiple LLM providers with automatic fallback and response format adaptation
"""

import os
from typing import Any, Dict, List, Optional
from datetime import datetime

from .gemini_integration import GeminiProcessor, GEMINI_AVAILABLE
from .chutes_integration import ChutesProcessor
from ...core.basic_models import BasicClaim
from ...config.simple_config import Config


class LLMManager:
    """Unified LLM manager supporting multiple providers"""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.processors = {}
        self.primary_provider = None
        self._initialize_processors()

    def _initialize_processors(self):
        """Initialize available LLM processors"""
        # Initialize Chutes.ai as primary if configured
        if self.config.llm_provider == "chutes":
            api_key = os.getenv("CHUTES_API_KEY") or os.getenv("Conjecture_LLM_API_KEY")
            if api_key:
                try:
                    self.processors["chutes"] = ChutesProcessor(
                        api_key=api_key,
                        api_url=self.config.llm_api_url,
                        model_name=self.config.llm_model
                    )
                    self.primary_provider = "chutes"
                    print(f"[LLM] Chutes.ai processor initialized with model: {self.config.llm_model}")
                except Exception as e:
                    print(f"[LLM] Failed to initialize Chutes.ai: {e}")

        # Initialize Gemini as fallback
        if GEMINI_AVAILABLE:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                try:
                    self.processors["gemini"] = GeminiProcessor(
                        api_key=api_key,
                        model_name="gemini-1.5-flash"
                    )
                    if not self.primary_provider:
                        self.primary_provider = "gemini"
                        print("[LLM] Gemini processor initialized as fallback")
                except Exception as e:
                    print(f"[LLM] Failed to initialize Gemini: {e}")

        if not self.processors:
            print("[LLM] Warning: No LLM processors available")

    def get_processor(self, provider: Optional[str] = None) -> Optional[Any]:
        """Get LLM processor by provider name"""
        if provider and provider in self.processors:
            return self.processors[provider]
        
        # Return primary provider or first available
        if self.primary_provider and self.primary_provider in self.processors:
            return self.processors[self.primary_provider]
        
        if self.processors:
            return list(self.processors.values())[0]
        
        return None

    def process_claims(
        self, 
        claims: List[BasicClaim], 
        task: str = "analyze",
        provider: Optional[str] = None,
        **kwargs
    ):
        """Process claims using specified or primary LLM provider"""
        processor = self.get_processor(provider)
        if not processor:
            raise RuntimeError("No LLM processor available")

        try:
            return processor.process_claims(claims, task, **kwargs)
        except Exception as e:
            print(f"[LLM] Processing failed with {provider or 'primary'}: {e}")
            
            # Try fallback provider if available
            if provider and len(self.processors) > 1:
                fallback = self.get_processor()
                if fallback and fallback != processor:
                    print(f"[LLM] Trying fallback provider...")
                    return fallback.process_claims(claims, task, **kwargs)
            
            raise

    def generate_response(
        self, 
        prompt: str, 
        provider: Optional[str] = None,
        **kwargs
    ):
        """Generate response using specified or primary LLM provider"""
        processor = self.get_processor(provider)
        if not processor:
            raise RuntimeError("No LLM processor available")

        try:
            return processor.generate_response(prompt, **kwargs)
        except Exception as e:
            print(f"[LLM] Generation failed with {provider or 'primary'}: {e}")
            
            # Try fallback provider if available
            if provider and len(self.processors) > 1:
                fallback = self.get_processor()
                if fallback and fallback != processor:
                    print(f"[LLM] Trying fallback provider...")
                    return fallback.generate_response(prompt, **kwargs)
            
            raise

    def get_available_providers(self) -> List[str]:
        """Get list of available LLM providers"""
        return list(self.processors.keys())

    def get_provider_stats(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for specified or primary provider"""
        processor = self.get_processor(provider)
        if not processor:
            return {}

        if hasattr(processor, 'get_stats'):
            return processor.get_stats()
        
        return {}

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all LLM providers"""
        health_status = {
            "total_providers": len(self.processors),
            "primary_provider": self.primary_provider,
            "providers": {},
            "overall_status": "healthy" if self.processors else "unavailable"
        }

        for name, processor in self.processors.items():
            try:
                # Simple health check - try to generate a short response
                result = processor.generate_response("Hello", 
                    config=processor.GenerationConfig(max_tokens=10))
                health_status["providers"][name] = {
                    "status": "healthy" if result.success else "unhealthy",
                    "last_check": datetime.now().isoformat(),
                    "model": processor.model_name
                }
            except Exception as e:
                health_status["providers"][name] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "last_check": datetime.now().isoformat(),
                    "model": processor.model_name
                }

        return health_status

    def reset_stats(self, provider: Optional[str] = None):
        """Reset statistics for specified or all providers"""
        if provider and provider in self.processors:
            processor = self.processors[provider]
            if hasattr(processor, 'reset_stats'):
                processor.reset_stats()
        else:
            for processor in self.processors.values():
                if hasattr(processor, 'reset_stats'):
                    processor.reset_stats()

    def get_combined_stats(self) -> Dict[str, Any]:
        """Get combined statistics from all providers"""
        combined_stats = {
            "total_providers": len(self.processors),
            "primary_provider": self.primary_provider,
            "providers": {}
        }

        total_requests = 0
        total_successful = 0
        total_tokens = 0

        for name, processor in self.processors.items():
            if hasattr(processor, 'get_stats'):
                stats = processor.get_stats()
                combined_stats["providers"][name] = stats
                total_requests += stats.get("total_requests", 0)
                total_successful += stats.get("successful_requests", 0)
                total_tokens += stats.get("total_tokens", 0)

        combined_stats["total_requests"] = total_requests
        combined_stats["total_successful"] = total_successful
        combined_stats["total_tokens"] = total_tokens
        
        if total_requests > 0:
            combined_stats["overall_success_rate"] = total_successful / total_requests
        else:
            combined_stats["overall_success_rate"] = 0.0

        return combined_stats