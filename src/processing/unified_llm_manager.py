"""
Unified LLM Manager for Conjecture
Consolidates all LLM provider functionality into a single, clean system
"""

import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from .llm.gemini_integration import GeminiProcessor, GEMINI_AVAILABLE
from .llm.chutes_integration import ChutesProcessor
from .llm.openrouter_integration import OpenRouterProcessor
from .llm.groq_integration import GroqProcessor
from .llm.openai_integration import OpenAIProcessor
from .llm.anthropic_integration import AnthropicProcessor
from .llm.google_integration import GoogleProcessor, GOOGLE_AVAILABLE
from .llm.cohere_integration import CohereProcessor
from .llm.local_providers_adapter import LocalProviderProcessor
from ..core.models import Claim
from ..config.unified_config import UnifiedConfig


class UnifiedLLMManager:
    """Unified LLM manager supporting all providers with intelligent fallback"""

    def __init__(self, config: Optional[UnifiedConfig] = None):
        self.config = config or UnifiedConfig()
        self.processors = {}
        self.provider_priorities = {}
        self.primary_provider = None
        self.failed_providers = set()
        self._provider_config = self._load_provider_config()
        self._initialize_processors()

    def _load_provider_config(self) -> Dict[str, Any]:
        """Load provider configuration from JSON file."""
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'providers.json')
            with open(config_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"[LLM] Warning: Could not load provider config: {e}")
            return {}

    def _initialize_processors(self):
        """Initialize all available LLM processors with priority-based ordering"""
        print("[LLM] Initializing all available LLM providers...")
        
        if self._provider_config:
            provider_configs = [
                (name, info.get("api_url", ""), os.getenv(info.get("api_key_env", "")), info.get("default_model", ""), info.get("priority", 99))
                for name, info in self._provider_config.items()
            ]
            for provider_name, api_url, api_key, default_model, priority in provider_configs:
                self._initialize_provider(provider_name, api_url, api_key, default_model, priority)
        else:
            # Fallback to unified config
            self._initialize_from_unified_config()
        
        # Set primary provider based on priority and health
        self._set_primary_provider()
        
        if not self.processors:
            print("[LLM] Warning: No LLM processors available")
        else:
            print(f"[LLM] Initialized {len(self.processors)} providers: {list(self.processors.keys())}")

    def _initialize_from_unified_config(self):
        """Initialize providers from unified config system"""
        providers = self.config.settings.providers
        
        for provider in providers:
            provider_name = provider.name
            api_url = provider.url
            api_key = provider.api
            model = provider.model
            priority = provider.priority
            
            self._initialize_provider(provider_name, api_url, api_key, model, priority)

    def _initialize_provider(self, provider_name: str, api_url: str, api_key: str, model: str, priority: int):
        """Initialize a specific provider"""
        if provider_name in self.failed_providers:
            return

        provider_info = self._provider_config.get(provider_name)
        if not provider_info:
            # Try to create from unified config
            try:
                if provider_name == "ollama":
                    processor = LocalProviderProcessor(
                        provider_type="ollama",
                        base_url=api_url,
                        model_name=model
                    )
                elif provider_name == "lm_studio":
                    processor = LocalProviderProcessor(
                        provider_type="lm_studio",
                        base_url=api_url,
                        model_name=model
                    )
                elif api_key:  # Cloud provider
                    processor_class_name = provider_name + "Processor"
                    processor_class = globals().get(processor_class_name)
                    if processor_class:
                        processor = processor_class(
                            api_key=api_key,
                            api_url=api_url,
                            model_name=model
                        )
                    else:
                        return
                else:
                    return

                if processor:
                    self.processors[provider_name] = processor
                    self.provider_priorities[provider_name] = priority
                    print(f"[LLM] {provider_name.capitalize()} processor initialized (priority: {priority})")

            except Exception as e:
                self.failed_providers.add(provider_name)
                print(f"[LLM] Failed to initialize {provider_name}: {e}")

    def _set_primary_provider(self):
        """Set primary provider based on priority and availability"""
        if not self.processors:
            self.primary_provider = None
            return

        # Sort providers by priority (lower number = higher priority)
        sorted_providers = sorted(
            self.processors.items(),
            key=lambda x: self.provider_priorities.get(x[0], 999)
        )

        # Try to set primary based on health check
        for provider_name, processor in sorted_providers:
            try:
                health = processor.health_check()
                if health.get("status") == "healthy":
                    self.primary_provider = provider_name
                    print(f"[LLM] Primary provider set to: {provider_name}")
                    return
            except Exception as e:
                print(f"[LLM] Health check failed for {provider_name}: {e}")
                self.failed_providers.add(provider_name)

        # If all health checks failed, use highest priority available
        if sorted_providers:
            self.primary_provider = sorted_providers[0][0]
            print(f"[LLM] Warning: All health checks failed. Using {self.primary_provider} as primary")

    def get_processor(self, provider: Optional[str] = None) -> Optional[Any]:
        """Get LLM processor by provider name with intelligent fallback"""
        # Specific provider requested
        if provider and provider in self.processors and provider not in self.failed_providers:
            return self.processors[provider]

        # Return primary provider if available
        if self.primary_provider and self.primary_provider in self.processors and self.primary_provider not in self.failed_providers:
            return self.processors[self.primary_provider]

        # Find next best available provider by priority
        available_providers = [
            (name, processor) for name, processor in self.processors.items()
            if name not in self.failed_providers
        ]

        if available_providers:
            # Sort by priority and return the best available
            sorted_providers = sorted(
                available_providers,
                key=lambda x: self.provider_priorities.get(x[0], 999)
            )
            best_provider = sorted_providers[0][0]
            print(f"[LLM] Using fallback provider: {best_provider}")
            return sorted_providers[0][1]

        # If all providers failed, try to reset failed list and retry once
        if self.failed_providers and self.processors:
            print("[LLM] All providers failed, attempting reset...")
            self.failed_providers.clear()
            return self.get_processor()

        return None

    def process_claims(
        self,
        claims: List[Claim],
        task: str = "analyze",
        provider: Optional[str] = None,
        **kwargs
    ):
        """Process claims using specified or primary LLM provider with automatic fallback"""
        attempted_providers = set()
        
        while True:
            processor = self.get_processor(provider)
            if not processor:
                raise RuntimeError("No LLM processor available")

            provider_name = self._get_provider_name(processor)
            if provider_name in attempted_providers:
                raise RuntimeError("All available providers failed")

            attempted_providers.add(provider_name)

            try:
                return processor.process_claims(claims, task, **kwargs)
            except Exception as e:
                print(f"[LLM] Processing failed with {provider_name}: {e}")
                self.failed_providers.add(provider_name)
                
                # If this was a specific provider request, try fallback
                if provider:
                    provider = None  # Allow fallback to primary or next best
                    continue
                
                # Try next available provider
                if len(attempted_providers) < len(self.processors):
                    continue
                else:
                    raise

    def generate_response(
        self,
        prompt: str,
        provider: Optional[str] = None,
        **kwargs
    ):
        """Generate response using specified or primary LLM provider with automatic fallback"""
        attempted_providers = set()
        
        while True:
            processor = self.get_processor(provider)
            if not processor:
                raise RuntimeError("No LLM processor available")

            provider_name = self._get_provider_name(processor)
            if provider_name in attempted_providers:
                raise RuntimeError("All available providers failed")

            attempted_providers.add(provider_name)

            try:
                return processor.generate_response(prompt, **kwargs)
            except Exception as e:
                print(f"[LLM] Generation failed with {provider_name}: {e}")
                self.failed_providers.add(provider_name)
                
                # If this was a specific provider request, try fallback
                if provider:
                    provider = None  # Allow fallback to primary or next best
                    continue
                
                # Try next available provider
                if len(attempted_providers) < len(self.processors):
                    continue
                else:
                    raise

    @with_llm_retry(max_attempts=3, base_delay=5.0, max_delay=30.0)
    def generate_response_with_retry(self, processor, provider_name: str, prompt: str, **kwargs):
        """Generate response with retry logic for a specific provider"""
        try:
            result = processor.generate_response(prompt, **kwargs)
            logger.info(f"Successfully generated response using {provider_name}")
            return result
        except Exception as e:
            logger.error(f"Generation failed with {provider_name}: {e}")
            self.failed_providers.add(provider_name)
            raise

    def _get_provider_name(self, processor) -> str:
        """Get provider name from processor instance"""
        for name, proc in self.processors.items():
            if proc is processor:
                return name
        return "unknown"

    def get_available_providers(self) -> List[str]:
        """Get list of available (not failed) LLM providers"""
        return [name for name in self.processors.keys() if name not in self.failed_providers]

    def get_all_configured_providers(self) -> List[str]:
        """Get list of all configured providers (including failed ones)"""
        return list(self.processors.keys())

    def get_provider_stats(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for specified or primary provider"""
        processor = self.get_processor(provider)
        if not processor:
            return {}

        if hasattr(processor, 'get_stats'):
            stats = processor.get_stats()
            stats['provider_name'] = self._get_provider_name(processor)
            return stats

        return {
            'provider_name': self._get_provider_name(processor),
            'message': 'Stats not available for this provider'
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check on all LLM providers"""
        health_status = {
            "total_providers": len(self.processors),
            "available_providers": len(self.get_available_providers()),
            "failed_providers": list(self.failed_providers),
            "primary_provider": self.primary_provider,
            "providers": {},
            "overall_status": "healthy" if self.processors else "unavailable"
        }

        for name, processor in self.processors.items():
            provider_health = self._get_provider_health(name, processor)
            health_status["providers"][name] = provider_health
            if provider_health["status"] == "unhealthy" and name in self.get_available_providers():
                self.failed_providers.add(name)

        available_count = len(self.get_available_providers())
        if available_count == 0:
            health_status["overall_status"] = "unavailable"
        elif available_count < len(self.processors):
            health_status["overall_status"] = "degraded"
        else:
            health_status["overall_status"] = "healthy"

        return health_status

    def _get_provider_health(self, name: str, processor: Any) -> Dict[str, Any]:
        """Get health status for a single provider."""
        try:
            if hasattr(processor, 'health_check'):
                return processor.health_check()
            else:
                result = processor.generate_response("Hello", config=getattr(processor, 'GenerationConfig', type('Config', (), {'max_tokens': 10}))())
                return {
                    "status": "healthy" if result.success else "unhealthy",
                    "last_check": datetime.now().isoformat(),
                    "model": getattr(processor, 'model_name', 'Unknown'),
                    "error": None if result.success else "Generation failed"
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "last_check": datetime.now().isoformat(),
                "model": getattr(processor, 'model_name', 'Unknown'),
                "error": str(e)
            }

    def get_provider_info(self) -> Dict[str, Any]:
        """Get detailed information about all providers"""
        info = {
            "configured_providers": {},
            "provider_priorities": self.provider_priorities,
            "primary_provider": self.primary_provider,
            "failed_providers": list(self.failed_providers),
            "available_providers": self.get_available_providers()
        }

        for name, processor in self.processors.items():
            info["configured_providers"][name] = {
                "priority": self.provider_priorities.get(name, 999),
                "status": "failed" if name in self.failed_providers else "available",
                "is_primary": name == self.primary_provider
            }

            if hasattr(processor, 'model_name'):
                info["configured_providers"][name]["model"] = processor.model_name

            if hasattr(processor, 'get_stats'):
                info["configured_providers"][name]["stats"] = processor.get_stats()

        return info


# Global instance for easy access
_unified_llm_manager = None


def get_unified_llm_manager() -> UnifiedLLMManager:
    """Get the global unified LLM manager instance"""
    global _unified_llm_manager
    if _unified_llm_manager is None:
        _unified_llm_manager = UnifiedLLMManager()
    return _unified_llm_manager