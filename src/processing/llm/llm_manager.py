"""
Unified LLM Manager for Conjecture
Handles multiple LLM providers with automatic fallback and response format adaptation
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from .gemini_integration import GeminiProcessor, GEMINI_AVAILABLE
from .chutes_integration import ChutesProcessor
from .openrouter_integration import OpenRouterProcessor
from .groq_integration import GroqProcessor
from .openai_integration import OpenAIProcessor
from .anthropic_integration import AnthropicProcessor
from .google_integration import GoogleProcessor, GOOGLE_AVAILABLE
from .cohere_integration import CohereProcessor
from .local_providers_adapter import LocalProviderProcessor
from ...core.basic_models import BasicClaim
from ...config.simple_config import Config


class LLMManager:
    """Unified LLM manager supporting all 9 providers with intelligent fallback"""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.processors = {}
        self.provider_priorities = {}
        self.primary_provider = None
        self.failed_providers = set()
        self._initialize_processors()

    def _initialize_processors(self):
        """Initialize all available LLM processors with priority-based ordering"""
        print("[LLM] Initializing all available LLM providers...")
        
        # Get unified provider configuration
        provider_config = self._get_unified_provider_config()
        
        if provider_config:
            # Initialize the configured provider
            self._initialize_provider(
                provider_config["name"].lower(),
                provider_config["api_url"],
                provider_config["api_key"],
                provider_config["model"],
                provider_config["priority"]
            )
        else:
            # Fallback to environment variable method and auto-detection
            self._initialize_from_environment()
        
        # Set primary provider based on priority and health
        self._set_primary_provider()
        
        if not self.processors:
            print("[LLM] Warning: No LLM processors available")
        else:
            print(f"[LLM] Initialized {len(self.processors)} providers: {list(self.processors.keys())}")

    def _get_unified_provider_config(self) -> Optional[Dict[str, Any]]:
        """Get unified provider configuration from environment variables"""
        api_url = os.getenv("PROVIDER_API_URL", "").strip()
        api_key = os.getenv("PROVIDER_API_KEY", "").strip()
        model = os.getenv("PROVIDER_MODEL", "").strip()

        if not api_url:
            return None

        # Detect provider type and priority from URL
        provider_info = self._detect_provider_from_url(api_url)
        
        return {
            "name": provider_info["name"],
            "api_url": api_url,
            "api_key": api_key,
            "model": model,
            "priority": provider_info["priority"],
            "is_local": provider_info["is_local"]
        }

    def _detect_provider_from_url(self, api_url: str) -> Dict[str, Any]:
        """Detect provider type, priority, and local status from URL"""
        url_lower = api_url.lower()

        # Provider detection patterns with priorities
        providers = {
            "ollama": {
                "pattern": r"localhost:11434|ollama",
                "name": "Ollama",
                "priority": 1,
                "is_local": True
            },
            "lm_studio": {
                "pattern": r"localhost:1234|lmstudio",
                "name": "LM Studio", 
                "priority": 2,
                "is_local": True
            },
            "chutes": {
                "pattern": r"chutes\.ai|llm\.chutes\.ai",
                "name": "Chutes.ai",
                "priority": 3,
                "is_local": False
            },
            "openrouter": {
                "pattern": r"openrouter\.ai",
                "name": "OpenRouter",
                "priority": 4,
                "is_local": False
            },
            "groq": {
                "pattern": r"api\.groq\.com|groq",
                "name": "Groq",
                "priority": 5,
                "is_local": False
            },
            "openai": {
                "pattern": r"api\.openai\.com|openai",
                "name": "OpenAI",
                "priority": 6,
                "is_local": False
            },
            "anthropic": {
                "pattern": r"api\.anthropic\.com|anthropic",
                "name": "Anthropic",
                "priority": 7,
                "is_local": False
            },
            "google": {
                "pattern": r"generativelanguage\.googleapis\.com|google",
                "name": "Google",
                "priority": 8,
                "is_local": False
            },
            "cohere": {
                "pattern": r"api\.cohere\.ai|cohere",
                "name": "Cohere",
                "priority": 9,
                "is_local": False
            }
        }

        # Find matching provider
        for provider_key, info in providers.items():
            if re.search(info["pattern"], url_lower):
                return info

        # Default fallback
        return {
            "name": "Unknown",
            "priority": 99,
            "is_local": False
        }

    def _initialize_from_environment(self):
        """Initialize providers from traditional environment variables"""
        # Try each provider in priority order
        provider_configs = [
            ("ollama", "http://localhost:11434", "", "llama2", 1),
            ("lm_studio", "http://localhost:1234/v1", "", "microsoft/DialoGPT-medium", 2),
            ("chutes", "https://llm.chutes.ai/v1", os.getenv("CHUTES_API_KEY") or os.getenv("Conjecture_LLM_API_KEY"), "zai-org/GLM-4.6", 3),
            ("openrouter", "https://openrouter.ai/api/v1", os.getenv("OPENROUTER_API_KEY"), "openai/gpt-3.5-turbo", 4),
            ("groq", "https://api.groq.com/openai/v1", os.getenv("GROQ_API_KEY"), "llama3-8b-8192", 5),
            ("openai", "https://api.openai.com/v1", os.getenv("OPENAI_API_KEY"), "gpt-3.5-turbo", 6),
            ("anthropic", "https://api.anthropic.com", os.getenv("ANTHROPIC_API_KEY"), "claude-3-haiku-20240307", 7),
            ("google", "https://generativelanguage.googleapis.com", os.getenv("GOOGLE_API_KEY"), "gemini-pro", 8),
            ("cohere", "https://api.cohere.ai/v1", os.getenv("COHERE_API_KEY"), "command", 9),
        ]

        for provider_name, api_url, api_key, default_model, priority in provider_configs:
            self._initialize_provider(provider_name, api_url, api_key, default_model, priority)

    def _initialize_provider(self, provider_name: str, api_url: str, api_key: str, model: str, priority: int):
        """Initialize a specific provider"""
        if provider_name in self.failed_providers:
            return  # Skip previously failed providers

        try:
            processor = None

            # Cloud providers (require API keys)
            if provider_name == "chutes":
                if api_key:
                    processor = ChutesProcessor(
                        api_key=api_key,
                        api_url=api_url,
                        model_name=model or "zai-org/GLM-4.6"
                    )
            elif provider_name == "openrouter":
                if api_key:
                    processor = OpenRouterProcessor(
                        api_key=api_key,
                        api_url=api_url,
                        model_name=model or "openai/gpt-3.5-turbo"
                    )
            elif provider_name == "groq":
                if api_key:
                    processor = GroqProcessor(
                        api_key=api_key,
                        api_url=api_url,
                        model_name=model or "llama3-8b-8192"
                    )
            elif provider_name == "openai":
                if api_key:
                    processor = OpenAIProcessor(
                        api_key=api_key,
                        api_url=api_url,
                        model_name=model or "gpt-3.5-turbo"
                    )
            elif provider_name == "anthropic":
                if api_key:
                    processor = AnthropicProcessor(
                        api_key=api_key,
                        api_url=api_url,
                        model_name=model or "claude-3-haiku-20240307"
                    )
            elif provider_name == "google":
                if api_key and GOOGLE_AVAILABLE:
                    processor = GoogleProcessor(
                        api_key=api_key,
                        api_url=api_url,
                        model_name=model or "gemini-pro"
                    )
            elif provider_name == "cohere":
                if api_key:
                    processor = CohereProcessor(
                        api_key=api_key,
                        api_url=api_url,
                        model_name=model or "command"
                    )
            elif provider_name == "gemini":  # Legacy Google support
                if api_key and GEMINI_AVAILABLE:
                    processor = GeminiProcessor(
                        api_key=api_key,
                        model_name="gemini-1.5-flash"
                    )

            # Local providers (don't require API keys)
            elif provider_name == "ollama":
                try:
                    processor = LocalProviderProcessor(
                        provider_type="ollama",
                        base_url=api_url,
                        model_name=model
                    )
                except Exception as e:
                    print(f"[LLM] Ollama not available: {e}")
            elif provider_name == "lm_studio":
                try:
                    processor = LocalProviderProcessor(
                        provider_type="lm_studio",
                        base_url=api_url,
                        model_name=model
                    )
                except Exception as e:
                    print(f"[LLM] LM Studio not available: {e}")

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
        claims: List[BasicClaim],
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
            try:
                # Simple health check - try to generate a short response
                if hasattr(processor, 'health_check'):
                    provider_health = processor.health_check()
                else:
                    # Fallback health check
                    try:
                        result = processor.generate_response("Hello", 
                            config=getattr(processor, 'GenerationConfig', type('Config', (), {'max_tokens': 10}))())
                        provider_health = {
                            "status": "healthy" if result.success else "unhealthy",
                            "last_check": datetime.now().isoformat(),
                            "model": getattr(processor, 'model_name', 'Unknown'),
                            "error": None if result.success else "Generation failed"
                        }
                    except Exception as e:
                        provider_health = {
                            "status": "unhealthy",
                            "last_check": datetime.now().isoformat(),
                            "model": getattr(processor, 'model_name', 'Unknown'),
                            "error": str(e)
                        }

                health_status["providers"][name] = provider_health

                # Update overall status
                if provider_health["status"] == "unhealthy" and name in self.get_available_providers():
                    self.failed_providers.add(name)

            except Exception as e:
                health_status["providers"][name] = {
                    "status": "unhealthy",
                    "last_check": datetime.now().isoformat(),
                    "model": getattr(processor, 'model_name', 'Unknown'),
                    "error": str(e)
                }

        # Re-evaluate overall status
        available_count = len(self.get_available_providers())
        if available_count == 0:
            health_status["overall_status"] = "unavailable"
        elif available_count < len(self.processors):
            health_status["overall_status"] = "degraded"
        else:
            health_status["overall_status"] = "healthy"

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
            "available_providers": len(self.get_available_providers()),
            "failed_providers": list(self.failed_providers),
            "primary_provider": self.primary_provider,
            "providers": {}
        }

        total_requests = 0
        total_successful = 0
        total_tokens = 0
        total_time = 0.0

        for name, processor in self.processors.items():
            if hasattr(processor, 'get_stats'):
                stats = processor.get_stats()
                combined_stats["providers"][name] = stats
                total_requests += stats.get("total_requests", 0)
                total_successful += stats.get("successful_requests", 0)
                total_tokens += stats.get("total_tokens", 0)
                total_time += stats.get("total_processing_time", 0.0)
            else:
                combined_stats["providers"][name] = {"message": "Stats not available"}

        combined_stats["total_requests"] = total_requests
        combined_stats["total_successful"] = total_successful
        combined_stats["total_tokens"] = total_tokens
        combined_stats["total_processing_time"] = total_time

        if total_requests > 0:
            combined_stats["overall_success_rate"] = total_successful / total_requests
            combined_stats["average_processing_time"] = total_time / total_requests
            combined_stats["average_tokens_per_request"] = total_tokens / total_requests
        else:
            combined_stats["overall_success_rate"] = 0.0
            combined_stats["average_processing_time"] = 0.0
            combined_stats["average_tokens_per_request"] = 0.0

        return combined_stats

    def switch_provider(self, provider_name: str) -> bool:
        """Manually switch to a specific provider"""
        if provider_name in self.processors and provider_name not in self.failed_providers:
            old_primary = self.primary_provider
            self.primary_provider = provider_name
            print(f"[LLM] Switched primary provider from {old_primary} to {provider_name}")
            return True
        else:
            print(f"[LLM] Cannot switch to {provider_name}: not available or failed")
            return False

    def reset_failed_providers(self):
        """Reset the list of failed providers and retry health checks"""
        self.failed_providers.clear()
        print("[LLM] Reset failed providers list")
        self._set_primary_provider()

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
            provider_info = {
                "priority": self.provider_priorities.get(name, 999),
                "status": "failed" if name in self.failed_providers else "available",
                "is_primary": name == self.primary_provider
            }

            # Add model information
            if hasattr(processor, 'model_name'):
                provider_info["model"] = processor.model_name
            elif hasattr(processor, '_get_model_name'):
                try:
                    provider_info["model"] = processor._get_model_name()
                except:
                    provider_info["model"] = "Unknown"

            # Add stats if available
            if hasattr(processor, 'get_stats'):
                provider_info["stats"] = processor.get_stats()

            # Add health status if available
            if hasattr(processor, 'health_check'):
                try:
                    health = processor.health_check()
                    provider_info["health"] = health
                except:
                    provider_info["health"] = {"status": "unknown"}

            info["configured_providers"][name] = provider_info

        return info