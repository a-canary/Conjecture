"""
Simplified LLM Manager for Conjecture
Uses only OpenAI-compatible providers for maximum compatibility and simplicity
"""

import os
import json
from typing import Any, Dict, List, Optional
from datetime import datetime

from .llm.openai_compatible_provider import OpenAICompatibleProcessor, create_openai_compatible_processor
from src.core.models import Claim

class SimplifiedLLMManager:
    """Simplified LLM manager supporting only OpenAI-compatible providers"""

    def __init__(self, providers: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize simplified LLM manager
        
        Args:
            providers: List of provider configurations. If None, loads from config.
        """
        self.processors = {}
        self.provider_priorities = {}
        self.primary_provider = None
        self.failed_providers = set()
        
        if providers is None:
            providers = self._load_providers_from_config()
        
        self._initialize_processors(providers)

    def _load_providers_from_config(self) -> List[Dict[str, Any]]:
        """Load provider configurations from config file"""
        try:
            # Try to load from user config first
            config_path = os.path.expanduser("~/.conjecture/config.json")
            if not os.path.exists(config_path):
                # Fallback to default config
                config_path = "src/config/default_config.json"
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            providers = []
            providers_config = config.get("providers", [])
            
            # Handle both list and dict formats for backward compatibility
            if isinstance(providers_config, list):
                # New format: list of provider objects
                for provider_config in providers_config:
                    providers.append({
                        "name": provider_config.get("name", f"provider_{len(providers)}"),
                        "url": provider_config.get("url", ""),
                        "api": provider_config.get("api", provider_config.get("key", "")),
                        "model": provider_config.get("model", ""),
                        "priority": provider_config.get("priority", 999)
                    })
            elif isinstance(providers_config, dict):
                # Old format: dict of named provider configs
                for name, provider_config in providers_config.items():
                    providers.append({
                        "name": name,
                        "url": provider_config.get("url", ""),
                        "api": provider_config.get("api", provider_config.get("key", "")),
                        "model": provider_config.get("model", ""),
                        "priority": provider_config.get("priority", 999)
                    })
            
            return providers
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"[LLM] Warning: Could not load provider config: {e}")
            return []

    def _validate_provider_config(self, provider_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate provider configuration with clear error messages"""
        errors = []
        
        # Validate required fields
        if "name" not in provider_config:
            errors.append("Provider name is required")
        elif not provider_config["name"] or not isinstance(provider_config["name"], str):
            errors.append("Provider name must be a non-empty string")
        
        if "url" not in provider_config:
            errors.append("Provider URL is required")
        elif not provider_config["url"] or not isinstance(provider_config["url"], str):
            errors.append("Provider URL must be a non-empty string")
        
        # Validate optional fields
        url = provider_config.get("url", "")
        if url and not (url.startswith("http://") or url.startswith("https://")):
            errors.append("Provider URL must start with http:// or https://")
        
        api_key = provider_config.get("api", "")
        if api_key and not isinstance(api_key, str):
            errors.append("API key must be a string")
        
        model = provider_config.get("model", "gpt-3.5-turbo")
        if not isinstance(model, str) or not model.strip():
            errors.append("Model must be a non-empty string")
        
        priority = provider_config.get("priority", 999)
        if not isinstance(priority, int) or priority < 0:
            errors.append("Priority must be a non-negative integer")
        
        if errors:
            error_msg = f"Configuration validation failed for {provider_config.get('name', 'unknown')}: {'; '.join(errors)}"
            raise ValueError(error_msg)
        
        return provider_config
    
    def _initialize_processors(self, providers: List[Dict[str, Any]]):
        """Initialize OpenAI-compatible processors with validation"""
        print("[LLM] Initializing OpenAI-compatible providers...")
        
        for provider_config in providers:
            try:
                # Validate configuration first
                validated_config = self._validate_provider_config(provider_config)
                
                provider_name = validated_config["name"]
                api_url = validated_config["url"]
                api_key = validated_config.get("api", "")
                model = validated_config.get("model", "gpt-3.5-turbo")
                priority = validated_config.get("priority", 999)
                
                # Create unified processor
                processor = create_openai_compatible_processor(
                    provider_name=provider_name,
                    api_url=api_url,
                    api_key=api_key,
                    model=model
                )
                
                self.processors[provider_name] = processor
                self.provider_priorities[provider_name] = priority
                print(f"[LLM] {provider_name} processor initialized (priority: {priority})")
                
            except ValueError as e:
                print(f"[LLM] Configuration validation failed: {e}")
                continue  # Skip this provider but continue with others
            except Exception as e:
                print(f"[LLM] Failed to initialize {provider_config.get('name', 'unknown')}: {e}")
        
        # Set primary provider based on priority and health
        self._set_primary_provider()
        
        if not self.processors:
            print("[LLM] Warning: No LLM processors available")
        else:
            print(f"[LLM] Initialized {len(self.processors)} providers: {list(self.processors.keys())}")

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

    def get_processor(self, provider: Optional[str] = None) -> Optional[OpenAICompatibleProcessor]:
        """Get LLM processor by provider name with intelligent fallback"""
        # Specific provider requested
        if provider and provider in self.processors and provider not in self.failed_providers:
            return self.processors[provider]

        # Return primary provider if available
        if (self.primary_provider and 
            self.primary_provider in self.processors and 
            self.primary_provider not in self.failed_providers):
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

    def _get_provider_name(self, processor: OpenAICompatibleProcessor) -> str:
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

        stats = processor.get_stats()
        stats['provider_name'] = self._get_provider_name(processor)
        return stats

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

    def _get_provider_health(self, name: str, processor: OpenAICompatibleProcessor) -> Dict[str, Any]:
        """Get health status for a single provider."""
        try:
            return processor.health_check()
        except Exception as e:
            return {
                "status": "unhealthy",
                "last_check": datetime.now().isoformat(),
                "model": getattr(processor, 'model_name', 'Unknown'),
                "provider": name,
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
                "is_primary": name == self.primary_provider,
                "model": processor.model_name,
                "url": processor.api_url,
                "is_local": processor._is_local_provider()
            }

            if hasattr(processor, 'get_stats'):
                info["configured_providers"][name]["stats"] = processor.get_stats()

        return info

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
            stats = processor.get_stats()
            combined_stats["providers"][name] = stats
            total_requests += stats.get("total_requests", 0)
            total_successful += stats.get("successful_requests", 0)
            total_tokens += stats.get("total_tokens", 0)
            total_time += stats.get("total_processing_time", 0.0)

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

# Global instance for easy access
_simplified_llm_manager = None

def get_simplified_llm_manager() -> SimplifiedLLMManager:
    """Get the global simplified LLM manager instance"""
    global _simplified_llm_manager
    if _simplified_llm_manager is None:
        _simplified_llm_manager = SimplifiedLLMManager()
    return _simplified_llm_manager