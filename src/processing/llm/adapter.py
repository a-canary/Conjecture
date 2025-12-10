"""
LLM Adapter
Single adapter pattern for all LLM providers using the registry
"""

from typing import Optional, Dict, Any
from ...config.config import Config

class LLMAdapter:
    """Single adapter for all LLM providers"""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._primary_provider = None
        self._fallback_provider = None
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize primary and fallback providers"""
        # Get preferred provider from config
        preferred = (
            self.config.llm_provider.lower() if self.config.llm_provider else "chutes"
        )

        # Try to set primary provider
        self._primary_provider = registry.get_provider(preferred, self.config)

        # Set fallback provider if primary is not available
        if not self._primary_provider or not self._primary_provider.is_available():
            available = registry.list_available(self.config)
            if available:
                fallback_name = available[0]
                if fallback_name != preferred:
                    self._fallback_provider = registry.get_provider(
                        fallback_name, self.config
                    )

    def is_available(self) -> bool:
        """Check if any provider is available"""
        return (self._primary_provider and self._primary_provider.is_available()) or (
            self._fallback_provider and self._fallback_provider.is_available()
        )

    def process_request(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Process request using available provider"""
        # Try primary provider first
        if self._primary_provider and self._primary_provider.is_available():
            try:
                return self._primary_provider.process_request(prompt, **kwargs)
            except Exception:
                pass

        # Try fallback provider
        if self._fallback_provider and self._fallback_provider.is_available():
            try:
                return self._fallback_provider.process_request(prompt, **kwargs)
            except Exception:
                pass

        # Return error response
        return {"success": False, "content": "", "error": "No LLM provider available"}

    def get_provider_name(self) -> str:
        """Get the name of the active provider"""
        if self._primary_provider and self._primary_provider.is_available():
            return self._primary_provider.name
        elif self._fallback_provider and self._fallback_provider.is_available():
            return self._fallback_provider.name
        return "none"

def create_adapter(config: Optional[Config] = None) -> LLMAdapter:
    """Factory function to create adapter"""
    return LLMAdapter(config)
