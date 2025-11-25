"""
Unified LLM Provider Registry
Consolidates all LLM providers into a single adapter pattern
"""

from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

from ..config.simple_config import Config


class LLMProvider(ABC):
    """Base class for all LLM providers"""

    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available"""
        pass

    @abstractmethod
    def process_request(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Process a request and return standardized response"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name"""
        pass


class ProviderRegistry:
    """Registry for managing LLM providers"""

    def __init__(self):
        self._providers: Dict[str, LLMProvider] = {}
        self._register_default_providers()

    def _register_default_providers(self):
        """Register all available providers"""
        try:
            from .chutes_provider import ChutesProvider

            self.register("chutes", ChutesProvider)
        except ImportError:
            pass

        try:
            from .lm_studio_provider import LMStudioProvider

            self.register("lm_studio", LMStudioProvider)
        except ImportError:
            pass

    def register(self, name: str, provider_class: type):
        """Register a provider class"""
        self._providers[name] = provider_class

    def get_provider(self, name: str, config: Config) -> Optional[LLMProvider]:
        """Get an instance of a provider"""
        if name not in self._providers:
            return None

        try:
            return self._providers[name](config)
        except Exception:
            return None

    def list_available(self, config: Config) -> List[str]:
        """List all available providers"""
        available = []
        for name in self._providers:
            provider = self.get_provider(name, config)
            if provider and provider.is_available():
                available.append(name)
        return available


# Global registry instance
registry = ProviderRegistry()
