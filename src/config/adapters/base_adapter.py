"""
Base Adapter Class
Provides common interface and data structures for all configuration adapters
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
from enum import Enum

from .common import ProviderConfig, ValidationResult


class FormatPriority(Enum):
    """Priority levels for configuration formats"""

    HIGHEST = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    LOWEST = 5
    errors: List[str]
    warnings: List[str]
    format_type: str
    priority: FormatPriority

    def __post_init__(self):
        if self.providers is None:
            self.providers = []
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class BaseAdapter(ABC):
    """
    Abstract base class for configuration format adapters
    All adapters must implement this interface to ensure consistency
    """

    def __init__(self, format_type: str, priority: FormatPriority):
        self.format_type = format_type
        self.priority = priority

    @abstractmethod
    def detect_format(self, env_vars: Dict[str, str]) -> bool:
        """
        Detect if the environment contains this format
        Returns True if format is detected
        """
        pass

    @abstractmethod
    def load_providers(
        self, env_vars: Dict[str, str]
    ) -> Tuple[bool, List[ProviderConfig], List[str]]:
        """
        Load providers from environment variables
        Returns: (success, providers, errors)
        """
        pass

    @abstractmethod
    def validate_format(self, env_vars: Dict[str, str]) -> Tuple[bool, List[str]]:
        """
        Validate the format structure without loading providers
        Returns: (is_valid, errors)
        """
        pass

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about this adapter"""
        return {
            "format_type": self.format_type,
            "priority": self.priority.value,
            "description": self.__class__.__doc__ or "No description available",
        }

    def create_provider_config(self, **kwargs) -> ProviderConfig:
        """Create a ProviderConfig with default values from this adapter"""
        return ProviderConfig(format_type=self.format_type, **kwargs)

    def normalize_url(self, url: str) -> str:
        """Normalize URL format"""
        if not url:
            return ""

        url = url.strip()
        if not url.startswith(("http://", "https://")):
            # Default to http for local, https for remote
            if any(
                local_indicator in url.lower()
                for local_indicator in ["localhost", "127.0.0.1", ".local"]
            ):
                url = f"http://{url}"
            else:
                url = f"https://{url}"

        return url

    def detect_provider_type(self, url: str) -> Dict[str, Any]:
        """Detect provider type and properties from URL"""
        url_lower = url.lower()

        # Provider patterns for detection
        provider_patterns = {
            "ollama": {
                "patterns": ["localhost:11434", "ollama", "/ollama"],
                "is_local": True,
                "default_protocol": "ollama",
                "default_model": "llama2",
            },
            "lm_studio": {
                "patterns": ["localhost:1234", "lmstudio", "/v1"],
                "is_local": True,
                "default_protocol": "openai",
                "default_model": "microsoft/DialoGPT-medium",
            },
            "chutes": {
                "patterns": ["chutes.ai", "llm.chutes.ai"],
                "is_local": False,
                "default_protocol": "openai",
                "default_model": "chutes-gpt-3.5-turbo",
            },
            "openrouter": {
                "patterns": ["openrouter.ai"],
                "is_local": False,
                "default_protocol": "openai",
                "default_model": "openai/gpt-3.5-turbo",
            },
            "openai": {
                "patterns": ["api.openai.com", "openai"],
                "is_local": False,
                "default_protocol": "openai",
                "default_model": "gpt-3.5-turbo",
            },
            "anthropic": {
                "patterns": ["api.anthropic.com", "anthropic"],
                "is_local": False,
                "default_protocol": "anthropic",
                "default_model": "claude-3-haiku-20240307",
            },
            "google": {
                "patterns": ["generativelanguage.googleapis.com", "google"],
                "is_local": False,
                "default_protocol": "google",
                "default_model": "gemini-pro",
            },
            "groq": {
                "patterns": ["api.groq.com", "groq"],
                "is_local": False,
                "default_protocol": "openai",
                "default_model": "llama3-8b-8192",
            },
            "cohere": {
                "patterns": ["api.cohere.ai", "cohere"],
                "is_local": False,
                "default_protocol": "cohere",
                "default_model": "command",
            },
        }

        for provider_name, info in provider_patterns.items():
            for pattern in info["patterns"]:
                if pattern in url_lower:
                    return {
                        "name": provider_name.replace("_", " ").title(),
                        "key": provider_name,
                        "is_local": info["is_local"],
                        "protocol": info["default_protocol"],
                        "default_model": info["default_model"],
                    }

        # Default fallback
        return {
            "name": "Unknown",
            "key": "unknown",
            "is_local": "localhost" in url_lower or "127.0.0.1" in url_lower,
            "protocol": "openai",
            "default_model": "default",
        }

    def get_priority_for_provider(self, provider_key: str) -> int:
        """Get the priority for a specific provider"""
        priorities = {
            "ollama": 1,
            "lm_studio": 2,
            "chutes": 3,
            "openrouter": 4,
            "groq": 5,
            "openai": 6,
            "anthropic": 7,
            "google": 8,
            "cohere": 9,
            "unknown": 99,
        }
        return priorities.get(provider_key.lower(), 99)
