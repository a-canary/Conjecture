"""
Provider Discovery System for Conjecture CLI

Automatic discovery of LLM providers including:
- Local services (Ollama, LM Studio)
- Cloud services via API keys
- Configuration management
- Security-focused implementation
"""

from .provider_discovery import ProviderDiscovery
from .service_detector import ServiceDetector
from .config_updater import ConfigUpdater

__all__ = [
    'ProviderDiscovery',
    'ServiceDetector', 
    'ConfigUpdater'
]

__version__ = '1.0.0'