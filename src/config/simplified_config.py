"""
Simplified Config - Compatibility Layer
Redirects to unified config for backward compatibility
"""

from .unified_config import UnifiedConfig as SimplifiedConfigManager
from .pydantic_config import ProviderConfig

# Export the main classes
__all__ = ['SimplifiedConfigManager', 'ProviderConfig']