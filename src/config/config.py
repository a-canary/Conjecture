"""
Config - Compatibility Layer
Redirects to unified config for backward compatibility
"""

from .unified_config import UnifiedConfig as Config

# Export the main class
__all__ = ['Config']