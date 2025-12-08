"""
LM Studio Adapter - Compatibility Layer
Redirects to the local providers adapter for backward compatibility
"""

from .local_providers_adapter import LocalProviderProcessor as LMStudioAdapter

def create_lm_studio_adapter_from_config(config):
    """Create LM Studio adapter from configuration"""
    return LMStudioAdapter(config)

# Export the main class and function
__all__ = ['LMStudioAdapter', 'create_lm_studio_adapter_from_config']