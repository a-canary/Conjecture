"""
LLM Manager - Compatibility Layer
Redirects to the unified LLM manager for backward compatibility
"""

# Import the actual LLM manager for backward compatibility
from ..unified_llm_manager import UnifiedLLMManager as LLMManager

# Export the main class
__all__ = ['LLMManager']