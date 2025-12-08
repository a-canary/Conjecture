"""
Simple LLM Processor - Compatibility Layer
Redirects to the simplified LLM manager for backward compatibility
"""

from .simplified_llm_manager import SimplifiedLLMManager as SimpleLLMProcessor

# Export the main class
__all__ = ['SimpleLLMProcessor']