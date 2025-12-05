"""
Simplified LLM Processing Package for Conjecture
Provides unified OpenAI-compatible provider integration
"""

from .openai_compatible_provider import OpenAICompatibleProcessor, create_openai_compatible_processor
from .common import GenerationConfig, LLMProcessingResult
from .error_handling import LLMErrorHandler, RetryConfig

try:
    from .llm_evaluation_framework import LLMEvaluator, LLMEvaluationResult
except ImportError:
    LLMEvaluator = None
    LLMEvaluationResult = None

__all__ = [
    "OpenAICompatibleProcessor",
    "create_openai_compatible_processor", 
    "GenerationConfig",
    "LLMProcessingResult",
    "LLMErrorHandler",
    "RetryConfig",
    "LLMEvaluator",
    "LLMEvaluationResult"
]