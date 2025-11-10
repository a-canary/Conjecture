"""
LLM Processing Package for Conjecture
Provides LLM integration capabilities with multiple providers
"""

from .gemini_integration import GeminiProcessor, GEMINI_AVAILABLE
from .chutes_integration import ChutesProcessor
from .llm_manager import LLMManager
from .llm_evaluation_framework import LLMEvaluator, LLMEvaluationResult

__all__ = [
    "GeminiProcessor",
    "ChutesProcessor",
    "LLMManager",
    "GEMINI_AVAILABLE", 
    "LLMEvaluator",
    "LLMEvaluationResult"
]