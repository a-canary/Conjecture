"""
LLM Processing Package for Conjecture
Provides LLM integration capabilities with multiple providers
"""

try:
    from .gemini_integration import GeminiProcessor, GEMINI_AVAILABLE
except ImportError:
    GeminiProcessor = None
    GEMINI_AVAILABLE = False

try:
    from .chutes_integration import ChutesProcessor
except ImportError:
    ChutesProcessor = None

try:
    from .llm_manager import LLMManager
except ImportError:
    LLMManager = None

try:
    from .llm_evaluation_framework import LLMEvaluator, LLMEvaluationResult
except ImportError:
    LLMEvaluator = None
    LLMEvaluationResult = None

__all__ = [
    "GeminiProcessor",
    "ChutesProcessor",
    "LLMManager",
    "GEMINI_AVAILABLE", 
    "LLMEvaluator",
    "LLMEvaluationResult"
]