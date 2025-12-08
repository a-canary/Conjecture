"""
Evaluation Module for Conjecture DeepEval Integration
Provides comprehensive evaluation framework with DeepEval metrics
"""

from .conjecture_llm_wrapper import (
    ConjectureLLMWrapper,
    GraniteTinyWrapper,
    GLM46Wrapper,
    GptOss20bWrapper,
    create_conjecture_wrapper,
    get_available_conjecture_providers,
    test_conjecture_wrapper
)

from .evaluation_framework import (
    EvaluationFramework,
    evaluate_single_provider,
    evaluate_all_providers
)

__all__ = [
    # LLM Wrappers
    'ConjectureLLMWrapper',
    'GraniteTinyWrapper',
    'GLM46Wrapper',
    'GptOss20bWrapper',
    'create_conjecture_wrapper',
    'get_available_conjecture_providers',
    'test_conjecture_wrapper',
    
    # Evaluation Framework
    'EvaluationFramework',
    'evaluate_single_provider',
    'evaluate_all_providers'
]

__version__ = "1.0.0"
__author__ = "Conjecture Team"
__description__ = "DeepEval integration for Conjecture's LLM providers"