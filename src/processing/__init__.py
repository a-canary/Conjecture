"""
Simplified Conjecture Processing Package
Unified processing system with OpenAI-compatible providers only
"""

# Core simplified processing components
from .simplified_llm_manager import SimplifiedLLMManager, get_simplified_llm_manager

# Unified bridge components (optional)
try:
    from .unified_bridge import (
        UnifiedLLMBridge,
        get_unified_bridge,
        LLMRequest,
        LLMResponse,
    )
except ImportError:
    UnifiedLLMBridge = None
    get_unified_bridge = None
    LLMRequest = None
    LLMResponse = None

# Legacy components (maintained for backward compatibility)
try:
    from .response_parser import ResponseParser
    from .tool_executor import ToolExecutor, ExecutionLimits, SafeExecutor
    from .example_generator import ExampleGenerator, ExampleQualityAssessor
except ImportError:
    # Handle import errors gracefully for modular testing
    ResponseParser = None
    ToolExecutor = None
    ExecutionLimits = None
    SafeExecutor = None
    ExampleGenerator = None
    ExampleQualityAssessor = None

__version__ = "2.0.0"
__author__ = "Conjecture Team"

__all__ = [
    # Simplified components
    "SimplifiedLLMManager",
    "get_simplified_llm_manager",
    "UnifiedLLMBridge",
    "get_unified_bridge",
    "LLMRequest",
    "LLMResponse",
    # Legacy components (backward compatibility)
    "ResponseParser",
    "ToolExecutor",
    "ExecutionLimits",
    "SafeExecutor",
    "ExampleGenerator",
    "ExampleQualityAssessor",
]
