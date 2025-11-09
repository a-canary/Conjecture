"""
Conjecture Processing Package

This package provides the skill-based agency functionality for the Conjecture project,
including skill management, tool execution, response parsing, and example generation.

Components:
- SkillManager: Manages skill claims and execution coordination
- ResponseParser: Parses LLM responses for structured tool calls
- ToolExecutor: Safe execution engine with resource limits
- ExampleGenerator: Automatic example generation from successful executions

Usage:
    from src.processing import SkillManager, ResponseParser, ToolExecutor
    
    # Parse LLM response
    parser = ResponseParser()
    parsed = parser.parse_response(llm_response)
    
    # Execute tool calls
    executor = ToolExecutor()
    for tool_call in parsed.tool_calls:
        result = await executor.execute_tool_call(tool_call)
"""

from .skill_manager import SkillManager
from .response_parser import ResponseParser
from .tool_executor import ToolExecutor, ExecutionLimits, SafeExecutor
from .example_generator import ExampleGenerator, ExampleQualityAssessor

__version__ = "1.0.0"
__author__ = "Conjecture Team"

__all__ = [
    # Main components
    "SkillManager",
    "ResponseParser", 
    "ToolExecutor",
    "ExampleGenerator",
    
    # Supporting classes
    "ExecutionLimits",
    "SafeExecutor",
    "ExampleQualityAssessor",
]