"""
Core Tools Registry System for Conjecture.

This module provides the ToolRegistry system and register_tool decorator
for dynamically managing available tools in the Conjecture system.
"""

from .registry import register_tool, ToolRegistry

__all__ = ['register_tool', 'ToolRegistry']