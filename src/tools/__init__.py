"""
Core Tools Registry System for Conjecture.

This module provides the ToolRegistry system and register_tool decorator
for dynamically managing available tools in the Conjecture system.
"""

from .registry import register_tool, ToolRegistry


# Add ToolManager for backward compatibility with tests
class ToolManager:
    """Simple tool manager for backward compatibility"""

    def __init__(self):
        self.tools = {}
        self.registry = ToolRegistry()

    def register_tool(self, name: str, tool_func):
        """Register a tool function"""
        self.tools[name] = tool_func
        return register_tool(name, tool_func)

    def get_tool(self, name: str):
        """Get a tool by name"""
        return self.tools.get(name)

    def list_tools(self):
        """List all available tools"""
        return list(self.tools.keys())


__all__ = ["register_tool", "ToolRegistry", "ToolManager"]
