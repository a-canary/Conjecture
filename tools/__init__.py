"""
Tools package for Conjecture.

This package contains all available tools for the Conjecture system.
Each tool module should be imported to register its functions with the ToolRegistry.
"""

# Import the registry system
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.tools.registry import register_tool

# Import all tool modules to register their functions
from . import webSearch
from . import readFiles
from . import writeFiles
from . import apply_diff

# Ensure registry auto-discovers tools from this directory
from src.tools.registry import ToolRegistry
ToolRegistry.auto_discover_tools(os.path.dirname(__file__))

__all__ = ['ToolRegistry', 'register_tool']