"""
Core Tools Registry System
Manages tool registration, discovery, and context generation
"""

import inspect
import importlib.util
from typing import Dict, List, Any, Callable, Optional
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ToolInfo:
    """Information about a registered tool"""
    name: str
    func: Callable
    description: str
    signature: str
    is_core: bool
    module: str


class ToolRegistry:
    """Registry for managing tools and their availability"""
    
    def __init__(self):
        self.core_tools: Dict[str, ToolInfo] = {}
        self.optional_tools: Dict[str, ToolInfo] = {}
        self._discover_tools()
    
    def _discover_tools(self):
        """Auto-discover tools from tools/ directory"""
        tools_dir = Path(__file__).parent.parent.parent / "tools"
        
        if not tools_dir.exists():
            print(f"Warning: tools directory not found at {tools_dir}")
            return
        
        for tool_file in tools_dir.glob("*.py"):
            if tool_file.name.startswith("__"):
                continue
                
            try:
                self._load_tool_module(tool_file)
            except Exception as e:
                print(f"Warning: Failed to load tool module {tool_file}: {e}")
    
    def _load_tool_module(self, tool_file: Path):
        """Load a tool module and register its tools"""
        spec = importlib.util.spec_from_file_location(
            tool_file.stem, tool_file
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Look for registered tools in the module
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if hasattr(attr, '_tool_info'):
                tool_info = attr._tool_info
                self._register_tool(tool_info)
    
    def _register_tool(self, tool_info: ToolInfo):
        """Register a tool in the appropriate registry"""
        if tool_info.is_core:
            self.core_tools[tool_info.name] = tool_info
        else:
            self.optional_tools[tool_info.name] = tool_info
        print(f"Registered {'Core' if tool_info.is_core else 'Optional'} tool: {tool_info.name}")
    
    def get_core_tools_context(self) -> str:
        """Generate Core Tools section for LLM context"""
        if not self.core_tools:
            return "# Core Tools\n\nNo core tools available."
        
        context_parts = ["# Core Tools"]
        
        for tool_name, tool_info in sorted(self.core_tools.items()):
            context_parts.append(f"**{tool_name}{tool_info.signature}**: {tool_info.description}")
        
        return "\n".join(context_parts)
    
    def get_available_tools_list(self) -> List[str]:
        """Get list of all available tool names"""
        all_tools = list(self.core_tools.keys()) + list(self.optional_tools.keys())
        return sorted(all_tools)
    
    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with given arguments"""
        # Check core tools first
        if tool_name in self.core_tools:
            tool_info = self.core_tools[tool_name]
        elif tool_name in self.optional_tools:
            tool_info = self.optional_tools[tool_name]
        else:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found",
                "available_tools": self.get_available_tools_list()
            }
        
        try:
            # Validate arguments
            sig = inspect.signature(tool_info.func)
            bound_args = sig.bind(**args)
            bound_args.apply_defaults()
            
            # Execute the tool
            result = tool_info.func(**bound_args.arguments)
            
            return {
                "success": True,
                "result": result,
                "tool_name": tool_name
            }
            
        except TypeError as e:
            return {
                "success": False,
                "error": f"Argument error for {tool_name}: {str(e)}",
                "expected_signature": tool_info.signature
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Execution error for {tool_name}: {str(e)}"
            }
    
    def get_tool_info(self, tool_name: str) -> Optional[ToolInfo]:
        """Get information about a specific tool"""
        if tool_name in self.core_tools:
            return self.core_tools[tool_name]
        elif tool_name in self.optional_tools:
            return self.optional_tools[tool_name]
        return None


def register_tool(is_core: bool = False, name: Optional[str] = None):
    """Decorator to register a function as a tool"""
    def decorator(func: Callable) -> Callable:
        # Extract tool information
        sig = inspect.signature(func)
        signature_str = str(sig)

        # Get description from docstring
        description = "No description available"
        if func.__doc__:
            description = func.__doc__.strip().split('\n')[0]

        # Use provided name or function name
        tool_name = name or func.__name__

        # Create tool info
        tool_info = ToolInfo(
            name=tool_name,
            func=func,
            description=description,
            signature=signature_str,
            is_core=is_core,
            module=func.__module__
        )

        # Store tool info on the function
        func._tool_info = tool_info

        return func
    
    return decorator


# Global registry instance
_tool_registry = None

def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry instance"""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry