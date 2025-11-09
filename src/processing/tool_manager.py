"""
Tool System for the Conjecture skill-based agency system.
Manages dynamic Python function tools with security and resource limitations.
"""
import os
import sys
import importlib.util
import inspect
import ast
import tempfile
import shutil
from typing import Dict, List, Any, Optional, Callable, Tuple
from pathlib import Path
import logging
import hashlib

from .tool_executor import ExecutionLimits, SecurityValidator


logger = logging.getLogger(__name__)


class Tool:
    """Represents a dynamically loaded Python tool."""
    
    def __init__(self, name: str, file_path: str, function: Callable, 
                 description: str = "", version: str = "1.0.0"):
        self.name = name
        self.file_path = file_path
        self.function = function
        self.description = description
        self.version = version
        self.signature = inspect.signature(function)
        self.parameters = self._extract_parameters()
        self.return_type = self._extract_return_type()
        self.created_at = Path(file_path).stat().st_mtime if os.path.exists(file_path) else None
        self.execution_count = 0
        self.success_count = 0
    
    def _extract_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Extract parameter information from function signature."""
        parameters = {}
        
        for param_name, param in self.signature.parameters.items():
            param_info = {
                'name': param_name,
                'required': param.default == inspect.Parameter.empty,
                'default': param.default if param.default != inspect.Parameter.empty else None,
                'type_hint': str(param.annotation) if param.annotation != inspect.Parameter.empty else 'any'
            }
            parameters[param_name] = param_info
        
        return parameters
    
    def _extract_return_type(self) -> str:
        """Extract return type from function signature."""
        if self.signature.return_annotation == inspect.Signature.empty:
            return 'any'
        return str(self.signature.return_annotation)
    
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate parameters against function signature."""
        errors = []
        
        # Check required parameters
        required_params = {name for name, info in self.parameters.items() if info['required']}
        provided_params = set(params.keys())
        
        missing = required_params - provided_params
        for param in missing:
            errors.append(f"Missing required parameter: {param}")
        
        # Check unknown parameters
        unknown = provided_params - set(self.parameters.keys())
        for param in unknown:
            errors.append(f"Unknown parameter: {param}")
        
        # Type validation (basic)
        for param_name, param_value in params.items():
            if param_name in self.parameters:
                param_info = self.parameters[param_name]
                # Basic type checking could be enhanced here
                if param_info['type_hint'] != 'any':
                    # For now, just ensure the value exists
                    if param_value is None and param_info['required']:
                        errors.append(f"Parameter {param_name} cannot be None")
        
        return len(errors) == 0, errors
    
    async def execute(self, parameters: Dict[str, Any]) -> Any:
        """Execute the tool with given parameters."""
        try:
            # Validate parameters
            is_valid, errors = self.validate_parameters(parameters)
            if not is_valid:
                raise ValueError(f"Parameter validation failed: {'; '.join(errors)}")
            
            # Execute the function
            if inspect.iscoroutinefunction(self.function):
                result = await self.function(**parameters)
            else:
                result = self.function(**parameters)
            
            # Update statistics
            self.execution_count += 1
            self.success_count += 1
            
            return result
            
        except Exception as e:
            self.execution_count += 1
            logger.error(f"Tool {self.name} execution failed: {e}")
            raise
    
    def get_success_rate(self) -> float:
        """Calculate success rate."""
        if self.execution_count == 0:
            return 0.0
        return self.success_count / self.execution_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary representation."""
        return {
            'name': self.name,
            'file_path': self.file_path,
            'description': self.description,
            'version': self.version,
            'parameters': self.parameters,
            'return_type': self.return_type,
            'execution_count': self.execution_count,
            'success_count': self.success_count,
            'success_rate': self.get_success_rate(),
            'created_at': self.created_at
        }


class ToolManager:
    """Manages dynamic loading and execution of Python tools."""
    
    def __init__(self, tools_directory: str = "tools", 
                 execution_limits: Optional[ExecutionLimits] = None):
        self.tools_directory = Path(tools_directory)
        self.tools_directory.mkdir(exist_ok=True)
        self.execution_limits = execution_limits or ExecutionLimits()
        self.security_validator = SecurityValidator(self.execution_limits)
        self.loaded_tools: Dict[str, Tool] = {}
        self._ensure_tools_directory()
    
    def _ensure_tools_directory(self) -> None:
        """Ensure tools directory exists with proper structure."""
        # Create __init__.py to make it a package
        init_file = self.tools_directory / "__init__.py"
        if not init_file.exists():
            init_file.write_text('"""Conjecture Tools Package"""\n')
        
        # Add tools directory to Python path if not already there
        tools_path_str = str(self.tools_directory.parent)
        if tools_path_str not in sys.path:
            sys.path.insert(0, tools_path_str)
    
    async def load_tool_from_file(self, file_path: str) -> Optional[Tool]:
        """Load a tool from a Python file."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"Tool file not found: {file_path}")
                return None
            
            # Read and validate the file content
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Security validation
            is_safe, errors = self.security_validator.validate_code(code)
            if not is_safe:
                logger.error(f"Tool security validation failed: {'; '.join(errors)}")
                return None
            
            # Parse the file to find functions
            tree = ast.parse(code)
            functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Skip private functions
                    if not node.name.startswith('_'):
                        functions.append(node.name)
            
            if not functions:
                logger.warning(f"No public functions found in {file_path}")
                return None
            
            # Load the module
            spec = importlib.util.spec_from_file_location(
                file_path.stem, file_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Create tool objects for each function
            tools = []
            for func_name in functions:
                if hasattr(module, func_name):
                    func = getattr(module, func_name)
                    if callable(func):
                        tool = Tool(
                            name=func_name,
                            file_path=str(file_path),
                            function=func,
                            description=inspect.getdoc(func) or "",
                            version="1.0.0"
                        )
                        tools.append(tool)
            
            # Register tools
            for tool in tools:
                self.loaded_tools[tool.name] = tool
                logger.info(f"Loaded tool: {tool.name} from {file_path}")
            
            return tools[0] if tools else None
            
        except Exception as e:
            logger.error(f"Failed to load tool from {file_path}: {e}")
            return None
    
    async def load_all_tools(self) -> int:
        """Load all tools from the tools directory."""
        loaded_count = 0
        
        for file_path in self.tools_directory.glob("*.py"):
            if file_path.name != "__init__.py":
                tool = await self.load_tool_from_file(file_path)
                if tool:
                    loaded_count += 1
        
        logger.info(f"Loaded {loaded_count} tools from {self.tools_directory}")
        return loaded_count
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a loaded tool by name."""
        return self.loaded_tools.get(name)
    
    def list_tools(self) -> List[Tool]:
        """List all loaded tools."""
        return list(self.loaded_tools.values())
    
    def search_tools(self, query: str) -> List[Tool]:
        """Search tools by name or description."""
        query_lower = query.lower()
        results = []
        
        for tool in self.loaded_tools.values():
            if (query_lower in tool.name.lower() or 
                query_lower in tool.description.lower()):
                results.append(tool)
        
        return results
    
    async def create_tool_file(self, name: str, code: str, 
                             description: str = "") -> Optional[str]:
        """
        Create a new tool file with the given code.
        
        Args:
            name: Tool name (will become filename)
            code: Python code for the tool
            description: Optional description
            
        Returns:
            Path to created file or None if failed
        """
        try:
            # Validate the code
            is_safe, errors = self.security_validator.validate_code(code)
            if not is_safe:
                logger.error(f"Cannot create tool {name}: {'; '.join(errors)}")
                return None
            
            # Create file path
            file_path = self.tools_directory / f"{name}.py"
            
            # Prepare file content with header
            file_content = f'''"""
{description or f"Tool: {name}"}
Auto-generated by Conjecture Tool Creator
"""

{code}
'''
            
            # Write the file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(file_content)
            
            logger.info(f"Created tool file: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to create tool {name}: {e}")
            return None
    
    async def update_tool_file(self, name: str, code: str, 
                             description: str = "") -> bool:
        """Update an existing tool file."""
        file_path = self.tools_directory / f"{name}.py"
        
        if not file_path.exists():
            logger.error(f"Tool {name} does not exist")
            return False
        
        # Create new version
        new_file_path = await self.create_tool_file(name, code, description)
        if new_file_path:
            # Reload the tool
            await self.load_tool_from_file(new_file_path)
            return True
        
        return False
    
    def delete_tool(self, name: str) -> bool:
        """Delete a tool."""
        try:
            # Remove from loaded tools
            if name in self.loaded_tools:
                del self.loaded_tools[name]
            
            # Remove file
            file_path = self.tools_directory / f"{name}.py"
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted tool: {name}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete tool {name}: {e}")
            return False
    
    async def execute_tool(self, name: str, parameters: Dict[str, Any]) -> Any:
        """Execute a tool by name."""
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")
        
        return await tool.execute(parameters)
    
    def get_tool_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded tools."""
        total_tools = len(self.loaded_tools)
        total_executions = sum(tool.execution_count for tool in self.loaded_tools.values())
        total_successes = sum(tool.success_count for tool in self.loaded_tools.values())
        
        avg_success_rate = (total_successes / total_executions 
                           if total_executions > 0 else 0.0)
        
        # Most used tools
        most_used = sorted(
            self.loaded_tools.values(),
            key=lambda t: t.execution_count,
            reverse=True
        )[:5]
        
        return {
            'total_tools': total_tools,
            'total_executions': total_executions,
            'total_successes': total_successes,
            'average_success_rate': avg_success_rate,
            'most_used_tools': [tool.name for tool in most_used],
            'tools_directory': str(self.tools_directory)
        }
    
    def validate_tool_file(self, file_path: str) -> Tuple[bool, List[str]]:
        """Validate a tool file without loading it."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            return self.security_validator.validate_code(code)
            
        except Exception as e:
            return False, [f"File validation error: {e}"]
    
    def get_tool_file_hash(self, file_path: str) -> str:
        """Get hash of tool file for change detection."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
    
    async def reload_changed_tools(self) -> int:
        """Reload tools that have been modified."""
        reloaded_count = 0
        
        for file_path in self.tools_directory.glob("*.py"):
            if file_path.name == "__init__.py":
                continue
            
            # Check if any loaded tools are from this file
            file_tools = [
                tool for tool in self.loaded_tools.values()
                if Path(tool.file_path) == file_path
            ]
            
            if file_tools:
                # Reload the file
                tool = await self.load_tool_from_file(file_path)
                if tool:
                    reloaded_count += 1
        
        return reloaded_count