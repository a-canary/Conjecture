"""
Pure Tool Registry for the Conjecture skill-based agency system.
Pure functions for tool registration, discovery, and execution coordination.
"""
import os
import sys
import importlib.util
import inspect
import ast
import tempfile
import shutil
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from pathlib import Path
import logging
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime

from .tool_executor import ExecutionLimits, SecurityValidator, ExecutionResult

logger = logging.getLogger(__name__)

@dataclass
class ToolFunction:
    """Pure data representation of a loaded tool function."""
    name: str
    file_path: str
    description: str
    version: str
    parameters: Dict[str, Dict[str, Any]]
    return_type: str
    created_at: Optional[float]
    execution_count: int
    success_count: int
    function: Optional[Callable] = None  # Runtime only field
    
    def get_success_rate(self) -> float:
        """Calculate success rate."""
        if self.execution_count == 0:
            return 0.0
        return self.success_count / self.execution_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding the function object)."""
        data = asdict(self)
        data['success_rate'] = self.get_success_rate()
        data.pop('function', None)  # Remove the function object
        return data

@dataclass
class ToolCall:
    """Pure data representation of a tool call."""
    name: str
    parameters: Dict[str, Any]
    call_id: Optional[str] = None

@dataclass
class ToolRegistry:
    """Pure data structure for tool registry state."""
    tools_directory: str
    tools: Dict[str, ToolFunction]
    execution_limits: ExecutionLimits
    
    def copy_with_updates(self, **updates) -> 'ToolRegistry':
        """Create a copy with updated fields."""
        fields = asdict(self)
        fields.update(updates)
        return ToolRegistry(**fields)

# Pure Functions for Tool Operations

def create_tool_registry(tools_directory: str = "tools",
                        execution_limits: Optional[ExecutionLimits] = None) -> ToolRegistry:
    """Pure function to initialize tool registry."""
    return ToolRegistry(
        tools_directory=tools_directory,
        tools={},
        execution_limits=execution_limits or ExecutionLimits()
    )

def ensure_tools_directory(tools_directory: str) -> str:
    """Pure function to ensure tools directory exists."""
    tools_path = Path(tools_directory)
    tools_path.mkdir(exist_ok=True)
    
    # Create __init__.py to make it a package
    init_file = tools_path / "__init__.py"
    if not init_file.exists():
        init_file.write_text('"""Conjecture Tools Package"""\n')
    
    return str(tools_path)

def extract_function_signature(func: Callable) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """Pure function to extract parameter and return type information."""
    signature = inspect.signature(func)
    parameters = {}
    
    for param_name, param in signature.parameters.items():
        param_info = {
            'name': param_name,
            'required': param.default == inspect.Parameter.empty,
            'default': param.default if param.default != inspect.Parameter.empty else None,
            'type_hint': str(param.annotation) if param.annotation != inspect.Parameter.empty else 'any'
        }
        parameters[param_name] = param_info
    
    return_type = str(signature.return_annotation) if signature.return_annotation != inspect.Signature.empty else 'any'
    return parameters, return_type

def validate_function_parameters(tool_func: ToolFunction, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Pure function to validate parameters against function signature."""
    errors = []
    
    # Check required parameters
    required_params = {name for name, info in tool_func.parameters.items() if info['required']}
    provided_params = set(params.keys())
    
    missing = required_params - provided_params
    for param in missing:
        errors.append(f"Missing required parameter: {param}")
    
    # Check unknown parameters
    unknown = provided_params - set(tool_func.parameters.keys())
    for param in unknown:
        errors.append(f"Unknown parameter: {param}")
    
    # Type validation (basic)
    for param_name, param_value in params.items():
        if param_name in tool_func.parameters:
            param_info = tool_func.parameters[param_name]
            if param_info['type_hint'] != 'any':
                if param_value is None and param_info['required']:
                    errors.append(f"Parameter {param_name} cannot be None")
    
    return len(errors) == 0, errors

def validate_tool_code(code: str, execution_limits: ExecutionLimits) -> Tuple[bool, List[str]]:
    """Pure function to validate tool code for security."""
    security_validator = SecurityValidator(execution_limits)
    return security_validator.validate_code(code)

def parse_tool_functions(code: str) -> List[str]:
    """Pure function to parse Python code and extract function names."""
    try:
        tree = ast.parse(code)
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip private functions
                if not node.name.startswith('_'):
                    functions.append(node.name)
        
        return functions
    except SyntaxError:
        return []

def load_tool_function_from_code(name: str, file_path: str, code: str) -> Optional[ToolFunction]:
    """Pure function to load a tool function from code."""
    try:
        # Parse code to find functions
        function_names = parse_tool_functions(code)
        if not function_names:
            return None
        
        if name not in function_names:
            return None
        
        # Load the module
        spec = importlib.util.spec_from_file_location(
            Path(file_path).stem, file_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get the function
        if not hasattr(module, name):
            return None
        
        func = getattr(module, name)
        if not callable(func):
            return None
        
        # Extract signature information
        parameters, return_type = extract_function_signature(func)
        description = inspect.getdoc(func) or ""
        
        # Create tool function
        return ToolFunction(
            name=name,
            file_path=file_path,
            description=description,
            version="1.0.0",
            parameters=parameters,
            return_type=return_type,
            created_at=Path(file_path).stat().st_mtime if os.path.exists(file_path) else datetime.utcnow().timestamp(),
            execution_count=0,
            success_count=0,
            function=func
        )
    
    except Exception as e:
        logger.error(f"Failed to load tool function {name}: {e}")
        return None

def register_tool_function(registry: ToolRegistry, tool_func: ToolFunction) -> ToolRegistry:
    """Pure function to register a tool function."""
    new_tools = registry.tools.copy()
    new_tools[tool_func.name] = tool_func
    
    return registry.copy_with_updates(tools=new_tools)

def unregister_tool_function(registry: ToolRegistry, name: str) -> ToolRegistry:
    """Pure function to unregister a tool function."""
    new_tools = registry.tools.copy()
    new_tools.pop(name, None)
    
    return registry.copy_with_updates(tools=new_tools)

def get_tool_function(registry: ToolRegistry, name: str) -> Optional[ToolFunction]:
    """Pure function to get a registered tool function."""
    return registry.tools.get(name)

def list_tool_functions(registry: ToolRegistry) -> List[ToolFunction]:
    """Pure function to list all registered tool functions."""
    return list(registry.tools.values())

def search_tool_functions(registry: ToolRegistry, query: str) -> List[ToolFunction]:
    """Pure function to search tool functions by name or description."""
    query_lower = query.lower()
    results = []
    
    for tool_func in registry.tools.values():
        if (query_lower in tool_func.name.lower() or
            query_lower in tool_func.description.lower()):
            results.append(tool_func)
    
    return results

def create_tool_file_content(name: str, code: str, description: str = "") -> str:
    """Pure function to create tool file content."""
    return f'''"""
{description or f"Tool: {name}"}
Auto-generated by Conjecture Tool Creator
"""

{code}
'''

def create_tool_file_path(tools_directory: str, name: str) -> str:
    """Pure function to get tool file path."""
    return str(Path(tools_directory) / f"{name}.py")

def write_tool_file(file_path: str, content: str) -> bool:
    """Pure function to write a tool file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        logger.error(f"Failed to write tool file {file_path}: {e}")
        return False

def load_tool_from_file(registry: ToolRegistry, file_path: str) -> Tuple[ToolRegistry, Optional[ToolFunction]]:
    """Pure function to load a tool from a file and register it."""
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            return registry, None
        
        # Read and validate the file content
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Security validation
        is_safe, errors = validate_tool_code(code, registry.execution_limits)
        if not is_safe:
            logger.error(f"Tool security validation failed: {'; '.join(errors)}")
            return registry, None
        
        # Parse the file to find functions
        function_names = parse_tool_functions(code)
        if not function_names:
            return registry, None
        
        # Load the first found function as the main tool
        tool_func = load_tool_function_from_code(function_names[0], str(file_path), code)
        if tool_func:
            new_registry = register_tool_function(registry, tool_func)
            return new_registry, tool_func
        
        return registry, None
    
    except Exception as e:
        logger.error(f"Failed to load tool from {file_path}: {e}")
        return registry, None

def load_all_tools_from_directory(registry: ToolRegistry) -> Tuple[ToolRegistry, int]:
    """Pure function to load all tools from the tools directory."""
    tools_dir = Path(registry.tools_directory)
    if not tools_dir.exists():
        return registry, 0
    
    current_registry = registry
    loaded_count = 0
    
    for file_path in tools_dir.glob("*.py"):
        if file_path.name != "__init__.py":
            current_registry, tool_func = load_tool_from_file(current_registry, file_path)
            if tool_func:
                loaded_count += 1
    
    return current_registry, loaded_count

def create_and_register_tool(registry: ToolRegistry, name: str, code: str, description: str = "") -> Tuple[ToolRegistry, Optional[ToolFunction], Optional[str]]:
    """Pure function to create and register a new tool."""
    # Validate the code
    is_safe, errors = validate_tool_code(code, registry.execution_limits)
    if not is_safe:
        return registry, None, f"Security validation failed: {'; '.join(errors)}"
    
    # Create file content and path
    content = create_tool_file_content(name, code, description)
    file_path = create_tool_file_path(registry.tools_directory, name)
    
    # Ensure tools directory exists
    ensure_tools_directory(registry.tools_directory)
    
    # Write the file
    if not write_tool_file(file_path, content):
        return registry, None, f"Failed to write tool file: {file_path}"
    
    # Load the tool
    updated_registry, tool_func = load_tool_from_file(registry, file_path)
    return updated_registry, tool_func, None

def delete_tool(registry: ToolRegistry, name: str) -> Tuple[ToolRegistry, bool]:
    """Pure function to delete a tool."""
    # Remove from registry
    updated_registry = unregister_tool_function(registry, name)
    
    # Remove file
    file_path = create_tool_file_path(registry.tools_directory, name)
    deleted = False
    
    if os.path.exists(file_path):
        try:
            os.unlink(file_path)
            deleted = True
        except Exception as e:
            logger.error(f"Failed to delete tool file {file_path}: {e}")
    
    return updated_registry, deleted

def get_tool_file_hash(file_path: str) -> str:
    """Pure function to get hash of tool file for change detection."""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return ""

def validate_tool_function(tool_func: ToolFunction, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Pure function to validate tool function and parameters."""
    # Validate tool function exists
    if not tool_func:
        return False, ["Tool function not found"]
    
    # Validate parameters
    is_valid, errors = validate_function_parameters(tool_func, parameters)
    return is_valid, errors

def get_tool_registry_stats(registry: ToolRegistry) -> Dict[str, Any]:
    """Pure function to get statistics about tool registry."""
    total_tools = len(registry.tools)
    total_executions = sum(tool_func.execution_count for tool_func in registry.tools.values())
    total_successes = sum(tool_func.success_count for tool_func in registry.tools.values())
    
    avg_success_rate = (total_successes / total_executions if total_executions > 0 else 0.0)
    
    # Most used tools
    most_used = sorted(
        registry.tools.values(),
        key=lambda t: t.execution_count,
        reverse=True
    )[:5]
    
    return {
        'total_tools': total_tools,
        'total_executions': total_executions,
        'total_successes': total_successes,
        'average_success_rate': avg_success_rate,
        'most_used_tools': [tool.name for tool in most_used],
        'tools_directory': registry.tools_directory
    }