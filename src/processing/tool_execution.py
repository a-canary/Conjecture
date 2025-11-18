"""
Pure Tool Execution Engine for the Conjecture system.
Pure functions for coordinating tool execution with separate registry.
"""
import time
import inspect
from typing import Dict, List, Any, Optional, Callable
import logging

from .tool_registry import ToolFunction, ToolCall, ToolRegistry
from .tool_executor import ExecutionLimits, ExecutionResult


logger = logging.getLogger(__name__)


def execute_tool_call(tool_call: ToolCall, 
                      tool_func: ToolFunction,
                      execution_limits: Optional[ExecutionLimits] = None) -> ExecutionResult:
    """Pure function to execute a tool call."""
    start_time = time.time()
    
    try:
        # Validate tool function
        if not tool_func.function:
            return ExecutionResult(
                success=False,
                error_message="Tool function object not available",
                execution_time_ms=int((time.time() - start_time) * 1000),
                skill_id=tool_call.name,
                parameters_used=tool_call.parameters
            )
        
        # Validate parameters
        if not validate_tool_parameters(tool_func, tool_call.parameters):
            return ExecutionResult(
                success=False,
                error_message="Parameter validation failed",
                execution_time_ms=int((time.time() - start_time) * 1000),
                skill_id=tool_call.name,
                parameters_used=tool_call.parameters
            )
        
        # Execute the function
        if inspect.iscoroutinefunction(tool_func.function):
            # For async functions, we'd need to handle this at the call site
            # This is a limitation of pure functions - returning a coroutine
            result = tool_func.function(**tool_call.parameters)
            is_async = True
        else:
            result = tool_func.function(**tool_call.parameters)
            is_async = False
        
        execution_time = int((time.time() - start_time) * 1000)
        
        return ExecutionResult(
            success=True,
            result=result,
            execution_time_ms=execution_time,
            skill_id=tool_call.name,
            parameters_used=tool_call.parameters,
            _is_async_result=is_async  # Private field to mark async results
        )
    
    except Exception as e:
        execution_time = int((time.time() - start_time) * 1000)
        logger.error(f"Tool execution error for {tool_call.name}: {e}")
        
        return ExecutionResult(
            success=False,
            error_message=f"Tool execution error: {str(e)}",
            execution_time_ms=execution_time,
            skill_id=tool_call.name,
            parameters_used=tool_call.parameters
        )


def validate_tool_parameters(tool_func: ToolFunction, parameters: Dict[str, Any]) -> bool:
    """Pure function to validate parameters quickly."""
    # Check required parameters
    required_params = {name for name, info in tool_func.parameters.items() if info['required']}
    provided_params = set(parameters.keys())
    
    missing = required_params - provided_params
    if missing:
        return False
    
    # Check unknown parameters
    unknown = provided_params - set(tool_func.parameters.keys())
    if unknown:
        return False
    
    return True


def create_tool_call(name: str, parameters: Dict[str, Any], call_id: Optional[str] = None) -> ToolCall:
    """Pure function to create a tool call."""
    return ToolCall(
        name=name,
        parameters=parameters.copy(),
        call_id=call_id
    )


def validate_tool_call(tool_call: ToolCall, registry: ToolRegistry) -> tuple[bool, List[str]]:
    """Pure function to validate a tool call against registry."""
    errors = []
    
    # Check if tool exists
    tool_func = get_tool_function(registry, tool_call.name)
    if not tool_func:
        errors.append(f"Tool '{tool_call.name}' not found in registry")
        return False, errors
    
    # Validate parameters
    is_valid, param_errors = validate_function_parameters(tool_func, tool_call.parameters)
    errors.extend(param_errors)
    
    return len(errors) == 0, errors


def execute_tool_from_registry(tool_call: ToolCall, 
                              registry: ToolRegistry,
                              execution_limits: Optional[ExecutionLimits] = None) -> ExecutionResult:
    """Pure function to execute a tool call from registry."""
    # Get the tool function
    tool_func = get_tool_function(registry, tool_call.name)
    if not tool_func:
        return ExecutionResult(
            success=False,
            error_message=f"Tool '{tool_call.name}' not found in registry",
            execution_time_ms=0,
            skill_id=tool_call.name,
            parameters_used=tool_call.parameters
        )
    
    # Validate the call
    is_valid, errors = validate_tool_call(tool_call, registry)
    if not is_valid:
        return ExecutionResult(
            success=False,
            error_message=f"Tool call validation failed: {'; '.join(errors)}",
            execution_time_ms=0,
            skill_id=tool_call.name,
            parameters_used=tool_call.parameters
        )
    
    # Execute the tool
    return execute_tool_call(tool_call, tool_func, execution_limits)


def batch_execute_tool_calls(tool_calls: List[ToolCall], 
                            registry: ToolRegistry,
                            execution_limits: Optional[ExecutionLimits] = None) -> List[ExecutionResult]:
    """Pure function to execute multiple tool calls."""
    results = []
    
    for tool_call in tool_calls:
        result = execute_tool_from_registry(tool_call, registry, execution_limits)
        results.append(result)
    
    return results


def create_execution_summary(results: List[ExecutionResult]) -> Dict[str, Any]:
    """Pure function to create execution summary from results."""
    total_calls = len(results)
    successful_calls = sum(1 for result in results if result.success)
    failed_calls = total_calls - successful_calls
    
    execution_times = [result.execution_time_ms for result in results if result.execution_time_ms]
    avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
    
    # Tool usage statistics
    tool_counts = {}
    for result in results:
        tool_counts[result.skill_id] = tool_counts.get(result.skill_id, 0) + 1
    
    return {
        'total_calls': total_calls,
        'successful_calls': successful_calls,
        'failed_calls': failed_calls,
        'success_rate': successful_calls / total_calls if total_calls > 0 else 0.0,
        'average_execution_time_ms': avg_execution_time,
        'tool_usage': tool_counts,
        'errors': [result.error_message for result in results if not result.success]
    }


async def execute_async_tool_calls(tool_calls: List[ToolCall], 
                                  registry: ToolRegistry,
                                  execution_limits: Optional[ExecutionLimits] = None) -> List[ExecutionResult]:
    """Async function to execute multiple tool calls with concurrency."""
    import asyncio
    
    results = []
    tasks = []
    
    for tool_call in tool_calls:
        # Get the tool function
        tool_func = get_tool_function(registry, tool_call.name)
        if not tool_func:
            result = ExecutionResult(
                success=False,
                error_message=f"Tool '{tool_call.name}' not found in registry",
                execution_time_ms=0,
                skill_id=tool_call.name,
                parameters_used=tool_call.parameters
            )
            results.append(result)
            continue
        
        # Validate the call
        is_valid, errors = validate_tool_call(tool_call, registry)
        if not is_valid:
            result = ExecutionResult(
                success=False,
                error_message=f"Tool call validation failed: {'; '.join(errors)}",
                execution_time_ms=0,
                skill_id=tool_call.name,
                parameters_used=tool_call.parameters
            )
            results.append(result)
            continue
        
        # Create async task
        task = asyncio.create_task(_execute_async_tool_call(tool_call, tool_func, execution_limits))
        tasks.append((tool_call.name, task))
    
    # Wait for all tasks
    for tool_name, task in tasks:
        try:
            result = await task
            results.append(result)
        except Exception as e:
            logger.error(f"Async tool execution error for {tool_name}: {e}")
            results.append(ExecutionResult(
                success=False,
                error_message=f"Async execution error: {str(e)}",
                execution_time_ms=0,
                skill_id=tool_name,
                parameters_used={}
            ))
    
    return results


async def _execute_async_tool_call(tool_call: ToolCall, 
                                  tool_func: ToolFunction,
                                  execution_limits: Optional[ExecutionLimits] = None) -> ExecutionResult:
    """Internal async function to execute a tool call."""
    start_time = time.time()
    
    try:
        # Execute the async function
        result = await tool_func.function(**tool_call.parameters)
        
        execution_time = int((time.time() - start_time) * 1000)
        
        return ExecutionResult(
            success=True,
            result=result,
            execution_time_ms=execution_time,
            skill_id=tool_call.name,
            parameters_used=tool_call.parameters
        )
    
    except Exception as e:
        execution_time = int((time.time() - start_time) * 1000)
        
        return ExecutionResult(
            success=False,
            error_message=f"Async tool execution error: {str(e)}",
            execution_time_ms=execution_time,
            skill_id=tool_call.name,
            parameters_used=tool_call.parameters
        )


def get_available_tools(registry: ToolRegistry) -> List[Dict[str, Any]]:
    """Pure function to get available tools as list of dictionaries."""
    return [tool.to_dict() for tool in registry.tools.values()]


def get_tool_by_purpose(registry: ToolRegistry, purpose: str, limit: int = 5) -> List[ToolFunction]:
    """Pure function to find tools by purpose/description."""
    purpose_lower = purpose.lower()
    matches = []
    
    for tool_func in registry.tools.values():
        # Search in name and description
        if (purpose_lower in tool_func.name.lower() or
            purpose_lower in tool_func.description.lower()):
            matches.append(tool_func)
    
    # Return top matches by relevance (could be enhanced with better scoring)
    matches.sort(key=lambda t: (
        purpose_lower in t.name.lower() * 2 + 
        purpose_lower in t.description.lower()
    ), reverse=True)
    
    return matches[:limit]


# Import from tool_registry for pure functions
from .tool_registry import get_tool_function, validate_function_parameters