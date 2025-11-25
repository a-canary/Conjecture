"""
Simple LLM Processor for Conjecture

A simplified LLM processor that focuses on tool execution instead of 
complex instruction parsing. Expects JSON tool calls and executes them
through the ToolRegistry system.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

from ..tools.registry import ToolRegistry, get_tool_registry
from ..interfaces.llm_interface import LLMInterface


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""
    name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None


@dataclass
class ToolResult:
    """Represents the result of a tool execution."""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time_ms: Optional[int] = None


class SimpleLLMProcessor:
    """
    Simplified LLM processor that handles direct tool execution.
    
    This processor:
    1. Takes context and sends to LLM
    2. Expects JSON response with tool calls
    3. Executes tool calls through ToolRegistry
    4. Returns structured results
    """
    
    def __init__(self, llm_interface: LLMInterface, max_tool_calls_per_request: int = 10):
        """
        Initialize the processor.
        
        Args:
            llm_interface: The LLM interface to use
            max_tool_calls_per_request: Maximum tool calls to execute per request
        """
        self.llm_interface = llm_interface
        self.max_tool_calls_per_request = max_tool_calls_per_request
        self.logger = logging.getLogger(__name__)
        
        # Ensure tools are loaded
        self.tool_registry = ToolRegistry()
    
    def process_request(self, context: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a request with context and expect tool calls.
        
        Args:
            context: The context to send to the LLM
            request_id: Optional request ID for tracking
            
        Returns:
            Dictionary with processing results
        """
        import time
        start_time = time.time()
        
        try:
            # Send context to LLM
            self.logger.info(f"Sending context to LLM (request: {request_id})")
            llm_response = self.llm_interface.generate_response(context)
            
            if not llm_response.strip():
                return {
                    'success': False,
                    'error': 'Empty response from LLM',
                    'request_id': request_id,
                    'execution_time_ms': int((time.time() - start_time) * 1000)
                }
            
            # Parse tool calls from response
            tool_calls = self._parse_tool_calls(llm_response)
            
            if not tool_calls:
                return {
                    'success': True,
                    'message': 'No tool calls found in LLM response',
                    'llm_response': llm_response,
                    'tool_calls': [],
                    'tool_results': [],
                    'request_id': request_id,
                    'execution_time_ms': int((time.time() - start_time) * 1000)
                }
            
            # Execute tool calls
            tool_results = self._execute_tool_calls(tool_calls)
            
            return {
                'success': True,
                'llm_response': llm_response,
                'tool_calls': [
                    {
                        'name': tc.name,
                        'arguments': tc.arguments,
                        'call_id': tc.call_id
                    } for tc in tool_calls
                ],
                'tool_results': [
                    {
                        'tool_name': tr.tool_name,
                        'success': tr.success,
                        'result': tr.result,
                        'error': tr.error,
                        'execution_time_ms': tr.execution_time_ms
                    } for tr in tool_results
                ],
                'total_tool_calls': len(tool_calls),
                'successful_calls': sum(1 for tr in tool_results if tr.success),
                'failed_calls': sum(1 for tr in tool_results if not tr.success),
                'request_id': request_id,
                'execution_time_ms': int((time.time() - start_time) * 1000)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing request: {str(e)}")
            return {
                'success': False,
                'error': f'Processing failed: {str(e)}',
                'request_id': request_id,
                'execution_time_ms': int((time.time() - start_time) * 1000)
            }
    
    def _parse_tool_calls(self, llm_response: str) -> List[ToolCall]:
        """
        Parse tool calls from LLM response.
        
        Expects JSON format like:
        {"tool_calls": [{"name": "ToolName", "arguments": {...}}]}
        
        Args:
            llm_response: The raw response from the LLM
            
        Returns:
            List of parsed tool calls
        """
        tool_calls = []
        
        try:
            # Try to parse as JSON directly
            response_data = json.loads(llm_response.strip())
            
            if isinstance(response_data, dict) and 'tool_calls' in response_data:
                tool_calls_data = response_data['tool_calls']
                
                if isinstance(tool_calls_data, list):
                    for i, call_data in enumerate(tool_calls_data[:self.max_tool_calls_per_request]):
                        if isinstance(call_data, dict):
                            name = call_data.get('name')
                            arguments = call_data.get('arguments', {})
                            call_id = call_data.get('call_id', f"call_{i}")
                            
                            if name and isinstance(arguments, dict):
                                tool_calls.append(ToolCall(name, arguments, call_id))
                            else:
                                self.logger.warning(f"Invalid tool call format: {call_data}")
            
        except json.JSONDecodeError:
            # Try to extract JSON from text response
            json_match = self._extract_json_from_text(llm_response)
            if json_match:
                try:
                    response_data = json.loads(json_match)
                    if isinstance(response_data, dict) and 'tool_calls' in response_data:
                        return self._parse_tool_calls(json.dumps(response_data))
                except json.JSONDecodeError:
                    self.logger.warning("Failed to parse extracted JSON")
            
            self.logger.warning(f"Could not parse tool calls from LLM response: {llm_response[:200]}...")
        
        return tool_calls
    
    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """Extract JSON object from text using regex."""
        import re
        
        # Look for JSON object patterns
        json_patterns = [
            r'\{[^{}]*"tool_calls"[^{}]*\}',  # Objects with tool_calls
            r'\{[^{}]*"name"[^{}]*"arguments"[^{}]*\}',  # Objects with name/arguments
            r'\{.*?\}'  # Any JSON object
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    json.loads(match)  # Validate JSON
                    return match
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _execute_tool_calls(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """
        Execute a list of tool calls.
        
        Args:
            tool_calls: List of tool calls to execute
            
        Returns:
            List of tool execution results
        """
        results = []
        
        for tool_call in tool_calls:
            result = self._execute_single_tool_call(tool_call)
            results.append(result)
        
        return results
    
    def _execute_single_tool_call(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a single tool call.
        
        Args:
            tool_call: The tool call to execute
            
        Returns:
            Tool result
        """
        import time
        start_time = time.time()
        
        try:
            # Validate tool exists
            registry = get_tool_registry()
            tool_info = registry.get_tool_info(tool_call.name)
            if tool_info is None:
                return ToolResult(
                    tool_name=tool_call.name,
                    success=False,
                    result=None,
                    error=f"Tool '{tool_call.name}' not found",
                    execution_time_ms=int((time.time() - start_time) * 1000)
                )
            
            # Validate arguments
            if not isinstance(tool_call.arguments, dict):
                return ToolResult(
                    tool_name=tool_call.name,
                    success=False,
                    result=None,
                    error="Tool arguments must be a dictionary",
                    execution_time_ms=int((time.time() - start_time) * 1000)
                )
            
            # Execute tool
            self.logger.info(f"Executing tool: {tool_call.name} with args: {tool_call.arguments}")
            registry = get_tool_registry()
            tool_result = registry.execute_tool(tool_call.name, tool_call.arguments)
            
            return ToolResult(
                tool_name=tool_call.name,
                success=True,
                result=tool_result,
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
            
        except Exception as e:
            self.logger.error(f"Error executing tool {tool_call.name}: {str(e)}")
            return ToolResult(
                tool_name=tool_call.name,
                success=False,
                result=None,
                error=f"Tool execution failed: {str(e)}",
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
    
    def get_available_tools(self) -> Dict[str, Any]:
        """
        Get information about available tools.
        
        Returns:
            Dictionary with core and optional tools information
        """
        registry = ToolRegistry()
        core_tools = registry.core_tools
        optional_tools = registry.optional_tools
        
        return {
            'core_tools': {
                name: {
                    'name': name,
                    'description': info.description,
                    'signature': info.signature
                } for name, info in core_tools.items()
            },
            'optional_tools': {
                name: {
                    'name': name,
                    'description': info.description,
                    'signature': info.signature
                } for name, info in optional_tools.items()
            },
            'total_core_tools': len(core_tools),
            'total_optional_tools': len(optional_tools),
            'total_tools': len(core_tools) + len(optional_tools)
        }
    
    def validate_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a tool call without executing it.
        
        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            
        Returns:
            Validation result
        """
        try:
            registry = ToolRegistry()
            tool_info = registry.get_tool_info(tool_name)
            if tool_info is None:
                return {
                    'valid': False,
                    'error': f"Tool '{tool_name}' not found"
                }
            
            if not isinstance(arguments, dict):
                return {
                    'valid': False,
                    'error': 'Arguments must be a dictionary'
                }
            
            # For now, just validate basic structure
            
            return {
                'valid': True,
                'tool_name': tool_name,
                'tool_signature': tool_info.signature
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f'Validation failed: {str(e)}'
            }


def create_simple_response(tool_calls: List[Dict[str, Any]]) -> str:
    """
    Create a simple JSON response for tool calls.
    
    Args:
        tool_calls: List of tool call dictionaries
        
    Returns:
        JSON string response
    """
    response = {
        'tool_calls': tool_calls
    }
    return json.dumps(response, indent=2)


def examples() -> List[str]:
    """Return example usage patterns for the SimpleLLMProcessor."""
    return [
        "process_request(context) processes context and executes returned tool calls",
        "get_available_tools() returns list of all available core and optional tools",
        "validate_tool_call('WebSearch', {'query': 'Rust tutorial'}) validates tool call parameters",
        "create_simple_response([{'name': 'WebSearch', 'arguments': {'query': 'test'}}]) creates JSON response format"
    ]


if __name__ == "__main__":
    # Test the SimpleLLMProcessor with mock LLM
    from ..interfaces.llm_interface import LLMInterface
    
    class MockLLM(LLMInterface):
        def generate_response(self, prompt: str) -> str:
            # Mock response with tool calls
            mock_tool_calls = [
                {
                    'name': 'WebSearch',
                    'arguments': {'query': 'Rust game development', 'max_results': 5},
                    'call_id': 'test_call_1'
                }
            ]
            return json.dumps({'tool_calls': mock_tool_calls})
    
    print("Testing SimpleLLMProcessor...")
    
    # Create processor
    processor = SimpleLLMProcessor(MockLLM())
    
    # Test getting available tools
    tools_info = processor.get_available_tools()
    print(f"\nAvailable tools: {tools_info['total_tools']} total")
    print(f"Core tools: {tools_info['total_core_tools']}")
    print(f"Optional tools: {tools_info['total_optional_tools']}")
    
    # Test validation
    validation = processor.validate_tool_call('WebSearch', {'query': 'test'})
    print(f"\nTool validation: {validation['valid']}")
    
    # Test creating simple response
    response_json = create_simple_response([{'name': 'Reason', 'arguments': {'thought_process': 'test'}}])
    print(f"\nSample response format:\n{response_json}")
    
    print("\nExamples:")
    for example in examples():
        print(f"- {example}")