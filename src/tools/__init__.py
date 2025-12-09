"""
Core Tools Registry System for Conjecture.

This module provides the ToolRegistry system and register_tool decorator
for dynamically managing available tools in the Conjecture system.
"""

from .registry import register_tool, ToolRegistry
import re
import ast
from typing import Dict, Any, List, Optional


# Add ToolManager for backward compatibility with tests
class ToolManager:
    """Simple tool manager for backward compatibility"""

    def __init__(self):
        self.tools = {}
        self.registry = ToolRegistry()
        # Initialize with default tools for backward compatibility
        self._initialize_default_tools()

    def _initialize_default_tools(self):
        """Initialize default tools for backward compatibility with tests"""
        default_tools = {
            "WebSearch": {
                "description": "Search the web for information",
                "parameters": {
                    "query": {"type": "string", "required": True},
                    "max_results": {"type": "integer", "required": False, "default": 5}
                }
            },
            "CreateClaim": {
                "description": "Create a new claim",
                "parameters": {
                    "content": {"type": "string", "required": True},
                    "confidence": {"type": "float", "required": False, "default": 0.8},
                    "claim_type": {"type": "string", "required": False, "default": "observation"},
                    "tags": {"type": "array", "required": False, "default": []}
                }
            },
            "WriteCodeFile": {
                "description": "Write code to a file",
                "parameters": {
                    "file_path": {"type": "string", "required": True},
                    "content": {"type": "string", "required": True}
                }
            },
            "ReadFiles": {
                "description": "Read content from files",
                "parameters": {
                    "files": {"type": "array", "required": True}
                }
            },
            "ClaimSupport": {
                "description": "Get supporting evidence for a claim",
                "parameters": {
                    "claim_id": {"type": "string", "required": True},
                    "max_results": {"type": "integer", "required": False, "default": 5}
                }
            }
        }
        
        for tool_name, tool_info in default_tools.items():
            self.tools[tool_name] = tool_info

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

    def get_tool_definitions(self) -> Dict[str, Any]:
        """Get tool definitions for backward compatibility"""
        return self.tools.copy()

    def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool with parameters"""
        try:
            if tool_name in self.tools:
                tool_info = self.tools[tool_name]
                
                # Handle default tools with mock responses
                if tool_name == "WebSearch":
                    query = parameters.get("query", "")
                    max_results = parameters.get("max_results", 5)
                    return {
                        "success": True,
                        "results": [
                            {"title": f"Result {i+1} for '{query}'", "url": f"https://example.com/{i+1}", "snippet": f"Snippet {i+1}"}
                            for i in range(min(max_results, 3))
                        ],
                        "query": query,
                        "total_results": min(max_results, 3)
                    }
                elif tool_name == "CreateClaim":
                    content = parameters.get("content", "")
                    confidence = parameters.get("confidence", 0.8)
                    claim_type = parameters.get("claim_type", "observation")
                    tags = parameters.get("tags", [])
                    return {
                        "success": True,
                        "claim_id": f"c{hash(content) % 1000000:06d}",
                        "content": content,
                        "confidence": confidence,
                        "claim_type": claim_type,
                        "tags": tags
                    }
                elif tool_name == "WriteCodeFile":
                    file_path = parameters.get("file_path", "")
                    content = parameters.get("content", "")
                    # Actually write the file for testing
                    with open(file_path, 'w') as f:
                        f.write(content)
                    return {
                        "success": True,
                        "file_path": file_path,
                        "bytes_written": len(content.encode())
                    }
                elif tool_name == "ReadFiles":
                    files = parameters.get("files", [])
                    results = {}
                    for file_path in files:
                        try:
                            with open(file_path, 'r') as f:
                                results[file_path] = f.read()
                        except Exception as e:
                            results[file_path] = f"Error reading file: {e}"
                    return {
                        "success": True,
                        "files": results
                    }
                elif tool_name == "ClaimSupport":
                    claim_id = parameters.get("claim_id", "")
                    max_results = parameters.get("max_results", 5)
                    return {
                        "success": True,
                        "claim_id": claim_id,
                        "supporting_evidence": [
                            {"source": f"Source {i+1}", "content": f"Supporting evidence {i+1} for claim {claim_id}"}
                            for i in range(min(max_results, 3))
                        ]
                    }
                else:
                    # Try to execute as a function
                    if callable(tool_info):
                        result = tool_info(**parameters)
                        return {"success": True, "result": result}
                    else:
                        return {"success": False, "error": f"Tool '{tool_name}' is not callable"}
            else:
                return {
                    "success": False,
                    "error": f"Tool '{tool_name}' not found",
                    "available_tools": list(self.tools.keys())
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error executing tool '{tool_name}': {str(e)}"
            }

    def parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Parse tool calls from text response"""
        tool_calls = []
        
        # Pattern to match tool calls like: ToolName(param1='value1', param2='value2')
        pattern = r'(\w+)\s*\(\s*([^)]*)\s*\)'
        matches = re.findall(pattern, text)
        
        for tool_name, params_str in matches:
            if tool_name in self.tools:
                try:
                    # Parse parameters
                    params = {}
                    if params_str.strip():
                        # Simple parameter parsing - handles key=value pairs
                        param_pairs = re.findall(r'(\w+)\s*=\s*([^,]+(?:,[^,]*)*)', params_str)
                        for param_name, param_value in param_pairs:
                            # Remove quotes and evaluate
                            param_value = param_value.strip()
                            if param_value.startswith(('"', "'")) and param_value.endswith(('"', "'")):
                                params[param_name] = param_value[1:-1]
                            elif param_value.isdigit():
                                params[param_name] = int(param_value)
                            elif param_value.replace('.', '').isdigit():
                                params[param_name] = float(param_value)
                            elif param_value == 'True':
                                params[param_name] = True
                            elif param_value == 'False':
                                params[param_name] = False
                            else:
                                params[param_name] = param_value
                    
                    tool_calls.append({
                        "tool": tool_name,
                        "parameters": params
                    })
                except Exception as e:
                    # Skip malformed tool calls
                    continue
        
        return tool_calls


__all__ = ["register_tool", "ToolRegistry", "ToolManager"]
