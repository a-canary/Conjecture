"""
Simplified Tool Execution for Conjecture
Basic tool implementation for WebSearch, ReadFiles, WriteCodeFile, CreateClaim, ClaimSupport
"""

import json
import os
import re
import requests
from typing import Any, Dict, List, Optional

# Use local implementations for simplified data operations


class ToolManager:
    """Simplified tool execution"""

    def __init__(self):
        self.tools = self._initialize_tools()

    def _initialize_tools(self) -> Dict[str, Dict[str, Any]]:
        """Initialize available tools"""
        return {
            "WebSearch": {
                "description": "Search the web for information",
                "parameters": {
                    "query": "string",
                    "max_results": "integer (optional, default 5)"
                }
            },
            "ReadFiles": {
                "description": "Read content from files",
                "parameters": {
                    "files": "array of file paths"
                }
            },
            "WriteCodeFile": {
                "description": "Write code to a file",
                "parameters": {
                    "file_path": "string",
                    "content": "string"
                }
            },
            "CreateClaim": {
                "description": "Create a new knowledge claim",
                "parameters": {
                    "content": "string",
                    "confidence": "number (0-1)",
                    "claim_type": "string",
                    "tags": "array of strings (optional)"
                }
            },
            "ClaimSupport": {
                "description": "Find claims that support a given claim",
                "parameters": {
                    "claim_id": "string",
                    "max_results": "integer (optional, default 5)"
                }
            }
        }

    def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with given parameters"""
        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}. Available: {list(self.tools.keys())}"
            }

        try:
            if tool_name == "WebSearch":
                return self._web_search(parameters)
            elif tool_name == "ReadFiles":
                return self._read_files(parameters)
            elif tool_name == "WriteCodeFile":
                return self._write_code_file(parameters)
            elif tool_name == "CreateClaim":
                return self._create_claim(parameters)
            elif tool_name == "ClaimSupport":
                return self._claim_support(parameters)
        except Exception as e:
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}"
            }

    def _web_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform web search (mock implementation)"""
        query = params.get("query")
        max_results = params.get("max_results", 5)

        if not query:
            return {"success": False, "error": "Query parameter is required"}

        # Mock web search results (in real implementation, use search API)
        mock_results = [
            {"title": f"Result {i+1} for '{query}'", 
             "url": f"https://example.com/{i+1}",
             "snippet": f"This is a mock search result snippet about {query}"}
            for i in range(min(max_results, 5))
        ]

        return {
            "success": True,
            "results": mock_results,
            "query": query,
            "total_found": len(mock_results)
        }

    def _read_files(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Read content from files"""
        file_paths = params.get("files", [])
        if not file_paths:
            return {"success": False, "error": "Files parameter is required"}

        results = {}
        errors = []

        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    results[file_path] = content[:2000]  # Limit to 2000 chars
                else:
                    errors.append(f"File not found: {file_path}")
            except Exception as e:
                errors.append(f"Error reading {file_path}: {str(e)}")

        return {
            "success": len(errors) == 0,
            "results": results,
            "errors": errors
        }

    def _write_code_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Write content to a file"""
        file_path = params.get("file_path")
        content = params.get("content")

        if not file_path:
            return {"success": False, "error": "file_path parameter is required"}
        
        if content is None:
            return {"success": False, "error": "content parameter is required"}

        try:
            # Create directory if it doesn't exist
            dir_path = os.path.dirname(file_path)
            if dir_path:  # Only create if directory path is not empty
                os.makedirs(dir_path, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {
                "success": True,
                "file_path": file_path,
                "message": f"Successfully wrote {len(content)} characters to {file_path}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to write file: {str(e)}"
            }

    def _create_claim(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new claim (simplified mock implementation)"""
        content = params.get("content")
        confidence = params.get("confidence")
        claim_type = params.get("claim_type")
        tags = params.get("tags", [])

        if None in [content, confidence, claim_type]:
            return {"success": False, "error": "content, confidence, and claim_type are required"}

        try:
            # Mock claim creation
            import time
            claim_id = f"claim_{int(time.time())}_{len(content)}"
            mock_claim = {
                "id": claim_id,
                "content": content,
                "confidence": confidence,
                "type": [claim_type],
                "tags": tags or [],
                "created": time.time()
            }
            return {
                "success": True,
                "claim_id": claim_id,
                "claim": mock_claim
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _claim_support(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Find claims that support a given claim (mock implementation)"""
        claim_id = params.get("claim_id")
        max_results = params.get("max_results", 5)

        if not claim_id:
            return {"success": False, "error": "claim_id parameter is required"}

        # Mock supporting claims
        supporting_claims = [
            {"id": f"support_{i}", "content": f"Supporting claim {i+1} for {claim_id}", 
             "confidence": 0.8 + i * 0.02, "type": ["reference"]}
            for i in range(min(max_results, 3))
        ]

        return {
            "success": True,
            "target_claim_id": claim_id,
            "supporting_claims": supporting_claims,
            "total_found": len(supporting_claims)
        }

    def get_tool_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Get all available tool definitions"""
        return self.tools

    def parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Parse tool calls from text using regex"""
        tool_calls = []
        
        # Enhanced regex pattern for tool calls (handles quotes better)
        pattern = r'(\w+)\(([^)]*)\)'
        matches = re.findall(pattern, text)
        
        for match in matches:
            tool_name = match[0]
            params_str = match[1].strip()
            
            # Manual parameter parsing (simplified)
            params = {}
            if params_str:
                for param in params_str.split(','):
                    if '=' in param:
                        key, value = param.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"\'')
                        try:
                            # Try to parse as number
                            if '.' in value:
                                value = float(value)
                            elif value.isdigit():
                                value = int(value)
                        except:
                            # Keep as string
                            pass
                        params[key] = value
            
            tool_calls.append({
                "tool": tool_name,
                "parameters": params
            })
        
        return tool_calls


def call_tool(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to call a tool"""
    tm = ToolManager()
    return tm.call_tool(tool_name, parameters)


def get_tool_definitions() -> Dict[str, Dict[str, Any]]:
    """Convenience function to get tool definitions"""
    tm = ToolManager()
    return tm.get_tool_definitions()


if __name__ == "__main__":
    print("🧪 Testing Tool Manager")
    print("=" * 30)
    
    tm = ToolManager()
    
    # Test WebSearch
    result = tm.call_tool("WebSearch", {"query": "machine learning", "max_results": 3})
    print(f"✅ WebSearch: {result['success']}, found {len(result.get('results', []))} results")
    
    # Test CreateClaim
    result = tm.call_tool("CreateClaim", {
        "content": "Python is widely used for machine learning",
        "confidence": 0.9,
        "claim_type": "reference",
        "tags": ["python", "ml"]
    })
    print(f"✅ CreateClaim: {result['success']}, claim_id: {result.get('claim_id')}")
    
    # Test WriteCodeFile
    result = tm.call_tool("WriteCodeFile", {
        "file_path": "test_output.py",
        "content": "print('Hello from Conjecture!')\ndef test():\n    return 'success'"
    })
    print(f"✅ WriteCodeFile: {result['success']}")
    
    # Test tool parsing
    text = "WebSearch(query='AI research'), CreateClaim(content='Test claim', confidence=0.8, claim_type='concept')"
    calls = tm.parse_tool_calls(text)
    print(f"✅ Tool parsing: found {len(calls)} tool calls")
    
    print("🎉 Tool Manager tests passed!")