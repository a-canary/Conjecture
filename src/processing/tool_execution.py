"""
Tool Execution - Compatibility Layer
Provides tool execution functionality for testing
"""

from typing import Dict, Any, List, Optional, Callable
from pydantic import BaseModel
from datetime import datetime

class ToolResult(BaseModel):
    """Result of tool execution"""
    success: bool = True
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0

class ToolExecutionContext(BaseModel):
    """Context for tool execution"""
    tools: Dict[str, Callable] = {}
    variables: Dict[str, Any] = {}
    timestamp: datetime = datetime.now()

class ToolExecutor:
    """Real
    
    def __init__(self, context: Optional[ToolExecutionContext] = None):
        self.context = context or ToolExecutionContext()
        self.execution_history: List[ToolResult] = []
    
    def execute_tool(self, tool_name: str, *args, **kwargs) -> ToolResult:
        """Execute a tool"""
        if tool_name in self.context.tools:
            try:
                result = self.context.tools[tool_name](*args, **kwargs)
                tool_result = ToolResult(success=True, result=result)
            except Exception as e:
                tool_result = ToolResult(success=False, error=str(e))
        else:
            tool_result = ToolResult(success=False, error=f"Tool {tool_name} not found")
        
        self.execution_history.append(tool_result)
        return tool_result
    
    def register_tool(self, name: str, tool: Callable) -> bool:
        """Register a new tool"""
        self.context.tools[name] = tool
        return True
    
    def list_tools(self) -> List[str]:
        """List all available tools"""
        return list(self.context.tools.keys())
    
    def get_execution_history(self) -> List[ToolResult]:
        """Get execution history"""
        return self.execution_history.copy()

def execute_tool_from_registry(tool_name: str, *args, **kwargs) -> ToolResult:
    """Execute a tool from the registry"""
    