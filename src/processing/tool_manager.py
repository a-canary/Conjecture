"""
Dynamic Tool Creation System
Allows LLM to discover, create, and validate tools dynamically
"""

import asyncio
import ast
import os
import tempfile
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import logging
import hashlib

from ..core.models import Claim, ClaimState
from .bridge import LLMBridge, LLMRequest

# Try to import core tools, but handle gracefully if not available
try:
    from core_tools.webSearch import WebSearch
    from core_tools.readFiles import ReadFiles
except ImportError:
    WebSearch = None
    ReadFiles = None


class ToolValidator:
    """Validates tool security and functionality"""

    def __init__(self):
        self.dangerous_imports = {
            "os",
            "sys",
            "subprocess",
            "eval",
            "exec",
            "compile",
            "open",
            "file",
            "input",
            "raw_input",
            "__import__",
            "reload",
            "vars",
            "globals",
            "locals",
            "dir",
            "getattr",
            "setattr",
            "delattr",
            "hasattr",
        }

        self.dangerous_functions = {
            "exec",
            "eval",
            "compile",
            "__import__",
            "reload",
            "open",
            "file",
            "input",
            "raw_input",
        }

        self.allowed_modules = {
            "math",
            "datetime",
            "json",
            "re",
            "string",
            "random",
            "collections",
            "itertools",
            "functools",
            "operator",
            "typing",
            "dataclasses",
            "enum",
            "pathlib",
            "urllib.parse",
        }

    def validate_tool_code(self, code: str) -> Tuple[bool, List[str]]:
        """
        Validate tool code for security and functionality

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        try:
            # Parse AST to check for dangerous operations
            tree = ast.parse(code)

            for node in ast.walk(tree):
                # Check for dangerous imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in self.dangerous_imports:
                            issues.append(f"Dangerous import: {alias.name}")

                elif isinstance(node, ast.ImportFrom):
                    if node.module in self.dangerous_imports:
                        issues.append(f"Dangerous import from: {node.module}")

                # Check for dangerous function calls
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in self.dangerous_functions:
                            issues.append(f"Dangerous function call: {node.func.id}")

                # Check for file operations
                elif isinstance(node, ast.Name):
                    if node.id in {"open", "file"}:
                        issues.append(f"File operation detected: {node.id}")

            # Check for required function structure
            required_functions = ["execute"]
            found_functions = []

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    found_functions.append(node.name)

            for func in required_functions:
                if func not in found_functions:
                    issues.append(f"Missing required function: {func}")

            # Check for proper docstring
            # Skip docstring check for now to focus on security validation
            pass

        except SyntaxError as e:
            issues.append(f"Syntax error: {e}")
        except Exception as e:
            issues.append(f"Parse error: {e}")

        return len(issues) == 0, issues

    def validate_tool_security(self, tool_path: str) -> Tuple[bool, List[str]]:
        """
        Additional security validation for created tool files

        Returns:
            Tuple of (is_secure, list_of_security_issues)
        """
        issues = []

        try:
            # Check file permissions
            if os.path.exists(tool_path):
                stat_info = os.stat(tool_path)
                mode = oct(stat_info.st_mode)[-3:]
                if mode != "644":
                    issues.append(f"Insecure file permissions: {mode}")

            # Check file size (prevent extremely large files)
            if os.path.exists(tool_path):
                size = os.path.getsize(tool_path)
                if size > 10240:  # 10KB limit
                    issues.append(f"File too large: {size} bytes")

        except Exception as e:
            issues.append(f"Security check error: {e}")

        return len(issues) == 0, issues


class DynamicToolCreator:
    """
    Dynamic Tool Creation System
    Enables LLM to discover needs and create tools dynamically
    """

    def __init__(
        self,
        llm_bridge: LLMBridge,
        tools_dir: str = "tools",
        validator: Optional[ToolValidator] = None,
    ):
        self.llm_bridge = llm_bridge
        self.tools_dir = Path(tools_dir)
        self.validator = validator or ToolValidator()

        # Ensure tools directory exists
        self.tools_dir.mkdir(exist_ok=True)

        # Track created tools
        self.created_tools: Dict[str, Dict[str, Any]] = {}

        self.logger = logging.getLogger(__name__)

    async def discover_tool_need(self, claim: Claim) -> Optional[str]:
        """
        Analyze a claim to discover if a new tool is needed

        Returns:
            Tool need description or None if no tool needed
        """
        try:
            prompt = f"""Analyze this claim to determine if a new tool is needed:

CLAIM: {claim.content}
TAGS: {", ".join(claim.tags)}

Available tools: WebSearch, ReadFiles, WriteCodeFile, CreateClaim, ClaimSupport

Determine if:
1. Existing tools can handle this need
2. A new specialized tool would be beneficial
3. What the new tool should do

If a new tool is needed, describe:
- Tool purpose and functionality
- Input parameters needed
- Expected output format
- Why existing tools are insufficient

If no new tool is needed, respond: NO_TOOL_NEEDED"""

            llm_request = LLMRequest(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.3,
                task_type="tool_discovery",
            )

            response = self.llm_bridge.process(llm_request)

            if response.success and response.content:
                if "NO_TOOL_NEEDED" in response.content.upper():
                    return None

                return response.content.strip()

        except Exception as e:
            self.logger.error(f"Error discovering tool need: {e}")

        return None

    async def websearch_tool_methods(self, tool_need: str) -> List[str]:
        """Search for existing tool implementation methods

        Returns:
            List of implementation approaches found
        """
        try:
            search_query = f"python implementation {tool_need} function example"

            # Use WebSearch tool to find implementation methods
            if WebSearch is None:
                self.logger.warning(
                    "WebSearch tool not available, returning mock methods"
                )
                return [
                    f"Mock implementation method for {tool_need}",
                    f"Alternative approach for {tool_need}",
                ]

            web_search = WebSearch()
            search_results = await web_search.search(search_query, max_results=5)

            methods = []
            for result in search_results:
                if result.get("content"):
                    methods.append(result["content"])

            self.logger.info(
                f"Found {len(methods)} implementation methods for {tool_need}"
            )
            return methods

        except Exception as e:
            self.logger.error(f"Error searching tool methods: {e}")
            return []

    async def create_tool_file(
        self, tool_name: str, tool_description: str, implementation_methods: List[str]
    ) -> Optional[str]:
        """
        Create a new tool file based on need and research

        Returns:
            Path to created tool file or None if creation failed
        """
        try:
            # Generate tool code
            tool_code = await self._generate_tool_code(
                tool_name, tool_description, implementation_methods
            )

            if not tool_code:
                return None

            # Validate the generated code
            is_valid, issues = self.validator.validate_tool_code(tool_code)
            if not is_valid:
                self.logger.error(f"Generated tool code validation failed: {issues}")
                return None

            # Create tool file
            tool_path = self.tools_dir / f"{tool_name.lower()}.py"

            # Write tool file
            with open(tool_path, "w", encoding="utf-8") as f:
                f.write(tool_code)

            # Validate security
            is_secure, security_issues = self.validator.validate_tool_security(
                str(tool_path)
            )
            if not is_secure:
                self.logger.error(f"Tool security validation failed: {security_issues}")
                # Remove insecure file
                tool_path.unlink(missing_ok=True)
                return None

            # Track created tool
            self.created_tools[tool_name] = {
                "path": str(tool_path),
                "description": tool_description,
                "created_at": datetime.utcnow().isoformat(),
                "methods_used": len(implementation_methods),
            }

            self.logger.info(f"Created tool file: {tool_path}")
            return str(tool_path)

        except Exception as e:
            self.logger.error(f"Error creating tool file: {e}")
            return None

    async def _generate_tool_code(
        self, tool_name: str, tool_description: str, implementation_methods: List[str]
    ) -> Optional[str]:
        """Generate tool code using LLM"""
        try:
            methods_text = "\n\n".join(
                [
                    f"Method {i + 1}:\n{method}"
                    for i, method in enumerate(implementation_methods)
                ]
            )

            template_code = f'''
"""
{tool_description}
"""

from typing import Dict, Any, Optional
import json
import re
from datetime import datetime

def execute(parameter1: str, parameter2: Optional[str] = None) -> Dict[str, Any]:
    """
    Execute the {tool_name} tool.
    
    Args:
        parameter1: Description of parameter1
        parameter2: Description of parameter2 (optional)
    
    Returns:
        Dictionary with success status and result or error message
    """
    try:
        # Implementation goes here
        result = "Tool execution result"
        
        return {{
            "success": True,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }}
    except Exception as e:
        return {{
            "success": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }}
'''

            prompt = f"""Create a Python tool file for: {tool_name}

DESCRIPTION:
{tool_description}

IMPLEMENTATION METHODS FOUND:
{methods_text}

REQUIREMENTS:
1. Create a function called 'execute' that takes parameters as needed
2. Include proper error handling
3. Add comprehensive docstring
4. Use only safe modules (math, datetime, json, re, string, random, etc.)
5. No file I/O, network calls, or system operations
6. Return structured results as dictionaries
7. Include type hints where appropriate

TOOL TEMPLATE:
{template_code}

Generate the complete tool code following this template and requirements."""

            llm_request = LLMRequest(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.3,
                task_type="code_generation",
            )

            response = self.llm_bridge.process(llm_request)

            if response.success and response.content:
                # Extract code from response
                code = self._extract_code_from_response(response.content)
                if code:
                    return code

        except Exception as e:
            self.logger.error(f"Error generating tool code: {e}")

        return None

    def _extract_code_from_response(self, response: str) -> Optional[str]:
        """Extract Python code from LLM response"""
        # Look for code blocks
        import re

        # Pattern for ```python code blocks
        pattern = r"```python\n(.*?)\n```"
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Pattern for ``` code blocks
        pattern = r"```\n(.*?)\n```"
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            return matches[0].strip()

        # If no code blocks, return the whole response
        return response.strip()

    async def create_skill_claim(
        self, tool_name: str, tool_description: str, tool_path: str
    ) -> Claim:
        """Create a skill claim describing how to use the tool"""
        try:
            # Read the tool file to understand its interface
            with open(tool_path, "r", encoding="utf-8") as f:
                tool_code = f.read()

            # Extract function signature
            function_info = self._extract_function_info(tool_code)

            prompt = f"""Create a skill claim for using this tool:

TOOL: {tool_name}
DESCRIPTION: {tool_description}
FUNCTION: {function_info}

Create a procedural skill claim that explains:
1. When to use this tool
2. How to prepare inputs
3. How to call the execute function
4. How to interpret results
5. Common usage patterns

Format as a clear, step-by-step procedure for LLM to follow."""

            llm_request = LLMRequest(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.3,
                task_type="skill_creation",
            )

            response = self.llm_bridge.process(llm_request)

            if response.success and response.content:
                skill_claim = Claim(
                    id=f"skill_{tool_name}_{int(datetime.utcnow().timestamp())}",
                    content=response.content.strip(),
                    confidence=0.85,
                    tags=["skill", "tool", tool_name, "concept"],
                    state=ClaimState.VALIDATED,
                )

                return skill_claim

        except Exception as e:
            self.logger.error(f"Error creating skill claim: {e}")

        # Fallback skill claim
        return Claim(
            id=f"skill_{tool_name}_{int(datetime.utcnow().timestamp())}",
            content=f"To use {tool_name}: 1) Prepare required parameters, 2) Call execute() function, 3) Handle response appropriately",
            confidence=0.7,
            tags=["skill", "tool", tool_name, "concept"],
            state=ClaimState.EXPLORE,
        )

    def _extract_function_info(self, tool_code: str) -> str:
        """Extract function signature and docstring from tool code"""
        try:
            tree = ast.parse(tool_code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "execute":
                    # Get function signature
                    args = []
                    for arg in node.args.args:
                        args.append(arg.arg)

                    signature = f"execute({', '.join(args)})"

                    # Get docstring
                    if (
                        node.body
                        and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)
                    ):
                        docstring = node.body[0].value.value
                        return f"{signature}\n\n{docstring}"

                    return signature

        except Exception as e:
            self.logger.error(f"Error extracting function info: {e}")

        return "execute(parameters)"

    async def create_sample_claim(self, tool_name: str, tool_path: str) -> Claim:
        """Create a sample claim showing exact tool usage"""
        try:
            # Generate a sample usage
            prompt = f"""Create a sample claim showing exact usage of this tool:

TOOL: {tool_name}

Create a realistic example showing:
1. The exact function call with parameters
2. The expected response format
3. How to handle the response

Format as a concrete example that can be used as a reference."""

            llm_request = LLMRequest(
                prompt=prompt,
                max_tokens=500,
                temperature=0.2,
                task_type="sample_creation",
            )

            response = self.llm_bridge.process(llm_request)

            if response.success and response.content:
                sample_claim = Claim(
                    id=f"sample_{tool_name}_{int(datetime.utcnow().timestamp())}",
                    content=response.content.strip(),
                    confidence=0.9,
                    tags=["sample", "tool", tool_name, "example"],
                    state=ClaimState.VALIDATED,
                )

                return sample_claim

        except Exception as e:
            self.logger.error(f"Error creating sample claim: {e}")

        # Fallback sample claim
        return Claim(
            id=f"sample_{tool_name}_{int(datetime.utcnow().timestamp())}",
            content=f"Example: result = execute(parameter1='value', parameter2='optional')",
            confidence=0.8,
            tags=["sample", "tool", tool_name, "example"],
            state=ClaimState.EXPLORE,
        )

    def get_created_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all created tools"""
        return self.created_tools.copy()

    def tool_exists(self, tool_name: str) -> bool:
        """Check if a tool already exists"""
        tool_path = self.tools_dir / f"{tool_name.lower()}.py"
        return tool_path.exists()

    async def validate_created_tool(self, tool_name: str) -> Tuple[bool, List[str]]:
        """Validate a created tool"""
        tool_path = self.tools_dir / f"{tool_name.lower()}.py"

        if not tool_path.exists():
            return False, [f"Tool file not found: {tool_path}"]

        # Read and validate code
        try:
            with open(tool_path, "r", encoding="utf-8") as f:
                code = f.read()

            is_valid, issues = self.validator.validate_tool_code(code)
            is_secure, security_issues = self.validator.validate_tool_security(
                str(tool_path)
            )

            all_issues = issues + security_issues
            return is_valid and is_secure, all_issues

        except Exception as e:
            return False, [f"Validation error: {e}"]
