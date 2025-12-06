"""
Tool Execution Engine for the Conjecture skill-based agency system.
Provides safe execution of Python code with timeout and resource limits.
"""

import asyncio
import time
import sys
import traceback
import signal
import platform

# Resource module is Unix-only
if platform.system() != "Windows":
    import resource
import os
import tempfile
import shutil
from typing import Dict, Any, Optional, List, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging
import ast
import inspect

from ..core.models import ExecutionResult, ToolCall


logger = logging.getLogger(__name__)


@dataclass
class ExecutionLimits:
    """Resource limits for code execution."""

    max_execution_time: float = 30.0  # seconds
    max_memory_mb: int = 100  # MB
    max_cpu_time: float = 10.0  # seconds
    max_output_chars: int = 10000  # characters
    allow_network: bool = False
    allow_file_access: bool = False
    allowed_modules: List[str] = None

    def __post_init__(self):
        if self.allowed_modules is None:
            self.allowed_modules = [
                "math",
                "random",
                "datetime",
                "json",
                "re",
                "string",
                "collections",
                "itertools",
                "functools",
                "operator",
                "statistics",
                "fractions",
                "decimal",
                "typing",
            ]


class SecurityValidator:
    """Validates code for security violations before execution."""

    def __init__(self, limits: ExecutionLimits):
        self.limits = limits
        self.dangerous_functions = {
            "exec",
            "eval",
            "compile",
            "__import__",
            "open",
            "file",
            "input",
            "raw_input",
            "reload",
            "__builtins__",
            "globals",
            "locals",
            "vars",
            "dir",
            "getattr",
            "setattr",
            "delattr",
            "hasattr",
            "callable",
            "isinstance",
            "issubclass",
            "iter",
            "next",
            "range",
            "xrange",
            "slice",
            "buffer",
            "memoryview",
            "bytearray",
            "bytes",
            "type",
            "super",
            "property",
            "classmethod",
            "staticmethod",
            "zip",
            "map",
            "filter",
            "reduce",
            "apply",
        }

        self.dangerous_modules = {
            "os",
            "sys",
            "subprocess",
            "shutil",
            "glob",
            "pickle",
            "marshal",
            "imp",
            "importlib",
            "pkgutil",
            "modulefinder",
            "inspect",
            "traceback",
            "linecache",
            "pdb",
            "bdb",
            "cmd",
            "code",
            "codeop",
            "dis",
            "parser",
            "symbol",
            "token",
            "keyword",
            "tokenize",
            "py_compile",
            "compileall",
            "pyclbr",
            "pydoc",
            "doctest",
            "unittest",
            "test",
            "socket",
            "urllib",
            "httplib",
            "ftplib",
            "poplib",
            "imaplib",
            "nntplib",
            "smtplib",
            "telnetlib",
            "uuid",
            "ssl",
            "hashlib",
            "hmac",
            "secrets",
            "base64",
            "binascii",
            "tempfile",
            "shelve",
            "dbm",
            "sqlite3",
            "csv",
            "configparser",
            "xml",
            "xmlrpc",
            "email",
            "mimetypes",
            "mailbox",
            "mbox",
            "distutils",
            "setuptools",
            "pip",
            "venv",
            "site",
            "usercustomize",
        }

    def validate_code(self, code: str) -> tuple[bool, List[str]]:
        """
        Validate code for security violations.

        Args:
            code: Python code to validate

        Returns:
            Tuple of (is_safe, error_messages)
        """
        errors = []

        try:
            # Parse the code to check for syntax errors
            tree = ast.parse(code)

            # Analyze AST for dangerous constructs
            for node in ast.walk(tree):
                errors.extend(self._check_node(node))

            # Check for dangerous string patterns
            errors.extend(self._check_string_patterns(code))

        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
        except Exception as e:
            errors.append(f"Parse error: {e}")

        return len(errors) == 0, errors

    def _check_node(self, node: ast.AST) -> List[str]:
        """Check an AST node for dangerous constructs."""
        errors = []

        # Check for function calls to dangerous functions
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in self.dangerous_functions:
                    errors.append(f"Dangerous function call: {node.func.id}")

            elif isinstance(node.func, ast.Attribute):
                if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "__builtins__"
                    and node.func.attr in self.dangerous_functions
                ):
                    errors.append(
                        f"Dangerous built-in access: __builtins__.{node.func.attr}"
                    )

        # Check for imports
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self.dangerous_modules:
                        errors.append(f"Dangerous module import: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module in self.dangerous_modules:
                    errors.append(f"Dangerous module import: {node.module}")

        # Check for attribute access on dangerous objects
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in [
                "__builtins__",
                "__import__",
            ]:
                errors.append(
                    f"Dangerous attribute access: {node.value.id}.{node.attr}"
                )

        # Check for exec/eval as statements
        elif isinstance(node, ast.Expr):
            if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id in ["exec", "eval"]
            ):
                errors.append(f"Dangerous statement: {node.value.func.id}")

        return errors

    def _check_string_patterns(self, code: str) -> List[str]:
        """Check code string for dangerous patterns."""
        errors = []

        # Check for file system access patterns
        file_patterns = [
            r"open\s*\(",
            r"file\s*\(",
            r"\.read\s*\(",
            r"\.write\s*\(",
            r"__import__\s*\(",
            r"exec\s*\(",
            r"eval\s*\(",
        ]

        for pattern in file_patterns:
            if re.search(pattern, code):
                errors.append(f"Dangerous pattern detected: {pattern}")

        return errors


class SafeExecutor:
    """Executes Python code in a safe, restricted environment."""

    def __init__(self, limits: ExecutionLimits):
        self.limits = limits
        self.security_validator = SecurityValidator(limits)
        self.temp_dir = None

    async def execute_code(
        self, code: str, context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """
        Execute Python code safely with resource limits.

        Args:
            code: Python code to execute
            context: Optional context variables

        Returns:
            ExecutionResult with outcome and metadata
        """
        start_time = time.time()

        try:
            # Validate code for security
            is_safe, errors = self.security_validator.validate_code(code)
            if not is_safe:
                return ExecutionResult(
                    success=False,
                    error_message=f"Security validation failed: {'; '.join(errors)}",
                    execution_time_ms=int((time.time() - start_time) * 1000),
                    skill_id="safe_executor",
                    parameters_used={
                        "code": code[:100] + "..." if len(code) > 100 else code
                    },
                )

            # Prepare execution environment
            exec_context = self._prepare_execution_context(context or {})

            # Execute with timeout and resource limits
            result = await self._execute_with_limits(code, exec_context)

            execution_time = int((time.time() - start_time) * 1000)

            return ExecutionResult(
                success=result["success"],
                result=result.get("result"),
                error_message=result.get("error"),
                execution_time_ms=execution_time,
                stdout=result.get("stdout"),
                stderr=result.get("stderr"),
                skill_id="safe_executor",
                parameters_used={
                    "code": code[:100] + "..." if len(code) > 100 else code
                },
            )

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            logger.error(f"Safe execution error: {e}")

            return ExecutionResult(
                success=False,
                error_message=f"Execution error: {str(e)}",
                execution_time_ms=execution_time,
                skill_id="safe_executor",
                parameters_used={
                    "code": code[:100] + "..." if len(code) > 100 else code
                },
            )

    def _prepare_execution_context(
        self, user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare a safe execution context."""
        # Start with safe built-ins
        safe_context = {
            # Safe built-in functions
            "__builtins__": {
                "abs": abs,
                "all": all,
                "any": any,
                "bin": bin,
                "bool": bool,
                "chr": chr,
                "dict": dict,
                "divmod": divmod,
                "enumerate": enumerate,
                "float": float,
                "hex": hex,
                "int": int,
                "len": len,
                "list": list,
                "max": max,
                "min": min,
                "oct": oct,
                "ord": ord,
                "pow": pow,
                "range": range,
                "repr": repr,
                "reversed": reversed,
                "round": round,
                "set": set,
                "slice": slice,
                "sorted": sorted,
                "str": str,
                "sum": sum,
                "tuple": tuple,
                "type": type,
                "zip": zip,
            }
        }

        # Add allowed modules
        for module_name in self.limits.allowed_modules:
            try:
                module = __import__(module_name)
                safe_context[module_name] = module
            except ImportError:
                logger.warning(f"Could not import allowed module: {module_name}")

        # Add user context (filtered for safety)
        for key, value in user_context.items():
            if not key.startswith("__") and key not in ["builtins", "import"]:
                safe_context[key] = value

        return safe_context

    async def _execute_with_limits(
        self, code: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute code with timeout and resource limits."""
        # Create a temporary file for the code
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            # Execute in a separate process with limits
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                f'''
import sys
import time
import signal
import resource
import traceback
import json

# Set resource limits (Unix only)
if platform.system() != 'Windows':
    try:
        resource.setrlimit(resource.RLIMIT_AS, ({self.limits.max_memory_mb * 1024 * 1024}, {self.limits.max_memory_mb * 1024 * 1024}))
        resource.setrlimit(resource.RLIMIT_CPU, (int({self.limits.max_cpu_time}), int({self.limits.max_cpu_time})))
    except (ValueError, OSError):
        pass  # Limits not supported on this system

# Load context
context = {json.dumps(context)}

# Capture output
import io
import sys
old_stdout = sys.stdout
old_stderr = sys.stderr

stdout_capture = io.StringIO()
stderr_capture = io.StringIO()

sys.stdout = stdout_capture
sys.stderr = stderr_capture

result = {{"success": False, "error": None, "stdout": "", "stderr": ""}}

try:
    # Execute the code
    exec(open("{temp_file}").read(), context)
    result["success"] = True
except Exception as e:
    result["error"] = str(e)
    result["traceback"] = traceback.format_exc()

# Restore output
sys.stdout = old_stdout
sys.stderr = old_stderr

result["stdout"] = stdout_capture.getvalue()
result["stderr"] = stderr_capture.getvalue()

# Limit output size
if len(result["stdout"]) > {self.limits.max_output_chars}:
    result["stdout"] = result["stdout"][:{self.limits.max_output_chars}] + "... (truncated)"
if len(result["stderr"]) > {self.limits.max_output_chars}:
    result["stderr"] = result["stderr"][:{self.limits.max_output_chars}] + "... (truncated)"

print(json.dumps(result))
                ''',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=self.limits.max_execution_time
                )

                # Parse result with better error handling
                result_json = stdout.decode().strip()
                if result_json:
                    try:
                        result = json.loads(result_json)
                    except json.JSONDecodeError:
                        result = {
                            "success": False,
                            "error": "Invalid JSON output from execution",
                            "stdout": result_json[:500] + "..." if len(result_json) > 500 else result_json,
                            "stderr": stderr.decode(),
                        }
                else:
                    result = {
                        "success": False,
                        "error": "No output from execution",
                        "stdout": "",
                        "stderr": stderr.decode(),
                    }

            except asyncio.TimeoutError:
                # More graceful timeout handling
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                
                result = {
                    "success": False,
                    "error": f"Execution timeout after {self.limits.max_execution_time} seconds",
                    "stdout": "",
                    "stderr": "",
                }

        except Exception as e:
            result = {
                "success": False,
                "error": f"Process execution error: {str(e)}",
                "stdout": "",
                "stderr": "",
            }

        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except OSError:
                pass

        return result


class ToolExecutor:
    """
    Main tool execution engine that coordinates safe execution.
    """

    def __init__(self, limits: Optional[ExecutionLimits] = None):
        self.limits = limits or ExecutionLimits()
        self.safe_executor = SafeExecutor(self.limits)
        self.execution_history: List[ExecutionResult] = []
        self.max_history_size = 100

    async def execute_tool_call(
        self, tool_call: ToolCall, skill_functions: Optional[Dict[str, Callable]] = None
    ) -> ExecutionResult:
        """
        Execute a tool call safely.

        Args:
            tool_call: Tool call to execute
            skill_functions: Optional mapping of skill names to functions

        Returns:
            ExecutionResult with outcome
        """
        start_time = time.time()

        try:
            # Check if it's a built-in skill function
            if skill_functions and tool_call.name in skill_functions:
                return await self._execute_builtin_function(
                    tool_call, skill_functions[tool_call.name]
                )

            # Otherwise, try to execute as Python code
            return await self._execute_code_tool(tool_call)

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            logger.error(f"Tool execution error: {e}")

            return ExecutionResult(
                success=False,
                error_message=f"Tool execution error: {str(e)}",
                execution_time_ms=execution_time,
                skill_id=tool_call.name,
                parameters_used=tool_call.parameters,
            )

    async def _execute_builtin_function(
        self, tool_call: ToolCall, function: Callable
    ) -> ExecutionResult:
        """Execute a built-in function."""
        start_time = time.time()

        try:
            # Check if function is async
            if inspect.iscoroutinefunction(function):
                result = await function(**tool_call.parameters)
            else:
                result = function(**tool_call.parameters)

            execution_time = int((time.time() - start_time) * 1000)

            return ExecutionResult(
                success=True,
                result=result,
                execution_time_ms=execution_time,
                skill_id=tool_call.name,
                parameters_used=tool_call.parameters,
            )

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)

            return ExecutionResult(
                success=False,
                error_message=f"Function execution error: {str(e)}",
                execution_time_ms=execution_time,
                skill_id=tool_call.name,
                parameters_used=tool_call.parameters,
            )

    async def _execute_code_tool(self, tool_call: ToolCall) -> ExecutionResult:
        """Execute a tool call as Python code."""
        # Extract code from parameters
        code = tool_call.parameters.get("code")
        if not code:
            return ExecutionResult(
                success=False,
                error_message="Missing 'code' parameter for code execution",
                execution_time_ms=0,
                skill_id=tool_call.name,
                parameters_used=tool_call.parameters,
            )

        # Prepare context from other parameters
        context = {k: v for k, v in tool_call.parameters.items() if k != "code"}

        # Execute safely
        return await self.safe_executor.execute_code(code, context)

    def add_to_history(self, result: ExecutionResult) -> None:
        """Add execution result to history."""
        self.execution_history.append(result)

        # Maintain history size limit
        if len(self.execution_history) > self.max_history_size:
            self.execution_history = self.execution_history[-self.max_history_size :]

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        if not self.execution_history:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
                "average_execution_time_ms": 0.0,
                "most_used_tools": [],
            }

        total_executions = len(self.execution_history)
        successful_executions = sum(
            1 for result in self.execution_history if result.success
        )
        success_rate = successful_executions / total_executions

        avg_execution_time = (
            sum(result.execution_time_ms for result in self.execution_history)
            / total_executions
        )

        # Most used tools
        tool_counts = {}
        for result in self.execution_history:
            tool_counts[result.skill_id] = tool_counts.get(result.skill_id, 0) + 1

        most_used_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[
            :5
        ]

        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "success_rate": success_rate,
            "average_execution_time_ms": avg_execution_time,
            "most_used_tools": most_used_tools,
        }


# Import required modules
import re
import json
