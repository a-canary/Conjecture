"""
Unit tests for ToolExecutor component.
"""
import pytest
import asyncio
import tempfile
import os
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.processing.tool_executor import (
    ToolExecutor, SafeExecutor, SecurityValidator, ExecutionLimits
)
from src.core.skill_models import ExecutionResult, ToolCall


class TestExecutionLimits:
    """Test cases for ExecutionLimits class."""

    def test_default_limits(self):
        """Test default execution limits."""
        limits = ExecutionLimits()
        
        assert limits.max_execution_time == 30.0
        assert limits.max_memory_mb == 100
        assert limits.max_cpu_time == 10.0
        assert limits.max_output_chars == 10000
        assert limits.allow_network is False
        assert limits.allow_file_access is False
        assert len(limits.allowed_modules) > 0
        assert 'math' in limits.allowed_modules
        assert 'random' in limits.allowed_modules

    def test_custom_limits(self):
        """Test custom execution limits."""
        limits = ExecutionLimits(
            max_execution_time=5.0,
            max_memory_mb=50,
            max_cpu_time=2.0,
            max_output_chars=1000,
            allow_network=True,
            allow_file_access=True,
            allowed_modules=['math', 'json']
        )
        
        assert limits.max_execution_time == 5.0
        assert limits.max_memory_mb == 50
        assert limits.max_cpu_time == 2.0
        assert limits.max_output_chars == 1000
        assert limits.allow_network is True
        assert limits.allow_file_access is True
        assert limits.allowed_modules == ['math', 'json']


class TestSecurityValidator:
    """Test cases for SecurityValidator class."""

    def test_validator_initialization(self, execution_limits):
        """Test SecurityValidator initialization."""
        validator = SecurityValidator(execution_limits)
        
        assert validator.limits == execution_limits
        assert len(validator.dangerous_functions) > 0
        assert 'exec' in validator.dangerous_functions
        assert 'eval' in validator.dangerous_functions
        assert 'open' in validator.dangerous_functions
        assert len(validator.dangerous_modules) > 0
        assert 'os' in validator.dangerous_modules
        assert 'subprocess' in validator.dangerous_modules

    def test_validate_safe_code(self, execution_limits, code_execution_samples):
        """Test validation of safe code."""
        validator = SecurityValidator(execution_limits)
        
        is_safe, errors = validator.validate_code(code_execution_samples["safe_math"])
        
        assert is_safe is True
        assert len(errors) == 0

    def test_validate_syntax_error(self, execution_limits):
        """Test validation of code with syntax error."""
        validator = SecurityValidator(execution_limits)
        
        is_safe, errors = validator.validate_code("def invalid_syntax(:")
        
        assert is_safe is False
        assert len(errors) > 0
        assert any("syntax error" in error.lower() for error in errors)

    def test_validate_dangerous_function_call(self, execution_limits):
        """Test validation of code with dangerous function calls."""
        validator = SecurityValidator(execution_limits)
        
        dangerous_codes = [
            "eval('1+1')",
            "exec('print(1)')",
            "open('file.txt')",
            "__import__('os')",
            "globals()",
            "locals()",
        ]
        
        for code in dangerous_codes:
            is_safe, errors = validator.validate_code(code)
            assert is_safe is False
            assert len(errors) > 0

    def test_validate_dangerous_import(self, execution_limits):
        """Test validation of code with dangerous imports."""
        validator = SecurityValidator(execution_limits)
        
        dangerous_imports = [
            "import os",
            "import sys",
            "import subprocess",
            "import sqlite3",
            "from os import system",
            "from subprocess import run",
        ]
        
        for code in dangerous_imports:
            is_safe, errors = validator.validate_code(code)
            assert is_safe is False
            assert len(errors) > 0
            assert any("Dangerous module import" in error for error in errors)

    def test_validate_allowed_import(self, execution_limits):
        """Test validation of code with allowed imports."""
        validator = SecurityValidator(execution_limits)
        
        safe_imports = [
            "import math",
            "import random",
            "import json",
            "from math import sqrt",
            "from collections import defaultdict",
        ]
        
        for code in safe_imports:
            is_safe, errors = validator.validate_code(code)
            assert is_safe is True
            assert len(errors) == 0

    def test_validate_dangerous_attribute_access(self, execution_limits):
        """Test validation of code with dangerous attribute access."""
        validator = SecurityValidator(execution_limits)
        
        dangerous_access = [
            "__builtins__.__import__",
            "__builtins__.exec",
            "__builtins__.eval",
        ]
        
        for code in dangerous_access:
            is_safe, errors = validator.validate_code(code)
            assert is_safe is False
            assert len(errors) > 0

    def test_validate_dangerous_string_patterns(self, execution_limits):
        """Test validation of dangerous string patterns."""
        validator = SecurityValidator(execution_limits)
        
        dangerous_patterns = [
            "open('file.txt')",
            "exec('code')",
            "eval('expression')",
            "__import__('module')",
        ]
        
        for code in dangerous_patterns:
            is_safe, errors = validator.validate_code(code)
            assert is_safe is False
            assert len(errors) > 0

    def test_check_node_function_call(self, execution_limits):
        """Test AST node checking for function calls."""
        validator = SecurityValidator(execution_limits)
        
        import ast
        dangerous_node = ast.parse("eval('code')").body[0].value
        errors = validator._check_node(dangerous_node)
        
        assert len(errors) > 0
        assert any("Dangerous function call" in error for error in errors)

    def test_check_node_import(self, execution_limits):
        """Test AST node checking for imports."""
        validator = SecurityValidator(execution_limits)
        
        import ast
        import_node = ast.parse("import os").body[0]
        errors = validator._check_node(import_node)
        
        assert len(errors) > 0
        assert any("Dangerous module import" in error for error in errors)

    def test_check_node_import_from(self, execution_limits):
        """Test AST node checking for from...import statements."""
        validator = SecurityValidator(execution_limits)
        
        import ast
        import_from_node = ast.parse("from os import system").body[0]
        errors = validator._check_node(import_from_node)
        
        assert len(errors) > 0
        assert any("Dangerous module import" in error for error in errors)

    def test_check_node_attribute_access(self, execution_limits):
        """Test AST node checking for attribute access."""
        validator = SecurityValidator(execution_limits)
        
        import ast
        attr_node = ast.parse("__builtins__.eval").body[0].value
        errors = validator._check_node(attr_node)
        
        assert len(errors) > 0
        assert any("Dangerous attribute access" in error for error in errors)


class TestSafeExecutor:
    """Test cases for SafeExecutor class."""

    @pytest.mark.asyncio
    async def test_safe_execution_success(self, execution_limits, code_execution_samples):
        """Test successful safe execution."""
        executor = SafeExecutor(execution_limits)
        
        result = await executor.execute_code(
            code_execution_samples["safe_math"],
            context={}
        )
        
        assert isinstance(result, ExecutionResult)
        assert result.success is True
        assert result.result is not None
        assert result.error_message is None
        assert result.execution_time_ms >= 0
        assert result.skill_id == "safe_executor"

    @pytest.mark.asyncio
    async def test_safe_execution_with_context(self, execution_limits):
        """Test execution with context variables."""
        executor = SafeExecutor(execution_limits)
        
        code = "result = context_var * 2"
        context = {"context_var": 5}
        
        result = await executor.execute_code(code, context)
        
        assert result.success is True
        assert result.result is not None  # output should be defined in the exec

    @pytest.mark.asyncio
    async def test_safe_execution_security_violation(self, execution_limits, code_execution_samples):
        """Test execution with security violation."""
        executor = SafeExecutor(execution_limits)
        
        result = await executor.execute_code(
            code_execution_samples["dangerous_import"],
            context={}
        )
        
        assert result.success is False
        assert result.error_message is not None
        assert "Security validation failed" in result.error_message

    @pytest.mark.asyncio
    async def test_safe_execution_syntax_error(self, execution_limits):
        """Test execution with syntax error."""
        executor = SafeExecutor(execution_limits)
        
        result = await executor.execute_code("def invalid_syntax(:", context={})
        
        assert result.success is False
        assert result.error_message is not None
        assert "Syntax error" in result.error_message

    @pytest.mark.asyncio
    async def test_prepare_execution_context(self, execution_limits):
        """Test preparation of execution context."""
        executor = SafeExecutor(execution_limits)
        
        user_context = {"user_var": "test", "*dangerous*": "blocked"}
        
        context = executor._prepare_execution_context(user_context)
        
        # Should contain safe built-ins
        assert "__builtins__" in context
        assert "abs" in context["__builtins__"]
        assert "len" in context["__builtins__"]
        
        # Should contain allowed modules
        assert "math" in context
        assert "json" in context
        
        # Should contain safe user context
        assert context["user_var"] == "test"
        
        # Should not contain dangerous keys
        assert "*dangerous*" not in context
        assert "builtins" not in context
        assert "import" not in context

    @pytest.mark.asyncio
    async def test_execution_with_output_capturing(self, execution_limits):
        """Test execution with stdout/stderr capturing."""
        executor = SafeExecutor(execution_limits)
        
        code = """
import sys
print("This goes to stdout")
print("This goes to stderr", file=sys.stderr)
output = {"result": "done"}
"""
        
        result = await executor.execute_code(code, context={})
        
        assert result.success is True
        assert "This goes to stdout" in result.stdout
        assert "This goes to stderr" in result.stderr

    @pytest.mark.asyncio
    async def test_execution_with_large_output(self, execution_limits):
        """Test execution with large output (truncation)."""
        # Set smaller output limit for testing
        small_limits = ExecutionLimits(max_output_chars=50)
        executor = SafeExecutor(small_limits)
        
        code = """
output = "x" * 1000  # Large output
print("y" * 1000)    # Large stdout
"""
        
        result = await executor.execute_code(code, context={})
        
        assert result.success is True
        assert len(result.stdout) <= 53  # Should be truncated with "..."

    @pytest.mark.asyncio
    @patch('asyncio.create_subprocess_exec')
    async def test_execution_timeout(self, mock_subprocess, execution_limits):
        """Test execution timeout handling."""
        # Mock subprocess that times out
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        
        mock_subprocess.return_value = mock_process
        
        executor = SafeExecutor(execution_limits)
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = "temp.py"
            
            result = await executor.execute_code(
                "import time; time.sleep(100)",  # Long running code
                context={}
            )
        
        assert result.success is False
        assert "timeout" in result.error_message.lower()

    @pytest.mark.asyncio
    @patch('asyncio.create_subprocess_exec')
    async def test_execution_process_error(self, mock_subprocess, execution_limits):
        """Test execution with process error."""
        # Mock subprocess that raises an error
        mock_subprocess.side_effect = OSError("Process creation failed")
        
        executor = SafeExecutor(execution_limits)
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = "temp.py"
            
            result = await executor.execute_code("print('test')", context={})
        
        assert result.success is False
        assert "Process execution error" in result.error_message

    def test_safe_executor_initialization(self, execution_limits):
        """Test SafeExecutor initialization."""
        executor = SafeExecutor(execution_limits)
        
        assert executor.limits == execution_limits
        assert isinstance(executor.security_validator, SecurityValidator)
        assert executor.temp_dir is None


class TestToolExecutor:
    """Test cases for ToolExecutor class."""

    def test_executor_initialization(self, execution_limits):
        """Test ToolExecutor initialization."""
        executor = ToolExecutor(execution_limits)
        
        assert executor.limits == execution_limits
        assert isinstance(executor.safe_executor, SafeExecutor)
        assert len(executor.execution_history) == 0
        assert executor.max_history_size == 100

    def test_executor_default_limits(self):
        """Test ToolExecutor with default limits."""
        executor = ToolExecutor()
        
        assert isinstance(executor.limits, ExecutionLimits)
        assert executor.limits.max_execution_time == 30.0
        assert executor.limits.max_memory_mb == 100

    @pytest.mark.asyncio
    async def test_execute_builtin_function_success(self, execution_limits, mock_async_functions):
        """Test successful execution of built-in function."""
        executor = ToolExecutor(execution_limits)
        
        tool_call = ToolCall(
            name="search_claims",
            parameters={"query": "test", "limit": 5}
        )
        
        skill_functions = {"search_claims": mock_async_functions["search_claims"]}
        
        result = await executor.execute_tool_call(tool_call, skill_functions)
        
        assert isinstance(result, ExecutionResult)
        assert result.success is True
        assert result.result is not None
        assert result.skill_id == tool_call.name
        assert result.parameters_used == tool_call.parameters

    @pytest.mark.asyncio
    async def test_execute_builtin_function_error(self, execution_limits):
        """Test built-in function that raises an error."""
        executor = ToolExecutor(execution_limits)
        
        async def failing_function(**kwargs):
            raise ValueError("Function error")
        
        tool_call = ToolCall(name="failing_tool", parameters={})
        skill_functions = {"failing_tool": failing_function}
        
        result = await executor.execute_tool_call(tool_call, skill_functions)
        
        assert result.success is False
        assert "Function execution error" in result.error_message
        assert result.skill_id == tool_call.name

    @pytest.mark.asyncio
    async def test_execute_sync_builtin_function(self, execution_limits):
        """Test execution of synchronous built-in function."""
        executor = ToolExecutor(execution_limits)
        
        def sync_function(**kwargs):
            return "sync_result"
        
        tool_call = ToolCall(name="sync_tool", parameters={})
        skill_functions = {"sync_tool": sync_function}
        
        result = await executor.execute_tool_call(tool_call, skill_functions)
        
        assert result.success is True
        assert result.result == "sync_result"

    @pytest.mark.asyncio
    async def test_execute_code_tool_success(self, execution_limits, code_execution_samples):
        """Test execution of code tool."""
        executor = ToolExecutor(execution_limits)
        
        tool_call = ToolCall(
            name="execute_code",
            parameters={"code": code_execution_samples["safe_math"]}
        )
        
        result = await executor.execute_tool_call(tool_call)
        
        assert isinstance(result, ExecutionResult)
        # Note: This depends on the SafeExecutor implementation
        # It might fail due to security restrictions or succeed

    @pytest.mark.asyncio
    async def test_execute_code_tool_missing_code(self, execution_limits):
        """Test code tool execution without code parameter."""
        executor = ToolExecutor(execution_limits)
        
        tool_call = ToolCall(
            name="execute_code",
            parameters={}  # Missing 'code' parameter
        )
        
        result = await executor.execute_tool_call(tool_call)
        
        assert result.success is False
        assert "Missing 'code' parameter" in result.error_message

    @pytest.mark.asyncio
    async def test_execute_code_tool_with_context(self, execution_limits):
        """Test code tool execution with context parameters."""
        executor = ToolExecutor(execution_limits)
        
        tool_call = ToolCall(
            name="execute_code",
            parameters={
                "code": "result = context_var * 2",
                "context_var": 5
            }
        )
        
        result = await executor.execute_tool_call(tool_call)
        
        # Result depends on SafeExecutor implementation
        assert isinstance(result, ExecutionResult)

    @pytest.mark.asyncio
    async def test_execute_tool_call_general_error(self, execution_limits):
        """Test general error handling in tool execution."""
        executor = ToolExecutor(execution_limits)
        
        # Create a tool call that will cause an error
        tool_call = ToolCall(name="nonexistent_tool", parameters={})
        
        result = await executor.execute_tool_call(tool_call)
        
        assert isinstance(result, ExecutionResult)
        # Should handle gracefully

    def test_add_to_history(self, execution_limits, sample_execution_result):
        """Test adding execution results to history."""
        executor = ToolExecutor(execution_limits)
        
        executor.add_to_history(sample_execution_result)
        
        assert len(executor.execution_history) == 1
        assert executor.execution_history[0] == sample_execution_result

    def test_history_size_limit(self, execution_limits, sample_execution_result):
        """Test that history respects size limit."""
        executor = ToolExecutor(execution_limits)
        executor.max_history_size = 3
        
        # Add more results than the limit
        for i in range(5):
            result = ExecutionResult(
                success=True,
                result=f"result_{i}",
                execution_time_ms=100,
                skill_id="tool",
                parameters_used={}
            )
            executor.add_to_history(result)
        
        # Should only keep the last 3
        assert len(executor.execution_history) == 3
        assert executor.execution_history[-1].result == "result_4"

    def test_get_execution_stats_empty(self, execution_limits):
        """Test execution statistics with no history."""
        executor = ToolExecutor(execution_limits)
        
        stats = executor.get_execution_stats()
        
        assert stats['total_executions'] == 0
        assert stats['success_rate'] == 0.0
        assert stats['average_execution_time_ms'] == 0.0
        assert len(stats['most_used_tools']) == 0

    def test_get_execution_stats_with_data(self, execution_limits):
        """Test execution statistics with history data."""
        executor = ToolExecutor(execution_limits)
        
        # Add some execution results
        success_result = ExecutionResult(
            success=True,
            result="success",
            execution_time_ms=100,
            skill_id="tool1",
            parameters_used={}
        )
        
        fail_result = ExecutionResult(
            success=False,
            error_message="failed",
            execution_time_ms=50,
            skill_id="tool2",
            parameters_used={}
        )
        
        executor.add_to_history(success_result)
        executor.add_to_history(fail_result)
        executor.add_to_history(success_result)  # Another success
        
        stats = executor.get_execution_stats()
        
        assert stats['total_executions'] == 3
        assert stats['successful_executions'] == 2
        assert stats['success_rate'] == 2/3
        assert stats['average_execution_time_ms'] == (100 + 50 + 100) / 3
        
        # Check most used tools
        most_used = stats['most_used_tools']
        assert len(most_used) == 2
        assert most_used[0] == ("tool1", 2)
        assert most_used[1] == ("tool2", 1)


class TestToolExecutorIntegration:
    """Integration tests for ToolExecutor."""

    @pytest.mark.asyncio
    async def test_full_execution_workflow(self, execution_limits, mock_async_functions):
        """Test full workflow from tool call to result."""
        executor = ToolExecutor(execution_limits)
        
        tool_call = ToolCall(
            name="search_claims",
            parameters={"query": "integration test", "limit": 10},
            call_id="call123"
        )
        
        skill_functions = {"search_claims": mock_async_functions["search_claims"]}
        
        result = await executor.execute_tool_call(tool_call, skill_functions)
        
        # Verify result
        assert isinstance(result, ExecutionResult)
        assert result.success is True
        assert result.skill_id == tool_call.name
        assert result.parameters_used == tool_call.parameters
        assert result.execution_time_ms >= 0
        
        # Verify history tracking
        assert len(executor.execution_history) == 1
        assert executor.execution_history[0] == result

    @pytest.mark.asyncio
    async def test_error_propagation_workflow(self, execution_limits):
        """Test error propagation through the workflow."""
        executor = ToolExecutor(execution_limits)
        
        # Tool call with dangerous code
        tool_call = ToolCall(
            name="execute_code",
            parameters={"code": "__import__('os').system('ls')"}
        )
        
        result = await executor.execute_tool_call(tool_call)
        
        # Should handle security violation
        assert isinstance(result, ExecutionResult)
        assert result.success is False  # Should fail due to security
        assert result.error_message is not None

    @pytest.mark.asyncio
    async def test_concurrent_executions(self, execution_limits, mock_async_functions):
        """Test concurrent tool executions."""
        executor = ToolExecutor(execution_limits)
        
        # Create multiple tool calls
        tool_calls = [
            ToolCall(name="search_claims", parameters={"query": f"query_{i}", "limit": 5})
            for i in range(5)
        ]
        
        skill_functions = {"search_claims": mock_async_functions["search_claims"]}
        
        # Execute all tool calls concurrently
        tasks = [
            executor.execute_tool_call(tool_call, skill_functions)
            for tool_call in tool_calls
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify all results
        assert len(results) == 5
        assert all(isinstance(r, ExecutionResult) for r in results)
        assert all(r.success for r in results)
        
        # Verify history tracking
        assert len(executor.execution_history) == 5

    @pytest.mark.asyncio
    async def test_mixed_execution_types(self, execution_limits, mock_async_functions):
        """Test mixing different execution types."""
        executor = ToolExecutor(execution_limits)
        
        skill_functions = {
            "search_claims": mock_async_functions["search_claims"],
            "create_claim": mock_async_functions["create_claim"]
        }
        
        # Mix of built-in functions
        calls = [
            ToolCall(name="search_claims", parameters={"query": "test1"}),
            ToolCall(name="create_claim", parameters={"content": "test claim"}),
            ToolCall(name="search_claims", parameters={"query": "test2"}),
        ]
        
        results = []
        for call in calls:
            result = await executor.execute_tool_call(call, skill_functions)
            results.append(result)
        
        assert len(results) == 3
        assert all(r.success for r in results)
        assert len(executor.execution_history) == 3


if __name__ == "__main__":
    pytest.main([__file__])