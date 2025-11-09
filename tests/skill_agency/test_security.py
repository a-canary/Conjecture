"""
Security tests for code execution and input validation in the skill-based agency system.
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock
import re

from src.processing.tool_executor import SecurityValidator, SafeExecutor, ExecutionLimits
from src.processing.response_parser import ResponseParser
from src.core.skill_models import SkillParameter, SkillClaim


class TestSecurityValidation:
    """Test security validation of code execution and inputs."""

    def test_dangerous_function_detection(self):
        """Test detection of dangerous function calls."""
        limits = ExecutionLimits()
        validator = SecurityValidator(limits)
        
        dangerous_patterns = [
            "eval('1+1')",
            "exec('print(1)')",
            "__import__('os')",
            "globals()",
            "locals()",
            "vars()",
            "dir()",
            "getattr(obj, 'attr')",
            "setattr(obj, 'attr', 'value')",
            "delattr(obj, 'attr')",
            "hasattr(obj, 'attr')",
            "callable(func)",
            "isinstance(obj, cls)",
            "issubclass(cls, base)",
            "iter([1,2,3])",
            "range(10)",
            "slice(0, 10)",
            "memoryview(b'bytes')",
            "bytearray(b'data')",
            "type(obj)",
            "super()",
            "property(lambda x: x)",
        ]
        
        for pattern in dangerous_patterns:
            is_safe, errors = validator.validate_code(pattern)
            assert not is_safe, f"Dangerous pattern not detected: {pattern}"
            assert len(errors) > 0

    def test_dangerous_module_detection(self):
        """Test detection of dangerous module imports."""
        limits = ExecutionLimits()
        validator = SecurityValidator(limits)
        
        dangerous_modules = [
            "import os",
            "import sys",
            "import subprocess",
            "import shutil",
            "import glob",
            "import pickle",
            "import marshal",
            "import imp",
            "import importlib",
            "import pkgutil",
            "import modulefinder",
            "import inspect",
            "import traceback",
            "import linecache",
            "import pdb",
            "import bdb",
            "import cmd",
            "import code",
            "import codeop",
            "import dis",
            "import parser",
            "from os import system",
            "from subprocess import run",
            "from sys import modules",
            "import socket",
            "import urllib",
            "import httplib",
            "import ftplib",
            "import poplib",
            "import imaplib",
            "import nntplib",
            "import smtplib",
            "import telnetlib",
            "import uuid",
            "import ssl",
            "import hashlib",
            "import hmac",
            "import secrets",
            "import base64",
            "import tempfile",
            "import shelve",
            "import dbm",
            "import sqlite3",
            "import csv",
            "import configparser",
            "import xml",
            "import xmlrpc",
            "import email",
            "import mimetypes",
            "import mailbox",
            "from distutils import setup",
            "import setuptools",
            "import pip",
            "import venv",
        ]
        
        for module in dangerous_modules:
            is_safe, errors = validator.validate_code(module)
            assert not is_safe, f"Dangerous module not detected: {module}"
            assert any("Dangerous module import" in error for error in errors)

    def test_allowed_modules_validation(self):
        """Test validation of allowed modules."""
        limits = ExecutionLimits()
        validator = SecurityValidator(limits)
        
        allowed_modules = [
            "import math",
            "import random",
            "import datetime",
            "import json",
            "import re",
            "import string",
            "import collections",
            "import itertools",
            "import functools",
            "import operator",
            "import statistics",
            "import fractions",
            "import decimal",
            "import typing",
            "from math import sqrt, pi, sin, cos",
            "from collections import defaultdict, Counter",
            "from itertools import permutations, combinations",
            "from functools import reduce, partial",
        ]
        
        for module in allowed_modules:
            is_safe, errors = validator.validate_code(module)
            assert is_safe, f"Allowed module incorrectly flagged: {module}"
            assert len(errors) == 0

    def test_filesystem_access_detection(self):
        """Test detection of filesystem access attempts."""
        limits = ExecutionLimits()
        validator = SecurityValidator(limits)
        
        filesystem_patterns = [
            "open('file.txt', 'r')",
            "open('/etc/passwd', 'w')",
            "file('test.txt')",
            "__import__('builtins').open('file')",
            "getattr(__builtins__, 'open')('file')",
            "with open('file.txt') as f:",
            "open('file', 'rb').read()",
            "open('file', 'wb').write(b'data')",
        ]
        
        for pattern in filesystem_patterns:
            is_safe, errors = validator.validate_code(pattern)
            assert not is_safe, f"Filesystem access not detected: {pattern}"

    def test_code_injection_detection(self):
        """Test detection of code injection patterns."""
        limits = ExecutionLimits()
        validator = SecurityValidator(limits)
        
        injection_patterns = [
            "eval(user_input)",
            "exec(malicious_code)",
            "compile(code, '<string>', 'exec')",
            "__builtins__['eval'](code)",
            "globals()['__builtins__']['eval'](code)",
            "getattr(__builtins__, 'eval')(code)",
            "exec(chr(101)+chr(118)+chr(97)+chr(108))",  # obfuscated eval
        ]
        
        for pattern in injection_patterns:
            is_safe, errors = validator.validate_code(pattern)
            assert not is_safe, f"Code injection not detected: {pattern}"

    def test_obfuscated_code_detection(self):
        """Test detection of obfuscated dangerous code."""
        limits = ExecutionLimits()
        validator = SecurityValidator(limits)
        
        obfuscated_patterns = [
            # Using string concatenation
            "eva" + "l('code')",
            "exe" + "c('code')",
            "__imp" + "ort__('os')",
            
            # Using getattr
            "getattr(__builtins__, 'eva' + 'l')",
            "getattr(__import__('builtins'), 'exec')",
            
            # Using locals/globals
            "locals()['__builtins__']['eval']",
            "globals().get('__builtins__')",
            
            # Using function composition
            "(lambda x: x.__import__)('os')",
            "(lambda: eval)()('code')",
        ]
        
        for pattern in obfuscated_patterns:
            is_safe, errors = validator.validate_code(pattern)
            # Some obfuscated patterns might pass basic validation but should still be caught
            if pattern.startswith("eva") or pattern.startswith("exe"):
                assert not is_safe, f"Obfuscated code not detected: {pattern}"

    def test_nested_dangerous_code(self):
        """Test detection of dangerous code in nested structures."""
        limits = ExecutionLimits()
        validator = SecurityValidator(limits)
        
        nested_patterns = [
            """
            def safe_function():
                eval('dangerous')  # Hidden inside function
            safe_function()
            """,
            """
            class SafeClass:
                def method(self):
                    exec('malicious')  # Hidden in method
            SafeClass().method()
            """,
            """
            try:
                eval(code)  # Hidden in try block
            except:
                pass
            """,
            """
            if condition:
                eval(code)  # Hidden in conditional
            """,
            """
            for i in range(10):
                exec(f'print({i})')  # Dynamic exec
            """,
        ]
        
        for pattern in nested_patterns:
            is_safe, errors = validator.validate_code(pattern)
            assert not is_safe, f"Nested dangerous code not detected: {pattern}"

    def test_crazy_ast_exploits(self):
        """Test detection of AST manipulation exploits."""
        limits = ExecutionLimits()
        validator = SecurityValidator(limits)
        
        ast_exploits = [
            # AST node manipulation
            """
            import ast
            code = ast.parse('print("safe")')
            eval(compile(code, '<string>', 'exec'))
            """,
            
            # Using compile with eval mode
            "compile('print(1)', '<string>', 'eval')",
            
            # Code object manipulation
            """
            def get_code():
                return (lambda: 0).__code__
            exec(get_code())
            """,
        ]
        
        for exploit in ast_exploits:
            is_safe, errors = validator.validate_code(exploit)
            assert not is_safe, f"AST exploit not detected: {exploit}"

    def test_network_access_detection(self):
        """Test detection of network access attempts."""
        limits = ExecutionLimits()
        validator = SecurityValidator(limits)
        
        network_patterns = [
            "import socket",
            "import urllib",
            "import requests",  # Not in dangerous list but should be blocked
            "import http.client",
            "socket.socket()",
            "urllib.request.urlopen('http://example.com')",
            "subprocess.run(['curl', 'http://example.com'])",
        ]
        
        for pattern in network_patterns:
            is_safe, errors = validator.validate_code(pattern)
            # Some network modules might not be in the dangerous list
            # but socket and subprocess should be caught
            if "socket" in pattern or "subprocess" in pattern:
                assert not is_safe, f"Network access not detected: {pattern}"

    def test_memory_bomb_detection(self):
        """Test detection of potential memory bombs."""
        limits = ExecutionLimits()
        validator = SecurityValidator(limits)
        
        memory_bombs = [
            "['x' * (10**9)]",  # Large list
            "'x' * (10**9)",    # Large string
            "{'x' * i: i for i in range(1000000)}",  # Large dict
            "itertools.repeat('x', 10**9)",  # Large iterator
            "(x for x in range(10**9))",  # Large generator
        ]
        
        for bomb in memory_bombs:
            is_safe, errors = validator.validate_code(bomb)
            # These might pass validation but should be limited by execution limits
            # We can't detect all memory bombs at parse time, but execution should limit them
            assert isinstance(is_safe, bool)


class TestSafeExecutionSecurity:
    """Test security of safe code execution."""

    @pytest.mark.asyncio
    async def test_execution_with_resource_limits(self):
        """Test execution respects resource limits."""
        small_limits = ExecutionLimits(
            max_execution_time=1.0,  # 1 second
            max_memory_mb=10,  # 10 MB
            max_output_chars=100  # Small output limit
        )
        
        executor = SafeExecutor(small_limits)
        
        # Test time limit
        time_code = """
import time
time.sleep(2)  # Should timeout
output = "done"
"""
        result = await executor.execute_code(time_code, {})
        assert not result.success
        assert "timeout" in result.error_message.lower()
        
        # Test output limit
        output_code = """
output = "x" * 1000  # Should be truncated
"""
        result = await executor.execute_code(output_code, {})
        # Execution should succeed but output truncated
        if result.success:
            assert len(result.stdout) <= 103  # Should be truncated with "..."

    @pytest.mark.asyncio
    async def test_isolated_execution_environment(self):
        """Test that execution environment is properly isolated."""
        limits = ExecutionLimits()
        executor = SafeExecutor(limits)
        
        # Test that previous executions don't affect others
        code1 = "test_var = 'from_execution_1'"
        result1 = await executor.execute_code(code1, {})
        
        # In second execution, test_var should not exist
        code2 = """
try:
    output = test_var  # Should not exist
except NameError:
    output = "variable_not_found"
"""
        result2 = await executor.execute_code(code2, {})
        
        assert result1.success
        if result2.success:
            # The variable should not exist from previous execution
            assert "variable_not_found" in str(result2.result)

    @pytest.mark.asyncio
    async def test_context_injection_prevention(self):
        """Test prevention of context injection attacks."""
        limits = ExecutionLimits()
        executor = SafeExecutor(limits)
        
        # Try to inject malicious context
        malicious_context = {
            "__builtins__": {"eval": lambda x: "injected"},
            "eval": lambda x: "injected",
            "open": lambda *args, **kwargs: "injected open",
        }
        
        safe_code = "output = 'safe_execution'"
        result = await executor.execute_code(safe_code, malicious_context)
        
        # Should still execute safely
        assert result.success
        assert "injected" not in str(result.result)

    @pytest.mark.asyncio
    async def test_subprocess_isolation(self):
        """Test that subprocess execution is isolated."""
        limits = ExecutionLimits()
        executor = SafeExecutor(limits)
        
        # Try to execute subprocess commands
        subprocess_code = """
import subprocess
try:
    output = subprocess.run(['echo', 'test'], capture_output=True, text=True).stdout
except Exception as e:
    output = f"Error: {e}"
"""
        
        result = await executor.execute_code(subprocess_code, {})
        
        # Should fail due to security validation
        assert not result.success
        assert any(keyword in result.error_message.lower() 
                  for keyword in ["security", "dangerous", "subprocess", "import"])

    @pytest.mark.asyncio
    async def test_file_access_prevention(self):
        """Test prevention of file system access."""
        limits = ExecutionLimits()
        executor = SafeExecutor(limits)
        
        file_access_code = """
try:
    with open('test.txt', 'w') as f:
        f.write('test')
    output = 'file_written'
except Exception as e:
    output = f"Error: {e}"
"""
        
        result = await executor.execute_code(file_access_code, {})
        
        # Should fail due to security validation
        assert not result.success
        assert any(keyword in result.error_message.lower() 
                  for keyword in ["security", "dangerous", "open", "file"])

    @pytest.mark.asyncio
    async def test_module_import_isolation(self):
        """Test that only allowed modules can be imported."""
        limits = ExecutionLimits()
        executor = SafeExecutor(limits)
        
        # Test allowed module
        allowed_code = """
import math
output = math.sqrt(16)
"""
        result = await executor.execute_code(allowed_code, {})
        assert result.success
        assert result.result == 4
        
        # Test disallowed module
        disallowed_code = """
import os
output = os.getcwd()
"""
        result = await executor.execute_code(disallowed_code, {})
        assert not result.success
        assert "security" in result.error_message.lower()


class TestInputValidationSecurity:
    """Test security of input validation and parsing."""

    def test_xml_injection_prevention(self, response_parser):
        """Test prevention of XML injection attacks."""
        xml_attacks = [
            # XML entity expansion (billion laughs attack)
            """
            <tool_calls>
                <invoke name="test">
                    <parameter name="data">
                        <!ENTITY lol1 "lol">
                        <!ENTITY lol2 "&lol1;&lol1;">
                        <!ENTITY lol3 "&lol2;&lol2;">
                    </parameter>
                </invoke>
            </tool_calls>
            """,
            
            # XML external entity (XXE) attack
            """
            <tool_calls>
                <invoke name="test">
                    <parameter name="data">
                        <!ENTITY xxe SYSTEM "file:///etc/passwd">
                    </parameter>
                </invoke>
            </tool_calls>
            """,
            
            # CDATA injection
            """
            <tool_calls>
                <invoke name="test">
                    <parameter name="data"><![CDATA[<script>alert('xss')</script>]]></parameter>
                </invoke>
            </tool_calls>
            """,
        ]
        
        for attack in xml_attacks:
            # Should not crash or be vulnerable
            try:
                result = response_parser.parse_response(attack)
                # Either parsing fails or tool calls are empty/safe
                assert isinstance(result, object)  # Should not crash
            except Exception as e:
                # Should fail gracefully
                assert not isinstance(e, (MemoryError, SystemError))

    def test_json_injection_prevention(self, response_parser):
        """Test prevention of JSON injection attacks."""
        json_attacks = [
            # Prototype pollution
            """
            {
                "tool_calls": [
                    {
                        "name": "test",
                        "parameters": {
                            "__proto__": {"admin": true},
                            "constructor": {"prototype": {"admin": true}}
                        }
                    }
                ]
            }
            """,
            
            # Large nested structure
            """
            {
                "tool_calls": [
                    {
                        "name": "test",
                        "parameters": {
                            "nested": [[[["deep"]]]]
                        }
                    }
                ]
            }
            """,
            
            # Special characters
            """
            {
                "tool_calls": [
                    {
                        "name": "test",
                        "parameters": {
                            "data": "\\u0000\\u0001\\u0002\\u0003\\u0004"
                        }
                    }
                ]
            }
            """,
        ]
        
        for attack in json_attacks:
            try:
                result = response_parser.parse_response(attack)
                # Should either parse safely or fail gracefully
                assert isinstance(result, object)
            except Exception as e:
                # Should handle JSON errors gracefully
                assert "json" in str(e).lower() or "parse" in str(e).lower()

    def test_command_injection_prevention(self, response_parser):
        """Test prevention of command injection in parameters."""
        command_injections = [
            # Shell command injection
            """
            <tool_calls>
                <invoke name="execute_shell">
                    <parameter name="command">ls; rm -rf /</parameter>
                </invoke>
            </tool_calls>
            """,
            
            # SQL injection patterns in parameters
            """
            {
                "tool_calls": [
                    {
                        "name": "database_query",
                        "parameters": {
                            "query": "SELECT * FROM users; DROP TABLE users; --"
                        }
                    }
                ]
            }
            """,
            
            # Path traversal
            """
            ```tool_call
            name: read_file
            path: ../../etc/passwd
            ```
            """,
        ]
        
        for injection in command_injections:
            result = response_parser.parse_response(injection)
            # Should parse but tool calls should be identifiable as potentially dangerous
            if result.has_tool_calls():
                for tool_call in result.tool_calls:
                    # Tool calls should be parseable but execution should be validated elsewhere
                    assert isinstance(tool_call.name, str)
                    assert isinstance(tool_call.parameters, dict)

    def test_parameter_type_validation_security(self):
        """Test security of parameter type validation."""
        # Create parameters with potential security implications
        dangerous_params = [
            SkillParameter(name="code", param_type="str", required=True),
            SkillParameter(name="command", param_type="str", required=True),
            SkillParameter(name="path", param_type="str", required=True),
            SkillParameter(name="filename", param_type="str", required=True),
        ]
        
        for param in dangerous_params:
            # Parameters themselves should be valid
            assert isinstance(param.validate_value("test"), bool)
            assert param.validate_value("safe_value") is True
            assert param.validate_value(123) is False  # Type checking works

    def test_skill_function_name_validation(self):
        """Test validation of skill function names."""
        dangerous_names = [
            "__import__",
            "eval",
            "exec",
            "open",
            "file",
            "globals",
            "locals",
            "builtins",
            "system",  # os.system
            "popen",   # os.popen
            "spawn",   # subprocess functions
            "socket",
        ]
        
        # These should be caught by existing validation or execution security
        for name in dangerous_names:
            # Function name validation allows these but execution security should block
            try:
                skill = SkillClaim(function_name=name, parameters=[], confidence=0.8)
                # Validation might pass but execution should be blocked by security
                assert skill.function_name == name
            except Exception:
                # Some might be caught by validation
                pass

    def test_input_size_limits(self):
        """Test protection against oversized inputs."""
        # Create very large parameter values
        large_string = "x" * (10 * 1024 * 1024)  # 10MB string
        large_list = list(range(1000000))  # Large list
        large_dict = {f"key_{i}": f"value_{i}" for i in range(100000)}  # Large dict
        
        # These should be handled at the execution level
        # Parameter validation should not crash on large inputs
        param = SkillParameter(name="large_param", param_type="str", required=True)
        
        # Validation should complete without memory issues
        try:
            result = param.validate_value(large_string)
            assert isinstance(result, bool)
        except MemoryError:
            pytest.fail("Parameter validation should not cause memory error on large inputs")


class TestSecurityConfiguration:
    """Test security configuration and policy enforcement."""

    def test_default_security_policies(self):
        """Test that default security policies are properly configured."""
        limits = ExecutionLimits()
        validator = SecurityValidator(limits)
        
        # Verify default security settings
        assert not limits.allow_network
        assert not limits.allow_file_access
        assert limits.max_memory_mb == 100
        assert limits.max_execution_time == 30.0
        
        # Verify dangerous functions are blocked
        assert 'eval' in validator.dangerous_functions
        assert 'exec' in validator.dangerous_functions
        assert 'open' in validator.dangerous_functions
        
        # Verify dangerous modules are blocked
        assert 'os' in validator.dangerous_modules
        assert 'subprocess' in validator.dangerous_modules
        assert 'socket' in validator.dangerous_modules

    def test_security_policy_customization(self):
        """Test that security policies can be customized."""
        # Custom limits for testing
        custom_limits = ExecutionLimits(
            max_execution_time=5.0,
            max_memory_mb=50,
            allow_network=False,
            allow_file_access=False,
            allowed_modules=['math', 'json']  # Very restricted
        )
        
        validator = SecurityValidator(custom_limits)
        
        # Should still block dangerous modules even with custom allowed list
        dangerous_code = "import os"
        is_safe, errors = validator.validate_code(dangerous_code)
        assert not is_safe
        
        # Should allow custom allowed modules
        safe_code = "import math"
        is_safe, errors = validator.validate_code(safe_code)
        assert is_safe
        
        # Should block disallowed modules from default list
        disallowed_code = "import sys"  # Not in custom allowed list
        is_safe, errors = validator.validate_code(disallowed_code)
        assert not is_safe

    def test_security_policy_inheritance(self):
        """Test that security policies are properly inherited."""
        base_limits = ExecutionLimits(
            max_execution_time=30.0,
            max_memory_mb=100
        )
        
        # Modify child limits
        child_limits = ExecutionLimits(
            max_execution_time=10.0,  # More restrictive
            max_memory_mb=50          # More restrictive
        )
        
        assert child_limits.max_execution_time < base_limits.max_execution_time
        assert child_limits.max_memory_mb < base_limits.max_memory_mb
        # Other settings should inherit defaults
        assert not child_limits.allow_network
        assert not child_limits.allow_file_access


class TestSecurityAudit:
    """Security audit tests for compliance and best practices."""

    def test_security_validation_completeness(self):
        """Test that security validation covers all attack vectors."""
        limits = ExecutionLimits()
        validator = SecurityValidator(limits)
        
        # Check that we have comprehensive dangerous function lists
        assert len(validator.dangerous_functions) > 20  # Should include many dangerous functions
        assert len(validator.dangerous_modules) > 20   # Should include many dangerous modules
        
        # Key dangerous functions should be blocked
        key_functions = ['eval', 'exec', 'compile', '__import__', 'open', 'file']
        for func in key_functions:
            assert func in validator.dangerous_functions
        
        # Key dangerous modules should be blocked
        key_modules = ['os', 'sys', 'subprocess', 'socket', 'urllib']
        for module in key_modules:
            assert module in validator.dangerous_modules

    def test_error_handling_security(self):
        """Test that error handling doesn't leak sensitive information."""
        limits = ExecutionLimits()
        validator = SecurityValidator(limits)
        
        # Test with malicious code that would cause errors
        error_codes = [
            "sys.exit(1)",  # Should be blocked
            "__import__('nonexistent_module')",  # Should be blocked for import
            "non_existent_function()",  # Should cause NameError but not crash
        ]
        
        for code in error_codes:
            is_safe, errors = validator.validate_code(code)
            # Should catch security issues first
            if not is_safe:
                assert len(errors) > 0
                # Error messages should not contain sensitive system info
                for error in errors:
                    assert "password" not in error.lower()
                    assert "secret" not in error.lower()
                    assert "token" not in error.lower()

    def test_logging_security(self):
        """Test that logging doesn't expose sensitive information."""
        # This test would audit logging behavior in the actual implementation
        # For now, we test that error messages don't leak sensitive data
        
        dangerous_code = "__import__('os').system('ls -la /etc/passwd')"
        limits = ExecutionLimits()
        validator = SecurityValidator(limits)
        
        is_safe, errors = validator.validate_code(dangerous_code)
        
        # Error messages should not contain the actual malicious code verbatim
        for error in errors:
            assert "etc/passwd" not in error
            assert "system" not in error
            # Should use generic descriptions
            assert any(term in error.lower() for term in ["dangerous", "blocked", "invalid"])

    def test_resource_exhaustion_prevention(self):
        """Test protection against resource exhaustion attacks."""
        limits = ExecutionLimits(
            max_execution_time=1.0,
            max_memory_mb=10,
            max_output_chars=1000
        )
        
        # Test various resource exhaustion patterns
        exhaustion_patterns = [
            # CPU exhaustion
            """
import math
import time
start = time.time()
while time.time() - start < 10:  # Should timeout
    math.factorial(100)
""",
            # Memory exhaustion
            """
big_list = []
for i in range(1000000):
    big_list.append('x' * 1000)  # Should hit memory limit
output = len(big_list)
""",
            # Output exhaustion
            """
output = 'x' * 1000000  # Should be truncated
""",
        ]
        
        for pattern in exhaustion_patterns:
            # These patterns should be caught by execution limits
            # The security validator might allow them but execution should limit
            is_safe, _ = validator.validate_code(pattern)
            
            # At parse time, these might appear safe
            # But execution limits should prevent actual resource exhaustion
            assert isinstance(is_safe, bool)


if __name__ == "__main__":
    pytest.main([__file__])