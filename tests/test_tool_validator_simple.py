"""
Simplified Tool Validator for Testing
Core validation functionality without complex dependencies
"""

import ast
import re
from typing import List, Tuple


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

            # Check for proper docstring (simplified check)
            # Skip docstring check for now to focus on security validation
            pass

        except SyntaxError as e:
            issues.append(f"Syntax error: {e}")
        except Exception as e:
            issues.append(f"Parse error: {e}")

        return len(issues) == 0, issues


def test_tool_validator():
    """Test the tool validator functionality"""
    validator = ToolValidator()

    print("Testing Tool Validator")
    print("=" * 30)

    # Test 1: Safe code
    print("\n1. Testing safe code validation...")
    safe_code = '''
def execute(param: str) -> dict:
    """Execute a safe operation"""
    return {"success": True, "result": param}
'''

    is_valid, issues = validator.validate_tool_code(safe_code)
    assert is_valid, f"Safe code should be valid: {issues}"
    print("Safe code validation passed")

    # Test 2: Unsafe code
    print("\n2. Testing unsafe code validation...")
    unsafe_code = '''
import os

def execute(param: str) -> dict:
    """Execute with dangerous import"""
    return {"success": True, "result": param}
'''

    is_valid, issues = validator.validate_tool_code(unsafe_code)
    assert not is_valid, "Unsafe code should not be valid"
    assert len(issues) > 0
    print("Unsafe code validation passed")

    # Test 3: Missing function
    print("\n3. Testing missing function validation...")
    incomplete_code = '''
def helper_function(param: str) -> dict:
    """Helper function"""
    return {"success": True, "result": param}
'''

    is_valid, issues = validator.validate_tool_code(incomplete_code)
    assert not is_valid, "Code missing execute function should not be valid"
    assert any("execute" in issue.lower() for issue in issues)
    print("Missing function validation passed")

    # Test 4: Allowed modules
    print("\n4. Testing allowed modules validation...")
    allowed_code = '''
import math
import datetime
from typing import Dict, Any

def execute(param: str) -> Dict[str, Any]:
    """Execute with safe imports"""
    result = math.sqrt(len(param))
    return {
        "success": True,
        "result": result,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }
'''

    is_valid, issues = validator.validate_tool_code(allowed_code)
    assert is_valid, f"Code with allowed modules should be valid: {issues}"
    print("Allowed modules validation passed")

    print("\nAll Tool Validator tests passed!")


if __name__ == "__main__":
    test_tool_validator()
