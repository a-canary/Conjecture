"""
Standalone Tool Validator Test
Tests the tool validation functionality independently
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from processing.dynamic_tool_creator import ToolValidator


class TestToolValidatorStandalone:
    """Standalone test for tool validation functionality"""

    def test_safe_code_validation(self):
        """Test validation of safe code"""
        validator = ToolValidator()

        safe_code = '''
def execute(param: str) -> dict:
    """Execute a safe operation"""
    return {"success": True, "result": param}
'''

        is_valid, issues = validator.validate_tool_code(safe_code)
        assert is_valid, f"Safe code should be valid: {issues}"
        print("✅ Safe code validation passed")

    def test_unsafe_code_validation(self):
        """Test validation of unsafe code"""
        validator = ToolValidator()

        unsafe_code = '''
import os

def execute(param: str) -> dict:
    """Execute with dangerous import"""
    return {"success": True, "result": param}
'''

        is_valid, issues = validator.validate_tool_code(unsafe_code)
        assert not is_valid, "Unsafe code should not be valid"
        assert len(issues) > 0
        print("✅ Unsafe code validation passed")

    def test_missing_function_validation(self):
        """Test validation of code missing required function"""
        validator = ToolValidator()

        incomplete_code = '''
def helper_function(param: str) -> dict:
    """Helper function"""
    return {"success": True, "result": param}
'''

        is_valid, issues = validator.validate_tool_code(incomplete_code)
        assert not is_valid, "Code missing execute function should not be valid"
        assert any("execute" in issue.lower() for issue in issues)
        print("✅ Missing function validation passed")

    def test_dangerous_imports_detection(self):
        """Test detection of dangerous imports"""
        validator = ToolValidator()

        dangerous_imports = [
            "import subprocess",
            "import sys",
            "import eval",
            "from os import *",
            "import exec",
        ]

        for dangerous_import in dangerous_imports:
            code = f"""
{dangerous_import}

def execute(param: str) -> dict:
    return {{"success": True}}
"""

            is_valid, issues = validator.validate_tool_code(code)
            assert not is_valid, f"Code with {dangerous_import} should not be valid"

        print("✅ Dangerous imports detection passed")

    def test_allowed_modules_validation(self):
        """Test validation with allowed modules"""
        validator = ToolValidator()

        safe_code = '''
import math
import datetime
import json
import re
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

        is_valid, issues = validator.validate_tool_code(safe_code)
        assert is_valid, f"Code with allowed modules should be valid: {issues}"
        print("✅ Allowed modules validation passed")


if __name__ == "__main__":
    # Run standalone tests
    test_instance = TestToolValidatorStandalone()

    print("🧪 Running Tool Validator Standalone Tests")
    print("=" * 50)

    try:
        test_instance.test_safe_code_validation()
        test_instance.test_unsafe_code_validation()
        test_instance.test_missing_function_validation()
        test_instance.test_dangerous_imports_detection()
        test_instance.test_allowed_modules_validation()

        print("\n🎉 All Tool Validator tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
