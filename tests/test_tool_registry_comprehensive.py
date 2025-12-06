#!/usr/bin/env python3
"""
Comprehensive Tests for Tool Registry System
Tests tool registration, discovery, execution, and error handling
"""

import pytest
import sys
import os
import tempfile
import inspect
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Callable

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.tools.registry import (
    ToolInfo, ToolRegistry, register_tool, get_tool_registry
)


class TestToolInfo:
    """Test ToolInfo dataclass"""
    
    def test_tool_info_creation(self):
        """Test creating ToolInfo"""
        def sample_func(x: int, y: str) -> str:
            """Sample function for testing"""
            return f"{x}:{y}"
        
        tool_info = ToolInfo(
            name="sample_tool",
            func=sample_func,
            description="Sample function for testing",
            signature="(x: int, y: str) -> str",
            is_core=True,
            module="test_module"
        )
        
        assert tool_info.name == "sample_tool"
        assert tool_info.func == sample_func
        assert tool_info.description == "Sample function for testing"
        assert tool_info.signature == "(x: int, y: str) -> str"
        assert tool_info.is_core == True
        assert tool_info.module == "test_module"
    
    def test_tool_info_equality(self):
        """Test ToolInfo equality"""
        def sample_func():
            pass
        
        tool_info1 = ToolInfo(
            name="tool",
            func=sample_func,
            description="desc",
            signature="()",
            is_core=False,
            module="mod"
        )
        
        tool_info2 = ToolInfo(
            name="tool",
            func=sample_func,
            description="desc",
            signature="()",
            is_core=False,
            module="mod"
        )
        
        # ToolInfo should be comparable by attributes
        assert tool_info1.name == tool_info2.name
        assert tool_info1.description == tool_info2.description


class TestToolRegistry:
    """Test ToolRegistry class"""
    
    @pytest.fixture
    def registry(self):
        """Create a fresh ToolRegistry instance"""
        return ToolRegistry()
    
    def test_registry_initialization(self, registry):
        """Test registry initialization"""
        assert isinstance(registry.core_tools, dict)
        assert isinstance(registry.optional_tools, dict)
        assert len(registry.core_tools) >= 0
        assert len(registry.optional_tools) >= 0
    
    def test_discover_tools_missing_directory(self, registry):
        """Test tool discovery when tools directory doesn't exist"""
        with patch('src.tools.registry.Path') as mock_path:
            # Mock tools directory as non-existent
            mock_tools_dir = Mock()
            mock_tools_dir.exists.return_value = False
            mock_path.return_value = mock_tools_dir
            
            # Should not raise an exception
            registry._discover_tools()
            
            # Should print warning
            # (We can't easily test print output without capturing stdout)
    
    def test_discover_tools_with_valid_files(self, registry):
        """Test tool discovery with valid tool files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock tool files
            tool_file1 = Path(temp_dir) / "tool1.py"
            tool_file2 = Path(temp_dir) / "tool2.py"
            tool_file3 = Path(temp_dir) / "__init__.py"  # Should be ignored
            
            tool_file1.write_text("# Tool 1")
            tool_file2.write_text("# Tool 2")
            tool_file3.write_text("# Init file")
            
            with patch('src.tools.registry.Path') as mock_path:
                mock_tools_dir = Path(temp_dir)
                mock_tools_dir.exists.return_value = True
                mock_tools_dir.glob.return_value = [tool_file1, tool_file2, tool_file3]
                
                with patch.object(registry, '_load_tool_module') as mock_load:
                    registry._discover_tools()
                    
                    # Should attempt to load only non-__init__ files
                    assert mock_load.call_count == 2
                    mock_load.assert_any_call(tool_file1)
                    mock_load.assert_any_call(tool_file2)
    
    def test_load_tool_module_success(self, registry):
        """Test successful tool module loading"""
        # Create a temporary module file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def sample_tool(x: int) -> str:
    \"\"\"Sample tool function\"\"\"
    return f"result: {x}"

# Attach tool info for testing
sample_tool._tool_info = ToolInfo(
    name="sample_tool",
    func=sample_tool,
    description="Sample tool function",
    signature="(x: int) -> str",
    is_core=False,
    module="test_module"
)
""")
            temp_file = Path(f.name)
        
        try:
            with patch('src.tools.registry.ToolInfo') as mock_tool_info_class:
                mock_tool_info = Mock()
                mock_tool_info_class.return_value = mock_tool_info
                
                with patch.object(registry, '_register_tool') as mock_register:
                    registry._load_tool_module(temp_file)
                    
                    # Should register the tool
                    mock_register.assert_called_once_with(mock_tool_info)
        
        finally:
            os.unlink(temp_file)
    
    def test_load_tool_module_failure(self, registry):
        """Test tool module loading failure"""
        # Create an invalid Python file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("invalid python syntax !!!")
            temp_file = Path(f.name)
        
        try:
            # Should handle the syntax error gracefully
            with patch('builtins.print') as mock_print:
                registry._load_tool_module(temp_file)
                
                # Should print warning about failed loading
                mock_print.assert_called()
                call_args = mock_print.call_args[0][0]
                assert "Failed to load tool module" in call_args
        
        finally:
            os.unlink(temp_file)
    
    def test_register_core_tool(self, registry):
        """Test registering a core tool"""
        def sample_func():
            """Sample function"""
            pass
        
        tool_info = ToolInfo(
            name="core_tool",
            func=sample_func,
            description="Core tool",
            signature="()",
            is_core=True,
            module="test_module"
        )
        
        with patch('builtins.print') as mock_print:
            registry._register_tool(tool_info)
            
            assert "core_tool" in registry.core_tools
            assert registry.core_tools["core_tool"] == tool_info
            assert "core_tool" not in registry.optional_tools
            
            # Should print registration message
            mock_print.assert_called_with("Registered Core tool: core_tool")
    
    def test_register_optional_tool(self, registry):
        """Test registering an optional tool"""
        def sample_func():
            """Sample function"""
            pass
        
        tool_info = ToolInfo(
            name="optional_tool",
            func=sample_func,
            description="Optional tool",
            signature="()",
            is_core=False,
            module="test_module"
        )
        
        with patch('builtins.print') as mock_print:
            registry._register_tool(tool_info)
            
            assert "optional_tool" in registry.optional_tools
            assert registry.optional_tools["optional_tool"] == tool_info
            assert "optional_tool" not in registry.core_tools
            
            # Should print registration message
            mock_print.assert_called_with("Registered Optional tool: optional_tool")
    
    def test_get_core_tools_context_empty(self, registry):
        """Test getting core tools context when no tools are registered"""
        context = registry.get_core_tools_context()
        
        assert context == "# Core Tools\n\nNo core tools available."
    
    def test_get_core_tools_context_with_tools(self, registry):
        """Test getting core tools context with registered tools"""
        # Add some core tools
        def func1():
            """Function 1"""
            pass
        
        def func2(x: int) -> str:
            """Function 2"""
            return str(x)
        
        tool_info1 = ToolInfo(
            name="tool1",
            func=func1,
            description="Function 1",
            signature="()",
            is_core=True,
            module="mod1"
        )
        
        tool_info2 = ToolInfo(
            name="tool2",
            func=func2,
            description="Function 2",
            signature="(x: int) -> str",
            is_core=True,
            module="mod2"
        )
        
        registry.core_tools = {"tool1": tool_info1, "tool2": tool_info2}
        
        context = registry.get_core_tools_context()
        
        assert "# Core Tools" in context
        assert "**tool1()**: Function 1" in context
        assert "**tool2(x: int) -> str**: Function 2" in context
        # Tools should be sorted alphabetically
        assert context.find("**tool1()**") < context.find("**tool2(x: int) -> str**")
    
    def test_get_available_tools_list(self, registry):
        """Test getting list of all available tools"""
        # Add tools to both registries
        def func1():
            pass
        
        def func2():
            pass
        
        def func3():
            pass
        
        tool_info1 = ToolInfo("tool1", func1, "desc1", "()", True, "mod1")
        tool_info2 = ToolInfo("tool2", func2, "desc2", "()", False, "mod2")
        tool_info3 = ToolInfo("tool3", func3, "desc3", "()", False, "mod3")
        
        registry.core_tools = {"tool1": tool_info1}
        registry.optional_tools = {"tool2": tool_info2, "tool3": tool_info3}
        
        tools_list = registry.get_available_tools_list()
        
        assert isinstance(tools_list, list)
        assert len(tools_list) == 3
        assert "tool1" in tools_list
        assert "tool2" in tools_list
        assert "tool3" in tools_list
        # Should be sorted
        assert tools_list == ["tool1", "tool2", "tool3"]
    
    def test_execute_tool_core_tool_success(self, registry):
        """Test successful execution of core tool"""
        def sample_func(x: int, y: str = "default") -> str:
            """Sample function"""
            return f"{x}:{y}"
        
        tool_info = ToolInfo(
            name="sample_tool",
            func=sample_func,
            description="Sample function",
            signature="(x: int, y: str = 'default') -> str",
            is_core=True,
            module="test_module"
        )
        
        registry.core_tools = {"sample_tool": tool_info}
        
        result = registry.execute_tool("sample_tool", {"x": 42, "y": "test"})
        
        assert result["success"] is True
        assert result["result"] == "42:test"
        assert result["tool_name"] == "sample_tool"
    
    def test_execute_tool_optional_tool_success(self, registry):
        """Test successful execution of optional tool"""
        def sample_func(data: Dict[str, Any]) -> List[str]:
            """Sample function"""
            return list(data.keys())
        
        tool_info = ToolInfo(
            name="sample_tool",
            func=sample_func,
            description="Sample function",
            signature="(data: Dict[str, Any]) -> List[str]",
            is_core=False,
            module="test_module"
        )
        
        registry.optional_tools = {"sample_tool": tool_info}
        
        result = registry.execute_tool("sample_tool", {"data": {"a": 1, "b": 2}})
        
        assert result["success"] is True
        assert set(result["result"]) == {"a", "b"}
        assert result["tool_name"] == "sample_tool"
    
    def test_execute_tool_not_found(self, registry):
        """Test execution of non-existent tool"""
        result = registry.execute_tool("non_existent_tool", {"arg": "value"})
        
        assert result["success"] is False
        assert "not found" in result["error"]
        assert "available_tools" in result
        assert isinstance(result["available_tools"], list)
    
    def test_execute_tool_type_error(self, registry):
        """Test tool execution with type error"""
        def sample_func(x: int) -> str:
            """Sample function requiring int"""
            return str(x)
        
        tool_info = ToolInfo(
            name="sample_tool",
            func=sample_func,
            description="Sample function",
            signature="(x: int) -> str",
            is_core=True,
            module="test_module"
        )
        
        registry.core_tools = {"sample_tool": tool_info}
        
        # Pass string instead of int
        result = registry.execute_tool("sample_tool", {"x": "not_an_int"})
        
        assert result["success"] is False
        assert "Argument error" in result["error"]
        assert "expected_signature" in result
        assert result["expected_signature"] == "(x: int) -> str"
    
    def test_execute_tool_runtime_error(self, registry):
        """Test tool execution with runtime error"""
        def sample_func():
            """Sample function that raises error"""
            raise ValueError("Something went wrong")
        
        tool_info = ToolInfo(
            name="sample_tool",
            func=sample_func,
            description="Sample function",
            signature="()",
            is_core=True,
            module="test_module"
        )
        
        registry.core_tools = {"sample_tool": tool_info}
        
        result = registry.execute_tool("sample_tool", {})
        
        assert result["success"] is False
        assert "Execution error" in result["error"]
        assert "Something went wrong" in result["error"]
    
    def test_execute_tool_with_default_parameters(self, registry):
        """Test tool execution with default parameters"""
        def sample_func(x: int, y: str = "default", z: float = 1.5) -> str:
            """Sample function with defaults"""
            return f"{x}:{y}:{z}"
        
        tool_info = ToolInfo(
            name="sample_tool",
            func=sample_func,
            description="Sample function",
            signature="(x: int, y: str = 'default', z: float = 1.5) -> str",
            is_core=True,
            module="test_module"
        )
        
        registry.core_tools = {"sample_tool": tool_info}
        
        # Only provide required parameter
        result = registry.execute_tool("sample_tool", {"x": 42})
        
        assert result["success"] is True
        assert result["result"] == "42:default:1.5"
    
    def test_get_tool_info_core_tool(self, registry):
        """Test getting info about core tool"""
        def sample_func():
            """Sample function"""
            pass
        
        tool_info = ToolInfo(
            name="sample_tool",
            func=sample_func,
            description="Sample function",
            signature="()",
            is_core=True,
            module="test_module"
        )
        
        registry.core_tools = {"sample_tool": tool_info}
        
        result = registry.get_tool_info("sample_tool")
        
        assert result is not None
        assert result.name == "sample_tool"
        assert result.description == "Sample function"
        assert result.is_core is True
    
    def test_get_tool_info_optional_tool(self, registry):
        """Test getting info about optional tool"""
        def sample_func():
            """Sample function"""
            pass
        
        tool_info = ToolInfo(
            name="sample_tool",
            func=sample_func,
            description="Sample function",
            signature="()",
            is_core=False,
            module="test_module"
        )
        
        registry.optional_tools = {"sample_tool": tool_info}
        
        result = registry.get_tool_info("sample_tool")
        
        assert result is not None
        assert result.name == "sample_tool"
        assert result.description == "Sample function"
        assert result.is_core is False
    
    def test_get_tool_info_not_found(self, registry):
        """Test getting info about non-existent tool"""
        result = registry.get_tool_info("non_existent_tool")
        
        assert result is None


class TestRegisterToolDecorator:
    """Test register_tool decorator"""
    
    def test_register_tool_decorator_default(self):
        """Test register_tool decorator with default parameters"""
        @register_tool
        def sample_function(x: int, y: str) -> str:
            """Sample function for testing"""
            return f"{x}:{y}"
        
        # Check that tool info was attached
        assert hasattr(sample_function, '_tool_info')
        
        tool_info = sample_function._tool_info
        assert tool_info.name == "sample_function"
        assert tool_info.func == sample_function
        assert tool_info.description == "Sample function for testing"
        assert tool_info.signature == "(x: int, y: str) -> str"
        assert tool_info.is_core is False  # Default
        assert tool_info.module == sample_function.__module__
    
    def test_register_tool_decorator_core(self):
        """Test register_tool decorator with core=True"""
        @register_tool(is_core=True)
        def core_function(x: int) -> int:
            """Core function for testing"""
            return x * 2
        
        tool_info = core_function._tool_info
        assert tool_info.name == "core_function"
        assert tool_info.is_core is True
        assert tool_info.description == "Core function for testing"
    
    def test_register_tool_decorator_custom_name(self):
        """Test register_tool decorator with custom name"""
        @register_tool(name="custom_tool_name")
        def sample_function():
            """Sample function"""
            pass
        
        tool_info = sample_function._tool_info
        assert tool_info.name == "custom_tool_name"
    
    def test_register_tool_decorator_no_docstring(self):
        """Test register_tool decorator with no docstring"""
        @register_tool
        def no_doc_function(x: int) -> int:
            return x * 2
        
        tool_info = no_doc_function._tool_info
        assert tool_info.description == "No description available"
    
    def test_register_tool_decorator_multiline_docstring(self):
        """Test register_tool decorator with multiline docstring"""
        @register_tool
        def multiline_function(x: int) -> int:
            """
            This is a multiline docstring.
            
            The first line should be used as description.
            
            Args:
                x: An integer to double
            """
            return x * 2
        
        tool_info = multiline_function._tool_info
        assert tool_info.description == "This is a multiline docstring."
    
    def test_register_tool_decorator_complex_signature(self):
        """Test register_tool decorator with complex function signature"""
        @register_tool
        def complex_function(
            x: int,
            y: str = "default",
            *args: Any,
            **kwargs: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Complex function signature"""
            return {"x": x, "y": y, "args": args, "kwargs": kwargs}
        
        tool_info = complex_function._tool_info
        assert "x: int" in tool_info.signature
        assert "y: str = 'default'" in tool_info.signature
        assert "*args: Any" in tool_info.signature
        assert "**kwargs: Dict[str, Any]" in tool_info.signature


class TestGetToolRegistry:
    """Test get_tool_registry function"""
    
    def test_get_tool_registry_singleton(self):
        """Test that get_tool_registry returns singleton instance"""
        registry1 = get_tool_registry()
        registry2 = get_tool_registry()
        
        assert registry1 is registry2
        assert isinstance(registry1, ToolRegistry)
    
    def test_get_tool_registry_initialization(self):
        """Test that get_tool_registry initializes registry on first call"""
        # Reset global registry
        import src.tools.registry
        src.tools.registry._tool_registry = None
        
        registry = get_tool_registry()
        
        assert isinstance(registry, ToolRegistry)
        assert src.tools.registry._tool_registry is registry


class TestToolRegistryIntegration:
    """Integration tests for tool registry"""
    
    def test_full_tool_lifecycle(self):
        """Test complete tool lifecycle: registration -> discovery -> execution"""
        # Create a temporary tools directory
        with tempfile.TemporaryDirectory() as temp_dir:
            tool_file = Path(temp_dir) / "integration_test_tool.py"
            
            # Create a tool file with registered functions
            tool_content = '''
from src.tools.registry import register_tool

@register_tool(is_core=True)
def integration_core_tool(x: int, y: str) -> str:
    """Integration test core tool"""
    return f"core:{x}:{y}"

@register_tool(name="custom_optional_tool")
def integration_optional_tool(data: dict) -> list:
    """Integration test optional tool"""
    return list(data.values())
'''
            tool_file.write_text(tool_content)
            
            # Create registry and load the tool
            registry = ToolRegistry()
            
            with patch('src.tools.registry.Path') as mock_path:
                mock_tools_dir = Path(temp_dir)
                mock_tools_dir.exists.return_value = True
                mock_tools_dir.glob.return_value = [tool_file]
                
                # Load the tool
                registry._load_tool_module(tool_file)
                
                # Check tools were registered
                assert "integration_core_tool" in registry.core_tools
                assert "custom_optional_tool" in registry.optional_tools
                
                # Test execution
                result1 = registry.execute_tool("integration_core_tool", {"x": 42, "y": "test"})
                assert result1["success"] is True
                assert result1["result"] == "core:42:test"
                
                result2 = registry.execute_tool("custom_optional_tool", {"data": {"a": 1, "b": 2}})
                assert result2["success"] is True
                assert set(result2["result"]) == {1, 2}
                
                # Test tool info
                info1 = registry.get_tool_info("integration_core_tool")
                assert info1 is not None
                assert info1.is_core is True
                
                info2 = registry.get_tool_info("custom_optional_tool")
                assert info2 is not None
                assert info2.is_core is False
                
                # Test available tools list
                tools_list = registry.get_available_tools_list()
                assert "integration_core_tool" in tools_list
                assert "custom_optional_tool" in tools_list
    
    def test_error_handling_in_discovery(self):
        """Test error handling during tool discovery"""
        # Create a temporary tools directory with both valid and invalid files
        with tempfile.TemporaryDirectory() as temp_dir:
            valid_file = Path(temp_dir) / "valid_tool.py"
            invalid_file = Path(temp_dir) / "invalid_tool.py"
            
            valid_file.write_text("""
@register_tool
def valid_tool():
    \"\"\"Valid tool\"\"\"
    pass
""")
            
            invalid_file.write_text("invalid python syntax !!!")
            
            registry = ToolRegistry()
            
            with patch('src.tools.registry.Path') as mock_path:
                mock_tools_dir = Path(temp_dir)
                mock_tools_dir.exists.return_value = True
                mock_tools_dir.glob.return_value = [valid_file, invalid_file]
                
                with patch('builtins.print') as mock_print:
                    # Should handle invalid file gracefully
                    registry._discover_tools()
                    
                    # Should print warning for invalid file
                    warning_calls = [call for call in mock_print.call_args_list 
                                  if "Failed to load tool module" in str(call)]
                    assert len(warning_calls) > 0
    
    def test_concurrent_access(self):
        """Test thread safety of tool registry"""
        import threading
        import time
        
        results = []
        errors = []
        
        def worker():
            try:
                for _ in range(10):
                    registry = get_tool_registry()
                    tools_list = registry.get_available_tools_list()
                    results.append(len(tools_list))
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should have no errors
        assert len(errors) == 0
        
        # Should have consistent results
        assert len(set(results)) <= 1  # All results should be the same


if __name__ == "__main__":
    pytest.main([__file__, "-v"])