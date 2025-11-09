"""
Error handling and edge case tests for the skill-based agency system.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import json
import xml.etree.ElementTree as ET

from src.processing.skill_manager import SkillManager
from src.processing.response_parser import ResponseParser
from src.processing.tool_executor import ToolExecutor, ExecutionLimits, SecurityValidator
from src.processing.example_generator import ExampleGenerator, ExampleQualityAssessor
from src.core.skill_models import (
    SkillClaim, ExampleClaim, ExecutionResult, ToolCall, 
    SkillParameter, SkillRegistry, ParsedResponse
)
from src.data.models import ClaimNotFoundError, InvalidClaimError


class TestSkillModelsEdgeCases:
    """Edge cases for skill models."""

    def test_skill_parameter_edge_cases(self):
        """Test extreme values for SkillParameter."""
        # Empty name
        with pytest.raises(Exception):  # Should validate name is not empty
            SkillParameter(name="", param_type="str", required=True)
        
        # Maximum length name
        long_name = "a" * 1000
        param = SkillParameter(name=long_name, param_type="str", required=True)
        assert param.name == long_name
        
        # Special characters in name
        special_name = "param-with_underscores.andNumbers123"
        param = SkillParameter(name=special_name, param_type="str", required=True)
        assert param.name == special_name
        
        # Type validation with None
        param = SkillParameter(name="test", param_type="str", required=False, default_value=None)
        assert param.default_value is None
        
        # Complex default values
        complex_default = {"key": [1, 2, 3], "nested": {"data": "value"}}
        param = SkillParameter(name="complex", param_type="dict", required=False, default_value=complex_default)
        assert param.default_value == complex_default

    def test_skill_claim_edge_cases(self):
        """Test extreme values for SkillClaim."""
        # Very long content
        long_content = "x" * 10000
        skill = SkillClaim(
            function_name="long_content_skill",
            content=long_content,
            parameters=[],
            confidence=0.8
        )
        assert skill.content == long_content
        
        # Special characters in function name
        special_names = [
            "skill-with-dashes",
            "skill_with_underscores",
            "skill123withNumbers",
            "SkillWithMixedCase"
        ]
        
        for name in special_names:
            skill = SkillClaim(function_name=name, parameters=[], confidence=0.8)
            assert skill.function_name == name
        
        # Edge case confidence values
        edge_confidences = [0.0, 0.01, 0.99, 1.0]
        for conf in edge_confidences:
            skill = SkillClaim(function_name="test", parameters=[], confidence=conf)
            assert skill.confidence == conf
        
        # Maximum tag list
        max_tags = ["tag"] * 1000
        skill = SkillClaim(function_name="test", parameters=[], tags=max_tags, confidence=0.8)
        assert len(skill.tags) == 1000

    def test_execution_result_edge_cases(self):
        """Test extreme values for ExecutionResult."""
        # Zero execution time
        result = ExecutionResult(
            success=True,
            result="test",
            execution_time_ms=0,
            skill_id="c123",
            parameters_used={}
        )
        assert result.execution_time_ms == 0
        
        # Very large execution time
        result = ExecutionResult(
            success=True,
            result="test",
            execution_time_ms=2**31 - 1,  # Max 32-bit int
            skill_id="c123",
            parameters_used={}
        )
        assert result.execution_time_ms == 2**31 - 1
        
        # Very large result
        large_result = "x" * 1000000
        result = ExecutionResult(
            success=True,
            result=large_result,
            execution_time_ms=100,
            skill_id="c123",
            parameters_used={}
        )
        assert result.result == large_result
        
        # Nested complex result
        complex_result = {
            "level1": {
                "level2": {
                    "level3": ["item1", "item2", "item3"]
                }
            },
            "lists": [[1, 2, 3], [4, 5, 6]],
            "mixed": [1, "string", {"nested": True}]
        }
        result = ExecutionResult(
            success=True,
            result=complex_result,
            execution_time_ms=100,
            skill_id="c123",
            parameters_used={}
        )
        assert result.result == complex_result

    def test_example_claim_edge_cases(self):
        """Test extreme values for ExampleClaim."""
        # Edge case quality values
        edge_qualities = [0.0, 0.5, 1.0]
        for quality in edge_qualities:
            example = ExampleClaim(
                skill_id="c123",
                input_parameters={},
                example_quality=quality,
                confidence=0.8
            )
            assert example.example_quality == quality
        
        # Very high usage count
        example = ExampleClaim(
            skill_id="c123",
            input_parameters={},
            usage_count=2**31 - 1,
            confidence=0.8
        )
        assert example.usage_count == 2**31 - 1
        
        # Complex parameters and results
        complex_params = {
            "nested": {"deep": {"deeper": {"data": "value"}}},
            "arrays": [[1, 2], [3, 4], [5, 6]],
            "mixed": None
        }
        complex_output = {"result": "success", "data": list(range(1000))}
        
        example = ExampleClaim(
            skill_id="c123",
            input_parameters=complex_params,
            output_result=complex_output,
            confidence=0.8
        )
        assert example.input_parameters == complex_params
        assert example.output_result == complex_output

    def test_tool_call_edge_cases(self):
        """Test extreme values for ToolCall."""
        # Empty parameters
        tool_call = ToolCall(name="test", parameters={})
        assert tool_call.parameters == {}
        
        # Very long call ID
        long_call_id = "call_" + "x" * 1000
        tool_call = ToolCall(name="test", parameters={}, call_id=long_call_id)
        assert tool_call.call_id == long_call_id
        
        # Complex parameters
        complex_params = {
            "string": "unicode: 🔥 🚀 ✨",
            "number": 3.14159265359,
            "boolean": True,
            "null": None,
            "array": [1, "two", {"three": 3}],
            "object": {"nested": {"deep": {"value": "test"}}}
        }
        tool_call = ToolCall(name="complex", parameters=complex_params)
        assert tool_call.parameters == complex_params

    def test_skill_registry_edge_cases(self):
        """Test edge cases for SkillRegistry."""
        registry = SkillRegistry()
        
        # Register skill with same name (should update)
        skill1 = SkillClaim(
            function_name="duplicate_skill",
            content="First version",
            parameters=[],
            confidence=0.8
        )
        skill2 = SkillClaim(
            function_name="duplicate_skill",
            content="Second version",
            parameters=[],
            confidence=0.9
        )
        
        registry.register_skill(skill1)
        assert len(registry.skills) == 1
        assert registry.get_skill("duplicate_skill").content == "First version"
        
        registry.register_skill(skill2)  # Should update
        assert len(registry.skills) == 1
        assert registry.get_skill("duplicate_skill").content == "Second version"
        
        # Search with empty query
        results = registry.search_skills("")
        assert isinstance(results, list)
        
        # Search with very long query
        long_query = "x" * 1000
        results = registry.search_skills(long_query)
        assert isinstance(results, list)


class TestResponseParserEdgeCases:
    """Edge cases for response parser."""

    def test_parse_malformed_xml(self, response_parser):
        """Test parsing malformed XML."""
        malformed_xml_cases = [
            # Unclosed tags
            "<tool_calls><invoke name='test'>",
            # Mismatched tags
            "<tool_calls><invoke></tool_calls>",
            # Invalid characters
            "<tool_calls><invoke name='test\0'></invoke></tool_calls>",
            # Invalid attribute syntax
            "<tool_calls><invoke name=test></invoke></tool_calls>",
            # Multiple root elements
            "<tool_calls></tool_calls><tool_calls></tool_calls>",
            # Self-closing with content
            "<invoke name='test'/>content</invoke>",
        ]
        
        for malformed in malformed_xml_cases:
            result = response_parser.parse_response(malformed)
            # Should handle gracefully (either parse partial content or return empty)
            assert isinstance(result, ParsedResponse)

    def test_parse_malformed_json(self, response_parser):
        """Test parsing malformed JSON."""
        malformed_json_cases = [
            # Trailing comma
            '{"tool_calls": [{"name": "test",}]}',
            # Unclosed braces
            '{"tool_calls": [{"name": "test"}]',
            # Unquoted keys
            "{tool_calls: [{name: 'test'}]}",
            # Invalid escape sequences
            '{"tool_calls": [{"name": "test\\x"}]}',
            # Invalid unicode
            '{"tool_calls": [{"name": "test\\uZZZZ"}]}',
            # Multiple JSON objects
            '{"name": "first"}{"name": "second"}',
        ]
        
        for malformed in malformed_json_cases:
            result = response_parser.parse_response(malformed)
            # Should handle gracefully
            assert isinstance(result, ParsedResponse)

    def test_parse_edge_case_markdown(self, response_parser):
        """Test parsing edge case markdown."""
        edge_case_markdown = [
            # Unclosed code block
            "```tool_call\nname: test\nparam: value",
            # Multiple code blocks
            "```tool_call\nname: test1\n```\n```tool_call\nname: test2\n```",
            # Empty code block
            "```\n```",
            # Code block with invalid language
            "```invalid_language\nname: test\n```",
            # Code block with no content
            "```tool_call\n```",
            # Mixed indentation
            "```tool_call\n    name: test\n  param: value\n```",
        ]
        
        for markdown in edge_case_markdown:
            result = response_parser.parse_response(markdown)
            assert isinstance(result, ParsedResponse)

    def test_parse_large_responses(self, response_parser):
        """Test parsing very large responses."""
        # Large XML response
        large_xml = "<tool_calls>"
        for i in range(1000):
            large_xml += f'<invoke name="tool_{i}"><parameter name="param_{i}">value_{i}</parameter></invoke>'
        large_xml += "</tool_calls>"
        
        result = response_parser.parse_response(large_xml)
        assert isinstance(result, ParsedResponse)
        
        # Large JSON response
        large_json = {"tool_calls": []}
        for i in range(1000):
            large_json["tool_calls"].append({
                "name": f"tool_{i}",
                "parameters": {"param": f"value_{i}"}
            })
        
        result = response_parser.parse_response(json.dumps(large_json))
        assert isinstance(result, ParsedResponse)

    def test_parse_special_characters(self, response_parser):
        """Test parsing responses with special characters."""
        special_chars = [
            # Unicode characters
            '<tool_calls><invoke name="test"><parameter name="unicode">🔥 🚀 ✨ ❤️ ✅</parameter></invoke></tool_calls>',
            # HTML entities
            '<tool_calls><invoke name="test"><parameter name="html">&lt;script&gt;alert(&quot;xss&quot;);&lt;/script&gt;</parameter></invoke></tool_calls>',
            # Whitespace variations
            '<tool_calls>\n\t<invoke name="test">\n\t\t<parameter name="whitespace">  value  \n\t</parameter>\n\t</invoke>\n</tool_calls>',
            # Null bytes and control characters
            '<tool_calls><invoke name="test"><parameter name="control">value\x00\x01\x02</parameter></invoke></tool_calls>',
        ]
        
        for chars in special_chars:
            result = response_parser.parse_response(chars)
            assert isinstance(result, ParsedResponse)

    def test_parse_empty_and_null_cases(self, response_parser):
        """Test parsing empty and null cases."""
        empty_cases = [
            None,
            "",
            "   ",
            "\n\t\r",
            "\x00",  # Null byte
        ]
        
        for empty in empty_cases:
            result = response_parser.parse_response(empty)
            assert isinstance(result, ParsedResponse)
            assert len(result.tool_calls) == 0

    def test_validate_response_structure_edge_cases(self, response_parser):
        """Test response validation with edge cases."""
        edge_cases = [
            # Valid but with extra whitespace
            "   <tool_calls><invoke name='test'></invoke></tool_calls>   ",
            # Valid with different quote styles
            "<tool_calls><invoke name=\"test\"><parameter name='param'>value</parameter></invoke></tool_calls>",
            # Empty tool_calls
            "<tool_calls></tool_calls>",
            # Tool call with no parameters
            "<tool_calls><invoke name='test'></invoke></tool_calls>",
            # Tool call with empty parameters
            "<tool_calls><invoke name='test'><parameter name='empty'></parameter></invoke></tool_calls>",
        ]
        
        for case in edge_cases:
            is_valid, errors = response_parser.validate_response_structure(case)
            # Should not crash
            assert isinstance(is_valid, bool)
            assert isinstance(errors, list)


class TestToolExecutorEdgeCases:
    """Edge cases for tool executor."""

    @pytest.mark.asyncio
    async def test_execute_with_timeout_edge_cases(self, execution_limits):
        """Test execution with various timeout scenarios."""
        # Very short timeout
        short_limits = ExecutionLimits(max_execution_time=0.01)
        executor = ToolExecutor(short_limits)
        
        code = "import time; time.sleep(1)"  # Longer than timeout
        result = await executor.execute_tool_call(
            ToolCall(name="execute_code", parameters={"code": code})
        )
        
        # Should timeout
        assert not result.success
        assert "timeout" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_execute_with_memory_limits(self, execution_limits):
        """Test execution with memory limits."""
        # Very low memory limit
        low_mem_limits = ExecutionLimits(max_memory_mb=1)
        executor = ToolExecutor(low_mem_limits)
        
        # Code that uses significant memory
        memory_code = "big_list = ['x'] * 1000000; output = len(big_list)"
        
        result = await executor.execute_tool_call(
            ToolCall(name="execute_code", parameters={"code": memory_code})
        )
        
        # Should either succeed (if limit not reached) or fail gracefully
        assert isinstance(result, ExecutionResult)

    @pytest.mark.asyncio
    async def test_execute_builtin_function_edge_cases(self, execution_limits):
        """Test executing built-in functions with edge cases."""
        executor = ToolExecutor(execution_limits)
        
        # Function that returns None
        async def none_func(**kwargs):
            return None
        
        # Function that raises different types of exceptions
        async def exception_func(**kwargs):
            raise KeyboardInterrupt("Test interrupt")
        
        # Function that returns very large result
        async def large_result_func(**kwargs):
            return {"data": "x" * 1000000}
        
        test_functions = {
            "none_func": none_func,
            "exception_func": exception_func,
            "large_result_func": large_result_func,
        }
        
        for name, func in test_functions.items():
            result = await executor._execute_builtin_function(
                ToolCall(name=name, parameters={}),
                func
            )
            
            # Should handle all edge cases gracefully
            assert isinstance(result, ExecutionResult)

    def test_security_validation_edge_cases(self, execution_limits):
        """Test security validation with edge cases."""
        validator = SecurityValidator(execution_limits)
        
        edge_case_codes = [
            # Empty code
            "",
            # Only whitespace
            "   \n\t   ",
            # Only comments
            "# This is a comment\n# Another comment",
            # Complex but safe code
            """
import math
import itertools
def complex_calculation():
    return sum(math.sqrt(i) for i in range(100))
output = complex_calculation()
""",
            # Code with Unicode
            "# Comment with unicode: 🔥 🚀 ✨\noutput = 'unicode test'",
            # Very long code
            "output = 'x' * 10000",
        ]
        
        for code in edge_case_codes:
            is_safe, errors = validator.validate_code(code)
            # Should not crash
            assert isinstance(is_safe, bool)
            assert isinstance(errors, list)

    @pytest.mark.asyncio
    async def test_execution_with_corrupted_context(self, execution_limits):
        """Test execution with corrupted or malicious context."""
        executor = ToolExecutor(execution_limits)
        
        # Malicious context
        malicious_context = {
            "__builtins__": {"eval": lambda x: "injected"},
            "exec": lambda x: None,
            "__import__": lambda x: os,
        }
        
        safe_code = "output = 'safe_execution'"
        result = await executor.safe_executor.execute_code(safe_code, malicious_context)
        
        # Should execute safely despite malicious context
        if result.success:
            assert "injected" not in str(result.result)

    @pytest.mark.asyncio
    async def test_concurrent_execution_edge_cases(self, execution_limits):
        """Test concurrent execution with edge cases."""
        executor = ToolExecutor(execution_limits)
        
        # Mix of fast and slow executions
        fast_codes = ["output = i" for i in range(5)]
        slow_codes = ["import time; time.sleep(0.1); output = 'slow'"]
        
        all_codes = fast_codes + slow_codes
        
        tasks = [
            executor.execute_tool_call(
                ToolCall(name="execute_code", parameters={"code": code})
            )
            for code in all_codes
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should complete without crashing
        assert len(results) == len(all_codes)
        for result in results:
            assert not isinstance(result, Exception)


class TestSkillManagerEdgeCases:
    """Edge cases for skill manager."""

    @pytest.mark.asyncio
    async def test_skill_registration_edge_cases(self, skill_manager):
        """Test skill registration with edge cases."""
        # Skill with no parameters (empty list vs None)
        skill1 = SkillClaim(
            function_name="no_params_skill",
            content="Skill with no parameters",
            parameters=None,
            confidence=0.8
        )
        
        skill2 = SkillClaim(
            function_name="empty_params_skill",
            content="Skill with empty parameters list",
            parameters=[],
            confidence=0.8
        )
        
        # Both should work
        await skill_manager.register_skill_claim(skill1)
        await skill_manager.register_skill_claim(skill2)
        
        assert len(skill_manager.registry.skills) == 2
        
        # Try to register invalid skill (database fails)
        invalid_skill = SkillClaim(
            function_name="invalid_skill",
            content="This should fail during database save",
            parameters=[]
        )
        
        with patch.object(skill_manager.data_manager, 'create_claim', side_effect=Exception("DB error")):
            result = await skill_manager.register_skill_claim(invalid_skill)
            assert result is False

    @pytest.mark.asyncio
    async def test_skill_execution_edge_cases(self, skill_manager, mock_async_functions):
        """Test skill execution with edge cases."""
        # Register skill
        skill = SkillClaim(
            function_name="edge_case_skill",
            content="Skill for edge case testing",
            parameters=[
                SkillParameter(name="int_param", param_type="int", required=True),
                SkillParameter(name="opt_param", param_type="str", required=False, default_value="default")
            ],
            confidence=0.8
        )
        await skill_manager.register_skill_claim(skill)
        
        # Edge case parameter sets
        edge_cases = [
            # Valid parameters
            {"int_param": 1},
            {"int_param": 1, "opt_param": "custom"},
            # Invalid parameters (these should fail gracefully)
            {"int_param": "not_int"},
            {"opt_param": "missing_required"},
            {},  # Missing required parameter
            {"int_param": None},
            {"int_param": [], "opt_param": {}},
        ]
        
        for params in edge_cases:
            result = await skill_manager.execute_skill("edge_case_skill", params)
            # Should handle all cases without crashing
            assert isinstance(result, ExecutionResult)
            if result.success:
                assert result.result is not None
            else:
                assert result.error_message is not None

    @pytest.mark.asyncio
    async def test_concurrent_skill_operations(self, skill_manager, mock_async_functions):
        """Test concurrent skill manager operations."""
        # Register multiple skills concurrently
        skills = [
            SkillClaim(
                function_name=f"concurrent_skill_{i}",
                content=f"Concurrent skill {i}",
                parameters=[],
                confidence=0.8
            )
            for i in range(10)
        ]
        
        registration_tasks = [
            skill_manager.register_skill_claim(skill)
            for skill in skills
        ]
        
        results = await asyncio.gather(*registration_tasks)
        
        # All should register successfully
        assert all(results)
        assert len(skill_manager.registry.skills) >= 10
        
        # Execute multiple skills concurrently
        for i, skill in enumerate(skills[:5]):
            skill_manager.builtin_skills[skill.function_name] = mock_async_functions["search_claims"]
        
        execution_tasks = [
            skill_manager.execute_skill(f"concurrent_skill_{i}", {})
            for i in range(5)
        ]
        
        execution_results = await asyncio.gather(*execution_tasks)
        
        # All should complete successfully
        assert all(isinstance(r, ExecutionResult) for r in execution_results)

    @pytest.mark.asyncio
    async def test_skill_discovery_edge_cases(self, skill_manager):
        """Test skill discovery with edge cases."""
        # Register skills with various content
        skills = [
            SkillClaim(
                function_name=f"skill_{i}",
                content=f"Content {i} with special chars: 🔥 🚀 ✨",
                parameters=[],
                confidence=0.8,
                tags=[f"tag_{i}", "common_tag"]
            )
            for i in range(20)
        ]
        
        for skill in skills:
            await skill_manager.register_skill_claim(skill)
        
        # Edge case queries
        edge_queries = [
            "",  # Empty query
            "   ",  # Whitespace only
            "nonexistent_query_12345",
            "🔥",  # Unicode emoji
            "a" * 1000,  # Very long query
            "common",  # Should match multiple
            "Content 10",  # Specific match
        ]
        
        for query in edge_queries:
            results = await skill_manager.find_relevant_skills(query, limit=5)
            # Should not crash
            assert isinstance(results, list)


class TestExampleGeneratorEdgeCases:
    """Edge cases for example generator."""

    @pytest.mark.asyncio
    async def test_example_generation_edge_cases(self, example_generator):
        """Test example generation with edge cases."""
        # Execution results with edge cases
        edge_results = [
            # Very fast execution
            ExecutionResult(
                success=True,
                result="fast",
                execution_time_ms=1,
                skill_id="c123",
                parameters_used={"param": "test"}
            ),
            # Very slow execution
            ExecutionResult(
                success=True,
                result="slow",
                execution_time_ms=60000,  # 60 seconds
                skill_id="c123",
                parameters_used={"param": "test"}
            ),
            # Very large result
            ExecutionResult(
                success=True,
                result="x" * 10000,
                execution_time_ms=100,
                skill_id="c123",
                parameters_used={"param": "test"}
            ),
            # Complex nested result
            ExecutionResult(
                success=True,
                result={"nested": {"deep": {"data": list(range(1000))}}},
                execution_time_ms=200,
                skill_id="c123",
                parameters_used={"param": "test"}
            ),
            # Result with special characters
            ExecutionResult(
                success=True,
                result="Special chars: 🔥 🚀 ✨ and unicode: 中文",
                execution_time_ms=50,
                skill_id="c123",
                parameters_used={"param": "test"}
            ),
        ]
        
        example_generator.get_examples_for_skill = AsyncMock(return_value=[])
        
        for result in edge_results:
            example = await example_generator.generate_example_from_execution(result)
            # Should handle all edge cases gracefully
            if example:
                assert isinstance(example, ExampleClaim)
                assert example.skill_id == result.skill_id

    def test_quality_assessment_edge_cases(self, example_generator):
        """Test quality assessment with edge cases."""
        assessor = example_generator.quality_assessor
        
        # Edge case execution results
        edge_results = [
            # Zero execution time
            ExecutionResult(
                success=True,
                result="instant",
                execution_time_ms=0,
                skill_id="c123",
                parameters_used={}
            ),
            # Very high execution time
            ExecutionResult(
                success=True,
                result="took_long",
                execution_time_ms=2**31 - 1,
                skill_id="c123",
                parameters_used={}
            ),
            # None result
            ExecutionResult(
                success=True,
                result=None,
                execution_time_ms=100,
                skill_id="c123",
                parameters_used={}
            ),
            # Empty result
            ExecutionResult(
                success=True,
                result="",
                execution_time_ms=100,
                skill_id="c123",
                parameters_used={}
            ),
        ]
        
        for result in edge_results:
            quality = assessor.assess_example_quality(result, [])
            # Should not crash
            assert 0.0 <= quality <= 1.0

    def test_parameter_signature_edge_cases(self, example_generator):
        """Test parameter signature creation with edge cases."""
        assessor = example_generator.quality_assessor
        
        edge_params = [
            # Empty parameters
            {},
            # Very long values
            {"param1": "x" * 1000, "param2": list(range(1000))},
            # Special characters
            {"param": "🔥 🚀 ✨ \n\t\r\x00"},
            # Complex nested structures
            {"param": {"nested": {"deep": {"data": list(range(100))}}}},
            # None values
            {"param1": None, "param2": "value"},
            # Mixed types
            {"str": "test", "int": 42, "bool": True, "none": None, "list": [1, "two"]},
        ]
        
        for params in edge_params:
            signature = assessor._create_parameter_signature(params)
            # Should not crash
            assert isinstance(signature, str)
            assert len(signature) > 0

    @pytest.mark.asyncio
    async def test_batch_generation_edge_cases(self, example_generator):
        """Test batch example generation with edge cases."""
        # Mixed successful and failed executions
        mixed_results = [
            ExecutionResult(success=True, result=f"success_{i}", execution_time_ms=100, skill_id="c123", parameters_used={"i": i})
            for i in range(5)
        ] + [
            ExecutionResult(success=False, error_message="failed", execution_time_ms=50, skill_id="c123", parameters_used={})
            for _ in range(3)
        ]
        
        example_generator.generate_example_from_execution = AsyncMock(side_effect=lambda r: 
            ExampleClaim(skill_id=r.skill_id, input_parameters=r.parameters_used, confidence=0.8) 
            if r.success else None
        )
        
        generated = await example_generator.batch_generate_examples(mixed_results)
        
        # Should generate examples only for successful ones
        assert len(generated) == 5
        assert all(isinstance(e, ExampleClaim) for e in generated)


class TestSystemWideEdgeCases:
    """System-wide edge case tests."""

    @pytest.mark.asyncio
    async def test_cascade_failures(self, skill_manager, data_manager_mock):
        """Test system behavior during cascade failures."""
        # Make database operations fail
        data_manager_mock.create_claim.side_effect = Exception("Database failure")
        data_manager_mock.update_claim.side_effect = Exception("Update failure")
        data_manager_mock.filter_claims.side_effect = Exception("Query failure")
        
        skill = SkillClaim(
            function_name="cascade_test_skill",
            content="Skill to test cascade failures",
            parameters=[],
            confidence=0.8
        )
        
        # Registration should fail gracefully
        result = await skill_manager.register_skill_claim(skill)
        assert result is False
        
        # Execution should handle database failures
        try:
            await skill_manager.initialize()  # Should not crash
        except Exception:
            pass  # Expected to fail

    @pytest.mark.asyncio
    async def test_resource_exhaustion_recovery(self, skill_manager):
        """Test system recovery from resource exhaustion."""
        # Simulate memory pressure
        large_objects = []
        try:
            # Create pressure
            for i in range(100):
                large_objects.append(['x'] * 10000)
            
            # Try to perform normal operations under pressure
            skills = [
                SkillClaim(
                    function_name=f"pressure_test_{i}",
                    content="Skill under memory pressure",
                    parameters=[],
                    confidence=0.8
                )
                for i in range(10)
            ]
            
            for skill in skills:
                await skill_manager.register_skill_claim(skill)
            
            # System should still function
            assert len(skill_manager.registry.skills) >= 10
            
        except MemoryError:
            # If we run out of memory, system should handle gracefully
            pytest.skip("Not enough memory for pressure test")
        finally:
            # Clean up
            del large_objects

    @pytest.mark.asyncio
    async def test_concurrent_stress(self, skill_manager, mock_async_functions):
        """Test system under concurrent stress."""
        # Register skills
        for i in range(20):
            skill = SkillClaim(
                function_name=f"stress_skill_{i}",
                content=f"Stress test skill {i}",
                parameters=[SkillParameter(name="param", param_type="str", required=True)],
                confidence=0.8
            )
            await skill_manager.register_skill_claim(skill)
            skill_manager.builtin_skills[f"stress_skill_{i}"] = mock_async_functions["search_claims"]
        
        # Create many concurrent operations
        async def mixed_operations():
            # Mix of different operations
            tasks = []
            
            # Skill executions
            for i in range(10):
                tasks.append(skill_manager.execute_skill(f"stress_skill_{i}", {"param": "test"}))
            
            # Skill discovery
            for i in range(5):
                tasks.append(skill_manager.find_relevant_skills(f"query_{i}"))
            
            return await asyncio.gather(*tasks, return_exceptions=True)
        
        # Run multiple batches concurrently
        batch_tasks = [mixed_operations() for _ in range(3)]
        all_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # System should not crash
        batch_count = 0
        for batch in all_results:
            if not isinstance(batch, Exception):
                batch_count += 1
                # All operations in batch should complete without exceptions
                for result in batch:
                    assert not isinstance(result, Exception)
        
        assert batch_count > 0  # At least some batches should succeed

    def test_unicode_and_encoding_edge_cases(self, response_parser):
        """Test Unicode handling and encoding edge cases."""
        unicode_responses = [
            # Various Unicode characters
            '<tool_calls><invoke name="test"><parameter name="unicode">🔥 🚀 ✨ ❤️ ✅ 中文 한국어 日本语 عربي</parameter></invoke></tool_calls>',
            # Mixed encodings
            '{"tool_calls": [{"name": "test", "parameters": {"mixed": "ASCII: hello, 中文: world"}}]}',
            # Special whitespace
            '```tool_call\nname: test\nparam: "unicode\\u2600\\u2601\\u2602"\n```',
            # Invalid Unicode sequences
            '{"tool_calls": [{"name": "test", "parameters": {"invalid": "\\uZZZZ"}}]}',
        ]
        
        for response in unicode_responses:
            result = response_parser.parse_response(response)
            # Should handle Unicode gracefully
            assert isinstance(result, ParsedResponse)
            if result.has_tool_calls():
                for tool_call in result.tool_calls:
                    assert isinstance(tool_call.name, str)
                    assert isinstance(tool_call.parameters, dict)

    @pytest.mark.asyncio
    async def test_long_running_operations(self, skill_manager, execution_limits):
        """Test handling of long-running operations."""
        # Create custom limits for testing
        long_limits = ExecutionLimits(max_execution_time=10.0)
        tool_executor = ToolExecutor(long_limits)
        
        # Mock function that takes time
        async def slow_function(**kwargs):
            await asyncio.sleep(0.1)  # 100ms
            return "slow_result"
        
        skill = SkillClaim(
            function_name="slow_skill",
            content="Skill that takes time",
            parameters=[],
            confidence=0.8
        )
        await skill_manager.register_skill_claim(skill)
        skill_manager.builtin_skills["slow_skill"] = slow_function
        
        # Execution should work but take time
        result = await skill_manager.execute_skill("slow_skill", {})
        
        assert result.success is True
        assert result.execution_time_ms >= 100  # Should take at least 100ms


if __name__ == "__main__":
    pytest.main([__file__])