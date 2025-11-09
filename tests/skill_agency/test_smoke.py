"""
Smoke tests for the skill-based agency system.
Quick verification that basic functionality works.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from src.processing.skill_manager import SkillManager
from src.processing.response_parser import ResponseParser
from src.processing.tool_executor import ToolExecutor
from src.processing.example_generator import ExampleGenerator
from src.core.skill_models import SkillClaim, SkillParameter


@pytest.mark.smoke
@pytest.mark.unit
class TestBasicSmoke:
    """Basic smoke tests to verify core functionality."""

    def test_skill_models_basic_creation(self):
        """Test basic skill model creation."""
        # Create skill parameter
        param = SkillParameter(
            name="test_param",
            param_type="str",
            required=True,
            description="Test parameter"
        )
        assert param.name == "test_param"
        assert param.param_type == "str"
        assert param.required is True
        
        # Create skill claim
        skill = SkillClaim(
            function_name="test_skill",
            content="Test skill for smoke testing",
            parameters=[param],
            confidence=0.8
        )
        assert skill.function_name == "test_skill"
        assert len(skill.parameters) == 1
        assert "type.skill" in skill.tags

    def test_response_parser_basic(self, response_parser):
        """Test basic response parsing."""
        # Simple XML response
        xml_response = """
        <tool_calls>
            <invoke name="test_tool">
                <parameter name="param">value</parameter>
            </invoke>
        </tool_calls>
        """
        
        result = response_parser.parse_response(xml_response)
        assert result.has_tool_calls()
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "test_tool"
        
        # Simple JSON response
        json_response = """
        {
            "tool_calls": [
                {"name": "test_tool", "parameters": {"param": "value"}}
            ]
        }
        """
        
        result = response_parser.parse_response(json_response)
        assert result.has_tool_calls()
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "test_tool"

    def test_tool_executor_basic(self, execution_limits):
        """Test basic tool executor setup."""
        executor = ToolExecutor(execution_limits)
        assert executor.limits == execution_limits
        assert executor.safe_executor is not None
        assert executor.security_validator is not None

    def test_example_generator_basic(self, data_manager_mock):
        """Test basic example generator setup."""
        generator = ExampleGenerator(data_manager_mock)
        assert generator.data_manager == data_manager_mock
        assert generator.quality_assessor is not None
        assert generator.min_quality_threshold == 0.3


@pytest.mark.smoke
@pytest.mark.integration
class TestIntegrationSmoke:
    """Integration smoke tests for component interaction."""

    @pytest.mark.asyncio
    async def test_skill_registration_and_discovery(self, skill_manager):
        """Test skill registration and discovery workflow."""
        # Create and register skill
        skill = SkillClaim(
            function_name="smoke_test_skill",
            content="Smoke test skill",
            parameters=[
                SkillParameter(name="query", param_type="str", required=True)
            ],
            confidence=0.8
        )
        
        # Mock successful registration
        skill_manager.data_manager.create_claim = AsyncMock(return_value=MagicMock(id="c123"))
        result = await skill_manager.register_skill_claim(skill)
        assert result is True
        
        # Verify skill is registered
        retrieved = skill_manager.registry.get_skill("smoke_test_skill")
        assert retrieved is not None
        assert retrieved.function_name == "smoke_test_skill"

    @pytest.mark.asyncio
    async def test_skill_execution_workflow(self, skill_manager):
        """Test basic skill execution workflow."""
        # Mock skill function
        async def mock_skill_func(**kwargs):
            return {"result": f"Executed with {kwargs}"}
        
        # Register skill
        skill = SkillClaim(
            id="c123",
            function_name="smoke_exec_skill",
            content="Execute smoke test",
            parameters=[SkillParameter(name="param", param_type="str", required=True)],
            confidence=0.8
        )
        await skill_manager.register_skill_claim(skill)
        skill_manager.builtin_skills["smoke_exec_skill"] = mock_skill_func
        
        # Execute skill
        result = await skill_manager.execute_skill("smoke_exec_skill", {"param": "test"})
        
        # Verify execution
        assert result.success is True
        assert result.result is not None
        assert "test" in str(result.result)

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, skill_manager, response_parser, mock_async_functions):
        """Test complete end-to-end workflow."""
        # Register skill
        skill = SkillClaim(
            id="c123",
            function_name="search_claims",
            content="Search for claims",
            parameters=[SkillParameter(name="query", param_type="str", required=True)],
            confidence=0.8
        )
        await skill_manager.register_skill_claim(skill)
        skill_manager.builtin_skills["search_claims"] = mock_async_functions["search_claims"]
        
        # Parse LLM response
        llm_response = """
        <tool_calls>
            <invoke name="search_claims">
                <parameter name="query">smoke test</parameter>
            </invoke>
        </tool_calls>
        """
        
        parsed = response_parser.parse_response(llm_response)
        assert parsed.has_tool_calls()
        
        # Execute parsed tool calls
        tool_call = parsed.tool_calls[0]
        result = await skill_manager.execute_skill(tool_call.name, tool_call.parameters)
        
        # Verify complete workflow
        assert result.success is True
        assert result.result is not None


@pytest.mark.smoke
@pytest.mark.security
class TestSecuritySmoke:
    """Security smoke tests for basic validation."""

    def test_dangerous_code_detection(self, execution_limits):
        """Test that dangerous code is detected."""
        from src.processing.tool_executor import SecurityValidator
        
        validator = SecurityValidator(execution_limits)
        
        # Test dangerous function detection
        dangerous_codes = [
            "eval('1+1')",
            "exec('print(1)')",
            "__import__('os')",
        ]
        
        for code in dangerous_codes:
            is_safe, errors = validator.validate_code(code)
            assert not is_safe, f"Dangerous code not detected: {code}"
            assert len(errors) > 0

    def test_safe_code_allowed(self, execution_limits):
        """Test that safe code is allowed."""
        from src.processing.tool_executor import SecurityValidator
        
        validator = SecurityValidator(execution_limits)
        
        # Test safe code
        safe_codes = [
            "import math",
            "import random",
            "result = 1 + 1",
            "def test(): return 'safe'",
        ]
        
        for code in safe_codes:
            is_safe, errors = validator.validate_code(code)
            assert is_safe, f"Safe code incorrectly flagged: {code}"
            assert len(errors) == 0


@pytest.mark.smoke
@pytest.mark.asyncio
class TestExampleGenerationSmoke:
    """Example generation smoke tests."""

    async def test_basic_example_generation(self, example_generator):
        """Test basic example generation."""
        from src.core.skill_models import ExecutionResult
        
        # Create sample execution result
        result = ExecutionResult(
            success=True,
            result="test result",
            execution_time_ms=100,
            skill_id="c123",
            parameters_used={"query": "test"}
        )
        
        # Mock dependencies
        example_generator.get_examples_for_skill = AsyncMock(return_value=[])
        example_generator._create_example_claim = AsyncMock()
        
        # Generate example
        await example_generator.generate_example_from_execution(result)
        
        # Verify process completed without errors
        example_generator.get_examples_for_skill.assert_called_once_with("c123")
        example_generator._create_example_claim.assert_called_once()


@pytest.mark.smoke
def test_configuration_and_setup():
    """Test that test configuration is properly set up."""
    # Test that fixtures are available
    from tests.skill_agency.conftest import (
        data_manager_mock, skill_manager, response_parser,
        tool_executor, example_generator, execution_limits
    )
    
    # Basic assertions that fixtures return expected types
    # (Note: Can't actually call fixtures here, just验证 they exist)
    assert True  # If imports work, setup is good


if __name__ == "__main__":
    pytest.main([__file__])