"""
Comprehensive unit tests for skill_models.py components.
"""
import pytest
from datetime import datetime
from typing import Dict, Any, List
from pydantic import ValidationError

from src.core.skill_models import (
    SkillParameter, SkillClaim, ExampleClaim, ExecutionResult,
    ToolCall, ParsedResponse, SkillRegistry
)


class TestSkillParameter:
    """Test cases for SkillParameter model."""

    def test_valid_parameter_creation(self):
        """Test creating a valid SkillParameter."""
        param = SkillParameter(
            name="query",
            param_type="str",
            required=True,
            description="Search query string"
        )
        
        assert param.name == "query"
        assert param.param_type == "str"
        assert param.required is True
        assert param.description == "Search query string"
        assert param.default_value is None

    def test_parameter_with_default(self):
        """Test parameter with default value."""
        param = SkillParameter(
            name="limit",
            param_type="int",
            required=False,
            default_value=10,
            description="Maximum number of results"
        )
        
        assert param.required is False
        assert param.default_value == 10

    def test_invalid_parameter_type(self):
        """Test validation of invalid parameter types."""
        with pytest.raises(ValueError, match="param_type must be one of"):
            SkillParameter(
                name="invalid",
                param_type="invalid_type",
                required=True
            )

    def test_validate_value_string(self):
        """Test value validation for string type."""
        param = SkillParameter(name="text", param_type="str", required=True)
        
        assert param.validate_value("hello") is True
        assert param.validate_value(123) is False
        assert param.validate_value(None) is False

    def test_validate_value_integer(self):
        """Test value validation for integer type."""
        param = SkillParameter(name="count", param_type="int", required=True)
        
        assert param.validate_value(123) is True
        assert param.validate_value("123") is False
        assert param.validate_value(12.3) is True  # Float accepted as int
        assert param.validate_value(None) is False

    def test_validate_value_float(self):
        """Test value validation for float type."""
        param = SkillParameter(name="score", param_type="float", required=True)
        
        assert param.validate_value(12.3) is True
        assert param.validate_value(123) is True
        assert param.validate_value("12.3") is False
        assert param.validate_value(None) is False

    def test_validate_value_boolean(self):
        """Test value validation for boolean type."""
        param = SkillParameter(name="active", param_type="bool", required=True)
        
        assert param.validate_value(True) is True
        assert param.validate_value(False) is True
        assert param.validate_value("true") is False
        assert param.validate_value(1) is False
        assert param.validate_value(None) is False

    def test_validate_value_dict(self):
        """Test value validation for dictionary type."""
        param = SkillParameter(name="config", param_type="dict", required=True)
        
        assert param.validate_value({"key": "value"}) is True
        assert param.validate_value({}) is True
        assert param.validate_value("string") is False
        assert param.validate_value([]) is False
        assert param.validate_value(None) is False

    def test_validate_value_list(self):
        """Test value validation for list type."""
        param = SkillParameter(name="items", param_type="list", required=True)
        
        assert param.validate_value([1, 2, 3]) is True
        assert param.validate_value([]) is True
        assert param.validate_value("string") is False
        assert param.validate_value({}) is False
        assert param.validate_value(None) is False

    def test_validate_value_any(self):
        """Test value validation for any type."""
        param = SkillParameter(name="anything", param_type="any", required=True)
        
        assert param.validate_value("string") is True
        assert param.validate_value(123) is True
        assert param.validate_value([1, 2, 3]) is True
        assert param.validate_value({"key": "value"}) is True
        assert param.validate_value(None) is True


class TestSkillClaim:
    """Test cases for SkillClaim model."""

    def test_valid_skill_claim_creation(self, sample_skill_parameter):
        """Test creating a valid SkillClaim."""
        skill = SkillClaim(
            id="c123",
            content="Search for claims matching query",
            function_name="search_claims",
            parameters=[sample_skill_parameter],
            return_type="List[Dict]",
            skill_category="search",
            confidence=0.9
        )
        
        assert skill.id == "c123"
        assert skill.function_name == "search_claims"
        assert len(skill.parameters) == 1
        assert skill.return_type == "List[Dict]"
        assert skill.skill_category == "search"
        assert "type.skill" in skill.tags
        assert skill.execution_count == 0
        assert skill.success_count == 0

    def test_skill_tag_auto_addition(self):
        """Test that skill tag is automatically added."""
        skill = SkillClaim(
            function_name="test_skill",
            parameters=[],
            confidence=0.8
        )
        
        assert "type.skill" in skill.tags

    def test_invalid_function_name(self):
        """Test validation of invalid function names."""
        with pytest.raises(ValueError, match="function_name must contain only"):
            SkillClaim(
                function_name="invalid name with spaces!",
                parameters=[],
                confidence=0.8
            )

    def test_validate_parameters_success(self, sample_skill_claim):
        """Test successful parameter validation."""
        params = {"query": "test search"}
        is_valid, errors = sample_skill_claim.validate_parameters(params)
        
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_missing_required(self, sample_skill_claim):
        """Test validation with missing required parameter."""
        params = {}  # Missing required 'query'
        is_valid, errors = sample_skill_claim.validate_parameters(params)
        
        assert is_valid is False
        assert any("Missing required parameter: query" in error for error in errors)

    def test_validate_unknown_parameter(self, sample_skill_claim):
        """Test validation with unknown parameter."""
        params = {"query": "test", "unknown": "value"}
        is_valid, errors = sample_skill_claim.validate_parameters(params)
        
        assert is_valid is False
        assert any("Unknown parameter: unknown" in error for error in errors)

    def test_validate_wrong_type(self, sample_skill_claim):
        """Test validation with wrong parameter type."""
        params = {"query": 123}  # Should be string
        is_valid, errors = sample_skill_claim.validate_parameters(params)
        
        assert is_valid is False
        assert any("Parameter query must be of type str" in error for error in errors)

    def test_get_parameter_defaults(self):
        """Test getting default values for optional parameters."""
        skill = SkillClaim(
            function_name="test_skill",
            parameters=[
                SkillParameter(name="required", param_type="str", required=True),
                SkillParameter(name="optional1", param_type="int", required=False, default_value=10),
                SkillParameter(name="optional2", param_type="str", required=False, default_value="default")
            ],
            confidence=0.8
        )
        
        defaults = skill.get_parameter_defaults()
        expected = {"optional1": 10, "optional2": "default"}
        
        assert defaults == expected

    def test_get_success_rate(self):
        """Test success rate calculation."""
        skill = SkillClaim(
            function_name="test_skill",
            parameters=[],
            confidence=0.8
        )
        
        # No executions yet
        assert skill.get_success_rate() == 0.0
        
        # Add some execution history
        skill.execution_count = 10
        skill.success_count = 7
        
        assert skill.get_success_rate() == 0.7

    def test_update_execution_stats(self):
        """Test updating execution statistics."""
        skill = SkillClaim(
            function_name="test_skill",
            parameters=[],
            confidence=0.8
        )
        
        original_updated_at = skill.updated_at
        
        # Successful execution
        skill.update_execution_stats(True)
        assert skill.execution_count == 1
        assert skill.success_count == 1
        assert skill.updated_at > original_updated_at
        
        # Failed execution
        skill.update_execution_stats(False)
        assert skill.execution_count == 2
        assert skill.success_count == 1

    def test_example_management(self):
        """Test example list management."""
        skill = SkillClaim(
            function_name="test_skill",
            parameters=[],
            examples=["example 1", "example 2"],
            confidence=0.8
        )
        
        assert len(skill.examples) == 2
        assert skill.examples == ["example 1", "example 2"]


class TestExampleClaim:
    """Test cases for ExampleClaim model."""

    def test_valid_example_claim_creation(self):
        """Test creating a valid ExampleClaim."""
        example = ExampleClaim(
            id="c456",
            content="Example execution",
            skill_id="c123",
            input_parameters={"query": "test"},
            output_result={"id": "result", "data": "found"},
            execution_time_ms=150,
            confidence=0.85
        )
        
        assert example.id == "c456"
        assert example.skill_id == "c123"
        assert example.input_parameters == {"query": "test"}
        assert example.output_result == {"id": "result", "data": "found"}
        assert example.execution_time_ms == 150
        assert example.example_quality == 0.5  # Default value
        assert "type.example" in example.tags

    def test_invalid_skill_id(self):
        """Test validation of invalid skill ID."""
        with pytest.raises(ValueError, match="skill_id must be a valid claim ID"):
            ExampleClaim(
                skill_id="invalid_id",  # Should start with 'c'
                input_parameters={},
                confidence=0.8
            )

    def test_example_quality_bounds(self):
        """Test example quality bounds."""
        # Valid within bounds
        example = ExampleClaim(
            skill_id="c123",
            input_parameters={},
            example_quality=0.8,
            confidence=0.8
        )
        assert example.example_quality == 0.8
        
        # Below minimum
        example = ExampleClaim(
            skill_id="c123",
            input_parameters={},
            example_quality=0.0,
            confidence=0.8
        )
        assert example.example_quality == 0.0
        
        # Above maximum
        example = ExampleClaim(
            skill_id="c123",
            input_parameters={},
            example_quality=1.0,
            confidence=0.8
        )
        assert example.example_quality == 1.0

    def test_usage_count_tracking(self):
        """Test usage count tracking."""
        example = ExampleClaim(
            skill_id="c123",
            input_parameters={},
            confidence=0.8
        )
        
        assert example.usage_count == 0
        
        # Usage count would typically be updated by other components
        example.usage_count = 5
        assert example.usage_count == 5


class TestExecutionResult:
    """Test cases for ExecutionResult model."""

    def test_successful_execution_result(self, sample_tool_call):
        """Test successful execution result creation."""
        result = ExecutionResult(
            success=True,
            result={"output": "success"},
            execution_time_ms=100,
            skill_id="c123",
            parameters_used=sample_tool_call.parameters
        )
        
        assert result.success is True
        assert result.result == {"output": "success"}
        assert result.error_message is None
        assert result.execution_time_ms == 100
        assert result.skill_id == "c123"
        assert isinstance(result.timestamp, datetime)

    def test_failed_execution_result(self, sample_tool_call):
        """Test failed execution result creation."""
        result = ExecutionResult(
            success=False,
            error_message="Something went wrong",
            execution_time_ms=50,
            skill_id="c123",
            parameters_used=sample_tool_call.parameters
        )
        
        assert result.success is False
        assert result.result is None
        assert result.error_message == "Something went wrong"
        assert result.execution_time_ms == 50

    def test_execution_result_with_streams(self, sample_tool_call):
        """Test execution result with stdout/stderr."""
        result = ExecutionResult(
            success=True,
            result="output",
            stdout="Process output",
            stderr="Process warnings",
            execution_time_ms=75,
            skill_id="c123",
            parameters_used=sample_tool_call.parameters
        )
        
        assert result.stdout == "Process output"
        assert result.stderr == "Process warnings"

    def test_to_example_data_success(self, sample_tool_call):
        """Test converting successful result to example data."""
        result = ExecutionResult(
            success=True,
            result={"data": "result"},
            execution_time_ms=150,
            skill_id="c123",
            parameters_used=sample_tool_call.parameters
        )
        
        example_data = result.to_example_data()
        
        assert example_data["skill_id"] == "c123"
        assert example_data["input_parameters"] == sample_tool_call.parameters
        assert example_data["output_result"] == {"data": "result"}
        assert example_data["execution_time_ms"] == 150
        assert example_data["example_quality"] == 1.0
        assert example_data["confidence"] == 0.9
        assert "type.example" in example_data["tags"]

    def test_to_example_data_failure(self, sample_tool_call):
        """Test converting failed result to example data."""
        result = ExecutionResult(
            success=False,
            error_message="Failed",
            execution_time_ms=50,
            skill_id="c123",
            parameters_used=sample_tool_call.parameters
        )
        
        example_data = result.to_example_data()
        
        assert example_data["example_quality"] == 0.0
        assert example_data["confidence"] == 0.1
        assert example_data["output_result"] is None


class TestToolCall:
    """Test cases for ToolCall model."""

    def test_valid_tool_call(self):
        """Test creating a valid ToolCall."""
        call = ToolCall(
            name="search_claims",
            parameters={"query": "test", "limit": 10},
            call_id="call123"
        )
        
        assert call.name == "search_claims"
        assert call.parameters == {"query": "test", "limit": 10}
        assert call.call_id == "call123"

    def test_tool_call_without_call_id(self):
        """Test ToolCall without call ID."""
        call = ToolCall(
            name="search_claims",
            parameters={"query": "test"}
        )
        
        assert call.name == "search_claims"
        assert call.call_id is None

    def test_tool_call_empty_parameters(self):
        """Test ToolCall with empty parameters."""
        call = ToolCall(
            name="simple_tool",
            parameters={}
        )
        
        assert call.name == "simple_tool"
        assert call.parameters == {}

    def test_to_skill_execution_params(self):
        """Test converting to skill execution parameters."""
        call = ToolCall(
            name="search_claims",
            parameters={"query": "test"},
            call_id="call123"
        )
        
        exec_params = call.to_skill_execution_params()
        
        assert exec_params["skill_name"] == "search_claims"
        assert exec_params["parameters"] == {"query": "test"}
        assert exec_params["call_id"] == "call123"


class TestParsedResponse:
    """Test cases for ParsedResponse model."""

    def test_parsed_response_with_tool_calls(self):
        """Test ParsedResponse containing tool calls."""
        tool_calls = [
            ToolCall(name="tool1", parameters={"param": "value1"}),
            ToolCall(name="tool2", parameters={"param": "value2"})
        ]
        
        response = ParsedResponse(
            tool_calls=tool_calls,
            text_content="Some text",
            parsing_errors=[]
        )
        
        assert len(response.tool_calls) == 2
        assert response.text_content == "Some text"
        assert len(response.parsing_errors) == 0

    def test_has_tool_calls(self):
        """Test checking if response has tool calls."""
        # With tool calls
        response_with_calls = ParsedResponse(
            tool_calls=[ToolCall(name="tool", parameters={})]
        )
        assert response_with_calls.has_tool_calls() is True
        
        # Without tool calls
        response_without_calls = ParsedResponse(tool_calls=[])
        assert response_without_calls.has_tool_calls() is False

    def test_get_tool_call_names(self):
        """Test getting tool call names."""
        tool_calls = [
            ToolCall(name="search_claims", parameters={}),
            ToolCall(name="create_claim", parameters={}),
            ToolCall(name="search_claims", parameters={})  # Duplicate
        ]
        
        response = ParsedResponse(tool_calls=tool_calls)
        names = response.get_tool_call_names()
        
        assert names == ["search_claims", "create_claim", "search_claims"]

    def test_parsed_response_with_errors(self):
        """Test ParsedResponse with parsing errors."""
        response = ParsedResponse(
            tool_calls=[],
            text_content="Invalid response",
            parsing_errors=["No valid tool calls found", "Malformed XML"]
        )
        
        assert len(response.tool_calls) == 0
        assert response.text_content == "Invalid response"
        assert len(response.parsing_errors) == 2


class TestSkillRegistry:
    """Test cases for SkillRegistry model."""

    def test_empty_registry(self):
        """Test creating an empty registry."""
        registry = SkillRegistry()
        
        assert len(registry.skills) == 0
        assert len(registry.categories) == 0

    def test_register_skill(self, sample_skill_claim):
        """Test registering a skill."""
        registry = SkillRegistry()
        registry.register_skill(sample_skill_claim)
        
        assert len(registry.skills) == 1
        assert sample_skill_claim.function_name in registry.skills
        assert registry.skills[sample_skill_claim.function_name] == sample_skill_claim

    def test_register_skill_categorization(self):
        """Test skill categorization during registration."""
        registry = SkillRegistry()
        
        skill1 = SkillClaim(
            function_name="search_tool",
            skill_category="search",
            parameters=[],
            confidence=0.8
        )
        
        skill2 = SkillClaim(
            function_name="another_search",
            skill_category="search",
            parameters=[],
            confidence=0.8
        )
        
        skill3 = SkillClaim(
            function_name="create_tool",
            skill_category="creation",
            parameters=[],
            confidence=0.8
        )
        
        registry.register_skill(skill1)
        registry.register_skill(skill2)
        registry.register_skill(skill3)
        
        # Check categories
        assert "search" in registry.categories
        assert "creation" in registry.categories
        assert "uncategorized" not in registry.categories
        
        assert len(registry.categories["search"]) == 2
        assert len(registry.categories["creation"]) == 1

    def test_register_skill_uncategorized(self):
        """Test registering skill without category."""
        registry = SkillRegistry()
        
        skill = SkillClaim(
            function_name="uncategorized_tool",
            parameters=[],
            confidence=0.8
        )
        
        registry.register_skill(skill)
        
        assert "uncategorized" in registry.categories
        assert skill.function_name in registry.categories["uncategorized"]

    def test_get_skill(self, sample_skill_claim):
        """Test getting a skill by name."""
        registry = SkillRegistry()
        registry.register_skill(sample_skill_claim)
        
        # Existing skill
        retrieved = registry.get_skill(sample_skill_claim.function_name)
        assert retrieved == sample_skill_claim
        
        # Non-existing skill
        not_found = registry.get_skill("non_existent")
        assert not_found is None

    def test_find_skills_by_category(self):
        """Test finding skills by category."""
        registry = SkillRegistry()
        
        # Add skills to different categories
        for i in range(3):
            skill = SkillClaim(
                function_name=f"search_{i}",
                skill_category="search",
                parameters=[],
                confidence=0.8
            )
            registry.register_skill(skill)
        
        for i in range(2):
            skill = SkillClaim(
                function_name=f"create_{i}",
                skill_category="creation",
                parameters=[],
                confidence=0.8
            )
            registry.register_skill(skill)
        
        search_skills = registry.find_skills_by_category("search")
        create_skills = registry.find_skills_by_category("creation")
        none_skills = registry.find_skills_by_category("nonexistent")
        
        assert len(search_skills) == 3
        assert len(create_skills) == 2
        assert len(none_skills) == 0

    def test_search_skills(self):
        """Test searching skills by query."""
        registry = SkillRegistry()
        
        # Add test skills
        skill1 = SkillClaim(
            function_name="search_claims",
            content="Search for claims in database",
            skill_category="search",
            tags=["search", "database"],
            parameters=[],
            confidence=0.8
        )
        
        skill2 = SkillClaim(
            function_name="create_claim",
            content="Create a new claim",
            skill_category="creation",
            tags=["create", "claim"],
            parameters=[],
            confidence=0.8
        )
        
        skill3 = SkillClaim(
            function_name="find_data",
            content="Find specific data points",
            skill_category="search",
            tags=["search", "data"],
            parameters=[],
            confidence=0.8
        )
        
        registry.register_skill(skill1)
        registry.register_skill(skill2)
        registry.register_skill(skill3)
        
        # Search by name
        results = registry.search_skills("search")
        assert len(results) >= 2  # search_claims, find_data
        
        # Search by content
        results = registry.search_skills("claim")
        assert len(results) >= 2  # search_claims, create_claim
        
        # Search by tag
        results = registry.search_skills("database")
        assert len(results) == 1  # search_claims
        
        # Search with no results
        results = registry.search_skills("nonexistent")
        assert len(results) == 0

    def test_get_skill_stats(self):
        """Test skill statistics calculation."""
        registry = SkillRegistry()
        
        # Add skills with different execution counts
        skill1 = SkillClaim(
            function_name="popular_skill",
            execution_count=100,
            success_count=90,
            parameters=[],
            confidence=0.8
        )
        
        skill2 = SkillClaim(
            function_name="moderate_skill",
            execution_count=50,
            success_count=40,
            parameters=[],
            confidence=0.7
        )
        
        skill3 = SkillClaim(
            function_name="unused_skill",
            execution_count=0,
            success_count=0,
            parameters=[],
            confidence=0.6
        )
        
        registry.register_skill(skill1)
        registry.register_skill(skill2)
        registry.register_skill(skill3)
        
        stats = registry.get_skill_stats()
        
        assert stats["total_skills"] == 3
        assert stats["total_executions"] == 150
        assert stats["total_successes"] == 130
        assert stats["average_success_rate"] == 130/150
        assert stats["categories"] == 1  # All in uncategorized
        
        # Check most used skills
        most_used = stats["most_used_skills"]
        assert len(most_used) == 3
        assert most_used[0].function_name == "popular_skill"
        assert most_used[1].function_name == "moderate_skill"
        assert most_used[2].function_name == "unused_skill"


if __name__ == "__main__":
    pytest.main([__file__])