"""
Unit tests for SkillManager component.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.processing.skill_manager import SkillManager
from src.core.skill_models import SkillClaim, ExampleClaim, ExecutionResult, ToolCall
from src.data.models import ClaimNotFoundError, InvalidClaimError


class TestSkillManager:
    """Test cases for SkillManager class."""

    @pytest.mark.asyncio
    async def test_initialization(self, skill_manager, data_manager_mock):
        """Test SkillManager initialization."""
        # Mock filter_claims to return skill claims
        skill_claim_dict = {
            'id': 'c123',
            'function_name': 'test_skill',
            'content': 'Test skill',
            'parameters': [],
            'tags': ['type.skill'],
            'confidence': 0.8,
            'created_by': 'test',
            'execution_count': 5,
            'success_count': 4
        }
        
        data_manager_mock.filter_claims.return_value = [skill_claim_dict]
        
        await skill_manager.initialize()
        
        # Verify skill was loaded
        assert 'test_skill' in skill_manager.registry.skills
        assert skill_manager.registry.skills['test_skill'].execution_count == 5
        
        # Verify filter_claims was called
        data_manager_mock.filter_claims.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialization_error_handling(self, skill_manager, data_manager_mock):
        """Test error handling during initialization."""
        # Simulate database error
        data_manager_mock.filter_claims.side_effect = Exception("Database error")
        
        with pytest.raises(Exception, match="Failed to initialize skill manager"):
            await skill_manager.initialize()

    @pytest.mark.asyncio
    async def test_initialization_invalid_skill_claim(self, skill_manager, data_manager_mock):
        """Test handling of invalid skill claims during initialization."""
        # Mix valid and invalid claims
        valid_claim = {
            'id': 'c123',
            'function_name': 'valid_skill',
            'content': 'Valid skill',
            'parameters': [],
            'tags': ['type.skill'],
            'confidence': 0.8
        }
        
        invalid_claim = {
            'id': 'c456',
            'function_name': 'invalid skill!',  # Invalid name
            'parameters': [],
            'tags': ['type.skill']
        }
        
        data_manager_mock.filter_claims.return_value = [valid_claim, invalid_claim]
        
        await skill_manager.initialize()
        
        # Only valid skill should be loaded
        assert len(skill_manager.registry.skills) == 1
        assert 'valid_skill' in skill_manager.registry.skills
        assert 'invalid skill!' not in skill_manager.registry.skills

    def test_builtin_skills_registration(self, skill_manager):
        """Test registration of built-in skills."""
        # Check that built-in skills are registered
        expected_builtins = {
            'search_claims',
            'create_claim',
            'get_claim',
            'create_relationship',
            'get_relationships',
            'get_stats'
        }
        
        builtin_names = set(skill_manager.builtin_skills.keys())
        assert expected_builtins.issubset(builtin_names)

    @pytest.mark.asyncio
    async def test_register_skill_claim_success(self, skill_manager, data_manager_mock, sample_skill_claim):
        """Test successful skill claim registration."""
        data_manager_mock.create_claim.return_value = MagicMock(id="c123", created_at=datetime.utcnow())
        
        result = await skill_manager.register_skill_claim(sample_skill_claim)
        
        assert result is True
        assert sample_skill_claim.function_name in skill_manager.registry.skills
        
        # Verify database call
        data_manager_mock.create_claim.assert_called_once()
        
        # Verify local registration
        registered_skill = skill_manager.registry.get_skill(sample_skill_claim.function_name)
        assert registered_skill == sample_skill_claim

    @pytest.mark.asyncio
    async def test_register_skill_claim_no_function_name(self, skill_manager, data_manager_mock):
        """Test registration of skill claim without function name."""
        skill_claim = SkillClaim(
            content="Skill without function name",
            parameters=[],  # Empty parameters
            confidence=0.8
        )
        # function_name is not set, should be empty
        skill_claim.function_name = ""
        
        result = await skill_manager.register_skill_claim(skill_claim)
        
        assert result is False
        assert skill_claim.function_name not in skill_manager.registry.skills
        
        # Should not call database
        data_manager_mock.create_claim.assert_not_called()

    @pytest.mark.asyncio
    async def test_register_skill_claim_database_error(self, skill_manager, data_manager_mock, sample_skill_claim):
        """Test handling database error during registration."""
        data_manager_mock.create_claim.side_effect = Exception("Database error")
        
        result = await skill_manager.register_skill_claim(sample_skill_claim)
        
        assert result is False
        assert sample_skill_claim.function_name not in skill_manager.registry.skills

    @pytest.mark.asyncio
    async def test_register_skill_claim_update_existing(self, skill_manager, data_manager_mock, sample_skill_claim):
        """Test updating existing skill."""
        # Register the skill first
        await skill_manager.register_skill_claim(sample_skill_claim)
        
        # Create updated version
        updated_skill = SkillClaim(
            id=sample_skill_claim.id,
            **sample_skill_claim.dict(exclude={'id', 'created_at', 'updated_at'}),
            content="Updated content"
        )
        
        data_manager_mock.create_claim.return_value = MagicMock(id="c123", created_at=datetime.utcnow())
        
        result = await skill_manager.register_skill_claim(updated_skill)
        
        assert result is True
        # Should still be registered (replaced)
        assert updated_skill.function_name in skill_manager.registry.skills

    @pytest.mark.asyncio
    async def test_execute_skill_success_builtin(self, skill_manager, sample_tool_call, mock_async_functions):
        """Test successful execution of built-in skill."""
        # Register the skill manually
        skill = SkillClaim(
            id="c123",
            function_name="search_claims",
            parameters=[
                MagicMock(name="query", param_type="str", required=True, validate_value=lambda x: True)
            ],
            confidence=0.8
        )
        skill_manager.registry.register_skill(skill)
        
        # Mock the built-in function
        skill_manager.builtin_skills["search_claims"] = mock_async_functions["search_claims"]
        
        result = await skill_manager.execute_skill(
            skill.function_name,
            sample_tool_call.parameters
        )
        
        assert result.success is True
        assert result.result is not None
        assert result.skill_id == "c123"
        assert result.parameters_used == sample_tool_call.parameters
        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_execute_skill_not_found(self, skill_manager, sample_tool_call):
        """Test execution of non-existent skill."""
        result = await skill_manager.execute_skill(
            "non_existent_skill",
            sample_tool_call.parameters
        )
        
        assert result.success is False
        assert "not found" in result.error_message
        assert result.skill_id == "unknown"

    @pytest.mark.asyncio
    async def test_execute_skill_parameter_validation_failed(self, skill_manager, sample_tool_call):
        """Test execution with parameter validation failure."""
        # Create skill with parameter validation
        skill = SkillClaim(
            id="c123",
            function_name="test_skill",
            parameters=[
                MagicMock(name="query", param_type="str", required=True, validate_value=lambda x: isinstance(x, str))
            ],
            confidence=0.8
        )
        skill_manager.registry.register_skill(skill)
        
        # Provide invalid parameter (int instead of str)
        result = await skill_manager.execute_skill(
            skill.function_name,
            {"query": 123}  # Should be string
        )
        
        assert result.success is False
        assert "Parameter validation failed" in result.error_message

    @pytest.mark.asyncio
    async def test_execute_skill_missing_required_parameter(self, skill_manager):
        """Test execution with missing required parameter."""
        skill = SkillClaim(
            id="c123",
            function_name="test_skill",
            parameters=[
                MagicMock(name="query", param_type="str", required=True, validate_value=lambda x: True)
            ],
            confidence=0.8
        )
        skill_manager.registry.register_skill(skill)
        
        # Don't provide required parameter
        result = await skill_manager.execute_skill(
            skill.function_name,
            {}
        )
        
        assert result.success is False
        assert "Missing required parameter" in result.error_message

    @pytest.mark.asyncio
    async def test_execute_skill_custom_skill_not_implemented(self, skill_manager, sample_tool_call):
        """Test execution of custom skill (not yet implemented)."""
        # Register a non-built-in skill
        skill = SkillClaim(
            id="c123",
            function_name="custom_skill",
            parameters=[],
            confidence=0.8
        )
        skill_manager.registry.register_skill(skill)
        
        result = await skill_manager.execute_skill(
            skill.function_name,
            sample_tool_call.parameters
        )
        
        assert result.success is False
        assert "Custom skill execution not yet implemented" in result.error_message

    @pytest.mark.asyncio
    async def test_execute_skill_function_exception(self, skill_manager, sample_tool_call):
        """Test execution when function raises exception."""
        skill = SkillClaim(
            id="c123",
            function_name="failing_skill",
            parameters=[],
            confidence=0.8
        )
        skill_manager.registry.register_skill(skill)
        
        # Mock function that raises exception
        async def failing_function(**kwargs):
            raise ValueError("Function error")
        
        skill_manager.builtin_skills["failing_skill"] = failing_function
        
        result = await skill_manager.execute_skill(
            skill.function_name,
            sample_tool_call.parameters
        )
        
        assert result.success is False
        assert "Function execution error" in result.error_message

    @pytest.mark.asyncio
    async def test_execute_skill_stats_update(self, skill_manager, data_manager_mock, sample_tool_call, mock_async_functions):
        """Test that execution statistics are updated."""
        skill = SkillClaim(
            id="c123",
            function_name="search_claims",
            parameters=[
                MagicMock(name="query", param_type="str", required=True, validate_value=lambda x: True)
            ],
            execution_count=5,
            success_count=4,
            confidence=0.8
        )
        skill_manager.registry.register_skill(skill)
        
        skill_manager.builtin_skills["search_claims"] = mock_async_functions["search_claims"]
        
        result = await skill_manager.execute_skill(
            skill.function_name,
            sample_tool_call.parameters
        )
        
        # Check stats were updated
        assert skill.execution_count == 6
        assert skill.success_count == 5
        
        # Check database update was called
        data_manager_mock.update_claim.assert_called_once_with(
            skill.id,
            execution_count=6,
            success_count=5
        )

    @pytest.mark.asyncio
    async def test_find_relevant_skills_found(self, skill_manager, data_manager_mock, sample_skill_claim):
        """Test finding relevant skills when skills exist."""
        # Setup data manager mock for search
        similar_claims = [sample_skill_claim]
        data_manager_mock.search_similar.return_value = similar_claims
        
        results = await skill_manager.find_relevant_skills("search query")
        
        assert len(results) >= 1
        # Should include both registry and database results

    @pytest.mark.asyncio
    async def test_find_relevant_skills_not_found(self, skill_manager, data_manager_mock):
        """Test finding relevant skills when none exist."""
        data_manager_mock.search_similar.return_value = []
        
        results = await skill_manager.find_relevant_skills("nonexistent query")
        
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_find_relevant_skills_error(self, skill_manager, data_manager_mock):
        """Test error handling in find_relevant_skills."""
        data_manager_mock.search_similar.side_effect = Exception("Search error")
        
        results = await skill_manager.find_relevant_skills("test query")
        
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_get_skill_examples_found(self, skill_manager, data_manager_mock, sample_example_claim):
        """Test getting examples for a skill when examples exist."""
        # Mock example claims
        example_dict = {
            'id': sample_example_claim.id,
            'skill_id': sample_example_claim.skill_id,
            'input_parameters': sample_example_claim.input_parameters,
            'output_result': sample_example_claim.output_result,
            'example_quality': sample_example_claim.example_quality,
            'tags': sample_example_claim.tags,
            'content': sample_example_claim.content,
            'confidence': sample_example_claim.confidence,
            'created_by': sample_example_claim.created_by
        }
        
        data_manager_mock.filter_claims.return_value = [example_dict]
        
        examples = await skill_manager.get_skill_examples("c123")
        
        assert len(examples) == 1
        assert examples[0].skill_id == "c123"

    @pytest.mark.asyncio
    async def test_get_skill_examples_not_found(self, skill_manager, data_manager_mock):
        """Test getting examples when none exist."""
        data_manager_mock.filter_claims.return_value = []
        
        examples = await skill_manager.get_skill_examples("c999")
        
        assert len(examples) == 0

    @pytest.mark.asyncio
    async def test_get_skill_examples_error(self, skill_manager, data_manager_mock):
        """Test error handling in get_skill_examples."""
        data_manager_mock.filter_claims.side_effect = Exception("Database error")
        
        examples = await skill_manager.get_skill_examples("c123")
        
        assert len(examples) == 0

    def test_get_execution_stats_empty(self, skill_manager):
        """Test execution statistics when no executions have occurred."""
        stats = skill_manager.get_execution_stats()
        
        assert stats['total_executions'] == 0
        assert stats['success_rate'] == 0.0
        assert stats['average_execution_time_ms'] == 0.0
        assert len(stats['most_used_skills']) == 0

    def test_get_execution_stats_with_data(self, skill_manager, sample_execution_result):
        """Test execution statistics with execution history."""
        # Add some execution results
        successful_result = sample_execution_result
        failed_result = ExecutionResult(
            success=False,
            error_message="Failed",
            execution_time_ms=50,
            skill_id="c456",
            parameters_used={"param": "value"}
        )
        
        skill_manager._add_to_history(successful_result)
        skill_manager._add_to_history(failed_result)
        skill_manager._add_to_history(successful_result)  # Another success
        
        stats = skill_manager.get_execution_stats()
        
        assert stats['total_executions'] == 3
        assert stats['successful_executions'] == 2
        assert stats['success_rate'] == 2/3
        assert stats['average_execution_time_ms'] == (150 + 50 + 150) / 3
        
        # Check most used skills
        most_used = stats['most_used_skills']
        assert len(most_used) == 2
        assert most_used[0] == ("c123", 2)  # Most used
        assert most_used[1] == ("c456", 1)

    def test_add_to_history_size_limit(self, skill_manager, sample_execution_result):
        """Test that execution history respects size limit."""
        # Set a small limit for testing
        skill_manager.max_history_size = 3
        
        # Add more results than the limit
        for i in range(5):
            result = ExecutionResult(
                success=True,
                result=f"result_{i}",
                execution_time_ms=100,
                skill_id="c123",
                parameters_used={"i": i}
            )
            skill_manager._add_to_history(result)
        
        # Should only keep the last 3
        assert len(skill_manager.execution_history) == 3
        assert skill_manager.execution_history[-1].result == "result_4"

    @pytest.mark.asyncio
    async def test_generate_example_from_execution(self, skill_manager, data_manager_mock, sample_execution_result):
        """Test generating example from execution."""
        success_result = sample_execution_result
        
        await skill_manager._generate_example_from_execution(success_result)
        
        # Should create example claim
        data_manager_mock.create_claim.assert_called_once()
        call_args = data_manager_mock.create_claim.call_args[1]
        
        assert 'type.example' in call_args['tags']
        assert 'auto_generated' in call_args['tags']
        assert call_args['created_by'] == 'system'
        assert call_args['confidence'] == 0.9  # High confidence for successful execution

    @pytest.mark.asyncio
    async def test_generate_example_from_execution_failed(self, skill_manager, data_manager_mock):
        """Test that example is not generated from failed execution."""
        failed_result = ExecutionResult(
            success=False,
            error_message="Failed",
            execution_time_ms=50,
            skill_id="c123",
            parameters_used={}
        )
        
        await skill_manager._generate_example_from_execution(failed_result)
        
        # Should not create example
        data_manager_mock.create_claim.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_example_from_execution_error(self, skill_manager, data_manager_mock, sample_execution_result):
        """Test error handling in example generation."""
        data_manager_mock.create_claim.side_effect = Exception("Database error")
        
        # Should not raise exception
        await skill_manager._generate_example_from_execution(sample_execution_result)
        
        # Should still attempt to create
        data_manager_mock.create_claim.assert_called_once()

    def test_builtin_skill_functions_signatures(self, skill_manager):
        """Test that built-in skill functions have expected signatures."""
        # Check search_claims function
        search_func = skill_manager.builtin_skills.get('search_claims')
        assert search_func is not None
        
        # Check create_claim function
        create_func = skill_manager.builtin_skills.get('create_claim')
        assert create_func is not None
        
        # Verify they're async functions
        import inspect
        assert inspect.iscoroutinefunction(search_func)
        assert inspect.iscoroutinefunction(create_func)


class TestSkillManagerBuiltinFunctions:
    """Test cases specifically for built-in skill functions."""

    @pytest.mark.asyncio
    async def test_builtin_search_claims(self, skill_manager, data_manager_mock):
        """Test built-in search_claims function."""
        # Setup mock returns
        similar_claims = [
            MagicMock(id="c1", content="Test claim 1", tags=["test"], confidence=0.8),
            MagicMock(id="c2", content="Test claim 2", tags=["test"], confidence=0.7)
        ]
        data_manager_mock.search_similar.return_value = similar_claims
        
        search_func = skill_manager.builtin_skills['search_claims']
        
        result = await search_func(query="test", limit=5, tags=["test"])
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]['id'] == "c1"
        assert result[1]['id'] == "c2"
        
        data_manager_mock.search_similar.assert_called_once_with("test", 5)

    @pytest.mark.asyncio
    async def test_builtin_create_claim(self, skill_manager, data_manager_mock):
        """Test built-in create_claim function."""
        mock_claim = MagicMock(
            id="c123",
            content="New claim",
            confidence=0.8,
            tags=["test"],
            created_at=datetime.utcnow()
        )
        data_manager_mock.create_claim.return_value = mock_claim
        
        create_func = skill_manager.builtin_skills['create_claim']
        
        result = await create_func(
            content="New claim",
            confidence=0.8,
            tags=["test"],
            created_by="skill_execution"
        )
        
        assert result['id'] == "c123"
        assert result['content'] == "New claim"
        assert result['confidence'] == 0.8
        assert "test" in result['tags']
        
        data_manager_mock.create_claim.assert_called_once()

    @pytest.mark.asyncio
    async def test_builtin_get_claim_found(self, skill_manager, data_manager_mock):
        """Test built-in get_claim function when claim exists."""
        mock_claim = MagicMock(
            id="c123",
            content="Test claim",
            confidence=0.8,
            tags=["test"],
            created_at=datetime.utcnow(),
            created_by="user"
        )
        data_manager_mock.get_claim.return_value = mock_claim
        
        get_func = skill_manager.builtin_skills['get_claim']
        
        result = await get_func("c123")
        
        assert result['id'] == "c123"
        assert result['content'] == "Test claim"
        assert 'error' not in result
        
        data_manager_mock.get_claim.assert_called_once_with("c123")

    @pytest.mark.asyncio
    async def test_builtin_get_claim_not_found(self, skill_manager, data_manager_mock):
        """Test built-in get_claim function when claim doesn't exist."""
        data_manager_mock.get_claim.return_value = None
        
        get_func = skill_manager.builtin_skills['get_claim']
        
        result = await get_func("c999")
        
        assert 'error' in result
        assert "not found" in result['error'].lower()

    @pytest.mark.asyncio
    async def test_builtin_create_relationship(self, skill_manager, data_manager_mock):
        """Test built-in create_relationship function."""
        data_manager_mock.add_relationship.return_value = "rel123"
        
        create_rel_func = skill_manager.builtin_skills['create_relationship']
        
        result = await create_rel_func(
            supporter_id="c1",
            supported_id="c2",
            relationship_type="supports",
            created_by="skill_execution"
        )
        
        assert result['relationship_id'] == "rel123"
        assert result['supporter_id'] == "c1"
        assert result['supported_id'] == "c2"
        assert result['relationship_type'] == "supports"
        
        data_manager_mock.add_relationship.assert_called_once()

    @pytest.mark.asyncio
    async def test_builtin_get_relationships(self, skill_manager, data_manager_mock):
        """Test built-in get_relationships function."""
        mock_rels = [
            MagicMock(id="rel1", supporter_id="c1", supported_id="c2", 
                     relationship_type="supports", created_at=datetime.utcnow()),
            MagicMock(id="rel2", supporter_id="c3", supported_id="c2", 
                     relationship_type="opposes", created_at=datetime.utcnow())
        ]
        data_manager_mock.get_relationships.return_value = mock_rels
        
        get_rels_func = skill_manager.builtin_skills['get_relationships']
        
        result = await get_rels_func("c2")
        
        assert result['claim_id'] == "c2"
        assert len(result['relationships']) == 2
        assert result['relationships'][0]['relationship_type'] == "supports"
        assert result['relationships'][1]['relationship_type'] == "opposes"

    @pytest.mark.asyncio
    async def test_builtin_get_stats(self, skill_manager, data_manager_mock):
        """Test built-in get_stats function."""
        data_manager_mock.get_stats.return_value = {
            'total_claims': 100,
            'total_relationships': 50
        }
        
        get_stats_func = skill_manager.builtin_skills['get_stats']
        
        result = await get_stats()
        
        assert 'data_layer' in result
        assert 'skills' in result
        assert 'execution_history_size' in result
        assert result['data_layer']['total_claims'] == 100


if __name__ == "__main__":
    pytest.main([__file__])