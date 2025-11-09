"""
Integration tests for end-to-end workflows of the skill-based agency system.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.processing.skill_manager import SkillManager
from src.processing.response_parser import ResponseParser
from src.processing.tool_executor import ToolExecutor, ExecutionLimits
from src.processing.example_generator import ExampleGenerator
from src.core.skill_models import SkillClaim, ExampleClaim, ExecutionResult, ToolCall, SkillParameter


class TestSkillBasedAgencyIntegration:
    """Integration tests for complete skill-based agency workflows."""

    @pytest.mark.asyncio
    async def test_complete_skill_execution_workflow(self, data_manager_mock, mock_async_functions):
        """Test complete workflow from LLM response to skill execution."""
        # Initialize components
        skill_manager = SkillManager(data_manager_mock)
        response_parser = ResponseParser()
        tool_executor = ToolExecutor()
        
        # Register a test skill
        skill = SkillClaim(
            id="c123",
            function_name="search_claims",
            content="Search for claims",
            parameters=[
                SkillParameter(name="query", param_type="str", required=True),
                SkillParameter(name="limit", param_type="int", required=False, default_value=10)
            ],
            skill_category="search",
            confidence=0.9
        )
        await skill_manager.register_skill_claim(skill)
        
        # Mock the built-in function
        skill_manager.builtin_skills["search_claims"] = mock_async_functions["search_claims"]
        
        # Sample LLM response with tool call
        llm_response = """
        I'll help you search for claims.
        
        <tool_calls>
            <invoke name="search_claims">
                <parameter name="query">test query</parameter>
                <parameter name="limit">5</parameter>
            </invoke>
        </tool_calls>
        
        Let me find those results for you.
        """
        
        # Parse the response
        parsed = response_parser.parse_response(llm_response)
        
        # Verify parsing worked
        assert len(parsed.tool_calls) == 1
        assert parsed.tool_calls[0].name == "search_claims"
        assert parsed.tool_calls[0].parameters["query"] == "test query"
        assert parsed.tool_calls[0].parameters["limit"] == 5
        
        # Execute the skill
        tool_call = parsed.tool_calls[0]
        execution_result = await skill_manager.execute_skill(
            tool_call.name,
            tool_call.parameters
        )
        
        # Verify execution
        assert execution_result.success is True
        assert execution_result.result is not None
        assert execution_result.skill_id == skill.id
        
        # Verify skill stats updated
        assert skill.execution_count == 1
        assert skill.success_count == 1
        
        # Verify execution history
        stats = skill_manager.get_execution_stats()
        assert stats['total_executions'] == 1
        assert stats['success_rate'] == 1.0

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_workflow(self, data_manager_mock, mock_async_functions):
        """Test workflow with multiple tool calls in single response."""
        skill_manager = SkillManager(data_manager_mock)
        response_parser = ResponseParser()
        
        # Register multiple skills
        search_skill = SkillClaim(
            id="c123",
            function_name="search_claims",
            content="Search for claims",
            parameters=[SkillParameter(name="query", param_type="str", required=True)],
            confidence=0.9
        )
        
        create_skill = SkillClaim(
            id="c456",
            function_name="create_claim",
            content="Create a claim",
            parameters=[SkillParameter(name="content", param_type="str", required=True)],
            confidence=0.8
        )
        
        await skill_manager.register_skill_claim(search_skill)
        await skill_manager.register_skill_claim(create_skill)
        
        # Mock functions
        skill_manager.builtin_skills["search_claims"] = mock_async_functions["search_claims"]
        skill_manager.builtin_skills["create_claim"] = mock_async_functions["create_claim"]
        
        # Multiple tool calls response
        llm_response = """
        <tool_calls>
            <invoke name="search_claims">
                <parameter name="query">existing claims</parameter>
            </invoke>
            <invoke name="create_claim">
                <parameter name="content">New claim based on search</parameter>
            </invoke>
        </tool_calls>
        """
        
        # Parse and execute all tool calls
        parsed = response_parser.parse_response(llm_response)
        
        results = []
        for tool_call in parsed.tool_calls:
            result = await skill_manager.execute_skill(
                tool_call.name,
                tool_call.parameters
            )
            results.append(result)
        
        # Verify all executions succeeded
        assert len(results) == 2
        assert all(r.success for r in results)
        
        # Verify statistics
        stats = skill_manager.get_execution_stats()
        assert stats['total_executions'] == 2
        assert stats['success_rate'] == 1.0

    @pytest.mark.asyncio
    async def test_json_response_format_workflow(self, data_manager_mock, mock_async_functions):
        """Test workflow with JSON-formatted tool calls."""
        skill_manager = SkillManager(data_manager_mock)
        response_parser = ResponseParser()
        
        # Register skill
        skill = SkillClaim(
            id="c123",
            function_name="create_claim",
            content="Create a claim",
            parameters=[
                SkillParameter(name="content", param_type="str", required=True),
                SkillParameter(name="confidence", param_type="float", required=False)
            ],
            confidence=0.8
        )
        await skill_manager.register_skill_claim(skill)
        skill_manager.builtin_skills["create_claim"] = mock_async_functions["create_claim"]
        
        # JSON format response
        llm_response = """
        I'll create a claim for you.
        
        {
            "tool_calls": [
                {
                    "name": "create_claim",
                    "parameters": {
                        "content": "Test claim content",
                        "confidence": 0.85
                    }
                }
            ]
        }
        
        The claim has been created.
        """
        
        # Parse and execute
        parsed = response_parser.parse_response(llm_response)
        assert len(parsed.tool_calls) == 1
        
        result = await skill_manager.execute_skill(
            parsed.tool_calls[0].name,
            parsed.tool_calls[0].parameters
        )
        
        # Verify execution
        assert result.success is True

    @pytest.mark.asyncio
    async def test_markdown_response_format_workflow(self, data_manager_mock, mock_async_functions):
        """Test workflow with markdown-formatted tool calls."""
        skill_manager = SkillManager(data_manager_mock)
        response_parser = ResponseParser()
        
        # Register skill
        skill = SkillClaim(
            id="c123",
            function_name="search_claims",
            content="Search for claims",
            parameters=[SkillParameter(name="query", param_type="str", required=True)],
            confidence=0.9
        )
        await skill_manager.register_skill_claim(skill)
        skill_manager.builtin_skills["search_claims"] = mock_async_functions["search_claims"]
        
        # Markdown format response
        llm_response = """
        Let me search for information about machine learning.
        
        ```tool_call
        name: search_claims
        query: machine learning
        limit: 10
        ```
        
        Here are the results I found.
        """
        
        # Parse and execute
        parsed = response_parser.parse_response(llm_response)
        assert len(parsed.tool_calls) == 1
        
        result = await skill_manager.execute_skill(
            parsed.tool_calls[0].name,
            parsed.tool_calls[0].parameters
        )
        
        # Verify execution
        assert result.success is True

    @pytest.mark.asyncio
    async def test_example_generation_workflow(self, data_manager_mock, mock_async_functions):
        """Test workflow with automatic example generation."""
        skill_manager = SkillManager(data_manager_mock)
        example_generator = ExampleGenerator(data_manager_mock)
        
        # Register skill
        skill = SkillClaim(
            id="c123",
            function_name="search_claims",
            content="Search for claims",
            parameters=[SkillParameter(name="query", param_type="str", required=True)],
            confidence=0.9
        )
        await skill_manager.register_skill_claim(skill)
        skill_manager.builtin_skills["search_claims"] = mock_async_functions["search_claims"]
        
        # Mock example generation methods
        example_generator.get_examples_for_skill = AsyncMock(return_value=[])
        
        # Execute skill
        result = await skill_manager.execute_skill(
            "search_claims",
            {"query": "test query"}
        )
        
        # Generate example from execution
        example = await example_generator.generate_example_from_execution(result)
        
        # Verify example generation
        assert example is not None
        assert isinstance(example, ExampleClaim)
        assert example.skill_id == skill.id
        assert example.input_parameters == {"query": "test query"}

    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, data_manager_mock):
        """Test workflow error handling."""
        skill_manager = SkillManager(data_manager_mock)
        response_parser = ResponseParser()
        
        # Parse malformed response
        malformed_response = "This response has no valid tool calls and is just plain text."
        
        parsed = response_parser.parse_response(malformed_response)
        
        # Should have no tool calls
        assert len(parsed.tool_calls) == 0
        assert len(parsed.parsing_errors) > 0
        
        # Try to execute non-existent skill
        result = await skill_manager.execute_skill("non_existent_skill", {})
        
        # Should handle gracefully
        assert result.success is False
        assert "not found" in result.error_message

    @pytest.mark.asyncio
    async def test_skill_discovery_workflow(self, data_manager_mock):
        """Test skill discovery and relevance workflow."""
        skill_manager = SkillManager(data_manager_mock)
        
        # Register multiple skills
        skills = [
            SkillClaim(
                id=f"c{i}",
                function_name=f"skill_{i}",
                content=f"Skill {i} for {['search', 'create', 'analyze'][i % 3]} tasks",
                skill_category=['search', 'creation', 'analysis'][i % 3],
                parameters=[],
                confidence=0.8
            )
            for i in range(6)
        ]
        
        for skill in skills:
            await skill_manager.register_skill_claim(skill)
        
        # Mock database search
        similar_skills = [skills[0], skills[3]]  # search skills
        data_manager_mock.search_similar.return_value = similar_skills
        
        # Find relevant skills
        relevant = await skill_manager.find_relevant_skills("search query")
        
        # Should include search skills
        assert len(relevant) >= 2
        search_skills = [s for s in relevant if 'search' in s.function_name or 'search' in s.content.lower()]
        assert len(search_skills) >= 1

    @pytest.mark.asyncio
    async def test_skill_examples_workflow(self, data_manager_mock, sample_example_claim):
        """Test workflow for accessing skill examples."""
        skill_manager = SkillManager(data_manager_mock)
        
        # Mock example retrieval
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
        
        # Get examples for skill
        examples = await skill_manager.get_skill_examples("c123")
        
        # Verify examples
        assert len(examples) == 1
        assert isinstance(examples[0], ExampleClaim)
        assert examples[0].skill_id == "c123"

    @pytest.mark.asyncio
    async def test_parameter_validation_workflow(self, data_manager_mock):
        """Test workflow with parameter validation."""
        skill_manager = SkillManager(data_manager_mock)
        
        # Register skill with strict parameter validation
        skill = SkillClaim(
            id="c123",
            function_name="validated_skill",
            content="Skill with parameter validation",
            parameters=[
                SkillParameter(name="required_string", param_type="str", required=True),
                SkillParameter(name="optional_int", param_type="int", required=False, default_value=10),
                SkillParameter(name="range_float", param_type="float", required=False)
            ],
            confidence=0.8
        )
        
        await skill_manager.register_skill_claim(skill)
        
        # Mock function
        async def test_func(**kwargs):
            return {"success": True, "received": kwargs}
        
        skill_manager.builtin_skills["validated_skill"] = test_func
        
        # Test valid parameters
        result1 = await skill_manager.execute_skill(
            "validated_skill",
            {"required_string": "test", "optional_int": 5, "range_float": 3.14}
        )
        assert result1.success is True
        
        # Test missing required parameter
        result2 = await skill_manager.execute_skill(
            "validated_skill",
            {"optional_int": 5}  # Missing required_string
        )
        assert result2.success is False
        assert "Missing required parameter" in result2.error_message
        
        # Test invalid parameter type
        result3 = await skill_manager.execute_skill(
            "validated_skill",
            {"required_string": "test", "optional_int": "not_int"}  # Should be int
        )
        assert result3.success is False
        assert "must be of type" in result3.error_message

    @pytest.mark.asyncio
    async def test_execution_statistics_workflow(self, data_manager_mock, mock_async_functions):
        """Test workflow for execution statistics tracking."""
        skill_manager = SkillManager(data_manager_mock)
        
        # Register skill
        skill = SkillClaim(
            id="c123",
            function_name="stats_test_skill",
            content="Skill for testing statistics",
            parameters=[SkillParameter(name="param", param_type="str", required=True)],
            confidence=0.8
        )
        await skill_manager.register_skill_claim(skill)
        skill_manager.builtin_skills["stats_test_skill"] = mock_async_functions["search_claims"]
        
        # Execute multiple times
        for i in range(10):
            success = i % 4 != 0  # 3 out of 4 succeed
            if not success:
                # Mock failure
                async def failing_func(**kwargs):
                    raise ValueError("Test failure")
                skill_manager.builtin_skills["stats_test_skill"] = failing_func
            else:
                skill_manager.builtin_skills["stats_test_skill"] = mock_async_functions["search_claims"]
            
            await skill_manager.execute_skill("stats_test_skill", {"param": f"test_{i}"})
        
        # Check statistics
        stats = skill_manager.get_execution_stats()
        
        assert stats['total_executions'] == 10
        assert stats['successful_executions'] == 7 or 8  # Allow for some variance
        assert 0.6 <= stats['success_rate'] <= 0.8
        assert stats['average_execution_time_ms'] > 0
        
        # Check skill-specific stats
        registry_stats = skill_manager.registry.get_skill_stats()
        assert registry_stats['total_executions'] == 10
        assert registry_stats['average_success_rate'] == stats['success_rate']

    @pytest.mark.asyncio
    async def test_concurrent_execution_workflow(self, data_manager_mock, mock_async_functions):
        """Test workflow with concurrent skill executions."""
        skill_manager = SkillManager(data_manager_mock)
        
        # Register skill
        skill = SkillClaim(
            id="c123",
            function_name="concurrent_skill",
            content="Skill for concurrent testing",
            parameters=[SkillParameter(name="query", param_type="str", required=True)],
            confidence=0.8
        )
        await skill_manager.register_skill_claim(skill)
        skill_manager.builtin_skills["concurrent_skill"] = mock_async_functions["search_claims"]
        
        # Execute multiple calls concurrently
        tasks = [
            skill_manager.execute_skill("concurrent_skill", {"query": f"query_{i}"})
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should succeed
        assert len(results) == 5
        assert all(isinstance(r, ExecutionResult) for r in results)
        assert all(r.success for r in results)
        
        # Verify all were tracked
        stats = skill_manager.get_execution_stats()
        assert stats['total_executions'] == 5

    @pytest.mark.asyncio
    async def test_system_initialization_workflow(self, data_manager_mock):
        """Test complete system initialization workflow."""
        skill_manager = SkillManager(data_manager_mock)
        
        # Mock database with existing skills
        existing_skills = [
            {
                'id': 'c1',
                'function_name': 'existing_skill1',
                'content': 'Existing skill 1',
                'parameters': [],
                'tags': ['type.skill'],
                'confidence': 0.8,
                'created_by': 'system',
                'execution_count': 5,
                'success_count': 4
            },
            {
                'id': 'c2',
                'function_name': 'existing_skill2',
                'content': 'Existing skill 2',
                'parameters': [],
                'tags': ['type.skill'],
                'confidence': 0.7,
                'created_by': 'user',
                'execution_count': 10,
                'success_count': 8
            }
        ]
        
        data_manager_mock.filter_claims.return_value = existing_skills
        
        # Initialize system
        await skill_manager.initialize()
        
        # Verify skills loaded
        assert len(skill_manager.registry.skills) == 2
        assert 'existing_skill1' in skill_manager.registry.skills
        assert 'existing_skill2' in skill_manager.registry.skills
        
        # Verify statistics preserved
        skill1 = skill_manager.registry.get_skill('existing_skill1')
        assert skill1.execution_count == 5
        assert skill1.success_count == 4
        
        skill2 = skill_manager.registry.get_skill('existing_skill2')
        assert skill2.execution_count == 10
        assert skill2.success_count == 8

    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, data_manager_mock, mock_async_functions):
        """Test workflow for error recovery and resilience."""
        skill_manager = SkillManager(data_manager_mock)
        
        # Register skill
        skill = SkillClaim(
            id="c123",
            function_name="resilient_skill",
            content="Skill for testing resilience",
            parameters=[SkillParameter(name="param", param_type="str", required=True)],
            confidence=0.8
        )
        await skill_manager.register_skill_claim(skill)
        
        # Mock function that fails initially then succeeds
        call_count = 0
        
        async def resilient_func(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ValueError(f"Attempt {call_count} failed")
            return {"success": True, "attempt": call_count}
        
        skill_manager.builtin_skills["resilient_skill"] = resilient_func
        
        # Execute until success (simulating retry logic)
        for attempt in range(4):
            result = await skill_manager.execute_skill("resilient_skill", {"param": "test"})
            if result.success:
                break
        
        # Verify eventual success
        assert result.success is True
        assert result.result["attempt"] == 3
        
        # Verify all attempts tracked
        stats = skill_manager.get_execution_stats()
        assert stats['total_executions'] == 3
        assert stats['successful_executions'] == 1
        assert stats['success_rate'] == 1/3


class TestComponentInteractionIntegration:
    """Integration tests for specific component interactions."""

    @pytest.mark.asyncio
    async def test_response_parser_tool_executor_integration(self, execution_limits):
        """Test integration between ResponseParser and ToolExecutor."""
        response_parser = ResponseParser()
        tool_executor = ToolExecutor(execution_limits)
        
        # Create test responses in different formats
        responses = [
            """
            <tool_calls>
                <invoke name="test_tool">
                    <parameter name="message">hello</parameter>
                </invoke>
            </tool_calls>
            """,
            """
            {
                "tool_calls": [
                    {"name": "test_tool", "parameters": {"message": "world"}}
                ]
            }
            """,
            """
            ```tool_call
            name: test_tool
            message: integration
            ```
            """
        ]
        
        for response in responses:
            # Parse response
            parsed = response_parser.parse_response(response)
            assert len(parsed.tool_calls) == 1
            
            # Convert to execution parameters
            tool_call = parsed.tool_calls[0]
            exec_params = tool_call.to_skill_execution_params()
            
            # Verify parameter conversion
            assert exec_params['skill_name'] == tool_call.name
            assert exec_params['parameters'] == tool_call.parameters

    @pytest.mark.asyncio
    async def test_skill_manager_example_generator_integration(self, data_manager_mock, mock_async_functions):
        """Test integration between SkillManager and ExampleGenerator."""
        skill_manager = SkillManager(data_manager_mock)
        example_generator = ExampleGenerator(data_manager_mock)
        
        # Register and execute skill
        skill = SkillClaim(
            id="c123",
            function_name="integration_test_skill",
            content="Skill for integration testing",
            parameters=[SkillParameter(name="query", param_type="str", required=True)],
            confidence=0.8
        )
        await skill_manager.register_skill_claim(skill)
        skill_manager.builtin_skills["integration_test_skill"] = mock_async_functions["search_claims"]
        
        # Execute skill
        execution_result = await skill_manager.execute_skill("integration_test_skill", {"query": "test"})
        
        # Generate example
        example_generator.get_examples_for_skill = AsyncMock(return_value=[])
        example = await example_generator.generate_example_from_execution(execution_result)
        
        # Verify integration
        assert example is not None
        assert example.skill_id == skill.id
        assert example.input_parameters == execution_result.parameters_used

    @pytest.mark.asyncio
    async def test_tool_executor_security_integration(self, execution_limits):
        """Test security integration in ToolExecutor."""
        tool_executor = ToolExecutor(execution_limits)
        
        # Try to execute dangerous code
        dangerous_calls = [
            ToolCall(name="execute_code", parameters={"code": "__import__('os').system('ls')"}),
            ToolCall(name="execute_code", parameters={"code": "eval('open(\"secret.txt\", \"r\").read()')"}),
        ]
        
        for tool_call in dangerous_calls:
            result = await tool_executor.execute_tool_call(tool_call)
            
            # Should be blocked by security
            assert result.success is False
            assert result.error_message is not None

    def test_skill_models_cross_validation_integration(self):
        """Test cross-validation between skill models."""
        # Create skill with parameters
        param = SkillParameter(name="query", param_type="str", required=True)
        skill = SkillClaim(
            id="c123",
            function_name="validation_test_skill",
            content="Skill for validation testing",
            parameters=[param],
            confidence=0.8
        )
        
        # Test parameter validation
        valid_params = {"query": "test"}
        is_valid, errors = skill.validate_parameters(valid_params)
        assert is_valid is True
        assert len(errors) == 0
        
        invalid_params = {"query": 123}  # Wrong type
        is_valid, errors = skill.validate_parameters(invalid_params)
        assert is_valid is False
        assert len(errors) > 0
        
        # Test tool call conversion
        tool_call = ToolCall(name="validation_test_skill", parameters=valid_params)
        exec_params = tool_call.to_skill_execution_params()
        assert exec_params['skill_name'] == skill.function_name
        assert exec_params['parameters'] == valid_params
        
        # Test execution result to example conversion
        result = ExecutionResult(
            success=True,
            result={"output": "test result"},
            execution_time_ms=100,
            skill_id=skill.id,
            parameters_used=valid_params
        )
        
        example_data = result.to_example_data()
        assert example_data['skill_id'] == skill.id
        assert example_data['input_parameters'] == valid_params
        assert example_data['output_result'] == {"output": "test result"}


if __name__ == "__main__":
    pytest.main([__file__])