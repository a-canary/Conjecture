"""
Unit tests for ExampleGenerator component.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from src.processing.example_generator import ExampleGenerator, ExampleQualityAssessor
from src.core.skill_models import ExampleClaim, ExecutionResult


class TestExampleQualityAssessor:
    """Test cases for ExampleQualityAssessor class."""

    def test_assessor_initialization(self):
        """Test ExampleQualityAssessor initialization."""
        assessor = ExampleQualityAssessor()
        
        assert len(assessor.quality_factors) == 5
        assert assessor.quality_factors['execution_success'] == 0.4
        assert assessor.quality_factors['execution_time'] == 0.2
        assert assessor.quality_factors['output_complexity'] == 0.2
        assert assessor.quality_factors['parameter_diversity'] == 0.1
        assert assessor.quality_factors['result_uniqueness'] == 0.1

    def test_assess_successful_execution(self, sample_execution_result):
        """Test quality assessment of successful execution."""
        assessor = ExampleQualityAssessor()
        
        quality = assessor.assess_example_quality(sample_execution_result, [])
        
        assert 0.0 <= quality <= 1.0
        assert quality > 0.5  # Successful execution should have decent quality

    def test_assess_failed_execution(self):
        """Test quality assessment of failed execution."""
        assessor = ExampleQualityAssessor()
        
        failed_result = ExecutionResult(
            success=False,
            error_message="Failed",
            execution_time_ms=100,
            skill_id="c123",
            parameters_used={}
        )
        
        quality = assessor.assess_example_quality(failed_result, [])
        
        assert quality == 0.0  # Failed execution should have zero quality

    def test_assess_execution_time(self):
        """Test execution time quality assessment."""
        assessor = ExampleQualityAssessor()
        
        # Fast execution
        fast_result = ExecutionResult(
            success=True,
            result="fast",
            execution_time_ms=50,  # < 100ms
            skill_id="c123",
            parameters_used={}
        )
        
        # Medium execution
        medium_result = ExecutionResult(
            success=True,
            result="medium",
            execution_time_ms=500,  # < 1000ms
            skill_id="c123",
            parameters_used={}
        )
        
        # Slow execution
        slow_result = ExecutionResult(
            success=True,
            result="slow",
            execution_time_ms=5000,  # > 5000ms
            skill_id="c123",
            parameters_used={}
        )
        
        fast_quality = assessor.assess_example_quality(fast_result, [])
        medium_quality = assessor.assess_example_quality(medium_result, [])
        slow_quality = assessor.assess_example_quality(slow_result, [])
        
        assert fast_quality > medium_quality > slow_quality

    def test_assess_output_complexity(self):
        """Test output complexity quality assessment."""
        assessor = ExampleQualityAssessor()
        
        # Simple outputs
        simple_results = [
            ExecutionResult(success=True, result=None, execution_time_ms=100, skill_id="c123", parameters_used={}),
            ExecutionResult(success=True, result=True, execution_time_ms=100, skill_id="c123", parameters_used={}),
            ExecutionResult(success=True, result="short", execution_time_ms=100, skill_id="c123", parameters_used={}),
            ExecutionResult(success=True, result=42, execution_time_ms=100, skill_id="c123", parameters_used={})
        ]
        
        # Complex outputs
        complex_results = [
            ExecutionResult(success=True, result=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], execution_time_ms=100, skill_id="c123", parameters_used={}),
            ExecutionResult(success=True, result={"key1": "value1", "key2": "value2", "key3": "value3"}, execution_time_ms=100, skill_id="c123", parameters_used={})
        ]
        
        simple_qualities = [assessor.assess_example_quality(r, []) for r in simple_results]
        complex_qualities = [assessor.assess_example_quality(r, []) for r in complex_results]
        
        avg_simple = sum(simple_qualities) / len(simple_qualities)
        avg_complex = sum(complex_qualities) / len(complex_qualities)
        
        assert avg_complex > avg_simple

    def test_assess_parameter_diversity(self):
        """Test parameter diversity quality assessment."""
        assessor = ExampleQualityAssessor()
        
        # Create existing examples with similar parameters
        existing_example = ExampleClaim(
            skill_id="c123",
            input_parameters={"query": "search", "limit": 10},
            confidence=0.8
        )
        
        # New execution with similar parameters
        similar_result = ExecutionResult(
            success=True,
            result="result",
            execution_time_ms=100,
            skill_id="c123",
            parameters_used={"query": "search", "limit": 10}
        )
        
        # New execution with different parameters  
        diverse_result = ExecutionResult(
            success=True,
            result="result",
            execution_time_ms=100,
            skill_id="c123",
            parameters_used={"query": "different", "category": "new"}
        )
        
        similar_quality = assessor.assess_example_quality(similar_result, [existing_example])
        diverse_quality = assessor.assess_example_quality(diverse_result, [existing_example])
        
        assert diverse_quality > similar_quality

    def test_assess_result_uniqueness(self):
        """Test result uniqueness quality assessment."""
        assessor = ExampleQualityAssessor()
        
        # Create existing example with result
        existing_example = ExampleClaim(
            skill_id="c123",
            input_parameters={},
            output_result="same_result",
            confidence=0.8
        )
        
        # New execution with same result
        same_result = ExecutionResult(
            success=True,
            result="same_result",
            execution_time_ms=100,
            skill_id="c123",
            parameters_used={}
        )
        
        # New execution with different result
        different_result = ExecutionResult(
            success=True,
            result="different_result",
            execution_time_ms=100,
            skill_id="c123",
            parameters_used={}
        )
        
        same_quality = assessor.assess_example_quality(same_result, [existing_example])
        different_quality = assessor.assess_example_quality(different_result, [existing_example])
        
        assert different_quality > same_quality

    def test_create_parameter_signature(self):
        """Test parameter signature creation."""
        assessor = ExampleQualityAssessor()
        
        params = {
            "query": "test search",
            "limit": 10,
            "category": "research"
        }
        
        signature = assessor._create_parameter_signature(params)
        
        # Should create consistent signature
        assert isinstance(signature, str)
        assert "str13" in signature or "str11" in signature  # String with length
        assert "int" in signature

    def test_create_result_signature(self):
        """Test result signature creation."""
        assessor = ExampleQualityAssessor()
        
        # Test different result types
        test_cases = [
            None,
            True,
            42,
            "string_result",
            [1, 2, 3],
            {"key": "value"}
        ]
        
        signatures = [assessor._create_result_signature(result) for result in test_cases]
        
        # All should be different
        assert len(set(signatures)) == len(signatures)
        
        # Should contain type information
        assert "null" in signatures[0]
        assert "bool:" in signatures[1]
        assert "number:" in signatures[2]
        assert "str:" in signatures[3]
        assert "list:" in signatures[4]
        assert "dict:" in signatures[5]

    def test_signatures_similar(self):
        """Test signature similarity checking."""
        assessor = ExampleQualityAssessor()
        
        sig1 = '{"category":"str8","limit":"int10","query":"str4"}'
        sig2 = '{"category":"str8","limit":"int10","query":"str4"}'
        sig3 = '{"category":"str5","limit":"int5","query":"str4"}'
        
        assert assessor._signatures_similar(sig1, sig2) is True
        assert assessor._signatures_similar(sig1, sig3) is False


class TestExampleGenerator:
    """Test cases for ExampleGenerator class."""

    def test_generator_initialization(self, data_manager_mock):
        """Test ExampleGenerator initialization."""
        generator = ExampleGenerator(data_manager_mock)
        
        assert generator.data_manager == data_manager_mock
        assert isinstance(generator.quality_assessor, ExampleQualityAssessor)
        assert generator.min_quality_threshold == 0.3
        assert generator.max_examples_per_skill == 50
        assert generator.generation_cooldown_minutes == 5
        assert len(generator.generation_history) == 0

    def test_generator_custom_settings(self, data_manager_mock):
        """Test ExampleGenerator with custom settings."""
        generator = ExampleGenerator(
            data_manager_mock,
            min_quality_threshold=0.5,
            max_examples_per_skill=100,
            generation_cooldown_minutes=10
        )
        
        assert generator.min_quality_threshold == 0.5
        assert generator.max_examples_per_skill == 100
        assert generator.generation_cooldown_minutes == 10

    @pytest.mark.asyncio
    async def test_generate_example_from_successful_execution(self, example_generator, sample_execution_result, data_manager_mock):
        """Test generating example from successful execution."""
        # Mock get_examples_for_skill
        example_generator.get_examples_for_skill = AsyncMock(return_value=[])
        
        # Mock _create_example_claim
        expected_example = ExampleClaim(
            skill_id=sample_execution_result.skill_id,
            input_parameters=sample_execution_result.parameters_used,
            confidence=0.8
        )
        example_generator._create_example_claim = AsyncMock(return_value=expected_example)
        
        result = await example_generator.generate_example_from_execution(sample_execution_result)
        
        assert result == expected_example
        example_generator.get_examples_for_skill.assert_called_once_with(sample_execution_result.skill_id)
        example_generator._create_example_claim.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_example_from_failed_execution(self, example_generator):
        """Test that example is not generated from failed execution."""
        failed_result = ExecutionResult(
            success=False,
            error_message="Failed",
            execution_time_ms=100,
            skill_id="c123",
            parameters_used={}
        )
        
        result = await example_generator.generate_example_from_execution(failed_result)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_generate_example_low_quality(self, example_generator, sample_execution_result):
        """Test that low quality examples are not generated."""
        # Mock low quality assessment
        example_generator.quality_assessor.assess_example_quality = MagicMock(return_value=0.1)
        example_generator.get_examples_for_skill = AsyncMock(return_value=[])
        
        result = await example_generator.generate_example_from_execution(sample_execution_result)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_generate_example_too_many_examples(self, example_generator, sample_execution_result):
        """Test that examples are not generated if limit reached."""
        # Mock that skill already has max examples
        many_examples = [ExampleClaim(skill_id=sample_execution_result.skill_id, input_parameters={}, confidence=0.8)] * 50
        example_generator.get_examples_for_skill = AsyncMock(return_value=many_examples)
        
        result = await example_generator.generate_example_from_execution(sample_execution_result)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_generate_example_within_cooldown(self, example_generator, sample_execution_result):
        """Test that examples are not generated during cooldown."""
        # Add recent generation to history
        example_generator.generation_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'skill_id': sample_execution_result.skill_id,
            'execution_id': 'prev_exec',
            'example_id': 'prev_example',
            'quality': 0.8,
            'execution_time_ms': 100,
            'parameters_count': 2
        })
        
        example_generator.get_examples_for_skill = AsyncMock(return_value=[])
        
        result = await example_generator.generate_example_from_execution(sample_execution_result)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_generate_example_after_cooldown(self, example_generator, sample_execution_result):
        """Test that examples are generated after cooldown."""
        # Add old generation to history (beyond cooldown)
        old_timestamp = (datetime.utcnow() - timedelta(minutes=10)).isoformat()
        example_generator.generation_history.append({
            'timestamp': old_timestamp,
            'skill_id': sample_execution_result.skill_id,
            'execution_id': 'prev_exec',
            'example_id': 'prev_example',
            'quality': 0.8,
            'execution_time_ms': 100,
            'parameters_count': 2
        })
        
        example_generator.get_examples_for_skill = AsyncMock(return_value=[])
        
        # Mock successful example creation
        expected_example = ExampleClaim(
            skill_id=sample_execution_result.skill_id,
            input_parameters=sample_execution_result.parameters_used,
            confidence=0.8
        )
        example_generator._create_example_claim = AsyncMock(return_value=expected_example)
        
        result = await example_generator.generate_example_from_execution(sample_execution_result)
        
        assert result == expected_example

    @pytest.mark.asyncio
    async def test_get_examples_for_skill(self, example_generator, data_manager_mock, sample_example_claim):
        """Test getting examples for a skill."""
        # Mock database response
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
        
        examples = await example_generator.get_examples_for_skill("c123")
        
        assert len(examples) == 1
        assert examples[0].skill_id == "c123"
        assert isinstance(examples[0], ExampleClaim)

    @pytest.mark.asyncio
    async def test_get_examples_for_skill_with_invalid_claims(self, example_generator, data_manager_mock):
        """Test handling invalid example claims."""
        # Mix valid and invalid claims
        valid_dict = {
            'id': 'c123',
            'skill_id': 'c123',
            'input_parameters': {},
            '-tags': ['type.example'],  # Should be 'tags'
            'confidence': 0.8
        }
        
        data_manager_mock.filter_claims.return_value = [valid_dict]
        
        examples = await example_generator.get_examples_for_skill("c123")
        
        # Should handle gracefully and return empty list
        assert len(examples) == 0

    @pytest.mark.asyncio
    async def test_get_examples_for_skill_error(self, example_generator, data_manager_mock):
        """Test error handling in get_examples_for_skill."""
        data_manager_mock.filter_claims.side_effect = Exception("Database error")
        
        examples = await example_generator.get_examples_for_skill("c123")
        
        assert len(examples) == 0

    def test_should_generate_example_success(self, example_generator, sample_execution_result):
        """Test should_generate_example with successful execution."""
        examples = []
        
        should_generate = example_generator._should_generate_example(sample_execution_result, examples)
        
        # New skill, successful execution should generate
        assert should_generate is True

    def test_should_generate_example_too_many_examples(self, example_generator, sample_execution_result):
        """Test should_generate_example when at limit."""
        many_examples = [ExampleClaim(skill_id=sample_execution_result.skill_id, input_parameters={}, confidence=0.8)] * 50
        
        should_generate = example_generator._should_generate_example(sample_execution_result, many_examples)
        
        assert should_generate is False

    def test_should_generate_example_failed_execution(self, example_generator):
        """Test should_generate_example with failed execution."""
        failed_result = ExecutionResult(
            success=False,
            error_message="Failed",
            execution_time_ms=100,
            skill_id="c123",
            parameters_used={}
        )
        
        should_generate = example_generator._should_generate_example(failed_result, [])
        
        assert should_generate is False

    def test_should_generate_example_no_parameters(self, example_generator):
        """Test should_generate_example with no parameters."""
        no_params_result = ExecutionResult(
            success=True,
            result="result",
            execution_time_ms=100,
            skill_id="c123",
            parameters_used={}  # Empty parameters
        )
        
        should_generate = example_generator._should_generate_example(no_params_result, [])
        
        assert should_generate is False

    def test_should_generate_example_in_cooldown(self, example_generator, sample_execution_result):
        """Test should_generate_example during cooldown."""
        # Add recent generation
        example_generator.generation_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'skill_id': sample_execution_result.skill_id,
            'execution_id': 'prev_exec',
            'example_id': 'prev_example',
            'quality': 0.8,
            'execution_time_ms': 100,
            'parameters_count': 2
        })
        
        should_generate = example_generator._should_generate_example(sample_execution_result, [])
        
        assert should_generate is False

    @pytest.mark.asyncio
    async def test_create_example_claim(self, example_generator, sample_execution_result, data_manager_mock):
        """Test creating an example claim."""
        quality = 0.85
        
        mock_claim = MagicMock(id="c789", created_at=datetime.utcnow())
        data_manager_mock.create_claim.return_value = mock_claim
        
        result = await example_generator._create_example_claim(sample_execution_result, quality)
        
        assert result is not None
        assert isinstance(result, ExampleClaim)
        assert result.skill_id == sample_execution_result.skill_id
        assert result.input_parameters == sample_execution_result.parameters_used
        assert result.output_result == sample_execution_result.result
        assert result.example_quality == quality
        
        # Verify database call
        data_manager_mock.create_claim.assert_called_once()
        call_args = data_manager_mock.create_claim.call_args[1]
        assert 'type.example' in call_args['tags']
        assert 'auto_generated' in call_args['tags']
        assert call_args['created_by'] == 'example_generator'

    @pytest.mark.asyncio
    async def test_create_example_claim_error(self, example_generator, sample_execution_result, data_manager_mock):
        """Test error handling in create_example_claim."""
        data_manager_mock.create_claim.side_effect = Exception("Database error")
        
        result = await example_generator._create_example_claim(sample_execution_result, 0.8)
        
        assert result is None

    def test_generate_example_content(self, example_generator, sample_execution_result):
        """Test example content generation."""
        content = example_generator._generate_example_content(sample_execution_result)
        
        assert sample_execution_result.skill_id in content
        assert "query=" in content or "limit=" in content  # Parameters should be included
        assert "->" in content  # Arrow indicating result
        
        # Fast execution should not include time
        if sample_execution_result.execution_time_ms <= 1000:
            assert "took" not in content

    def test_generate_example_content_with_slow_execution(self, example_generator):
        """Test example content generation with slow execution."""
        slow_result = ExecutionResult(
            success=True,
            result="result",
            execution_time_ms=2000,  # > 1000ms
            skill_id="skill",
            parameters_used={"param": "value"}
        )
        
        content = example_generator._generate_example_content(slow_result)
        
        assert "took 2000ms" in content

    def test_format_parameters(self, example_generator):
        """Test parameter formatting."""
        params = {
            "query": "test search",
            "limit": 10,
            "active": True,
            "data": {"key": "value"}
        }
        
        formatted = example_generator._format_parameters(params)
        
        assert "query=\"test search\"" in formatted
        assert "limit=10" in formatted
        assert "active=True" in formatted
        assert "data={'key': 'value'}" in formatted

    def test_format_parameters_empty(self, example_generator):
        """Test formatting empty parameters."""
        formatted = example_generator._format_parameters({})
        
        assert formatted == ""

    def test_format_parameters_long_string(self, example_generator):
        """Test formatting long string parameters."""
        long_string = "x" * 60  # Longer than 50 chars
        params = {"long_param": long_string}
        
        formatted = example_generator._format_parameters(params)
        
        # Should be truncated
        assert "..." in formatted
        assert len(formatted) < len(long_string)

    def test_format_result(self, example_generator):
        """Test result formatting."""
        test_cases = [
            None,
            "string result",
            [1, 2, 3, 4, 5],
            {"key": "value", "number": 42},
            42,
            True
        ]
        
        for result in test_cases:
            formatted = example_generator._format_result(result)
            
            if result is None:
                assert formatted == "None"
            elif isinstance(result, str):
                if len(result) <= 100:
                    assert f'"{result}"' in formatted
                else:
                    assert "..." in formatted
            elif isinstance(result, (int, float, bool)):
                assert str(result) in formatted
            else:
                # Complex types should be JSON formatted
                assert "{" in formatted or "[" in formatted or str(result) in formatted

    def test_record_generation(self, example_generator, sample_execution_result, sample_example_claim):
        """Test generation recording."""
        quality = 0.85
        
        example_generator._record_generation(sample_execution_result, sample_example_claim, quality)
        
        assert len(example_generator.generation_history) == 1
        
        record = example_generator.generation_history[0]
        assert record['skill_id'] == sample_execution_result.skill_id
        assert record['example_id'] == sample_example_claim.id
        assert record['quality'] == quality
        assert record['execution_time_ms'] == sample_execution_result.execution_time_ms
        assert 'timestamp' in record

    def test_generation_history_size_limit(self, example_generator, sample_execution_result, sample_example_claim):
        """Test that generation history respects size limit."""
        example_generator.max_history_size = 3
        
        # Add more records than the limit
        for i in range(5):
            result = ExecutionResult(
                success=True,
                result=f"result_{i}",
                execution_time_ms=100,
                skill_id="c123",
                parameters_used={"i": i}
            )
            example = ExampleClaim(
                id=f"c{i}",
                skill_id="c123",
                input_parameters={"i": i},
                confidence=0.8
            )
            example_generator._record_generation(result, example, 0.8)
        
        # Should only keep the last 3
        assert len(example_generator.generation_history) == 3

    @pytest.mark.asyncio
    async def test_batch_generate_examples(self, example_generator, data_manager_mock):
        """Test batch generation of examples."""
        # Create multiple execution results
        execution_results = [
            ExecutionResult(
                success=True,
                result=f"result_{i}",
                execution_time_ms=100 + i * 10,
                skill_id="c123",
                parameters_used={"param": i}
            )
            for i in range(3)
        ]
        
        # Mock individual generation
        async def mock_generate(execution_result):
            return ExampleClaim(
                id=f"c{100 + execution_results.index(execution_result)}",
                skill_id=execution_result.skill_id,
                input_parameters=execution_result.parameters_used,
                confidence=0.8
            )
        
        example_generator.generate_example_from_execution = AsyncMock(side_effect=mock_generate)
        
        generated_examples = await example_generator.batch_generate_examples(execution_results)
        
        assert len(generated_examples) == 3
        assert all(isinstance(example, ExampleClaim) for example in generated_examples)
        
        # Verify individual generation was called for each result
        assert example_generator.generate_example_from_execution.call_count == 3

    @pytest.mark.asyncio
    async def test_batch_generate_examples_with_errors(self, example_generator, data_manager_mock):
        """Test batch generation with some errors."""
        execution_results = [
            ExecutionResult(success=True, result="result1", execution_time_ms=100, skill_id="c123", parameters_used={}),
            ExecutionResult(success=False, error_message="failed", execution_time_ms=100, skill_id="c123", parameters_used={}),
            ExecutionResult(success=True, result="result3", execution_time_ms=100, skill_id="c123", parameters_used={})
        ]
        
        # Mock individual generation (success for successful results, None for failed ones)
        async def mock_generate(execution_result):
            if execution_result.success:
                return ExampleClaim(skill_id=execution_result.skill_id, input_parameters={}, confidence=0.8)
            return None
        
        example_generator.generate_example_from_execution = AsyncMock(side_effect=mock_generate)
        
        generated_examples = await example_generator.batch_generate_examples(execution_results)
        
        assert len(generated_examples) == 2  # Only successful ones

    def test_get_generation_stats_empty(self, example_generator):
        """Test generation statistics with no history."""
        stats = example_generator.get_generation_stats()
        
        assert stats['total_generated'] == 0
        assert stats['average_quality'] == 0.0
        assert len(stats['most_generated_skills']) == 0
        assert stats['generation_rate_per_hour'] == 0

    def test_get_generation_stats_with_data(self, example_generator):
        """Test generation statistics with history data."""
        # Add some history records
        base_time = datetime.utcnow()
        
        for i in range(5):
            timestamp = base_time - timedelta(hours=i)
            example_generator.generation_history.append({
                'timestamp': timestamp.isoformat(),
                'skill_id': f'skill_{i % 2}',  # Two different skills
                'execution_id': f'exec_{i}',
                'example_id': f'example_{i}',
                'quality': 0.6 + i * 0.1,
                'execution_time_ms': 100 + i * 10,
                'parameters_count': 2
            })
        
        stats = example_generator.get_generation_stats()
        
        assert stats['total_generated'] == 5
        assert stats['average_quality'] > 0.6
        assert len(stats['most_generated_skills']) == 2
        
        # Most generated skills should be sorted by count
        skill_counts = dict(stats['most_generated_skills'])
        assert sum(skill_counts.values()) == 5

    def test_get_generation_stats_recent_rate(self, example_generator):
        """Test generation rate calculation for recent generations."""
        # Add recent records (within last hour)
        recent_time = datetime.utcnow() - timedelta(minutes=30)
        
        for i in range(3):
            example_generator.generation_history.append({
                'timestamp': recent_time.isoformat(),
                'skill_id': 'recent_skill',
                'execution_id': f'recent_exec_{i}',
                'example_id': f'recent_example_{i}',
                'quality': 0.8,
                'execution_time_ms': 100,
                'parameters_count': 2
            })
        
        stats = example_generator.get_generation_stats()
        
        assert stats['generation_rate_per_hour'] == 3

    @pytest.mark.asyncio
    async def test_cleanup_low_quality_examples(self, example_generator, data_manager_mock):
        """Test cleanup of low-quality examples."""
        # Mix of high and low quality examples
        example_dicts = [
            {
                'id': 'c123',
                'skill_id': 'skill1',
                'input_parameters': {},
                'example_quality': 0.8,
                'tags': ['type.example'],
                'content': 'content',
                'confidence': 0.8
            },
            {
                'id': 'c456',
                'skill_id': 'skill1', 
                'input_parameters': {},
                'example_quality': 0.1,  # Low quality
                'tags': ['type.example'],
                'content': 'content',
                'confidence': 0.8
            },
            {
                'id': 'c789',
                'skill_id': 'skill2',
                'input_parameters': {},
                'example_quality': 0.3,  # Above threshold
                'tags': ['type.example'],
                'content': 'content',
                'confidence': 0.8
            }
        ]
        
        data_manager_mock.filter_claims.return_value = example_dicts
        data_manager_mock.delete_claim.return_value = True
        
        removed_count = await example_generator.cleanup_low_quality_examples(min_quality=0.2)
        
        # Should remove only the one below threshold
        assert removed_count == 1
        data_manager_mock.delete_claim.assert_called_once_with('c456')

    @pytest.mark.asyncio
    async def test_cleanup_low_quality_examples_error(self, example_generator, data_manager_mock):
        """Test error handling in cleanup."""
        data_manager_mock.filter_claims.side_effect = Exception("Database error")
        
        removed_count = await example_generator.cleanup_low_quality_examples()
        
        assert removed_count == 0


class TestExampleGeneratorIntegration:
    """Integration tests for ExampleGenerator."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, example_generator, sample_execution_result, data_manager_mock):
        """Test full example generation workflow."""
        # Mock dependencies
        example_generator.get_examples_for_skill = AsyncMock(return_value=[])
        data_manager_mock.create_claim.return_value = MagicMock(id="c789")
        
        # Generate example
        example = await example_generator.generate_example_from_execution(sample_execution_result)
        
        # Verify result
        assert isinstance(example, ExampleClaim)
        assert example.skill_id == sample_execution_result.skill_id
        assert example.example_quality > 0.3  # Should pass quality threshold
        
        # Verify recording
        assert len(example_generator.generation_history) == 1
        assert example_generator.generation_history[0]['skill_id'] == sample_execution_result.skill_id

    @pytest.mark.asyncio
    async def test_quality_threshold_workflow(self, example_generator, data_manager_mock):
        """Test workflow with quality threshold filtering."""
        # Create execution with potentially low quality
        low_quality_result = ExecutionResult(
            success=True,
            result="simple",  # Might get low complexity score
            execution_time_ms=10000,  # Slow execution
            skill_id="c123",
            parameters_used={}
        )
        
        example_generator.get_examples_for_skill = AsyncMock(return_value=[])
        
        # Generate example
        example = await example_generator.generate_example_from_execution(low_quality_result)
        
        # May or may not generate depending on quality assessment
        if example:
            assert example.example_quality >= example_generator.min_quality_threshold
        else:
            # Should not have been generated if quality was too low
            assert len(example_generator.generation_history) == 0


if __name__ == "__main__":
    pytest.main([__file__])