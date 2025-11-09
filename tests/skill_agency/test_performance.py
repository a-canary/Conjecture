"""
Performance tests and benchmarks for the skill-based agency system.
"""
import pytest
import asyncio
import time
import psutil
import os
from unittest.mock import AsyncMock, MagicMock
from typing import List, Dict, Any
import statistics

from src.processing.skill_manager import SkillManager
from src.processing.response_parser import ResponseParser
from src.processing.tool_executor import ToolExecutor, ExecutionLimits
from src.processing.example_generator import ExampleGenerator
from src.core.skill_models import SkillClaim, ExecutionResult, ToolCall, SkillParameter


class PerformanceBenchmark:
    """Utility class for performance benchmarking."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.start_memory = None
        self.end_time = None
        self.end_memory = None
        self.process = psutil.Process(os.getpid())
    
    def start(self):
        """Start benchmarking."""
        self.start_time = time.perf_counter()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
    
    def stop(self):
        """Stop benchmarking and return results."""
        self.end_time = time.perf_counter()
        self.end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'name': self.name,
            'duration_ms': (self.end_time - self.start_time) * 1000,
            'memory_delta_mb': self.end_memory - self.start_memory,
            'peak_memory_mb': self.end_memory
        }


class TestResponseParsingPerformance:
    """Performance tests for response parsing."""
    
    def test_xml_parsing_performance(self, response_parser, xml_response_samples):
        """Benchmark XML parsing performance."""
        benchmark = PerformanceBenchmark("XML Parsing")
        
        # Single small XML response
        benchmark.start()
        for _ in range(100):
            response_parser.parse_response(xml_response_samples["valid_single"])
        benchmark.stop()
        
        result = benchmark.stop()
        print(f"XML Parsing (100 iterations): {result['duration_ms']:.2f}ms, {result['memory_delta_mb']:.2f}MB")
        
        # Should be under 10ms per response
        assert result['duration_ms'] / 100 < 10, f"XML parsing too slow: {result['duration_ms'] / 100}ms per response"

    def test_json_parsing_performance(self, response_parser, json_response_samples):
        """Benchmark JSON parsing performance."""
        benchmark = PerformanceBenchmark("JSON Parsing")
        
        benchmark.start()
        for _ in range(100):
            response_parser.parse_response(json_response_samples["valid_single"])
        benchmark.stop()
        
        result = benchmark.stop()
        print(f"JSON Parsing (100 iterations): {result['duration_ms']:.2f}ms, {result['memory_delta_mb']:.2f}MB")
        
        # Should be under 10ms per response
        assert result['duration_ms'] / 100 < 10, f"JSON parsing too slow: {result['duration_ms'] / 100}ms per response"

    def test_markdown_parsing_performance(self, response_parser, markdown_response_samples):
        """Benchmark markdown parsing performance."""
        benchmark = PerformanceBenchmark("Markdown Parsing")
        
        benchmark.start()
        for _ in range(100):
            response_parser.parse_response(markdown_response_samples["valid_simple"])
        benchmark.stop()
        
        result = benchmark.stop()
        print(f"Markdown Parsing (100 iterations): {result['duration_ms']:.2f}ms, {result['memory_delta_mb']:.2f}MB")
        
        # Should be under 10ms per response
        assert result['duration_ms'] / 100 < 10, f"Markdown parsing too slow: {result['duration_ms'] / 100}ms per response"

    def test_large_response_parsing_performance(self, response_parser):
        """Test parsing performance with large responses."""
        # Create large XML response
        large_xml = """<tool_calls>"""
        for i in range(100):
            large_xml += f"""
            <invoke name="tool_{i}">
                <parameter name="param_{i}">value_{i}</parameter>
                <parameter name="data">{'x' * 100}</parameter>
            </invoke>"""
        large_xml += "</tool_calls>"
        
        benchmark = PerformanceBenchmark("Large XML Parsing")
        benchmark.start()
        response_parser.parse_response(large_xml)
        result = benchmark.stop()
        
        print(f"Large XML parsing (100 tools): {result['duration_ms']:.2f}ms, {result['memory_delta_mb']:.2f}MB")
        
        # Should still be reasonably fast even with large responses
        assert result['duration_ms'] < 100, f"Large XML parsing too slow: {result['duration_ms']}ms"

    def test_parsing_memory_efficiency(self, response_parser):
        """Test memory efficiency of parsing operations."""
        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        # Parse many responses
        responses = [
            '<tool_calls><invoke name="tool"><parameter name="param">value</parameter></invoke></tool_calls>',
            '{"tool_calls": [{"name": "tool", "parameters": {"param": "value"}}]}',
            '```tool_call\nname: tool\nparam: value\n```'
        ]
        
        for i in range(1000):
            response = responses[i % 3]
            response_parser.parse_response(response)
        
        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        print(f"Parsing memory increase after 1000 operations: {memory_increase:.2f}MB")
        
        # Memory increase should be reasonable (< 10MB for 1000 operations)
        assert memory_increase < 10, f"Memory increase too high: {memory_increase}MB"


class TestSkillExecutionPerformance:
    """Performance tests for skill execution."""
    
    @pytest.mark.asyncio
    async def test_skill_execution_performance(self, skill_manager, mock_async_functions):
        """Benchmark skill execution performance."""
        # Register skill
        skill = SkillClaim(
            id="c123",
            function_name="performance_test_skill",
            content="Skill for performance testing",
            parameters=[SkillParameter(name="query", param_type="str", required=True)],
            confidence=0.8
        )
        await skill_manager.register_skill_claim(skill)
        skill_manager.builtin_skills["performance_test_skill"] = mock_async_functions["search_claims"]
        
        benchmark = PerformanceBenchmark("Skill Execution")
        benchmark.start()
        
        # Execute skill multiple times
        for i in range(50):
            result = await skill_manager.execute_skill(
                "performance_test_skill",
                {"query": f"test_query_{i}"}
            )
            assert result.success
        
        result = benchmark.stop()
        print(f"Skill Execution (50 iterations): {result['duration_ms']:.2f}ms, {result['memory_delta_mb']:.2f}MB")
        
        # Should be under 100ms per execution
        avg_time = result['duration_ms'] / 50
        assert avg_time < 100, f"Skill execution too slow: {avg_time}ms per execution"

    @pytest.mark.asyncio
    async def test_concurrent_skill_execution_performance(self, skill_manager, mock_async_functions):
        """Benchmark concurrent skill execution performance."""
        # Register skill
        skill = SkillClaim(
            id="c123",
            function_name="concurrent_skill",
            content="Concurrent execution test skill",
            parameters=[SkillParameter(name="query", param_type="str", required=True)],
            confidence=0.8
        )
        await skill_manager.register_skill_claim(skill)
        skill_manager.builtin_skills["concurrent_skill"] = mock_async_functions["search_claims"]
        
        benchmark = PerformanceBenchmark("Concurrent Skill Execution")
        benchmark.start()
        
        # Execute 20 skills concurrently
        tasks = [
            skill_manager.execute_skill("concurrent_skill", {"query": f"query_{i}"})
            for i in range(20)
        ]
        
        results = await asyncio.gather(*tasks)
        result = benchmark.stop()
        
        # Verify all succeeded
        assert all(r.success for r in results)
        
        print(f"Concurrent Skill Execution (20 concurrent): {result['duration_ms']:.2f}ms, {result['memory_delta_mb']:.2f}MB")
        
        # Concurrent should be faster than sequential
        avg_time = result['duration_ms'] / 20
        assert avg_time < 50, f"Concurrent execution too slow: {avg_time}ms per execution"

    @pytest.mark.asyncio
    async def test_skill_discovery_performance(self, skill_manager, data_manager_mock):
        """Benchmark skill discovery performance."""
        # Register many skills
        for i in range(100):
            skill = SkillClaim(
                id=f"c{i}",
                function_name=f"skill_{i}",
                content=f"Skill {i} for testing discovery performance",
                skill_category="test",
                parameters=[],
                confidence=0.8
            )
            await skill_manager.register_skill_claim(skill)
        
        benchmark = PerformanceBenchmark("Skill Discovery")
        benchmark.start()
        
        # Search for relevant skills multiple times
        for _ in range(10):
            results = await skill_manager.find_relevant_skills("test query", limit=10)
        
        result = benchmark.stop()
        print(f"Skill Discovery (10 searches in 100 skills): {result['duration_ms']:.2f}ms")
        
        # Search should be fast even with many skills
        avg_time = result['duration_ms'] / 10
        assert avg_time < 50, f"Skill discovery too slow: {avg_time}ms per search"

    @pytest.mark.asyncio
    async def test_parameter_validation_performance(self, skill_manager):
        """Benchmark parameter validation performance."""
        # Create skill with complex parameters
        skill = SkillClaim(
            id="c123",
            function_name="validation_test_skill",
            content="Skill with complex parameters for validation testing",
            parameters=[
                SkillParameter(name="param1", param_type="str", required=True),
                SkillParameter(name="param2", param_type="int", required=False, default_value=10),
                SkillParameter(name="param3", param_type="float", required=False, default_value=3.14),
                SkillParameter(name="param4", param_type="bool", required=False, default_value=True),
                SkillParameter(name="param5", param_type="dict", required=False),
                SkillParameter(name="param6", param_type="list", required=False),
                SkillParameter(name="param7", param_type="any", required=False),
            ],
            confidence=0.8
        )
        await skill_manager.register_skill_claim(skill)
        
        benchmark = PerformanceBenchmark("Parameter Validation")
        benchmark.start()
        
        # Validate parameters multiple times
        test_params = {
            "param1": "test_string",
            "param2": 42,
            "param3": 2.71,
            "param4": False,
            "param5": {"key": "value"},
            "param6": [1, 2, 3],
            "param7": "any_value"
        }
        
        for _ in range(1000):
            is_valid, errors = skill.validate_parameters(test_params)
            assert is_valid
            assert len(errors) == 0
        
        result = benchmark.stop()
        print(f"Parameter Validation (1000 validations): {result['duration_ms']:.2f}ms")
        
        # Validation should be very fast
        avg_time = result['duration_ms'] / 1000
        assert avg_time < 1, f"Parameter validation too slow: {avg_time}ms per validation"


class TestToolExecutorPerformance:
    """Performance tests for tool execution."""
    
    @pytest.mark.asyncio
    async def test_safe_execution_performance(self, execution_limits, code_execution_samples):
        """Benchmark safe code execution performance."""
        executor = ToolExecutor(execution_limits)
        
        benchmark = PerformanceBenchmark("Safe Code Execution")
        benchmark.start()
        
        # Execute safe code multiple times
        for _ in range(20):
            result = await executor.safe_executor.execute_code(
                code_execution_samples["safe_math"],
                {}
            )
            assert result.success
        
        result = benchmark.stop()
        print(f"Safe Code Execution (20 iterations): {result['duration_ms']:.2f}ms, {result['memory_delta_mb']:.2f}MB")
        
        # Safe execution should complete in reasonable time
        avg_time = result['duration_ms'] / 20
        assert avg_time < 200, f"Safe execution too slow: {avg_time}ms per execution"

    @pytest.mark.asyncio
    async def test_security_validation_performance(self, execution_limits):
        """Benchmark security validation performance."""
        from src.processing.tool_executor import SecurityValidator
        
        validator = SecurityValidator(execution_limits)
        
        safe_codes = [
            "import math\nresult = math.sqrt(16)",
            "import json\nresult = json.dumps({'key': 'value'})",
            "import random\nresult = random.randint(1, 100)",
            "import datetime\nresult = datetime.datetime.now()",
            "import collections\nresult = collections.Counter('test')",
        ]
        
        benchmark = PerformanceBenchmark("Security Validation")
        benchmark.start()
        
        # Validate many code snippets
        for i in range(500):
            code = safe_codes[i % len(safe_codes)]
            is_safe, errors = validator.validate_code(code)
            assert is_safe
        
        result = benchmark.stop()
        print(f"Security Validation (500 validations): {result['duration_ms']:.2f}ms")
        
        # Security validation should be very fast
        avg_time = result['duration_ms'] / 500
        assert avg_time < 5, f"Security validation too slow: {avg_time}ms per validation"

    @pytest.mark.asyncio
    async def test_execution_memory_usage(self, execution_limits):
        """Test memory usage during execution."""
        executor = ToolExecutor(execution_limits)
        
        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        # Execute multiple operations
        codes = [
            "import math\nresult = [math.sqrt(i) for i in range(100)]",
            "import itertools\nresult = list(itertools.permutations('abc', 3))",
            "import collections\nresult = collections.defaultdict(list)",
        ]
        
        for code in codes:
            for _ in range(10):
                await executor.execute_tool_call(
                    ToolCall(name="execute_code", parameters={"code": code})
                )
        
        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        print(f"Execution memory increase: {memory_increase:.2f}MB")
        
        # Memory increase should be reasonable
        assert memory_increase < 20, f"Memory usage too high: {memory_increase}MB"


class TestExampleGenerationPerformance:
    """Performance tests for example generation."""
    
    @pytest.mark.asyncio
    async def test_example_generation_performance(self, example_generator):
        """Benchmark example generation performance."""
        # Create execution results
        execution_results = [
            ExecutionResult(
                success=True,
                result=f"result_{i}",
                execution_time_ms=50 + i * 5,
                skill_id="c123",
                parameters_used={"param": f"value_{i}"}
            )
            for i in range(50)
        ]
        
        # Mock get_examples_for_skill to return empty list
        example_generator.get_examples_for_skill = AsyncMock(return_value=[])
        
        benchmark = PerformanceBenchmark("Example Generation")
        benchmark.start()
        
        # Generate examples
        examples = await example_generator.batch_generate_examples(execution_results)
        
        result = benchmark.stop()
        print(f"Example Generation (50 examples): {result['duration_ms']:.2f}ms, {result['memory_delta_mb']:.2f}MB")
        
        # Example generation should be fast
        avg_time = result['duration_ms'] / 50
        assert avg_time < 20, f"Example generation too slow: {avg_time}ms per example"

    def test_quality_assessment_performance(self, example_generator, sample_execution_result):
        """Benchmark quality assessment performance."""
        benchmark = PerformanceBenchmark("Quality Assessment")
        benchmark.start()
        
        # Assess quality many times
        for _ in range(1000):
            quality = example_generator.quality_assessor.assess_example_quality(
                sample_execution_result,
                []
            )
            assert 0.0 <= quality <= 1.0
        
        result = benchmark.stop()
        print(f"Quality Assessment (1000 assessments): {result['duration_ms']:.2f}ms")
        
        # Quality assessment should be very fast
        avg_time = result['duration_ms'] / 1000
        assert avg_time < 2, f"Quality assessment too slow: {avg_time}ms per assessment"


class TestSystemPerformance:
    """System-wide performance tests."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow_performance(self, data_manager_mock, response_parser, skill_manager, mock_async_functions):
        """Benchmark complete workflow performance."""
        # Register skill
        skill = SkillClaim(
            id="c123",
            function_name="workflow_test_skill",
            content="Skill for end-to-end workflow testing",
            parameters=[SkillParameter(name="query", param_type="str", required=True)],
            confidence=0.8
        )
        await skill_manager.register_skill_claim(skill)
        skill_manager.builtin_skills["workflow_test_skill"] = mock_async_functions["search_claims"]
        
        # Sample LLM responses
        responses = [
            """<tool_calls><invoke name="workflow_test_skill"><parameter name="query">test1</parameter></invoke></tool_calls>""",
            """{"tool_calls": [{"name": "workflow_test_skill", "parameters": {"query": "test2"}}]}""",
            """```tool_call\nname: workflow_test_skill\nquery: test3\n```"""
        ]
        
        benchmark = PerformanceBenchmark("End-to-End Workflow")
        benchmark.start()
        
        # Process complete workflows
        for response in responses:
            # Parse response
            parsed = response_parser.parse_response(response)
            
            # Execute tool calls
            for tool_call in parsed.tool_calls:
                result = await skill_manager.execute_skill(
                    tool_call.name,
                    tool_call.parameters
                )
                assert result.success
        
        result = benchmark.stop()
        print(f"End-to-End Workflow (3 complete workflows): {result['duration_ms']:.2f}ms")
        
        # End-to-end should complete in reasonable time
        avg_time = result['duration_ms'] / 3
        assert avg_time < 150, f"End-to-end workflow too slow: {avg_time}ms per workflow"

    @pytest.mark.asyncio
    async def test_system_memory_efficiency(self, skill_manager, data_manager_mock):
        """Test overall system memory efficiency."""
        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        # Load many skills
        for i in range(50):
            skill = SkillClaim(
                id=f"c{i}",
                function_name=f"memory_test_skill_{i}",
                content=f"Skill {i} for memory testing",
                parameters=[
                    SkillParameter(name=f"param_{j}", param_type="str", required=True)
                    for j in range(5)
                ],
                examples=[f"Example {k}" for k in range(3)],
                confidence=0.8
            )
            await skill_manager.register_skill_claim(skill)
        
        # Execute many operations
        for i in range(20):
            skill_id = f"memory_test_skill_{i % 50}"
            # Mock execution
            result = ExecutionResult(
                success=True,
                result=f"result_{i}",
                execution_time_ms=50,
                skill_id=f"c{i % 50}",
                parameters_used={"param_0": "test"}
            )
            skill_manager._add_to_history(result)
        
        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        print(f"System memory increase (50 skills + 20 executions): {memory_increase:.2f}MB")
        
        # Memory increase should be reasonable
        assert memory_increase < 50, f"System memory usage too high: {memory_increase}MB"

    def test_component_startup_performance(self):
        """Test performance of component startup."""
        benchmark = PerformanceBenchmark("Component Startup")
        
        benchmark.start()
        
        # Initialize components
        data_manager = MagicMock()
        skill_manager = SkillManager(data_manager)
        response_parser = ResponseParser()
        execution_limits = ExecutionLimits()
        tool_executor = ToolExecutor(execution_limits)
        example_generator = ExampleGenerator(data_manager)
        
        result = benchmark.stop()
        print(f"Component Startup: {result['duration_ms']:.2f}ms, {result['memory_delta_mb']:.2f}MB")
        
        # Startup should be fast
        assert result['duration_ms'] < 100, f"Component startup too slow: {result['duration_ms']}ms"


class TestPerformanceRegression:
    """Performance regression tests."""
    
    @pytest.mark.asyncio
    async def test_parsing_performance_regression(self, response_parser, performance_benchmarks):
        """Test that parsing performance doesn't regress."""
        response = '<tool_calls><invoke name="test"><parameter name="param">value</parameter></invoke></tool_calls>'
        
        # Measure parsing time
        start_time = time.perf_counter()
        for _ in range(100):
            response_parser.parse_response(response)
        duration = (time.perf_counter() - start_time) * 1000
        avg_time = duration / 100
        
        print(f"Average parsing time: {avg_time:.3f}ms")
        assert avg_time < performance_benchmarks["max_parsing_time_ms"], \
            f"Parsing performance regression: {avg_time}ms > {performance_benchmarks['max_parsing_time_ms']}ms"

    @pytest.mark.asyncio
    async def test_execution_performance_regression(self, skill_manager, mock_async_functions, performance_benchmarks):
        """Test that execution performance doesn't regress."""
        # Register skill
        skill = SkillClaim(
            id="c123",
            function_name="perf_test_skill",
            content="Performance test skill",
            parameters=[SkillParameter(name="query", param_type="str", required=True)],
            confidence=0.8
        )
        await skill_manager.register_skill_claim(skill)
        skill_manager.builtin_skills["perf_test_skill"] = mock_async_functions["search_claims"]
        
        # Measure execution time
        start_time = time.perf_counter()
        for _ in range(10):
            result = await skill_manager.execute_skill("perf_test_skill", {"query": "test"})
            assert result.success
        duration = (time.perf_counter() - start_time) * 1000
        avg_time = duration / 10
        
        print(f"Average execution time: {avg_time:.3f}ms")
        assert avg_time < performance_benchmarks["max_execution_time_ms"], \
            f"Execution performance regression: {avg_time}ms > {performance_benchmarks['max_execution_time_ms']}ms"

    def test_memory_usage_regression(self, performance_benchmarks):
        """Test that memory usage doesn't regress."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Perform memory-intensive operations
        response_parser = ResponseParser()
        
        # Parse many responses
        for i in range(1000):
            response = f'<tool_calls><invoke name="tool_{i}"><parameter name="param">value_{i}</parameter></invoke></tool_calls>'
            response_parser.parse_response(response)
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        print(f"Memory increase: {memory_increase:.2f}MB")
        assert memory_increase < performance_benchmarks["max_memory_mb"], \
            f"Memory usage regression: {memory_increase}MB > {performance_benchmarks['max_memory_mb']}MB"


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-v"])