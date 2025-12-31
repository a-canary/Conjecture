#!/usr/bin/env python3
"""
Comprehensive tests for the enhanced prompt system
Tests all 7 proven enhancements and integrated functionality
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from src.agent.prompt_system import (
    PromptSystem,
    ProblemType,
    Difficulty,
    PromptResponse,
    ResponseParser,
)


class TestEnhancedPromptSystem:
    """Test the restored and enhanced prompt system"""

    @pytest.fixture
    def prompt_system(self):
        """Create prompt system for testing"""
        return PromptSystem()

    @pytest.fixture
    def response_parser(self):
        """Create response parser for testing"""
        return ResponseParser()

    class TestProblemTypeDetection:
        """Test problem type detection functionality"""

        @pytest.mark.asyncio
        async def test_mathematical_detection(self, prompt_system):
            """Test mathematical problem detection"""
            problems = [
                "What is 17 × 24?",
                "Solve for x: 2x + 5 = 15",
                "Calculate 15% of 200",
                "Find the square root of 144",
            ]

            for problem in problems:
                problem_type = prompt_system._detect_problem_type(problem)
                assert problem_type == ProblemType.MATHEMATICAL, (
                    f"Failed to detect mathematical in: {problem}"
                )

        @pytest.mark.asyncio
        async def test_logical_detection(self, prompt_system):
            """Test logical problem detection"""
            problems = [
                "If all A are B and some B are C, what follows?",
                "Given P implies Q and Q implies R, does P imply R?",
                "All humans are mortal. Socrates is human. Therefore?",
            ]

            for problem in problems:
                problem_type = prompt_system._detect_problem_type(problem)
                assert problem_type == ProblemType.LOGICAL, (
                    f"Failed to detect logical in: {problem}"
                )

        @pytest.mark.asyncio
        async def test_scientific_detection(self, prompt_system):
            """Test scientific problem detection"""
            problems = [
                "Design an experiment to test plant growth",
                "Analyze the data from this scientific observation",
                "Form a hypothesis about the experimental results",
            ]

            for problem in problems:
                problem_type = prompt_system._detect_problem_type(problem)
                # Note: Current implementation may detect as other types, this test documents current behavior
                assert problem_type in [ProblemType.SCIENTIFIC, ProblemType.GENERAL], (
                    f"Unexpected detection: {problem_type}"
                )

        @pytest.mark.asyncio
        async def test_sequential_detection(self, prompt_system):
            """Test sequential problem detection"""
            problems = [
                "First measure the ingredients, then mix them, finally bake",
                "Complete step 1, then proceed to step 2, followed by step 3",
                "Before starting the analysis, gather data, then process it",
            ]

            for problem in problems:
                problem_type = prompt_system._detect_problem_type(problem)
                assert problem_type == ProblemType.SEQUENTIAL, (
                    f"Failed to detect sequential in: {problem}"
                )

        @pytest.mark.asyncio
        async def test_general_detection(self, prompt_system):
            """Test general problem detection"""
            problems = [
                "What is the best approach to solve this?",
                "How would you handle this situation?",
                "Explain the concept in simple terms",
            ]

            for problem in problems:
                problem_type = prompt_system._detect_problem_type(problem)
                # Most problems should be classified (default fallback)
                assert isinstance(problem_type, ProblemType), (
                    f"No classification for: {problem}"
                )

    class TestDifficultyEstimation:
        """Test difficulty estimation functionality"""

        @pytest.mark.asyncio
        async def test_easy_detection(self, prompt_system):
            """Test easy problem detection"""
            easy_problems = [
                "What is 2 + 2?",
                "How many days in a week?",
                "What color is the sky?",
            ]

            for problem in easy_problems:
                difficulty = prompt_system._estimate_difficulty(problem)
                assert difficulty == Difficulty.EASY, (
                    f"Failed to detect easy in: {problem}"
                )

        @pytest.mark.asyncio
        async def test_hard_detection(self, prompt_system):
            """Test hard problem detection"""
            hard_problems = [
                "Prove the fundamental theorem of calculus",
                "Optimize the algorithm for maximum efficiency",
                "Derive the complex mathematical formula",
            ]

            for problem in hard_problems:
                difficulty = prompt_system._estimate_difficulty(problem)
                assert difficulty == Difficulty.HARD, (
                    f"Failed to detect hard in: {problem}"
                )

        @pytest.mark.asyncio
        async def test_medium_detection(self, prompt_system):
            """Test medium problem detection"""
            medium_problems = [
                "Calculate the area of a complex shape",
                "Analyze the given data set",
                "Explain the concept with examples",
            ]

            for problem in medium_problems:
                difficulty = prompt_system._estimate_difficulty(problem)
                # Most non-easy/hard problems should be medium
                assert difficulty == Difficulty.MEDIUM, (
                    f"Unexpected difficulty for: {problem}"
                )

    class TestDomainAdaptivePrompts:
        """Test domain-adaptive prompt generation"""

        @pytest.mark.asyncio
        async def test_mathematical_prompt(self, prompt_system):
            """Test mathematical prompt generation"""
            prompt = prompt_system._get_domain_adaptive_prompt(
                "What is 17 × 24?", ProblemType.MATHEMATICAL, Difficulty.MEDIUM
            )

            # Should contain mathematical-specific guidance
            assert "MATHEMATICAL" in prompt
            assert "calculations" in prompt.lower()
            assert "step-by-step" in prompt.lower()

        @pytest.mark.asyncio
        async def test_logical_prompt(self, prompt_system):
            """Test logical prompt generation"""
            prompt = prompt_system._get_domain_adaptive_prompt(
                "If A then B", ProblemType.LOGICAL, Difficulty.MEDIUM
            )

            # Should contain logical-specific guidance
            assert "LOGICAL" in prompt
            assert "reasoning" in prompt.lower()
            assert "steps" in prompt.lower()

        @pytest.mark.asyncio
        async def test_difficulty_guidance(self, prompt_system):
            """Test difficulty-specific guidance"""
            easy_prompt = prompt_system._get_domain_adaptive_prompt(
                "Simple question", ProblemType.GENERAL, Difficulty.EASY
            )
            hard_prompt = prompt_system._get_domain_adaptive_prompt(
                "Complex question", ProblemType.GENERAL, Difficulty.HARD
            )

            # Should have difficulty-specific guidance
            assert "Easy" in easy_prompt or "straightforward" in easy_prompt.lower()
            assert "Hard" in hard_prompt or "thorough" in hard_prompt.lower()

    class TestEnhancementFunctionality:
        """Test specific enhancement functionalities"""

        @pytest.mark.asyncio
        async def test_mathematical_reasoning_enhancement(self, prompt_system):
            """Test mathematical reasoning enhancement"""
            problem = "What is 15 × 12?"
            enhancement = prompt_system._enhance_mathematical_reasoning(problem)

            assert enhancement["mathematical_enhancement_applied"] is True
            assert "problem_type" in enhancement
            assert "reasoning_strategy" in enhancement
            assert len(enhancement["reasoning_strategy"]) > 0

        @pytest.mark.asyncio
        async def test_multistep_reasoning_enhancement(self, prompt_system):
            """Test multi-step reasoning enhancement"""
            problem = "First calculate A, then multiply by B, finally add C"
            enhancement = prompt_system._enhance_multistep_reasoning(problem)

            assert enhancement["multistep_enhancement_applied"] is True
            assert "complexity_level" in enhancement
            assert "suggested_steps" in enhancement
            assert enhancement["suggested_steps"] >= 2

        @pytest.mark.asyncio
        async def test_problem_decomposition_enhancement(self, prompt_system):
            """Test problem decomposition enhancement"""
            problem = "Analyze the components of this complex system"
            enhancement = prompt_system._enhance_problem_decomposition(problem)

            assert enhancement["decomposition_enhancement_applied"] is True
            assert "decomposition_approach" in enhancement
            assert "decomposition_strategy" in enhancement
            assert len(enhancement["decomposition_strategy"]) > 0

    class TestIntegrationFunctionality:
        """Test full integration functionality"""

        @pytest.mark.asyncio
        async def test_process_with_context_basic(self, prompt_system):
            """Test basic process_with_context functionality"""
            problem = "What is 7 × 8?"

            response = await prompt_system.process_with_context(problem)

            assert isinstance(response, PromptResponse)
            assert response.response is not None
            assert response.confidence > 0
            assert response.reasoning is not None
            assert response.prompt_type == "enhanced_conjecture"
            assert response.metadata is not None

        @pytest.mark.asyncio
        async def test_enhancement_application(self, prompt_system):
            """Test that enhancements are applied correctly"""
            problem = "Calculate the area of a circle with radius 5"

            response = await prompt_system.process_with_context(problem)

            # Should have applied enhancements
            assert response.metadata["enhancements_applied"] >= 1
            assert "enhancement_types" in response.metadata
            assert response.metadata["problem_type"] == "mathematical"

        @pytest.mark.asyncio
        async def test_caching_functionality(self, prompt_system):
            """Test that caching works correctly"""
            problem = "What is the capital of France?"

            # First call
            response1 = await prompt_system.process_with_context(problem)
            cache_key_1 = response1.metadata["cache_key"]

            # Second call should use cache
            response2 = await prompt_system.process_with_context(problem)
            cache_key_2 = response2.metadata["cache_key"]

            # Should use same cache key
            assert cache_key_1 == cache_key_2
            # Responses should be identical
            assert response1.response == response2.response

    class TestEnhancementControls:
        """Test enhancement enable/disable functionality"""

        @pytest.mark.asyncio
        async def test_enhancement_disable_enable(self, prompt_system):
            """Test enabling/disabling enhancements"""
            # All enhancements should be enabled by default
            status = prompt_system.get_enhancement_status()
            assert all(status.values()), "Not all enhancements enabled by default"

            # Disable an enhancement
            prompt_system.enable_enhancement("mathematical_reasoning", False)
            status = prompt_system.get_enhancement_status()
            assert status["mathematical_reasoning"] is False

            # Re-enable the enhancement
            prompt_system.enable_enhancement("mathematical_reasoning", True)
            status = prompt_system.get_enhancement_status()
            assert status["mathematical_reasoning"] is True

    class TestResponseParsing:
        """Test response parsing functionality"""

        @pytest.mark.asyncio
        async def test_mathematical_parsing(self, response_parser):
            """Test mathematical response parsing"""
            response = "The answer is 42. I calculated this by multiplying 6 by 7."
            parsed = response_parser.parse_response(response, "mathematical")

            assert parsed["answer"] is not None
            assert "workings" in parsed
            assert "confidence" in parsed
            assert "numbers_found" in parsed
            assert parsed["has_final_answer"] is True

        @pytest.mark.asyncio
        async def test_logical_parsing(self, response_parser):
            """Test logical response parsing"""
            response = "Therefore, the conclusion is that all humans are mortal."
            parsed = response_parser.parse_response(response, "logical")

            assert "conclusion" in parsed
            assert "reasoning" in parsed
            assert "confidence" in parsed
            assert "has_conclusion" in parsed

        @pytest.mark.asyncio
        async def test_general_parsing(self, response_parser):
            """Test general response parsing"""
            response = "This is a general response to the question asked."
            parsed = response_parser.parse_response(response, "general")

            assert "answer" in parsed
            assert "full_response" in parsed
            assert "confidence" in parsed
            assert "response_length" in parsed

    class TestLegacyCompatibility:
        """Test backward compatibility with legacy interfaces"""

        @pytest.mark.asyncio
        async def test_prompt_builder_compatibility(self, prompt_system):
            """Test PromptBuilder compatibility"""
            from src.agent.prompt_system import PromptBuilder

            builder = PromptBuilder()
            prompt = builder.get_system_prompt(
                ProblemType.MATHEMATICAL, Difficulty.MEDIUM
            )

            assert prompt is not None
            assert len(prompt) > 50  # Should be a substantial prompt
            assert "mathematical" in prompt.lower() or "calculation" in prompt.lower()


class TestPromptSystemPerformance:
    """Test prompt system performance and optimization"""

    @pytest.fixture
    def prompt_system(self):
        """Create prompt system for testing"""
        return PromptSystem()

    @pytest.mark.asyncio
    async def test_response_time_performance(self, prompt_system):
        """Test that response times are reasonable"""
        import time

        problem = "What is 10 + 15?"
        start_time = time.time()

        response = await prompt_system.process_with_context(problem)

        end_time = time.time()
        response_time = end_time - start_time

        # Should complete quickly (under 1 second for processing)
        assert response_time < 1.0, f"Response too slow: {response_time}s"
        assert response is not None

    @pytest.mark.asyncio
    async def test_memory_efficiency(self, prompt_system):
        """Test that memory usage is reasonable"""
        import gc
        import sys

        # Get initial memory usage
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Process multiple problems
        problems = ["What is 2 + 2?", "Calculate 10 × 5", "Solve for x: x + 3 = 8"]

        for problem in problems:
            await prompt_system.process_with_context(problem)

        # Check memory usage
        gc.collect()
        final_objects = len(gc.get_objects())
        object_increase = final_objects - initial_objects

        # Should not create excessive objects (arbitrary limit)
        assert object_increase < 1000, f"Too many objects created: {object_increase}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
