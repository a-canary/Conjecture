#!/usr/bin/env python3
"""
Comprehensive tests for the enhanced GLM-4.6 judge system
Tests advanced evaluation methodology and domain-specific criteria
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from benchmarks.benchmarking.enhanced_glm46_judge import (
    EnhancedGLM46Judge,
    JudgeEvaluation,
)


class TestEnhancedGLM46Judge:
    """Test the enhanced GLM-4.6 judge system"""

    @pytest.fixture
    def judge_config(self):
        """Create judge configuration for testing"""
        return {"key": "test_api_key", "url": "http://test-api.com", "model": "glm-4.6"}

    @pytest.fixture
    def sample_evaluation_response(self):
        """Sample GLM-4.6 evaluation response"""
        return """{
            "is_correct": true,
            "confidence": 85,
            "reasoning_quality": "good",
            "problem_type_match": true,
            "enhancement_usage": "moderate",
            "feedback": "The response correctly calculates the answer and shows good mathematical reasoning.",
            "detailed_scores": {
                "correctness": 90,
                "methodology": 80,
                "clarity": 85,
                "completeness": 88,
                "enhancement_usage": 75
            }
        }"""

    class TestJudgeInitialization:
        """Test judge initialization and setup"""

        def test_successful_initialization(self, judge_config):
            """Test successful judge initialization"""
            judge = EnhancedGLM46Judge(judge_config)
            assert judge.api_key == "test_api_key"
            assert judge.base_url == "http://test-api.com"
            assert judge.model == "glm-4.6"
            assert judge.evaluation_cache == {}

        def test_incomplete_config_raises_error(self):
            """Test that incomplete configuration raises error"""
            with pytest.raises(ValueError):
                EnhancedGLM46Judge({"key": "incomplete"})

        def test_empty_config_raises_error(self):
            """Test that empty configuration raises error"""
            with pytest.raises(ValueError):
                EnhancedGLM46Judge({})

    class TestEnhancedEvaluationPrompts:
        """Test enhanced evaluation prompt generation"""

        def test_mathematical_evaluation_prompt(self, judge_config):
            """Test mathematical evaluation prompt generation"""
            judge = EnhancedGLM46Judge(judge_config)

            prompt = judge._create_enhanced_judge_prompt(
                problem="What is 15 × 12?",
                response="The answer is 180. I calculated 15 × 12 = 180.",
                expected="180",
                problem_type="mathematical",
                difficulty="medium",
            )

            # Should contain mathematical-specific criteria
            assert "MATHEMATICAL EVALUATION CRITERIA" in prompt
            assert "calculation" in prompt
            assert "methodology" in prompt
            assert "step-by-step reasoning" in prompt

        def test_logical_evaluation_prompt(self, judge_config):
            """Test logical evaluation prompt generation"""
            judge = EnhancedGLM46Judge(judge_config)

            prompt = judge._create_enhanced_judge_prompt(
                problem="If all A are B and some B are C, what follows?",
                response="Since all A are B and some B are C, some A might be C.",
                expected="Some A might be C",
                problem_type="logical",
                difficulty="medium",
            )

            # Should contain logical-specific criteria
            assert "LOGICAL EVALUATION CRITERIA" in prompt
            assert "logical structure" in prompt
            assert "reasoning process" in prompt

        def test_difficulty_specific_guidance(self, judge_config):
            """Test difficulty-specific guidance in prompts"""
            judge = EnhancedGLM46Judge(judge_config)

            easy_prompt = judge._create_enhanced_judge_prompt(
                problem="What is 2 + 2?",
                response="4",
                expected="4",
                problem_type="mathematical",
                difficulty="easy",
            )

            hard_prompt = judge._create_enhanced_judge_prompt(
                problem="Prove the fundamental theorem of calculus",
                response="Complex proof...",
                expected="Complex proof...",
                problem_type="mathematical",
                difficulty="hard",
            )

            assert "EASY" in easy_prompt
            assert "HARD" in hard_prompt

    class TestEvaluationParsing:
        """Test evaluation response parsing"""

        def test_json_parsing_success(self, judge_config, sample_evaluation_response):
            """Test successful JSON parsing"""
            judge = EnhancedGLM46Judge(judge_config)
            from datetime import datetime

            start_time = datetime.now()

            evaluation = judge._parse_evaluation_response(
                sample_evaluation_response, "mathematical", start_time
            )

            assert isinstance(evaluation, JudgeEvaluation)
            assert evaluation.is_correct is True
            assert evaluation.confidence == 85
            assert evaluation.reasoning_quality == "good"
            assert evaluation.problem_type_match is True
            assert evaluation.enhancement_usage == "moderate"
            assert evaluation.detailed_scores["correctness"] == 90

        def test_text_parsing_fallback(self, judge_config):
            """Test text parsing fallback for non-JSON responses"""
            judge = EnhancedGLM46Judge(judge_config)
            from datetime import datetime

            start_time = datetime.now()

            text_response = "The answer is CORRECT and the reasoning is EXCELLENT."
            evaluation = judge._parse_evaluation_response(
                text_response, "general", start_time
            )

            assert isinstance(evaluation, JudgeEvaluation)
            assert evaluation.is_correct is True
            assert evaluation.reasoning_quality == "fair"  # fallback default
            assert evaluation.problem_type_match is True

        def test_malformed_json_handling(self, judge_config):
            """Test handling of malformed JSON"""
            judge = EnhancedGLM46Judge(judge_config)
            from datetime import datetime

            start_time = datetime.now()

            malformed_response = '{"is_correct": true, "confidence": invalid}'
            evaluation = judge._parse_evaluation_response(
                malformed_response, "general", start_time
            )

            # Should fall back gracefully
            assert isinstance(evaluation, JudgeEvaluation)

    class TestFallbackEvaluation:
        """Test fallback evaluation functionality"""

        def test_fallback_evaluation_correct(self, judge_config):
            """Test fallback evaluation for correct responses"""
            judge = EnhancedGLM46Judge(judge_config)
            from datetime import datetime

            start_time = datetime.now()

            evaluation = judge._fallback_evaluation(
                problem="What is 2+2?",
                response="The answer is 4",
                expected="4",
                start_time=start_time,
            )

            assert isinstance(evaluation, JudgeEvaluation)
            assert evaluation.is_correct is True
            assert evaluation.confidence >= 60  # Should be higher for correct
            assert evaluation.feedback == "Fallback evaluation - GLM-4.6 unavailable"

        def test_fallback_evaluation_incorrect(self, judge_config):
            """Test fallback evaluation for incorrect responses"""
            judge = EnhancedGLM46Judge(judge_config)
            from datetime import datetime

            start_time = datetime.now()

            evaluation = judge._fallback_evaluation(
                problem="What is 2+2?",
                response="The answer is 5",
                expected="4",
                start_time=start_time,
            )

            assert isinstance(evaluation, JudgeEvaluation)
            assert evaluation.is_correct is False
            assert evaluation.confidence <= 40  # Should be lower for incorrect

    class TestCachingFunctionality:
        """Test evaluation caching"""

        @pytest.mark.asyncio
        async def test_caching_mechanism(self, judge_config):
            """Test that evaluations are cached"""
            judge = EnhancedGLM46Judge(judge_config)

            # Mock the API call
            with patch.object(judge, "_call_glm46_judge") as mock_call:
                mock_call.return_value = '{"is_correct": true, "confidence": 80}'

                # First evaluation
                eval1 = await judge.evaluate_response(
                    "What is 2+2?", "4", "4", "mathematical"
                )

                # Second evaluation with same inputs should use cache
                eval2 = await judge.evaluate_response(
                    "What is 2+2?", "4", "4", "mathematical"
                )

                # API should only be called once due to caching
                assert mock_call.call_count == 1
                assert eval1.is_correct == eval2.is_correct
                assert eval1.confidence == eval2.confidence

    class TestEvaluationSummary:
        """Test evaluation summary functionality"""

        def test_empty_evaluations_summary(self, judge_config):
            """Test summary with empty evaluations list"""
            judge = EnhancedGLM46Judge(judge_config)
            summary = judge.get_evaluation_summary([])

            assert "error" in summary
            assert summary["error"] == "No evaluations provided"

        def test_evaluation_summary_statistics(self, judge_config):
            """Test evaluation summary statistics"""
            judge = EnhancedGLM46Judge(judge_config)

            # Create sample evaluations
            evaluations = [
                JudgeEvaluation(
                    is_correct=True,
                    confidence=90,
                    reasoning_quality="excellent",
                    problem_type_match=True,
                    enhancement_usage="extensive",
                    feedback="Great response",
                    detailed_scores={
                        "correctness": 95,
                        "methodology": 90,
                        "clarity": 88,
                        "completeness": 92,
                        "enhancement_usage": 85,
                    },
                    evaluation_time=1.5,
                ),
                JudgeEvaluation(
                    is_correct=False,
                    confidence=30,
                    reasoning_quality="fair",
                    problem_type_match=True,
                    enhancement_usage="minimal",
                    feedback="Needs improvement",
                    detailed_scores={
                        "correctness": 40,
                        "methodology": 50,
                        "clarity": 45,
                        "completeness": 35,
                        "enhancement_usage": 25,
                    },
                    evaluation_time=1.2,
                ),
                JudgeEvaluation(
                    is_correct=True,
                    confidence=75,
                    reasoning_quality="good",
                    problem_type_match=True,
                    enhancement_usage="moderate",
                    feedback="Good response",
                    detailed_scores={
                        "correctness": 80,
                        "methodology": 75,
                        "clarity": 78,
                        "completeness": 72,
                        "enhancement_usage": 60,
                    },
                    evaluation_time=1.3,
                ),
            ]

            summary = judge.get_evaluation_summary(evaluations)

            assert summary["total_evaluations"] == 3
            assert summary["correct_count"] == 2
            assert summary["accuracy"] == 2 / 3
            assert summary["average_confidence"] == (90 + 30 + 75) / 3
            assert "average_scores" in summary
            assert "reasoning_quality_distribution" in summary

    class TestBenchmarkComparison:
        """Test benchmark comparison functionality"""

        @pytest.mark.asyncio
        async def test_enhanced_vs_standard_benchmark(self, judge_config):
            """Test benchmark comparison between enhanced and standard evaluation"""
            judge = EnhancedGLM46Judge(judge_config)

            # Mock API responses for different evaluation types
            with patch.object(judge, "_call_glm46_judge") as mock_call:
                # Enhanced evaluation response
                enhanced_response = """{
                    "is_correct": true,
                    "confidence": 85,
                    "reasoning_quality": "good",
                    "problem_type_match": true,
                    "enhancement_usage": "moderate",
                    "feedback": "Good mathematical reasoning",
                    "detailed_scores": {"correctness": 90, "methodology": 80, "clarity": 85, "completeness": 88, "enhancement_usage": 75}
                }"""
                # Standard evaluation response
                standard_response = "CORRECT"

                # Configure mock to return different responses based on prompt content
                def side_effect(prompt):
                    if "ENHANCED EVALUATION TASK" in prompt:
                        return enhanced_response
                    else:
                        return standard_response

                mock_call.side_effect = side_effect

                test_problems = [
                    {
                        "problem": "What is 15 × 12?",
                        "response": "180",
                        "expected": "180",
                        "type": "mathematical",
                    }
                ]

                results = await judge.benchmark_enhanced_vs_standard(test_problems)

                assert "enhanced_evaluations" in results
                assert "standard_evaluations" in results
                assert "comparison" in results
                assert len(results["enhanced_evaluations"]) == 1
                assert len(results["standard_evaluations"]) == 1

                # Enhanced evaluation should have more detailed information
                enhanced_eval = results["enhanced_evaluations"][0]
                standard_eval = results["standard_evaluations"][0]

                assert enhanced_eval.detailed_scores is not None
                assert len(enhanced_eval.detailed_scores) > 1
                assert standard_eval.detailed_scores == {
                    "correctness": 80
                }  # Standard fallback

    class TestErrorHandling:
        """Test error handling and robustness"""

        @pytest.mark.asyncio
        async def test_api_error_handling(self, judge_config):
            """Test handling of API errors"""
            judge = EnhancedGLM46Judge(judge_config)

            # Mock API error
            with patch.object(judge, "_call_glm46_judge") as mock_call:
                mock_call.side_effect = Exception("API Error")

                evaluation = await judge.evaluate_response(
                    "Test problem", "Test response", "Test expected", "general"
                )

                # Should fall back to evaluation
                assert isinstance(evaluation, JudgeEvaluation)
                assert (
                    evaluation.feedback == "Fallback evaluation - GLM-4.6 unavailable"
                )

        @pytest.mark.asyncio
        async def test_timeout_handling(self, judge_config):
            """Test handling of timeout scenarios"""
            judge = EnhancedGLM46Judge(judge_config)

            # Mock timeout
            with patch("aiohttp.ClientSession.post") as mock_post:
                mock_post.side_effect = asyncio.TimeoutError("Request timeout")

                evaluation = await judge.evaluate_response(
                    "Test problem", "Test response", "Test expected", "general"
                )

                # Should fall back gracefully
                assert isinstance(evaluation, JudgeEvaluation)
                assert evaluation.is_correct is False  # Conservative fallback

        @pytest.mark.asyncio
        async def test_malformed_response_handling(self, judge_config):
            """Test handling of malformed API responses"""
            judge = EnhancedGLM46Judge(judge_config)

            # Mock malformed response
            with patch.object(judge, "_call_glm46_judge") as mock_call:
                mock_call.return_value = "This is not valid JSON {"

                evaluation = await judge.evaluate_response(
                    "Test problem", "Test response", "Test expected", "general"
                )

                # Should handle gracefully
                assert isinstance(evaluation, JudgeEvaluation)
                assert evaluation.reasoning_quality == "fair"  # Fallback default


class TestJudgePerformance:
    """Test judge performance and optimization"""

    @pytest.mark.asyncio
    async def test_evaluation_performance(self, judge_config):
        """Test that evaluation completes in reasonable time"""
        import time

        judge = EnhancedGLM46Judge(judge_config)

        # Mock fast API response
        with patch.object(judge, "_call_glm46_judge") as mock_call:
            mock_call.return_value = '{"is_correct": true, "confidence": 80}'

            start_time = time.time()
            evaluation = await judge.evaluate_response(
                "Test problem", "Test response", "Test expected", "general"
            )
            end_time = time.time()

            # Should complete quickly
            assert end_time - start_time < 2.0, (
                f"Evaluation too slow: {end_time - start_time}s"
            )
            assert isinstance(evaluation, JudgeEvaluation)

    @pytest.mark.asyncio
    async def test_batch_evaluation_performance(self, judge_config):
        """Test batch evaluation performance"""
        import time

        judge = EnhancedGLM46Judge(judge_config)

        # Mock API response
        with patch.object(judge, "_call_glm46_judge") as mock_call:
            mock_call.return_value = '{"is_correct": true, "confidence": 80}'

            # Create multiple test cases
            test_cases = [
                ("Problem 1", "Response 1", "Expected 1", "mathematical"),
                ("Problem 2", "Response 2", "Expected 2", "logical"),
                ("Problem 3", "Response 3", "Expected 3", "general"),
            ]

            start_time = time.time()
            evaluations = []
            for problem, response, expected, ptype in test_cases:
                eval_result = await judge.evaluate_response(
                    problem, response, expected, ptype
                )
                evaluations.append(eval_result)
            end_time = time.time()

            # Should handle multiple evaluations efficiently
            assert end_time - start_time < 5.0, (
                f"Batch evaluation too slow: {end_time - start_time}s"
            )
            assert len(evaluations) == 3
            assert all(isinstance(e, JudgeEvaluation) for e in evaluations)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
