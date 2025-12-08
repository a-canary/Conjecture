"""
Tests for Conjecture providers integration with DeepEval
Tests the custom LLM wrappers and provider-specific functionality
"""

import asyncio
import pytest
import sys
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.evaluation.conjecture_llm_wrapper import (
        ConjectureLLMWrapper,
        GraniteTinyWrapper,
        GLM46Wrapper,
        GptOss20bWrapper,
        create_conjecture_wrapper
    )
    from src.evaluation.evaluation_framework import (
        EvaluationFramework,
        TestCase,
        EvaluationResult
    )
    EVALUATION_FRAMEWORK_AVAILABLE = True
except ImportError:
    EVALUATION_FRAMEWORK_AVAILABLE = False


class TestConjectureLLMWrapper:
    """Test the base Conjecture LLM wrapper"""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        return {
            "providers": [
                {
                    "name": "test-provider",
                    "url": "http://test.com",
                    "api": "test-key",
                    "model": "test-model"
                }
            ]
        }

    @pytest.mark.skipif(not EVALUATION_FRAMEWORK_AVAILABLE, reason="Evaluation framework not available")
    def test_base_wrapper_initialization(self, mock_config):
        """Test base wrapper initialization"""
        wrapper = ConjectureLLMWrapper("test-model", use_conjecture=False)
        assert wrapper.model_name == "test-model"
        assert wrapper.use_conjecture is False
        assert wrapper.model is None  # Not loaded yet

    @pytest.mark.skipif(not EVALUATION_FRAMEWORK_AVAILABLE, reason="Evaluation framework not available")
    def test_base_wrapper_with_conjecture(self, mock_config):
        """Test base wrapper with Conjecture enhancement"""
        wrapper = ConjectureLLMWrapper("test-model", use_conjecture=True)
        assert wrapper.use_conjecture is True

    @pytest.mark.skipif(not EVALUATION_FRAMEWORK_AVAILABLE, reason="Evaluation framework not available")
    @pytest.mark.asyncio
    async def test_base_wrapper_load_model(self):
        """Test model loading in base wrapper"""
        wrapper = ConjectureLLMWrapper("test-model")
        
        # Mock the model integration
        with patch('src.evaluation.conjecture_llm_wrapper.ModelIntegration') as mock_integration_class:
            mock_integration = AsyncMock()
            mock_integration_class.return_value = mock_integration
            
            await wrapper.load_model()
            
            assert wrapper.model is not None
            mock_integration_class.assert_called_once()

    @pytest.mark.skipif(not EVALUATION_FRAMEWORK_AVAILABLE, reason="Evaluation framework not available")
    @pytest.mark.asyncio
    async def test_base_wrapper_a_generate(self):
        """Test async generation in base wrapper"""
        wrapper = ConjectureLLMWrapper("test-model")
        
        # Mock model and integration
        mock_model = AsyncMock()
        mock_model.get_model_response = AsyncMock(return_value="Test response")
        wrapper.model = mock_model
        
        response = await wrapper.a_generate("Test prompt")
        assert response == "Test response"
        mock_model.get_model_response.assert_called_once_with("test-model", "Test prompt")

    @pytest.mark.skipif(not EVALUATION_FRAMEWORK_AVAILABLE, reason="Evaluation framework not available")
    @pytest.mark.asyncio
    async def test_base_wrapper_a_generate_with_conjecture(self):
        """Test async generation with Conjecture enhancement"""
        wrapper = ConjectureLLMWrapper("test-model", use_conjecture=True)
        
        # Mock model and integration
        mock_model = AsyncMock()
        mock_model.get_conjecture_enhanced_response = AsyncMock(return_value="Enhanced response")
        wrapper.model = mock_model
        
        response = await wrapper.a_generate("Test prompt")
        assert response == "Enhanced response"
        mock_model.get_conjecture_enhanced_response.assert_called_once_with("test-model", "Test prompt")

    @pytest.mark.skipif(not EVALUATION_FRAMEWORK_AVAILABLE, reason="Evaluation framework not available")
    @pytest.mark.asyncio
    async def test_base_wrapper_a_generate_model_not_loaded(self):
        """Test generation when model is not loaded"""
        wrapper = ConjectureLLMWrapper("test-model")
        
        # Should auto-load the model
        with patch.object(wrapper, 'load_model', new_callable=AsyncMock) as mock_load:
            mock_model = AsyncMock()
            mock_model.get_model_response = AsyncMock(return_value="Auto-loaded response")
            wrapper.model = mock_model
            
            response = await wrapper.a_generate("Test prompt")
            assert response == "Auto-loaded response"
            mock_load.assert_called_once()


@pytest.mark.skipif(not EVALUATION_FRAMEWORK_AVAILABLE, reason="Evaluation framework not available")
class TestGraniteTinyWrapper:
    """Test the Granite Tiny wrapper"""

    def test_granite_tiny_wrapper_initialization(self):
        """Test Granite Tiny wrapper initialization"""
        wrapper = GraniteTinyWrapper(use_conjecture=False)
        assert wrapper.model_name == "ibm/granite-4-h-tiny"
        assert wrapper.use_conjecture is False

    def test_granite_tiny_wrapper_with_conjecture(self):
        """Test Granite Tiny wrapper with Conjecture"""
        wrapper = GraniteTinyWrapper(use_conjecture=True)
        assert wrapper.use_conjecture is True

    @pytest.mark.asyncio
    async def test_granite_tiny_wrapper_generation(self):
        """Test Granite Tiny wrapper generation"""
        wrapper = GraniteTinyWrapper()
        
        # Mock model
        mock_model = AsyncMock()
        mock_model.get_model_response = AsyncMock(return_value="Granite Tiny response")
        wrapper.model = mock_model
        
        response = await wrapper.a_generate("Test prompt")
        assert response == "Granite Tiny response"
        mock_model.get_model_response.assert_called_once_with("granite-tiny", "Test prompt")


@pytest.mark.skipif(not EVALUATION_FRAMEWORK_AVAILABLE, reason="Evaluation framework not available")
class TestGLM46Wrapper:
    """Test the GLM-4.6 wrapper"""

    def test_glm46_wrapper_initialization(self):
        """Test GLM-4.6 wrapper initialization"""
        wrapper = GLM46Wrapper(use_conjecture=False)
        assert wrapper.model_name == "zai/GLM-4.6"
        assert wrapper.use_conjecture is False

    def test_glm46_wrapper_with_conjecture(self):
        """Test GLM-4.6 wrapper with Conjecture"""
        wrapper = GLM46Wrapper(use_conjecture=True)
        assert wrapper.use_conjecture is True

    @pytest.mark.asyncio
    async def test_glm46_wrapper_generation(self):
        """Test GLM-4.6 wrapper generation"""
        wrapper = GLM46Wrapper()
        
        # Mock model
        mock_model = AsyncMock()
        mock_model.get_model_response = AsyncMock(return_value="GLM-4.6 response")
        wrapper.model = mock_model
        
        response = await wrapper.a_generate("Test prompt")
        assert response == "GLM-4.6 response"
        mock_model.get_model_response.assert_called_once_with("glm-4.6", "Test prompt")


@pytest.mark.skipif(not EVALUATION_FRAMEWORK_AVAILABLE, reason="Evaluation framework not available")
class TestGptOss20bWrapper:
    """Test the GPT-OSS-20B wrapper"""

    def test_gpt_oss20b_wrapper_initialization(self):
        """Test GPT-OSS-20B wrapper initialization"""
        wrapper = GptOss20bWrapper(use_conjecture=False)
        assert wrapper.model_name == "openrouter/gpt-oss-20b"
        assert wrapper.use_conjecture is False

    def test_gpt_oss20b_wrapper_with_conjecture(self):
        """Test GPT-OSS-20B wrapper with Conjecture"""
        wrapper = GptOss20bWrapper(use_conjecture=True)
        assert wrapper.use_conjecture is True

    @pytest.mark.asyncio
    async def test_gpt_oss20b_wrapper_generation(self):
        """Test GPT-OSS-20B wrapper generation"""
        wrapper = GptOss20bWrapper()
        
        # Mock model
        mock_model = AsyncMock()
        mock_model.get_model_response = AsyncMock(return_value="GPT-OSS-20B response")
        wrapper.model = mock_model
        
        response = await wrapper.a_generate("Test prompt")
        assert response == "GPT-OSS-20B response"
        mock_model.get_model_response.assert_called_once_with("gpt-oss-20b", "Test prompt")


@pytest.mark.skipif(not EVALUATION_FRAMEWORK_AVAILABLE, reason="Evaluation framework not available")
class TestWrapperFactory:
    """Test the wrapper factory functions"""

    def test_create_granite_tiny_wrapper(self):
        """Test creating Granite Tiny wrapper"""
        wrapper = create_conjecture_wrapper("ibm/granite-4-h-tiny", use_conjecture=False)
        assert isinstance(wrapper, GraniteTinyWrapper)
        assert wrapper.use_conjecture is False

    def test_create_granite_tiny_wrapper_with_conjecture(self):
        """Test creating Granite Tiny wrapper with Conjecture"""
        wrapper = create_conjecture_wrapper("ibm/granite-4-h-tiny", use_conjecture=True)
        assert isinstance(wrapper, GraniteTinyWrapper)
        assert wrapper.use_conjecture is True

    def test_create_glm46_wrapper(self):
        """Test creating GLM-4.6 wrapper"""
        wrapper = create_conjecture_wrapper("zai/GLM-4.6", use_conjecture=False)
        assert isinstance(wrapper, GLM46Wrapper)
        assert wrapper.use_conjecture is False

    def test_create_gpt_oss20b_wrapper(self):
        """Test creating GPT-OSS-20B wrapper"""
        wrapper = create_conjecture_wrapper("openrouter/gpt-oss-20b", use_conjecture=False)
        assert isinstance(wrapper, GptOss20bWrapper)
        assert wrapper.use_conjecture is True

    def test_create_wrapper_unknown_model(self):
        """Test creating wrapper for unknown model"""
        wrapper = create_conjecture_wrapper("unknown/model", use_conjecture=False)
        assert isinstance(wrapper, ConjectureLLMWrapper)
        assert wrapper.model_name == "unknown/model"


@pytest.mark.skipif(not EVALUATION_FRAMEWORK_AVAILABLE, reason="Evaluation framework not available")
class TestEvaluationFramework:
    """Test the evaluation framework"""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        return {
            "providers": [
                {
                    "name": "test-granite",
                    "url": "http://localhost:11434",
                    "model": "ibm/granite-4-h-tiny"
                },
                {
                    "name": "test-glm",
                    "url": "https://api.test.com",
                    "api": "test-key",
                    "model": "zai/GLM-4.6"
                },
                {
                    "name": "test-gpt",
                    "url": "https://openrouter.ai",
                    "api": "test-key",
                    "model": "openrouter/gpt-oss-20b"
                }
            ]
        }

    @pytest.fixture
    def evaluation_framework(self, mock_config):
        """Create evaluation framework instance"""
        return EvaluationFramework(mock_config)

    def test_framework_initialization(self, evaluation_framework):
        """Test framework initialization"""
        assert evaluation_framework is not None
        assert hasattr(evaluation_framework, 'metrics')
        assert len(evaluation_framework.metrics) > 0

    def test_create_test_case(self, evaluation_framework):
        """Test test case creation"""
        test_case = evaluation_framework.create_test_case(
            input_text="What is 2+2?",
            expected_output="4",
            actual_output="The answer is 4.",
            additional_metadata={"category": "mathematics"}
        )
        
        assert isinstance(test_case, TestCase)
        assert test_case.input == "What is 2+2?"
        assert test_case.expected_output == "4"
        assert test_case.actual_output == "The answer is 4."
        assert test_case.additional_metadata["category"] == "mathematics"

    def test_create_test_cases_from_tasks(self, evaluation_framework):
        """Test creating test cases from task dictionaries"""
        tasks = [
            {
                "prompt": "What is the capital of France?",
                "expected_answer": "Paris",
                "model_response": "The capital of France is Paris.",
                "metadata": {"category": "geography"}
            },
            {
                "prompt": "Solve: 5+3",
                "expected_answer": "8",
                "model_response": "5+3 = 8",
                "metadata": {"category": "mathematics"}
            }
        ]
        
        test_cases = evaluation_framework.create_test_cases_from_tasks(tasks)
        
        assert len(test_cases) == 2
        assert all(isinstance(tc, TestCase) for tc in test_cases)
        assert test_cases[0].input == "What is the capital of France?"
        assert test_cases[1].input == "Solve: 5+3"

    @pytest.mark.asyncio
    async def test_evaluate_provider(self, evaluation_framework):
        """Test provider evaluation"""
        test_case = evaluation_framework.create_test_case(
            input_text="Test question",
            expected_output="Test answer",
            actual_output="Test response"
        )
        
        # Mock the wrapper
        with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
            mock_wrapper = AsyncMock()
            mock_wrapper.a_generate = AsyncMock(return_value="Mocked response")
            mock_create.return_value = mock_wrapper
            
            result = await evaluation_framework.evaluate_provider(
                "ibm/granite-4-h-tiny", [test_case], use_conjecture=False
            )
            
            assert isinstance(result, EvaluationResult)
            assert result.provider_name == "ibm/granite-4-h-tiny_direct"
            assert result.success is True
            assert result.overall_score >= 0

    @pytest.mark.asyncio
    async def test_evaluate_provider_with_conjecture(self, evaluation_framework):
        """Test provider evaluation with Conjecture enhancement"""
        test_case = evaluation_framework.create_test_case(
            input_text="Test question",
            expected_output="Test answer",
            actual_output="Test response"
        )
        
        # Mock the wrapper
        with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
            mock_wrapper = AsyncMock()
            mock_wrapper.a_generate = AsyncMock(return_value="Enhanced response")
            mock_create.return_value = mock_wrapper
            
            result = await evaluation_framework.evaluate_provider(
                "zai/GLM-4.6", [test_case], use_conjecture=True
            )
            
            assert result.provider_name == "zai/GLM-4.6_conjecture"
            assert result.success is True
            mock_create.assert_called_once_with("zai/GLM-4.6", use_conjecture=True)

    @pytest.mark.asyncio
    async def test_evaluate_multiple_providers(self, evaluation_framework):
        """Test multiple provider evaluation"""
        test_cases = [
            evaluation_framework.create_test_case(
                input_text="Question 1",
                expected_output="Answer 1",
                actual_output="Response 1"
            ),
            evaluation_framework.create_test_case(
                input_text="Question 2",
                expected_output="Answer 2",
                actual_output="Response 2"
            )
        ]
        
        providers = ["ibm/granite-4-h-tiny", "zai/GLM-4.6"]
        
        # Mock the wrapper
        with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
            mock_wrapper = AsyncMock()
            mock_wrapper.a_generate = AsyncMock(return_value="Mocked response")
            mock_create.return_value = mock_wrapper
            
            result = await evaluation_framework.evaluate_multiple_providers(
                providers, test_cases, compare_conjecture=True
            )
            
            assert "providers" in result
            assert "ibm/granite-4-h-tiny_direct" in result["providers"]
            assert "ibm/granite-4-h-tiny_conjecture" in result["providers"]
            assert "zai/GLM-4.6_direct" in result["providers"]
            assert "zai/GLM-4.6_conjecture" in result["providers"]
            assert "comparison" in result

    def test_generate_summary(self, evaluation_framework):
        """Test summary generation"""
        # Create mock results
        results = {
            "providers": {
                "ibm/granite-4-h-tiny_direct": {
                    "overall_score": 0.7,
                    "success": True,
                    "metrics_results": {"relevancy": {"score": 0.8}}
                },
                "ibm/granite-4-h-tiny_conjecture": {
                    "overall_score": 0.8,
                    "success": True,
                    "metrics_results": {"relevancy": {"score": 0.9}}
                }
            },
            "comparison": {
                "ibm/granite-4-h-tiny": {
                    "improvement": 14.3,
                    "direct_score": 0.7,
                    "conjecture_score": 0.8
                }
            }
        }
        
        summary = evaluation_framework.generate_summary(results)
        
        assert "EVALUATION SUMMARY" in summary
        assert "ibm/granite-4-h-tiny" in summary
        assert "14.3%" in summary
        assert "IMPROVEMENT ANALYSIS" in summary


class TestProviderSpecificFunctionality:
    """Test provider-specific functionality and edge cases"""

    @pytest.mark.skipif(not EVALUATION_FRAMEWORK_AVAILABLE, reason="Evaluation framework not available")
    @pytest.mark.asyncio
    async def test_granite_tiny_mathematics_evaluation(self):
        """Test Granite Tiny on mathematics tasks"""
        framework = EvaluationFramework()
        
        test_case = framework.create_test_case(
            input_text="Solve: √25 = ?",
            expected_output="5",
            actual_output="The square root of 25 is 5. This can be verified by calculating 5² = 25.",
            additional_metadata={"category": "mathematics", "difficulty": "easy"}
        )
        
        # Mock wrapper
        with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
            mock_wrapper = AsyncMock()
            mock_wrapper.a_generate = AsyncMock(return_value="The square root of 25 is 5.")
            mock_create.return_value = mock_wrapper
            
            result = await framework.evaluate_provider(
                "ibm/granite-4-h-tiny", [test_case], use_conjecture=False
            )
            
            assert result.success is True
            assert result.overall_score > 0.5  # Should score well on mathematics

    @pytest.mark.skipif(not EVALUATION_FRAMEWORK_AVAILABLE, reason="Evaluation framework not available")
    @pytest.mark.asyncio
    async def test_glm46_algorithm_evaluation(self):
        """Test GLM-4.6 on algorithm tasks"""
        framework = EvaluationFramework()
        
        test_case = framework.create_test_case(
            input_text="Write a function to find the maximum element in a list",
            expected_output="def find_max(lst): return max(lst)",
            actual_output="def find_max(lst):\n    if not lst:\n        return None\n    max_val = lst[0]\n    for item in lst[1:]:\n        if item > max_val:\n            max_val = item\n    return max_val",
            additional_metadata={"category": "algorithms", "complexity": "O(n)"}
        )
        
        # Mock wrapper
        with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
            mock_wrapper = AsyncMock()
            mock_wrapper.a_generate = AsyncMock(return_value="def find_max(lst): return max(lst)")
            mock_create.return_value = mock_wrapper
            
            result = await framework.evaluate_provider(
                "zai/GLM-4.6", [test_case], use_conjecture=False
            )
            
            assert result.success is True
            assert result.overall_score > 0.5  # Should score well on algorithms

    @pytest.mark.skipif(not EVALUATION_FRAMEWORK_AVAILABLE, reason="Evaluation framework not available")
    @pytest.mark.asyncio
    async def test_gpt_oss20b_creativity_evaluation(self):
        """Test GPT-OSS-20B on creativity tasks"""
        framework = EvaluationFramework()
        
        test_case = framework.create_test_case(
            input_text="Write a short story about a robot discovering music",
            expected_output="A creative story with musical elements",
            actual_output="In a world of silence, robot X-7 discovered an old vinyl record. As the needle touched the groove, sounds emerged that transformed its existence...",
            additional_metadata={"category": "creativity", "style": "narrative"}
        )
        
        # Mock wrapper
        with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
            mock_wrapper = AsyncMock()
            mock_wrapper.a_generate = AsyncMock(return_value="A creative story about a robot and music...")
            mock_create.return_value = mock_wrapper
            
            result = await framework.evaluate_provider(
                "openrouter/gpt-oss-20b", [test_case], use_conjecture=False
            )
            
            assert result.success is True
            assert result.overall_score > 0.5  # Should score well on creativity

    @pytest.mark.skipif(not EVALUATION_FRAMEWORK_AVAILABLE, reason="Evaluation framework not available")
    @pytest.mark.asyncio
    async def test_conjecture_enhancement_comparison(self):
        """Test Conjecture enhancement across all providers"""
        framework = EvaluationFramework()
        
        test_case = framework.create_test_case(
            input_text="Explain quantum computing in simple terms",
            expected_output="A simple explanation of quantum computing",
            actual_output="Quantum computing uses quantum bits that can be in multiple states simultaneously...",
            additional_metadata={"category": "science", "complexity": "high"}
        )
        
        providers = ["ibm/granite-4-h-tiny", "zai/GLM-4.6", "openrouter/gpt-oss-20b"]
        
        # Mock wrappers
        with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
            mock_wrapper = AsyncMock()
            # Simulate better performance with Conjecture
            mock_wrapper.a_generate.side_effect = [
                "Basic explanation",  # Direct response
                "Enhanced explanation with Conjecture",  # Conjecture response
                "Basic explanation",  # Direct response
                "Enhanced explanation with Conjecture",  # Conjecture response
                "Basic explanation",  # Direct response
                "Enhanced explanation with Conjecture",  # Conjecture response
            ]
            mock_create.return_value = mock_wrapper
            
            result = await framework.evaluate_multiple_providers(
                providers, [test_case], compare_conjecture=True
            )
            
            # Check that all providers were evaluated both direct and with Conjecture
            assert len(result["providers"]) == 6  # 3 providers × 2 modes
            assert "comparison" in result
            
            # Check that Conjecture scores are generally better
            for provider in providers:
                direct_key = f"{provider}_direct"
                conjecture_key = f"{provider}_conjecture"
                
                direct_score = result["providers"][direct_key]["overall_score"]
                conjecture_score = result["providers"][conjecture_key]["overall_score"]
                
                # Conjecture should generally perform better
                assert conjecture_score >= direct_score


class TestErrorHandling:
    """Test error handling and edge cases"""

    @pytest.mark.skipif(not EVALUATION_FRAMEWORK_AVAILABLE, reason="Evaluation framework not available")
    @pytest.mark.asyncio
    async def test_wrapper_generation_error(self):
        """Test wrapper error handling during generation"""
        wrapper = GraniteTinyWrapper()
        
        # Mock model that raises exception
        mock_model = AsyncMock()
        mock_model.get_model_response = AsyncMock(side_effect=Exception("Generation failed"))
        wrapper.model = mock_model
        
        with pytest.raises(Exception):
            await wrapper.a_generate("Test prompt")

    @pytest.mark.skipif(not EVALUATION_FRAMEWORK_AVAILABLE, reason="Evaluation framework not available")
    @pytest.mark.asyncio
    async def test_evaluation_framework_provider_error(self):
        """Test evaluation framework error handling"""
        framework = EvaluationFramework()
        
        test_case = framework.create_test_case(
            input_text="Test question",
            expected_output="Test answer",
            actual_output="Test response"
        )
        
        # Mock wrapper that raises exception
        with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
            mock_wrapper = AsyncMock()
            mock_wrapper.a_generate = AsyncMock(side_effect=Exception("Wrapper error"))
            mock_create.return_value = mock_wrapper
            
            result = await framework.evaluate_provider(
                "ibm/granite-4-h-tiny", [test_case], use_conjecture=False
            )
            
            assert result.success is False
            assert "error" in result.__dict__.values()

    @pytest.mark.skipif(not EVALUATION_FRAMEWORK_AVAILABLE, reason="Evaluation framework not available")
    def test_wrapper_configuration_validation(self):
        """Test wrapper configuration validation"""
        # Test with valid configurations
        wrapper1 = GraniteTinyWrapper()
        assert wrapper1.model_name == "ibm/granite-4-h-tiny"
        
        wrapper2 = GLM46Wrapper()
        assert wrapper2.model_name == "zai/GLM-4.6"
        
        wrapper3 = GptOss20bWrapper()
        assert wrapper3.model_name == "openrouter/gpt-oss-20b"

    @pytest.mark.skipif(not EVALUATION_FRAMEWORK_AVAILABLE, reason="Evaluation framework not available")
    def test_framework_metrics_initialization(self):
        """Test framework metrics initialization"""
        framework = EvaluationFramework()
        
        # Check that all expected metrics are initialized
        expected_metrics = [
            "relevancy", "faithfulness", "exact_match", 
            "summarization", "bias", "toxicity"
        ]
        
        for metric in expected_metrics:
            assert metric in framework.metrics
            assert hasattr(framework.metrics[metric], 'threshold')


class TestPerformance:
    """Test performance and scalability"""

    @pytest.mark.skipif(not EVALUATION_FRAMEWORK_AVAILABLE, reason="Evaluation framework not available")
    @pytest.mark.asyncio
    async def test_concurrent_provider_evaluation(self):
        """Test concurrent evaluation of multiple providers"""
        framework = EvaluationFramework()
        
        test_cases = [
            framework.create_test_case(
                input_text=f"Question {i}",
                expected_output=f"Answer {i}",
                actual_output=f"Response {i}"
            )
            for i in range(3)
        ]
        
        providers = ["ibm/granite-4-h-tiny", "zai/GLM-4.6"]
        
        # Mock wrappers
        with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
            mock_wrapper = AsyncMock()
            mock_wrapper.a_generate = AsyncMock(return_value="Mock response")
            mock_create.return_value = mock_wrapper
            
            # Run evaluations concurrently
            tasks = [
                framework.evaluate_provider(provider, test_cases, use_conjecture=False)
                for provider in providers
            ]
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == len(providers)
            assert all(result.success for result in results)

    @pytest.mark.skipif(not EVALUATION_FRAMEWORK_AVAILABLE, reason="Evaluation framework not available")
    @pytest.mark.asyncio
    async def test_large_dataset_evaluation(self):
        """Test evaluation with larger dataset"""
        framework = EvaluationFramework()
        
        # Create larger test dataset
        test_cases = [
            framework.create_test_case(
                input_text=f"Question {i}",
                expected_output=f"Answer {i}",
                actual_output=f"Response {i}"
            )
            for i in range(10)
        ]
        
        # Mock wrapper
        with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
            mock_wrapper = AsyncMock()
            mock_wrapper.a_generate = AsyncMock(return_value="Mock response")
            mock_create.return_value = mock_wrapper
            
            result = await framework.evaluate_provider(
                "ibm/granite-4-h-tiny", test_cases, use_conjecture=False
            )
            
            assert result.success is True
            assert result.overall_score >= 0
            assert len(result.metrics_results) > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])