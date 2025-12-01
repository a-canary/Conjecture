"""
Comprehensive Tests for LLM Provider Integration
Tests all 9 providers with mock responses and error scenarios
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import os
from typing import Dict, Any

from src.core.basic_models import BasicClaim, ClaimState, ClaimType
from src.processing.llm.llm_manager import LLMManager
from src.processing.llm.chutes_integration import ChutesProcessor
from src.processing.llm.openrouter_integration import OpenRouterProcessor
from src.processing.llm.groq_integration import GroqProcessor
from src.processing.llm.openai_integration import OpenAIProcessor
from src.processing.llm.anthropic_integration import AnthropicProcessor
from src.processing.llm.google_integration import GoogleProcessor, GOOGLE_AVAILABLE
from src.processing.llm.cohere_integration import CohereProcessor
from src.processing.llm.gemini_integration import GeminiProcessor, GEMINI_AVAILABLE
from src.processing.llm.local_providers_adapter import LocalProviderProcessor
from src.processing.llm.error_handling import LLMErrorHandler, RetryConfig


class TestLLMProviderBase(unittest.TestCase):
    """Base class for LLM provider tests"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_claims = [
            BasicClaim(
                claim_id="test_1",
                content="The sky is blue",
                claim_type=ClaimType.ASSERTION.value,
                confidence=0.8
            ),
            BasicClaim(
                claim_id="test_2", 
                content="Water boils at 100°C at sea level",
                claim_type=ClaimType.ASSERTION.value,
                confidence=0.95
            )
        ]
        
        # Mock response for successful processing
        self.mock_response = {
            "choices": [{"message": {"content": '''{
                "claims": [
                    {
                        "claim_id": "test_1",
                        "state": "VERIFIED",
                        "confidence": 0.85,
                        "analysis": "Empirically observable fact",
                        "verification": "Direct observation confirms"'''
                    },
                    {
                        "claim_id": "test_2", 
                        "state": "VERIFIED",
                        "confidence": 0.98,
                        "analysis": "Well-established scientific fact",
                        "verification": "Multiple scientific studies confirm"
                    }
                ]
            }'}}],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 100,
                "total_tokens": 150
            }
        }
        
        self.mock_error_response = {
            "error": {
                "message": "API Error",
                "type": "api_error"
            }
        }

    def create_mock_processor(self, processor_class, **init_kwargs):
        """Create a mock processor for testing"""
        processor = processor_class.__new__(processor_class)
        
        # Initialize common attributes
        processor.api_key = init_kwargs.get("api_key", "test_key")
        processor.api_url = init_kwargs.get("api_url", "https://api.test.com")
        processor.model_name = init_kwargs.get("model_name", "test-model")
        processor.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_processing_time": 0.0,
        }
        processor.error_handler = LLMErrorHandler(RetryConfig())
        
        return processor

    def assert_processing_result(self, result, should_succeed=True):
        """Assert processing result has expected properties"""
        self.assertIsInstance(result, dict)
        self.assertIn("success", result)
        self.assertIn("processed_claims", result)
        self.assertIn("errors", result)
        self.assertIn("processing_time", result)
        self.assertIn("tokens_used", result)
        
        if should_succeed:
            self.assertTrue(result["success"])
            self.assertIsInstance(result["processed_claims"], list)
            self.assertEqual(len(result["errors"]), 0)
        else:
            self.assertFalse(result["success"])
            self.assertGreater(len(result["errors"]), 0)


class TestChutesProcessor(TestLLMProviderBase):
    """Test Chutes.ai processor"""

    @patch('src.processing.llm.chutes_integration.requests.post')
    def test_successful_processing(self, mock_post):
        """Test successful claim processing with Chutes.ai"""
        mock_response = Mock()
        mock_response.json.return_value = self.mock_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        processor = ChutesProcessor(
            api_key="test_key",
            api_url="https://llm.chutes.ai/v1",
            model_name="test-model"
        )

        result = processor.process_claims(self.sample_claims, task="analyze")
        self.assert_processing_result(result, should_succeed=True)

    @patch('src.processing.llm.chutes_integration.requests.post')
    def test_api_error_handling(self, mock_post):
        """Test API error handling"""
        mock_post.side_effect = Exception("API Error")

        processor = ChutesProcessor(
            api_key="test_key",
            api_url="https://llm.chutes.ai/v1",
            model_name="test-model"
        )

        result = processor.process_claims(self.sample_claims, task="analyze")
        self.assert_processing_result(result, should_succeed=False)

    def test_stats_tracking(self):
        """Test statistics tracking"""
        processor = ChutesProcessor(
            api_key="test_key",
            api_url="https://llm.chutes.ai/v1",
            model_name="test-model"
        )

        initial_stats = processor.get_stats()
        self.assertEqual(initial_stats["total_requests"], 0)

        # Simulate processing
        processor.stats["total_requests"] = 1
        processor.stats["successful_requests"] = 1

        updated_stats = processor.get_stats()
        self.assertEqual(updated_stats["total_requests"], 1)
        self.assertEqual(updated_stats["success_rate"], 1.0)


class TestOpenRouterProcessor(TestLLMProviderBase):
    """Test OpenRouter processor"""

    @patch('src.processing.llm.openrouter_integration.requests.post')
    def test_successful_processing(self, mock_post):
        """Test successful claim processing with OpenRouter"""
        mock_response = Mock()
        mock_response.json.return_value = self.mock_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        processor = OpenRouterProcessor(
            api_key="test_key",
            api_url="https://openrouter.ai/api/v1",
            model_name="openai/gpt-3.5-turbo"
        )

        result = processor.process_claims(self.sample_claims, task="analyze")
        self.assert_processing_result(result, should_succeed=True)

    def test_headers_configuration(self):
        """Test that OpenRouter sends proper headers"""
        processor = OpenRouterProcessor(
            api_key="test_key",
            api_url="https://openrouter.ai/api/v1",
            model_name="openai/gpt-3.5-turbo"
        )

        # Headers should include OpenRouter-specific ones
        with patch('src.processing.llm.openrouter_integration.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"choices": [{"message": {"content": "test"}}]}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            processor.generate_response("test")

            # Check that proper headers were included
            call_args = mock_post.call_args
            headers = call_args[1]["headers"]
            self.assertIn("HTTP-Referer", headers)
            self.assertIn("X-Title", headers)


class TestGroqProcessor(TestLLMProviderBase):
    """Test Groq processor"""

    @patch('src.processing.llm.groq_integration.requests.post')
    def test_successful_processing(self, mock_post):
        """Test successful claim processing with Groq"""
        mock_response = Mock()
        mock_response.json.return_value = self.mock_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        processor = GroqProcessor(
            api_key="test_key",
            api_url="https://api.groq.com/openai/v1",
            model_name="llama3-8b-8192"
        )

        result = processor.process_claims(self.sample_claims, task="analyze")
        self.assert_processing_result(result, should_succeed=True)

    def test_fast_retry_config(self):
        """Test that Groq uses faster retry configuration"""
        processor = GroqProcessor(
            api_key="test_key",
            api_url="https://api.groq.com/openai/v1",
            model_name="llama3-8b-8192"
        )

        # Groq should use faster retry config
        retry_config = processor.error_handler.retry_config
        self.assertEqual(retry_config.base_delay, 0.5)
        self.assertEqual(retry_config.max_delay, 15.0)


class TestOpenAIProcessor(TestLLMProviderBase):
    """Test OpenAI processor"""

    @patch('src.processing.llm.openai_integration.requests.post')
    def test_successful_processing(self, mock_post):
        """Test successful claim processing with OpenAI"""
        mock_response = Mock()
        mock_response.json.return_value = self.mock_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        processor = OpenAIProcessor(
            api_key="test_key",
            api_url="https://api.openai.com/v1",
            model_name="gpt-3.5-turbo"
        )

        result = processor.process_claims(self.sample_claims, task="analyze")
        self.assert_processing_result(result, should_succeed=True)


class TestAnthropicProcessor(TestLLMProviderBase):
    """Test Anthropic processor"""

    @patch('src.processing.llm.anthropic_integration.requests.post')
    def test_successful_processing(self, mock_post):
        """Test successful claim processing with Anthropic"""
        # Anthropic response format is different
        anthropic_mock_response = {
            "content": [{"text": self.mock_response["choices"][0]["message"]["content"]}],
            "usage": {
                "input_tokens": 50,
                "output_tokens": 100
            }
        }

        mock_response = Mock()
        mock_response.json.return_value = anthropic_mock_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        processor = AnthropicProcessor(
            api_key="test_key",
            api_url="https://api.anthropic.com",
            model_name="claude-3-haiku-20240307"
        )

        result = processor.process_claims(self.sample_claims, task="analyze")
        self.assert_processing_result(result, should_succeed=True)

    def test_message_format_conversion(self):
        """Test that OpenAI messages are converted to Claude format"""
        processor = AnthropicProcessor(
            api_key="test_key",
            api_url="https://api.anthropic.com",
            model_name="claude-3-haiku-20240307"
        )

        openai_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"}
        ]

        claude_messages = processor._convert_messages_to_claude_format(openai_messages)
        
        # Should skip system messages and convert others
        expected = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"}
        ]
        
        self.assertEqual(claude_messages, expected)


@unittest.skipUnless(GOOGLE_AVAILABLE, "Google library not available")
class TestGoogleProcessor(TestLLMProviderBase):
    """Test Google processor"""

    @patch('src.processing.llm.google_integration.genai.GenerativeModel')
    @patch('src.processing.llm.google_integration.genai.configure')
    def test_successful_processing(self, mock_configure, mock_model):
        """Test successful claim processing with Google"""
        # Mock the response
        mock_response = Mock()
        mock_response.text = self.mock_response["choices"][0]["message"]["content"]
        mock_model.return_value.generate_content.return_value = mock_response

        processor = GoogleProcessor(
            api_key="test_key",
            api_url="https://generativelanguage.googleapis.com",
            model_name="gemini-pro"
        )

        result = processor.process_claims(self.sample_claims, task="analyze")
        self.assert_processing_result(result, should_succeed=True)

    def test_token_estimation(self):
        """Test token estimation for Google (which doesn't provide detailed usage)"""
        processor = GoogleProcessor(
            api_key="test_key",
            api_url="https://generativelanguage.googleapis.com",
            model_name="gemini-pro"
        )

        mock_response = Mock()
        mock_response.text = "This is a test response"

        total_tokens, completion_tokens = processor._extract_usage_stats(mock_response)
        
        # Should estimate based on character count (1 token ≈ 4 characters)
        expected_tokens = len("This is a test response") // 4
        self.assertEqual(total_tokens, expected_tokens)
        self.assertEqual(completion_tokens, expected_tokens)


class TestCohereProcessor(TestLLMProviderBase):
    """Test Cohere processor"""

    @patch('src.processing.llm.cohere_integration.requests.post')
    def test_successful_processing(self, mock_post):
        """Test successful claim processing with Cohere"""
        # Cohere response format is different
        cohere_mock_response = {
            "generations": [{"text": self.mock_response["choices"][0]["message"]["content"]}],
            "meta": {
                "billed_units": {
                    "input_tokens": 50,
                    "output_tokens": 100
                }
            }
        }

        mock_response = Mock()
        mock_response.json.return_value = cohere_mock_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        processor = CohereProcessor(
            api_key="test_key",
            api_url="https://api.cohere.ai/v1",
            model_name="command"
        )

        result = processor.process_claims(self.sample_claims, task="analyze")
        self.assert_processing_result(result, should_succeed=True)


@unittest.skipUnless(GEMINI_AVAILABLE, "Gemini library not available")
class TestGeminiProcessor(TestLLMProviderBase):
    """Test Gemini processor (legacy)"""

    @patch('src.processing.llm.gemini_integration.genai.GenerativeModel')
    @patch('src.processing.llm.gemini_integration.genai.configure')
    def test_successful_processing(self, mock_configure, mock_model):
        """Test successful claim processing with Gemini"""
        # Mock the response
        mock_response = Mock()
        mock_response.text = self.mock_response["choices"][0]["message"]["content"]
        mock_model.return_value.generate_content.return_value = mock_response

        processor = GeminiProcessor(
            api_key="test_key",
            model_name="gemini-1.5-flash"
        )

        result = processor.process_claims(self.sample_claims, task="analyze")
        self.assert_processing_result(result, should_succeed=True)


class TestLocalProvidersAdapter(TestLLMProviderBase):
    """Test Local Providers Adapter"""

    @patch('src.processing.llm.local_providers_adapter.asyncio.new_event_loop')
    @patch('src.processing.llm.local_providers_adapter.OllamaClient')
    def test_ollama_processor(self, mock_client, mock_loop):
        """Test Ollama processor"""
        # Mock the async client
        mock_client_instance = Mock()
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        mock_client.return_value.__aexit__.return_value = None
        
        mock_loop.return_value.__enter__.return_value = Mock()
        mock_client_instance.generate_response.return_value = "test response"

        processor = LocalProviderProcessor(
            provider_type="ollama",
            base_url="http://localhost:11434"
        )

        result = processor.generate_response("test")
        self.assert_processing_result(result, should_succeed=True)

    def test_failed_local_provider_initialization(self):
        """Test handling of failed local provider initialization"""
        with patch('src.processing.llm.local_providers_adapter.OllamaClient') as mock_client:
            mock_client.side_effect = Exception("Connection failed")

            with self.assertRaises(RuntimeError):
                LocalProviderProcessor(
                    provider_type="ollama",
                    base_url="http://localhost:11434"
                )


class TestLLMManagerComprehensive(unittest.TestCase):
    """Test the comprehensive LLM Manager"""

    def setUp(self):
        self.sample_claims = [
            BasicClaim(
                claim_id="test_1",
                content="Test claim",
                claim_type=ClaimType.ASSERTION.value,
                confidence=0.8
            )
        ]

    @patch.dict(os.environ, {
        'PROVIDER_API_URL': 'https://llm.chutes.ai/v1',
        'PROVIDER_API_KEY': 'test_key',
        'PROVIDER_MODEL': 'test-model'
    })
    @patch('src.processing.llm.chutes_integration.requests.post')
    def test_unified_provider_detection(self, mock_post):
        """Test unified provider detection from environment"""
        mock_response = Mock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "test"}}]}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        manager = LLMManager()
        
        # Should detect Chutes from URL
        self.assertIn("chutes", manager.processors)
        self.assertEqual(manager.provider_priorities.get("chutes"), 3)

    @patch.dict(os.environ, {}, clear=True)  # Clear all env vars
    def test_no_providers_available(self):
        """Test behavior when no providers are available"""
        manager = LLMManager()
        
        self.assertEqual(len(manager.processors), 0)
        self.assertIsNone(manager.primary_provider)

        # Should raise error when trying to process
        with self.assertRaises(RuntimeError):
            manager.process_claims(self.sample_claims)

    @patch.multiple(
        'src.processing.llm.llm_manager',
        ChutesProcessor=Mock(),
        OpenRouterProcessor=Mock(),
        GroqProcessor=Mock(),
        OpenAIProcessor=Mock(),
        AnthropicProcessor=Mock(),
        GoogleProcessor=Mock(),
        CohereProcessor=Mock(),
        LocalProviderProcessor=Mock()
    )
    @patch.dict(os.environ, {
        'CHUTES_API_KEY': 'chutes_key',
        'OPENROUTER_API_KEY': 'openrouter_key',
        'GROQ_API_KEY': 'groq_key',
        'OPENAI_API_KEY': 'openai_key',
        'ANTHROPIC_API_KEY': 'anthropic_key',
        'GOOGLE_API_KEY': 'google_key',
        'COHERE_API_KEY': 'cohere_key'
    })
    def test_multiple_provider_initialization(self):
        """Test initialization of multiple providers"""
        manager = LLMManager()
        
        # Should initialize all available providers
        self.assertGreater(len(manager.processors), 0)
        
        # Should have priorities set
        for provider_name in manager.processors:
            self.assertIn(provider_name, manager.provider_priorities)

    def test_provider_fallback_logic(self):
        """Test provider fallback logic"""
        manager = LLMManager()
        
        # Mock processors
        primary_processor = Mock()
        primary_processor.process_claims.side_effect = Exception("Primary failed")
        
        fallback_processor = Mock()
        fallback_processor.process_claims.return_value = {"success": True}
        
        manager.processors = {
            "primary": primary_processor,
            "fallback": fallback_processor
        }
        manager.provider_priorities = {"primary": 1, "fallback": 2}
        manager.primary_provider = "primary"

        # Should fallback when primary fails
        result = manager.process_claims(self.sample_claims)
        fallback_processor.process_claims.assert_called_once()

    def test_health_check_comprehensive(self):
        """Test comprehensive health check"""
        manager = LLMManager()
        
        # Mock processors with different health statuses
        healthy_processor = Mock()
        healthy_processor.health_check.return_value = {
            "status": "healthy",
            "model": "test-model"
        }
        
        unhealthy_processor = Mock()
        unhealthy_processor.health_check.return_value = {
            "status": "unhealthy",
            "error": "Connection failed"
        }
        
        manager.processors = {
            "healthy": healthy_processor,
            "unhealthy": unhealthy_processor
        }
        manager.provider_priorities = {"healthy": 1, "unhealthy": 2}

        health_status = manager.health_check()
        
        self.assertEqual(health_status["total_providers"], 2)
        self.assertEqual(health_status["providers"]["healthy"]["status"], "healthy")
        self.assertEqual(health_status["providers"]["unhealthy"]["status"], "unhealthy")
        self.assertEqual(health_status["overall_status"], "degraded")  # One unhealthy

    def test_combined_statistics(self):
        """Test combined statistics from all providers"""
        manager = LLMManager()
        
        # Mock processors with different stats
        processor1 = Mock()
        processor1.get_stats.return_value = {
            "total_requests": 10,
            "successful_requests": 8,
            "total_tokens": 1000,
            "total_processing_time": 5.0
        }
        
        processor2 = Mock()
        processor2.get_stats.return_value = {
            "total_requests": 20,
            "successful_requests": 15,
            "total_tokens": 2000,
            "total_processing_time": 10.0
        }
        
        manager.processors = {
            "processor1": processor1,
            "processor2": processor2
        }

        combined_stats = manager.get_combined_stats()
        
        self.assertEqual(combined_stats["total_requests"], 30)
        self.assertEqual(combined_stats["total_successful"], 23)
        self.assertEqual(combined_stats["total_tokens"], 3000)
        self.assertEqual(combined_stats["total_processing_time"], 15.0)
        self.assertAlmostEqual(combined_stats["overall_success_rate"], 23/30, places=3)

    def test_provider_switching(self):
        """Test manual provider switching"""
        manager = LLMManager()
        
        manager.processors = {
            "provider1": Mock(),
            "provider2": Mock()
        }
        manager.provider_priorities = {"provider1": 1, "provider2": 2}
        manager.primary_provider = "provider1"
        manager.failed_providers = set()

        # Should be able to switch to available provider
        result = manager.switch_provider("provider2")
        self.assertTrue(result)
        self.assertEqual(manager.primary_provider, "provider2")

        # Should not be able to switch to failed provider
        manager.failed_providers.add("provider1")
        result = manager.switch_provider("provider1")
        self.assertFalse(result)


class TestErrorHandlingAdvanced(unittest.TestCase):
    """Test advanced error handling scenarios"""

    def test_circuit_breaker(self):
        """Test circuit breaker functionality"""
        handler = LLMErrorHandler(
            retry_config=RetryConfig(max_attempts=1)  # Quick failure for testing
        )

        # Function that always fails
        @handler.handle_generation
        def failing_function():
            raise Exception("Always fails")

        # Should fail immediately and eventually open circuit
        for i in range(10):  # Multiple attempts to trigger circuit breaker
            try:
                failing_function()
            except Exception:
                pass

        # Circuit should be open after threshold
        circuit = handler.circuit_breakers["generation"]
        self.assertEqual(circuit.state, "OPEN")

    def test_retry_with_backoff(self):
        """Test retry with exponential backoff"""
        attempt_count = 0
        
        def function_with_retries():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:  # Fail first 2 attempts
                raise Exception("Temporary failure")
            return "success"

        handler = LLMErrorHandler(
            retry_config=RetryConfig(max_attempts=3, base_delay=0.01, max_delay=0.1)
        )

        @handler.handle_generation
        def protected_function():
            return function_with_retries()

        result = protected_function()
        self.assertEqual(result, "success")
        self.assertEqual(attempt_count, 3)  # Should have retried twice


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)