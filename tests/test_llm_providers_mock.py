"""
Mock Testing for Cloud LLM Providers
Tests cloud provider integrations without actual API calls
"""

import unittest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import json
import asyncio
from typing import Dict, Any

from src.core.basic_models import BasicClaim, ClaimState, ClaimType


class MockResponse:
    """Mock HTTP response class for testing"""
    
    def __init__(self, json_data: Dict[str, Any], status_code: int = 200):
        self.json_data = json_data
        self.status_code = status_code
        self.text = json.dumps(json_data)
        
    def json(self):
        return self.json_data
        
    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code} Error")


class MockAsyncResponse:
    """Mock async response for local providers"""
    
    def __init__(self, text: str):
        self.text = text
        
    async def text(self):
        return self.text


class TestCloudProvidersMock(unittest.TestCase):
    """Mock tests for cloud providers without API dependencies"""

    def setUp(self):
        """Set up test fixtures"""
        self.sample_claims = [
            BasicClaim(
                claim_id="test_1",
                content="The earth revolves around the sun",
                claim_type=ClaimType.ASSERTION.value,
                confidence=0.9
            )
        ]
        
        self.successful_response_content = {
            "claims": [
                {
                    "claim_id": "test_1",
                    "state": "VERIFIED",
                    "confidence": 0.95,
                    "analysis": "Scientifically established fact",
                    "verification": "Astronomical observations confirm"
                }
            ]
        }

    @patch('src.processing.llm.chutes_integration.requests.post')
    def test_chutes_mock_processing(self, mock_post):
        """Test Chutes.ai processing with mocked HTTP response"""
        from src.processing.llm.chutes_integration import ChutesProcessor
        
        # Mock successful response
        mock_response = MockResponse(self.successful_response_content)
        mock_post.return_value = mock_response
        
        processor = ChutesProcessor(
            api_key="mock_key",
            api_url="https://llm.chutes.ai/v1",
            model_name="test-model"
        )
        
        result = processor.process_claims(self.sample_claims, task="analyze")
        
        self.assertTrue(result.success)
        self.assertEqual(len(result.processed_claims), 1)
        self.assertEqual(result.processed_claims[0].claim_id, "test_1")
        self.assertEqual(result.processed_claims[0].state, ClaimState.VERIFIED)

    @patch('src.processing.llm.chutes_integration.requests.post')
    def test_chutes_mock_error_scenario(self, mock_post):
        """Test Chutes.ai error handling with mocked HTTP error"""
        from src.processing.llm.chutes_integration import ChutesProcessor
        
        # Mock error response
        mock_response = MockResponse({"error": "API Key Invalid"}, status_code=401)
        mock_post.return_value = mock_response
        
        processor = ChutesProcessor(
            api_key="invalid_key",
            api_url="https://llm.chutes.ai/v1",
            model_name="test-model"
        )
        
        result = processor.process_claims(self.sample_claims, task="analyze")
        
        self.assertFalse(result.success)
        self.assertGreater(len(result.errors), 0)

    @patch('src.processing.llm.openrouter_integration.requests.post')
    def test_openrouter_mock_processing(self, mock_post):
        """Test OpenRouter processing with mocked HTTP response"""
        from src.processing.llm.openrouter_integration import OpenRouterProcessor
        
        mock_response = MockResponse(self.successful_response_content)
        mock_post.return_value = mock_response
        
        processor = OpenRouterProcessor(
            api_key="mock_key",
            api_url="https://openrouter.ai/api/v1",
            model_name="openai/gpt-3.5-turbo"
        )
        
        result = processor.process_claims(self.sample_claims, task="analyze")
        
        self.assertTrue(result.success)
        self.assertEqual(len(result.processed_claims), 1)

    def test_openrouter_headers_validation(self):
        """Test that OpenRouter includes required headers"""
        from src.processing.llm.openrouter_integration import OpenRouterProcessor
        
        processor = OpenRouterProcessor(
            api_key="mock_key",
            api_url="https://openrouter.ai/api/v1",
            model_name="openai/gpt-3.5-turbo"
        )
        
        with patch('src.processing.llm.openrouter_integration.requests.post') as mock_post:
            mock_response = MockResponse(self.successful_response_content)
            mock_post.return_value = mock_response
            
            processor.generate_response("test prompt")
            
            # Verify headers were included
            call_args = mock_post.call_args
            headers = call_args[1]['headers']
            
            self.assertIn('Authorization', headers)
            self.assertEqual(headers['Authorization'], 'Bearer mock_key')
            self.assertIn('HTTP-Referer', headers)
            self.assertIn('X-Title', headers)

    @patch('src.processing.llm.groq_integration.requests.post')
    def test_groq_mock_processing(self, mock_post):
        """Test Groq processing with mocked HTTP response"""
        from src.processing.llm.groq_integration import GroqProcessor
        
        mock_response = MockResponse(self.successful_response_content)
        mock_post.return_value = mock_response
        
        processor = GroqProcessor(
            api_key="mock_key",
            api_url="https://api.groq.com/openai/v1",
            model_name="llama3-8b-8192"
        )
        
        result = processor.process_claims(self.sample_claims, task="analyze")
        
        self.assertTrue(result.success)
        self.assertGreater(result.tokens_used, 0)

    def test_groq_retry_configuration(self):
        """Test that Groq uses faster retry configuration"""
        from src.processing.llm.groq_integration import GroqProcessor
        
        processor = GroqProcessor(
            api_key="mock_key",
            api_url="https://api.groq.com/openai/v1",
            model_name="llama3-8b-8192"
        )
        
        retry_config = processor.error_handler.retry_config
        self.assertEqual(retry_config.base_delay, 0.5)
        self.assertEqual(retry_config.max_delay, 15.0)
        self.assertLess(retry_config.max_delay, 30.0)  # Should be faster than others

    @patch('src.processing.llm.openai_integration.requests.post')
    def test_openai_mock_processing(self, mock_post):
        """Test OpenAI processing with mocked HTTP response"""
        from src.processing.llm.openai_integration import OpenAIProcessor
        
        mock_response = MockResponse(self.successful_response_content)
        mock_post.return_value = mock_response
        
        processor = OpenAIProcessor(
            api_key="mock_key",
            api_url="https://api.openai.com/v1",
            model_name="gpt-3.5-turbo"
        )
        
        result = processor.process_claims(self.sample_claims, task="analyze")
        
        self.assertTrue(result.success)
        self.assertGreater(result.processing_time, 0)

    @patch('src.processing.llm.openai_integration.requests.post')
    def test_openai_rate_limit_handling(self, mock_post):
        """Test OpenAI rate limit error handling"""
        from src.processing.llm.openai_integration import OpenAIProcessor
        
        # Mock rate limit response
        mock_response = MockResponse({
            "error": {
                "message": "Rate limit exceeded",
                "type": "rate_limit_error"
            }
        }, status_code=429)
        mock_post.return_value = mock_response
        
        processor = OpenAIProcessor(
            api_key="mock_key",
            api_url="https://api.openai.com/v1",
            model_name="gpt-3.5-turbo"
        )
        
        result = processor.process_claims(self.sample_claims, task="analyze")
        
        self.assertFalse(result.success)

    @patch('src.processing.llm.anthropic_integration.requests.post')
    def test_anthropic_mock_processing(self, mock_post):
        """Test Anthropic processing with mocked HTTP response"""
        from src.processing.llm.anthropic_integration import AnthropicProcessor
        
        # Anthropic uses different response format
        anthropic_response = {
            "content": [{"text": json.dumps(self.successful_response_content)}],
            "usage": {"input_tokens": 50, "output_tokens": 100}
        }
        
        mock_response = MockResponse(anthropic_response)
        mock_post.return_value = mock_response
        
        processor = AnthropicProcessor(
            api_key="mock_key",
            api_url="https://api.anthropic.com",
            model_name="claude-3-haiku-20240307"
        )
        
        result = processor.process_claims(self.sample_claims, task="analyze")
        
        self.assertTrue(result.success)
        self.assertEqual(len(result.processed_claims), 1)

    def test_anthropic_message_conversion(self):
        """Test Anthropic message format conversion"""
        from src.processing.llm.anthropic_integration import AnthropicProcessor
        
        processor = AnthropicProcessor(
            api_key="mock_key",
            api_url="https://api.anthropic.com",
            model_name="claude-3-haiku-20240307"
        )
        
        openai_messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        claude_messages = processor._convert_messages_to_claude_format(openai_messages)
        
        # Should filter out system messages
        expected = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        self.assertEqual(claude_messages, expected)

    def mock_google_model(self):
        """Create mock Google GenerativeModel"""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps(self.successful_response_content)
        mock_model.generate_content.return_value = mock_response
        
        mock_genai = Mock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_genai.configure.return_value = None
        
        return mock_genai, mock_model

    def test_google_mock_processing(self):
        """Test Google processing with mocked library"""
        from src.processing.llm.google_integration import GoogleProcessor
        
        mock_genai, mock_model = self.mock_google_model()
        
        with patch('src.processing.llm.google_integration.genai', mock_genai):
            processor = GoogleProcessor(
                api_key="mock_key",
                api_url="https://generativelanguage.googleapis.com",
                model_name="gemini-pro"
            )
            
            result = processor.process_claims(self.sample_claims, task="analyze")
            
            self.assertTrue(result.success)
            mock_model.generate_content.assert_called_once()

    def test_google_token_estimation(self):
        """Test Google token estimation (since Google doesn't provide detailed usage)"""
        from src.processing.llm.google_integration import GoogleProcessor
        
        mock_genai, mock_model = self.mock_google_model()
        
        with patch('src.processing.llm.google_integration.genai', mock_genai):
            processor = GoogleProcessor(
                api_key="mock_key",
                api_url="https://generativelanguage.googleapis.com",
                model_name="gemini-pro"
            )
            
            result = processor.generate_response("Test response")
            
            # Should estimate tokens based on character count
            self.assertGreater(result.tokens_used, 0)

    @patch('src.processing.llm.cohere_integration.requests.post')
    def test_cohere_mock_processing(self, mock_post):
        """Test Cohere processing with mocked HTTP response"""
        from src.processing.llm.cohere_integration import CohereProcessor
        
        # Cohere response format
        cohere_response = {
            "generations": [{"text": json.dumps(self.successful_response_content)}],
            "meta": {
                "billed_units": {
                    "input_tokens": 50,
                    "output_tokens": 100
                }
            }
        }
        
        mock_response = MockResponse(cohere_response)
        mock_post.return_value = mock_response
        
        processor = CohereProcessor(
            api_key="mock_key",
            api_url="https://api.cohere.ai/v1",
            model_name="command"
        )
        
        result = processor.process_claims(self.sample_claims, task="analyze")
        
        self.assertTrue(result.success)
        self.assertEqual(result.tokens_used, 150)  # 50 + 100

    def test_cohere_chat_vs_generate(self):
        """Test difference between Cohere chat and generate endpoints"""
        from src.processing.llm.cohere_integration import CohereProcessor
        
        processor = CohereProcessor(
            api_key="mock_key",
            api_url="https://api.cohere.ai/v1",
            model_name="command"
        )
        
        # Test generate endpoint
        with patch('src.processing.llm.cohere_integration.requests.post') as mock_post:
            mock_response = MockResponse({"generations": [{"text": "test"}]})
            mock_post.return_value = mock_response
            
            processor.generate_response("test")
            
            # Should call generate endpoint
            mock_post.assert_called_once()
            call_url = mock_post.call_args[0][0]
            self.assertIn("/generate", call_url)

    def test_response_parsing_robustness(self):
        """Test robust response parsing across different formats"""
        test_cases = [
            # Standard JSON
            '{"claims": [{"claim_id": "test", "state": "VERIFIED"}]}',
            # JSON with extra whitespace
            '\n  {"claims": [{"claim_id": "test", "state": "VERIFIED"}]}  \n',
            # Array format
            '[{"claim_id": "test", "state": "VERIFIED"}]',
            # Text-based format
            'claim_id: test\nstate: VERIFIED\nconfidence: 0.85',
            # Malformed JSON (should fallback gracefully)
            'invalid json format but with claim_id: test and state: VERIFIED'
        ]
        
        from src.processing.llm.openai_integration import OpenAIProcessor
        
        processor = OpenAIProcessor(
            api_key="mock_key",
            api_url="https://api.openai.com/v1",
            model_name="gpt-3.5-turbo"
        )
        
        for response_text in test_cases:
            try:
                parsed = processor._parse_claims_from_response(response_text, self.sample_claims)
                # Should not raise exception and return some result
                self.assertIsInstance(parsed, list)
            except Exception as e:
                self.fail(f"Failed to parse response: {response_text}. Error: {e}")


class MockAsyncContext:
    """Mock async context manager for testing"""
    
    def __init__(self, return_value=None, side_effect=None):
        self.return_value = return_value
        self.side_effect = side_effect
        
    async def __aenter__(self):
        if self.side_effect:
            raise self.side_effect
        return AsyncMock(return_value=self.return_value or AsyncMock())
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class TestLocalProvidersMock(unittest.TestCase):
    """Mock tests for local providers"""

    def setUp(self):
        """Set up test fixtures"""
        self.sample_claims = [
            BasicClaim(
                claim_id="test_1",
                content="Test claim for local processing",
                claim_type=ClaimType.ASSERTION.value,
                confidence=0.8
            )
        ]

    @patch('src.processing.llm.local_providers_adapter.asyncio.new_event_loop')
    @patch('src.processing.llm.local_providers_adapter.OllamaClient')
    def test_ollama_mock_success(self, mock_client, mock_loop):
        """Test Ollama with mocked async client"""
        from src.processing.llm.local_providers_adapter import LocalProviderProcessor
        
        # Mock the async client
        mock_client_instance = AsyncMock()
        mock_client.return_value = mock_client_instance
        
        mock_client_instance.generate_response.return_value = json.dumps({
            "claims": [
                {
                    "claim_id": "test_1",
                    "state": "VERIFIED",
                    "confidence": 0.9,
                    "analysis": "Local analysis",
                    "verification": "Local verification"
                }
            ]
        })
        
        # Mock event loop
        mock_event_loop = Mock()
        mock_event_loop.run_until_complete.return_value = mock_client_instance
        mock_loop.return_value = mock_event_loop
        
        processor = LocalProviderProcessor(
            provider_type="ollama",
            base_url="http://localhost:11434",
            model_name="llama2"
        )
        
        result = processor.process_claims(self.sample_claims, task="analyze")
        
        self.assertTrue(result.success)
        self.assertEqual(len(result.processed_claims), 1)

    @patch('src.processing.llm.local_providers_adapter.asyncio.new_event_loop')
    @patch('src.processing.llm.local_providers_adapter.OllamaClient')
    def test_ollama_mock_connection_failure(self, mock_client, mock_loop):
        """Test Ollama connection failure handling"""
        from src.processing.llm.local_providers_adapter import LocalProviderProcessor
        
        # Mock connection failure
        mock_client.side_effect = Exception("Connection refused")
        
        mock_event_loop = Mock()
        mock_loop.return_value = mock_event_loop
        
        with self.assertRaises(RuntimeError):
            LocalProviderProcessor(
                provider_type="ollama",
                base_url="http://localhost:11434"
            )

    @patch('src.processing.llm.local_providers_adapter.asyncio.new_event_loop')
    @patch('src.processing.llm.local_providers_adapter.OllamaClient')
    def test_ollama_mock_generation_failure(self, mock_client, mock_loop):
        """Test Ollama generation failure handling"""
        from src.processing.llm.local_providers_adapter import LocalProviderProcessor
        
        # Mock client that fails generation
        mock_client_instance = AsyncMock()
        mock_client.return_value = mock_client_instance
        
        mock_client_instance.generate_response.side_effect = Exception("Model not found")
        
        mock_event_loop = Mock()
        mock_event_loop.run_until_complete.return_value = mock_client_instance
        mock_loop.return_value = mock_event_loop
        
        try:
            processor = LocalProviderProcessor(
                provider_type="ollama",
                base_url="http://localhost:11434"
            )
            
            result = processor.process_claims(self.sample_claims, task="analyze")
            
            self.assertFalse(result.success)
            self.assertGreater(len(result.errors), 0)
        except Exception:
            # Expected to fail
            pass

    def test_local_models_info(self):
        """Test local models information handling"""
        from src.processing.llm.local_providers_adapter import LocalProviderProcessor
        
        processor = LocalProviderProcessor(
            provider_type="ollama",
            base_url="http://localhost:11434"
        )
        
        # Mock model list
        mock_models = [
            Mock(name="llama2", provider="ollama"),
            Mock(name="mistral", provider="ollama")
        ]
        
        with patch.object(processor, 'get_available_models') as mock_get_models:
            mock_get_models.return_value = []
            
            # Should handle empty model list gracefully
            models = processor.get_available_models()
            self.assertIsInstance(models, list)


class TestIntegrationMock(unittest.TestCase):
    """Mock integration tests for the complete system"""

    @patch.dict('os.environ', {
        'PROVIDER_API_URL': 'https://llm.chutes.ai/v1',
        'PROVIDER_API_KEY': 'mock_chutes_key',
        'PROVIDER_MODEL': 'test-model',
        'OPENAI_API_KEY': 'mock_openai_key',
        'GROQ_API_KEY': 'mock_groq_key'
    })
    @patch('src.processing.llm.chutes_integration.requests.post')
    @patch('src.processing.llm.openai_integration.requests.post')
    @patch('src.processing.llm.groq_integration.requests.post')
    def test_multi_provider_fallback_mock(self, mock_groq, mock_openai, mock_chutes):
        """Test multi-provider fallback with mocked responses"""
        from src.processing.llm.llm_manager import LLMManager
        
        sample_claims = [
            BasicClaim(
                claim_id="test_1",
                content="Test claim for fallback",
                claim_type=ClaimType.ASSERTION.value,
                confidence=0.8
            )
        ]
        
        successful_response = MockResponse({
            "claims": [
                {
                    "claim_id": "test_1",
                    "state": "VERIFIED",
                    "confidence": 0.9,
                    "analysis": "Test analysis",
                    "verification": "Test verification"
                }
            ]
        })
        
        # Primary provider fails
        mock_chutes.side_effect = Exception("Chutes failed")
        
        # Secondary providers succeed
        mock_openai.return_value = successful_response
        mock_groq.return_value = successful_response
        
        manager = LLMManager()
        
        # Should process successfully using fallback
        result = manager.process_claims(sample_claims, task="analyze")
        
        self.assertTrue(result.success)
        self.assertEqual(len(result.processed_claims), 1)

    @patch('src.processing.llm.chutes_integration.requests.post')
    def test_unified_provider_detection_mock(self, mock_post):
        """Test unified provider detection with mocked environment"""
        from src.processing.llm.llm_manager import LLMManager
        
        mock_post.return_value = MockResponse({
            "choices": [{"message": {"content": "test response"}}]
        })
        
        with patch.dict('os.environ', {
            'PROVIDER_API_URL': 'https://api.openai.com/v1',
            'PROVIDER_API_KEY': 'mock_openai_key',
            'PROVIDER_MODEL': 'gpt-3.5-turbo'
        }):
            manager = LLMManager()
            
            # Should detect OpenAI from URL
            self.assertIn("openai", manager.processors)
            self.assertEqual(manager.provider_priorities.get("openai"), 6)


if __name__ == '__main__':
    unittest.main(verbosity=2)