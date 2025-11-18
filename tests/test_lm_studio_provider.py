"""
Test suite for LM Studio provider integration
Tests the LM Studio adapter implementation with the LLM bridge
"""

import os
import unittest
from unittest.mock import Mock, patch, MagicMock

# Import the necessary modules
from src.processing.llm.lm_studio_adapter import LMStudioAdapter, create_lm_studio_adapter_from_config
from src.processing.llm_bridge import LLMRequest
from src.config.simple_config import Config


class TestLMStudioAdapter(unittest.TestCase):
    """Test cases for LM Studio Adapter"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            "base_url": "http://localhost:1234/v1",
            "model": "ibm/granite-4-h-tiny"
        }
        self.adapter = LMStudioAdapter(self.config)

    def test_adapter_initialization(self):
        """Test that LM Studio adapter initializes correctly"""
        self.assertIsInstance(self.adapter, LMStudioAdapter)
        self.assertEqual(self.adapter.config["base_url"], "http://localhost:1234/v1")
        self.assertEqual(self.adapter.config["model"], "ibm/granite-4-h-tiny")

    @patch('src.processing.llm.local_providers_adapter.LocalProviderProcessor')
    def test_adapter_with_mock_processor(self, mock_processor_class):
        """Test adapter with a mocked processor"""
        # Create a mock processor instance
        mock_processor = Mock()
        mock_processor.health_check.return_value = {"status": "healthy"}
        
        # Mock the response from generate_response
        mock_result = Mock()
        mock_result.success = True
        mock_result.processed_claims = []
        mock_result.errors = []
        mock_result.processing_time = 0.1
        mock_result.tokens_used = 10
        mock_result.model_used = "ibm/granite-4-h-tiny"
        
        mock_processor.generate_response.return_value = mock_result
        
        # Assign the mock processor to the adapter
        self.adapter.processor = mock_processor
        
        # Test that the adapter is available
        self.assertTrue(self.adapter.is_available())

    def test_request_conversion(self):
        """Test that requests are properly converted for LM Studio"""
        request = LLMRequest(
            prompt="Test prompt",
            context_claims=[],
            max_tokens=100,
            temperature=0.7,
            task_type="explore"
        )
        
        converted = self.adapter._convert_request(request)
        
        self.assertEqual(converted["prompt"], request.prompt)
        self.assertEqual(converted["context_claims"], request.context_claims)

    def test_create_adapter_from_config(self):
        """Test factory function for creating LM Studio adapter from config"""
        # Test with environment variables
        os.environ["PROVIDER_API_URL"] = "http://localhost:1234/v1"
        os.environ["PROVIDER_MODEL"] = "ibm/granite-4-h-tiny"
        
        adapter = create_lm_studio_adapter_from_config()
        self.assertIsInstance(adapter, LMStudioAdapter)
        
        # Clean up environment variables
        if "PROVIDER_API_URL" in os.environ:
            del os.environ["PROVIDER_API_URL"]
        if "PROVIDER_MODEL" in os.environ:
            del os.environ["PROVIDER_MODEL"]


class TestLMStudioIntegration(unittest.TestCase):
    """Integration tests for LM Studio with Conjecture"""

    def test_config_with_lm_studio_provider(self):
        """Test that configuration properly recognizes LM Studio provider"""
        # Set environment to use LM Studio
        os.environ["Conjecture_LLM_PROVIDER"] = "lm_studio"
        os.environ["Conjecture_LLM_API_URL"] = "http://localhost:1234/v1"
        os.environ["Conjecture_LLM_MODEL"] = "ibm/granite-4-h-tiny"
        
        # Create config
        config = Config()
        
        # Verify settings
        self.assertEqual(config.llm_provider, "lm_studio")
        self.assertEqual(config.llm_api_url, "http://localhost:1234/v1")
        self.assertEqual(config.llm_model, "ibm/granite-4-h-tiny")
        self.assertTrue(config.llm_enabled)  # Should be true for LM Studio
        
        # Clean up environment
        del os.environ["Conjecture_LLM_PROVIDER"]
        del os.environ["Conjecture_LLM_API_URL"]
        del os.environ["Conjecture_LLM_MODEL"]

    def test_config_fallback_to_chutes(self):
        """Test that config defaults to chutes when no provider specified"""
        # Remove any existing provider env var
        if "Conjecture_LLM_PROVIDER" in os.environ:
            del os.environ["Conjecture_LLM_PROVIDER"]
        
        # Create config with default settings
        config = Config()
        
        # Should default to chutes
        self.assertEqual(config.llm_provider, "chutes")
        
        # Restore original if it was set
        if "Conjecture_LLM_PROVIDER" in os.environ:
            os.environ["Conjecture_LLM_PROVIDER"] = "chutes"


def run_tests():
    """Run all tests in this module"""
    unittest.main()


if __name__ == "__main__":
    print("Testing LM Studio provider integration...")
    run_tests()