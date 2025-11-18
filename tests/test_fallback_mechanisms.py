"""
Fallback Mechanism Tests for LM Studio Integration
Tests the fallback behavior when LM Studio is unavailable
"""

import unittest
import os
import sys

sys.path.insert(0, './src')

from contextflow import Conjecture


class TestFallbackMechanisms(unittest.TestCase):
    """Tests for fallback mechanisms when LM Studio is unavailable"""
    
    def test_fallback_with_invalid_lm_studio_url(self):
        """Test that the system falls back when LM Studio is unavailable"""
        # Set environment to use LM Studio with an invalid URL
        os.environ['Conjecture_LLM_PROVIDER'] = 'lm_studio'
        os.environ['Conjecture_LLM_API_URL'] = 'http://invalid-url:9999'
        os.environ['Conjecture_LLM_MODEL'] = 'ibm/granite-4-h-tiny'
        
        # Initialize Conjecture - this should handle the failure gracefully
        cf = Conjecture()
        
        # Even if primary provider is unavailable, the system should still work
        # using fallback mechanisms or mock results
        self.assertIsNotNone(cf)
        
        # Check that the LLM bridge is not available with invalid URL
        # but the system still functions for core operations
        print(f"LLM Bridge available with invalid URL: {cf.llm_bridge.is_available()}")
        
        # The exploration should still work (using fallback/mock)
        result = cf.explore("test query", max_claims=1)
        self.assertIsNotNone(result)
        # Should have mock results even if LLM is not available
        print(f"Got {len(result.claims)} claims (mock/fallback results)")
    
    def test_provider_switching(self):
        """Test switching between different providers"""
        # Test with Chutes as primary (though it may not be configured)
        os.environ['Conjecture_LLM_PROVIDER'] = 'chutes'
        os.environ['Conjecture_LLM_API_URL'] = 'https://llm.chutes.ai/v1'
        
        cf = Conjecture()
        self.assertEqual(cf.config.llm_provider, 'chutes')
        
        # Now test with LM Studio
        os.environ['Conjecture_LLM_PROVIDER'] = 'lm_studio'
        os.environ['Conjecture_LLM_API_URL'] = 'http://127.0.0.1:1234'
        os.environ['Conjecture_LLM_MODEL'] = 'ibm/granite-4-h-tiny'
        
        cf_lmstudio = Conjecture()
        self.assertEqual(cf_lmstudio.config.llm_provider, 'lm_studio')
        self.assertEqual(cf_lmstudio.config.llm_model, 'ibm/granite-4-h-tiny')
        
        # Verify that the LM Studio instance has the correct configuration
        print(f"LM Studio provider: {cf_lmstudio.config.llm_provider}")
        print(f"LM Studio model: {cf_lmstudio.config.llm_model}")
        print(f"LM Studio URL: {cf_lmstudio.config.llm_api_url}")


def run_fallback_tests():
    """Run the fallback mechanism tests"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestFallbackMechanisms)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    print("Running fallback mechanism tests...")
    run_fallback_tests()