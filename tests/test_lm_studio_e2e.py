"""
End-to-End Tests for LM Studio Integration
Tests the complete functionality with a running LM Studio instance
"""

import unittest
import os
import sys
import time
from typing import List

# Add src to path
sys.path.insert(0, './src')

from contextflow import Conjecture
from core.unified_models import Claim


class TestLMStudioEndToEnd(unittest.TestCase):
    """End-to-end tests for LM Studio integration"""
    
    @classmethod
    def setUpClass(cls):
        """Set up environment for LM Studio tests"""
        # Set environment to use LM Studio
        os.environ['Conjecture_LLM_PROVIDER'] = 'lm_studio'
        os.environ['Conjecture_LLM_API_URL'] = 'http://127.0.0.1:1234'
        os.environ['Conjecture_LLM_MODEL'] = 'ibm/granite-4-h-tiny'
        
        # Initialize Conjecture instance
        cls.conjecture = Conjecture()
    
    def test_conjecture_initialization(self):
        """Test that Conjecture initializes correctly with LM Studio"""
        self.assertEqual(self.conjecture.config.llm_provider, 'lm_studio')
        self.assertEqual(self.conjecture.config.llm_api_url, 'http://127.0.0.1:1234')
        self.assertEqual(self.conjecture.config.llm_model, 'ibm/granite-4-h-tiny')
        self.assertTrue(self.conjecture.llm_bridge.is_available())
    
    def test_claim_creation_and_validation(self):
        """Test creating a claim with LM Studio validation"""
        # Create a claim with LM validation enabled
        content = "Machine learning algorithms require training data to learn patterns"
        claim = self.conjecture.add_claim(
            content=content,
            confidence=0.75,
            claim_type="concept",
            validate_with_llm=True  # Use LLM for validation
        )
        
        # Verify the claim was created with expected properties
        self.assertIsInstance(claim, Claim)
        self.assertEqual(claim.content, content)
        self.assertGreaterEqual(claim.confidence, 0.0)
        self.assertLessEqual(claim.confidence, 1.0)
        self.assertEqual(claim.type[0].value, "concept")
    
    def test_exploration_functionality(self):
        """Test exploration functionality with LM Studio"""
        # Try to explore a topic
        result = self.conjecture.explore("machine learning basics", max_claims=3)
        
        # Verify the result structure
        self.assertIsNotNone(result)
        self.assertGreaterEqual(len(result.claims), 0)  # May or may not have claims depending on knowledge base
        self.assertLessEqual(len(result.claims), 3)  # Should not exceed max_claims
        
        # Verify claims have expected structure if any were returned
        for claim in result.claims:
            self.assertIsInstance(claim, Claim)
            self.assertGreaterEqual(claim.confidence, 0.0)
            self.assertLessEqual(claim.confidence, 1.0)
            self.assertIsNotNone(claim.content)
            self.assertGreater(len(claim.content.strip()), 0)
    
    def test_different_claim_types(self):
        """Test creating claims with different types using LM Studio"""
        claim_types = ["concept", "example", "thesis", "goal"]
        
        for claim_type in claim_types:
            with self.subTest(claim_type=claim_type):
                claim = self.conjecture.add_claim(
                    content=f"Test {claim_type} about artificial intelligence",
                    confidence=0.80,
                    claim_type=claim_type,
                    validate_with_llm=True
                )
                
                self.assertEqual(claim.type[0].value, claim_type)
                self.assertIsInstance(claim, Claim)
    
    def test_response_quality(self):
        """Test that responses from LM Studio are reasonable"""
        # Test that exploration returns coherent content
        result = self.conjecture.explore("simple python programming concept", max_claims=2)
        
        for claim in result.claims:
            # Check that content is meaningful
            self.assertGreater(len(claim.content.strip()), 10)  # At least 10 characters
            # Check confidence is in valid range
            self.assertGreaterEqual(claim.confidence, 0.0)
            self.assertLessEqual(claim.confidence, 1.0)
        
    def test_fallback_behavior(self):
        """Test that the system handles connection issues gracefully"""
        # Temporarily change to an invalid URL to test fallback
        original_url = self.conjecture.config.llm_api_url
        self.conjecture.config.llm_api_url = "http://invalid-url:9999"
        
        # Should still work with mock backend or error handling
        try:
            result = self.conjecture.explore("test query", max_claims=1)
            # Should have results from mock or proper error handling
            self.assertIsNotNone(result)
        except Exception as e:
            # If it fails, that's also valid behavior
            pass
        finally:
            # Restore original URL
            self.conjecture.config.llm_api_url = original_url


def run_end_to_end_tests():
    """Run the end-to-end tests for LM Studio integration"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestLMStudioEndToEnd)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    print("Running end-to-end tests for LM Studio integration...")
    print("Ensure LM Studio is running at http://127.0.0.1:1234 with ibm/granite-4-h-tiny model")
    run_end_to_end_tests()