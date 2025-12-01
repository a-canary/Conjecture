"""
Fallback Mechanism Tests for LM Studio Integration
Tests the fallback behavior when LM Studio is unavailable
"""

import unittest
import os
import sys

sys.path.insert(0, "./src")

from conjecture import Conjecture


class TestFallbackMechanisms(unittest.TestCase):
    """Tests for fallback mechanisms when LM Studio is unavailable"""

    def test_fallback_with_invalid_lm_studio_url(self):
        """Test that the system falls back when LM Studio is unavailable"""
        # Set environment to use LM Studio with an invalid URL
        os.environ["Conjecture_LLM_PROVIDER"] = "lm_studio"
        os.environ["Conjecture_LLM_API_URL"] = "http://invalid-url:9999"
        os.environ["Conjecture_LLM_MODEL"] = "ibm/granite-4-h-tiny"

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
        os.environ["Conjecture_LLM_PROVIDER"] = "chutes"
        os.environ["Conjecture_LLM_API_URL"] = "https://llm.chutes.ai/v1"

        cf = Conjecture()
        self.assertEqual(cf.config.llm_provider, "chutes")

        # Now test with LM Studio
        os.environ["Conjecture_LLM_PROVIDER"] = "lm_studio"
        os.environ["Conjecture_LLM_API_URL"] = "http://127.0.0.1:1234"
        os.environ["Conjecture_LLM_MODEL"] = "ibm/granite-4-h-tiny"

        cf_lmstudio = Conjecture()
        self.assertEqual(cf_lmstudio.config.llm_provider, "lm_studio")
        self.assertEqual(cf_lmstudio.config.llm_model, "ibm/granite-4-h-tiny")

        # Verify that the LM Studio instance has the correct configuration
        print(f"LM Studio provider: {cf_lmstudio.config.llm_provider}")
        print(f"LM Studio model: {cf_lmstudio.config.llm_model}")
        print(f"LM Studio URL: {cf_lmstudio.config.llm_api_url}")

    def test_hallucination_reduction_via_verification(self):
        """Test that low-confidence claims are flagged or confidence reduced during validation"""
        # Initialize Conjecture
        cf = Conjecture()

        # Create a mock response for the LLM validation
        # We want to simulate the LLM identifying a hallucination
        from src.processing.bridge import LLMResponse
        from src.core.models import Claim, ClaimState, ClaimType

        # Mock the LLM bridge process method
        # We need to handle the case where llm_bridge might not be fully initialized or using fallback
        # If using fallback (mock), validation might be skipped or different.
        # We assume for this test we can monkeypatch the instance.

        original_process = cf.llm_bridge.process

        def mock_validation_process(request):
            # Check if this is a validation request
            if request.task_type == "validate":
                # Return a response that downgrades confidence
                return LLMResponse(
                    success=True,
                    content="VALIDATED: False\nCONFIDENCE: 0.2\nREASONING: This claim contradicts known laws of physics.\nSUGGESTED_EDIT: NO_CHANGE",
                    generated_claims=[],
                )
            return original_process(request)

        # Apply mock
        cf.llm_bridge.process = mock_validation_process

        try:
            # Try to add a "hallucinated" claim
            # "Perpetual motion machines are widely used in industry"
            # We force validate_with_llm=True, but we also need to ensure is_available() returns True for the bridge
            # otherwise add_claim skips validation.

            if not cf.llm_bridge.is_available():
                # Force availability for test
                cf.llm_bridge.is_available = lambda: True

            claim = cf.add_claim(
                content="Perpetual motion machines are widely used in modern industrial manufacturing",
                confidence=0.9,  # User claims high confidence
                claim_type="thesis",
                validate_with_llm=True,
            )

            # Assert that the system (via our mock) reduced the confidence
            # Note: if validation failed entirely (exception), it keeps original confidence.
            # We want to ensure validation *happened*.

            self.assertLess(claim.confidence, 0.5)
            print(
                f"Hallucination check passed: Confidence reduced from 0.9 to {claim.confidence}"
            )

        finally:
            # Restore original method
            cf.llm_bridge.process = original_process


def run_fallback_tests():
    """Run the fallback mechanism tests"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestFallbackMechanisms)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    print("Running fallback mechanism tests...")
    run_fallback_tests()
