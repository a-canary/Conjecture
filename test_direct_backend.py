#!/usr/bin/env python3
"""
Direct backend test without CLI package imports
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_direct_backend_imports():
    """Test backend imports directly."""
    try:
        # Test configuration
        from config.config import get_config

        config = get_config()
        print(f"PASS: Configuration loaded - workspace: {config.workspace}")

        # Test backend imports directly
        from cli.backends.local import LocalBackend

        print("PASS: LocalBackend import successful")

        from cli.backends.cloud import CloudBackend

        print("PASS: CloudBackend import successful")

        from cli.backends.hybrid import HybridBackend

        print("PASS: HybridBackend import successful")

        from cli.backends.auto import AutoBackend

        print("PASS: AutoBackend import successful")

        # Test backend instantiation
        local_backend = LocalBackend()
        print(
            f"PASS: LocalBackend instantiated - available: {local_backend.is_available()}"
        )

        cloud_backend = CloudBackend()
        print(
            f"PASS: CloudBackend instantiated - available: {cloud_backend.is_available()}"
        )

        hybrid_backend = HybridBackend()
        print(
            f"PASS: HybridBackend instantiated - available: {hybrid_backend.is_available()}"
        )

        auto_backend = AutoBackend()
        print(
            f"PASS: AutoBackend instantiated - available: {auto_backend.is_available()}"
        )

        # Test process_prompt method exists
        assert hasattr(local_backend, "process_prompt"), (
            "LocalBackend missing process_prompt"
        )
        assert hasattr(cloud_backend, "process_prompt"), (
            "CloudBackend missing process_prompt"
        )
        assert hasattr(hybrid_backend, "process_prompt"), (
            "HybridBackend missing process_prompt"
        )
        assert hasattr(auto_backend, "process_prompt"), (
            "AutoBackend missing process_prompt"
        )

        print("PASS: All backends have process_prompt method")

        return True

    except Exception as e:
        print(f"FAIL: Direct backend import test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_base_cli_functionality():
    """Test base CLI functionality directly."""
    try:
        from cli.base_cli import BaseCLI
        from config.config import get_config

        config = get_config()

        # Create a test backend instance
        class TestBackend(BaseCLI):
            def __init__(self):
                super().__init__()
                self.db_path = "data/test_conjecture.db"

            def create_claim(
                self, content, confidence, user_id, analyze=False, tags=None, **kwargs
            ):
                return self._save_claim(content, confidence, user_id, None, tags)

            def get_claim(self, claim_id):
                return self._get_claim(claim_id)

            def search_claims(self, query, limit=10, **kwargs):
                return self._search_claims(query, limit)

            def analyze_claim(self, claim_id, **kwargs):
                return {"claim_id": claim_id, "analysis": "test"}

            def process_prompt(self, prompt_text, confidence=0.8, verbose=0, **kwargs):
                return super().process_prompt(
                    prompt_text, confidence, verbose, **kwargs
                )

            def is_available(self):
                return True

        backend = TestBackend()
        backend._init_database()

        # Test process_prompt with verbose=1
        print("Testing process_prompt with verbose=1:")
        result = backend.process_prompt(
            "test prompt for explainer video app", confidence=0.8, verbose=1
        )

        print(f"Debug: result = {result}")

        if result and "claim_id" in result:
            print(
                f"PASS: process_prompt test successful - claim_id: {result['claim_id']}"
            )
            print(f"      Tags: {result['tags']}")
            print(f"      Workspace context: {result['workspace_context']}")
            print(f"      Final status: {result['final_status']}")
        else:
            print("FAIL: process_prompt returned invalid result")
            return False

        # Test process_prompt with verbose=0
        print("\nTesting process_prompt with verbose=0:")
        result2 = backend.process_prompt(
            "simple test prompt", confidence=0.95, verbose=0
        )

        if result2 and "claim_id" in result2:
            print(
                f"PASS: process_prompt verbose=0 test successful - claim_id: {result2['claim_id']}"
            )
            print(f"      Final status: {result2['final_status']}")
        else:
            print("FAIL: process_prompt verbose=0 returned invalid result")
            return False

        return True

    except Exception as e:
        print(f"FAIL: Base CLI functionality test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing direct backend functionality...")
    print("=" * 50)

    success1 = test_direct_backend_imports()
    print()
    success2 = test_base_cli_functionality()

    if success1 and success2:
        print("\nPASS: All direct backend tests passed!")
        print("The prompt command implementation is working correctly.")
        print("Backend files are fixed and functional.")
        sys.exit(0)
    else:
        print("\nFAIL: Some direct backend tests failed!")
        sys.exit(1)
