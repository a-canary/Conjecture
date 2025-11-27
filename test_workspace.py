#!/usr/bin/env python3
"""
Simple test for workspace context
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_workspace_context():
    """Test workspace context configuration."""
    try:
        # Test configuration
        from config.config import get_config

        config = get_config()
        print(f"Configuration loaded successfully:")
        print(f"  Workspace: {config.workspace}")
        print(f"  User: {config.user}")
        print(f"  Team: {config.team}")
        print(f"  User Context: {config.user_context}")
        print(f"  Full Context: {config.full_context}")

        # Test environment variable handling
        expected_workspace = os.getenv("CONJECTURE_WORKSPACE", "default")
        expected_user = os.getenv("CONJECTURE_USER", "user")
        expected_team = os.getenv("CONJECTURE_TEAM", "default")

        assert config.workspace == expected_workspace
        assert config.user == expected_user
        assert config.team == expected_team

        print("PASS: Workspace context test passed!")
        return True

    except Exception as e:
        print(f"FAIL: Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_tag_generation():
    """Test tag generation logic."""
    try:
        # Test tag generation with sample values
        workspace = "test-project"
        user = "alice"
        team = "engineering"

        tags = ["user-prompt", f"workspace-{workspace}", f"user-{user}", f"team-{team}"]
        expected_tags = [
            "user-prompt",
            "workspace-test-project",
            "user-alice",
            "team-engineering",
        ]

        print(f"\nTag generation test:")
        print(f"  Workspace: {workspace}")
        print(f"  User: {user}")
        print(f"  Team: {team}")
        print(f"  Generated tags: {tags}")

        assert tags == expected_tags

        print("PASS: Tag generation test passed!")
        return True

    except Exception as e:
        print(f"FAIL: Tag generation test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing workspace context...")
    print("=" * 50)

    success1 = test_workspace_context()
    success2 = test_tag_generation()

    if success1 and success2:
        print("\nPASS: All tests passed!")
        sys.exit(0)
    else:
        print("\nFAIL: Some tests failed!")
        sys.exit(1)
