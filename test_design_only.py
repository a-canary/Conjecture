#!/usr/bin/env python3
"""
Simple test for workspace context and prompt command design
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
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_prompt_design():
    """Test prompt command design concepts."""
    try:
        # Test tag generation
        from config.config import get_config

        config = get_config()

        workspace = config.workspace
        user = config.user
        team = config.team

        # Simulate tag generation
        tags = ["user-prompt", f"workspace-{workspace}", f"user-{user}", f"team-{team}"]

        print(f"Generated tags: {tags}")

        # Test verbose level concepts
        verbose_levels = {
            0: "Final response only",
            1: "Tool calls visible",
            2: "Claims >90% visible",
        }

        print(f"Verbose levels: {verbose_levels}")

        print("✅ Prompt design test passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing workspace context and prompt command design...")
    print("=" * 60)

    success1 = test_workspace_context()
    print()
    success2 = test_prompt_design()

    if success1 and success2:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
