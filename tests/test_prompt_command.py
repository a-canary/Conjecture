#!/usr/bin/env python3
"""
Simple test for the prompt command
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

def test_prompt_command():
    """Test the prompt command functionality."""
    try:
        # Test basic imports
        from config.config import get_config
        from cli.modular_cli import app

        print("✅ Basic imports successful")

        # Test configuration
        config = get_config()
        print(
            f"✅ Configuration loaded: workspace={config.workspace}, user={config.user}, team={config.team}"
        )

        # Test that prompt command is registered
        commands = [cmd.name for cmd in app.commands.values()]
        if "prompt" in commands:
            print("✅ Prompt command is registered")
        else:
            print("❌ Prompt command is NOT registered")
            print(f"Available commands: {commands}")
            return False

        print("✅ All tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_prompt_command()
    sys.exit(0 if success else 1)
