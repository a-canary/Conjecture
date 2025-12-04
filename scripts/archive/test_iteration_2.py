#!/usr/bin/env python3
"""
Quick test of iteration 2 connectivity and basic functionality
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

try:
    from conjecture_iteration_2 import make_api_call, MODELS

    async def test_connectivity():
        print("Testing model connectivity...")
        for model_key, config in MODELS.items():
            test_prompt = "Say 'OK' if you can read this."
            result = make_api_call(
                test_prompt, {**config, "name": model_key}, max_tokens=10
            )
            if result["status"] == "success":
                print(f"  {model_key}: OK")
            else:
                print(f"  {model_key}: FAIL - {result['error']}")

    asyncio.run(test_connectivity())

except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Error: {e}")
