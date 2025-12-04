#!/usr/bin/env python3
"""
Quick test of comprehensive experiment runner
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from research.comprehensive_experiment_runner import test_model_connectivity, MODELS


async def main():
    print("Testing model connectivity...")
    await test_model_connectivity()

    print(f"\nModels configured: {list(MODELS.keys())}")
    print("Connectivity test complete.")


if __name__ == "__main__":
    asyncio.run(main())
