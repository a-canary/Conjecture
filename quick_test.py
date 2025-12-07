#!/usr/bin/env python3
"""
Quick test for optimized conjecture
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

async def main():
    print("Quick test of optimized conjecture")

    from conjecture_optimized import OptimizedConjecture

    cf = OptimizedConjecture()
    print(f"✅ Initialization successful")

    # Test LLM manager property
    try:
        llm_mgr = cf.llm_manager
        print(f"✅ LLM manager accessible: {llm_mgr is not None}")
        if llm_mgr:
            providers = llm_mgr.get_available_providers()
            print(f"  Available providers: {providers}")
    except Exception as e:
        print(f"❌ LLM manager error: {e}")

    await cf.stop_services()

if __name__ == "__main__":
    asyncio.run(main())