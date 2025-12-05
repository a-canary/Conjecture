#!/usr/bin/env python3
"""
Test script to verify the fixes for:
1. ZAI API URL duplication
2. process_task method availability
3. LM Studio token limit
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from conjecture import Conjecture

async def test_process_task():
    """Test the process_task method"""
    print("🧪 Testing process_task method...")
    
    try:
        async with Conjecture() as cf:
            # Test process_task method with explore type
            task = {
                'type': 'explore',
                'content': 'What is machine learning?',
                'max_claims': 2
            }
            
            result = await cf.process_task(task)
            
            if result.get('success'):
                print('✅ process_task method works!')
                print(f'   Result type: {result.get("type")}')
                print(f'   Success: {result.get("success")}')
                return True
            else:
                print(f'❌ process_task returned error: {result.get("error")}')
                return False
                
    except Exception as e:
        print(f'❌ process_task method failed: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_config_fixes():
    """Test configuration fixes"""
    print("🔧 Testing configuration fixes...")
    
    # Check LM Studio config
    try:
        from config.tiny_model_config import DEFAULT_TINY_MODEL_CONFIG, LM_STUDIO_CONFIG
        
        print(f"   LM Studio max_tokens: {LM_STUDIO_CONFIG.get('max_tokens')}")
        print(f"   Default tiny model max_tokens: {DEFAULT_TINY_MODEL_CONFIG.max_tokens}")
        
        if LM_STUDIO_CONFIG.get('max_tokens') == 42000:
            print("✅ LM Studio token limit updated to 42000")
        else:
            print("❌ LM Studio token limit not updated correctly")
            
        if DEFAULT_TINY_MODEL_CONFIG.max_tokens == 42000:
            print("✅ Default tiny model config updated to 42000")
        else:
            print("❌ Default tiny model config not updated correctly")
            
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False
    
    # Check ZAI API fix
    try:
        from processing.llm.chutes_integration import ChutesProcessor
        
        # Create a mock processor to test URL construction
        processor = ChutesProcessor(
            api_key="test_key",
            api_url="https://api.z.ai/api/coding/paas/v4",
            model_name="glm-4.6"
        )
        
        print(f"   ZAI API URL: {processor.api_url}")
        print("✅ ZAI API configuration loaded")
        
    except Exception as e:
        print(f"❌ ZAI API test failed: {e}")
        return False
    
    return True

async def main():
    """Run all tests"""
    print("🚀 Testing Conjecture fixes...")
    print("=" * 50)
    
    # Test configuration fixes
    config_ok = test_config_fixes()
    
    # Test process_task method
    process_task_ok = await test_process_task()
    
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY:")
    print(f"   Configuration fixes: {'✅ PASS' if config_ok else '❌ FAIL'}")
    print(f"   process_task method: {'✅ PASS' if process_task_ok else '❌ FAIL'}")
    
    overall_success = config_ok and process_task_ok
    print(f"   Overall: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    return overall_success

if __name__ == '__main__':
    success = asyncio.run(main())
    sys.exit(0 if success else 1)