#!/usr/bin/env python3
"""
Test script for Conjecture EndPoint App using EndpointManager
Demonstrates proper subprocess management and cleanup
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.endpoint_manager import EndpointManager, TemporaryEndpoint, with_endpoint
from test_endpoint_app import EndPointAppTester

async def test_with_manager():
    """Test using endpoint manager directly"""
    print("Testing with EndpointManager")
    print("=" * 60)
    
    # Create endpoint manager
    manager = EndpointManager(host="127.0.0.1", port=8001, log_level="warning")
    
    try:
        # Start endpoint
        if not await manager.start():
            print("Failed to start endpoint")
            return False
        
        print(f"✅ Endpoint started at {manager.base_url}")
        
        # Run tests
        tester = EndPointAppTester(manager.base_url, manager)
        await tester.run_all_tests()
        
        # Print summary
        passed = sum(1 for result in tester.test_results if result["success"])
        total = len(tester.test_results)
        print(f"\n📊 Test Results: {passed}/{total} passed ({(passed/total)*100:.1f}%)")
        
        return passed == total
        
    finally:
        # Always stop endpoint
        await manager.stop()
        print("🛑 Endpoint stopped")

async def test_with_context_manager():
    """Test using temporary endpoint context manager"""
    print("Testing with TemporaryEndpoint Context Manager")
    print("=" * 60)
    
    async with TemporaryEndpoint(host="127.0.0.1", port=8002, log_level="warning") as manager:
        print(f"Endpoint started at {manager.base_url}")
        
        # Run tests
        tester = EndPointAppTester(manager.base_url, manager)
        await tester.run_all_tests()
        
        # Print summary
        passed = sum(1 for result in tester.test_results if result["success"])
        total = len(tester.test_results)
        print(f"\nTest Results: {passed}/{total} passed ({(passed/total)*100:.1f}%)")
        
        return passed == total
    
    # Endpoint automatically stopped when exiting context

async def test_specific_functionality(manager: EndpointManager):
    """Example of testing specific functionality with endpoint"""
    print(f"Testing specific functionality with endpoint at {manager.base_url}")
    
    # Test health check
    if await manager.health_check():
        print("Health check passed")
    else:
        print("Health check failed")
        return False
    
    # Test process info
    info = manager.get_process_info()
    print(f"Process info: {info}")
    
    # Test basic API calls
    tester = EndPointAppTester(manager.base_url, manager)
    tester.test_health_check()
    tester.test_config_info()
    tester.test_session_creation()
    
    passed = sum(1 for result in tester.test_results if result["success"])
    total = len(tester.test_results)
    print(f"\nTest Results: {passed}/{total} passed ({(passed/total)*100:.1f}%)")
    
    return passed == total

async def main():
    """Main test execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Conjecture EndPoint App with EndpointManager")
    parser.add_argument("--mode", choices=["manager", "context", "function", "all"], 
                       default="all", help="Test mode to run")
    parser.add_argument("--host", default="127.0.0.1", help="Endpoint host")
    parser.add_argument("--port", type=int, default=8001, help="Endpoint port")
    
    args = parser.parse_args()
    
    success_count = 0
    total_tests = 0
    
    if args.mode in ["manager", "all"]:
        total_tests += 1
        print("\n" + "="*80)
        print("MODE: Direct EndpointManager")
        print("="*80)
        if await test_with_manager():
            success_count += 1
    
    if args.mode in ["context", "all"]:
        total_tests += 1
        print("\n" + "="*80)
        print("MODE: TemporaryEndpoint Context Manager")
        print("="*80)
        if await test_with_context_manager():
            success_count += 1
    
    if args.mode in ["function", "all"]:
        total_tests += 1
        print("\n" + "="*80)
        print("MODE: Function with Endpoint")
        print("="*80)
        
        # Use the utility function
        result = await with_endpoint(test_specific_functionality, args.host, args.port)
        if result:
            success_count += 1
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Successful test modes: {success_count}/{total_tests}")
    print(f"Success Rate: {(success_count/total_tests)*100:.1f}%")
    
    if success_count == total_tests:
        print("All test modes passed!")
        return 0
    else:
        print("Some test modes failed")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)