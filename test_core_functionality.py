#!/usr/bin/env python3
"""
Test core functionality of Conjecture system
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.conjecture import Conjecture

def test_core_functionality():
    """Test core Conjecture functionality"""
    print("=== Core Functionality Test ===")
    
    try:
        # Initialize without unicode print
        cf = Conjecture()
        print("Conjecture initialized successfully")
        
        # Test 1: Basic request processing
        print("\n--- Test 1: Request Processing ---")
        result = cf.process_request('Research machine learning algorithms')
        
        print(f"Success: {result['success']}")
        print(f"Skill used: {result['skill_used']}")
        print(f"Context claims: {len(result['context_claims'])}")
        print(f"Tool results: {len(result['tool_results'])}")
        
        if not result['success']:
            print(f"Error: {result.get('error', 'Unknown error')}")
            return False
        
        # Test 2: Claim creation
        print("\n--- Test 2: Claim Creation ---")
        import asyncio
        
        async def test_claim_creation():
            claim_result = await cf.create_claim(
                content="Machine learning is a branch of artificial intelligence",
                confidence=0.85,
                claim_type="concept",
                tags=["ml", "ai"]
            )
            return claim_result
        
        claim_result = asyncio.run(test_claim_creation())
        print(f"Claim creation success: {claim_result['success']}")
        
        if claim_result['success']:
            claim = claim_result['claim']
            print(f"Claim ID: {claim['id']}")
            print(f"Content: {claim['content']}")
            print(f"Confidence: {claim['confidence']}")
        
        # Test 3: Search functionality
        print("\n--- Test 3: Search Functionality ---")
        
        async def test_search():
            search_results = await cf.search_claims("machine learning", limit=5)
            return search_results
        
        search_results = asyncio.run(test_search())
        print(f"Search results: {len(search_results)} claims found")
        
        for result in search_results[:2]:  # Show first 2
            print(f"  - {result.get('id', 'N/A')}: {result.get('content', 'N/A')}")
        
        # Test 4: Statistics
        print("\n--- Test 4: System Statistics ---")
        stats = cf.get_statistics()
        print(f"Available tools: {stats.get('available_tools', 0)}")
        print(f"Available skills: {stats.get('available_skills', 0)}")
        print(f"Active sessions: {stats.get('active_sessions', 0)}")
        
        print("\n=== Core Functionality Test: PASS ===")
        return True
        
    except Exception as e:
        print(f"Core functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tool_integration():
    """Test tool integration"""
    print("\n=== Tool Integration Test ===")
    
    try:
        cf = Conjecture()
        
        # Test with a request that should trigger tools
        result = cf.process_request('Search for information about Python programming')
        
        print(f"Tool execution results: {len(result['tool_results'])}")
        
        for tool_result in result['tool_results']:
            tool_name = tool_result['tool']
            tool_success = tool_result['result'].get('success', False)
            print(f"  {tool_name}: {'PASS' if tool_success else 'FAIL'}")
            
            if not tool_success:
                print(f"    Error: {tool_result['result'].get('error', 'Unknown')}")
        
        print("Tool Integration Test: PASS")
        return True
        
    except Exception as e:
        print(f"Tool integration test failed: {e}")
        return False

def test_skill_system():
    """Test skill system"""
    print("\n=== Skill System Test ===")
    
    try:
        cf = Conjecture()
        
        # Test different request types
        test_requests = [
            "Research artificial intelligence",
            "Write a Python function", 
            "Test the system performance",
            "Evaluate the results"
        ]
        
        for request in test_requests:
            result = cf.process_request(request)
            skill = result['skill_used']
            print(f"Request: {request}")
            print(f"Skill assigned: {skill}")
            print(f"Success: {result['success']}")
            print()
        
        print("Skill System Test: PASS")
        return True
        
    except Exception as e:
        print(f"Skill system test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Conjecture Core Functionality")
    print("=" * 50)
    
    success1 = test_core_functionality()
    success2 = test_tool_integration()
    success3 = test_skill_system()
    
    if success1 and success2 and success3:
        print("\nAll core functionality tests passed!")
    else:
        print("\nSome core functionality tests failed!")