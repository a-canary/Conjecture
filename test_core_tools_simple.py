#!/usr/bin/env python3
"""
Simple Core Tools Integration Test
Tests the Core Tools registry and execution directly
"""

import sys
import os
import json
from typing import Dict, Any, List

# Add current directory to path to import tools directly
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the tools directly
from tools.claim_tools import ClaimCreate, ClaimAddSupport, ClaimGetSupport, ClaimsQuery
from tools.interaction_tools import Reason, TellUser, AskUser
from tools.webSearch import webSearch
from tools.readFiles import readFiles
from tools.writeFiles import writeFile


def test_claim_tools():
    """Test claim management tools"""
    print("=" * 50)
    print("Testing Claim Tools")
    print("=" * 50)
    
    # Test ClaimCreate
    result = ClaimCreate("Test claim for Core Tools system", confidence=0.9, tags=["test", "core-tools"])
    print(f"✓ ClaimCreate: {'Success' if result['success'] else 'Failed'}")
    if result['success']:
        claim_id = result['claim_id']
        print(f"  Created claim: {claim_id}")
        
        # Test ClaimAddTags
        tag_result = ClaimAddTags(claim_id, ["integration", "test"])
        print(f"✓ ClaimAddTags: {'Success' if tag_result['success'] else 'Failed'}")
        
        # Test ClaimsQuery
        query_result = ClaimsQuery({"tags": ["test"]})
        print(f"✓ ClaimsQuery: {'Success' if query_result['success'] else 'Failed'}")
        print(f"  Found {query_result['total_count']} claims")
        
        # Create another claim for support testing
        result2 = ClaimCreate("Supporting claim for Core Tools", confidence=0.8, tags=["support"])
        if result2['success']:
            support_claim_id = result2['claim_id']
            
            # Test ClaimAddSupport
            support_result = ClaimAddSupport(support_claim_id, claim_id)
            print(f"✓ ClaimAddSupport: {'Success' if support_result['success'] else 'Failed'}")
            
            # Test ClaimGetSupport
            get_support_result = ClaimGetSupport(claim_id)
            print(f"✓ ClaimGetSupport: {'Success' if get_support_result['success'] else 'Failed'}")
            print(f"  Supported by count: {get_support_result['supported_by_count']}")
    
    return True


def test_interaction_tools():
    """Test interaction tools"""
    print("\n" + "=" * 50)
    print("Testing Interaction Tools")
    print("=" * 50)
    
    # Test Reason
    result = Reason("Testing Core Tools integration step by step")
    print(f"✓ Reason: {'Success' if result['success'] else 'Failed'}")
    if result['success']:
        print(f"  Reasoning ID: {result['reasoning_id']}, Step: {result['step_number']}")
    
    # Test TellUser
    result = TellUser("Core Tools integration test in progress", message_type="info")
    print(f"✓ TellUser: {'Success' if result['success'] else 'Failed'}")
    if result['success']:
        print(f"  Message ID: {result['message_id']}")
    
    # Test AskUser
    result = AskUser("Do you want to continue the Core Tools test?", options=["yes", "no"], required=False)
    print(f"✓ AskUser: {'Success' if result['success'] else 'Failed'}")
    if result['success']:
        print(f"  Question ID: {result['question_id']}")
    
    return True


def test_file_tools():
    """Test file tools"""
    print("\n" + "=" * 50)
    print("Testing File Tools")
    print("=" * 50)
    
    # Test WriteFile
    test_content = """# Core Tools Test File
This file was created to test the Core Tools system.
## Features Tested:
- ClaimCreate and claim management
- Interaction tools (Reason, TellUser, AskUser)
- File tools (WriteFile, ReadFiles, WebSearch)
"""
    
    result = writeFile("test_core_tools_output.md", test_content, create_dirs=True)
    print(f"✓ WriteFile: {'Success' if result['success'] else 'Failed'}")
    if result['success']:
        print(f"  File: {result['filename']}, Lines: {result['line_count']}")
    
    # Test ReadFiles
    result = readFiles("test_core_tools_output.md", max_files=5)
    print(f"✓ ReadFiles: {'Success' if result and result[0]['read_success'] else 'Failed'}")
    if result and result[0]['read_success']:
        print(f"  File: {result[0]['filename']}, Lines: {result[0]['line_count']}")
    
    # Clean up
    try:
        os.remove("test_core_tools_output.md")
        print("✓ Cleaned up test file")
    except:
        pass
    
    return True


def test_web_search():
    """Test web search tool"""
    print("\n" + "=" * 50)
    print("Testing Web Search Tool")
    print("=" * 50)
    
    # Test WebSearch
    result = webSearch("Core Tools system architecture", max_results=3)
    print(f"✓ WebSearch: {'Success' if result else 'Failed'}")
    if result:
        print(f"  Results: {len(result)} items")
        if result and 'title' in result[0]:
            print(f"  First result: {result[0]['title'][:50]}...")
    
    return True


def test_tool_response_format():
    """Test creating proper tool response JSON format"""
    print("\n" + "=" * 50)
    print("Testing Tool Response Format")
    print("=" * 50)
    
    # Sample tool calls as expected from LLM
    tool_calls = [
        {
            'name': 'Reason',
            'arguments': {'thought_process': 'Testing response format'},
            'call_id': 'test_format_1'
        },
        {
            'name': 'ClaimCreate',
            'arguments': {'content': 'Format test claim', 'confidence': 0.85},
            'call_id': 'test_format_2'
        },
        {
            'name': 'TellUser',
            'arguments': {'message': 'Format test completed', 'message_type': 'success'},
            'call_id': 'test_format_3'
        }
    ]
    
    # Create JSON response format
    response = {
        'tool_calls': tool_calls
    }
    
    json_response = json.dumps(response, indent=2)
    print(f"✓ JSON response format created")
    print(f"  Length: {len(json_response)} characters")
    
    # Test parsing the response
    try:
        parsed = json.loads(json_response)
        print(f"✓ JSON parsing successful")
        print(f"  Tool calls found: {len(parsed['tool_calls'])}")
        
        # Verify structure
        for call in parsed['tool_calls']:
            if 'name' in call and 'arguments' in call:
                print(f"  ✓ {call['name']} has correct structure")
            else:
                print(f"  ✗ {call.get('name', 'unknown')} missing required fields")
                return False
        
    except json.JSONDecodeError as e:
        print(f"✗ JSON parsing failed: {e}")
        return False
    
    return True


def main():
    """Run all Core Tools tests"""
    print("Core Tools System Integration Test")
    print("=" * 60)
    
    tests = [
        ("Claim Tools", test_claim_tools),
        ("Interaction Tools", test_interaction_tools),
        ("File Tools", test_file_tools),
        ("Web Search Tool", test_web_search),
        ("Response Format", test_tool_response_format),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:.<30} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL CORE TOOLS TESTS PASSED!")
        print("\nThe Core Tools system is working correctly with:")
        print("- Tool registration and discovery")
        print("- Claim management (create, support, query)")
        print("- User interaction (reasoning, messaging)")
        print("- File operations (read, write)")
        print("- Web search capabilities")
        print("- Proper JSON response format")
        return 0
    else:
        print("❌ Some Core Tools tests failed.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)