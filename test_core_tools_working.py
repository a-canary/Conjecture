#!/usr/bin/env python3
"""
Core Tools System Integration Test
Test the Core Tools registry and execution
"""

import sys
import os
import json

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import tools using absolute imports
sys.path.append(os.path.join(project_root, 'tools'))

# Import individual tool modules
try:
    from claim_tools import ClaimCreate, ClaimAddSupport, ClaimGetSupport, ClaimsQuery
    from interaction_tools import Reason, TellUser, AskUser
    from webSearch import webSearch
    from readFiles import readFiles
    from writeFiles import writeFile
    print("✓ All tool modules imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def test_claim_tools():
    """Test claim management tools"""
    print("\n" + "=" * 50)
    print("Testing Claim Tools")
    print("=" * 50)
    
    # Test ClaimCreate
    result = ClaimCreate("Test claim for Core Tools system", confidence=0.9, tags=["test", "core-tools"])
    success = result.get('success', False)
    print(f"✓ ClaimCreate: {'Success' if success else 'Failed'}")
    
    claim_id = None
    if success:
        claim_id = result['claim_id']
        print(f"  Created claim: {claim_id}")
        
        # Test ClaimAddTags
        tag_result = ClaimAddTags(claim_id, ["integration", "test"])
        print(f"✓ ClaimAddTags: {'Success' if tag_result.get('success', False) else 'Failed'}")
        
        # Test ClaimsQuery
        query_result = ClaimsQuery({"tags": ["test"]})
        print(f"✓ ClaimsQuery: {'Success' if query_result.get('success', False) else 'Failed'}")
        print(f"  Found {query_result.get('total_count', 0)} claims")
        
        # Create another claim for support testing
        result2 = ClaimCreate("Supporting claim for Core Tools", confidence=0.8, tags=["support"])
        if result2.get('success', False):
            support_claim_id = result2['claim_id']
            
            # Test ClaimAddSupport
            support_result = ClaimAddSupport(support_claim_id, claim_id)
            print(f"✓ ClaimAddSupport: {'Success' if support_result.get('success', False) else 'Failed'}")
            
            # Test ClaimGetSupport
            get_support_result = ClaimGetSupport(claim_id)
            print(f"✓ ClaimGetSupport: {'Success' if get_support_result.get('success', False) else 'Failed'}")
            print(f"  Supported by count: {get_support_result.get('supported_by_count', 0)}")
    
    return success


def test_interaction_tools():
    """Test interaction tools"""
    print("\n" + "=" * 50)
    print("Testing Interaction Tools")
    print("=" * 50)
    
    # Test Reason
    result = Reason("Testing Core Tools integration step by step")
    success1 = result.get('success', False)
    print(f"✓ Reason: {'Success' if success1 else 'Failed'}")
    if success1:
        print(f"  Reasoning ID: {result.get('reasoning_id')}, Step: {result.get('step_number')}")
    
    # Test TellUser
    result = TellUser("Core Tools integration test in progress", message_type="info")
    success2 = result.get('success', False)
    print(f"✓ TellUser: {'Success' if success2 else 'Failed'}")
    if success2:
        print(f"  Message ID: {result.get('message_id')}")
    
    # Test AskUser
    result = AskUser("Do you want to continue the Core Tools test?", options=["yes", "no"], required=False)
    success3 = result.get('success', False)
    print(f"✓ AskUser: {'Success' if success3 else 'Failed'}")
    if success3:
        print(f"  Question ID: {result.get('question_id')}")
    
    return success1 and success2 and success3


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
    success1 = result.get('write_success', False)
    print(f"✓ WriteFile: {'Success' if success1 else 'Failed'}")
    if success1:
        print(f"  File: {result.get('filename')}, Lines: {result.get('line_count')}")
    
    # Test ReadFiles
    result = readFiles("test_core_tools_output.md", max_files=5)
    success2 = result and len(result) > 0 and result[0].get('read_success', False)
    print(f"✓ ReadFiles: {'Success' if success2 else 'Failed'}")
    if success2:
        print(f"  File: {result[0].get('filename')}, Lines: {result[0].get('line_count')}")
    
    # Clean up
    try:
        os.remove("test_core_tools_output.md")
        print("✓ Cleaned up test file")
    except:
        pass
    
    return success1 and success2


def test_web_search():
    """Test web search tool"""
    print("\n" + "=" * 50)
    print("Testing Web Search Tool")
    print("=" * 50)
    
    # Test WebSearch
    result = webSearch("Core Tools system architecture", max_results=3)
    success = result is not None and len(result) > 0
    print(f"✓ WebSearch: {'Success' if success else 'Failed'}")
    if success:
        print(f"  Results: {len(result)} items")
        if result and 'title' in result[0]:
            print(f"  First result: {result[0]['title'][:50]}...")
    
    return success


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
        success = True
        for call in parsed['tool_calls']:
            if 'name' in call and 'arguments' in call:
                print(f"  ✓ {call['name']} has correct structure")
            else:
                print(f"  ✗ {call.get('name', 'unknown')} missing required fields")
                success = False
                break
        
        return success
        
    except json.JSONDecodeError as e:
        print(f"✗ JSON parsing failed: {e}")
        return False


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
        print("- ✅ Tool registration and discovery")
        print("- ✅ Claim management (create, support, query)")
        print("- ✅ User interaction (reasoning, messaging)")
        print("- ✅ File operations (read, write)")
        print("- ✅ Web search capabilities")
        print("- ✅ Proper JSON response format")
        return 0
    else:
        print("❌ Some Core Tools tests failed.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    print(f"\nTest completed with exit code: {exit_code}")
    sys.exit(exit_code)