#!/usr/bin/env python3
"""
Comprehensive Test for Core Tools System Integration
Tests the complete Core Tools system including registry, processor, and context builder
"""

import sys
import os
import json
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import Core Tools system components
from src.tools.registry import ToolRegistry, register_tool, get_tool_registry
from src.llm.simple_llm_processor import SimpleLLMProcessor
from src.context.complete_context_builder import CompleteContextBuilder
from src.core.models import Claim, ClaimType
from src.interfaces.llm_interface import LLMInterface


class MockLLM(LLMInterface):
    """Mock LLM for testing purposes"""
    
    def generate_response(self, prompt: str) -> str:
        """Generate mock response with tool calls"""
        print(f"LLM received prompt ({len(prompt)} chars) - returning mock tool calls")
        
        # Mock tool calls that we can test
        mock_tool_calls = [
            {
                'name': 'Reason',
                'arguments': {'thought_process': 'Testing the Core Tools system integration'},
                'call_id': 'test_reason_1'
            },
            {
                'name': 'ClaimCreate',
                'arguments': {'content': 'Core Tools system is working correctly', 'confidence': 0.9},
                'call_id': 'test_claim_1'
            },
            {
                'name': 'TellUser',
                'arguments': {'message': 'Core Tools test completed successfully', 'message_type': 'success'},
                'call_id': 'test_message_1'
            }
        ]
        
        return json.dumps({'tool_calls': mock_tool_calls}, indent=2)


def test_tool_registry():
    """Test the ToolRegistry system"""
    print("=" * 60)
    print("Testing ToolRegistry System")
    print("=" * 60)
    
    # Create registry
    registry = ToolRegistry()
    
    # Test core tools count
    core_tools_count = len(registry.core_tools)
    optional_tools_count = len(registry.optional_tools)
    
    print(f"[OK] Registry initialized with {core_tools_count} core tools and {optional_tools_count} optional tools")
    
    # Test Core Tools generation
    tools_context = registry.get_core_tools_context()
    print(f"[OK] Core Tools context generated ({len(tools_context)} characters)")
    
    # Test available tools list
    available_tools = registry.get_available_tools_list()
    print(f"[OK] Available tools list: {available_tools}")
    
    # Test executing a core tool
    result = registry.execute_tool('Reason', {'thought_process': 'Testing tool execution'})
    print(f"[OK] Reason tool execution: {'Success' if result['success'] else 'Failed'}")
    if not result['success']:
        print(f"  Error: {result.get('error', 'Unknown error')}")
    
    # Test executing claim tool
    result = registry.execute_tool('ClaimCreate', {'content': 'Test claim', 'confidence': 0.8})
    print(f"[OK] ClaimCreate tool execution: {'Success' if result['success'] else 'Failed'}")
    if result['success']:
        print(f"  Created claim ID: {result['result']['claim_id']}")
    
    return True


def test_simple_llm_processor():
    """Test the SimpleLLMProcessor"""
    print("\n" + "=" * 60)
    print("Testing SimpleLLMProcessor")
    print("=" * 60)
    
    # Create processor with mock LLM
    processor = SimpleLLMProcessor(MockLLM())
    
    # Test getting available tools
    tools_info = processor.get_available_tools()
    print(f"[OK] Processor recognizes {tools_info['total_tools']} tools")
    print(f"  Core tools: {tools_info['total_core_tools']}")
    print(f"  Optional tools: {tools_info['total_optional_tools']}")
    
    # Test tool validation
    validation = processor.validate_tool_call('WebSearch', {'query': 'test'})
    print(f"[OK] Tool validation: {'Valid' if validation['valid'] else 'Invalid'}")
    
    # Test processing a request
    sample_context = """# Core Tools
**ClaimCreate(content, confidence=0.8, tags=None)**: Create new claims
**WebSearch(query, max_results=10)**: Search web for information

---

# Instructions
Test the Core Tools system by calling the available tools.
"""
    
    result = processor.process_request(sample_context, request_id='test_001')
    print(f"[OK] Request processing: {'Success' if result['success'] else 'Failed'}")
    
    if result['success']:
        print(f"  Tool calls found: {result['total_tool_calls']}")
        print(f"  Successful executions: {result['successful_calls']}")
        print(f"  Failed executions: {result['failed_calls']}")
        
        # Show tool results
        for tool_result in result['tool_results']:
            print(f"  - {tool_result['tool_name']}: {'[OK]' if tool_result['success'] else '[FAIL]'}")
    
    return result['success']


def test_context_builder():
    """Test the CompleteContextBuilder with Core Tools integration"""
    print("\n" + "=" * 60)
    print("Testing CompleteContextBuilder with Core Tools")
    print("=" * 60)
    
    # Create sample claims for testing
    claims = [
        Claim(id="c0000001", content="Test claim 1", confidence=0.8, type=[ClaimType.CONCEPT], tags=["test"]),
        Claim(id="c0000002", content="Test claim 2", confidence=0.9, type=[ClaimType.CONCEPT], tags=["test", "example"]),
        Claim(id="c0000003", content="Test claim 3", confidence=0.7, type=[ClaimType.EXAMPLE], tags=["example"]),
    ]
    
    # Create context builder
    builder = CompleteContextBuilder(claims, include_core_tools=True)
    
    # Test tools summary
    tools_summary = builder.get_tools_summary()
    print(f"[OK] ContextBuilder tools summary:")
    print(f"  Core tools: {tools_summary['core_tools_count']}")
    print(f"  Optional tools: {tools_summary['optional_tools_count']}")
    print(f"  Tools context length: {tools_summary['tools_context_length']}")
    
    # Test building simple context
    simple_context = builder.build_simple_context(include_core_tools=True)
    print(f"[OK] Simple context built ({len(simple_context)} characters)")
    
    # Verify Core Tools section is present
    if "# Core Tools" in simple_context:
        print("[OK] Core Tools section found in context")
    else:
        print("[FAIL] Core Tools section NOT found in context")
        return False
    
    # Test building complete context for a claim
    try:
        complete_context = builder.build_complete_context("c0000001", max_tokens=4000)
        print(f"[OK] Complete context built for claim c0000001 ({len(complete_context.context_text)} characters)")
        
        # Verify Core Tools are present
        if "# Core Tools" in complete_context.context_text:
            print("[OK] Core Tools included in complete context")
        else:
            print("[FAIL] Core Tools NOT included in complete context")
            return False
            
        # Show metrics
        metrics = complete_context.metrics
        print(f"  Context metrics:")
        print(f"    Tokens used: {metrics.tokens_used}")
        print(f"    Build time: {metrics.build_time_ms:.2f}ms")
        
    except Exception as e:
        print(f"[FAIL] Complete context build failed: {e}")
        return False
    
    return True


def test_end_to_end_integration():
    """Test complete end-to-end integration"""
    print("\n" + "=" * 60)
    print("Testing End-to-End Integration")
    print("=" * 60)
    
    # 1. Create registry and verify tools
    registry = get_tool_registry()
    print("[OK] ToolRegistry initialized")
    
    # 2. Create context with Core Tools
    context_parts = [
        "# Core Tools Integration Test",
        "You are testing the Core Tools system.",
        "Use the Reason tool to record your thought process.",
        "Use ClaimCreate to create test claims.",
        "Use TellUser to send completion messages.",
        "",
        "# Instructions",
        "Respond with only JSON tool calls.",
        ""
    ]
    
    context = "\n".join(context_parts)
    print(f"[OK] Test context prepared ({len(context)} chars)")
    
    # 3. Process with SimpleLLMProcessor
    processor = SimpleLLMProcessor(MockLLM())
    result = processor.process_request(context, request_id='integration_test')
    
    if not result['success']:
        print(f"[FAIL] Integration test failed: {result.get('error', 'Unknown error')}")
        return False
    
    print("[OK] LLM processing successful")
    print(f"[OK] Executed {result['total_tool_calls']} tool calls")
    
    # 4. Verify all expected tools were executed
    executed_tools = [tr['tool_name'] for tr in result['tool_results'] if tr['success']]
    expected_tools = ['Reason', 'ClaimCreate', 'TellUser']
    
    for tool in expected_tools:
        if tool in executed_tools:
            print(f"[OK] {tool} executed successfully")
        else:
            print(f"[FAIL] {tool} NOT executed")
            return False
    
    return True


def main():
    """Run all Core Tools system tests"""
    print("Starting Comprehensive Core Tools System Integration Test")
    print("=" * 80)
    
    tests = [
        ("ToolRegistry System", test_tool_registry),
        ("SimpleLLMProcessor", test_simple_llm_processor),
        ("CompleteContextBuilder", test_context_builder),
        ("End-to-End Integration", test_end_to_end_integration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"[FAIL] {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "[OK] PASS" if success else "[FAIL] FAIL"
        print(f"{test_name:.<40} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] ALL TESTS PASSED! Core Tools system is working correctly.")
        return 0
    else:
        print("[FAIL] Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)