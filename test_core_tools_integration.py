#!/usr/bin/env python3
"""
Comprehensive test script for the Core Tools system integration.

Tests:
1. Tool Registry functionality
2. Core Tools registration and discovery
3. Simple LLM Processor
4. Context Builder integration
5. End-to-end tool execution
"""

import sys
import os
import json
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import all components
from src.tools.registry import ToolRegistry, register_tool
from src.llm.simple_llm_processor import SimpleLLMProcessor, create_simple_response
from src.context.complete_context_builder import CompleteContextBuilder
from src.core.models import Claim

# Import tools to trigger registration
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))
import claim_tools
import interaction_tools
import file_tools
import webSearch
import readFiles
import writeFiles
import apply_diff


class MockLLM:
    """Mock LLM for testing purposes"""
    
    def __init__(self, response_content: str = None):
        self.response_content = response_content or self._create_default_response()
    
    def generate_response(self, prompt: str) -> str:
        """Generate mock response with tool calls"""
        print(f"\n=== Mock LLM received prompt ({len(prompt)} chars) ===")
        print("First 200 chars of prompt:")
        print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
        print("=" * 50)
        
        return self.response_content
    
    def _create_default_response(self) -> str:
        """Create default tool call response"""
        tool_calls = [
            {
                "name": "Reason",
                "arguments": {"thought_process": "Testing Core Tools integration"},
                "call_id": "test_reason_1"
            },
            {
                "name": "WebSearch",
                "arguments": {"query": "Core Tools Python integration", "max_results": 3},
                "call_id": "test_search_1"
            },
            {
                "name": "ClaimCreate",
                "arguments": {
                    "content": "Core Tools system provides simplified tool execution",
                    "confidence": 0.9,
                    "tags": ["testing", "core-tools"]
                },
                "call_id": "test_claim_1"
            }
        ]
        return json.dumps({"tool_calls": tool_calls})


def test_tool_registry():
    """Test Tool Registry functionality"""
    print("\n" + "="*60)
    print("TESTING TOOL REGISTRY")
    print("="*60)
    
    # Test tool auto-discovery
    ToolRegistry.auto_discover_tools()
    
    # Get all tools
    all_tools = ToolRegistry.get_all_tools()
    core_tools = ToolRegistry.get_core_tools()
    optional_tools = ToolRegistry.get_optional_tools()
    
    print(f"✓ Total tools registered: {len(all_tools)}")
    print(f"✓ Core tools: {len(core_tools)}")
    print(f"✓ Optional tools: {len(optional_tools)}")
    
    # List core tools
    print("\nCore Tools:")
    for name, tool_info in core_tools.items():
        print(f"  - {name}: {tool_info.description}")
    
    # Test tool info
    websearch_tool = ToolRegistry.get_tool("WebSearch")
    if websearch_tool:
        print(f"\n✓ WebSearch tool found:")
        print(f"  - Description: {websearch_tool.description}")
        print(f"  - Parameters: {websearch_tool.parameters}")
        print(f"  - Is core: {websearch_tool.is_core}")
    else:
        print("✗ WebSearch tool not found")
    
    return len(all_tools) > 0 and len(core_tools) > 0


def test_tool_execution():
    """Test individual tool execution"""
    print("\n" + "="*60)
    print("TESTING TOOL EXECUTION")
    print("="*60)
    
    # Test Reason tool
    try:
        result = ToolRegistry.execute_tool("Reason", thought_process="Testing tool execution")
        print(f"✓ Reason tool executed: {result.get('success', False)}")
        if result.get('success'):
            print(f"  - Reasoning ID: {result.get('reasoning_id')}")
    except Exception as e:
        print(f"✗ Reason tool failed: {e}")
    
    # Test ClaimCreate tool
    try:
        claim_result = ToolRegistry.execute_tool(
            "ClaimCreate",
            content="Testing claim creation during integration",
            confidence=0.8,
            tags=["integration", "test"]
        )
        print(f"✓ ClaimCreate tool executed: {claim_result.get('success', False)}")
        if claim_result.get('success'):
            claim_id = claim_result.get('claim_id')
            print(f"  - Claim ID: {claim_id}")
            
            # Test ClaimGetSupport
            support_result = ToolRegistry.execute_tool("ClaimGetSupport", claim_id=claim_id)
            print(f"✓ ClaimGetSupport tool executed: {support_result.get('success', False)}")
        
    except Exception as e:
        print(f"✗ ClaimCreate tool failed: {e}")
    
    return True


def test_simple_llm_processor():
    """Test SimpleLLMProcessor"""
    print("\n" + "="*60)
    print("TESTING SIMPLE LLM PROCESSOR")
    print("="*60)
    
    # Create mock LLM with test response
    mock_llm = MockLLM()
    
    # Create processor
    processor = SimpleLLMProcessor(mock_llm)
    
    # Test getting available tools
    tools_info = processor.get_available_tools()
    print(f"✓ Processor found {tools_info['total_tools']} tools")
    print(f"  - Core tools: {tools_info['total_core_tools']}")
    print(f"  - Optional tools: {tools_info['total_optional_tools']}")
    
    # Test tool validation
    validation = processor.validate_tool_call("WebSearch", {"query": "test"})
    print(f"✓ Tool validation: {validation.get('valid', False)}")
    
    # Test processing request with simple context
    simple_context = """# Test Context
This is a test context for the SimpleLLMProcessor.
Please use the available tools to test the system.
"""
    
    result = processor.process_request(simple_context, request_id="test_001")
    print(f"✓ Request processed: {result.get('success', False)}")
    
    if result.get('success'):
        print(f"  - Tool calls found: {result.get('total_tool_calls', 0)}")
        print(f"  - Successful calls: {result.get('successful_calls', 0)}")
        print(f"  - Failed calls: {result.get('failed_calls', 0)}")
        print(f"  - Execution time: {result.get('execution_time_ms', 0)}ms")
        
        # Show tool results summary
        tool_results = result.get('tool_results', [])
        for tr in tool_results:
            status = "✓" if tr.get('success', False) else "✗"
            print(f"  {status} {tr.get('tool_name', 'Unknown')}: {tr.get('error', 'Success')}")
    
    return result.get('success', False)


def test_context_builder():
    """Test Context Builder integration"""
    print("\n" + "="*60)
    print("TESTING CONTEXT BUILDER")
    print("="*60)
    
    # Create some test claims
    test_claims = [
        Claim(
            id="claim_1",
            content="Core Tools system provides simplified tool execution",
            confidence=0.9,
            tags=["core-tools", "testing"],
            supports=[],
            supported_by=[]
        ),
        Claim(
            id="claim_2", 
            content="Tool Registry manages tool discovery and execution",
            confidence=0.8,
            tags=["registry", "tools"],
            supports=[],
            supported_by=[]
        ),
        Claim(
            id="claim_3",
            content="SimpleLLMProcessor focuses on tool execution complexity",
            confidence=0.85,
            tags=["llm", "processor"],
            supports=[], 
            supported_by=[]
        )
    ]
    
    # Create context builder
    builder = CompleteContextBuilder(test_claims, include_core_tools=True)
    
    # Test tools summary
    tools_summary = builder.get_tools_summary()
    print(f"✓ Context builder tools summary:")
    print(f"  - Core tools: {tools_summary['core_tools_count']}")
    print(f"  - Optional tools: {tools_summary['optional_tools_count']}")
    print(f"  - Tools context length: {tools_summary['tools_context_length']} chars")
    
    # Test simple context building
    simple_context = builder.build_simple_context()
    print(f"✓ Simple context built: {len(simple_context)} characters")
    
    # Show first 500 chars of context
    if len(simple_context) > 500:
        print("Context preview (first 500 chars):")
        print(simple_context[:500] + "...")
    else:
        print("Full context:")
        print(simple_context)
    
    return len(simple_context) > 0


def test_end_to_end_integration():
    """Test complete end-to-end integration"""
    print("\n" + "="*60)
    print("TESTING END-TO-END INTEGRATION") 
    print("="*60)
    
    # Create processor with real tool responses
    mock_llm = MockLLM()
    processor = SimpleLLMProcessor(mock_llm)
    
    # Create context that should trigger tool usage
    context_builder = CompleteContextBuilder([], include_core_tools=True)
    context = context_builder.build_simple_context()
    
    # Add a specific task to the context
    task_section = """
# User Task
Please search for information about "Rust programming language" and create a claim about it.
Then tell the user what you found.
"""
    
    full_context = context + "\n" + task_section
    
    print(f"✓ Built context for end-to-end test: {len(full_context)} chars")
    
    # Process the request
    result = processor.process_request(full_context, request_id="e2e_test")
    
    success = result.get('success', False)
    print(f"✓ End-to-end processing: {success}")
    
    if success:
        tool_calls = result.get('tool_calls', [])
        tool_results = result.get('tool_results', [])
        
        print(f"  - Tool calls executed: {len(tool_calls)}")
        print(f"  - Successful results: {len([tr for tr in tool_results if tr.get('success', False)])}")
        
        # Verify expected tools were called
        called_tools = {tc.get('name') for tc in tool_calls}
        expected_tools = {'Reason', 'WebSearch', 'ClaimCreate'}
        
        print(f"  - Called tools: {called_tools}")
        print(f"  - Expected in call set: {expected_tools.intersection(called_tools)}")
    
    return success


def test_response_formatting():
    """Test JSON response formatting"""
    print("\n" + "="*60) 
    print("TESTING RESPONSE FORMATTING")
    print("="*60)
    
    # Test create_simple_response
    tool_calls = [
        {"name": "Reason", "arguments": {"thought_process": "Testing format"}},
        {"name": "WebSearch", "arguments": {"query": "test", "max_results": 5}}
    ]
    
    json_response = create_simple_response(tool_calls)
    print(f"✓ Created JSON response: {len(json_response)} chars")
    
    # Parse and validate
    try:
        parsed = json.loads(json_response)
        print(f"✓ JSON is valid")
        print(f"  - Tool calls in response: {len(parsed.get('tool_calls', []))}")
        
        for i, call in enumerate(parsed.get('tool_calls', [])):
            print(f"    {i+1}. {call.get('name')} with args {list(call.get('arguments', {}).keys())}")
            
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON: {e}")
        return False
    
    return True


def main():
    """Run all tests"""
    print("STARTING CORE TOOLS INTEGRATION TESTS")
    print("=" * 60)
    
    tests = [
        ("Tool Registry", test_tool_registry),
        ("Tool Execution", test_tool_execution), 
        ("Simple LLM Processor", test_simple_llm_processor),
        ("Context Builder", test_context_builder),
        ("End-to-End Integration", test_end_to_end_integration),
        ("Response Formatting", test_response_formatting)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n--- Running {test_name} ---")
            success = test_func()
            results[test_name] = success
            print(f"--- {test_name}: {'PASSED' if success else 'FAILED'} ---")
        except Exception as e:
            print(f"--- {test_name}: ERROR - {e} ---")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:.<30} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Core Tools system is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)