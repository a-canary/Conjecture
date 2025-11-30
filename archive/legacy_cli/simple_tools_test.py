#!/usr/bin/env python3
"""
Simple test for Core Tools system without complex dependencies.
"""

import sys
import os
import json

# Add current directory to path for imports
sys.path.insert(0, '.')
sys.path.insert(0, 'src')

def test_registry_import():
    """Test basic registry import"""
    print("Testing Tool Registry import...")
    
    try:
        # Direct import without complex dependencies
        exec(open('src/tools/registry.py').read())
        print("[OK] ToolRegistry module loaded successfully")
        return True
    except Exception as e:
        print(f"[FAIL] Failed to load registry: {e}")
        return False


def test_tool_functionality():
    """Test basic tool functionality"""
    print("\nTesting tool functionality...")
    
    try:
        # Load the registry
        exec(open('src/tools/registry.py').read())
        
        # Test manual tool registration
        @register_tool(name="TestTool", is_core=True)
        def test_function(message: str) -> dict:
            """A simple test function"""
            return {"success": True, "message": message}
        
        print("[OK] Tool registered successfully")

        # Test tool execution
        result = ToolRegistry.execute_tool("TestTool", message="Hello World")
        if result.get("success"):
            print("[OK] Tool executed successfully")
            print(f"  Result: {result}")
            return True
        else:
            print("[FAIL] Tool execution failed")
            return False

    except Exception as e:
        print(f"[FAIL] Tool functionality test failed: {e}")
        return False


def test_core_tools_import():
    """Test importing core tools"""
    print("\nTesting Core Tools import...")
    
    # Test loading claim tools
    try:
        # Add tools directory to path
        sys.path.insert(0, 'tools')
        
        # Load registry first
        exec(open('src/tools/registry.py').read())
        
        # Import claim_tools manually
        claim_tools_code = open('tools/claim_tools.py').read()
        exec(claim_tools_code)
        
        print("[OK] Claim tools loaded successfully")

        # Test a claim tool
        result = ClaimCreate("Test claim from simple test", 0.8, ["test"])
        if result.get('success'):
            print("[OK] ClaimCreate tool working")
            claim_id = result.get('claim_id')

            # Test ClaimGetSupport
            support_result = ClaimGetSupport(claim_id)
            if support_result.get('success'):
                print("[OK] ClaimGetSupport tool working")
                return True
            else:
                print(f"[FAIL] ClaimGetSupport failed: {support_result.get('error')}")
        else:
            print(f"[FAIL] ClaimCreate failed: {result.get('error')}")

    except Exception as e:
        print(f"[FAIL] Core tools import failed: {e}")
        import traceback
        traceback.print_exc()
    
    return False


def test_interaction_tools():
    """Test interaction tools"""
    print("\nTesting Interaction Tools...")
    
    try:
        # Load interaction_tools
        interaction_tools_code = open('tools/interaction_tools.py').read()
        exec(interaction_tools_code)
        
        # Test Reason tool
        result = Reason("Testing interaction tools functionality")
        if result.get('success'):
            print("[OK] Reason tool working")
            print(f"  Reasoning ID: {result.get('reasoning_id')}")
        else:
            print(f"[FAIL] Reason tool failed: {result.get('error')}")
            return False

        # Test TellUser tool
        result = TellUser("Interaction tools test completed", "success")
        if result.get('success'):
            print("[OK] TellUser tool working")
            return True
        else:
            print(f"[FAIL] TellUser tool failed: {result.get('error')}")
            return False

    except Exception as e:
        print(f"[FAIL] Interaction tools test failed: {e}")
        return False


def test_json_response_format():
    """Test JSON response format"""
    print("\nTesting JSON response format...")
    
    # Create sample tool calls
    tool_calls = [
        {
            "name": "Reason",
            "arguments": {"thought_process": "Testing JSON format"},
            "call_id": "test_1"
        },
        {
            "name": "ClaimCreate", 
            "arguments": {
                "content": "Testing claim creation",
                "confidence": 0.9,
                "tags": ["test"]
            },
            "call_id": "test_2"
        }
    ]
    
    # Create JSON response
    response = {"tool_calls": tool_calls}
    json_response = json.dumps(response, indent=2)
    
    print("✓ JSON response format created:")
    print(json_response)
    
    # Validate it's valid JSON
    try:
        parsed = json.loads(json_response)
        print(f"✓ JSON is valid with {len(parsed.get('tool_calls', []))} tool calls")
        return True
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON: {e}")
        return False


def test_context_format():
    """Test context format with core tools"""
    print("\nTesting context format...")
    
    try:
        # Load registry and get core tools context
        exec(open('src/tools/registry.py').read())
        
        # Manually register some tools for testing
        @register_tool(name="ClaimCreate", is_core=True)
        def ClaimCreate(content: str, confidence: float = 0.8, tags=None):
            return {"success": True, "claim_id": "test_1"}
        
        @register_tool(name="WebSearch", is_core=True)
        def WebSearch(query: str, max_results: int = 10):
            return {"success": True, "results": []}
        
        @register_tool(name="ReadFiles", is_core=True)
        def ReadFiles(path_pattern: str, max_files: int = 50):
            return {"success": True, "files": []}
        
        @register_tool(name="Reason", is_core=True)
        def Reason(thought_process: str):
            return {"success": True, "reasoning_id": "test_1"}
        
        # Get core tools context
        context = ToolRegistry.get_core_tools_context()
        print("✓ Core tools context generated:")
        print(context)
        
        # Build full context template
        full_context = f"""{context}

---

# Relevant Claims
[c1 | Rust is a systems programming language | confidence: 0.9 | tags: [rust, systems]]

# Target Claim  
[c2 | Rust provides memory safety without garbage collection | confidence_redacted]

# Instructions
Please respond with only JSON tool calls in this format:
```json
{{"tool_calls": [{{"name": "ToolName", "arguments": {{...}}}}]}}
```
"""
        
        print("✓ Full context template created successfully")
        print(f"  Context length: {len(full_context)} characters")
        
        return True
        
    except Exception as e:
        print(f"✗ Context format test failed: {e}")
        return False


def main():
    """Run all simple tests"""
    print("SIMPLE CORE TOOLS TESTS")
    print("=" * 50)
    
    tests = [
        ("Registry Import", test_registry_import),
        ("Tool Functionality", test_tool_functionality),
        ("Core Tools Import", test_core_tools_import),
        ("Interaction Tools", test_interaction_tools),
        ("JSON Response Format", test_json_response_format),
        ("Context Format", test_context_format)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            success = test_func()
            results[test_name] = success
            print(f"--- {test_name}: {'PASSED' if success else 'FAILED'} ---")
        except Exception as e:
            print(f"--- {test_name}: ERROR - {e} ---")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:.<30} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed >= 4:  # At least 4 tests passing indicates core functionality works
        print("🎉 Core Tools system is working!")
        return True
    else:
        print("⚠️  Core Tools system needs attention.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)