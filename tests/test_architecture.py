#!/usr/bin/env python3
"""
Direct test of 3-part architecture without CLI Unicode issues
Testing: Claims → LLM Inference → Tools → New Claims
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_three_part_architecture():
    """Test the complete 3-part architecture flow."""
    
    print("=" * 60)
    print("Testing 3-Part Architecture Integration")
    print("=" * 60)
    
    # Test 1: Import core components
    print("\n1. Testing Core Imports...")
    try:
        from core.models import Claim, ClaimState, ClaimType
        from agent.llm_inference import build_llm_context, coordinate_three_part_flow
        from processing.tool_registry import create_tool_registry, ToolCall
        print("   ✓ Core models imported successfully")
        print("   ✓ LLM inference layer imported successfully")
        print("   ✓ Tool registry imported successfully")
    except Exception as e:
        print(f"   X Import failed: {e}")
        return False
    
    # Test 2: Create test claims
    print("\n2. Testing Claims Layer...")
    try:
        from core.models import Claim
        
        claim1 = Claim(
            id="test_claim_1",
            content="Python is a popular programming language",
            confidence=0.9,
            state=ClaimState.VALIDATED,
            type=[ClaimType.CONCEPT],
            tags=["programming", "python"]
        )
        
        claim2 = Claim(
            id="test_claim_2", 
            content="Machine learning requires data",
            confidence=0.8,
            state=ClaimState.EXPLORE,
            type=[ClaimType.CONCEPT],
            tags=["ml", "data"]
        )
        
        all_claims = [claim1, claim2]
        print(f"   ✓ Created {len(all_claims)} test claims")
        print(f"   ✓ Claim 1: {claim1.content[:50]}...")
        print(f"   ✓ Claim 2: {claim2.content[:50]}...")
    except Exception as e:
        print(f"   X Claims layer failed: {e}")
        return False
    
    # Test 3: Setup Tool Registry  
    print("\n3. Testing Tools Layer...")
    try:
        from processing.tool_registry import create_tool_registry
        
        tool_registry = create_tool_registry("test_tools")
        print(f"   ✓ Tool registry created: {len(tool_registry.tools)} tools")
        print(f"   ✓ Tools directory: {tool_registry.tools_directory}")
    except Exception as e:
        print(f"   X Tools layer failed: {e}")
        return False
    
    # Test 4: LLM Inference Layer 
    print("\n4. Testing LLM Inference Layer...")
    try:
        from agent.llm_inference import build_llm_context, find_relevant_claims
        
        # Test context building
        context = build_llm_context(
            session_id="test_session",
            user_request="Tell me about programming languages",
            all_claims=all_claims,
            tool_registry=tool_registry
        )
        
        print(f"   ✓ LLM context created")
        print(f"   ✓ Relevant claims found: {len(context.relevant_claims)}")
        print(f"   ✓ Available tools: {len(context.available_tools)}")
        
        # Test claim relevance finding
        relevant = find_relevant_claims("programming", all_claims)
        print(f"   ✓ Found {len(relevant)} relevant claims for 'programming'")
        
    except Exception as e:
        print(f"   X LLM inference layer failed: {e}")
        return False
    
    # Test 5: Complete Flow Coordination  
    print("\n5. Testing Complete 3-Part Flow...")
    try:
        from agent.llm_inference import coordinate_three_part_flow
        
        result = coordinate_three_part_flow(
            session_id="integration_test",
            user_request="What do you know about Python programming?",
            all_claims=all_claims,
            tool_registry=tool_registry,
            conversation_history=[]
        )
        
        if result['success']:
            print("   ✓ Complete flow executed successfully")
            print(f"   ✓ Context built: {type(result.get('context'))}")
            print(f"   ✓ LLM response generated: {type(result.get('llm_response'))}")
            print(f"   ✓ Processing plan created: {type(result.get('processing_plan'))}")
            print(f"   ✓ Tool results: {len(result.get('tool_results', []))}")
            print(f"   ✓ Updated claims: {len(result.get('updated_claims', []))}")
            print(f"   ✓ New claims: {len(result.get('new_claims', []))}")
        else:
            print(f"   X Flow failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"   X Complete flow failed: {e}")
        return False
    
    # Test 6: CLI Integration (without progress bars)
    print("\n6. Testing CLI Integration...")
    try:
        # Test that CLI components can be imported and initialized
        from cli.backends.auto_backend import AutoBackend
        
        backend = AutoBackend()
        print("   ✓ AutoBackend initialized")
        
        # Check backend info
        if backend.is_available():
            info = backend.get_backend_info()
            print(f"   ✓ Backend available: {info.get('name', 'Unknown')}")
        else:
            print("   ⚠ Backend not configured (expected for test)")
            
    except Exception as e:
        print(f"   X CLI integration failed: {e}")
        return False
    
    # Success!
    print("\n" + "=" * 60)
    print("+ 3-PART ARCHITECTURE INTEGRATION TEST: PASSED")
    print("=" * 60)
    print("+ Claims Layer: Working")
    print("+ LLM Inference Layer: Working") 
    print("+ Tools Layer: Working")
    print("+ Flow Coordination: Working")
    print("+ CLI Integration: Working")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_three_part_architecture()
    sys.exit(0 if success else 1)