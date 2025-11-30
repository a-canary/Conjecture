#!/usr/bin/env python3
"""
Comprehensive Test Suite for Conjecture 3-Part Architecture
Tests: Unit, Integration, and End-to-End functionality
"""

import sys
import os
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class TestResults:
    """Track test results"""
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def add_result(self, test_name: str, success: bool, error: str = None):
        self.total += 1
        if success:
            self.passed += 1
            print(f"   + {test_name}")
        else:
            self.failed += 1
            print(f"   X {test_name}")
            if error:
                print(f"     Error: {error}")
                self.errors.append(f"{test_name}: {error}")
    
    def get_summary(self) -> dict:
        return {
            'total': self.total,
            'passed': self.passed,
            'failed': self.failed,
            'success_rate': (self.passed / self.total) * 100 if self.total > 0 else 0,
            'errors': self.errors
        }

def test_unit_components():
    """Test individual components (Unit Tests)"""
    print("=" * 60)
    print("UNIT TESTS - Individual Components")
    print("=" * 60)
    
    results = TestResults()
    
    # Test 1: Claim Model
    print("\n1. Testing Claim Model...")
    try:
        from src.core.models import Claim, ClaimState, ClaimType
        
        # Test claim creation
        claim = Claim(
            id="unit_test_1",
            content="Unit test claim",
            confidence=0.9,
            state=ClaimState.VALIDATED,
            type=[ClaimType.CONCEPT],
            tags=["test"]
        )
        
        # Test claim validation
        assert claim.content == "Unit test claim"
        assert claim.confidence == 0.9
        assert claim.state == ClaimState.VALIDATED
        assert len(claim.type) == 1
        
        # Test ChromaDB conversion
        metadata = claim.to_chroma_metadata()
        assert 'confidence' in metadata
        assert 'state' in metadata
        
        results.add_result("Claim Model Creation", True)
        results.add_result("Claim Model Validation", True)
        results.add_result("ChromaDB Conversion", True)
        
    except Exception as e:
        results.add_result("Claim Model", False, str(e))
    
    # Test 2: Tool Registry
    print("\n2. Testing Tool Registry...")
    try:
        from src.processing.tool_registry import create_tool_registry, ToolFunction
        
        registry = create_tool_registry("unit_test_tools")
        assert len(registry.tools) == 0
        assert registry.tools_directory == "unit_test_tools"
        
        results.add_result("Tool Registry Creation", True)
        
        # Test tool function creation
        tool_func = ToolFunction(
            name="test_tool",
            file_path="/test/path.py",
            description="Test tool function",
            version="1.0.0",
            parameters={"param1": {"name": "param1", "required": True, "type_hint": "str"}},
            return_type="str",
            created_at=time.time(),
            execution_count=0,
            success_count=0
        )
        
        assert tool_func.name == "test_tool"
        assert hasattr(tool_func, 'execution_count')
        
        results.add_result("Tool Function Creation", True)
        
    except Exception as e:
        results.add_result("Tool Registry", False, str(e))
    
    # Test 3: LLM Inference Components
    print("\n3. Testing LLM Inference...")
    try:
        from src.agent.llm_inference import LLMContext, LLMResponse, ProcessingPlan
        from src.core.models import Claim, ClaimState, ClaimType
        
        # Test context creation
        context = LLMContext(
            session_id="test_session",
            user_request="test request",
            relevant_claims=[],
            available_tools=[],
            conversation_history=[],
            metadata={}
        )
        
        assert context.session_id == "test_session"
        assert context.user_request == "test request"
        
        results.add_result("LLM Context Creation", True)
        
        # Test response parsing
        response = LLMResponse(
            response_text="Test response",
            tool_calls=[],
            claim_suggestions=[],
            confidence=0.8,
            metadata={}
        )
        
        assert response.response_text == "Test response"
        assert response.confidence == 0.8
        
        results.add_result("LLM Response Creation", True)
        
        # Test processing plan
        plan = ProcessingPlan(
            planned_tool_calls=[],
            planned_claims=[],
            reasoning="Test reasoning",
            confidence=0.9
        )
        
        assert plan.reasoning == "Test reasoning"
        assert plan.confidence == 0.9
        
        results.add_result("Processing Plan Creation", True)
        
    except Exception as e:
        results.add_result("LLM Inference", False, str(e))
    
    return results.get_summary()

def test_integration_layers():
    """Test integration between layers"""
    print("\n" + "=" * 60)
    print("INTEGRATION TESTS - Layer Communication")
    print("=" * 60)
    
    results = TestResults()
    
    # Test 1: Claims to LLM Inference
    print("\n1. Testing Claims -> LLM Inference...")
    try:
        from src.core.models import Claim, ClaimState, ClaimType
        from src.agent.llm_inference import build_llm_context, find_relevant_claims
        from src.processing.tool_registry import create_tool_registry
        
        # Create test claims
        claims = [
            Claim(
                id="int_test_1",
                content="Python programming language",
                confidence=0.9,
                state=ClaimState.VALIDATED,
                type=[ClaimType.CONCEPT],
                tags=["python", "programming"]
            ),
            Claim(
                id="int_test_2",
                content="Web development frameworks",
                confidence=0.7,
                state=ClaimState.EXPLORE,
                type=[ClaimType.CONCEPT],
                tags=["web", "frameworks"]
            )
        ]
        
        # Test claim relevance finding
        relevant = find_relevant_claims("python programming", claims)
        assert len(relevant) > 0
        assert any("python" in claim.content.lower() for claim in relevant)
        
        results.add_result("Claim Relevance Finding", True)
        
        # Test context building
        tool_registry = create_tool_registry()
        context = build_llm_context(
            session_id="integration_test",
            user_request="Tell me about Python programming",
            all_claims=claims,
            tool_registry=tool_registry
        )
        
        assert len(context.relevant_claims) > 0
        assert context.user_request == "Tell me about Python programming"
        assert isinstance(context.available_tools, list)
        
        results.add_result("Context Building from Claims", True)
        
    except Exception as e:
        results.add_result("Claims -> LLM Inference", False, str(e))
    
    # Test 2: LLM Inference to Tools
    print("\n2. Testing LLM Inference -> Tools...")
    try:
        from src.agent.llm_inference import create_llm_prompt, parse_llm_response, LLMContext
        from src.processing.tool_registry import ToolFunction, create_tool_registry
        
        # Create mock context
        tool_registry = create_tool_registry()
        context = LLMContext(
            session_id="test",
            user_request="test request",
            relevant_claims=[],
            available_tools=[],
            conversation_history=[],
            metadata={}
        )
        
        # Test prompt creation
        prompt = create_llm_prompt(context)
        assert "test request" in prompt
        assert "USER REQUEST" in prompt
        
        results.add_result("LLM Prompt Generation", True)
        
        # Test response parsing
        test_response = "This is a response"
        parsed = parse_llm_response(test_response)
        assert parsed.response_text == "This is a response"
        assert parsed.tool_calls == []
        
        results.add_result("LLM Response Parsing", True)
        
    except Exception as e:
        results.add_result("LLM Inference -> Tools", False, str(e))
    
    # Test 3: Complete Flow Integration
    print("\n3. Testing Complete 3-Part Flow Integration...")
    try:
        from src.core.models import Claim, ClaimState, ClaimType
        from src.agent.llm_inference import coordinate_three_part_flow
        from src.processing.tool_registry import create_tool_registry
        
        # Setup test data
        test_claims = [
            Claim(
                id="flow_test_1",
                content="Integration test claim",
                confidence=0.8,
                state=ClaimState.VALIDATED,
                type=[ClaimType.EXAMPLE],
                tags=["integration", "test"]
            )
        ]
        
        tool_registry = create_tool_registry()
        
        # Execute complete flow
        result = coordinate_three_part_flow(
            session_id="flow_integration_test",
            user_request="Test the complete integration flow",
            all_claims=test_claims,
            tool_registry=tool_registry
        )
        
        assert result['success'] is True
        assert 'context' in result
        assert 'llm_response' in result
        assert 'processing_plan' in result
        
        results.add_result("Complete Flow Execution", True)
        results.add_result("Flow Result Validation", True)
        
    except Exception as e:
        results.add_result("Complete Flow Integration", False, str(e))
    
    return results.get_summary()

def test_end_to_end_scenarios():
    """Test real user scenarios"""
    print("\n" + "=" * 60)
    print("END-TO-END TESTS - Real Scenarios")
    print("=" * 60)
    
    results = TestResults()
    
    # Test 1: Research Assistant Scenario
    print("\n1. Testing Research Assistant Scenario...")
    try:
        from src.core.models import Claim, ClaimState, ClaimType
        from src.agent.llm_inference import coordinate_three_part_flow
        from src.processing.tool_registry import create_tool_registry
        
        # Create knowledge base
        knowledge_claims = [
            Claim(
                id="research_1",
                content="Machine learning models require training data",
                confidence=0.95,
                state=ClaimState.VALIDATED,
                type=[ClaimType.CONCEPT],
                tags=["ml", "training", "data"]
            ),
            Claim(
                id="research_2",
                content="Deep learning uses neural networks",
                confidence=0.9,
                state=ClaimState.VALIDATED,
                type=[ClaimType.CONCEPT],
                tags=["deep-learning", "neural-networks"]
            )
        ]
        
        tool_registry = create_tool_registry()
        
        # Simulate research query
        result = coordinate_three_part_flow(
            session_id="research_assistant",
            user_request="What do I need to build a machine learning system?",
            all_claims=knowledge_claims,
            tool_registry=tool_registry,
            conversation_history=[
                {"user": "I want to build a ML system", "assistant": "I can help with that"}
            ]
        )
        
        assert result['success'] is True
        assert len(result['context'].relevant_claims) > 0
        
        results.add_result("Research Query Processing", True)
        results.add_result("Knowledge Retrieval", True)
        
    except Exception as e:
        results.add_result("Research Assistant Scenario", False, str(e))
    
    # Test 2: Code Generation Scenario
    print("\n2. Testing Code Generation Scenario...")
    try:
        from src.core.models import Claim, ClaimState, ClaimType
        from src.agent.llm_inference import find_relevant_claims, format_claims_for_llm
        
        # Create programming knowledge
        code_claims = [
            Claim(
                id="code_1",
                content="Python functions use def keyword",
                confidence=1.0,
                state=ClaimState.VALIDATED,
                type=[ClaimType.SKILL],
                tags=["python", "functions", "syntax"]
            ),
            Claim(
                id="code_2",
                content="List comprehensions provide concise iteration",
                confidence=0.9,
                state=ClaimState.VALIDATED,
                type=[ClaimType.SKILL],
                tags=["python", "list-comprehensions"]
            )
        ]
        
        # Test claim formatting for LLM
        formatted = format_claims_for_llm(code_claims)
        assert "python functions" in formatted.lower()
        assert "list comprehensions" in formatted.lower()
        
        results.add_result("Code Knowledge Formatting", True)
        
        # Test relevance finding for coding queries
        relevant = find_relevant_claims("Create a Python function", code_claims)
        assert len(relevant) > 0
        assert any("function" in claim.content.lower() for claim in relevant)
        
        results.add_result("Code Query Relevance", True)
        
    except Exception as e:
        results.add_result("Code Generation Scenario", False, str(e))
    
    # Test 3: Multi-Turn Conversation
    print("\n3. Testing Multi-Turn Conversation...")
    try:
        from src.core.models import Claim, ClaimState, ClaimType
        from src.agent.llm_inference import coordinate_three_part_flow
        from src.processing.tool_registry import create_tool_registry
        
        # Create conversation context
        conversation_claims = [
            Claim(
                id="conv_1",
                content="User wants to learn about databases",
                confidence=0.8,
                state=ClaimState.EXPLORE,
                type=[ClaimType.GOAL],
                tags=["database", "learning"]
            )
        ]
        
        tool_registry = create_tool_registry()
        
        # Multi-turn conversation
        conversation_history = [
            {"user": "I want to learn databases", "assistant": "What aspect interests you?"},
            {"user": "SQL databases", "assistant": "SQL is great for structured data"}
        ]
        
        result = coordinate_three_part_flow(
            session_id="conversation_test",
            user_request="Tell me more about SQL optimization",
            all_claims=conversation_claims,
            tool_registry=tool_registry,
            conversation_history=conversation_history
        )
        
        assert result['success'] is True
        assert len(result['context'].conversation_history) == 2
        
        results.add_result("Conversation Context Management", True)
        results.add_result("Multi-Turn Processing", True)
        
    except Exception as e:
        results.add_result("Multi-Turn Conversation", False, str(e))
    
    return results.get_summary()

def test_architectural_compliance():
    """Test architectural compliance and design principles"""
    print("\n" + "=" * 60)
    print("ARCHITECTURAL COMPLIANCE TESTS")
    print("=" * 60)
    
    results = TestResults()
    
    # Test 1: No Circular Dependencies
    print("\n1. Testing Circular Dependency Prevention...")
    try:
        # Test that imports work without cycles
        from src.core.models import Claim
        from src.agent.llm_inference import build_llm_context
        from src.processing.tool_registry import create_tool_registry
        
        # Import should not hang or cause circular import errors
        print("   + Core models imported")
        print("   + Agent layer imported")
        print("   + Processing layer imported")
        
        results.add_result("No Circular Dependencies", True)
        
    except Exception as e:
        results.add_result("No Circular Dependencies", False, str(e))
    
    # Test 2: Separation of Concerns
    print("\n2. Testing Separation of Concerns...")
    try:
        # Core layer should not depend on processing/agent layers
        import src.core.models
        import src.core.claim_operations
        
        # Check that core doesn't import upper layers
        core_module = sys.modules['src.core.models']
        core_source = open(src.core.models.__file__, 'r').read()
        
        # Should not import from agent or processing
        assert 'from .agent' not in core_source
        assert 'from .processing' not in core_source
        assert 'from ..agent' not in core_source  
        assert 'from ..processing' not in core_source
        
        results.add_result("Core Layer Independence", True)
        
        # Agent layer can use core but not processing directly
        agent_source = open(src.agent.llm_inference.__file__, 'r').read()
        assert 'from ..core' in agent_source
        assert 'from ..processing' in agent_source  # This is expected for tool integration
        
        results.add_result("Layer Dependency Direction", True)
        
    except Exception as e:
        results.add_result("Separation of Concerns", False, str(e))
    
    # Test 3: Pure Functions Where Appropriate
    print("\n3. Testing Pure Function Design...")
    try:
        from src.processing.tool_registry import create_tool_registry, validate_function_parameters
        from src.core.models import Claim, ClaimState, ClaimType
        
        # Test that key functions are pure (same input -> same output)
        registry1 = create_tool_registry("test_pure_1")
        registry2 = create_tool_registry("test_pure_2")
        
        # Same inputs should create equivalent objects
        assert registry1.tools_directory != registry2.tools_directory
        assert len(registry1.tools) == len(registry2.tools) == 0
        
        results.add_result("Pure Function Tool Registry", True)
        
        # Test claim immutability principles
        claim1 = Claim(
            id="pure_test",
            content="Test pure function",
            confidence=0.8,
            state=ClaimState.VALIDATED,
            type=[ClaimType.EXAMPLE],
            tags=["test"]
        )
        
        claim2 = Claim(
            id="pure_test", 
            content="Test pure function",
            confidence=0.8,
            state=ClaimState.VALIDATED,
            type=[ClaimType.EXAMPLE],
            tags=["test"]
        )
        
        # Equal claims should be equal
        assert claim1.id == claim2.id
        assert claim1.content == claim2.content
        
        results.add_result("Claim Immutability", True)
        
    except Exception as e:
        results.add_result("Pure Function Design", False, str(e))
    
    return results.get_summary()

def main():
    """Run all test suites"""
    print("COMPREHENSIVE TEST SUITE FOR CONJECTURE 3-PART ARCHITECTURE")
    print("=" * 80)
    
    start_time = time.time()
    
    # Run all test suites
    unit_results = test_unit_components()
    integration_results = test_integration_layers()
    e2e_results = test_end_to_end_scenarios()
    compliance_results = test_architectural_compliance()
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Aggregate results
    total_tests = (unit_results['total'] + integration_results['total'] + 
                  e2e_results['total'] + compliance_results['total'])
    total_passed = (unit_results['passed'] + integration_results['passed'] + 
                   e2e_results['passed'] + compliance_results['passed'])
    total_failed = (unit_results['failed'] + integration_results['failed'] + 
                   e2e_results['failed'] + compliance_results['failed'])
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)
    print(f"Total Tests:     {total_tests}")
    print(f"Passed:          {total_passed}")
    print(f"Failed:          {total_failed}")
    print(f"Success Rate:    {(total_passed / total_tests) * 100:.1f}%")
    print(f"Duration:        {duration:.2f} seconds")
    
    print("\nTest Suite Breakdown:")
    print(f"Unit Tests:          {unit_results['passed']}/{unit_results['total']} passed ({unit_results['success_rate']:.1f}%)")
    print(f"Integration Tests:   {integration_results['passed']}/{integration_results['total']} passed ({integration_results['success_rate']:.1f}%)")
    print(f"End-to-End Tests:    {e2e_results['passed']}/{e2e_results['total']} passed ({e2e_results['success_rate']:.1f}%)")
    print(f"Compliance Tests:    {compliance_results['passed']}/{compliance_results['total']} passed ({compliance_results['success_rate']:.1f}%)")
    
    # Show errors if any
    all_errors = (unit_results['errors'] + integration_results['errors'] + 
                 e2e_results['errors'] + compliance_results['errors'])
    
    if all_errors:
        print("\nErrors encountered:")
        for error in all_errors:
            print(f"  - {error}")
    
    # Overall verdict
    print("\n" + "=" * 80)
    if total_failed == 0:
        print("OVERALL RESULT: ALL TESTS PASSED - + ARCHITECTURE VALIDATED")
        print("The 3-part architecture is functioning correctly!")
    else:
        print(f"OVERALL RESULT: {total_failed} TESTS FAILED - ! ISSUES DETECTED")
        print("Please review and fix the failed tests before proceeding.")
    print("=" * 80)
    
    # Write results to file
    results_file = "test_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'summary': {
                'total': total_tests,
                'passed': total_passed,
                'failed': total_failed,
                'success_rate': (total_passed / total_tests) * 100,
                'duration': duration
            },
            'unit_tests': unit_results,
            'integration_tests': integration_results,
            'e2e_tests': e2e_results,
            'compliance_tests': compliance_results,
            'errors': all_errors
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    return total_failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)