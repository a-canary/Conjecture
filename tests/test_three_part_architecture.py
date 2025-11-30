"""
Comprehensive Test Suite for 3-Part Architecture Implementation
Tests that all functionality is preserved while maintaining clean architectural separation.
"""
import sys
import os
import asyncio
from datetime import datetime
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.models import Claim, ClaimState, ClaimType, DirtyReason
from src.core.claim_operations import (
    update_confidence, add_support, mark_dirty, should_prioritize,
    find_supporting_claims, filter_claims_by_confidence
)
from src.core.relationship_manager import (
    establish_bidirectional_relationship, validate_relationship_consistency,
    analyze_claim_relationships, get_relationship_statistics
)
from src.processing.tool_registry import (
    create_tool_registry, register_tool_function, load_tool_from_file,
    create_and_register_tool, get_tool_registry_stats
)
from src.processing.tool_execution import (
    execute_tool_from_registry, create_tool_call, batch_execute_tool_calls
)
from src.agent.agent_coordination import (
    process_user_request, initialize_agent_system, create_agent_session
)
from src.agent.llm_inference import (
    build_llm_context, create_llm_prompt, parse_llm_response,
    coordinate_three_part_flow
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestSuite:
    """Comprehensive test suite for 3-part architecture."""
    
    def __init__(self):
        self.test_results = []
        self.failed_tests = []
        
    def run_test(self, test_name: str, test_func):
        """Run a single test and record results."""
        try:
            logger.info(f"Running test: {test_name}")
            result = test_func()
            if result:
                logger.info(f"✅ PASSED: {test_name}")
                self.test_results.append((test_name, True, None))
            else:
                logger.error(f"❌ FAILED: {test_name}")
                self.test_results.append((test_name, False, "Test returned False"))
                self.failed_tests.append(test_name)
        except Exception as e:
            logger.error(f"❌ ERROR in {test_name}: {e}")
            self.test_results.append((test_name, False, str(e)))
            self.failed_tests.append(f"{test_name}: {str(e)}")
    
    def run_all_tests(self):
        """Run all tests in the suite."""
        logger.info("=== Starting 3-Part Architecture Test Suite ===\n")
        
        # Claims Layer Tests
        self.run_test("Claims Layer - Pure Data Structure", self.test_claims_purity)
        self.run_test("Claims Layer - Pure Functions Operations", self.test_claim_operations)
        self.run_test("Claims Layer - Relationship Operations", self.test_claim_relationships)
        self.run_test("Claims Layer - Advanced Relationships", self.test_advanced_relationships)
        
        # Tools Layer Tests  
        self.run_test("Tools Layer - Pure Registry", self.test_tool_registry_purity)
        self.run_test("Tools Layer - Function Registration", self.test_tool_registration)
        self.run_test("Tools Layer - Pure Execution", self.test_tool_execution)
        self.run_test("Tools Layer - Batch Execution", self.test_batch_execution)
        
        # LLM Inference Layer Tests
        self.run_test("LLM Layer - Context Building", self.test_context_building)
        self.run_test("LLM Layer - Prompt Creation", self.test_prompt_creation)
        self.run_test("LLM Layer - Response Parsing", self.test_response_parsing)
        self.run_test("LLM Layer - Simulation", self.test_llm_simulation)
        
        # 3-Part Coordination Tests
        self.run_test("Coordination - Complete Flow", self.test_complete_coordination)
        self.run_test("Coordination - Session Management", self.test_session_management)
        self.run_test("Coordination - System Initialization", self.test_system_initialization)
        self.run_test("Coordination - Error Handling", self.test_error_handling)
        
        # Architectural Integrity Tests
        self.run_test("Architecture - Separation of Concerns", self.test_separation_of_concerns)
        self.run_test("Architecture - Data Flow Purity", self.test_data_flow_purity)
        self.run_test("Architecture - No Violations", self.test_no_violations)
        
        # Print results
        self.print_results()
    
    def print_results(self):
        """Print comprehensive test results."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for _, passed, _ in self.test_results if passed)
        failed_tests = total_tests - passed_tests
        
        print(f"\n=== TEST RESULTS ===")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        if self.failed_tests:
            print(f"\n=== FAILED TESTS ===")
            for failure in self.failed_tests:
                print(f"❌ {failure}")
        
        if failed_tests == 0:
            print(f"\n🎉 ALL TESTS PASSED! 3-Part Architecture Implementation is SUCCESSFUL!")
        else:
            print(f"\n⚠️  Some tests failed. Please review the failures above.")
    
    # Claims Layer Tests
    
    def test_claims_purity(self):
        """Test that claims are pure data structures."""
        claim = Claim(
            id="test_claim",
            content="Test claim content",
            confidence=0.8,
            state=ClaimState.VALIDATED,
            supported_by=[],
            supports=[],
            type=[ClaimType.CONCEPT],
            tags=["test"],
            created=datetime.utcnow(),
            updated=datetime.utcnow()
        )
        
        # Check no execution methods
        public_methods = [method for method in dir(claim) 
                         if not method.startswith('_') and callable(getattr(claim, method))]
        allowed_methods = {'format_for_context', 'to_chroma_metadata', 'to_chroma_batch'}
        
        for method in public_methods:
            if method not in allowed_methods:
                return False
        
        # Try to call an execution method (should not exist)
        try:
            result = claim.update_confidence(0.9)
            return False  # Should not reach here
        except AttributeError:
            pass  # Expected
        
        return True
    
    def test_claim_operations(self):
        """Test pure claim operations functions."""
        claim = Claim(
            id="test_claim",
            content="Test claim",
            confidence=0.7,
            state=ClaimState.EXPLORE,
            supported_by=[],
            supports=[],
            type=[ClaimType.CONCEPT],
            tags=["test"],
            created=datetime.utcnow(),
            updated=datetime.utcnow()
        )
        
        # Test pure function operations
        updated = update_confidence(claim, 0.9)
        if updated.confidence != 0.9 or claim.confidence != 0.7:
            return False  # Original should be unchanged
        
        # Test add support
        with_support = add_support(claim, "supporter_001")
        if "supporter_001" not in with_support.supported_by or "supporter_001" in claim.supported_by:
            return False
        
        # Test mark dirty
        dirty = mark_dirty(claim, DirtyReason.MANUAL_MARK, 3)
        if not dirty.is_dirty or dirty.dirty_priority != 3 or claim.is_dirty:
            return False
        
        return True
    
    def test_claim_relationships(self):
        """Test claim relationship functions."""
        claim1 = Claim(
            id="claim1",
            content="Base claim",
            confidence=0.9,
            state=ClaimState.VALIDATED,
            supported_by=[],
            supports=[],
            type=[ClaimType.CONCEPT],
            tags=["base"],
            created=datetime.utcnow(),
            updated=datetime.utcnow()
        )
        
        claim2 = Claim(
            id="claim2", 
            content="Supporting claim",
            confidence=0.85,
            state=ClaimState.VALIDATED,
            supported_by=[],
            supports=[],
            type=[ClaimType.EXAMPLE],
            tags=["support"],
            created=datetime.utcnow(),
            updated=datetime.utcnow()
        )
        
        # Test bidirectional relationship establishment
        updated1, updated2 = establish_bidirectional_relationship(claim1, claim2)
        
        if claim2.id not in updated1.supports:
            return False
        if claim1.id not in updated2.supported_by:
            return False
        
        # Test relationship validation
        errors = validate_relationship_consistency([updated1, updated2])
        if errors:
            return False
        
        return True
    
    def test_advanced_relationships(self):
        """Test advanced relationship management."""
        # Create claims with relationships
        claims = []
        
        base_claim = Claim(
            id="base",
            content="Base theory",
            confidence=0.9,
            state=ClaimState.VALIDATED,
            supported_by=[],
            supports=["derived1", "derived2"],
            type=[ClaimType.THESIS],
            tags=["theory"],
            created=datetime.utcnow(),
            updated=datetime.utcnow()
        )
        claims.append(base_claim)
        
        for i in range(1, 3):
            derived = Claim(
                id=f"derived{i}",
                content=f"Derived theory {i}",
                confidence=0.8,
                state=ClaimState.VALIDATED,
                supported_by=["base"],
                supports=[],
                type=[ClaimType.CONCEPT],
                tags=["derived"],
                created=datetime.utcnow(),
                updated=datetime.utcnow()
            )
            claims.append(derived)
        
        # Test relationship analysis
        analysis = analyze_claim_relationships(base_claim, claims)
        if analysis.supported_count != 2 or analysis.support_count != 0:
            return False
        
        # Test statistics
        stats = get_relationship_statistics(claims)
        if stats["root_claims"] != 1 or stats["leaf_claims"] != 2:
            return False
        
        return True
    
    # Tools Layer Tests
    
    def test_tool_registry_purity(self):
        """Test that tool registry is pure data structure."""
        registry = create_tool_registry()
        
        # Check that registry is pure data
        if hasattr(registry, 'load_tool') or hasattr(registry, 'execute'):
            return False
        
        if not isinstance(registry.tools, dict):
            return False
        
        return True
    
    def test_tool_registration(self):
        """Test tool registration and discovery."""
        registry = create_tool_registry()
        
        # Create a simple test tool function
        def test_tool_func(x: int, y: int = 1) -> int:
            """Simple test function that adds numbers."""
            return x + y
        
        # Create tool function data
        from src.processing.tool_registry import ToolFunction
        tool_func = ToolFunction(
            name="test_add",
            file_path="test.py",
            description="Adds two numbers",
            version="1.0.0",
            parameters={
                "x": {"name": "x", "required": True, "default": None, "type_hint": "int"},
                "y": {"name": "y", "required": False, "default": 1, "type_hint": "int"}
            },
            return_type="int",
            created_at=datetime.utcnow().timestamp(),
            execution_count=0,
            success_count=0,
            function=test_tool_func
        )
        
        # Register the tool
        updated_registry = register_tool_function(registry, tool_func)
        
        # Check registration
        if "test_add" not in updated_registry.tools:
            return False
        
        # Test stats
        stats = get_tool_registry_stats(updated_registry)
        if stats["total_tools"] != 1:
            return False
        
        return True
    
    def test_tool_execution(self):
        """Test pure tool execution."""
        def multiply(x: int, y: int = 2) -> int:
            """Multiplies two numbers."""
            return x * y
        
        # Create tool function
        from src.processing.tool_registry import ToolFunction
        tool_func = ToolFunction(
            name="test_multiply",
            file_path="test.py", 
            description="Multiplies two numbers",
            version="1.0.0",
            parameters={
                "x": {"name": "x", "required": True, "default": None, "type_hint": "int"},
                "y": {"name": "y", "required": False, "default": 2, "type_hint": "int"}
            },
            return_type="int",
            created_at=datetime.utcnow().timestamp(),
            execution_count=0,
            success_count=0,
            function=multiply
        )
        
        # Create tool call
        tool_call = create_tool_call("test_multiply", {"x": 5, "y": 3})
        
        # Execute
        result = execute_tool_from_registry(tool_call, create_tool_registry())  # Empty registry for now
        
        # Should fail because tool not in registry
        if result.success:
            return False
        
        # Now test with proper registry
        registry = register_tool_function(create_tool_registry(), tool_func)
        result = execute_tool_from_registry(tool_call, registry)
        
        if not result.success or result.result != 15:
            return False
        
        return True
    
    def test_batch_execution(self):
        """Test batch tool execution."""
        def add_one(x: int) -> int:
            """Adds one to input."""
            return x + 1
        
        # Create and register tool
        from src.processing.tool_registry import ToolFunction
        tool_func = ToolFunction(
            name="add_one",
            file_path="test.py",
            description="Adds one",
            version="1.0.0", 
            parameters={
                "x": {"name": "x", "required": True, "default": None, "type_hint": "int"}
            },
            return_type="int",
            created_at=datetime.utcnow().timestamp(),
            execution_count=0,
            success_count=0,
            function=add_one
        )
        
        registry = register_tool_function(create_tool_registry(), tool_func)
        
        # Create multiple tool calls
        tool_calls = [
            create_tool_call("add_one", {"x": 1}),
            create_tool_call("add_one", {"x": 5}),
            create_tool_call("add_one", {"x": 10})
        ]
        
        # Execute batch
        results = batch_execute_tool_calls(tool_calls, registry)
        
        if len(results) != 3:
            return False
        
        expected_results = [2, 6, 11]
        for i, result in enumerate(results):
            if not result.success or result.result != expected_results[i]:
                return False
        
        return True
    
    # LLM Inference Layer Tests
    
    def test_context_building(self):
        """Test LLM context building."""
        claims = [
            Claim(
                id="claim1",
                content="Test claim 1",
                confidence=0.9,
                state=ClaimState.VALIDATED,
                supported_by=[],
                supports=[],
                type=[ClaimType.CONCEPT],
                tags=["test"],
                created=datetime.utcnow(),
                updated=datetime.utcnow()
            )
        ]
        
        registry = create_tool_registry()
        context = build_llm_context(
            session_id="test_session",
            user_request="Test request",
            all_claims=claims,
            tool_registry=registry
        )
        
        if context.session_id != "test_session":
            return False
        if len(context.relevant_claims) != 1:
            return False
        if context.user_request != "Test request":
            return False
        
        return True
    
    def test_prompt_creation(self):
        """Test LLM prompt creation."""
        from src.agent.llm_inference import LLMContext
        from src.processing.tool_registry import ToolFunction
        
        context = LLMContext(
            session_id="test",
            user_request="Test request",
            relevant_claims=[],
            available_tools=[],
            conversation_history=None,
            metadata={}
        )
        
        prompt = create_llm_prompt(context)
        
        if not prompt or len(prompt) < 10:
            return False
        if "TEST REQUEST" not in prompt.upper():
            return False
        if "AVAILABLE TOOLS" not in prompt:
            return False
        
        return True
    
    def test_response_parsing(self):
        """Test LLM response parsing."""
        # Test simple text response
        simple_response = "This is a simple response."
        parsed = parse_llm_response(simple_response)
        
        if parsed.response_text != simple_response:
            return False
        if len(parsed.tool_calls) != 0:
            return False
        
        # Test response with tool calls
        tool_response = """I'll help you with that.

<tool_calls>
  <invoke name="test_tool">
    <parameter name="param1">value1</parameter>
  </invoke>
</tool_calls>

Done."""
        
        parsed = parse_llm_response(tool_response)
        if len(parsed.tool_calls) != 1:
            return False
        if parsed.tool_calls[0].name != "test_tool":
            return False
        
        return True
    
    def test_llm_simulation(self):
        """Test LLM simulation functionality."""
        from src.agent.llm_inference import LLMContext, simulate_llm_interaction
        
        # Create test context
        context = LLMContext(
            session_id="test",
            user_request="research quantum computing",
            relevant_claims=[],
            available_tools=[],
            conversation_history=None,
            metadata={}
        )
        
        response = simulate_llm_interaction(context)
        
        if not response.response_text:
            return False
        if response.confidence <= 0:
            return False
        
        return True
    
    # 3-Part Coordination Tests
    
    def test_complete_coordination(self):
        """Test complete 3-part coordination flow."""
        claims = [
            Claim(
                id="coord_test",
                content="Test claim for coordination",
                confidence=0.8,
                state=ClaimState.VALIDATED,
                supported_by=[],
                supports=[],
                type=[ClaimType.CONCEPT],
                tags=["test", "coordination"],
                created=datetime.utcnow(),
                updated=datetime.utcnow()
            )
        ]
        
        registry = create_tool_registry()
        result = process_user_request(
            user_request="Test coordination request",
            existing_claims=claims,
            tool_registry=registry
        )
        
        if not result.success:
            return False
        if not result.session_id:
            return False
        if not result.llm_response:
            return False
        
        return True
    
    def test_session_management(self):
        """Test session management."""
        registry = create_tool_registry()
        claims = []
        
        session = create_agent_session(
            user_request="Test session",
            existing_claims=claims,
            tool_registry=registry
        )
        
        if not session.session_id:
            return False
        if session.user_request != "Test session":
            return False
        if len(session.claims) != 0:
            return False
        
        return True
    
    def test_system_initialization(self):
        """Test system initialization."""
        # This test uses a temporary directory for tools
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        try:
            result = initialize_agent_system(tools_directory=temp_dir)
            
            if not result["success"]:
                return False
            if "tool_registry" not in result:
                return False
        finally:
            shutil.rmtree(temp_dir)
        
        return True
    
    def test_error_handling(self):
        """Test error handling in coordination."""
        registry = create_tool_registry()
        claims = []
        
        # Test with invalid request (should still work but might have limited functionality)
        result = process_user_request(
            user_request="",
            existing_claims=claims,
            tool_registry=registry
        )
        
        # Should either succeed with empty response or fail gracefully
        return result.success or result.errors
    
    # Architectural Integrity Tests
    
    def test_separation_of_concerns(self):
        """Test that concerns are properly separated."""
        # Claims should not have tool execution
        claim = Claim(
            id="test",
            content="Test",
            confidence=0.8,
            state=ClaimState.VALIDATED,
            supported_by=[],
            supports=[],
            type=[ClaimType.CONCEPT],
            tags=["test"],
            created=datetime.utcnow(),
            updated=datetime.utcnow()
        )
        
        if hasattr(claim, 'execute') or hasattr(claim, 'call_tool'):
            return False
        
        # Tool registry should not have claim processing
        registry = create_tool_registry()
        if hasattr(registry, 'create_claim') or hasattr(registry, 'update_claim'):
            return False
        
        return True
    
    def test_data_flow_purity(self):
        """Test that data flows in the correct direction."""
        # Test that claims are immutable (operations return new instances)
        original_claim = Claim(
            id="immutable_test",
            content="Original content",
            confidence=0.7,
            state=ClaimState.EXPLORE,
            supported_by=[],
            supports=[],
            type=[ClaimType.CONCEPT],
            tags=["immutable"],
            created=datetime.utcnow(),
            updated=datetime.utcnow()
        )
        
        # Perform operation
        updated_claim = update_confidence(original_claim, 0.9)
        
        # Original should be unchanged
        if original_claim.confidence != 0.7:
            return False
        if updated_claim.confidence != 0.9:
            return False
        
        return True
    
    def test_no_violations(self):
        """Test that no architectural violations exist."""
        violations = []
        
        # Check Claim model
        claim = Claim(
            id="violation_test",
            content="Test claim",
            confidence=0.8,
            state=ClaimState.VALIDATED,
            supported_by=[],
            supports=[],
            type=[ClaimType.CONCEPT],
            tags=["test"],
            created=datetime.utcnow(),
            updated=datetime.utcnow()
        )
        
        # These methods should NOT exist on claims
        forbidden_methods = [
            'execute', 'run', 'call', 'process', 'handle',
            'update_confidence', 'add_support', 'mark_dirty'
        ]
        
        for method in forbidden_methods:
            if hasattr(claim, method):
                violations.append(f"Claim has forbidden method: {method}")
        
        # Check tool registry
        registry = create_tool_registry()
        forbidden_registry_methods = [
            'create_claim', 'update_claim', 'process_llm', 'execute_llm'
        ]
        
        for method in forbidden_registry_methods:
            if hasattr(registry, method):
                violations.append(f"Tool registry has forbidden method: {method}")
        
        return len(violations) == 0


def main():
    """Run the complete test suite."""
    test_suite = TestSuite()
    test_suite.run_all_tests()
    
    # Return exit code based on results
    failed_count = len(test_suite.failed_tests)
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)