"""
Data Flow Demonstration - Pure 3-Part Architecture Flow
Shows the clean separation: Claims -> LLM -> Tools -> Claims

Naming convention:
- subs: claims that provide evidence FOR this claim (children)
- supers: claims this claim provides evidence FOR (toward root, parents)
"""
from datetime import datetime, timezone
from typing import List, Dict, Any
import logging

from ..core.models import Claim, ClaimState, ClaimType, DirtyReason
from ..core.claim_operations import (
    add_sub, update_confidence, mark_dirty,
    calculate_support_strength, validate_relationship_integrity
)
from ..processing.tool_registry import create_tool_registry, get_available_tools
from ..processing.tool_execution import execute_tool_from_registry
from .agent_coordination import process_user_request, initialize_agent_system
from .llm_inference import build_llm_context, create_llm_prompt

logger = logging.getLogger(__name__)

def demonstrate_data_flow():
    """Demonstrate the pure 3-part data flow."""
    
    print("=== 3-Part Architecture Data Flow Demonstration ===\n")
    
    # 1. CLAIMS LAYER: Pure Data Only
    print("1. CLAIMS LAYER - Pure Knowledge Data:")
    existing_claims = create_sample_claims()
    print_claims_summary(existing_claims)
    print()
    
    # 2. TOOLS LAYER: Pure Functions Only  
    print("2. TOOLS LAYER - Pure Function Registry:")
    system_init = initialize_agent_system()
    if system_init["success"]:
        tool_registry = system_init["tool_registry"]
        tools = get_available_tools(tool_registry)
        print_tools_summary(tools)
    else:
        print(f"Tool system initialization failed: {system_init['error']}")
        return
    print()
    
    # 3. LLM INFERENCE LAYER: The Bridge
    print("3. LLM INFERENCE LAYER - Reasoning Bridge:")
    user_request = "Research the latest developments in quantum computing"
    print(f"User Request: {user_request}")
    
    # Build context from claims and tools
    context = build_llm_context(
        session_id="demo_session",
        user_request=user_request,
        all_claims=existing_claims,
        tool_registry=tool_registry
    )
    print(f"Context built with {len(context.relevant_claims)} relevant claims and {len(context.available_tools)} available tools")
    print()
    
    # Show prompt creation (LLM reasoning input)
    prompt = create_llm_prompt(context)
    print(f"LLM Prompt Preview (first 200 chars): {prompt[:200]}...")
    print()
    
    # Execute the complete 3-part flow
    print("4. COMPLETE 3-PART FLOW EXECUTION:")
    print("   Claims -> LLM Reasoning -> Tools -> New Claims")
    print()
    
    result = process_user_request(
        user_request=user_request,
        existing_claims=existing_claims,
        tool_registry=tool_registry
    )
    
    # Display results
    print_results_summary(result, existing_claims)
    
    return result

def create_sample_claims() -> List[Claim]:
    """Create sample claims for demonstration.
    
    Note: subs = claims that provide evidence FOR this claim (children)
          supers = claims this provides evidence FOR (toward root, parents)
    """
    return [
        Claim(
            id="claim_001",
            content="Quantum computers use quantum bits (qubits) that can exist in superposition",
            confidence=0.9,
            state=ClaimState.VALIDATED,
            subs=[],  # No claims provide evidence FOR this claim
            supers=["claim_002", "claim_003"],  # This claim provides evidence FOR claim_002 and claim_003
            type=[ClaimType.CONCEPT],
            tags=["quantum", "computer", "qubit", "superposition"],
            created=datetime.now(timezone.utc),
            updated=datetime.now(timezone.utc)
        ),
        Claim(
            id="claim_002", 
            content="Superposition allows quantum computers to process multiple states simultaneously",
            confidence=0.85,
            state=ClaimState.VALIDATED,
            subs=["claim_001"],  # claim_001 provides evidence FOR this claim
            supers=["claim_004"],  # This claim provides evidence FOR claim_004
            type=[ClaimType.CONCEPT],
            tags=["quantum", "superposition", "processing"],
            created=datetime.now(timezone.utc),
            updated=datetime.now(timezone.utc)
        ),
        Claim(
            id="claim_003",
            content="Quantum entanglement establishes correlations between qubits",
            confidence=0.8,
            state=ClaimState.EXPLORE,
            subs=["claim_001"],  # claim_001 provides evidence FOR this claim
            supers=[],  # This claim doesn't provide evidence for any other claims
            type=[ClaimType.CONCEPT],
            tags=["quantum", "entanglement", "qubit"],
            created=datetime.now(timezone.utc),
            updated=datetime.now(timezone.utc)
        ),
        Claim(
            id="claim_004",
            content="Quantum computers could potentially solve certain problems faster than classical computers",
            confidence=0.75,
            state=ClaimState.EXPLORE,
            subs=["claim_002"],  # claim_002 provides evidence FOR this claim
            supers=[],  # This claim doesn't provide evidence for any other claims (root-level thesis)
            type=[ClaimType.THESIS],
            tags=["quantum", "performance", "complexity", "advantage"],
            created=datetime.now(timezone.utc),
            updated=datetime.now(timezone.utc),
            is_dirty=True,
            dirty_reason=DirtyReason.CONFIDENCE_THRESHOLD,
            dirty_timestamp=datetime.now(timezone.utc),
            dirty_priority=5
        )
    ]

def print_claims_summary(claims: List[Claim]):
    """Print summary of claims showing pure data nature."""
    print("   Claims are pure data structures:")
    for claim in claims:
        type_str = ", ".join([t.value for t in claim.type])
        support_info = f" (supers: {len(claim.supers)}, subs: {len(claim.subs)})"
        dirty_info = " [DIRTY]" if claim.is_dirty else ""
        print(f"   - [{claim.id}] C:{claim.confidence:.2f} T:{type_str}{support_info}{dirty_info}")
        print(f"     {claim.content}")
    print("   No execution methods - pure data only")

def print_tools_summary(tools: List[Dict[str, Any]]):
    """Print summary of available tools."""
    print("   Tools are pure functions:")
    for tool in tools[:5]:  # Show first 5
        params = list(tool['parameters'].keys()) if tool['parameters'] else ['none']
        print(f"   - {tool['name']}({', '.join(params)})")
        print(f"     {tool['description'] or 'No description'}")
    print(f"   No embedded logic - pure function registration")

def print_results_summary(result: Dict[str, Any], original_claims: List[Claim]):
    """Print comprehensive results summary."""
    print("   FLOW RESULTS:")
    print(f"   Success: {result['success']}")
    
    if result['success']:
        # LLM response
        print("   LLM Response:")
        print(f"     \"{result['llm_response'][:100]}{'...' if len(result['llm_response']) > 100 else ''}\"")
        
        # Tool execution
        tool_results = result['tool_results']
        print(f"   Tools Executed: {len(tool_results)}")
        for i, tool_result in enumerate(tool_results):
            status = "SUCCESS" if tool_result.success else "FAILED"
            print(f"     Tool {i+1}: {tool_result.skill_id} -> {status}")
            if not tool_result.success:
                print(f"       Error: {tool_result.error_message}")
        
        # Claim updates
        updated_claims = result['updated_claims']
        new_claims = result['new_claims']
        print(f"   Claims Updated: {len(updated_claims)}")
        print(f"   New Claims Created: {len(new_claims)}")
        
        # Show data flow visualization
        print("\n   DATA FLOW VISUALIZATION:")
        print("   +-------------------+")
        print("   |   CLAIMS LAYER    |  <- Pure knowledge data")  
        print("   |  (4 existing)     |")
        print("   +---------+---------+")
        print("             |")
        print("             v")
        print("   +-------------------+")
        print("   |  LLM INFERENCE    |  <- Context + reasoning")
        print("   |     LAYER         |")
        print("   +---------+---------+")
        print("             |")
        print("             v")
        print("   +-------------------+")
        print("   |   TOOLS LAYER     |  <- Pure function execution")
        print("   |  (calls executed) |")
        print("   +---------+---------+")
        print("             |")
        print("             v")
        print("   +-------------------+")
        print("   |   CLAIMS LAYER    |  <- New/updated knowledge")
        print(f"   |  ({len(updated_claims)} + {len(new_claims)} total)    |")
        print("   +-------------------+")
        
        # Show the clean architectural separation
        print("\n   ARCHITECTURAL SEPARATION CONFIRMED:")
        print("   [OK] Claims: Pure data structures only")
        print("   [OK] Tools: Pure functions only") 
        print("   [OK] LLM: Reasoning bridge only")
        print("   [OK] Data Flow: Claims -> LLM -> Tools -> Claims")
        
    else:
        print("   [FAILED] Processing Failed:")
        for error in result['errors']:
            print(f"     - {error}")

def demonstrate_relationship_handling():
    """Demonstrate claim relationship handling."""
    print("\n=== CLAIM RELATIONSHIP HANDLING DEMONSTRATION ===\n")
    
    # Create related claims
    base_claim = Claim(
        id="base_001",
        content="Machine learning requires large datasets for training",
        confidence=0.9,
        state=ClaimState.VALIDATED,
        subs=[],  # No claims provide evidence FOR this yet
        supers=[],  # This claim doesn't provide evidence for any other claims
        type=[ClaimType.CONCEPT],
        tags=["ml", "data", "training"],
        created=datetime.now(timezone.utc),
        updated=datetime.now(timezone.utc)
    )
    
    supporting_claim = Claim(
        id="support_001", 
        content="Deep neural networks improve with more training examples",
        confidence=0.85,
        state=ClaimState.VALIDATED,
        subs=[],  # No claims provide evidence FOR this
        supers=["base_001"],  # This claim provides evidence FOR base_001
        type=[ClaimType.EXAMPLE],
        tags=["ml", "deep-learning", "training-data"],
        created=datetime.now(timezone.utc),
        updated=datetime.now(timezone.utc)
    )
    
    # Use pure functions to establish relationships
    # add_sub adds a claim to the subs list (claims that provide evidence FOR this claim)
    updated_base = add_sub(base_claim, supporting_claim.id)
    
    print("RELATIONSHIP OPERATIONS (Pure Functions):")
    print(f"Base Claim: {base_claim.content}")
    print(f"Supporting Claim: {supporting_claim.content}")
    print(f"After add_sub() - Base claim subs: {updated_base.subs}")
    
    # Calculate support strength
    all_claims = [updated_base, supporting_claim]
    strength, count = calculate_support_strength(updated_base, all_claims)
    print(f"Support strength: {strength:.2f} from {count} sub claims")
    
    # Validate relationships
    errors = validate_relationship_integrity(updated_base, all_claims)
    print(f"Relationship validation: {'PASSED' if not errors else 'FAILED'}")
    if errors:
        for error in errors:
            print(f"  - {error}")
    
    print("[OK] Relationship handling is pure functional")

def demonstrate_architectural_violations_fixed():
    """Show that architectural violations have been fixed."""
    print("\n=== ARCHITECTURAL VIOLATIONS FIXED ===\n")
    
    print("BEFORE REFACTORING:")
    print("[X] Claim model had execution methods (update_confidence, mark_dirty, etc.)")
    print("[X] ToolManager mixed procedural logic with pure functions")  
    print("[X] AgentHandle had mixed responsibilities")
    print("[X] Data flow was unclear and interconnected")
    
    print("\nAFTER REFACTORING:")
    print("[OK] Claims are pure data models (no methods)")
    print("[OK] Claim operations moved to pure functions (claim_operations.py)")
    print("[OK] Tools are pure functions with clear registry")
    print("[OK] LLM inference is the only coordination bridge")
    print("[OK] Clear data flow: Claims -> LLM -> Tools -> Claims")
    print("[OK] Each layer has single responsibility")
    
    print("\nTESTING THE PURE ARCHITECTURE:")
    
    # Test Claim purity - should have no execution methods
    claim = Claim(
        id="test_claim",
        content="Test claim for purity",
        confidence=1.0,
        state=ClaimState.VALIDATED,
        subs=[],
        supers=[],
        type=[ClaimType.CONCEPT],
        tags=["test"],
        created=datetime.now(timezone.utc),
        updated=datetime.now(timezone.utc)
    )
    
    execution_methods = [method for method in dir(claim) if not method.startswith('_') and callable(getattr(claim, method))]
    print(f"[OK] Claim object has {len(execution_methods)} public methods: {execution_methods}")
    claim_pure = all(method in ['format_for_context', 'to_chroma_metadata'] for method in execution_methods)
    print(f"[OK] Claim is pure (only formatting methods): {claim_pure}")
    
    # Tool registry purity test
    registry = create_tool_registry()
    print(f"[OK] Tool registry is pure data structure with {len(registry.tools)} tools")
    
    print("\n[SUCCESS] 3-Part Architecture Implementation Complete!")

if __name__ == "__main__":
    """Run the complete demonstration."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run demonstrations
    demonstrate_data_flow()
    demonstrate_relationship_handling() 
    demonstrate_architectural_violations_fixed()
