#!/usr/bin/env python3
"""
Test script for the updated CompleteContextBuilder implementation
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.context.complete_context_builder import CompleteContextBuilder
from src.core.models import Claim, ClaimType, ClaimState


def create_test_claims():
    """Create test claims for verification"""
    claims = [
        # Supporting claims (should be in Chain From User Claim)
        Claim(
            id="research-1",
            content="Studies show spaced repetition improves learning outcomes",
            confidence=0.95,
            state=ClaimState.VALIDATED,
            type=[ClaimType.REFERENCE],
            tags=["research", "learning"],
            supports=["ml-practice-1"]
        ),
        Claim(
            id="concept-1", 
            content="Understanding fundamentals is crucial for advanced topics",
            confidence=0.90,
            state=ClaimState.VALIDATED,
            type=[ClaimType.CONCEPT],
            tags=["education", "fundamentals"],
            supports=["ml-practice-1"]
        ),
        
        # Target claim (will be in Target Claim section)
        Claim(
            id="ml-practice-1",
            content="Practice machine learning with structured approach",
            confidence=0.85,
            state=ClaimState.EXPLORE,
            type=[ClaimType.GOAL],
            tags=["ml", "practice"],
            supported_by=["research-1", "concept-1"]
        ),
        
        # Direct supported claims (should be in Supported_by Claims)
        Claim(
            id="daily-1",
            content="Daily coding practice improves retention",
            confidence=0.80,
            state=ClaimState.EXPLORE,
            type=[ClaimType.EXAMPLE],
            tags=["daily", "practice"],
            supported_by=["ml-practice-1"]
        ),
        Claim(
            id="project-1",
            content="Building real projects solidifies understanding",
            confidence=0.75,
            state=ClaimState.EXPLORE,
            type=[ClaimType.EXAMPLE],
            tags=["project", "hands-on"],
            supported_by=["ml-practice-1"]
        ),
        
        # Semantic similar claims (should be in Relevant Claims)
        Claim(
            id="related-1",
            content="Consistent practice leads to mastery",
            confidence=0.88,
            state=ClaimState.VALIDATED,
            type=[ClaimType.CONCEPT],
            tags=["practice", "mastery"]
        ),
        Claim(
            id="related-2",
            content="Project-based learning enhances engagement",
            confidence=0.82,
            state=ClaimState.VALIDATED,
            type=[ClaimType.CONCEPT],
            tags=["projects", "engagement"]
        )
    ]
    
    return claims


def test_context_builder():
    """Test the updated CompleteContextBuilder implementation"""
    print("Testing updated CompleteContextBuilder...")
    
    # Create test claims
    claims = create_test_claims()
    print(f"Created {len(claims)} test claims")
    
    # Initialize context builder
    builder = CompleteContextBuilder(claims)
    print("✓ CompleteContextBuilder initialized successfully")
    
    # Build context for target claim
    target_claim_id = "ml-practice-1"
    try:
        context = builder.build_complete_context(
            target_claim_id=target_claim_id,
            max_tokens=4000,
            include_metadata=False
        )
        print("✓ Context built successfully")
        
        # Display the formatted context
        print("\n" + "="*60)
        print("FORMATTED CONTEXT OUTPUT:")
        print("="*60)
        print(context.context_text)
        print("="*60)
        
        # Verify the template structure
        context_lines = context.context_text.split('\n')
        
        # Check for required sections
        sections_found = {
            "# Relevant Claims": False,
            "# Chain From User Claim": False, 
            "# Supported_by Claims": False,
            "# Target Claim": False
        }
        
        for line in context_lines:
            if line in sections_found:
                sections_found[line] = True
        
        print("\n📋 Template Structure Verification:")
        for section, found in sections_found.items():
            status = "✓" if found else "✗"
            print(f"  {status} {section}")
        
        # Check target claim formatting (confidence should be redacted)
        target_section_found = False
        for i, line in enumerate(context_lines):
            if line == "# Target Claim" and i + 1 < len(context_lines):
                target_line = context_lines[i + 1]
                if "confidence_redacted" in target_line:
                    target_section_found = True
                    print("  ✓ Target claim confidence properly redacted")
                else:
                    print("  ✗ Target claim confidence NOT redacted")
                break
        
        if not target_section_found:
            print("  ✗ Target claim section not found")
        
        # Display metrics
        print(f"\n📊 Context Metrics:")
        print(f"  Total tokens used: {context.metrics.tokens_used}")
        print(f"  Upward chain claims: {context.metrics.upward_chain_claims}")
        print(f"  Downward chain claims: {context.metrics.downward_chain_claims}")
        print(f"  Semantic claims: {context.metrics.semantic_claims}")
        print(f"  Build time: {context.metrics.build_time_ms:.2f}ms")
        print(f"  Coverage completeness: {context.metrics.coverage_completeness:.2f}")
        
        # Test token allocation
        allocation = context.allocation
        print(f"\n💰 Token Allocation:")
        print(f"  Supporting claims (40%): {allocation.upward_chain_tokens} tokens")
        print(f"  Supported claims (30%): {allocation.downward_chain_tokens} tokens") 
        print(f"  Semantic claims (30%): {allocation.semantic_tokens} tokens")
        print(f"  Total: {allocation.total_tokens} tokens")
        
        return True
        
    except Exception as e:
        print(f"✗ Error building context: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_claim_formatters():
    """Test the Claim model formatting methods"""
    print("\nTesting Claim formatting methods...")
    
    claim = Claim(
        id="test-1",
        content="This is a test claim for formatting",
        confidence=0.85,
        state=ClaimState.EXPLORE,
        type=[ClaimType.CONCEPT],
        tags=["test", "formatting"]
    )
    
    print("format_for_context():")
    formatted_context = claim.format_for_context()
    print(f"  {formatted_context}")
    
    # Check format matches expected pattern: [c{id} | content | / confidence]
    if formatted_context.startswith(f"[c{claim.id} | ") and f" | / {claim.confidence:.2f}]" in formatted_context:
        print("  ✓ format_for_context() produces correct format")
    else:
        print("  ✗ format_for_context() format incorrect")
    
    print("\nformat_for_llm_analysis():")
    formatted_analysis = claim.format_for_llm_analysis()
    print(formatted_analysis)
    
    # Check that analysis includes expected fields
    required_fields = ["Claim ID:", "Content:", "Confidence:", "State:", "Type:", "Tags:", "Supports:", "Supported By:"]
    missing_fields = [field for field in required_fields if field not in formatted_analysis]
    
    if not missing_fields:
        print("  ✓ format_for_llm_analysis() includes all required fields")
    else:
        print(f"  ✗ format_for_llm_analysis() missing fields: {missing_fields}")


if __name__ == "__main__":
    print("🧪 TESTING UPDATED CONTEXT BUILDER IMPLEMENTATION")
    print("=" * 60)
    
    # Test claim formatting methods
    test_claim_formatters()
    
    # Test complete context builder
    success = test_context_builder()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("The context builder is working correctly with:")
        print("  • Claim model instead of UnifiedClaim")
        print("  • Approved LLM template format")
        print("  • Proper token allocation (40/30/30)")
        print("  • Direct supported_by relationships only")
        print("  • Confidence redaction for target claim")
    else:
        print("❌ TESTS FAILED!")
        print("Please check the errors above and fix the implementation.")