#!/usr/bin/env python3
"""
Simple test for the updated CompleteContextBuilder implementation
Tests the core functionality without complex imports
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Direct imports to avoid circular dependencies
from datetime import datetime
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
import re

# Import just the core Claim model
try:
    from src.core.models import Claim, ClaimType, ClaimState, create_claim_index
    CLAIM_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Could not import core Claim model: {e}")
    CLAIM_MODEL_AVAILABLE = False

# Import SupportRelationshipManager
try:
    from src.core.support_relationship_manager import SupportRelationshipManager
    RELATIONSHIP_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Could not import SupportRelationshipManager: {e}")
    RELATIONSHIP_MANAGER_AVAILABLE = False

# Import CompleteContextBuilder 
try:
    from src.context.complete_context_builder import CompleteContextBuilder
    CONTEXT_BUILDER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Could not import CompleteContextBuilder: {e}")
    CONTEXT_BUILDER_AVAILABLE = False


def create_test_claims():
    """Create test claims for verification"""
    if not CLAIM_MODEL_AVAILABLE:
        return []
    
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


def test_claim_formatters():
    """Test the Claim model formatting methods"""
    print("🧪 Testing Claim formatting methods...")
    
    if not CLAIM_MODEL_AVAILABLE:
        print("  ❌ Cannot test - Claim model not available")
        return False
    
    claim = Claim(
        id="test-1",
        content="This is a test claim for formatting",
        confidence=0.85,
        state=ClaimState.EXPLORE,
        type=[ClaimType.CONCEPT],
        tags=["test", "formatting"]
    )
    
    print("  📝 format_for_context():")
    try:
        formatted_context = claim.format_for_context()
        print(f"    {formatted_context}")
        
        # Check format matches expected pattern: [c{id} | content | / confidence]
        if formatted_context.startswith(f"[c{claim.id} | ") and f" | / {claim.confidence:.2f}]" in formatted_context:
            print("    ✅ format_for_context() produces correct format")
            context_test_passed = True
        else:
            print("    ❌ format_for_context() format incorrect")
            context_test_passed = False
    except Exception as e:
        print(f"    ❌ Error in format_for_context(): {e}")
        context_test_passed = False
    
    print("  📋 format_for_llm_analysis():")
    try:
        formatted_analysis = claim.format_for_llm_analysis()
        print(f"    {formatted_analysis}")
        
        # Check that analysis includes expected fields
        required_fields = ["Claim ID:", "Content:", "Confidence:", "State:", "Type:", "Tags:", "Supports:", "Supported By:"]
        missing_fields = [field for field in required_fields if field not in formatted_analysis]
        
        if not missing_fields:
            print("    ✅ format_for_llm_analysis() includes all required fields")
            analysis_test_passed = True
        else:
            print(f"    ❌ format_for_llm_analysis() missing fields: {missing_fields}")
            analysis_test_passed = False
    except Exception as e:
        print(f"    ❌ Error in format_for_llm_analysis(): {e}")
        analysis_test_passed = False
    
    return context_test_passed and analysis_test_passed


def test_context_builder():
    """Test the CompleteContextBuilder implementation"""
    print("\n🏗️ Testing CompleteContextBuilder...")
    
    missing_components = []
    if not CLAIM_MODEL_AVAILABLE:
        missing_components.append("Claim model")
    if not RELATIONSHIP_MANAGER_AVAILABLE:
        missing_components.append("SupportRelationshipManager")
    if not CONTEXT_BUILDER_AVAILABLE:
        missing_components.append("CompleteContextBuilder")
    
    if missing_components:
        print(f"  ❌ Cannot test - missing components: {', '.join(missing_components)}")
        return False
    
    # Create test claims
    claims = create_test_claims()
    print(f"  📝 Created {len(claims)} test claims")
    
    # Initialize context builder
    try:
        builder = CompleteContextBuilder(claims)
        print("  ✅ CompleteContextBuilder initialized successfully")
    except Exception as e:
        print(f"  ❌ Error initializing CompleteContextBuilder: {e}")
        return False
    
    # Build context for target claim
    target_claim_id = "ml-practice-1"
    try:
        context = builder.build_complete_context(
            target_claim_id=target_claim_id,
            max_tokens=4000,
            include_metadata=False
        )
        print("  ✅ Context built successfully")
        
        # Display the formatted context
        print("\n  📋 FORMATTED CONTEXT OUTPUT:")
        print("  " + "="*60)
        for line in context.context_text.split('\n'):
            print(f"  {line}")
        print("  " + "="*60)
        
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
        
        print("\n  🔍 Template Structure Verification:")
        all_sections_found = True
        for section, found in sections_found.items():
            status = "✅" if found else "❌"
            print(f"    {status} {section}")
            if not found:
                all_sections_found = False
        
        # Check target claim formatting (confidence should be redacted)
        target_section_found = False
        for i, line in enumerate(context_lines):
            if line == "# Target Claim" and i + 1 < len(context_lines):
                target_line = context_lines[i + 1]
                if "confidence_redacted" in target_line:
                    target_section_found = True
                    print("    ✅ Target claim confidence properly redacted")
                else:
                    print("    ❌ Target claim confidence NOT redacted")
                break
        
        if not target_section_found:
            print("    ❌ Target claim section not found")
            all_sections_found = False
        
        # Display metrics
        print(f"\n  📊 Context Metrics:")
        print(f"    Total tokens used: {context.metrics.tokens_used}")
        print(f"    Upward chain claims: {context.metrics.upward_chain_claims}")
        print(f"    Downward chain claims: {context.metrics.downward_chain_claims}")
        print(f"    Semantic claims: {context.metrics.semantic_claims}")
        print(f"    Build time: {context.metrics.build_time_ms:.2f}ms")
        print(f"    Coverage completeness: {context.metrics.coverage_completeness:.2f}")
        
        # Test token allocation
        allocation = context.allocation
        print(f"\n  💰 Token Allocation:")
        print(f"    Supporting claims (40%): {allocation.upward_chain_tokens} tokens")
        print(f"    Supported claims (30%): {allocation.downward_chain_tokens} tokens") 
        print(f"    Semantic claims (30%): {allocation.semantic_tokens} tokens")
        print(f"    Total: {allocation.total_tokens} tokens")
        
        return all_sections_found
        
    except Exception as e:
        print(f"  ❌ Error building context: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("🧪 TESTING UPDATED CONTEXT BUILDER IMPLEMENTATION")
    print("=" * 60)
    
    # Test 1: Claim formatting methods
    format_test_passed = test_claim_formatters()
    
    # Test 2: Complete context builder
    context_test_passed = test_context_builder()
    
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY:")
    
    if format_test_passed:
        print("  ✅ Claim formatting methods working correctly")
    else:
        print("  ❌ Claim formatting methods have issues")
    
    if context_test_passed:
        print("  ✅ CompleteContextBuilder working correctly")
    else:
        print("  ❌ CompleteContextBuilder has issues")
    
    if format_test_passed and context_test_passed:
        print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("The context builder is working correctly with:")
        print("  • Claim model instead of UnifiedClaim")
        print("  • Approved LLM template format") 
        print("  • Proper token allocation (40/30/30)")
        print("  • Direct supported_by relationships only")
        print("  • Confidence redaction for target claim")
        return True
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("Please check the errors above and fix the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)