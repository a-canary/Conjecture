#!/usr/bin/env python3
"""
Standalone test for the updated context formatting logic
Tests the formatting methods without complex dependencies
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# Define minimal models for testing
class ClaimState(str, Enum):
    EXPLORE = "Explore"
    VALIDATED = "Validated"
    ORPHANED = "Orphaned"
    QUEUED = "Queued"


class ClaimType(str, Enum):
    CONCEPT = "concept"
    REFERENCE = "reference"
    THESIS = "thesis"
    SKILL = "skill"
    EXAMPLE = "example"
    GOAL = "goal"


class Claim(BaseModel):
    """Core claim model with validation"""
    id: str
    content: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    state: ClaimState = ClaimState.EXPLORE
    supported_by: List[str] = Field(default_factory=list)
    supports: List[str] = Field(default_factory=list)
    type: List[ClaimType] = Field(..., min_items=1)
    tags: List[str] = Field(default_factory=list)
    created: datetime = Field(default_factory=datetime.utcnow)
    updated: datetime = Field(default_factory=datetime.utcnow)

    def format_for_context(self) -> str:
        """Format claim for LLM context in standard [c{id} | content | / confidence] format"""
        return f"[c{self.id} | {self.content} | / {self.confidence:.2f}]"

    def format_for_llm_analysis(self) -> str:
        """Format claim for detailed LLM analysis with metadata"""
        type_str = ",".join([t.value for t in self.type])
        tags_str = ",".join(self.tags) if self.tags else "none"
        return (
            f"Claim ID: {self.id}\n"
            f"Content: {self.content}\n"
            f"Confidence: {self.confidence:.2f}\n"
            f"State: {self.state.value}\n"
            f"Type: {type_str}\n"
            f"Tags: {tags_str}\n"
            f"Supports: {', '.join(self.supports) if self.supports else 'none'}\n"
            f"Supported By: {', '.join(self.supported_by) if self.supported_by else 'none'}"
        )


def format_context_template(target_claim, upward_claims, downward_claims, semantic_claims):
    """Format the complete context using the approved LLM template"""
    context_parts = []

    # Relevant Claims section (semantic claims first for LLM attention)
    if semantic_claims:
        context_parts.append("# Relevant Claims")
        for claim in semantic_claims:
            context_parts.append(claim.format_for_context())
        context_parts.append("")

    # Chain From User Claim section (supporting claims)
    if upward_claims:
        context_parts.append("# Chain From User Claim")
        for claim in upward_claims:
            context_parts.append(claim.format_for_context())
        context_parts.append("")

    # Supported_by Claims section (direct supported claims)
    if downward_claims:
        context_parts.append("# Supported_by Claims")
        for claim in downward_claims:
            context_parts.append(claim.format_for_context())
        context_parts.append("")

    # Target Claim (last for LLM attention, confidence redacted)
    context_parts.append("# Target Claim")
    # Redact confidence for target claim as required
    target_formatted = f"[c{target_claim.id} | {target_claim.content} | / confidence_redacted]"
    context_parts.append(target_formatted)

    return "\n".join(context_parts)


def create_test_claims():
    """Create test claims for verification"""
    claims = {
        # Supporting claims (should be in Chain From User Claim)
        "research-1": Claim(
            id="research-1",
            content="Studies show spaced repetition improves learning outcomes",
            confidence=0.95,
            state=ClaimState.VALIDATED,
            type=[ClaimType.REFERENCE],
            tags=["research", "learning"],
            supports=["ml-practice-1"]
        ),
        "concept-1": Claim(
            id="concept-1", 
            content="Understanding fundamentals is crucial for advanced topics",
            confidence=0.90,
            state=ClaimState.VALIDATED,
            type=[ClaimType.CONCEPT],
            tags=["education", "fundamentals"],
            supports=["ml-practice-1"]
        ),
        
        # Target claim (will be in Target Claim section)
        "ml-practice-1": Claim(
            id="ml-practice-1",
            content="Practice machine learning with structured approach",
            confidence=0.85,
            state=ClaimState.EXPLORE,
            type=[ClaimType.GOAL],
            tags=["ml", "practice"],
            supported_by=["research-1", "concept-1"]
        ),
        
        # Direct supported claims (should be in Supported_by Claims)
        "daily-1": Claim(
            id="daily-1",
            content="Daily coding practice improves retention",
            confidence=0.80,
            state=ClaimState.EXPLORE,
            type=[ClaimType.EXAMPLE],
            tags=["daily", "practice"],
            supported_by=["ml-practice-1"]
        ),
        "project-1": Claim(
            id="project-1",
            content="Building real projects solidifies understanding",
            confidence=0.75,
            state=ClaimState.EXPLORE,
            type=[ClaimType.EXAMPLE],
            tags=["project", "hands-on"],
            supported_by=["ml-practice-1"]
        ),
        
        # Semantic similar claims (should be in Relevant Claims)
        "related-1": Claim(
            id="related-1",
            content="Consistent practice leads to mastery",
            confidence=0.88,
            state=ClaimState.VALIDATED,
            type=[ClaimType.CONCEPT],
            tags=["practice", "mastery"]
        ),
        "related-2": Claim(
            id="related-2",
            content="Project-based learning enhances engagement",
            confidence=0.82,
            state=ClaimState.VALIDATED,
            type=[ClaimType.CONCEPT],
            tags=["projects", "engagement"]
        )
    }
    
    return claims


def test_claim_formatters():
    """Test the Claim model formatting methods"""
    print("Testing Claim formatting methods...")
    
    claim = Claim(
        id="test-1",
        content="This is a test claim for formatting",
        confidence=0.85,
        state=ClaimState.EXPLORE,
        type=[ClaimType.CONCEPT],
        tags=["test", "formatting"]
    )
    
    print("format_for_context():")
    try:
        formatted_context = claim.format_for_context()
        print(f"  {formatted_context}")
        
        # Check format matches expected pattern: [c{id} | content | / confidence]
        if formatted_context.startswith(f"[c{claim.id} | ") and f" | / {claim.confidence:.2f}]" in formatted_context:
            print("  PASS: format_for_context() produces correct format")
            context_test_passed = True
        else:
            print("  FAIL: format_for_context() format incorrect")
            context_test_passed = False
    except Exception as e:
        print(f"  ERROR in format_for_context(): {e}")
        context_test_passed = False
    
    print("format_for_llm_analysis():")
    try:
        formatted_analysis = claim.format_for_llm_analysis()
        print(f"  {formatted_analysis}")
        
        # Check that analysis includes expected fields
        required_fields = ["Claim ID:", "Content:", "Confidence:", "State:", "Type:", "Tags:", "Supports:", "Supported By:"]
        missing_fields = [field for field in required_fields if field not in formatted_analysis]
        
        if not missing_fields:
            print("  PASS: format_for_llm_analysis() includes all required fields")
            analysis_test_passed = True
        else:
            print(f"  FAIL: format_for_llm_analysis() missing fields: {missing_fields}")
            analysis_test_passed = False
    except Exception as e:
        print(f"  ERROR in format_for_llm_analysis(): {e}")
        analysis_test_passed = False
    
    return context_test_passed and analysis_test_passed


def test_context_formatting():
    """Test the context formatting template"""
    print("\nTesting context formatting template...")
    
    # Create test claims
    claims = create_test_claims()
    print(f"Created {len(claims)} test claims")
    
    # Set up the different claim groups
    target_claim = claims["ml-practice-1"]
    upward_claims = [claims["research-1"], claims["concept-1"]]  # Supporting claims
    downward_claims = [claims["daily-1"], claims["project-1"]]   # Direct supported claims  
    semantic_claims = [claims["related-1"], claims["related-2"]] # Semantic similar claims
    
    try:
        # Format the context
        formatted_context = format_context_template(
            target_claim, upward_claims, downward_claims, semantic_claims
        )
        print("PASS: Context formatted successfully")
        
        # Display the formatted context
        print("\nFORMATTED CONTEXT OUTPUT:")
        print("=" * 60)
        print(formatted_context)
        print("=" * 60)
        
        # Verify the template structure
        context_lines = formatted_context.split('\n')
        
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
        
        print("\nTemplate Structure Verification:")
        all_sections_found = True
        for section, found in sections_found.items():
            status = "PASS" if found else "FAIL"
            print(f"  {status}: {section}")
            if not found:
                all_sections_found = False
        
        # Check target claim formatting (confidence should be redacted)
        target_section_found = False
        for i, line in enumerate(context_lines):
            if line == "# Target Claim" and i + 1 < len(context_lines):
                target_line = context_lines[i + 1]
                if "confidence_redacted" in target_line:
                    target_section_found = True
                    print("  PASS: Target claim confidence properly redacted")
                else:
                    print("  FAIL: Target claim confidence NOT redacted")
                break
        
        if not target_section_found:
            print("  FAIL: Target claim section not found")
            all_sections_found = False
        
        # Check claim format consistency
        claim_format_lines = [line for line in context_lines if line.startswith("[c") ]
        wrong_format_claims = []
        
        for line in claim_format_lines:
            if not (line.endswith("]") and " | / " in line):
                wrong_format_claims.append(line)
        
        if wrong_format_claims:
            print(f"  FAIL: {len(wrong_format_claims)} claims with wrong format")
            for claim in wrong_format_claims:
                print(f"    {claim}")
            all_sections_found = False
        else:
            print(f"  PASS: All {len(claim_format_lines)} claims have correct format")
        
        return all_sections_found
        
    except Exception as e:
        print(f"ERROR formatting context: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("TESTING UPDATED CONTEXT FORMATTING")
    print("=" * 60)
    
    # Test 1: Claim formatting methods
    format_test_passed = test_claim_formatters()
    
    # Test 2: Context formatting template
    context_test_passed = test_context_formatting()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY:")
    
    if format_test_passed:
        print("  PASS: Claim formatting methods working correctly")
    else:
        print("  FAIL: Claim formatting methods have issues")
    
    if context_test_passed:
        print("  PASS: Context formatting template working correctly")
    else:
        print("  FAIL: Context formatting template has issues")
    
    if format_test_passed and context_test_passed:
        print("\nALL TESTS COMPLETED SUCCESSFULLY!")
        print("The context formatting is working correctly with:")
        print("  • Claim model with proper formatting methods")
        print("  • Approved LLM template format") 
        print("  • Proper section organization")
        print("  • Confidence redaction for target claim")
        print("  • [c{id} | content | / confidence] formatting")
        return True
    else:
        print("\nSOME TESTS FAILED!")
        print("Please check the errors above and fix the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)