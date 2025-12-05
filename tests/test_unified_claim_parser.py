"""
Test suite for Unified Claim Parser

Validates that the unified parser can handle all three incompatible formats
and convert them to the standard bracket format with 90%+ success rate.
"""

import pytest
from typing import List, Dict, Any
from src.processing.unified_claim_parser import UnifiedClaimParser, parse_claims_from_response
from src.core.models import Claim, ClaimType, ClaimState
from datetime import datetime


class TestUnifiedClaimParser:
    """Test suite for unified claim parser"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.parser = UnifiedClaimParser()
        
    def test_parse_bracket_format_success(self):
        """Test successful parsing of bracket format"""
        response_text = """Here are some claims:
[c123 | Machine learning requires large datasets | / 0.9]
[c124 | Neural networks can approximate any function | / 0.8]
[c125 | Deep learning uses multiple layers | / 0.85]"""
        
        claims = parse_claims_from_response(response_text)
        
        assert len(claims) == 3, f"Expected 3 claims, got {len(claims)}"
        
        # Check first claim
        claim1 = claims[0]
        assert claim1.id == "c123"
        assert claim1.content == "Machine learning requires large datasets"
        assert claim1.confidence == 0.9
        
        # Check formatted output
        formatted = claim1.format_for_output()
        assert formatted == "[c123 | Machine learning requires large datasets | / 0.90]"
    
    def test_parse_xml_format_success(self):
        """Test successful parsing of XML format"""
        response_text = """Analysis results:
<claim type="concept" confidence="0.85">Quantum entanglement creates correlations</claim>
<claim type="thesis" confidence="0.75">This enables secure communication</claim>
<claim type="example" confidence="0.9">GPS uses relativistic corrections</claim>"""
        
        claims = parse_claims_from_response(response_text)
        
        assert len(claims) == 3, f"Expected 3 claims, got {len(claims)}"
        
        # Check that claims are converted to standard format
        for claim in claims:
            assert claim.id.startswith("c")
            assert 0.0 <= claim.confidence <= 1.0
            assert len(claim.content) > 0
            assert claim.state == ClaimState.EXPLORE
    
    def test_parse_structured_format_success(self):
        """Test successful parsing of structured format"""
        response_text = """Research findings:
Claim: "Python is dynamically typed" Confidence: 0.95 Type: concept
Claim: "Static typing catches errors early" Confidence: 0.8 Type: thesis
Claim: "Type hints improve documentation" Confidence: 0.85 Type: example"""
        
        claims = parse_claims_from_response(response_text)
        
        assert len(claims) == 3, f"Expected 3 claims, got {len(claims)}"
        
        # Check first claim
        claim1 = claims[0]
        assert claim1.content == "Python is dynamically typed"
        assert claim1.confidence == 0.95
        assert claim1.state == ClaimState.EXPLORE
    
    def test_parse_freeform_fallback(self):
        """Test freeform parsing as fallback"""
        response_text = """Here are some insights:
Machine learning models improve with more data
Neural networks are inspired by biological systems
Deep learning has revolutionized computer vision"""
        
        claims = parse_claims_from_response(response_text)
        
        assert len(claims) >= 2, f"Expected at least 2 claims, got {len(claims)}"
        
        for claim in claims:
            assert claim.id.startswith("c")
            assert claim.confidence >= 0.0
            assert len(claim.content) > 10  # Substantial content
            assert "fallback" in claim.tags or "freeform" in claim.tags
    
    def test_mixed_format_parsing(self):
        """Test parsing response with mixed formats"""
        response_text = """Mixed analysis results:
[c123 | Standard bracket format works | / 0.9]
<claim type="concept" confidence="0.8">XML format also supported</claim>
Claim: "Structured format handled too" Confidence: 0.75 Type: thesis
Freeform text claim as fallback"""
        
        claims = parse_claims_from_response(response_text)
        
        assert len(claims) >= 3, f"Expected at least 3 claims, got {len(claims)}"
        
        # Check that all claims are in standard format
        for claim in claims:
            formatted = claim.format_for_output()
            assert formatted.startswith("[c")
            assert "|" in formatted
            assert "/ " in formatted
            assert formatted.endswith("]")
    
    def test_invalid_confidence_handling(self):
        """Test handling of invalid confidence values"""
        response_text = """Claims with invalid confidence:
[c123 | Valid confidence | / 0.8]
<claim type="concept" confidence="1.5">Too high confidence</claim>
Claim: "Negative confidence" Confidence: -0.2 Type: concept"""
        
        claims = parse_claims_from_response(response_text)
        
        # Should parse valid claims and handle invalid ones gracefully
        assert len(claims) >= 1, "Should parse at least the valid claim"
        
        # Check that valid claim is parsed correctly
        valid_claims = [c for c in claims if c.confidence >= 0.0 and c.confidence <= 1.0]
        assert len(valid_claims) >= 1, "Should have at least one valid claim"
    
    def test_empty_and_invalid_input(self):
        """Test handling of empty and invalid input"""
        # Empty input
        claims = parse_claims_from_response("")
        assert len(claims) == 0, "Empty input should return no claims"
        
        # Only comments and whitespace
        claims = parse_claims_from_response("# Just a comment\n\n   \n# Another comment")
        assert len(claims) == 0, "Comments only should return no claims"
        
        # Malformed bracket format
        claims = parse_claims_from_response("[c123 | Missing confidence]")
        assert len(claims) == 0, "Malformed format should return no claims"
    
    def test_statistics_tracking(self):
        """Test that parsing statistics are tracked correctly"""
        response_text = """Mixed formats:
[c123 | Bracket format | / 0.9]
<claim type="concept" confidence="0.8">XML format</claim>
Claim: "Structured" Confidence: 0.7 Type: thesis
Freeform claim text"""
        
        # Reset and parse
        self.parser.reset_statistics()
        claims = self.parser.parse_claims_from_response(response_text)
        stats = self.parser.get_parse_statistics()
        
        assert stats['bracket'] >= 1, "Should track bracket format parsing"
        assert stats['xml'] >= 1, "Should track XML format parsing"
        assert stats['structured'] >= 1, "Should track structured format parsing"
        assert stats['freeform'] >= 1, "Should track freeform format parsing"
    
    def test_format_validation(self):
        """Test claim format validation"""
        # Valid bracket formats
        valid_formats = [
            "[c123 | Content | / 0.9]",
            "[c456|More content|/0.8]",
            "[c789 | Spaced content | / 0.75 ]",
        ]
        
        for format_str in valid_formats:
            claims = parse_claims_from_response(format_str)
            assert len(claims) == 1, f"Should parse valid format: {format_str}"
            assert claims[0].confidence >= 0.0 and claims[0].confidence <= 1.0
    
    def test_confidence_extraction_from_text(self):
        """Test confidence extraction from freeform text"""
        response_text = """Various confidence indicators:
This claim has confidence: 0.85
Another one with certainty: 75%
This one has probability: 0.9
Final claim with 80% probability"""
        
        claims = parse_claims_from_response(response_text)
        
        assert len(claims) >= 2, "Should extract confidence from text"
        
        # Check that percentages are converted to 0-1 scale
        for claim in claims:
            assert 0.0 <= claim.confidence <= 1.0, f"Invalid confidence: {claim.confidence}"
    
    def test_overall_success_rate(self):
        """Test that overall success rate meets 90%+ target"""
        test_cases = [
            # Bracket format test cases
            "[c123 | Test claim 1 | / 0.9]",
            "[c124 | Test claim 2 | / 0.8]",
            
            # XML format test cases
            '<claim type="concept" confidence="0.85">XML claim 1</claim>',
            '<claim type="thesis" confidence="0.75">XML claim 2</claim>',
            
            # Structured format test cases
            'Claim: "Structured claim 1" Confidence: 0.9 Type: concept',
            'Claim: "Structured claim 2" Confidence: 0.8 Type: thesis',
            
            # Freeform test cases
            "This is a freeform claim with substantial content",
            "Another freeform claim that should be parsed",
        ]
        
        successful_parses = 0
        total_tests = len(test_cases)
        
        for test_case in test_cases:
            try:
                claims = parse_claims_from_response(test_case)
                if len(claims) > 0:
                    # Validate the parsed claim
                    claim = claims[0]
                    if (claim.id and claim.content and 
                        0.0 <= claim.confidence <= 1.0 and
                        claim.state):
                        successful_parses += 1
            except Exception as e:
                print(f"Parse failed for: {test_case[:50]}... - {e}")
                continue
        
        success_rate = (successful_parses / total_tests) * 100
        print(f"Success rate: {successful_parses}/{total_tests} = {success_rate:.1f}%")
        
        # Assert 90%+ success rate
        assert success_rate >= 90.0, f"Success rate {success_rate:.1f}% is below 90% target"
    
    def test_integration_with_existing_code(self):
        """Test integration with existing codebase"""
        # Test that the parser can be imported and used
        from src.processing.unified_claim_parser import get_unified_parser
        
        parser = get_unified_parser()
        assert parser is not None, "Should be able to get parser instance"
        
        # Test parsing with the instance
        claims = parser.parse_claims_from_response("[c123 | Integration test | / 0.9]")
        assert len(claims) == 1, "Parser instance should work correctly"
    
    def test_error_recovery(self):
        """Test error recovery and graceful degradation"""
        # Mix of valid and invalid formats
        response_text = """
[c123 | Valid claim | / 0.9]
[invalid | malformed bracket]
<claim type="concept" confidence="invalid">Invalid confidence</claim>
Claim: "Missing confidence" Type: concept
Valid freeform claim with substantial content"""
        
        # Should not raise exception and should parse valid parts
        claims = parse_claims_from_response(response_text)
        
        # Should parse at least the valid claims
        assert len(claims) >= 2, "Should parse valid claims despite invalid ones"
        
        # All parsed claims should be valid
        for claim in claims:
            assert claim.id, "All claims should have IDs"
            assert claim.content, "All claims should have content"
            assert 0.0 <= claim.confidence <= 1.0, "All claims should have valid confidence"


if __name__ == "__main__":
    # Run basic validation
    parser = UnifiedClaimParser()
    
    print("Testing unified claim parser...")
    
    # Test each format
    bracket_test = "[c123 | Test bracket format | / 0.9]"
    xml_test = '<claim type="concept" confidence="0.8">Test XML format</claim>'
    structured_test = 'Claim: "Test structured" Confidence: 0.75 Type: thesis'
    freeform_test = "Test freeform claim with substantial content"
    
    for test_name, test_input in [
        ("Bracket", bracket_test),
        ("XML", xml_test), 
        ("Structured", structured_test),
        ("Freeform", freeform_test)
    ]:
        claims = parser.parse_claims_from_response(test_input)
        print(f"{test_name}: {len(claims)} claims parsed")
        if claims:
            print(f"  Example: {claims[0].format_for_output()}")
    
    print("All tests completed successfully!")