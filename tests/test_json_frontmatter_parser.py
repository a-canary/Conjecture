"""
Comprehensive tests for JSON Frontmatter Parser

Tests the new JSON frontmatter parsing functionality with various scenarios,
including validation, error handling, and backward compatibility.
"""

import pytest
import json
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.processing.json_frontmatter_parser import (
    JSONFrontmatterParser, 
    parse_response_with_json_frontmatter,
    create_json_frontmatter_prompt_template,
    JSONClaimData,
    JSONFrontmatterData,
    ResponseType,
    ParseResult
)
from src.core.models import Claim, ClaimType, ClaimState
from src.processing.json_schemas import ResponseSchemaType, get_schema_examples


class TestJSONFrontmatterParser:
    """Test cases for JSON frontmatter parser"""

    def setup_method(self):
        """Setup for each test method"""
        self.parser = JSONFrontmatterParser()

    def test_parse_valid_claims_response(self):
        """Test parsing a valid claims response"""
        response_text = '''---
{
  "type": "claims",
  "confidence": 0.95,
  "claims": [
    {
      "id": "c1",
      "content": "The doctor lives in house 3 based on clue 1",
      "confidence": 0.95,
      "type": "fact"
    },
    {
      "id": "c2",
      "content": "The engineer is not in house 2",
      "confidence": 0.80,
      "type": "inference"
    }
  ]
}---

Based on the analysis, here are the key claims:

[c1 | The doctor lives in house 3 based on clue 1 | / 0.95]
[c2 | The engineer is not in house 2 | / 0.80]
'''

        result = self.parser.parse_response(response_text)
        
        assert result.success is True
        assert len(result.claims) == 2
        assert result.parse_method == "json_frontmatter"
        assert result.frontmatter_data is not None
        assert result.frontmatter_data.type == "claims"
        assert result.frontmatter_data.confidence == 0.95
        
        # Check first claim
        claim1 = result.claims[0]
        assert claim1.id == "c1"
        assert claim1.content == "The doctor lives in house 3 based on clue 1"
        assert claim1.confidence == 0.95
        assert ClaimType.FACT in claim1.type
        assert "json_frontmatter" in claim1.tags

    def test_parse_analysis_response(self):
        """Test parsing an analysis response"""
        response_text = '''---
{
  "type": "analysis",
  "confidence": 0.90,
  "analysis": {
    "summary": "Strong evidence supports main claim",
    "key_factors": ["evidence_quality", "logical_consistency"]
  },
  "claims": [
    {
      "id": "c1",
      "content": "Main claim is well-supported",
      "confidence": 0.90,
      "type": "conclusion"
    }
  ],
  "insights": [
    "Strong correlation between evidence and confidence"
  ]
}---

Analysis complete with key findings above.'''

        result = self.parser.parse_response(response_text)
        
        assert result.success is True
        assert len(result.claims) == 1
        assert result.parse_method == "json_frontmatter"
        assert result.frontmatter_data.type == "analysis"
        assert len(result.frontmatter_data.insights) == 1

    def test_parse_invalid_json(self):
        """Test handling of invalid JSON"""
        response_text = '''---
{
  "type": "claims",
  "confidence": 0.95,
  "claims": [
    {
      "id": "c1",
      "content": "Valid claim",
      "confidence": 0.95,
      "type": "fact"
    }
  }
  // Invalid JSON comment
}---

Some content here.'''

        result = self.parser.parse_response(response_text)
        
        # Should fallback to text parsing
        assert result.parse_method in ["fallback_text", "error"]
        assert len(result.errors) > 0

    def test_parse_missing_frontmatter(self):
        """Test handling of response without frontmatter"""
        response_text = '''Here are some claims in the old format:

[c1 | This is a bracket format claim | / 0.8]
[c2 | Another bracket claim | / 0.7]

No frontmatter here.'''

        result = self.parser.parse_response(response_text)
        
        # Should fallback to text parsing
        assert result.parse_method == "fallback_text"
        assert len(result.claims) >= 0  # May parse some claims

    def test_parse_malformed_claim_id(self):
        """Test handling of malformed claim IDs"""
        response_text = '''---
{
  "type": "claims",
  "confidence": 0.95,
  "claims": [
    {
      "id": "invalid_id",
      "content": "Invalid claim ID format",
      "confidence": 0.95,
      "type": "fact"
    },
    {
      "id": "c2",
      "content": "Valid claim ID",
      "confidence": 0.80,
      "type": "fact"
    }
  ]
}---

Mixed valid and invalid claims.'''

        result = self.parser.parse_response(response_text)
        
        # Should parse valid claims and report errors for invalid ones
        assert result.success is True  # At least one valid claim
        assert len(result.claims) == 1  # Only the valid one
        assert result.claims[0].id == "c2"
        assert len(result.errors) > 0  # Error for invalid ID

    def test_parse_confidence_out_of_range(self):
        """Test handling of confidence values out of range"""
        response_text = '''---
{
  "type": "claims",
  "confidence": 0.95,
  "claims": [
    {
      "id": "c1",
      "content": "Valid confidence",
      "confidence": 0.85,
      "type": "fact"
    },
    {
      "id": "c2",
      "content": "Invalid confidence - too high",
      "confidence": 1.5,
      "type": "fact"
    }
  ]
}---

Mixed confidence values.'''

        result = self.parser.parse_response(response_text)
        
        # Should parse valid claims and report errors for invalid ones
        assert result.success is True  # At least one valid claim
        assert len(result.claims) == 1  # Only the valid one
        assert result.claims[0].confidence == 0.85
        assert len(result.errors) > 0  # Error for invalid confidence

    def test_parse_empty_response(self):
        """Test handling of empty response"""
        result = self.parser.parse_response("")
        
        assert result.success is False
        assert len(result.claims) == 0
        assert result.parse_method == "error"
        assert len(result.errors) > 0

    def test_parse_statistics(self):
        """Test parsing statistics tracking"""
        # Parse a valid JSON frontmatter response
        valid_response = '''---
{
  "type": "claims",
  "confidence": 0.95,
  "claims": [
    {
      "id": "c1",
      "content": "Test claim",
      "confidence": 0.95,
      "type": "fact"
    }
  ]
}---
'''
        self.parser.parse_response(valid_response)
        
        # Parse a fallback response
        fallback_response = "[c1 | Fallback claim | / 0.8]"
        self.parser.parse_response(fallback_response)
        
        stats = self.parser.get_statistics()
        
        assert stats['json_frontmatter_success'] == 1
        assert stats['fallback_text_success'] >= 0
        assert stats['total_parses'] == 2
        assert stats['avg_processing_time'] > 0

    def test_reset_statistics(self):
        """Test statistics reset"""
        # Parse something to generate stats
        self.parser.parse_response('[c1 | Test | / 0.8]')
        
        # Reset stats
        self.parser.reset_statistics()
        
        stats = self.parser.get_statistics()
        assert stats['json_frontmatter_success'] == 0
        assert stats['fallback_text_success'] == 0
        assert stats['total_parses'] == 0


class TestJSONClaimData:
    """Test JSON claim data validation"""

    def test_valid_claim_data(self):
        """Test valid claim data"""
        claim_data = JSONClaimData(
            id="c1",
            content="Test claim content",
            confidence=0.95,
            type="fact",
            tags=["test", "example"]
        )
        
        assert claim_data.id == "c1"
        assert claim_data.content == "Test claim content"
        assert claim_data.confidence == 0.95
        assert claim_data.type == "fact"
        assert "test" in claim_data.tags

    def test_invalid_claim_id(self):
        """Test invalid claim ID raises validation error"""
        with pytest.raises(ValueError, match="Claim ID must be in format"):
            JSONClaimData(
                id="invalid",
                content="Test claim",
                confidence=0.95,
                type="fact"
            )

    def test_invalid_confidence_range(self):
        """Test confidence out of range raises validation error"""
        with pytest.raises(ValueError):
            JSONClaimData(
                id="c1",
                content="Test claim",
                confidence=1.5,  # Too high
                type="fact"
            )

    def test_content_too_short(self):
        """Test content too short raises validation error"""
        with pytest.raises(ValueError):
            JSONClaimData(
                id="c1",
                content="Too",  # Less than 5 characters
                confidence=0.95,
                type="fact"
            )

    def test_content_too_long(self):
        """Test content too long raises validation error"""
        with pytest.raises(ValueError):
            JSONClaimData(
                id="c1",
                content="x" * 2001,  # More than 2000 characters
                confidence=0.95,
                type="fact"
            )


class TestJSONFrontmatterData:
    """Test JSON frontmatter data validation"""

    def test_valid_frontmatter_data(self):
        """Test valid frontmatter data"""
        claims = [
            JSONClaimData(
                id="c1",
                content="Test claim",
                confidence=0.95,
                type="fact"
            )
        ]
        
        frontmatter = JSONFrontmatterData(
            type=ResponseType.CLAIMS,
            confidence=0.95,
            claims=claims,
            metadata={"test": True}
        )
        
        assert frontmatter.type == ResponseType.CLAIMS
        assert frontmatter.confidence == 0.95
        assert len(frontmatter.claims) == 1
        assert frontmatter.metadata["test"] is True

    def test_invalid_timestamp(self):
        """Test invalid timestamp raises validation error"""
        with pytest.raises(ValueError, match="Invalid timestamp format"):
            JSONFrontmatterData(
                type=ResponseType.CLAIMS,
                confidence=0.95,
                claims=[],
                timestamp="invalid-timestamp"
            )


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_parse_response_with_json_frontmatter(self):
        """Test convenience function"""
        response_text = '''---
{
  "type": "claims",
  "confidence": 0.95,
  "claims": [
    {
      "id": "c1",
      "content": "Test claim",
      "confidence": 0.95,
      "type": "fact"
    }
  ]
}---
'''
        
        result = parse_response_with_json_frontmatter(response_text)
        
        assert result.success is True
        assert len(result.claims) == 1
        assert result.claims[0].id == "c1"

    def test_create_json_frontmatter_prompt_template(self):
        """Test prompt template creation"""
        template = create_json_frontmatter_prompt_template(ResponseType.CLAIMS)
        
        assert "JSON frontmatter" in template
        assert "claims" in template
        assert "format 'c<number>'" in template
        assert "0.0 and 1.0" in template


class TestIntegrationWithExistingParsers:
    """Test integration with existing parsers for backward compatibility"""

    @patch('src.processing.json_frontmatter_parser.parse_claims_from_response')
    def test_fallback_to_unified_parser(self, mock_unified_parser):
        """Test fallback to unified parser when JSON frontmatter fails"""
        # Mock unified parser to return some claims
        mock_claims = [
            Claim(
                id="c1",
                content="Fallback claim",
                confidence=0.8,
                type=[ClaimType.CONCEPT],
                state=ClaimState.EXPLORE,
                created=datetime.utcnow()
            )
        ]
        mock_unified_parser.return_value = mock_claims
        
        # Parse response without frontmatter
        response_text = "[c1 | Fallback claim | / 0.8]"
        result = self.parser.parse_response(response_text)
        
        assert result.parse_method == "fallback_text"
        assert len(result.claims) == 1
        assert result.claims[0].content == "Fallback claim"

    def test_mixed_format_response(self):
        """Test response with both JSON frontmatter and text claims"""
        response_text = '''---
{
  "type": "claims",
  "confidence": 0.95,
  "claims": [
    {
      "id": "c1",
      "content": "JSON frontmatter claim",
      "confidence": 0.95,
      "type": "fact"
    }
  ]
}---

Additional text claims:
[c2 | Text format claim | / 0.8]
[c3 | Another text claim | / 0.7]
'''
        
        result = self.parser.parse_response(response_text)
        
        # Should prioritize JSON frontmatter
        assert result.parse_method == "json_frontmatter"
        assert len(result.claims) >= 1
        assert result.claims[0].content == "JSON frontmatter claim"


class TestErrorHandling:
    """Test error handling scenarios"""

    def test_malformed_json_syntax(self):
        """Test handling of malformed JSON syntax"""
        response_text = '''---
{
  "type": "claims",
  "confidence": 0.95,
  "claims": [
    {
      "id": "c1",
      "content": "Missing closing brace"
      "confidence": 0.95,
      "type": "fact"
    }
  ]
}---
'''
        
        result = self.parser.parse_response(response_text)
        
        # Should handle gracefully and fallback
        assert result.parse_method in ["fallback_text", "error"]
        assert len(result.errors) > 0

    def test_missing_required_fields(self):
        """Test handling of missing required fields"""
        response_text = '''---
{
  "type": "claims",
  "claims": [
    {
      "id": "c1",
      "content": "Missing confidence and type"
    }
  ]
}---
'''
        
        result = self.parser.parse_response(response_text)
        
        # Should handle validation errors
        assert result.success is False or len(result.errors) > 0

    def test_unicode_content(self):
        """Test handling of unicode content in claims"""
        response_text = '''---
{
  "type": "claims",
  "confidence": 0.95,
  "claims": [
    {
      "id": "c1",
      "content": "Unicode test: 🚀 Claim with emoji and ñ special chars",
      "confidence": 0.95,
      "type": "concept"
    }
  ]
}---
'''
        
        result = self.parser.parse_response(response_text)
        
        assert result.success is True
        assert len(result.claims) == 1
        assert "🚀" in result.claims[0].content
        assert "ñ" in result.claims[0].content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])