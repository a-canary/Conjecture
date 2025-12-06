"""
JSON Frontmatter Parser for LLM Responses

Provides structured parsing of LLM responses using JSON frontmatter format.
This replaces the multiple incompatible text formats with a single, reliable
JSON-based approach for 99%+ parsing success rates.

JSON Frontmatter Format:
```json
---
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
}
---

Based on the analysis, here are the key claims:

[c1 | The doctor lives in house 3 based on clue 1 | / 0.95]
[c2 | The engineer is not in house 2 | / 0.80]
```
"""

import json
import re
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field, field_validator, ValidationError

from src.core.models import Claim, ClaimType, ClaimState
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ResponseType(str, Enum):
    """Supported response types in JSON frontmatter"""
    CLAIMS = "claims"
    ANALYSIS = "analysis"
    VALIDATION = "validation"
    INSTRUCTION_SUPPORT = "instruction_support"
    ERROR = "error"


class JSONClaimData(BaseModel):
    """Individual claim data in JSON frontmatter"""
    id: str = Field(..., description="Unique claim identifier")
    content: str = Field(..., min_length=5, max_length=2000, description="Claim content")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    type: Optional[str] = Field(None, description="Claim type")
    tags: Optional[List[str]] = Field(default_factory=list, description="Additional tags")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

    @field_validator('id')
    @classmethod
    def validate_claim_id(cls, v):
        """Validate claim ID format"""
        if not re.match(r'^c\d+$', v):
            raise ValueError(f"Claim ID must be in format 'c<number>', got: {v}")
        return v

    @field_validator('type')
    @classmethod
    def validate_claim_type(cls, v):
        """Validate claim type if provided"""
        if v is not None:
            try:
                # Validate against known claim types
                ClaimType(v.lower())
            except ValueError:
                # Allow custom types but log warning
                logger.warning(f"Unknown claim type: {v}")
        return v


class JSONFrontmatterData(BaseModel):
    """Complete JSON frontmatter structure"""
    type: ResponseType = Field(..., description="Response type")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Overall confidence")
    claims: Optional[List[JSONClaimData]] = Field(default_factory=list, description="List of claims")
    analysis: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Analysis data")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    version: str = Field(default="1.0", description="Format version")
    timestamp: Optional[str] = Field(None, description="Response timestamp")

    @field_validator('timestamp')
    def validate_timestamp(cls, v):
        """Validate timestamp format"""
        if v is not None:
            try:
                datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError(f"Invalid timestamp format: {v}")
        return v


@dataclass
class ParseResult:
    """Result of parsing JSON frontmatter"""
    success: bool
    claims: List[Claim]
    frontmatter_data: Optional[JSONFrontmatterData]
    content_text: str
    errors: List[str]
    parse_method: str  # 'json_frontmatter', 'fallback_text', 'error'
    processing_time: float


class JSONFrontmatterParser:
    """
    Parser for JSON frontmatter format with robust error handling and fallback.
    
    Primary parsing method: JSON frontmatter
    Fallback methods: Existing text formats for backward compatibility
    """

    def __init__(self):
        self.stats = {
            'json_frontmatter_success': 0,
            'fallback_text_success': 0,
            'parse_errors': 0,
            'total_parses': 0,
            'avg_processing_time': 0.0
        }
        self._frontmatter_pattern = re.compile(r'^---\s*\n(.*?)\n---\s*\n(.*)$', re.MULTILINE | re.DOTALL)

    def parse_response(self, response_text: str) -> ParseResult:
        """
        Parse LLM response using JSON frontmatter format with fallback.
        
        Args:
            response_text: Raw LLM response text
            
        Returns:
            ParseResult with parsed claims and metadata
        """
        start_time = time.time()
        self.stats['total_parses'] += 1
        
        # Try JSON frontmatter first
        try:
            result = self._parse_json_frontmatter(response_text)
            if result.success:
                processing_time = time.time() - start_time
                result.processing_time = processing_time
                self.stats['json_frontmatter_success'] += 1
                self._update_avg_processing_time(processing_time)
                logger.info(f"Successfully parsed {len(result.claims)} claims using JSON frontmatter")
                return result
        except Exception as e:
            logger.debug(f"JSON frontmatter parsing failed: {e}")
        
        # Fallback to text-based parsing
        try:
            result = self._parse_fallback_text(response_text)
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            self.stats['fallback_text_success'] += 1
            self._update_avg_processing_time(processing_time)
            logger.info(f"Fallback parsing succeeded with {len(result.claims)} claims")
            return result
        except Exception as e:
            logger.error(f"All parsing methods failed: {e}")
        
        # Create error result
        processing_time = time.time() - start_time
        self.stats['parse_errors'] += 1
        self._update_avg_processing_time(processing_time)
        
        return ParseResult(
            success=False,
            claims=[],
            frontmatter_data=None,
            content_text=response_text,
            errors=["All parsing methods failed"],
            parse_method="error",
            processing_time=processing_time
        )

    def _parse_json_frontmatter(self, response_text: str) -> ParseResult:
        """Parse JSON frontmatter from response text"""
        # Extract frontmatter
        match = self._frontmatter_pattern.match(response_text.strip())
        if not match:
            raise ValueError("No JSON frontmatter found in response")
        
        json_text, content_text = match.groups()
        
        # Parse JSON
        try:
            frontmatter_dict = json.loads(json_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in frontmatter: {e}")
        
        # Validate structure
        try:
            frontmatter_data = JSONFrontmatterData(**frontmatter_dict)
        except ValidationError as e:
            raise ValueError(f"Invalid frontmatter structure: {e}")
        
        # Extract claims
        claims = []
        errors = []
        
        if frontmatter_data.claims:
            for claim_data in frontmatter_data.claims:
                try:
                    claim = self._convert_json_claim_to_claim(claim_data, frontmatter_data)
                    claims.append(claim)
                except Exception as e:
                    error_msg = f"Failed to convert claim {claim_data.id}: {e}"
                    errors.append(error_msg)
                    logger.warning(error_msg)
        
        return ParseResult(
            success=len(claims) > 0 or frontmatter_data.type == ResponseType.ANALYSIS,
            claims=claims,
            frontmatter_data=frontmatter_data,
            content_text=content_text.strip(),
            errors=errors,
            parse_method="json_frontmatter",
            processing_time=0.0  # Will be set by caller
        )

    def _parse_fallback_text(self, response_text: str) -> ParseResult:
        """Fallback parsing using existing text-based methods"""
        try:
            from .unified_claim_parser import parse_claims_from_response
            claims = parse_claims_from_response(response_text)
            
            return ParseResult(
                success=len(claims) > 0,
                claims=claims,
                frontmatter_data=None,
                content_text=response_text,
                errors=[],
                parse_method="fallback_text",
                processing_time=0.0  # Will be set by caller
            )
        except ImportError:
            raise ValueError("Fallback parser not available")

    def _convert_json_claim_to_claim(self, json_claim: JSONClaimData, frontmatter: JSONFrontmatterData) -> Claim:
        """Convert JSON claim data to standard Claim object"""
        # Determine claim type
        claim_type = ClaimType.CONCEPT  # Default
        if json_claim.type:
            try:
                claim_type = ClaimType(json_claim.type.lower())
            except ValueError:
                # Use concept for unknown types
                claim_type = ClaimType.CONCEPT
        
        # Combine tags
        tags = ["json_frontmatter", "auto_generated"]
        if json_claim.tags:
            tags.extend(json_claim.tags)
        if frontmatter.type.value:
            tags.append(f"response_type:{frontmatter.type.value}")
        
        # Create claim
        claim = Claim(
            id=json_claim.id,
            content=json_claim.content,
            confidence=json_claim.confidence,
            type=[claim_type],
            state=ClaimState.EXPLORE,
            tags=tags,
            created=datetime.utcnow(),
            updated=datetime.utcnow()
        )
        
        return claim

    def _update_avg_processing_time(self, processing_time: float):
        """Update average processing time statistics"""
        total = self.stats['total_parses']
        current_avg = self.stats['avg_processing_time']
        self.stats['avg_processing_time'] = (current_avg * (total - 1) + processing_time) / total

    def get_statistics(self) -> Dict[str, Any]:
        """Get parsing statistics"""
        total = self.stats['total_parses']
        if total == 0:
            return self.stats.copy()
        
        return {
            **self.stats,
            'json_frontmatter_success_rate': self.stats['json_frontmatter_success'] / total,
            'fallback_success_rate': self.stats['fallback_text_success'] / total,
            'error_rate': self.stats['parse_errors'] / total
        }

    def reset_statistics(self):
        """Reset parsing statistics"""
        self.stats = {
            'json_frontmatter_success': 0,
            'fallback_text_success': 0,
            'parse_errors': 0,
            'total_parses': 0,
            'avg_processing_time': 0.0
        }


# Global instance for reuse
_json_parser = None

def get_json_frontmatter_parser() -> JSONFrontmatterParser:
    """Get or create the JSON frontmatter parser instance"""
    global _json_parser
    if _json_parser is None:
        _json_parser = JSONFrontmatterParser()
    return _json_parser


def parse_response_with_json_frontmatter(response_text: str) -> ParseResult:
    """
    Convenience function to parse response using JSON frontmatter.
    
    Args:
        response_text: Raw LLM response text
        
    Returns:
        ParseResult with parsed claims and metadata
    """
    parser = get_json_frontmatter_parser()
    return parser.parse_response(response_text)


# Utility functions for creating JSON frontmatter prompts
def create_json_frontmatter_prompt_template(response_type: ResponseType, examples: List[Dict[str, Any]] = None) -> str:
    """
    Create a prompt template that requests JSON frontmatter format.
    
    Args:
        response_type: Type of response expected
        examples: Optional examples to include in prompt
        
    Returns:
        Prompt template string
    """
    template_parts = [
        "Please format your response using JSON frontmatter for reliable parsing.",
        "",
        "## REQUIRED FORMAT:",
        "```json",
        "---",
        "{",
        '  "type": "' + response_type.value + '",',
        '  "confidence": 0.95,',
        '  "claims": [',
        '    {',
        '      "id": "c1",',
        '      "content": "Your claim here",',
        '      "confidence": 0.95,',
        '      "type": "fact"',
        '    }',
        "  ]",
        "}",
        "---",
        "```",
        "",
        "## REQUIREMENTS:",
        "- Include JSON frontmatter at the very beginning",
        "- Use valid JSON syntax",
        "- Include claim IDs in format 'c1', 'c2', etc.",
        "- Provide confidence scores between 0.0 and 1.0",
        "- Use appropriate claim types: fact, concept, example, goal, reference, assertion, thesis, hypothesis, question, task",
        "",
        "## EXAMPLE:"
    ]
    
    if examples:
        for i, example in enumerate(examples):
            template_parts.append(f"\n### Example {i+1}:")
            template_parts.append("```json")
            template_parts.append("---")
            template_parts.append(json.dumps(example, indent=2))
            template_parts.append("---")
            template_parts.append("```")
    else:
        # Add default example
        default_example = {
            "type": response_type.value,
            "confidence": 0.95,
            "claims": [
                {
                    "id": "c1",
                    "content": "Example claim based on analysis",
                    "confidence": 0.95,
                    "type": "fact"
                }
            ]
        }
        template_parts.append("```json")
        template_parts.append("---")
        template_parts.append(json.dumps(default_example, indent=2))
        template_parts.append("---")
        template_parts.append("```")
    
    template_parts.extend([
        "",
        "After the JSON frontmatter, you can include additional explanation or analysis in plain text.",
        "",
        "## YOUR TASK:"
    ])
    
    return "\n".join(template_parts)