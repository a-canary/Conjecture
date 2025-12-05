"""
Unified Claim Parser

Handles parsing of all three incompatible claim formats:
1. Bracket format: [c{id} | content | / confidence]
2. XML format: <claim type="" confidence="">content</claim>
3. Structured format: "Claim: "...", Confidence: ..., Type: ...

Converts all parsed claims to the standard [c{id} | content | / confidence] format.
"""

import re
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from src.core.models import Claim, ClaimType, ClaimState
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedClaim:
    """Represents a parsed claim before conversion to standard format"""
    id: str
    content: str
    confidence: float
    claim_type: Optional[str] = None
    raw_format: Optional[str] = None
    raw_text: Optional[str] = None


class UnifiedClaimParser:
    """
    Unified parser that handles multiple claim formats and converts them to standard format.
    
    Priority order for parsing (updated for XML optimization):
    1. JSON frontmatter → XML format → Bracket format → Structured format → Freeform
    """
    
    def __init__(self):
        self.bracket_patterns = [
            # Standard bracket format: [c123 | content | / 0.95]
            r'\[c(\d+)\s*\|\s*([^|]+?)\s*\|\s*/\s*([0-9.]+)\s*\]',
            # Compact bracket format: [c123|content|/0.95]
            r'\[c(\d+)\|([^|]+?)\|/([0-9.]+)\]',
            # Bracket with extra spaces
            r'\[c(\d+)\s*\|\s*([^|]+?)\s*\|\s*/\s*([0-9.]+)\s*\]',
        ]
        
        self.xml_pattern = r'<claim\s+type="([^"]*)"\s+confidence="([^"]*)"[^>]*>(.*?)</claim>'
        
        self.structured_patterns = [
            # "Claim: "...", Confidence: ..., Type: ..."
            r'Claim:\s*"([^"]+)"\s*Confidence:\s*([\d.]+)\s*Type:\s*(\w+)',
            # Case insensitive variations
            r'claim:\s*"([^"]+)"\s*confidence:\s*([\d.]+)\s*type:\s*(\w+)',
            # With optional quotes around content
            r'Claim:\s*"?([^"]+)"?\s*Confidence:\s*([\d.]+)\s*Type:\s*(\w+)',
        ]
        
        # Statistics for debugging
        self.parse_stats = {
            'json_frontmatter': 0,
            'bracket': 0,
            'xml': 0,
            'structured': 0,
            'freeform': 0,
            'errors': 0
        }
    
    def parse_claims_from_response(self, response_text: str) -> List[Claim]:
        """
        Parse claims from LLM response text using JSON frontmatter first, then fallback to legacy formats.
        
        Args:
            response_text: Raw LLM response text
            
        Returns:
            List of Claim objects in standard format
        """
        # Try JSON frontmatter first (new primary method)
        try:
            from .json_frontmatter_parser import parse_response_with_json_frontmatter
            result = parse_response_with_json_frontmatter(response_text)
            if result.success:
                self.parse_stats['json_frontmatter'] = len(result.claims)
                logger.info(f"Parsed {len(result.claims)} claims using JSON frontmatter format")
                return result.claims
        except ImportError:
            logger.debug("JSON frontmatter parser not available, using legacy formats")
        except Exception as e:
            logger.debug(f"JSON frontmatter parsing failed: {e}")
        
        # Fallback to legacy text formats
        parsed_claims = []
        
        # Try each format in order of preference (XML prioritized for optimization)
        for format_name, parser_func in [
            ('xml', self._parse_xml_format),  # Prioritize XML for experiment
            ('bracket', self._parse_bracket_format),
            ('structured', self._parse_structured_format),
            ('freeform', self._parse_freeform_format)
        ]:
            try:
                claims = parser_func(response_text)
                if claims:
                    parsed_claims.extend(claims)
                    self.parse_stats[format_name] += len(claims)
                    logger.debug(f"Parsed {len(claims)} claims using {format_name} format")
                    # Don't break - try all formats to catch mixed responses
            except Exception as e:
                logger.debug(f"Failed to parse with {format_name} format: {e}")
                self.parse_stats['errors'] += 1
                continue
        
        # Convert all parsed claims to standard format
        standard_claims = []
        for parsed_claim in parsed_claims:
            try:
                standard_claim = self._convert_to_standard_format(parsed_claim)
                if standard_claim:
                    standard_claims.append(standard_claim)
            except Exception as e:
                logger.warning(f"Failed to convert claim to standard format: {e}")
                continue
        
        logger.info(f"Parsed {len(standard_claims)} claims total using legacy fallback. Stats: {self.parse_stats}")
        return standard_claims
    
    def _parse_bracket_format(self, text: str) -> List[ParsedClaim]:
        """Parse bracket format: [c{id} | content | / confidence]"""
        claims = []
        
        for pattern in self.bracket_patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
            for match in matches:
                try:
                    claim_id, content, confidence_str = match
                    confidence = float(confidence_str)
                    
                    if self._validate_confidence(confidence):
                        claim = ParsedClaim(
                            id=f"c{claim_id}",
                            content=content.strip(),
                            confidence=confidence,
                            raw_format="bracket",
                            raw_text=f"[c{claim_id} | {content.strip()} | / {confidence}]"
                        )
                        claims.append(claim)
                except (ValueError, IndexError) as e:
                    logger.debug(f"Failed to parse bracket claim: {e}")
                    continue
        
        return claims
    
    def _parse_xml_format(self, text: str) -> List[ParsedClaim]:
        """Parse XML format: <claim type="" confidence="">content</claim>"""
        claims = []
        
        # Enhanced XML pattern to handle multiline content and optional attributes
        enhanced_xml_patterns = [
            # Standard XML with multiline content
            r'<claim\s+type="([^"]*)"\s+confidence="([^"]*)"[^>]*>(.*?)</claim>',
            # XML with id attribute
            r'<claim\s+id="([^"]*)"\s+type="([^"]*)"\s+confidence="([^"]*)"[^>]*>(.*?)</claim>',
            # XML with different attribute order
            r'<claim\s+confidence="([^"]*)"\s+type="([^"]*)"[^>]*>(.*?)</claim>',
            # Simplified XML format
            r'<claim[^>]*type="([^"]*)"[^>]*confidence="([^"]*)"[^>]*>(.*?)</claim>',
        ]
        
        claim_counter = 1
        
        for pattern in enhanced_xml_patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
            for match in matches:
                try:
                    if len(match) == 4:  # Pattern with id attribute
                        claim_id, claim_type, confidence_str, content = match
                        if not claim_id or claim_id.strip() == "":
                            claim_id = f"c{claim_counter:03d}"
                    else:  # Standard pattern
                        claim_type, confidence_str, content = match
                        claim_id = f"c{claim_counter:03d}"
                    
                    confidence = float(confidence_str)
                    
                    if self._validate_confidence(confidence):
                        # Clean up content - remove extra whitespace and XML tags if present
                        content = content.strip()
                        # Remove any nested XML tags that might be in the content
                        content = re.sub(r'<[^>]+>', '', content)
                        content = re.sub(r'\s+', ' ', content)
                        
                        claim = ParsedClaim(
                            id=claim_id,
                            content=content,
                            confidence=confidence,
                            claim_type=claim_type.strip() if claim_type else None,
                            raw_format="xml",
                            raw_text=f'<claim type="{claim_type}" confidence="{confidence}">{content[:100]}...</claim>'
                        )
                        claims.append(claim)
                        claim_counter += 1
                except (ValueError, IndexError) as e:
                    logger.debug(f"Failed to parse XML claim: {e}")
                    continue
        
        # If no claims found with enhanced patterns, try the original pattern as fallback
        if not claims:
            matches = re.findall(self.xml_pattern, text, re.MULTILINE | re.DOTALL)
            for match in matches:
                try:
                    claim_type, confidence_str, content = match
                    confidence = float(confidence_str)
                    
                    if self._validate_confidence(confidence):
                        claim_id = f"c{claim_counter:03d}"
                        content = content.strip()
                        content = re.sub(r'<[^>]+>', '', content)
                        content = re.sub(r'\s+', ' ', content)
                        
                        claim = ParsedClaim(
                            id=claim_id,
                            content=content,
                            confidence=confidence,
                            claim_type=claim_type.strip() if claim_type else None,
                            raw_format="xml",
                            raw_text=f'<claim type="{claim_type}" confidence="{confidence}">{content[:100]}...</claim>'
                        )
                        claims.append(claim)
                        claim_counter += 1
                except (ValueError, IndexError) as e:
                    logger.debug(f"Failed to parse XML claim: {e}")
                    continue
        
        return claims
    
    def _parse_structured_format(self, text: str) -> List[ParsedClaim]:
        """Parse structured format: "Claim: "...", Confidence: ..., Type: ..." """
        claims = []
        
        for pattern in self.structured_patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
            for match in matches:
                try:
                    content, confidence_str, claim_type = match
                    confidence = float(confidence_str)
                    
                    if self._validate_confidence(confidence):
                        # Generate ID since structured format doesn't include it
                        claim_id = f"c{int(time.time() * 1000) % 10000000:07d}"
                        
                        claim = ParsedClaim(
                            id=claim_id,
                            content=content.strip(),
                            confidence=confidence,
                            claim_type=claim_type.strip() if claim_type else None,
                            raw_format="structured",
                            raw_text=f'Claim: "{content}" Confidence: {confidence} Type: {claim_type}'
                        )
                        claims.append(claim)
                except (ValueError, IndexError) as e:
                    logger.debug(f"Failed to parse structured claim: {e}")
                    continue
        
        return claims
    
    def _parse_freeform_format(self, text: str) -> List[ParsedClaim]:
        """Parse freeform text as claims (fallback method)"""
        claims = []
        
        # Split by lines and look for substantial content
        lines = text.strip().split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Skip empty lines, comments, and obviously non-claim lines
            if (not line or 
                line.startswith('#') or 
                line.startswith('//') or 
                len(line) < 20 or
                line.startswith('<') and line.endswith('>') or  # Skip XML-like tags
                'Claim:' in line or  # Skip already parsed formats
                line.startswith('[')):  # Skip bracket formats
                continue
            
            # Try to extract confidence from the line
            confidence = self._extract_confidence_from_text(line)
            
            # Generate ID for freeform claims
            claim_id = f"c{int(time.time() * 1000 + i) % 10000000:07d}"
            
            claim = ParsedClaim(
                id=claim_id,
                content=line,
                confidence=confidence,
                raw_format="freeform",
                raw_text=line
            )
            claims.append(claim)
        
        return claims
    
    def _extract_confidence_from_text(self, text: str) -> float:
        """Extract confidence score from freeform text"""
        # Look for confidence indicators
        confidence_patterns = [
            r'confidence[:\s]*([0-9.]+)',
            r'certainty[:\s]*([0-9.]+)',
            r'probability[:\s]*([0-9.]+)',
            r'([0-9]+\.[0-9]+)%',  # Percentage
            r'([0-9]+)%',  # Whole percentage
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    conf_str = match.group(1)
                    conf_val = float(conf_str)
                    # Convert percentage to 0-1 scale
                    if conf_val > 1.0:
                        conf_val = conf_val / 100.0
                    return min(max(conf_val, 0.0), 1.0)
                except ValueError:
                    continue
        
        # Default confidence for freeform claims
        return 0.7
    
    def _validate_confidence(self, confidence: float) -> bool:
        """Validate confidence score is in valid range"""
        return 0.0 <= confidence <= 1.0
    
    def _convert_to_standard_format(self, parsed_claim: ParsedClaim) -> Optional[Claim]:
        """Convert parsed claim to standard Claim object with enhanced type mapping"""
        try:
            # Enhanced claim type mapping for XML optimization
            claim_type = ClaimType.CONCEPT  # Default
            if parsed_claim.claim_type:
                try:
                    # Normalize claim type string
                    claim_type_str = parsed_claim.claim_type.lower().strip()
                    
                    # Enhanced type mapping
                    type_mapping = {
                        'fact': ClaimType.FACT,
                        'concept': ClaimType.CONCEPT,
                        'example': ClaimType.EXAMPLE,
                        'goal': ClaimType.GOAL,
                        'reference': ClaimType.REFERENCE,
                        'hypothesis': ClaimType.HYPOTHESIS,
                        'assertion': ClaimType.ASSERTION,
                        'thesis': ClaimType.THESIS,
                        'question': ClaimType.QUESTION,
                        'task': ClaimType.TASK,
                    }
                    
                    claim_type = type_mapping.get(claim_type_str, ClaimType.CONCEPT)
                    
                except (ValueError, AttributeError):
                    # Fall back to concept if type is invalid
                    claim_type = ClaimType.CONCEPT
                    logger.debug(f"Unknown claim type '{parsed_claim.claim_type}', using CONCEPT")
            
            # Create claim with standard formatting
            claim = Claim(
                id=parsed_claim.id,
                content=parsed_claim.content,
                confidence=parsed_claim.confidence,
                type=[claim_type],
                state=ClaimState.EXPLORE,
                created=datetime.utcnow(),
                tags=[f"parsed_{parsed_claim.raw_format}", "auto_generated", "xml_optimized"]
            )
            
            return claim
            
        except Exception as e:
            logger.error(f"Failed to convert parsed claim to standard format: {e}")
            return None
    
    def format_claim_for_output(self, claim: Claim) -> str:
        """Format claim in standard bracket format for output"""
        return f"[c{claim.id} | {claim.content} | / {claim.confidence:.2f}]"
    
    def get_parse_statistics(self) -> Dict[str, int]:
        """Get parsing statistics for debugging"""
        return self.parse_stats.copy()
    
    def reset_statistics(self):
        """Reset parsing statistics"""
        self.parse_stats = {
            'json_frontmatter': 0,
            'bracket': 0,
            'xml': 0,
            'structured': 0,
            'freeform': 0,
            'errors': 0
        }


# Global instance for reuse
_unified_parser = None

def get_unified_parser() -> UnifiedClaimParser:
    """Get or create the unified parser instance"""
    global _unified_parser
    if _unified_parser is None:
        _unified_parser = UnifiedClaimParser()
    return _unified_parser


def parse_claims_from_response(response_text: str) -> List[Claim]:
    """
    Convenience function to parse claims from response text.
    
    Args:
        response_text: Raw LLM response text
        
    Returns:
        List of Claim objects in standard format
    """
    parser = get_unified_parser()
    return parser.parse_claims_from_response(response_text)


def format_claim_for_output(claim: Claim) -> str:
    """
    Convenience function to format claim in standard bracket format.
    
    Args:
        claim: Claim object to format
        
    Returns:
        Formatted claim string
    """
    parser = get_unified_parser()
    return parser.format_claim_for_output(claim)