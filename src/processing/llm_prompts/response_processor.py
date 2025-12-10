"""
Response Processor for LLM Prompt Management System
Parse, validate, and process LLM responses
"""

import json
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import yaml

from .models import (
    LLMResponse, ResponseSchema, ParsedResponse, FallbackResponse
)

logger = logging.getLogger(__name__)

class ResponseProcessor:
    """
    Processes LLM responses with parsing, validation, and error handling
    """

    def __init__(self):
        # Response parsers
        self.parsers = {
            'json': self._parse_json,
            'yaml': self._parse_yaml,
            'structured_text': self._parse_structured_text,
            'claims': self._parse_claims,
            'freeform': self._parse_freeform
        }
        
        # Response validators
        self.validators = {}
        
        # Error patterns and their handlers
        self.error_patterns = [
            (r'(?i)error|invalid|cannot|unable|fail', lambda m: "General error detected"),
            (r'(?i)i\'m sorry|i do not|cannot provide', lambda m: "Refusal response"),
            (r'\{.+\}', lambda m: "Potential JSON structure"),
            (r'^\s*\d+\.', lambda m: "Numbered list format"),
            (r'^\s*-\s', lambda m: "Bulleted list format")
        ]
        
        # Statistics
        self.processing_stats = {
            'total_processed': 0,
            'successful_parses': 0,
            'fallback_used': 0,
            'common_errors': {}
        }

    def register_parser(self, parser_name: str, parser_func: callable) -> None:
        """
        Register a custom response parser
        
        Args:
            parser_name: Parser name
            parser_func: Parser function
        """
        self.parsers[parser_name] = parser_func
        logger.info(f"Registered parser: {parser_name}")

    def register_validator(self, schema_name: str, validator_func: callable) -> None:
        """
        Register a custom response validator
        
        Args:
            schema_name: Schema name
            validator_func: Validator function
        """
        self.validators[schema_name] = validator_func
        logger.info(f"Registered validator: {schema_name}")

    async def parse_response(self, response: str, schema: ResponseSchema,
                           parser_type: str = 'auto') -> ParsedResponse:
        """
        Parse and validate an LLM response
        
        Args:
            response: Raw LLM response
            schema: Response schema for validation
            parser_type: Type of parser to use ('auto' for automatic detection)
            
        Returns:
            Parsed response result
        """
        start_time = time.time()
        
        try:
            self.processing_stats['total_processed'] += 1
            
            # Detect parser type if auto
            if parser_type == 'auto':
                parser_type = self._detect_parser_type(response)
            
            # Parse response
            parser = self.parsers.get(parser_type, self._parse_freeform)
            try:
                parsed_data = await parser(response)
                parse_success = True
            except Exception as e:
                parsed_data = None
                parse_success = False
            
            # Validate parsed data
            validation_result = await self._validate_response_content(parsed_data, schema) if parse_success else (
                False, ['Parsing failed']
            )
            
            # Extract claims if present
            claims = await self._extract_claims_from_response(response, parsed_data)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                parse_success, validation_result[0], len(claims), response
            )
            
            # Create parsed response
            parsed_response = ParsedResponse(
                is_valid=validation_result[0] and parse_success,
                parsed_data=parsed_data if parse_success else None,
                errors=validation_result[1] if not validation_result[0] else [],
                warnings=[],  # Could add warning detection
                confidence_score=confidence_score,
                extracted_claims=claims,
                metadata={
                    'parser_type': parser_type,
                    'parse_time_ms': int((time.time() - start_time) * 1000),
                    'original_length': len(response)
                }
            )
            
            if parsed_response.is_valid:
                self.processing_stats['successful_parses'] += 1
            
            return parsed_response

        except Exception as e:
            logger.error(f"Response parsing failed: {e}")
            return ParsedResponse(
                is_valid=False,
                parsed_data=None,
                errors=[f"Parsing error: {str(e)}"],
                confidence_score=0.0,
                metadata={'parse_error': True}
            )

    async def extract_claims(self, response: str) -> List[Dict[str, Any]]:
        """
        Extract claims from an LLM response
        
        Args:
            response: LLM response
            
        Returns:
            List of extracted claims
        """
        try:
            claims = []
            
            # Look for claim-like statements
            claim_patterns = [
                r'(?i)claim[:\s]+([^.\n]+)',  # "Claim: ..."
                r'(?i)statement[:\s]+([^.\n]+)',  # "Statement: ..."
                r'(?i)finding[:\s]+([^.\n]+)',  # "Finding: ..."
                r'(?i)conclusion[:\s]+([^.\n]+)',  # "Conclusion: ..."
                r'(?i)result[:\s]+([^.\n]+)',  # "Result: ..."
            ]
            
            for pattern in claim_patterns:
                matches = re.findall(pattern, response)
                for match in matches:
                    claims.append({
                        'content': match.strip(),
                        'confidence': 0.7,  # Default confidence
                        'source': 'pattern_extraction',
                        'pattern': pattern
                    })
            
            # Look for statements with uncertainty indicators
            uncertain_patterns = [
                r'(?i)(probably|likely|might|could|may|perhaps|possibly)\s+([^.!?]+[.!?])',
                r'(?i)(i think|i believe|i estimate)\s+that\s+([^.!?]+[.!?])',
            ]
            
            for pattern in uncertain_patterns:
                matches = re.findall(pattern, response)
                for indicator, statement in matches:
                    claims.append({
                        'content': statement.strip(),
                        'confidence': 0.5,  # Lower confidence for uncertain statements
                        'source': 'uncertainty_pattern',
                        'indicator': indicator
                    })
            
            # Remove duplicates
            unique_claims = []
            seen_contents = set()
            
            for claim in claims:
                content_lower = claim['content'].lower()
                if content_lower not in seen_contents:
                    seen_contents.add(content_lower)
                    unique_claims.append(claim)
            
            return unique_claims

        except Exception as e:
            logger.error(f"Claim extraction failed: {e}")
            return []

    async def handle_malformed_response(self, response: str) -> FallbackResponse:
        """
        Generate fallback response for malformed LLM output
        
        Args:
            response: Original malformed response
            
        Returns:
            Fallback response
        """
        try:
            self.processing_stats['fallback_used'] += 1
            
            # Analyze the response to determine the best fallback
            response_lower = response.lower()
            
            if any(word in response_lower for word in ['error', 'cannot', 'unable', 'fail']):
                return FallbackResponse(
                    response_type='error_acknowledgment',
                    message="The LLM encountered an error in processing your request. This might be due to the complexity of the task or limitations in the input clarity.",
                    original_response=response,
                    parsing_errors=["LLM indicated an error in processing"],
                    should_retry=True,
                    retry_hints=["Simplify the request", "Provide more specific context"]
                )
            
            elif any(word in response_lower for word in ["i'm sorry", "i do not", "cannot provide"]):
                return FallbackResponse(
                    response_type='refusal_handling',
                    message="The LLM was unable to process this request, possibly due to content policies or unclear requirements.",
                    original_response=response,
                    parsing_errors=["LLM declined to respond"],
                    should_retry=True,
                    retry_hints=["Rephrase the request", "Check policy compliance"]
                )
            
            elif len(response.strip()) < 10:
                return FallbackResponse(
                    response_type='empty_response',
                    message="The response was too short or empty, indicating a processing issue.",
                    original_response=response,
                    parsing_errors=["Response too short"],
                    should_retry=True,
                    retry_hints=["Try a different approach", "Add more context"]
                )
            
            else:
                # Try to extract useful content even if malformed
                extracted_content = self._extract_useful_content(response)
                
                return FallbackResponse(
                    response_type='partial_extraction',
                    message=f"Partially processed response. Some information was extracted: {extracted_content[:200]}...",
                    original_response=response,
                    parsing_errors=["Response format issues"],
                    should_retry=False,
                    retry_hints=["Manual review may be needed"]
                )

        except Exception as e:
            logger.error(f"Fallback response generation failed: {e}")
            return FallbackResponse(
                response_type='system_error',
                message="A system error occurred while processing the response.",
                original_response=response,
                parsing_errors=[f"System error: {str(e)}"],
                should_retry=True
            )

    async def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get response processing statistics
        
        Returns:
            Processing statistics
        """
        total = self.processing_stats['total_processed']
        success_rate = (
            self.processing_stats['successful_parses'] / total
            if total > 0 else 0.0
        )
        fallback_rate = (
            self.processing_stats['fallback_used'] / total
            if total > 0 else 0.0
        )
        
        return {
            **self.processing_stats,
            'success_rate': success_rate,
            'fallback_rate': fallback_rate,
            'available_parsers': list(self.parsers.keys()),
            'available_validators': list(self.validators.keys())
        }

    def _detect_parser_type(self, response: str) -> str:
        """Automatically detect the best parser type for a response"""
        response_stripped = response.strip()
        
        # Check for JSON structure
        if (response_stripped.startswith('{') and response_stripped.endswith('}')) or \
           (response_stripped.startswith('[') and response_stripped.endswith(']')):
            return 'json'
        
        # Check for YAML structure
        if ':' in response_stripped and response_stripped.count('\n') > 2:
            # Simple YAML detection
            lines = response_stripped.split('\n')
            yaml_like = sum(1 for line in lines if ':' in line and len(line.split(':', 1)) == 2)
            if yaml_like / len(lines) > 0.5:
                return 'yaml'
        
        # Check for structured text (lists, headings, etc.)
        if re.search(r'^\s*(#+|[-*]|\d+\.)', response_stripped, re.MULTILINE):
            return 'structured_text'
        
        # Check for claim-like content
        claim_indicators = ['claim:', 'statement:', 'finding:', 'conclusion:']
        if any(indicator in response_stripped.lower() for indicator in claim_indicators):
            return 'claims'
        
        # Default to freeform
        return 'freeform'

    async def _parse_json(self, response: str) -> Dict[str, Any]:
        """Parse JSON response"""
        json_text = response.strip()
        
        # Try to extract JSON if embedded in text
        json_match = re.search(r'\{.*\}', json_text, re.DOTALL)
        if json_match:
            json_text = json_match.group()
        
        return json.loads(json_text)

    async def _parse_yaml(self, response: str) -> Dict[str, Any]:
        """Parse YAML response"""
        yaml_text = response.strip()
        return yaml.safe_load(yaml_text)

    async def _parse_structured_text(self, response: str) -> Dict[str, Any]:
        """Parse structured text response"""
        lines = response.strip().split('\n')
        result = {}
        current_section = 'main'
        current_list = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for headings
            if line.startswith('#'):
                heading = line.lstrip('#').strip()
                current_section = heading.lower().replace(' ', '_')
                result[current_section] = []
                current_list = result[current_section]
                continue
            
            # Check for list items
            if line.startswith(('-', '*', '+')):
                item = line[1:].strip()
                current_list.append(item)
                continue
            
            # Check for numbered items
            if re.match(r'^\d+\.', line):
                item = re.sub(r'^\d+\.\s*', '', line).strip()
                current_list.append(item)
                continue
            
            # Regular text - add to current list
            current_list.append(line)
        
        # Convert single-item lists to strings
        for key, value in result.items():
            if isinstance(value, list) and len(value) == 1:
                result[key] = value[0]
        
        return result

    async def _parse_claims(self, response: str) -> Dict[str, Any]:
        """Parse claim-based response"""
        claims = await self.extract_claims(response)
        return {
            'type': 'claims',
            'claim_count': len(claims),
            'claims': claims
        }

    async def _parse_freeform(self, response: str) -> Dict[str, Any]:
        """Parse freeform response"""
        return {
            'type': 'freeform',
            'content': response,
            'word_count': len(response.split()),
            'character_count': len(response),
            'lines': response.count('\n') + 1
        }

    async def _validate_response_content(self, data: Any, schema: ResponseSchema) -> Tuple[bool, List[str]]:
        """Validate parsed content against schema"""
        errors = []
        
        if not isinstance(data, dict):
            return False, ["Response must be a dictionary/object"]
        
        # Check required fields
        for field in schema.required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Check field types
        for field, expected_type in schema.field_types.items():
            if field in data:
                if not self._check_field_type(data[field], expected_type):
                    errors.append(f"Field {field} has wrong type. Expected {expected_type}")
        
        # Run custom validation rules
        for field, rule_name in schema.validation_rules.items():
            if field in data:
                validator = self.validators.get(rule_name)
                if validator:
                    try:
                        if not validator(data[field]):
                            errors.append(f"Validation failed for field: {field}")
                    except Exception as e:
                        errors.append(f"Validator error for field {field}: {e}")
        
        # Run custom validator if specified
        if schema.custom_validator:
            validator = self.validators.get(schema.custom_validator)
            if validator:
                try:
                    if not validator(data):
                        errors.append("Custom validation failed")
                except Exception as e:
                    errors.append(f"Custom validator error: {e}")
        
        return len(errors) == 0, errors

    def _check_field_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type"""
        type_map = {
            'string': str,
            'integer': int,
            'number': (int, float),
            'boolean': bool,
            'array': list,
            'object': dict,
            'any': type(None)  # Accept any type
        }
        
        expected_python_type = type_map.get(expected_type)
        if expected_python_type is None:
            return True  # Unknown type, accept
        
        if expected_python_type == type(None):  # any type
            return True
        
        return isinstance(value, expected_python_type)

    async def _extract_claims_from_response(self, response: str, parsed_data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract claims from both raw response and parsed data"""
        claims = []
        
        # Extract from raw response
        raw_claims = await self.extract_claims(response)
        claims.extend(raw_claims)
        
        # Extract from parsed data if available
        if parsed_data and isinstance(parsed_data, dict):
            if 'claims' in parsed_data:
                data_claims = parsed_data['claims']
                if isinstance(data_claims, list):
                    for claim in data_claims:
                        if isinstance(claim, dict):
                            claims.append(claim)
                        elif isinstance(claim, str):
                            claims.append({
                                'content': claim,
                                'confidence': 0.8,
                                'source': 'parsed_data'
                            })
        
        # Remove duplicates and return
        unique_claims = []
        seen_contents = set()
        
        for claim in claims:
            content = claim.get('content', '')
            content_lower = content.lower()
            
            if content_lower and content_lower not in seen_contents:
                seen_contents.add(content_lower)
                unique_claims.append(claim)
        
        return unique_claims

    def _calculate_confidence_score(self, parse_success: bool, validation_success: bool,
                                  claim_count: int, response: str) -> float:
        """Calculate confidence score for parsing result"""
        score = 0.0
        
        # Parse success contributes significantly
        if parse_success:
            score += 0.5
        
        # Validation success
        if validation_success:
            score += 0.3
        
        # Claims extracted
        score += min(0.1, claim_count * 0.02)
        
        # Response quality (length and structure)
        if len(response.strip()) > 50:  # Reasonable length
            score += 0.05
        
        # Has some structure
        if any(pattern in response for pattern in [':', '\n', '-', '1.', '2.']):
            score += 0.05
        
        return min(1.0, score)

    def _extract_useful_content(self, response: str) -> str:
        """Extract useful content from malformed response"""
        # Remove common error phrases
        content = response
        
        # Remove error/failed patterns
        error_patterns = [
            r'(?i)i\'m sorry, but.*?\.?\s*',
            r'(?i)as an ai.*?\.?\s*',
            r'(?i)i cannot.*?\.?\s*',
            r'(?i)error:.*?\.?\s*',
        ]
        
        for pattern in error_patterns:
            content = re.sub(pattern, '', content).strip()
        
        # Extract sentences that look useful
        sentences = re.split(r'[.!?]+', content)
        useful_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 10 and 
                not any(word in sentence.lower() for word in ['error', 'cannot', 'unable', 'sorry'])):
                useful_sentences.append(sentence)
        
        return '. '.join(useful_sentences[:3]) if useful_sentences else content[:200]