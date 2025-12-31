"""
Enhanced Error Handling and Validation for JSON Frontmatter Processing

Provides comprehensive error handling, validation, and recovery mechanisms
for the JSON frontmatter parsing system.
"""

import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from src.utils.logging import get_logger

logger = get_logger(__name__)

class ErrorSeverity(str, Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(str, Enum):
    """Error categories for better classification"""
    JSON_SYNTAX = "json_syntax"
    JSON_VALIDATION = "json_validation"
    FRONTMATTER_PARSING = "frontmatter_parsing"
    CLAIM_VALIDATION = "claim_validation"
    SCHEMA_VALIDATION = "schema_validation"
    FALLBACK_PARSING = "fallback_parsing"
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"

@dataclass
class ParsingError:
    """Structured error information"""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    recovery_suggestions: List[str]
    retry_count: int = 0

@dataclass
class ValidationResult:
    """Result of validation operation"""
    is_valid: bool
    errors: List[ParsingError]
    warnings: List[str]
    data: Optional[Dict[str, Any]] = None

class ErrorRecoveryStrategy:
    """Strategies for recovering from parsing errors"""
    
    @staticmethod
    def get_recovery_actions(error: ParsingError) -> List[str]:
        """Get recovery actions for a specific error"""
        actions = []
        
        if error.category == ErrorCategory.JSON_SYNTAX:
            actions.extend([
                "Attempt to fix common JSON syntax errors",
                "Try parsing with fallback text parser",
                "Request LLM to regenerate response with proper format"
            ])
        
        elif error.category == ErrorCategory.FRONTMATTER_PARSING:
            actions.extend([
                "Check for malformed frontmatter delimiters",
                "Verify JSON structure within frontmatter",
                "Attempt to extract JSON from response manually"
            ])
        
        elif error.category == ErrorCategory.CLAIM_VALIDATION:
            actions.extend([
                "Skip invalid claims and process valid ones",
                "Attempt to correct claim ID format",
                "Normalize confidence values to valid range"
            ])
        
        elif error.category == ErrorCategory.SCHEMA_VALIDATION:
            actions.extend([
                "Use schema-compatible response format",
                "Include all required fields in response",
                "Validate field types and constraints"
            ])
        
        elif error.category == ErrorCategory.FALLBACK_PARSING:
            actions.extend([
                "Try alternative text parsing patterns",
                "Use simple line-by-line extraction",
                "Request simplified response format"
            ])
        
        return actions

class EnhancedErrorHandler:
    """Enhanced error handler with recovery and monitoring"""
    
    def __init__(self, max_retry_attempts: int = 3):
        self.max_retry_attempts = max_retry_attempts
        self.error_history: List[ParsingError] = []
        self.error_stats: Dict[ErrorCategory, int] = {}
        self.recovery_stats: Dict[str, int] = {}
        
    def handle_parsing_error(
        self, 
        error: Exception, 
        context: str = "",
        response_text: str = ""
    ) -> ParsingError:
        """Handle and categorize parsing errors"""
        
        # Determine error category and severity
        category, severity = self._categorize_error(error)
        
        # Generate error ID
        error_id = f"err_{int(time.time() * 1000) % 1000000:06d}"
        
        # Extract error details
        details = self._extract_error_details(error, response_text)
        
        # Generate recovery suggestions
        recovery_suggestions = ErrorRecoveryStrategy.get_recovery_actions(
            ParsingError(
                error_id=error_id,
                category=category,
                severity=severity,
                message=str(error),
                details=details,
                timestamp=datetime.utcnow(),
                recovery_suggestions=[]
            )
        )
        
        # Create structured error
        parsing_error = ParsingError(
            error_id=error_id,
            category=category,
            severity=severity,
            message=str(error),
            details=details,
            timestamp=datetime.utcnow(),
            recovery_suggestions=recovery_suggestions
        )
        
        # Track error
        self._track_error(parsing_error)
        
        # Log error with context
        self._log_error(parsing_error, context, response_text)
        
        return parsing_error
    
    def validate_json_frontmatter(
        self, 
        response_text: str,
        strict_mode: bool = False
    ) -> ValidationResult:
        """Validate JSON frontmatter format comprehensively"""
        
        errors = []
        warnings = []
        
        try:
            # Check for frontmatter delimiters
            if not response_text.strip().startswith('---'):
                errors.append(ParsingError(
                    error_id=f"val_{int(time.time() * 1000) % 1000000:06d}",
                    category=ErrorCategory.FRONTMATTER_PARSING,
                    severity=ErrorSeverity.HIGH,
                    message="Response does not start with frontmatter delimiter '---'",
                    details={"response_start": response_text[:100]},
                    timestamp=datetime.utcnow(),
                    recovery_suggestions=["Add '---' at the beginning of response"]
                ))
            
            # Extract frontmatter
            frontmatter_match = self._extract_frontmatter(response_text)
            if not frontmatter_match:
                errors.append(ParsingError(
                    error_id=f"val_{int(time.time() * 1000) % 1000000:06d}",
                    category=ErrorCategory.FRONTMATTER_PARSING,
                    severity=ErrorSeverity.HIGH,
                    message="No valid frontmatter structure found",
                    details={"response_length": len(response_text)},
                    timestamp=datetime.utcnow(),
                    recovery_suggestions=[
                        "Ensure frontmatter is properly formatted with '---' delimiters",
                        "Check JSON syntax within frontmatter"
                    ]
                ))
                return ValidationResult(False, errors, warnings)
            
            json_text, content_text = frontmatter_match
            
            # Validate JSON syntax
            try:
                json_data = json.loads(json_text)
            except json.JSONDecodeError as e:
                errors.append(ParsingError(
                    error_id=f"val_{int(time.time() * 1000) % 1000000:06d}",
                    category=ErrorCategory.JSON_SYNTAX,
                    severity=ErrorSeverity.HIGH,
                    message=f"JSON syntax error: {str(e)}",
                    details={
                        "json_text": json_text[:500],
                        "error_line": getattr(e, 'lineno', 'unknown'),
                        "error_column": getattr(e, 'colno', 'unknown')
                    },
                    timestamp=datetime.utcnow(),
                    recovery_suggestions=[
                        "Fix JSON syntax errors",
                        "Use JSON validator to check format",
                        "Ensure proper quoting and escaping"
                    ]
                ))
                return ValidationResult(False, errors, warnings)
            
            # Validate frontmatter structure
            structure_errors = self._validate_frontmatter_structure(json_data, strict_mode)
            errors.extend(structure_errors)
            
            # Validate claims if present
            if "claims" in json_data:
                claim_errors = self._validate_claims_array(json_data["claims"])
                errors.extend(claim_errors)
            
            # Generate warnings for potential issues
            warnings.extend(self._generate_warnings(json_data, content_text))
            
            is_valid = len(errors) == 0
            result_data = json_data if is_valid else None
            
            return ValidationResult(is_valid, errors, warnings, result_data)
            
        except Exception as e:
            errors.append(ParsingError(
                error_id=f"val_{int(time.time() * 1000) % 1000000:06d}",
                category=ErrorCategory.UNKNOWN_ERROR,
                severity=ErrorSeverity.CRITICAL,
                message=f"Validation error: {str(e)}",
                details={"exception_type": type(e).__name__},
                timestamp=datetime.utcnow(),
                recovery_suggestions=["Check validation logic", "Report this issue"]
            ))
            
            return ValidationResult(False, errors, warnings)
    
    def attempt_error_recovery(
        self, 
        error: ParsingError, 
        response_text: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """Attempt to recover from parsing errors"""
        
        recovery_attempts = 0
        max_attempts = 3
        
        while recovery_attempts < max_attempts:
            recovery_attempts += 1
            
            try:
                if error.category == ErrorCategory.JSON_SYNTAX:
                    success, result = self._recover_json_syntax_error(response_text)
                elif error.category == ErrorCategory.FRONTMATTER_PARSING:
                    success, result = self._recover_frontmatter_error(response_text)
                elif error.category == ErrorCategory.CLAIM_VALIDATION:
                    success, result = self._recover_claim_validation_error(response_text)
                else:
                    success, result = False, None
                
                if success:
                    logger.info(f"Error recovery successful on attempt {recovery_attempts}")
                    self._track_recovery(f"recovery_{error.category.value}")
                    return True, result
                    
            except Exception as e:
                logger.warning(f"Recovery attempt {recovery_attempts} failed: {e}")
        
        logger.error(f"Error recovery failed after {max_attempts} attempts")
        return False, None
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        total_errors = len(self.error_history)
        
        if total_errors == 0:
            return {
                "total_errors": 0,
                "error_rate": 0.0,
                "error_categories": {},
                "recovery_success_rate": 0.0,
                "most_common_error": None
            }
        
        # Calculate error rates by category
        category_stats = {}
        for category, count in self.error_stats.items():
            category_stats[category.value] = {
                "count": count,
                "percentage": (count / total_errors) * 100
            }
        
        # Calculate recovery success rate
        total_recovery_attempts = sum(self.recovery_stats.values())
        successful_recoveries = self.recovery_stats.get("successful", 0)
        recovery_success_rate = (successful_recoveries / total_recovery_attempts) if total_recovery_attempts > 0 else 0.0
        
        # Find most common error
        most_common_error = max(self.error_stats.items(), key=lambda x: x[1])[0].value if self.error_stats else None
        
        return {
            "total_errors": total_errors,
            "error_rate": total_errors / max(total_errors, 1),  # Avoid division by zero
            "error_categories": category_stats,
            "recovery_success_rate": recovery_success_rate,
            "recovery_attempts": total_recovery_attempts,
            "most_common_error": most_common_error,
            "recent_errors": [
                {
                    "error_id": error.error_id,
                    "category": error.category.value,
                    "severity": error.severity.value,
                    "message": error.message,
                    "timestamp": error.timestamp.isoformat()
                }
                for error in self.error_history[-10:]  # Last 10 errors
            ]
        }
    
    def reset_statistics(self):
        """Reset error statistics"""
        self.error_history.clear()
        self.error_stats.clear()
        self.recovery_stats.clear()
    
    def _categorize_error(self, error: Exception) -> Tuple[ErrorCategory, ErrorSeverity]:
        """Categorize error and determine severity"""
        error_message = str(error).lower()
        error_type = type(error).__name__
        
        # JSON syntax errors
        if "jsondecodeerror" in error_type or "json" in error_message:
            return ErrorCategory.JSON_SYNTAX, ErrorSeverity.HIGH
        
        # Frontmatter parsing errors
        if "frontmatter" in error_message or "delimiter" in error_message:
            return ErrorCategory.FRONTMATTER_PARSING, ErrorSeverity.HIGH
        
        # Validation errors
        if "validation" in error_message or "schema" in error_message:
            return ErrorCategory.SCHEMA_VALIDATION, ErrorSeverity.MEDIUM
        
        # Claim validation errors
        if "claim" in error_message or "confidence" in error_message:
            return ErrorCategory.CLAIM_VALIDATION, ErrorSeverity.MEDIUM
        
        # Network errors
        if "connection" in error_message or "timeout" in error_message:
            return ErrorCategory.NETWORK_ERROR, ErrorSeverity.MEDIUM
        
        # Default to unknown
        return ErrorCategory.UNKNOWN_ERROR, ErrorSeverity.LOW
    
    def _extract_error_details(self, error: Exception, response_text: str) -> Dict[str, Any]:
        """Extract detailed error information"""
        details = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "response_length": len(response_text) if response_text else 0,
            "response_preview": response_text[:200] if response_text else ""
        }
        
        # Add JSON-specific details
        if hasattr(error, 'lineno'):
            details["error_line"] = error.lineno
        if hasattr(error, 'colno'):
            details["error_column"] = error.colno
        if hasattr(error, 'pos'):
            details["error_position"] = error.pos
        
        return details
    
    def _extract_frontmatter(self, response_text: str) -> Optional[Tuple[str, str]]:
        """Extract frontmatter from response text"""
        import re
        frontmatter_pattern = re.compile(r'^---\s*\n(.*?)\n---\s*\n(.*)$', re.MULTILINE | re.DOTALL)
        match = frontmatter_pattern.match(response_text.strip())
        
        if match:
            return match.groups()
        return None
    
    def _validate_frontmatter_structure(
        self, 
        json_data: Dict[str, Any], 
        strict_mode: bool
    ) -> List[ParsingError]:
        """Validate frontmatter structure"""
        errors = []
        
        # Check required fields
        required_fields = ["type"]
        for field in required_fields:
            if field not in json_data:
                errors.append(ParsingError(
                    error_id=f"val_{int(time.time() * 1000) % 1000000:06d}",
                    category=ErrorCategory.SCHEMA_VALIDATION,
                    severity=ErrorSeverity.HIGH,
                    message=f"Missing required field: {field}",
                    details={"missing_field": field, "present_fields": list(json_data.keys())},
                    timestamp=datetime.utcnow(),
                    recovery_suggestions=[f"Add '{field}' field to response"]
                ))
        
        # Validate type field
        if "type" in json_data:
            valid_types = ["claims", "analysis", "validation", "instruction_support", "error"]
            if json_data["type"] not in valid_types:
                errors.append(ParsingError(
                    error_id=f"val_{int(time.time() * 1000) % 1000000:06d}",
                    category=ErrorCategory.SCHEMA_VALIDATION,
                    severity=ErrorSeverity.HIGH,
                    message=f"Invalid response type: {json_data['type']}",
                    details={"invalid_type": json_data["type"], "valid_types": valid_types},
                    timestamp=datetime.utcnow(),
                    recovery_suggestions=[f"Use one of: {', '.join(valid_types)}"]
                ))
        
        # Strict mode validations
        if strict_mode:
            if "timestamp" not in json_data:
                errors.append(ParsingError(
                    error_id=f"val_{int(time.time() * 1000) % 1000000:06d}",
                    category=ErrorCategory.SCHEMA_VALIDATION,
                    severity=ErrorSeverity.MEDIUM,
                    message="Missing timestamp field (required in strict mode)",
                    details={},
                    timestamp=datetime.utcnow(),
                    recovery_suggestions=["Add timestamp field to response"]
                ))
        
        return errors
    
    def _validate_claims_array(self, claims: List[Dict[str, Any]]) -> List[ParsingError]:
        """Validate claims array"""
        errors = []
        
        if not isinstance(claims, list):
            errors.append(ParsingError(
                error_id=f"val_{int(time.time() * 1000) % 1000000:06d}",
                category=ErrorCategory.CLAIM_VALIDATION,
                severity=ErrorSeverity.HIGH,
                message="Claims field must be an array",
                details={"claims_type": type(claims).__name__},
                timestamp=datetime.utcnow(),
                recovery_suggestions=["Ensure claims is formatted as an array"]
            ))
            return errors
        
        for i, claim in enumerate(claims):
            claim_errors = self._validate_single_claim(claim, i)
            errors.extend(claim_errors)
        
        return errors
    
    def _validate_single_claim(
        self, 
        claim: Dict[str, Any], 
        index: int
    ) -> List[ParsingError]:
        """Validate a single claim"""
        errors = []
        
        # Check required fields
        required_fields = ["id", "content", "confidence"]
        for field in required_fields:
            if field not in claim:
                errors.append(ParsingError(
                    error_id=f"val_{int(time.time() * 1000) % 1000000:06d}",
                    category=ErrorCategory.CLAIM_VALIDATION,
                    severity=ErrorSeverity.HIGH,
                    message=f"Claim {index}: Missing required field '{field}'",
                    details={"claim_index": index, "missing_field": field},
                    timestamp=datetime.utcnow(),
                    recovery_suggestions=[f"Add '{field}' field to claim"]
                ))
        
        # Validate claim ID format
        if "id" in claim:
            claim_id = str(claim["id"])
            if not claim_id.startswith("c") or not claim_id[1:].isdigit():
                errors.append(ParsingError(
                    error_id=f"val_{int(time.time() * 1000) % 1000000:06d}",
                    category=ErrorCategory.CLAIM_VALIDATION,
                    severity=ErrorSeverity.MEDIUM,
                    message=f"Claim {index}: Invalid ID format '{claim_id}'",
                    details={"claim_index": index, "claim_id": claim_id},
                    timestamp=datetime.utcnow(),
                    recovery_suggestions=["Use format 'c1', 'c2', etc. for claim IDs"]
                ))
        
        # Validate confidence range
        if "confidence" in claim:
            confidence = claim["confidence"]
            if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
                errors.append(ParsingError(
                    error_id=f"val_{int(time.time() * 1000) % 1000000:06d}",
                    category=ErrorCategory.CLAIM_VALIDATION,
                    severity=ErrorSeverity.MEDIUM,
                    message=f"Claim {index}: Invalid confidence value {confidence}",
                    details={"claim_index": index, "confidence": confidence},
                    timestamp=datetime.utcnow(),
                    recovery_suggestions=["Use confidence values between 0.0 and 1.0"]
                ))
        
        return errors
    
    def _generate_warnings(
        self, 
        json_data: Dict[str, Any], 
        content_text: str
    ) -> List[str]:
        """Generate warnings for potential issues"""
        warnings = []
        
        # Check for empty content
        if content_text and len(content_text.strip()) == 0:
            warnings.append("No content found after frontmatter")
        
        # Check for very long content
        if content_text and len(content_text) > 10000:
            warnings.append("Very long content after frontmatter - consider summarizing")
        
        # Check for missing optional but useful fields
        if "confidence" not in json_data and "claims" in json_data:
            warnings.append("Missing overall confidence field")
        
        return warnings
    
    def _recover_json_syntax_error(self, response_text: str) -> Tuple[bool, Dict[str, Any]]:
        """Attempt to recover from JSON syntax errors"""
        try:
            # Try to extract and fix common JSON syntax issues
            frontmatter_match = self._extract_frontmatter(response_text)
            if not frontmatter_match:
                return False, None
            
            json_text, _ = frontmatter_match
            
            # Common fixes
            json_text = json_text.replace("'", '"')  # Replace single quotes
            json_text = json_text.replace(',}', '}')  # Remove trailing commas
            json_text = json_text.replace(',]', ']')  # Remove trailing commas in arrays
            
            # Try to parse fixed JSON
            fixed_data = json.loads(json_text)
            return True, fixed_data
            
        except Exception:
            return False, None
    
    def _recover_frontmatter_error(self, response_text: str) -> Tuple[bool, Dict[str, Any]]:
        """Attempt to recover from frontmatter errors"""
        try:
            # Try to find JSON-like content even without proper delimiters
            import re
            
            # Look for JSON object patterns
            json_pattern = re.compile(r'\{.*\}', re.DOTALL)
            matches = json_pattern.findall(response_text)
            
            for match in matches:
                try:
                    json_data = json.loads(match)
                    # Add type if missing
                    if "type" not in json_data and "claims" in json_data:
                        json_data["type"] = "claims"
                    return True, json_data
                except json.JSONDecodeError:
                    continue
            
            return False, None
            
        except Exception:
            return False, None
    
    def _recover_claim_validation_error(self, response_text: str) -> Tuple[bool, Dict[str, Any]]:
        """Attempt to recover from claim validation errors"""
        try:
            # Try to extract claims using text parsing as fallback
            from .unified_claim_parser import parse_claims_from_response
            claims = parse_claims_from_response(response_text)
            
            if claims:
                # Convert to JSON format
                claims_data = []
                for claim in claims:
                    claim_dict = {
                        "id": claim.id,
                        "content": claim.content,
                        "confidence": claim.confidence,
                        "type": claim.type[0].value if claim.type else "concept"
                    }
                    claims_data.append(claim_dict)
                
                json_data = {
                    "type": "claims",
                    "confidence": 0.8,  # Default confidence
                    "claims": claims_data
                }
                return True, json_data
            
            return False, None
            
        except Exception:
            return False, None
    
    def _track_error(self, error: ParsingError):
        """Track error for statistics"""
        self.error_history.append(error)
        
        # Update category stats
        if error.category not in self.error_stats:
            self.error_stats[error.category] = 0
        self.error_stats[error.category] += 1
    
    def _track_recovery(self, recovery_type: str):
        """Track recovery attempts"""
        if recovery_type not in self.recovery_stats:
            self.recovery_stats[recovery_type] = 0
        self.recovery_stats[recovery_type] += 1
    
    def _log_error(
        self, 
        error: ParsingError, 
        context: str, 
        response_text: str
    ):
        """Log error with appropriate level"""
        log_message = f"[{error.error_id}] {error.category.value}: {error.message}"
        
        if error.severity == ErrorSeverity.CRITICAL:
            logger.error(log_message)
        elif error.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
        
        # Log recovery suggestions
        if error.recovery_suggestions:
            logger.info(f"Recovery suggestions: {', '.join(error.recovery_suggestions)}")
        
        # Log context if available
        if context:
            logger.debug(f"Error context: {context}")
        
        # Log response preview for debugging
        if response_text:
            logger.debug(f"Response preview: {response_text[:300]}...")

# Global error handler instance
_error_handler = None

def get_error_handler() -> EnhancedErrorHandler:
    """Get or create the global error handler instance"""
    global _error_handler
    if _error_handler is None:
        _error_handler = EnhancedErrorHandler()
    return _error_handler

def handle_parsing_error(
    error: Exception, 
    context: str = "",
    response_text: str = ""
) -> ParsingError:
    """Convenience function to handle parsing errors"""
    handler = get_error_handler()
    return handler.handle_parsing_error(error, context, response_text)

def validate_json_frontmatter(
    response_text: str,
    strict_mode: bool = False
) -> ValidationResult:
    """Convenience function to validate JSON frontmatter"""
    handler = get_error_handler()
    return handler.validate_json_frontmatter(response_text, strict_mode)