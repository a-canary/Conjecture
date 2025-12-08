"""
Unified Validator - Compatibility Layer
Provides validation functionality for testing
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, validator

class ValidationResult(BaseModel):
    """Validation result for testing"""
    is_valid: bool = True
    errors: List[str] = []
    warnings: List[str] = []

class UnifiedValidator:
    """Mock unified validator for testing"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate configuration"""
        return ValidationResult(is_valid=True, errors=[], warnings=[])
    
    def validate_provider(self, provider: Dict[str, Any]) -> ValidationResult:
        """Validate provider configuration"""
        return ValidationResult(is_valid=True, errors=[], warnings=[])
    
    def validate_all(self) -> ValidationResult:
        """Validate all configurations"""
        return ValidationResult(is_valid=True, errors=[], warnings=[])

# Export the main classes
__all__ = ['UnifiedValidator', 'ValidationResult']