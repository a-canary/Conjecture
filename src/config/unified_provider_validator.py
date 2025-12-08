"""
Unified Provider Validator - Compatibility Layer
Provides provider validation functionality for testing
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, validator

class ValidationResult(BaseModel):
    """Validation result for testing"""
    is_valid: bool = True
    errors: List[str] = []
    warnings: List[str] = []

class UnifiedProviderValidator:
    """Mock unified provider validator for testing"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    def validate_provider(self, provider: Dict[str, Any]) -> ValidationResult:
        """Validate provider configuration"""
        errors = []
        warnings = []
        
        # Basic validation
        if not provider.get('name'):
            errors.append("Provider name is required")
        if not provider.get('url'):
            errors.append("Provider URL is required")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def validate_all_providers(self, providers: List[Dict[str, Any]]) -> ValidationResult:
        """Validate all providers"""
        all_errors = []
        all_warnings = []
        
        for provider in providers:
            result = self.validate_provider(provider)
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
        
        return ValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings
        )

class UnifiedConfigValidator:
    """Mock unified config validator for testing"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.provider_validator = UnifiedProviderValidator(config)
    
    def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate configuration"""
        errors = []
        warnings = []
        
        # Validate providers if present
        if 'providers' in config:
            provider_result = self.provider_validator.validate_all_providers(config['providers'])
            errors.extend(provider_result.errors)
            warnings.extend(provider_result.warnings)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

# Export the main classes
__all__ = ['UnifiedProviderValidator', 'UnifiedConfigValidator', 'ValidationResult']