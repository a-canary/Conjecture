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
__all__ = ['UnifiedValidator', 'ValidationResult', 'UnifiedConfigValidator']

# Alias for backward compatibility
UnifiedConfigValidator = UnifiedValidator

# Additional exports for testing
from enum import Enum

class ConfigFormat(Enum):
    """Configuration format enum"""
    JSON = "json"
    YAML = "yaml"
    ENV = "env"

def get_unified_validator(config_path: Optional[str] = None) -> UnifiedValidator:
    """Get unified validator instance"""
    return UnifiedValidator()

def validate_config(config: Dict[str, Any]) -> ValidationResult:
    """Validate configuration"""
    validator = UnifiedValidator()
    return validator.validate_config(config)

def get_primary_provider(config: Dict[str, Any]) -> Optional[str]:
    """Get primary provider from config"""
    providers = config.get('providers', [])
    return providers[0].get('name') if providers else None

def show_configuration_status(config: Dict[str, Any]) -> Dict[str, Any]:
    """Show configuration status"""
    return {
        "configured": bool(config.get('providers')),
        "provider_count": len(config.get('providers', [])),
        "primary_provider": get_primary_provider(config)
    }