"""
Conjecture: Simplified configuration package
Now with unified configuration validation
"""

# Legacy configuration imports
from .simple_config import (
    Config,
    config,
    get_config,
    print_config_summary,
    validate_config as simple_validate_config,
)

# New unified configuration validator
from .unified_validator import (
    UnifiedConfigValidator,
    ConfigFormat,
    UnifiedValidationResult,
    ProviderConfig,
    get_unified_validator,
    validate_config as unified_validate_config,
    get_primary_provider,
    show_configuration_status,
)

# Backward compatibility layer
from .backward_compatibility import (
    SimpleProviderValidator,
    IndividualEnvValidator,
    UnifiedProviderValidator,
    SimpleValidator,
    show_migration_suggestions,
    check_api_usage,
)

# Migration utilities (commented out to avoid circular imports)
# from .migration_utils import (
#     ConfigMigrator,
#     analyze_migration,
#     execute_migration,
# )

# Default validate_config function now uses unified validator
def validate_config(env_file: str = ".env"):
    """Validate configuration using unified validator"""
    validator = get_unified_validator(env_file)
    return validator.validate_configuration()

__all__ = [
    # Legacy
    "Config", "config", "get_config", "print_config_summary",
    # Unified validation
    "UnifiedConfigValidator", "ConfigFormat", "UnifiedValidationResult", "ProviderConfig",
    "get_unified_validator", "unified_validate_config", "get_primary_provider",
    "show_configuration_status", "validate_config",
    # Backward compatibility
    "SimpleProviderValidator", "IndividualEnvValidator", "UnifiedProviderValidator",
    "SimpleValidator", "show_migration_suggestions", "check_api_usage",
    # Migration utilities commented out to avoid circular imports
    # "ConfigMigrator", "analyze_migration", "execute_migration"
]