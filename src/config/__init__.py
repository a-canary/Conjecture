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

# Simplified configuration - only use what actually exists
# Note: Unified validator components are not fully implemented yet
# Using simple config for now

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


# Default validate_config function uses simple validator
def validate_config(env_file: str = ".env"):
    """Validate configuration using simple validator"""
    return simple_validate_config(env_file)


__all__ = [
    # Legacy
    "Config",
    "config",
    "get_config",
    "print_config_summary",
    "simple_validate_config",
    # Simple validation
    "validate_config",
    # Backward compatibility
    "SimpleProviderValidator",
    "IndividualEnvValidator",
    "UnifiedProviderValidator",
    "SimpleValidator",
    "show_migration_suggestions",
    "check_api_usage",
]
