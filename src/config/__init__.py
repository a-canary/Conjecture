"""
Conjecture: Simplified configuration package
Now with unified configuration validation
"""

# Unified configuration imports
from .config import (
    Config,
    get_config,
    validate_config as simple_validate_config,
)

# Simplified configuration - only use what actually exists
# Note: Unified validator components are not fully implemented yet
# Using simple config for now

# Backward compatibility layer (archived)
# from .backward_compatibility import (
#     SimpleProviderValidator,
#     IndividualEnvValidator,
#     UnifiedProviderValidator,
#     SimpleValidator,
#     show_migration_suggestions,
#     check_api_usage,
# )

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
    # Core
    "Config",
    "get_config",
    "simple_validate_config",
    # Simple validation
    "validate_config",
]
