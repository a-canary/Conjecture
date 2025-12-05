"""
Conjecture: Unified configuration package
Consolidated configuration system with single source of truth
"""

# Unified configuration imports
from .unified_config import (
    UnifiedConfig,
    get_config,
    validate_config,
    reload_config
)

# Pydantic-based models and validation
from .settings_models import (
    ProviderConfig,
    ConjectureSettings,
    DatabaseSettings,
    LLMSettings,
    ProcessingSettings,
    DirtyFlagSettings,
    LoggingSettings,
    WorkspaceSettings
)

from .pydantic_config import (
    PydanticConfig,
    get_pydantic_config,
    reload_pydantic_config,
    validate_pydantic_config
)

# Backward compatibility aliases
Config = UnifiedConfig  # Alias for existing code

__all__ = [
    # Core unified configuration
    "UnifiedConfig",
    "Config",  # Backward compatibility
    "get_config",
    "validate_config",
    "reload_config",
    
    # Pydantic models
    "ProviderConfig",
    "ConjectureSettings",
    "DatabaseSettings",
    "LLMSettings",
    "ProcessingSettings",
    "DirtyFlagSettings",
    "LoggingSettings",
    "WorkspaceSettings",
    
    # Pydantic config system
    "PydanticConfig",
    "get_pydantic_config",
    "reload_pydantic_config",
    "validate_pydantic_config",
]