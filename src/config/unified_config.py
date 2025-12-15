"""
Unified Configuration System for Conjecture
Consolidates all configuration functionality into a single, clean system
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .pydantic_config import PydanticConfig, get_pydantic_config, reload_pydantic_config
from .settings_models import ProviderConfig, ConjectureSettings

class UnifiedConfig:
    """
    Unified configuration class that consolidates all config functionality
    Uses Pydantic settings for type safety and validation
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        # Initialize Pydantic configuration
        self.pydantic_config = PydanticConfig(config_path)
        self.settings = self.pydantic_config.settings

    # Backward compatibility properties
    @property
    def confidence_threshold(self) -> float:
        return self.settings.processing.confidence_threshold

    @property
    def confident_threshold(self) -> float:
        return self.settings.processing.confident_threshold

    @property
    def max_context_size(self) -> int:
        return self.settings.processing.max_context_size

    @property
    def batch_size(self) -> int:
        return self.settings.processing.batch_size

    @property
    def debug(self) -> bool:
        return self.settings.debug

    @property
    def database_path(self) -> str:
        return self.settings.database.database_path

    @property
    def data_dir(self) -> str:
        return self.settings.workspace.data_dir

    @property
    def workspace(self) -> str:
        return self.settings.workspace.workspace

    @property
    def user(self) -> str:
        return self.settings.workspace.user

    @property
    def team(self) -> str:
        return self.settings.workspace.team

    @property
    def providers(self) -> List[ProviderConfig]:
        """Get providers as list of ProviderConfig objects"""
        return self.settings.providers

    @property
    def config_path(self) -> Path:
        """Get the active config path"""
        return self.pydantic_config.config_hierarchy.get_active_config_path()

    def get_providers(self) -> List[ProviderConfig]:
        """Get list of configured providers"""
        return self.providers

    def get_primary_provider(self) -> Optional[ProviderConfig]:
        """Get the primary (first) provider"""
        return self.settings.get_primary_provider()

    def is_workspace_config(self) -> bool:
        """Check if using workspace-specific configuration"""
        return self.pydantic_config.is_workspace_config()

    def get_config_info(self) -> Dict[str, Any]:
        """Get configuration information"""
        return self.pydantic_config.get_config_info()

    def get_effective_confident_threshold(self) -> float:
        """Get effective confident threshold for claim evaluation"""
        return self.settings.processing.confident_threshold

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return self.pydantic_config.to_dict()

    def reload_config(self):
        """Reload configuration from files"""
        self.pydantic_config.reload_settings()
        self.settings = self.pydantic_config.settings

    def save_config(self, target: Optional[str] = None):
        """Save current configuration"""
        self.pydantic_config.save_settings(target)

# Global configuration instance
_config = None

def get_config() -> UnifiedConfig:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = UnifiedConfig()
    return _config

def validate_config() -> bool:
    """Validate configuration (simplified for JSON-based config)"""
    try:
        config = get_config()

        # Check if we have at least one provider
        if not config.providers:
            print("No providers configured")
            return False

        # Check if primary provider has required fields
        primary = config.get_primary_provider()
        if not primary:
            print("No primary provider found")
            return False

        required_fields = ["url", "model"]
        for field in required_fields:
            if not primary.get(field):
                print(f"Primary provider missing required field: {field}")
                return False

        return True
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False

def reload_config():
    """Reload configuration from file"""
    global _config
    _config = UnifiedConfig()