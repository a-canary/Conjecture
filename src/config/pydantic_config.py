"""
Pydantic-based configuration loader for Conjecture
Handles workspace → user → default config hierarchy with Pydantic settings
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .settings_models import ConjectureSettings, ProviderConfig


class ConfigHierarchy:
    """
    Manages configuration file hierarchy: workspace → user → default
    Works with Pydantic settings for type safety and validation
    """
    
    def __init__(self):
        # Define config paths in order of precedence
        self.workspace_config = Path.cwd() / ".conjecture" / "config.json"
        self.user_config = Path.home() / ".conjecture" / "config.json" 
        self.default_config = Path(__file__).parent / "default_config.json"
        
    def load_config_dict(self) -> Dict[str, Any]:
        """
        Load configuration from workspace → user → default config files
        Later configs override earlier ones
        """
        config = {}
        
        # Load default config first
        if self.default_config.exists():
            with open(self.default_config, 'r', encoding='utf-8') as f:
                config.update(json.load(f))
        
        # Load user config (overrides default)
        if self.user_config.exists():
            with open(self.user_config, 'r', encoding='utf-8') as f:
                user_config_data = json.load(f)
                self._merge_configs(config, user_config_data)
        
        # Load workspace config (overrides user and default)
        if self.workspace_config.exists():
            with open(self.workspace_config, 'r', encoding='utf-8') as f:
                workspace_config_data = json.load(f)
                self._merge_configs(config, workspace_config_data)
        
        return config
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]):
        """
        Deep merge two config dictionaries
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
    
    def get_active_config_path(self) -> Path:
        """Return the path of the active config file"""
        if self.workspace_config.exists():
            return self.workspace_config
        elif self.user_config.exists():
            return self.user_config
        else:
            return self.default_config
    
    def create_user_config(self, config_data: Dict[str, Any]):
        """Create user config file"""
        self.user_config.parent.mkdir(parents=True, exist_ok=True)
        with open(self.user_config, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
    
    def is_workspace_config(self) -> bool:
        """Check if using workspace-specific configuration"""
        return self.workspace_config.exists()
    
    def save_config(self, settings: ConjectureSettings, target: Optional[str] = None):
        """
        Save settings to config file
        
        Args:
            settings: ConjectureSettings to save
            target: Target path ('user', 'workspace', or custom path)
        """
        if target == 'user':
            config_path = self.user_config
        elif target == 'workspace':
            config_path = self.workspace_config
        elif target:
            config_path = Path(target)
        else:
            # Save to active config path
            config_path = self.get_active_config_path()
        
        # Ensure parent directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert settings to dictionary and save
        config_dict = settings.to_dict()
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2)


class PydanticConfig:
    """
    Pydantic-based configuration class for Conjecture
    Uses workspace → user → default config file hierarchy with Pydantic validation
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        # Initialize config hierarchy
        self.config_hierarchy = ConfigHierarchy()
        
        # Store custom config path if provided
        self.custom_config_path = Path(config_path) if config_path else None
        
        # Load settings
        self.settings = self._load_settings()
        
        # Setup workspace context
        self._setup_workspace_context()
        
        # Setup data directory
        self._setup_data_directory()

    def _load_settings(self) -> ConjectureSettings:
        """Load settings using Pydantic with config file hierarchy"""
        try:
            # Load configuration from hierarchy
            config_data = self.config_hierarchy.load_config_dict()
            
            # Override with custom config path if provided
            if self.custom_config_path and self.custom_config_path.exists():
                with open(self.custom_config_path, 'r', encoding='utf-8') as f:
                    custom_config_data = json.load(f)
                    self.config_hierarchy._merge_configs(config_data, custom_config_data)
            
            # Create ConjectureSettings from config data
            settings = ConjectureSettings.from_dict(config_data)
            
            # Apply environment variable overrides (Pydantic BaseSettings does this automatically)
            # But we need to handle the custom env vars for providers
            # No environment variable overrides
            
            return settings
            
        except json.JSONDecodeError as e:
            # Re-raise JSON errors instead of falling back to defaults
            raise ValueError(f"Invalid JSON in config file: {e}")
        except Exception as e:
            print(f"Error loading config: {e}")
            print("Creating default configuration...")
            return self._create_default_settings()


    def _create_default_settings(self) -> ConjectureSettings:
        """Create default settings when config loading fails"""
        # Create default providers
        default_providers = [
            ProviderConfig(
                name="ollama",
                url="http://localhost:11434",
                api="",
                model="llama2",
                priority=1,
                is_local=True
            ),
            ProviderConfig(
                name="lm_studio", 
                url="http://localhost:1234",
                api="",
                model="ibm/granite-4-h-tiny",
                priority=2,
                is_local=True
            )
        ]
        
        # Create settings with defaults
        settings = ConjectureSettings(providers=default_providers)
        
        # Create default user config file
        self.config_hierarchy.create_user_config(settings.to_dict())
        
        return settings

    def _setup_workspace_context(self):
        """Setup workspace context based on current directory"""
        # Only set workspace if it's the default value
        if self.settings.workspace.workspace == "default":
            if self.config_hierarchy.is_workspace_config():
                # We're in a workspace
                self.settings.workspace.workspace = Path.cwd().name
            else:
                # Global workspace
                self.settings.workspace.workspace = "default"

    def _setup_data_directory(self):
        """Setup data directory based on workspace settings"""
        workspace_name = self.settings.workspace.workspace
        
        # Only override database_path if it's the default value
        current_db_path = self.settings.database.database_path
        is_default_path = current_db_path in ["data/conjecture.db", "conjecture.db"]
        
        if is_default_path:
            if workspace_name != "default":
                # Workspace-specific data directory
                workspace_dir = Path.cwd() / ".conjecture"
                data_dir = workspace_dir / "data"
                db_path = str(workspace_dir / "data" / "conjecture.db")
            else:
                # Global data directory
                home_dir = Path.home() / ".conjecture"
                data_dir = home_dir / "data"
                db_path = str(home_dir / "data" / "conjecture.db")
            
            # Update settings only if using default path
            self.settings.workspace.data_dir = str(data_dir)
            self.settings.database.database_path = db_path
            
            # Ensure data directory exists
            data_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Respect custom database_path from config
            # Ensure data directory exists for custom path
            custom_db_path = Path(self.settings.database.database_path)
            custom_db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Set data_dir to parent directory of custom database path
            self.settings.workspace.data_dir = str(custom_db_path.parent)
        
        # Ensure data_dir is always set
        if not self.settings.workspace.data_dir:
            # Fallback to current directory if nothing else is set
            self.settings.workspace.data_dir = str(Path.cwd())

    # Backward compatibility methods
    @property
    def providers(self) -> List[ProviderConfig]:
        """Get list of configured providers"""
        return self.settings.providers

    @property
    def confidence_threshold(self) -> float:
        """Get confidence threshold"""
        return self.settings.processing.confidence_threshold

    @property
    def confident_threshold(self) -> float:
        """Get confident threshold"""
        return self.settings.processing.confident_threshold

    @property
    def max_context_size(self) -> int:
        """Get max context size"""
        return self.settings.processing.max_context_size

    @property
    def batch_size(self) -> int:
        """Get batch size"""
        return self.settings.processing.batch_size

    @property
    def debug(self) -> bool:
        """Get debug flag"""
        return self.settings.debug

    @property
    def database_path(self) -> str:
        """Get database path"""
        return self.settings.database.database_path

    @property
    def user(self) -> str:
        """Get user name"""
        return self.settings.workspace.user

    @property
    def team(self) -> str:
        """Get team name"""
        return self.settings.workspace.team

    @property
    def workspace(self) -> str:
        """Get workspace name"""
        return self.settings.workspace.workspace

    @property
    def data_dir(self) -> str:
        """Get data directory"""
        return self.settings.workspace.data_dir

    def get_providers(self) -> List[Dict[str, Any]]:
        """Get list of configured providers as dictionaries"""
        return [provider.to_dict() for provider in self.settings.providers]

    def get_primary_provider(self) -> Optional[Dict[str, Any]]:
        """Get the primary (first) provider as dictionary"""
        provider = self.settings.get_primary_provider()
        return provider.to_dict() if provider else None

    def is_workspace_config(self) -> bool:
        """Check if using workspace-specific configuration"""
        return self.config_hierarchy.is_workspace_config()

    def get_config_info(self) -> Dict[str, Any]:
        """Get configuration information"""
        return {
            "config_path": str(self.config_hierarchy.get_active_config_path()),
            "workspace": self.settings.workspace.workspace,
            "is_workspace_config": self.is_workspace_config(),
            "providers_count": len(self.settings.providers),
            "data_dir": self.settings.workspace.data_dir,
            "database_path": self.settings.database.database_path,
            "debug": self.settings.debug,
            "user_config_exists": self.config_hierarchy.user_config.exists(),
            "workspace_config_exists": self.config_hierarchy.workspace_config.exists(),
            "default_config_exists": self.config_hierarchy.default_config.exists(),
            "available_providers": len(self.settings.get_available_providers()),
            "primary_provider": self.settings.get_primary_provider().name if self.settings.get_primary_provider() else None,
            # Add backward compatibility keys for tests
            "has_providers": len(self.settings.providers) > 0,
            "provider_count": len(self.settings.providers),
        }

    def save_settings(self, target: Optional[str] = None):
        """Save current settings to config file"""
        self.config_hierarchy.save_config(self.settings, target)

    def reload_settings(self):
        """Reload settings from config files"""
        self.settings = self._load_settings()
        self._setup_workspace_context()
        self._setup_data_directory()

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return self.settings.to_dict()


# Global configuration instance
_pydantic_config = None


def get_pydantic_config() -> PydanticConfig:
    """Get the global Pydantic configuration instance"""
    global _pydantic_config
    if _pydantic_config is None:
        _pydantic_config = PydanticConfig()
    return _pydantic_config


def reload_pydantic_config():
    """Reload the global Pydantic configuration"""
    global _pydantic_config
    _pydantic_config = PydanticConfig()


def validate_pydantic_config() -> bool:
    """Validate Pydantic configuration"""
    config = get_pydantic_config()
    
    # Check if we have at least one provider
    if not config.settings.providers:
        print("No providers configured")
        return False
    
    # Check if primary provider has required fields
    primary = config.settings.get_primary_provider()
    if not primary:
        print("No primary provider found")
        return False
    
    # Check if primary provider is available
    if not primary.is_available():
        print(f"Primary provider '{primary.name}' is not available (missing API key or unreachable)")
        return False
    
    return True