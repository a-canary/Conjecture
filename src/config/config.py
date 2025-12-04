"""
Configuration for Conjecture
JSON-based configuration with workspace, user, and default config files
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class ConfigHierarchy:
    """
    Manages configuration file hierarchy: workspace → user → default
    """
    
    def __init__(self):
        # Define config paths in order of precedence
        self.workspace_config = Path.cwd() / ".conjecture" / "config.json"
        self.user_config = Path.home() / ".conjecture" / "config.json" 
        self.default_config = Path(__file__).parent / "default_config.json"
        
    def load_configs(self) -> Dict[str, Any]:
        """
        Load configuration from workspace → user → default config files
        Later configs override earlier ones
        """
        config = {}
        
        # Load default config first
        if self.default_config.exists():
            with open(self.default_config, 'r') as f:
                config.update(json.load(f))
        
        # Load user config (overrides default)
        if self.user_config.exists():
            with open(self.user_config, 'r') as f:
                user_config_data = json.load(f)
                self._merge_configs(config, user_config_data)
        
        # Load workspace config (overrides user and default)
        if self.workspace_config.exists():
            with open(self.workspace_config, 'r') as f:
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
        with open(self.user_config, 'w') as f:
            json.dump(config_data, f, indent=2)


class Config:
    """
    Configuration class for Conjecture
    Uses workspace → user → default config file hierarchy
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        # === Core Settings ===
        self.confidence_threshold = 0.95
        self.confident_threshold = 0.8
        self.session_confident_threshold = None
        self.max_context_size = 10
        self.batch_size = 10
        self.debug = False

        # === Database Settings ===
        self.database_type = "sqlite"
        self.database_path = "data/conjecture.db"
        self.data_dir = None

        # === Workspace Context ===
        self.workspace = "default"
        self.user = "user"
        self.team = "default"

        # === LLM Provider Configuration ===
        self.providers: List[Dict[str, Any]] = []
        
        # Initialize config hierarchy
        self.config_hierarchy = ConfigHierarchy()
        self.config_path = config_path or self.config_hierarchy.get_active_config_path()

        self._load_config()
        self._setup_data_directory()

    def _load_config(self):
        """Load configuration from config file hierarchy"""
        try:
            # Load configuration from hierarchy
            config_data = self.config_hierarchy.load_configs()

            # Load providers
            self.providers = config_data.get("providers", [])

            # Load other settings (with defaults)
            self.confidence_threshold = config_data.get("confidence_threshold", 0.95)
            self.confident_threshold = config_data.get("confident_threshold", 0.8)
            self.max_context_size = config_data.get("max_context_size", 10)
            self.batch_size = config_data.get("batch_size", 10)
            self.debug = config_data.get("debug", False)
            self.database_path = config_data.get("database_path", "data/conjecture.db")
            self.user = config_data.get("user", "user")
            self.team = config_data.get("team", "default")

            # Determine if we're in a workspace
            self.workspace = "default"
            if self.config_hierarchy.workspace_config.exists():
                self.workspace = Path.cwd().name

        except Exception as e:
            print(f"Error loading config: {e}")
            self._create_default_config()

    def _create_default_config(self):
        """Create a default user configuration file"""
        default_config = {
            "providers": [
                {"url": "http://localhost:11434", "api": "", "model": "llama2"}
            ],
            "confidence_threshold": 0.95,
            "confident_threshold": 0.8,
            "max_context_size": 10,
            "batch_size": 10,
            "debug": False,
            "database_path": "data/conjecture.db",
            "user": "user",
            "team": "default",
        }

        # Create user config file
        self.config_hierarchy.create_user_config(default_config)

        # Load the default config
        self.providers = default_config["providers"]

    def _setup_data_directory(self):
        """Setup data directory based on workspace"""
        if self.workspace != "default":
            # Workspace-specific data directory
            workspace_dir = Path.cwd() / ".conjecture"
            self.data_dir = workspace_dir / "data"
            self.database_path = str(workspace_dir / "data" / "conjecture.db")
        else:
            # Global data directory
            home_dir = Path.home() / ".conjecture"
            self.data_dir = home_dir / "data"
            self.database_path = str(home_dir / "data" / "conjecture.db")

        self.data_dir.mkdir(parents=True, exist_ok=True)

    def get_providers(self) -> List[Dict[str, Any]]:
        """Get list of configured providers"""
        return self.providers

    def get_primary_provider(self) -> Optional[Dict[str, Any]]:
        """Get the primary (first) provider"""
        return self.providers[0] if self.providers else None

    def is_workspace_config(self) -> bool:
        """Check if using workspace-specific configuration"""
        return (
            "workspace" in str(self.config_path).lower() or self.workspace != "default"
        )

    def get_config_info(self) -> Dict[str, Any]:
        """Get configuration information"""
        return {
            "config_path": str(self.config_path),
            "workspace": self.workspace,
            "is_workspace_config": self.is_workspace_config(),
            "providers_count": len(self.providers),
            "data_dir": str(self.data_dir),
            "database_path": self.database_path,
            "debug": self.debug,
            "user_config_exists": self.config_hierarchy.user_config.exists(),
            "workspace_config_exists": self.config_hierarchy.workspace_config.exists(),
            "default_config_exists": self.config_hierarchy.default_config.exists(),
        }


# Global configuration instance
_config = None


def get_config() -> Config:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = Config()
    return _config


def validate_config() -> bool:
    """Validate configuration (simplified for JSON-based config)"""
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


def reload_config():
    """Reload configuration from file"""
    global _config
    _config = Config()