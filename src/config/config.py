"""
Configuration for Conjecture
JSON-based configuration with workspace detection
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Load environment variables from .env files
try:
    from dotenv import load_dotenv
    # Try to load .env from project root
    project_root = Path(__file__).parent.parent.parent

    # Load .env files in order of precedence
    for env_file in [project_root / '.env']:
        if env_file.exists():
            load_dotenv(env_file)
except ImportError:
    # dotenv not available, use system environment variables only
    pass


def substitute_env_vars(config_dict):
    """
    Recursively substitute environment variables in configuration values
    Supports ${VAR} and ${VAR:-default} syntax
    """
    import re

    if isinstance(config_dict, dict):
        return {k: substitute_env_vars(v) for k, v in config_dict.items()}
    elif isinstance(config_dict, list):
        return [substitute_env_vars(item) for item in config_dict]
    elif isinstance(config_dict, str):
        # Replace ${VAR:-default} patterns
        def replace_var(match):
            var_expr = match.group(1)
            if ':-' in var_expr:
                var_name, default_value = var_expr.split(':-', 1)
                return os.getenv(var_name, default_value)
            else:
                return os.getenv(var_expr, '')

        # Handle both ${VAR} and ${VAR:-default} patterns
        pattern = r'\$\{([^}]+)\}'
        result = re.sub(pattern, replace_var, config_dict)

        # Convert string boolean/numeric values to proper types
        if result.lower() == 'true':
            return True
        elif result.lower() == 'false':
            return False
        elif result.isdigit():
            return int(result)
        elif result.replace('.', '').isdigit():
            try:
                return float(result)
            except ValueError:
                pass

        return result
    else:
        return config_dict


class Config:
    """
    Configuration class for Conjecture
    JSON-based configuration with workspace detection and provider management
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
        self.config_path = config_path or self._detect_config_path()

        self._load_config()
        self._setup_data_directory()

    def _detect_config_path(self) -> Path:
        """Detect configuration path (workspace or home)"""
        # Check for workspace config first
        workspace_config = Path.cwd() / ".conjecture" / "config.json"
        if workspace_config.exists():
            self.workspace = Path.cwd().name
            return workspace_config

        # Fall back to home config
        home_config = Path.home() / ".conjecture" / "config.json"
        return home_config

    def _load_config(self):
        """Load configuration from JSON file"""
        try:
            if not self.config_path.exists():
                self._create_default_config()
                return

            with open(self.config_path, "r") as f:
                config_data = json.load(f)

            # Substitute environment variables
            config_data = substitute_env_vars(config_data)

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

        except Exception as e:
            print(f"Error loading config: {e}")
            self._create_default_config()

    def _create_default_config(self):
        """Create a default configuration file"""
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

        # Create config directory if it doesn't exist
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.config_path, "w") as f:
            json.dump(default_config, f, indent=2)

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
