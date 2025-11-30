"""
Common configuration classes for provider management
Consolidated ProviderConfig to reduce duplication across validators
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class ProviderConfig:
    """Unified provider configuration structure"""

    # Core required fields
    name: str
    base_url: str
    api_key: str

    # Model information
    model: Optional[str] = None
    models: List[str] = field(default_factory=list)

    # Priority and classification
    priority: int = 99
    is_local: bool = False

    # Optional metadata
    protocol: Optional[str] = None
    format_type: str = "unknown"
    description: Optional[str] = None

    def __post_init__(self):
        """Post-processing to normalize data"""
        # Ensure models list is populated
        if self.models and not self.model:
            self.model = self.models[0]
        elif self.model and not self.models:
            self.models = [self.model]

        # Set protocol from URL if not specified
        if not self.protocol and self.base_url:
            if self.base_url.startswith("https://"):
                self.protocol = "https"
            elif self.base_url.startswith("http://"):
                self.protocol = "http"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "model": self.model,
            "models": self.models,
            "priority": self.priority,
            "is_local": self.is_local,
            "protocol": self.protocol,
            "format_type": self.format_type,
            "description": self.description,
        }

    @classmethod
    def from_env_vars(
        cls, prefix: str, env_vars: Dict[str, str]
    ) -> Optional["ProviderConfig"]:
        """Create ProviderConfig from environment variables with given prefix"""
        api_url = env_vars.get(f"{prefix}_API_URL", "")
        api_key = env_vars.get(f"{prefix}_API_KEY", "")
        model = env_vars.get(f"{prefix}_MODEL", "")

        if not api_url:
            return None

        # Detect provider type and set defaults
        provider_name = prefix.lower().replace("_", "")
        is_local = provider_name in ["ollama", "lm_studio"]

        # Set default priorities
        priorities = {
            "ollama": 1,
            "lm_studio": 2,
            "chutes": 3,
            "openrouter": 4,
            "groq": 5,
            "openai": 6,
            "anthropic": 7,
            "google": 8,
        }

        return cls(
            name=provider_name,
            base_url=api_url,
            api_key=api_key,
            model=model,
            priority=priorities.get(provider_name, 99),
            is_local=is_local,
        )


@dataclass
class ValidationResult:
    """Standard validation result structure"""

    is_valid: bool
    providers: List[ProviderConfig] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, error: str):
        """Add an error message"""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str):
        """Add a warning message"""
        self.warnings.append(warning)

    def add_provider(self, provider: ProviderConfig):
        """Add a valid provider"""
        self.providers.append(provider)
