"""
Pydantic settings models for Conjecture configuration system
Provides type-safe, validated configuration with environment variable support
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class ProviderType(str, Enum):
    """Provider type enumeration"""
    OLLAMA = "ollama"
    LM_STUDIO = "lm_studio"
    CHUTES = "chutes"
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    GOOGLE = "google"


class ProviderConfig(BaseModel):
    """Unified provider configuration using Pydantic"""
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        use_enum_values=True,
        populate_by_name=True
    )

    # Core required fields
    name: str = Field(..., description="Provider name")
    url: str = Field(..., description="Provider API URL")
    api: str = Field(default="", description="API key (empty for local providers)", alias="api_key")
    model: str = Field(..., description="Default model name")

    # Optional metadata
    priority: int = Field(default=99, description="Provider priority (lower = higher priority)")
    is_local: bool = Field(default=False, description="Whether this is a local provider")
    description: Optional[str] = Field(default=None, description="Provider description")
    max_tokens: Optional[int] = Field(default=4000, description="Maximum tokens for this provider")
    temperature: Optional[float] = Field(default=0.7, description="Default temperature")
    timeout: Optional[int] = Field(default=30, description="Request timeout in seconds")
    max_retries: Optional[int] = Field(default=3, description="Maximum retry attempts")

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        """Validate provider name"""
        if not v or not v.strip():
            raise ValueError("Provider name cannot be empty")
        return v.strip().lower()

    @field_validator('url')
    @classmethod
    def validate_url(cls, v):
        """Validate provider URL"""
        if not v or not v.strip():
            raise ValueError("Provider URL cannot be empty")
        return v.strip().rstrip('/')

    @field_validator('model')
    @classmethod
    def validate_model(cls, v):
        """Validate model name"""
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()

    @model_validator(mode='after')
    def set_local_provider_flags(self) -> 'ProviderConfig':
        """Set local provider flags based on URL and name"""
        # Modify self directly instead of returning a copy
        is_local = (
            self.name in ['ollama', 'lm_studio'] or
            'localhost' in self.url or '127.0.0.1' in self.url
        )
        # Use object.__setattr__ to bypass Pydantic's immutability during validation
        object.__setattr__(self, 'is_local', is_local)
        return self

    def get_provider_type(self) -> Optional[ProviderType]:
        """Get the provider type enum"""
        try:
            return ProviderType(self.name)
        except ValueError:
            return None

    def is_available(self) -> bool:
        """Check if provider is likely available (has API key or is local)"""
        return self.is_local or bool(self.api)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "url": self.url,
            "api": self.api,
            "model": self.model,
            "name": self.name,
            "priority": self.priority,
            "is_local": self.is_local,
            "description": self.description,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        
        # Include optional fields if they are set (not None)
        if self.timeout is not None:
            result["timeout"] = self.timeout
        if self.max_retries is not None:
            result["max_retries"] = self.max_retries
            
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProviderConfig":
        """Create ProviderConfig from dictionary"""
        # Handle api_key -> api field mapping
        if "api_key" in data and "api" not in data:
            data = data.copy()
            data["api"] = data.pop("api_key")
        
        # Filter out unknown fields to avoid extra_forbidden errors
        allowed_fields = {"name", "url", "api", "api_key", "model", "priority", "is_local",
                       "description", "max_tokens", "temperature", "timeout", "max_retries"}
        filtered_data = {k: v for k, v in data.items() if k in allowed_fields}
        
        return cls(**filtered_data)


class DatabaseSettings(BaseModel):
    """Database configuration settings"""
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        frozen=False
    )

    database_type: str = Field(default="sqlite", description="Database type")
    database_path: str = Field(default="data/conjecture.db", description="Database file path")
    chroma_path: str = Field(default="data/chroma", description="ChromaDB path")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Embedding model name")
    chroma_collection_name: str = Field(default="claims", description="ChromaDB collection name")
    max_tokens: int = Field(default=8000, ge=1000, description="Maximum context tokens")

    @field_validator('database_path', 'chroma_path')
    @classmethod
    def resolve_path(cls, v):
        """Resolve relative paths to absolute paths"""
        if not os.path.isabs(v):
            return str(Path.cwd() / v)
        return v


class LLMSettings(BaseModel):
    """LLM configuration settings"""
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        frozen=False
    )

    default_model: str = Field(default="gpt-3.5-turbo", description="Default LLM model")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="Default temperature")
    max_tokens: int = Field(default=2000, ge=1, description="Default max tokens")
    timeout: int = Field(default=30, ge=1, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, ge=0.1, description="Retry delay in seconds")


class ProcessingSettings(BaseModel):
    """Claim processing configuration settings"""
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        frozen=False
    )

    confidence_threshold: float = Field(default=0.95, ge=0.0, le=1.0, description="Confidence threshold for validation")
    confident_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence threshold for confidence")
    max_context_size: int = Field(default=10, ge=1, description="Maximum context size")
    batch_size: int = Field(default=10, ge=1, description="Batch size for processing")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Similarity threshold")
    validation_threshold: float = Field(default=0.95, ge=0.0, le=1.0, description="Validation threshold")

    # Context limits
    max_context_concepts: int = Field(default=10, ge=1, description="Maximum concepts in context")
    max_context_references: int = Field(default=8, ge=1, description="Maximum references in context")
    max_context_skills: int = Field(default=5, ge=1, description="Maximum skills in context")
    max_context_goals: int = Field(default=3, ge=1, description="Maximum goals in context")

    @model_validator(mode='after')
    def validate_thresholds(self):
        """Validate threshold relationships"""
        if self.confident_threshold > self.confidence_threshold:
            raise ValueError("confident_threshold cannot be greater than confidence_threshold")
        return self


class DirtyFlagSettings(BaseModel):
    """Dirty flag system configuration settings"""
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        frozen=False
    )

    confidence_threshold: float = Field(default=0.90, ge=0.0, le=1.0, description="Dirty flag confidence threshold")
    cascade_depth: int = Field(default=3, ge=1, description="Maximum cascade depth")
    batch_size: int = Field(default=5, ge=1, description="Dirty flag batch size")
    max_parallel_batches: int = Field(default=3, ge=1, description="Maximum parallel batches")
    confidence_boost_factor: float = Field(default=0.10, ge=0.0, description="Confidence boost factor")
    two_pass_evaluation: bool = Field(default=True, description="Enable two-pass evaluation")
    relationship_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Relationship threshold")
    timeout_seconds: int = Field(default=300, ge=1, description="Timeout in seconds")
    max_retries: int = Field(default=2, ge=0, description="Maximum retries")
    auto_evaluation_enabled: bool = Field(default=True, description="Enable auto evaluation")
    evaluation_interval_minutes: int = Field(default=30, ge=1, description="Evaluation interval in minutes")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Similarity threshold")

    # Priority weights
    priority_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "confidence_gap": 10.0,
            "new_claim": 5.0,
            "confidence_threshold": 15.0,
            "supporting_claim_changed": 8.0,
            "relationship_changed": 6.0,
            "manual_mark": 20.0,
            "batch_evaluation": 3.0,
            "system_trigger": 4.0
        },
        description="Priority weights for dirty flag evaluation"
    )


class LoggingSettings(BaseModel):
    """Logging configuration settings"""
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        frozen=False
    )

    level: str = Field(default="INFO", description="Log level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    file_path: Optional[str] = Field(default=None, description="Log file path")
    max_file_size: int = Field(default=10485760, description="Maximum log file size in bytes")
    backup_count: int = Field(default=5, description="Number of backup log files")

    @field_validator('level')
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()


class WorkspaceSettings(BaseModel):
    """Workspace configuration settings"""
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        frozen=False
    )

    workspace: str = Field(default="default", description="Workspace name")
    user: str = Field(default="user", description="User name")
    team: str = Field(default="default", description="Team name")
    data_dir: Optional[str] = Field(default=None, description="Data directory path")

    @field_validator('data_dir')
    @classmethod
    def resolve_data_dir(cls, v):
        """Resolve data directory path"""
        if v and not os.path.isabs(v):
            return str(Path.cwd() / v)
        return v


class ConjectureSettings(BaseModel):
    """Main Conjecture settings class using Pydantic BaseSettings"""
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        frozen=False
    )

    # Core settings
    debug: bool = Field(default=False, description="Enable debug mode")
    config_path: Optional[str] = Field(default=None, description="Custom config file path")

    # Sub-settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings, description="Database settings")
    llm: LLMSettings = Field(default_factory=LLMSettings, description="LLM settings")
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings, description="Processing settings")
    dirty_flag: DirtyFlagSettings = Field(default_factory=DirtyFlagSettings, description="Dirty flag settings")
    logging: LoggingSettings = Field(default_factory=LoggingSettings, description="Logging settings")
    workspace: WorkspaceSettings = Field(default_factory=WorkspaceSettings, description="Workspace settings")

    # Provider configurations
    providers: List[ProviderConfig] = Field(default_factory=list, description="LLM provider configurations")

    def get_primary_provider(self) -> Optional[ProviderConfig]:
        """Get the primary (highest priority) provider"""
        if not self.providers:
            return None
        return min(self.providers, key=lambda p: p.priority)

    def get_available_providers(self) -> List[ProviderConfig]:
        """Get all available providers (local or with API keys)"""
        return [p for p in self.providers if p.is_available()]

    def get_provider_by_name(self, name: str) -> Optional[ProviderConfig]:
        """Get provider by name"""
        for provider in self.providers:
            if provider.name == name.lower():
                return provider
        return None

    def add_provider(self, provider: ProviderConfig):
        """Add a provider configuration"""
        # Remove existing provider with same name if exists
        self.providers = [p for p in self.providers if p.name != provider.name]
        self.providers.append(provider)

    def remove_provider(self, name: str) -> bool:
        """Remove a provider by name"""
        original_count = len(self.providers)
        self.providers = [p for p in self.providers if p.name != name.lower()]
        return len(self.providers) < original_count

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary for JSON serialization"""
        return {
            "debug": self.debug,
            "providers": [p.to_dict() for p in self.providers],
            "confidence_threshold": self.processing.confidence_threshold,
            "confident_threshold": self.processing.confident_threshold,
            "max_context_size": self.processing.max_context_size,
            "batch_size": self.processing.batch_size,
            "database_path": self.database.database_path,
            "user": self.workspace.user,
            "team": self.workspace.team,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConjectureSettings":
        """Create settings from dictionary"""
        # Handle providers with proper field mapping
        providers_data = data.get("providers", [])
        providers = []
        for provider_data in providers_data:
            # Handle api_key -> api field mapping
            if "api_key" in provider_data and "api" not in provider_data:
                provider_data = provider_data.copy()
                provider_data["api"] = provider_data.pop("api_key")
            
            # Filter out unknown fields to avoid extra_forbidden errors
            allowed_fields = {"name", "url", "api", "api_key", "model", "priority", "is_local",
                           "description", "max_tokens", "temperature", "timeout", "max_retries"}
            filtered_data = {k: v for k, v in provider_data.items() if k in allowed_fields}
            providers.append(ProviderConfig(**filtered_data))
        
        # Extract other settings with proper handling
        kwargs = {}
        
        # Core settings
        if "debug" in data:
            kwargs["debug"] = data["debug"]
        
        # Processing settings
        processing_kwargs = {}
        if "confidence_threshold" in data:
            processing_kwargs["confidence_threshold"] = data["confidence_threshold"]
        if "confident_threshold" in data:
            processing_kwargs["confident_threshold"] = data["confident_threshold"]
        if "max_context_size" in data:
            processing_kwargs["max_context_size"] = data["max_context_size"]
        if "batch_size" in data:
            processing_kwargs["batch_size"] = data["batch_size"]
        
        # Database settings
        database_kwargs = {}
        if "database_path" in data:
            database_kwargs["database_path"] = data["database_path"]
        
        # Workspace settings
        workspace_kwargs = {}
        if "user" in data:
            workspace_kwargs["user"] = data["user"]
        if "team" in data:
            workspace_kwargs["team"] = data["team"]
        if "workspace" in data:
            workspace_kwargs["workspace"] = data["workspace"]
        
        # Create settings with all components
        kwargs["providers"] = providers
        if processing_kwargs:
            kwargs["processing"] = ProcessingSettings(**processing_kwargs)
        if database_kwargs:
            kwargs["database"] = DatabaseSettings(**database_kwargs)
        if workspace_kwargs:
            kwargs["workspace"] = WorkspaceSettings(**workspace_kwargs)
            
        return cls(**kwargs)