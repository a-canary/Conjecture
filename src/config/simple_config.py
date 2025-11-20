"""
Unified Configuration System for Conjecture
Single source of truth using .env file
"""

import os
from pathlib import Path
from typing import Any, Dict

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # dotenv not available, use system environment only
    pass


class Config:
    """
    Unified Configuration using .env file
    Simplified from 83+ constants to essential settings
    """

    def __init__(self):
        # === Provider Configuration (from .env) ===
        self.provider_api_url = os.getenv(
            "PROVIDER_API_URL", "https://llm.chutes.ai/v1"
        )
        self.provider_api_key = os.getenv("PROVIDER_API_KEY", "")
        self.provider_model = os.getenv("PROVIDER_MODEL", "zai-org/GLM-4.6-FP8")
        self.llm_provider = os.getenv("LLM_PROVIDER", "chutes")  # Add missing attribute

        # === Application Settings ===
        self.database_path = os.getenv("DB_PATH", "data/conjecture.db")
        self.database_type = os.getenv(
            "DATABASE_TYPE", "sqlite"
        )  # Add missing attribute
        self.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.95"))
        self.max_context_size = int(os.getenv("MAX_CONTEXT_SIZE", "10"))
        self.batch_size = int(os.getenv("BATCH_SIZE", "10"))
        self.debug = os.getenv("DEBUG", "false").lower() == "true"

        # === Derived Settings ===
        self.llm_enabled = (
            bool(self.provider_api_key) or "localhost" in self.provider_api_url
        )
        self.data_dir = Path(self.database_path).parent
        self.data_dir.mkdir(exist_ok=True)

    def __str__(self) -> str:
        """String representation for debugging"""
        return f"Config(db={self.database_type}, path={self.database_path}, confidence={self.confidence_threshold}, llm={'enabled' if self.llm_enabled else 'disabled'})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "database_type": self.database_type,
            "database_path": self.database_path,
            "database_type": self.database_type,
            "confidence_threshold": self.confidence_threshold,
            "max_context_size": self.max_context_size,
            "exploration_batch_size": self.exploration_batch_size,
            "embedding_model": self.embedding_model,
            "llm_enabled": self.llm_enabled,
            "llm_model": self.llm_model,
            "debug": self.debug,
        }

    @property
    def chroma_settings(self) -> Dict[str, Any]:
        """ChromaDB-specific settings (only when using chroma database)"""
        if self.database_type != "chroma":
            return {}

        return {
            "collection_name": "claims",
            "host": os.getenv("CHROMA_HOST", "localhost"),
            "port": int(os.getenv("CHROMA_PORT", "8000")),
            "path": os.getenv("CHROMA_PATH", "data/chroma_db"),
        }

    @property
    def llm_settings(self) -> Dict[str, Any]:
        """LLM-specific settings (only when LLM is enabled)"""
        if not self.llm_enabled:
            return {}

        return {
            "model": self.llm_model,
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.3")),
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "2000")),
            "timeout": int(os.getenv("LLM_TIMEOUT", "30")),
        }

    @property
    def validation_rules(self) -> Dict[str, Any]:
        """Validation rules for claims"""
        return {
            "min_content_length": int(os.getenv("MIN_CONTENT_LENGTH", "10")),
            "max_content_length": int(os.getenv("MAX_CONTENT_LENGTH", "2000")),
            "min_confidence": float(os.getenv("MIN_CONFIDENCE", "0.0")),
            "max_confidence": float(os.getenv("MAX_CONFIDENCE", "1.0")),
        }

    @property
    def performance_settings(self) -> Dict[str, Any]:
        """Performance-related settings"""
        return {
            "query_timeout_ms": int(os.getenv("QUERY_TIMEOUT_MS", "100")),
            "max_retries": int(os.getenv("MAX_RETRIES", "3")),
            "retry_delay": float(os.getenv("RETRY_DELAY", "1.0")),
        }


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance"""
    return config


def validate_config() -> bool:
    """Validate the configuration settings"""
    try:
        # Test basic settings
        assert config.confidence_threshold >= 0.0
        assert config.confidence_threshold <= 1.0
        assert config.max_context_size > 0
        assert config.exploration_batch_size > 0

        # Test file paths
        assert config.data_dir.exists() or config.data_dir.parent.exists()

        # Test LLM consistency
        if config.llm_enabled:
            assert config.llm_model is not None

        print("[OK] Configuration validation: PASS")
        return True
    except Exception as e:
        print(f"[ERROR] Configuration validation failed: {e}")
        return False


def print_config_summary():
    """Print a summary of current configuration"""
    print("[CONFIG] Conjecture Configuration Summary")
    print("=" * 40)
    print(f"Database Type: {config.database_type}")
    print(f"Database Path: {config.database_path}")
    print(f"Confidence Threshold: {config.confidence_threshold}")
    print(f"Max Context Size: {config.max_context_size}")
    print(f"Batch Size: {config.exploration_batch_size}")
    print(f"Embedding Model: {config.embedding_model}")
    print(f"LLM Enabled: {'Yes' if config.llm_enabled else 'No'}")
    if config.llm_enabled:
        print(f"LLM Provider: {config.llm_provider}")
        print(f"LLM Model: {config.llm_model}")
        print(f"LLM API URL: {config.llm_api_url}")
    print(f"Debug Mode: {'On' if config.debug else 'Off'}")
    print(f"Data Directory: {config.data_dir}")
    print()


if __name__ == "__main__":
    print("[TEST] Testing Simplified Configuration")
    print("=" * 40)

    if validate_config():
        print_config_summary()
        print("[SUCCESS] Configuration validation passed!")
    else:
        print("[ERROR] Configuration validation failed")
