"""
Unified Configuration for Conjecture
Single source of truth with no monkey-patching
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


class Config:
    """
    Unified configuration class for Conjecture
    All essential settings in one place with proper validation
    """

    def __init__(self):
        # === Core Settings ===
        self.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.95"))
        self.confident_threshold = float(os.getenv("CONFIDENT_THRESHOLD", "0.8"))
        self.session_confident_threshold = None
        self.max_context_size = int(os.getenv("MAX_CONTEXT_SIZE", "10"))
        self.batch_size = int(os.getenv("BATCH_SIZE", "10"))
        self.debug = os.getenv("DEBUG", "false").lower() == "true"

        # === Database Settings ===
        self.database_type = os.getenv("DATABASE_TYPE", "sqlite")
        self.database_path = os.getenv("DB_PATH", "data/conjecture.db")
        self.data_dir = Path(self.database_path).parent
        self.data_dir.mkdir(exist_ok=True)

        # === LLM Provider Settings ===
        self.llm_provider = os.getenv("LLM_PROVIDER", "chutes")
        self.provider_api_url = os.getenv(
            "PROVIDER_API_URL", "https://llm.chutes.ai/v1"
        )
        self.provider_api_key = os.getenv("PROVIDER_API_KEY", "")
        self.provider_model = os.getenv("PROVIDER_MODEL", "zai-org/GLM-4.6-FP8")

        # === Derived Settings ===
        self.llm_enabled = (
            bool(self.provider_api_key) or "localhost" in self.provider_api_url
        )

    # === Confidence Threshold Methods ===
    def set_confident_threshold(self, threshold: float):
        """Set session-level confident threshold override"""
        if 0.0 <= threshold <= 1.0:
            self.session_confident_threshold = threshold
        else:
            raise ValueError("Confident threshold must be between 0.0 and 1.0")

    def get_effective_confident_threshold(self) -> float:
        """Get effective confident threshold (session override or global default)"""
        return self.session_confident_threshold or self.confident_threshold

    def reset_confident_threshold(self):
        """Reset to global default confident threshold"""
        self.session_confident_threshold = None

    # === Utility Methods ===
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "database_type": self.database_type,
            "database_path": self.database_path,
            "confidence_threshold": self.confidence_threshold,
            "max_context_size": self.max_context_size,
            "batch_size": self.batch_size,
            "llm_enabled": self.llm_enabled,
            "llm_provider": self.llm_provider,
            "provider_model": self.provider_model,
            "provider_api_url": self.provider_api_url,
            "debug": self.debug,
        }

    def validate(self) -> bool:
        """Validate configuration settings"""
        try:
            assert 0.0 <= self.confidence_threshold <= 1.0
            assert self.max_context_size > 0
            assert self.batch_size > 0
            assert self.data_dir.exists() or self.data_dir.parent.exists()

            if self.llm_enabled:
                assert self.provider_model is not None

            return True
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False

    def __str__(self) -> str:
        """String representation for debugging"""
        return f"Config(db={self.database_type}, llm={self.llm_provider}, confidence={self.confidence_threshold})"

    # === Property Methods ===
    @property
    def chroma_settings(self) -> Dict[str, Any]:
        """ChromaDB settings (only when using chroma)"""
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
        """LLM settings (only when LLM is enabled)"""
        if not self.llm_enabled:
            return {}
        return {
            "model": self.provider_model,
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.3")),
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "2000")),
            "timeout": int(os.getenv("LLM_TIMEOUT", "30")),
        }


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance"""
    return config


if __name__ == "__main__":
    # Test configuration
    cfg = Config()
    print("Configuration Test:")
    print(f"Valid: {cfg.validate()}")
    print(f"Settings: {cfg.to_dict()}")
    print(f"LLM Enabled: {cfg.llm_enabled}")
    print(f"Debug Mode: {cfg.debug}")
