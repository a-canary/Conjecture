"""
Enhanced Configuration System with Local Services Support
Adds configuration for local embeddings, LLM, and vector storage
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
from enum import Enum

from .simple_config import Config

class ServiceMode(Enum):
    """Service operation modes"""
    LOCAL = "local"
    EXTERNAL = "external"
    DISABLED = "disabled"
    AUTO = "auto"  # Try local first, fallback to external

class LocalConfig(Config):
    """
    Extended configuration with local services support
    Inherits from simple_config and adds local service options
    """

    def __init__(self):
        super().__init__()
        
        # === Local Services Configuration ===
        
        # Embedding Service
        self.embedding_mode = ServiceMode(
            os.getenv("Conjecture_EMBEDDING_MODE", "local")
        )
        self.local_embedding_model = os.getenv(
            "Conjecture_LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )
        self.embedding_cache_dir = os.getenv(
            "Conjecture_EMBEDDING_CACHE_DIR", 
            str(Path.home() / ".conjecture" / "models")
        )
        self.embedding_batch_size = int(
            os.getenv("Conjecture_EMBEDDING_BATCH_SIZE", "8")
        )
        
        # Vector Storage
        self.vector_store_mode = ServiceMode(
            os.getenv("Conjecture_VECTOR_STORE_MODE", "local")
        )
        self.vector_store_type = os.getenv(
            "Conjecture_VECTOR_STORE_TYPE", "faiss_sqlite"
        )  # faiss_sqlite, chroma, sqlite_only
        self.vector_store_path = os.getenv(
            "Conjecture_VECTOR_STORE_PATH", "data/local_vector_store.db"
        )
        self.faiss_index_type = os.getenv(
            "Conjecture_FAISS_INDEX_TYPE", "flat"
        )  # flat, ivf_flat
        self.use_faiss = os.getenv(
            "Conjecture_USE_FAISS", "true"
        ).lower() == "true"
        
        # Local LLM (Ollama/LM Studio)
        self.llm_mode = ServiceMode(
            os.getenv("Conjecture_LLM_MODE", "auto")
        )
        self.ollama_base_url = os.getenv(
            "Conjecture_OLLAMA_URL", "http://localhost:11434"
        )
        self.lm_studio_url = os.getenv(
            "Conjecture_LM_STUDIO_URL", "http://localhost:1234"
        )
        self.llm_timeout = int(
            os.getenv("Conjecture_LLM_TIMEOUT", "60")
        )
        self.local_llm_model = os.getenv(
            "Conjecture_LOCAL_LLM_MODEL", ""
        )  # Empty = auto-select
        
        # Service Health and Fallback
        self.health_check_interval = int(
            os.getenv("Conjecture_HEALTH_CHECK_INTERVAL", "30")
        )  # seconds
        self.fallback_enabled = os.getenv(
            "Conjecture_FALLBACK_ENABLED", "true"
        ).lower() == "true"
        self.service_retry_count = int(
            os.getenv("Conjecture_SERVICE_RETRY_COUNT", "3")
        )
        
        # Performance Optimization
        self.enable_caching = os.getenv(
            "Conjecture_ENABLE_CACHING", "true"
        ).lower() == "true"
        self.cache_ttl = int(
            os.getenv("Conjecture_CACHE_TTL", "3600"
        ))  # seconds
        self.max_memory_usage = int(
            os.getenv("Conjecture_MAX_MEMORY_MB", "1024"
        ))  # MB
        
        # Initialize derived settings
        self._init_derived_settings()

    def _init_derived_settings(self):
        """Initialize derived configuration settings"""
        # Create necessary directories
        Path(self.embedding_cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.vector_store_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Effective configuration based on modes
        self.use_local_embeddings = (
            self.embedding_mode in (ServiceMode.LOCAL, ServiceMode.AUTO)
        )
        self.use_local_vector_store = (
            self.vector_store_mode in (ServiceMode.LOCAL, ServiceMode.AUTO)
        )
        self.use_local_llm = (
            self.llm_mode in (ServiceMode.LOCAL, ServiceMode.AUTO)
        )

    @property
    def embedding_config(self) -> Dict[str, Any]:
        """Get embedding service configuration"""
        return {
            "mode": self.embedding_mode.value,
            "model": self.local_embedding_model,
            "cache_dir": self.embedding_cache_dir,
            "batch_size": self.embedding_batch_size,
            "use_local": self.use_local_embeddings,
            "fallback_enabled": self.fallback_enabled
        }

    @property
    def vector_store_config(self) -> Dict[str, Any]:
        """Get vector store configuration"""
        return {
            "mode": self.vector_store_mode.value,
            "type": self.vector_store_type,
            "path": self.vector_store_path,
            "faiss_index_type": self.faiss_index_type,
            "use_faiss": self.use_faiss and self.use_local_vector_store,
            "use_local": self.use_local_vector_store,
            "fallback_enabled": self.fallback_enabled
        }

    @property
    def llm_config(self) -> Dict[str, Any]:
        """Get LLM service configuration"""
        base_config = super().llm_settings.copy()
        
        local_config = {
            "mode": self.llm_mode.value,
            "use_local": self.use_local_llm,
            "ollama_url": self.ollama_base_url,
            "lm_studio_url": self.lm_studio_url,
            "timeout": self.llm_timeout,
            "preferred_model": self.local_llm_model,
            "fallback_enabled": self.fallback_enabled and self.llm_mode == ServiceMode.AUTO
        }
        
        return {**base_config, **local_config}

    @property
    def performance_config(self) -> Dict[str, Any]:
        """Get performance optimization configuration"""
        base_config = super().performance_settings.copy()
        
        local_config = {
            "enable_caching": self.enable_caching,
            "cache_ttl": self.cache_ttl,
            "max_memory_mb": self.max_memory_usage,
            "health_check_interval": self.health_check_interval,
            "retry_count": self.service_retry_count
        }
        
        return {**base_config, **local_config}

    def is_local_first(self) -> bool:
        """Check if local services are preferred over external"""
        return (
            self.embedding_mode == ServiceMode.LOCAL or
            self.vector_store_mode == ServiceMode.LOCAL or
            self.llm_mode == ServiceMode.LOCAL
        )

    def supports_offline(self) -> bool:
        """Check if configuration supports offline operation"""
        return (
            self.use_local_embeddings and 
            self.use_local_vector_store and 
            (self.use_local_llm or not self.llm_enabled)
        )

    def get_primary_llm_url(self) -> str:
        """Get the primary LLM URL based on configuration"""
        if self.llm_mode == ServiceMode.LOCAL:
            # Try Ollama first, then LM Studio
            return self.ollama_base_url
        elif self.llm_mode == ServiceMode.AUTO:
            # Return Ollama as primary (fallback logic in service layer)
            return self.ollama_base_url
        else:
            # External mode - use configured API URL
            return self.llm_api_url

    def get_fallback_llm_url(self) -> Optional[str]:
        """Get fallback LLM URL for auto mode"""
        if self.llm_mode == ServiceMode.AUTO:
            # LM Studio as fallback for Ollama
            return self.lm_studio_url
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert full configuration to dictionary"""
        base_dict = super().to_dict()
        
        local_dict = {
            "embedding": self.embedding_config,
            "vector_store": self.vector_store_config,
            "llm": self.llm_config,
            "performance": self.performance_config,
            "offline_capable": self.supports_offline(),
            "local_first": self.is_local_first()
        }
        
        return {**base_dict, **local_dict}

    def validate_local_config(self) -> bool:
        """Validate local-specific configuration"""
        try:
            # Validate embedding configuration
            assert self.embedding_batch_size > 0
            assert self.embedding_cache_dir
            
            # Validate vector store configuration
            assert self.faiss_index_type in ("flat", "ivf_flat")
            assert self.vector_store_path
            
            # Validate LLM configuration
            assert self.llm_timeout > 0
            assert self.ollama_base_url.startswith("http")
            assert self.lm_studio_url.startswith("http")
            
            # Validate performance settings
            assert self.max_memory_usage > 0
            assert self.cache_ttl > 0
            
            print("[OK] Local configuration validation: PASS")
            return True
            
        except Exception as e:
            print(f"[ERROR] Local configuration validation failed: {e}")
            return False

    def print_local_config_summary(self):
        """Print local services configuration summary"""
        print("[LOCAL] Local Services Configuration")
        print("=" * 50)
        
        print(f"Embedding Mode: {self.embedding_mode.value}")
        print(f"  Model: {self.local_embedding_model}")
        print(f"  Cache: {self.embedding_cache_dir}")
        print(f"  Batch Size: {self.embedding_batch_size}")
        
        print(f"\nVector Store Mode: {self.vector_store_mode.value}")
        print(f"  Type: {self.vector_store_type}")
        print(f"  Path: {self.vector_store_path}")
        print(f"  FAISS Index: {self.faiss_index_type}")
        print(f"  Use FAISS: {self.use_faiss}")
        
        print(f"\nLLM Mode: {self.llm_mode.value}")
        print(f"  Ollama URL: {self.ollama_base_url}")
        print(f"  LM Studio URL: {self.lm_studio_url}")
        print(f"  Timeout: {self.llm_timeout}s")
        if self.local_llm_model:
            print(f"  Preferred Model: {self.local_llm_model}")
        
        print(f"\nPerformance Settings:")
        print(f"  Caching: {'Enabled' if self.enable_caching else 'Disabled'}")
        print(f"  Max Memory: {self.max_memory_usage}MB")
        print(f"  Health Checks: Every {self.health_check_interval}s")
        print(f"  Fallback: {'Enabled' if self.fallback_enabled else 'Disabled'}")
        
        print(f"\nCapabilities:")
        print(f"  Offline Operation: {'Supported' if self.supports_offline() else 'Limited'}")
        print(f"  Local-First: {'Yes' if self.is_local_first() else 'No'}")
        print()

    def create_env_template(self) -> str:
        """Create .env template file content"""
        return """# Conjecture Local Services Configuration
# Copy this to your .env file and modify as needed

# === Embedding Service ===
# Options: local, external, disabled, auto
Conjecture_EMBEDDING_MODE=local

# Local embedding model (sentence-transformers)
Conjecture_LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2

# Embedding cache directory
Conjecture_EMBEDDING_CACHE_DIR=~/.conjecture/models

# Embedding batch size for processing
Conjecture_EMBEDDING_BATCH_SIZE=8

# === Vector Storage ===
# Options: local, external, disabled, auto
Conjecture_VECTOR_STORE_MODE=local

# Storage type: faiss_sqlite, chroma, sqlite_only
Conjecture_VECTOR_STORE_TYPE=faiss_sqlite

# Vector store database path
Conjecture_VECTOR_STORE_PATH=data/local_vector_store.db

# FAISS index type: flat, ivf_flat
Conjecture_FAISS_INDEX_TYPE=flat

# Enable/disable FAISS acceleration
Conjecture_USE_FAISS=true

# === Local LLM (Ollama/LM Studio) ===
# Options: local, external, disabled, auto
Conjecture_LLM_MODE=auto

# Ollama service URL
Conjecture_OLLAMA_URL=http://localhost:11434

# LM Studio service URL
Conjecture_LM_STUDIO_URL=http://localhost:1234

# LLM request timeout (seconds)
Conjecture_LLM_TIMEOUT=60

# Preferred local model (empty = auto-select)
Conjecture_LOCAL_LLM_MODEL=

# === Performance & Reliability ===
# Health check interval (seconds)
Conjecture_HEALTH_CHECK_INTERVAL=30

# Enable fallback to external services
Conjecture_FALLBACK_ENABLED=true

# Service retry count on failures
Conjecture_SERVICE_RETRY_COUNT=3

# Enable response caching
Conjecture_ENABLE_CACHING=true

# Cache TTL (seconds)
Conjecture_CACHE_TTL=3600

# Maximum memory usage (MB)
Conjecture_MAX_MEMORY_MB=1024
"""

# Factory functions
def create_local_config() -> LocalConfig:
    """Create local configuration instance"""
    return LocalConfig()

def get_local_config() -> LocalConfig:
    """Get global local configuration instance"""
    global _local_config
    if '_local_config' not in globals():
        _local_config = LocalConfig()
    return _local_config

def validate_and_print_config():
    """Validate configuration and print summary"""
    print("[CONFIG] Validating Local Services Configuration")
    print("=" * 60)
    
    config = get_local_config()
    
    # Validate
    if config.validate_local_config():
        # Print summaries
        config.print_local_config_summary()
        print("[SUCCESS] Local services configuration is valid!")
    else:
        print("[ERROR] Local services configuration has issues")

if __name__ == "__main__":
    validate_and_print_config()
    
    # Generate .env template
    print("\n[TEMPLATE] .env file template:")
    print("=" * 40)
    config = create_local_config()
    print(config.create_env_template())