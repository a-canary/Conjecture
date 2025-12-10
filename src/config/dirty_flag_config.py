"""
Dirty Flag Configuration Management
Handles configuration loading and validation for dirty flag system
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

from .settings import (
    DIRTY_FLAG_CONFIDENCE_THRESHOLD,
    DIRTY_FLAG_CASCADE_DEPTH,
    DIRTY_FLAG_BATCH_SIZE,
    DIRTY_FLAG_MAX_PARALLEL_BATCHES,
    DIRTY_FLAG_CONFIDENCE_BOOST_FACTOR,
    DIRTY_FLAG_TWO_PASS_EVALUATION,
    DIRTY_FLAG_RELATIONSHIP_THRESHOLD,
    DIRTY_FLAG_TIMEOUT_SECONDS,
    DIRTY_FLAG_MAX_RETRIES,
    DIRTY_FLAG_AUTO_EVALUATION_ENABLED,
    DIRTY_FLAG_EVALUATION_INTERVAL_MINUTES,
    DIRTY_FLAG_SIMILARITY_THRESHOLD,
    DIRTY_FLAG_PRIORITY_WEIGHTS,
    DIRTY_FLAG_ENABLE_CASCADE,
    DIRTY_FLAG_MAX_CLAIMS_PER_EVALUATION,
    DIRTY_FLAG_MIN_DIRTY_CLAIMS_BATCH,
    DIRTY_FLAG_CACHE_INVALIDATION_MINUTES
)

@dataclass
class DirtyFlagConfig:
    """Configuration for dirty flag system"""
    
    # Core dirty flag settings
    confidence_threshold: float = DIRTY_FLAG_CONFIDENCE_THRESHOLD
    cascade_depth: int = DIRTY_FLAG_CASCADE_DEPTH
    enable_cascade: bool = DIRTY_FLAG_ENABLE_CASCADE
    
    # Evaluation settings
    batch_size: int = DIRTY_FLAG_BATCH_SIZE
    max_parallel_batches: int = DIRTY_FLAG_MAX_PARALLEL_BATCHES
    confidence_boost_factor: float = DIRTY_FLAG_CONFIDENCE_BOOST_FACTOR
    two_pass_evaluation: bool = DIRTY_FLAG_TWO_PASS_EVALUATION
    relationship_threshold: float = DIRTY_FLAG_RELATIONSHIP_THRESHOLD
    timeout_seconds: int = DIRTY_FLAG_TIMEOUT_SECONDS
    max_retries: int = DIRTY_FLAG_MAX_RETRIES
    
    # Auto-evaluation settings
    auto_evaluation_enabled: bool = DIRTY_FLAG_AUTO_EVALUATION_ENABLED
    evaluation_interval_minutes: int = DIRTY_FLAG_EVALUATION_INTERVAL_MINUTES
    max_claims_per_evaluation: int = DIRTY_FLAG_MAX_CLAIMS_PER_EVALUATION
    min_dirty_claims_batch: int = DIRTY_FLAG_MIN_DIRTY_CLAIMS_BATCH
    
    # Similarity and relationship settings
    similarity_threshold: float = DIRTY_FLAG_SIMILARITY_THRESHOLD
    cache_invalidation_minutes: int = DIRTY_FLAG_CACHE_INVALIDATION_MINUTES
    
    # Priority weights
    priority_weights: Dict[str, float] = None
    
    def __post_init__(self):
        """Validate configuration and set defaults"""
        if self.priority_weights is None:
            self.priority_weights = DIRTY_FLAG_PRIORITY_WEIGHTS.copy()
        
        self._validate()
    
    def _validate(self):
        """Validate configuration values"""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        
        if self.cascade_depth < 1:
            raise ValueError("cascade_depth must be at least 1")
        
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        
        if self.max_parallel_batches < 1:
            raise ValueError("max_parallel_batches must be at least 1")
        
        if not 0.0 <= self.confidence_boost_factor <= 1.0:
            raise ValueError("confidence_boost_factor must be between 0.0 and 1.0")
        
        if not 0.0 <= self.relationship_threshold <= 1.0:
            raise ValueError("relationship_threshold must be between 0.0 and 1.0")
        
        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be at least 1")
        
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        
        if self.evaluation_interval_minutes < 1:
            raise ValueError("evaluation_interval_minutes must be at least 1")
        
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        
        if self.cache_invalidation_minutes < 1:
            raise ValueError("cache_invalidation_minutes must be at least 1")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DirtyFlagConfig":
        """Create configuration from dictionary"""
        # Filter only valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        
        return cls(**filtered_dict)
    
    @classmethod
    def from_env(cls) -> "DirtyFlagConfig":
        """Create configuration from environment variables"""
        config_dict = {}
        
        # Map environment variables to config fields
        env_mappings = {
            "DIRTY_FLAG_CONFIDENCE_THRESHOLD": ("confidence_threshold", float),
            "DIRTY_FLAG_CASCADE_DEPTH": ("cascade_depth", int),
            "DIRTY_FLAG_ENABLE_CASCADE": ("enable_cascade", bool),
            "DIRTY_FLAG_BATCH_SIZE": ("batch_size", int),
            "DIRTY_FLAG_MAX_PARALLEL_BATCHES": ("max_parallel_batches", int),
            "DIRTY_FLAG_CONFIDENCE_BOOST_FACTOR": ("confidence_boost_factor", float),
            "DIRTY_FLAG_TWO_PASS_EVALUATION": ("two_pass_evaluation", bool),
            "DIRTY_FLAG_RELATIONSHIP_THRESHOLD": ("relationship_threshold", float),
            "DIRTY_FLAG_TIMEOUT_SECONDS": ("timeout_seconds", int),
            "DIRTY_FLAG_MAX_RETRIES": ("max_retries", int),
            "DIRTY_FLAG_AUTO_EVALUATION_ENABLED": ("auto_evaluation_enabled", bool),
            "DIRTY_FLAG_EVALUATION_INTERVAL_MINUTES": ("evaluation_interval_minutes", int),
            "DIRTY_FLAG_MAX_CLAIMS_PER_EVALUATION": ("max_claims_per_evaluation", int),
            "DIRTY_FLAG_MIN_DIRTY_CLAIMS_BATCH": ("min_dirty_claims_batch", int),
            "DIRTY_FLAG_SIMILARITY_THRESHOLD": ("similarity_threshold", float),
            "DIRTY_FLAG_CACHE_INVALIDATION_MINUTES": ("cache_invalidation_minutes", int),
        }
        
        for env_var, (field_name, field_type) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    if field_type == bool:
                        config_dict[field_name] = value.lower() in ["true", "1", "yes", "on"]
                    else:
                        config_dict[field_name] = field_type(value)
                except (ValueError, TypeError) as e:
                    print(f"Warning: Invalid value for {env_var}: {value} ({e})")
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    def update(self, **kwargs) -> "DirtyFlagConfig":
        """Create updated configuration with new values"""
        current_dict = self.to_dict()
        current_dict.update(kwargs)
        return DirtyFlagConfig(**current_dict)
    
    def get_priority_weight(self, reason: str) -> float:
        """Get priority weight for a specific dirty reason"""
        return self.priority_weights.get(reason, 0.0)
    
    def set_priority_weight(self, reason: str, weight: float) -> None:
        """Set priority weight for a specific dirty reason"""
        self.priority_weights[reason] = max(0.0, float(weight))
    
    def get_effective_timeout(self) -> int:
        """Calculate effective timeout based on batch size"""
        base_timeout = self.timeout_seconds
        batch_factor = min(2.0, 1.0 + (self.batch_size - 1) * 0.2)
        return int(base_timeout * batch_factor)
    
    def should_auto_evaluate(self, dirty_claim_count: int) -> bool:
        """Determine if auto-evaluation should trigger"""
        if not self.auto_evaluation_enabled:
            return False
        
        return dirty_claim_count >= self.min_dirty_claims_batch
    
    def get_max_evaluation_claims(self, total_dirty: int) -> int:
        """Get maximum number of claims to evaluate in one batch"""
        return min(total_dirty, self.max_claims_per_evaluation)
    
    def is_high_confidence(self, confidence: float) -> bool:
        """Check if confidence is considered high"""
        return confidence >= self.confidence_threshold
    
    def get_priority_bonus(self, confidence: float) -> float:
        """Calculate priority bonus based on confidence gap"""
        if confidence >= self.confidence_threshold:
            return 0.0
        
        gap = self.confidence_threshold - confidence
        return gap * self.priority_weights.get("confidence_gap", 10.0)
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return f"DirtyFlagConfig(threshold={self.confidence_threshold}, batch_size={self.batch_size}, cascade_depth={self.cascade_depth})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"DirtyFlagConfig({self.to_dict()})"

class DirtyFlagConfigManager:
    """Manager for dirty flag configuration"""
    
    def __init__(self, config: Optional[DirtyFlagConfig] = None):
        self.config = config or DirtyFlagConfig()
        self._config_history = []
    
    def get_config(self) -> DirtyFlagConfig:
        """Get current configuration"""
        return self.config
    
    def update_config(self, **kwargs) -> DirtyFlagConfig:
        """Update configuration with new values"""
        # Save current config to history
        self._config_history.append(self.config.to_dict())
        
        # Keep only last 10 configs in history
        if len(self._config_history) > 10:
            self._config_history.pop(0)
        
        # Create updated config
        self.config = self.config.update(**kwargs)
        return self.config
    
    def reset_to_defaults(self) -> DirtyFlagConfig:
        """Reset configuration to defaults"""
        self.config = DirtyFlagConfig()
        return self.config
    
    def load_from_file(self, file_path: str) -> DirtyFlagConfig:
        """Load configuration from JSON file"""
        import json
        
        try:
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
            
            self.config = DirtyFlagConfig.from_dict(config_dict)
            return self.config
            
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {file_path}: {e}")
    
    def save_to_file(self, file_path: str) -> None:
        """Save current configuration to JSON file"""
        import json
        
        try:
            with open(file_path, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
                
        except Exception as e:
            raise RuntimeError(f"Failed to save config to {file_path}: {e}")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration"""
        return {
            "confidence_threshold": self.config.confidence_threshold,
            "batch_size": self.config.batch_size,
            "max_parallel_batches": self.config.max_parallel_batches,
            "cascade_depth": self.config.cascade_depth,
            "auto_evaluation_enabled": self.config.auto_evaluation_enabled,
            "two_pass_evaluation": self.config.two_pass_evaluation,
            "priority_weight_count": len(self.config.priority_weights),
            "config_source": "loaded" if self._config_history else "default"
        }
    
    def get_config_history(self) -> list:
        """Get configuration change history"""
        return self._config_history.copy()

# Global configuration manager instance
_config_manager = None

def get_dirty_flag_config() -> DirtyFlagConfigManager:
    """Get global dirty flag configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = DirtyFlagConfigManager()
    return _config_manager

def update_dirty_flag_config(**kwargs) -> DirtyFlagConfig:
    """Update global dirty flag configuration"""
    return get_dirty_flag_config().update_config(**kwargs)