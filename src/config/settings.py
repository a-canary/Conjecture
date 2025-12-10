"""
Settings - Compatibility Layer
Provides settings functionality for testing
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel

# Dirty flag configuration constants
DIRTY_FLAG_CONFIDENCE_THRESHOLD = 0.8
DIRTY_FLAG_CONFIDENCE_BOOST_FACTOR = 1.2
DIRTY_FLAG_ENABLED = True
DIRTY_FLAG_AUTO_CLEAN = False
DIRTY_FLAG_CASCADE_DEPTH = 3
DIRTY_FLAG_BATCH_SIZE = 100
DIRTY_FLAG_MAX_PARALLEL_BATCHES = 10
DIRTY_FLAG_TWO_PASS_EVALUATION = True
DIRTY_FLAG_RELATIONSHIP_THRESHOLD = 0.7
DIRTY_FLAG_TIMEOUT_SECONDS = 300
DIRTY_FLAG_MAX_RETRIES = 3
DIRTY_FLAG_AUTO_EVALUATION_ENABLED = True
DIRTY_FLAG_EVALUATION_INTERVAL_MINUTES = 5
DIRTY_FLAG_SIMILARITY_THRESHOLD = 0.8
DIRTY_FLAG_PRIORITY_WEIGHTS = {"confidence": 0.4, "recency": 0.3, "importance": 0.3}
DIRTY_FLAG_ENABLE_CASCADE = True
DIRTY_FLAG_MAX_CLAIMS_PER_EVALUATION = 50
DIRTY_FLAG_MIN_DIRTY_CLAIMS_BATCH = 5
DIRTY_FLAG_CACHE_INVALIDATION_MINUTES = 15

class Settings(BaseModel):
    """Settings configuration for testing"""
    debug: bool = False
    log_level: str = "INFO"
    max_workers: int = 4
    timeout: int = 30
    retry_attempts: int = 3
    
    @classmethod
    def from_dict(cls, settings_dict: Dict[str, Any]) -> 'Settings':
        """Create settings from dictionary"""
        return cls(**settings_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return self.model_dump()

# Export the main class and constants
__all__ = ['Settings', 'DIRTY_FLAG_CONFIDENCE_THRESHOLD', 'DIRTY_FLAG_ENABLED', 'DIRTY_FLAG_AUTO_CLEAN']