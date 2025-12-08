"""
Settings - Compatibility Layer
Provides settings functionality for testing
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel

# Dirty flag configuration constants
DIRTY_FLAG_CONFIDENCE_THRESHOLD = 0.8
DIRTY_FLAG_ENABLED = True
DIRTY_FLAG_AUTO_CLEAN = False
DIRTY_FLAG_CASCADE_DEPTH = 3
DIRTY_FLAG_BATCH_SIZE = 100
DIRTY_FLAG_MAX_PARALLEL_BATCHES = 10

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
        return self.dict()

# Export the main class and constants
__all__ = ['Settings', 'DIRTY_FLAG_CONFIDENCE_THRESHOLD', 'DIRTY_FLAG_ENABLED', 'DIRTY_FLAG_AUTO_CLEAN']