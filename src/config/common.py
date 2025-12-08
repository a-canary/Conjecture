"""
Common Config - Compatibility Layer
Provides common configuration functionality for testing
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class ProviderConfig(BaseModel):
    """Provider configuration for testing"""
    name: str
    url: str
    api: str = ""
    model: str
    available: bool = True
    timeout: int = 30
    max_retries: int = 3

class CommonConfig(BaseModel):
    """Common configuration for testing"""
    providers: List[ProviderConfig] = []
    debug: bool = False
    confidence_threshold: float = 0.95
    max_context_size: int = 10
    database_path: str = "data/conjecture.db"
    user: str = "user"
    team: str = "default"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CommonConfig':
        """Create config from dictionary"""
        providers = []
        for provider_data in config_dict.get('providers', []):
            providers.append(ProviderConfig(**provider_data))
        
        return cls(
            providers=providers,
            debug=config_dict.get('debug', False),
            confidence_threshold=config_dict.get('confidence_threshold', 0.95),
            max_context_size=config_dict.get('max_context_size', 10),
            database_path=config_dict.get('database_path', 'data/conjecture.db'),
            user=config_dict.get('user', 'user'),
            team=config_dict.get('team', 'default')
        )

# Export the main classes
__all__ = ['ProviderConfig', 'CommonConfig']