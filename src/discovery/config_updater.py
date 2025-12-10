"""
Config Updater - Compatibility Layer
Provides configuration updating functionality for testing
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class ConfigUpdate(BaseModel):
    """Configuration update model for testing"""
    key: str
    value: Any
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.dict()

class ConfigUpdater(BaseModel):
    """Real
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.updates: List[ConfigUpdate] = []
        self.current_config: Dict[str, Any] = {}
    
    def add_update(self, key: str, value: Any) -> bool:
        """Add a configuration update"""
        update = ConfigUpdate(key=key, value=value)
        self.updates.append(update)
        return True
    
    def apply_updates(self) -> bool:
        """Apply all pending updates"""
        for update in self.updates:
            self.current_config[update.key] = update.value
        self.updates.clear()
        return True
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.current_config.copy()
    
    def set_config(self, config: Dict[str, Any]) -> bool:
        """Set the entire configuration"""
        self.current_config = config.copy()
        return True
    
    def save_config(self, path: Optional[str] = None) -> bool:
        """Save configuration to file"""
        