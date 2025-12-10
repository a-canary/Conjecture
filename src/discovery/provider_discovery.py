"""
Provider Discovery - Compatibility Layer
Provides provider discovery functionality for testing
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class ProviderInfo(BaseModel):
    """Provider information model for testing"""
    name: str
    url: str
    model: str
    available: bool = True
    latency_ms: float = 0.0
    capabilities: List[str] = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.model_dump()

class ProviderDiscovery(BaseModel):
    """Real
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.discovered_providers: List[ProviderInfo] = []
        self.scan_history: List[Dict[str, Any]] = []
    
    def discover_providers(self, scan_urls: Optional[List[str]] = None) -> List[ProviderInfo]:
        """Discover available providers"""
        