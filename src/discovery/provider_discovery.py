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
        return self.dict()

class ProviderDiscovery(BaseModel):
    """Mock provider discovery for testing"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.discovered_providers: List[ProviderInfo] = []
        self.scan_history: List[Dict[str, Any]] = []
    
    def discover_providers(self, scan_urls: Optional[List[str]] = None) -> List[ProviderInfo]:
        """Discover available providers"""
        # Mock implementation for testing
        mock_providers = [
            ProviderInfo(
                name="ollama",
                url="http://localhost:11434",
                model="llama2",
                available=True,
                latency_ms=50.0,
                capabilities=["text-generation", "chat"]
            ),
            ProviderInfo(
                name="chutes",
                url="https://llm.chutes.ai/v1",
                model="zai-org/GLM-4.6-FP8",
                available=True,
                latency_ms=150.0,
                capabilities=["text-generation", "chat", "analysis"]
            )
        ]
        
        self.discovered_providers = mock_providers
        self.scan_history.append({
            "timestamp": "2025-12-08T15:33:00Z",
            "providers_found": len(mock_providers),
            "scan_urls": scan_urls or []
        })
        
        return mock_providers
    
    def get_available_providers(self) -> List[ProviderInfo]:
        """Get only available providers"""
        return [p for p in self.discovered_providers if p.available]
    
    def get_provider_by_name(self, name: str) -> Optional[ProviderInfo]:
        """Get provider by name"""
        for provider in self.discovered_providers:
            if provider.name == name:
                return provider
        return None
    
    def test_provider(self, provider_info: ProviderInfo) -> bool:
        """Test a specific provider"""
        # Mock implementation for testing
        return provider_info.available
    
    def get_scan_history(self) -> List[Dict[str, Any]]:
        """Get scan history"""
        return self.scan_history.copy()
    
    def clear_discovered_providers(self) -> bool:
        """Clear all discovered providers"""
        self.discovered_providers.clear()
        return True
    
    def add_provider(self, provider_info: ProviderInfo) -> bool:
        """Add a discovered provider"""
        self.discovered_providers.append(provider_info)
        return True
    
    def remove_provider(self, name: str) -> bool:
        """Remove a discovered provider"""
        for i, provider in enumerate(self.discovered_providers):
            if provider.name == name:
                del self.discovered_providers[i]
                return True
        return False

# Export the main classes
__all__ = ['ProviderDiscovery', 'ProviderInfo']