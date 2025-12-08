"""
Service Detector - Compatibility Layer
Provides service detection functionality for testing
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class DetectedProvider(BaseModel):
    """Information about a detected provider"""
    name: str
    url: str
    model: str
    available: bool = True
    latency_ms: float = 0.0
    error: Optional[str] = None

class ServiceDetector:
    """Mock service detector for testing"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.detected_providers: List[DetectedProvider] = []
    
    def detect_providers(self) -> List[DetectedProvider]:
        """Detect available providers"""
        # Mock some common providers for testing
        mock_providers = [
            DetectedProvider(
                name="ollama",
                url="http://localhost:11434",
                model="llama2",
                available=True,
                latency_ms=50.0
            ),
            DetectedProvider(
                name="chutes",
                url="https://llm.chutes.ai/v1",
                model="zai-org/GLM-4.6-FP8",
                available=True,
                latency_ms=150.0
            )
        ]
        self.detected_providers = mock_providers
        return mock_providers
    
    def check_provider(self, url: str, api_key: Optional[str] = None) -> DetectedProvider:
        """Check a specific provider"""
        return DetectedProvider(
            name="unknown",
            url=url,
            model="unknown",
            available=True,
            latency_ms=100.0
        )
    
    def get_available_providers(self) -> List[DetectedProvider]:
        """Get only available providers"""
        return [p for p in self.detected_providers if p.available]
    
    def get_provider_by_name(self, name: str) -> Optional[DetectedProvider]:
        """Get provider by name"""
        for provider in self.detected_providers:
            if provider.name == name:
                return provider
        return None

# Export the main classes
__all__ = ['ServiceDetector', 'DetectedProvider']