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
    """Real-time service detection for testing"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.detected_providers: List[DetectedProvider] = []

    def detect_providers(self) -> List[DetectedProvider]:
        """Detect available providers"""
        return []
        