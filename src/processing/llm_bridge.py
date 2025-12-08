"""
LLM Bridge - Compatibility Layer
Provides LLM bridging functionality for testing
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class LLMRequest(BaseModel):
    """LLM request model for testing"""
    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.dict()

class LLMResponse(BaseModel):
    """LLM response model for testing"""
    content: str
    model: str
    provider: str
    tokens_used: Optional[int] = None
    response_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.dict()

class LLMBridge:
    """Mock LLM bridge for testing"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.providers = {}
    
    def add_provider(self, name: str, provider: Any) -> bool:
        """Add a provider to the bridge"""
        self.providers[name] = provider
        return True
    
    def remove_provider(self, name: str) -> bool:
        """Remove a provider from the bridge"""
        if name in self.providers:
            del self.providers[name]
            return True
        return False
    
    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response using the bridge"""
        # Mock implementation for testing
        return LLMResponse(
            content=f"Bridge mock response to: {request.prompt[:100]}...",
            model=request.model or "default",
            provider=request.provider or "default",
            tokens_used=100,
            response_time=0.5
        )
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return list(self.providers.keys())
    
    def is_provider_available(self, name: str) -> bool:
        """Check if a provider is available"""
        return name in self.providers

# Export the main classes
__all__ = ['LLMRequest', 'LLMResponse', 'LLMBridge']