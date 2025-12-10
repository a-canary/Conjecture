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
        return self.model_dump()

class LLMResponse(BaseModel):
    """LLM response model for testing"""
    content: str
    model: str
    provider: str
    tokens_used: Optional[int] = None
    response_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.model_dump()

class LLMBridge:
    """Real
    
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
        