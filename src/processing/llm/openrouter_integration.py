"""
OpenRouter Integration - Compatibility Layer
Provides OpenRouter integration for testing
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class OpenRouterProcessor(BaseModel):
    """Mock OpenRouter processor for testing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = "openrouter"
        self.url = config.get('url', 'https://openrouter.ai/api/v1')
        self.api_key = config.get('api', '')
        self.model = config.get('model', 'meta-llama/llama-3-8b-instruct')
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using OpenRouter"""
        # Mock implementation for testing
        return f"OpenRouter mock response to: {prompt[:100]}..."
    
    def is_available(self) -> bool:
        """Check if OpenRouter is available"""
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": self.name,
            "url": self.url,
            "model": self.model,
            "available": self.is_available()
        }

# Export the main class
__all__ = ['OpenRouterProcessor']