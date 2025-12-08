"""
Anthropic Integration - Compatibility Layer
Provides Anthropic integration for testing
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class AnthropicProcessor(BaseModel):
    """Mock Anthropic processor for testing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = "anthropic"
        self.url = config.get('url', 'https://api.anthropic.com')
        self.api_key = config.get('api', '')
        self.model = config.get('model', 'claude-3-sonnet-20240229')
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using Anthropic"""
        # Mock implementation for testing
        return f"Anthropic mock response to: {prompt[:100]}..."
    
    def is_available(self) -> bool:
        """Check if Anthropic is available"""
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": self.name,
            "url": self.url,
            "model": self.model,
            "available": self.is_available()
        }
    
    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate chat completion"""
        # Mock implementation for testing
        return f"Anthropic chat completion for {len(messages)} messages"
    
    def get_supported_models(self) -> List[str]:
        """Get supported models"""
        return [
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-opus-20240229"
        ]

# Export the main class
__all__ = ['AnthropicProcessor']