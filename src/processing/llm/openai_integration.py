"""
OpenAI Integration - Compatibility Layer
Provides OpenAI integration for testing
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class OpenAIProcessor(BaseModel):
    """Mock OpenAI processor for testing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = "openai"
        self.url = config.get('url', 'https://api.openai.com/v1')
        self.api_key = config.get('api', '')
        self.model = config.get('model', 'gpt-3.5-turbo')
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using OpenAI"""
        # Mock implementation for testing
        return f"OpenAI mock response to: {prompt[:100]}..."
    
    def is_available(self) -> bool:
        """Check if OpenAI is available"""
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
        return f"OpenAI chat completion for {len(messages)} messages"
    
    def get_supported_models(self) -> List[str]:
        """Get supported models"""
        return [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o"
        ]

# Export the main class
__all__ = ['OpenAIProcessor']