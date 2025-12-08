"""
Groq Integration - Compatibility Layer
Provides Groq integration for testing
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class GroqProcessor(BaseModel):
    """Mock Groq processor for testing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = "groq"
        self.url = config.get('url', 'https://api.groq.com/openai/v1')
        self.api_key = config.get('api', '')
        self.model = config.get('model', 'llama3-8b-8192')
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using Groq"""
        # Mock implementation for testing
        return f"Groq mock response to: {prompt[:100]}..."
    
    def is_available(self) -> bool:
        """Check if Groq is available"""
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
        return f"Groq chat completion for {len(messages)} messages"
    
    def get_supported_models(self) -> List[str]:
        """Get supported models"""
        return [
            "llama3-8b-8192",
            "llama3-70b-8192",
            "mixtral-8x7b-32768"
        ]

# Export the main class
__all__ = ['GroqProcessor']