"""
LLM Provider
Minimal provider-specific code using HTTP requests
"""

import json
import requests
from typing import Dict, Any, Optional
from ...config.config import Config


class Provider:
    """Single provider class that handles all LLM APIs from config file"""

    def __init__(self, config: Config):
        self.config = config
        self.current_provider_index = 0
        self.available_providers = []
        
        # Load providers from config
        self._load_providers_from_config()

    def _load_providers_from_config(self):
        """Load provider configurations from the config file"""
        provider_configs = self.config.get_providers()
        
        for provider_config in provider_configs:
            provider_name = provider_config.get("name", "unknown")
            provider_url = provider_config.get("url", "")
            provider_api = provider_config.get("api", "")
            provider_model = provider_config.get("model", "")
            
            # Skip if required fields are missing
            if not provider_url or not provider_model:
                continue
            
            # Build provider configuration
            if provider_name == "chutes":
                self.available_providers.append({
                    "name": provider_name,
                    "url": f"{provider_url}/chat/completions",
                    "headers": {"Authorization": f"Bearer {provider_api}"},
                    "model": provider_model,
                    "has_api_key": bool(provider_api),
                })
            elif provider_name == "lm_studio":
                self.available_providers.append({
                    "name": provider_name,
                    "url": f"{provider_url}/v1/chat/completions",
                    "headers": {},
                    "model": provider_model,
                    "has_api_key": True,  # Local providers don't need API keys
                })
            elif provider_name == "openrouter":
                self.available_providers.append({
                    "name": provider_name,
                    "url": f"{provider_url}/chat/completions",
                    "headers": {"Authorization": f"Bearer {provider_api}"},
                    "model": provider_model,
                    "has_api_key": bool(provider_api),
                })
            elif provider_name == "openai":
                self.available_providers.append({
                    "name": provider_name,
                    "url": f"{provider_url}/chat/completions",
                    "headers": {"Authorization": f"Bearer {provider_api}"},
                    "model": provider_model,
                    "has_api_key": bool(provider_api),
                })
            elif provider_name == "anthropic":
                self.available_providers.append({
                    "name": provider_name,
                    "url": f"{provider_url}/messages",
                    "headers": {"x-api-key": provider_api, "anthropic-version": "2023-06-01"},
                    "model": provider_model,
                    "has_api_key": bool(provider_api),
                })
            elif provider_name == "ollama":
                self.available_providers.append({
                    "name": provider_name,
                    "url": f"{provider_url}/api/generate",
                    "headers": {},
                    "model": provider_model,
                    "has_api_key": True,
                    "use_ollama_format": True,  # Special flag for Ollama format
                })
            else:
                # Generic provider configuration
                self.available_providers.append({
                    "name": provider_name,
                    "url": provider_url,
                    "headers": {"Authorization": f"Bearer {provider_api}"} if provider_api else {},
                    "model": provider_model,
                    "has_api_key": bool(provider_api),
                })

    def is_available(self) -> bool:
        """Check if any provider is available"""
        return len(self.available_providers) > 0

    def _test_provider(self, provider_config: Dict[str, Any]) -> bool:
        """Test if a specific provider is available"""
        try:
            url = provider_config["url"]
            headers = provider_config.get("headers", {})
            
            # Simple test request - just check if the endpoint responds
            response = requests.get(url.replace("/chat/completions", "/models"), 
                                  headers=headers, timeout=5)
            return response.status_code < 500
        except:
            return False

    def _get_current_provider(self) -> Optional[Dict[str, Any]]:
        """Get the current provider, cycling through if needed"""
        if not self.available_providers:
            return None
            
        # Try current provider
        provider = self.available_providers[self.current_provider_index]
        if self._test_provider(provider):
            return provider
        
        # If not available, try the next one
        for i in range(len(self.available_providers)):
            if i != self.current_provider_index:
                if self._test_provider(self.available_providers[i]):
                    self.current_provider_index = i
                    return self.available_providers[i]
        
        return None

    def generate(self, messages: list, model: str = None, **kwargs) -> Dict[str, Any]:
        """Generate response using available provider"""
        provider = self._get_current_provider()
        if not provider:
            raise Exception("No provider available")
        
        # Use provided model or provider's default
        model_to_use = model or provider["model"]
        
        # Handle different API formats
        if provider.get("use_ollama_format"):
            return self._generate_ollama(messages, provider, model_to_use, **kwargs)
        else:
            return self._generate_openai_format(messages, provider, model_to_use, **kwargs)

    def _generate_openai_format(self, messages: list, provider: Dict[str, Any], 
                                model: str, **kwargs) -> Dict[str, Any]:
        """Generate using OpenAI-compatible format"""
        url = provider["url"]
        headers = provider["headers"]
        
        # Default OpenAI request format
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
        }
        
        response = requests.post(url, json=data, headers=headers, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return {
            "content": result["choices"][0]["message"]["content"],
            "usage": result.get("usage", {}),
            "provider": provider["name"]
        }

    def _generate_ollama(self, messages: list, provider: Dict[str, Any], 
                        model: str, **kwargs) -> Dict[str, Any]:
        """Generate using Ollama format"""
        url = provider["url"]
        headers = provider["headers"]
        
        # Convert messages to prompt for Ollama
        prompt = ""
        for message in messages:
            if message["role"] == "system":
                prompt += f"System: {message['content']}\n\n"
            elif message["role"] == "user":
                prompt += f"User: {message['content']}\n\n"
            elif message["role"] == "assistant":
                prompt += f"Assistant: {message['content']}\n\n"
        
        prompt += "Assistant: "
        
        # Ollama request format
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "num_predict": kwargs.get("max_tokens", 1000),
            }
        }
        
        response = requests.post(url, json=data, headers=headers, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return {
            "content": result.get("response", ""),
            "usage": {},  # Ollama doesn't provide detailed usage info
            "provider": provider["name"]
        }

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about available providers"""
        return {
            "available_providers": len(self.available_providers),
            "current_provider": self.available_providers[self.current_provider_index]["name"] if self.available_providers else None,
            "providers": [
                {
                    "name": p["name"],
                    "url": p["url"],
                    "model": p["model"],
                    "has_api_key": p["has_api_key"]
                }
                for p in self.available_providers
            ]
        }


def create_provider(config: Config) -> Provider:
    """Create a provider instance from config"""
    return Provider(config)