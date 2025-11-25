"""
Simple Unified LLM Provider
Minimal provider-specific code using HTTP requests
"""

import json
import requests
from typing import Dict, Any, Optional
from ...config.simple_config import Config


class SimpleUnifiedProvider:
    """Single provider class that handles all LLM APIs"""

    def __init__(self, config: Config):
        self.config = config
        self.provider_configs = {
            "chutes": {
                "url": "https://llm.chutes.ai/v1/chat/completions",
                "headers": {"Authorization": f"Bearer {config.provider_api_key}"},
                "model": config.provider_model,
            },
            "lm_studio": {
                "url": "http://localhost:1234/v1/chat/completions",
                "headers": {},
                "model": "local-model",
            },
            "openai": {
                "url": "https://api.openai.com/v1/chat/completions",
                "headers": {"Authorization": f"Bearer {config.provider_api_key}"},
                "model": "gpt-3.5-turbo",
            },
            "anthropic": {
                "url": "https://api.anthropic.com/v1/messages",
                "headers": {"x-api-key": config.provider_api_key},
                "model": "claude-3-haiku-20240307",
            },
        }

    def is_available(self) -> bool:
        """Check if any provider is available"""
        for provider_name in self.provider_configs:
            if self._test_provider(provider_name):
                return True
        return False

    def _test_provider(self, provider_name: str) -> bool:
        """Test if a specific provider is available"""
        config = self.provider_configs[provider_name]
        try:
            # Simple health check - try a minimal request
            response = requests.post(
                config["url"],
                headers=config["headers"],
                json={
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 1,
                },
                timeout=5,
            )
            return response.status_code in [
                200,
                201,
                400,
            ]  # 400 might mean bad format but service is up
        except:
            return False

    def process_request(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Process request using first available provider"""
        preferred = (
            self.config.llm_provider.lower() if self.config.llm_provider else "chutes"
        )

        # Try preferred provider first
        if preferred in self.provider_configs:
            result = self._call_provider(preferred, prompt, **kwargs)
            if result["success"]:
                return result

        # Try other providers
        for provider_name in self.provider_configs:
            if provider_name != preferred:
                result = self._call_provider(provider_name, prompt, **kwargs)
                if result["success"]:
                    return result

        return {"success": False, "content": "", "error": "No providers available"}

    def _call_provider(
        self, provider_name: str, prompt: str, **kwargs
    ) -> Dict[str, Any]:
        """Call a specific provider"""
        config = self.provider_configs[provider_name]

        try:
            # Standardize request format
            if provider_name == "anthropic":
                payload = {
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": kwargs.get("max_tokens", 2000),
                    "model": config["model"],
                }
            else:
                payload = {
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": kwargs.get("max_tokens", 2000),
                    "model": config["model"],
                    "temperature": kwargs.get("temperature", 0.7),
                }

            response = requests.post(
                config["url"], headers=config["headers"], json=payload, timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                # Extract content from different response formats
                if provider_name == "anthropic":
                    content = data["content"][0]["text"]
                else:
                    content = data["choices"][0]["message"]["content"]

                return {"success": True, "content": content}
            else:
                return {
                    "success": False,
                    "content": "",
                    "error": f"HTTP {response.status_code}",
                }

        except Exception as e:
            return {"success": False, "content": "", "error": str(e)}


def create_simple_provider(config: Optional[Config] = None) -> SimpleUnifiedProvider:
    """Factory function"""
    return SimpleUnifiedProvider(config or Config())
