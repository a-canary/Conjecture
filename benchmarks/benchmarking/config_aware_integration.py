"""
Config-Aware Model Integration
Reads directly from Conjecture config to get API keys and providers
"""

import aiohttp
import asyncio
import json
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigAwareIntegration:
    """Integration that reads from Conjecture config"""

    def __init__(self, config_path: str = None):
        self.config_path = config_path or Path(__file__).parent.parent.parent / ".conjecture" / "config.json"
        self.providers = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.load_config()

    def load_config(self):
        """Load providers from Conjecture config"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)

            for provider in config.get('providers', []):
                provider_name = provider.get('name')
                if provider_name:
                    # Handle both 'api' and 'key' field names
                    api_key = provider.get('api') or provider.get('key', '')
                    self.providers[provider_name] = {
                        'url': provider.get('url'),
                        'api_key': api_key,
                        'model': provider.get('model'),
                        'is_local': provider.get('is_local', False),
                        'max_tokens': provider.get('max_tokens', 4000),
                        'temperature': provider.get('temperature', 0.1),
                        'timeout': provider.get('timeout', 30)
                    }

        except Exception as e:
            print(f"Failed to load config: {e}")

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_response(self, provider_name: str, prompt: str, **kwargs) -> str:
        """Get response from specified provider"""
        if provider_name not in self.providers:
            raise Exception(f"Provider {provider_name} not found in config")

        provider = self.providers[provider_name]

        if not self.session:
            self.session = aiohttp.ClientSession()

        # For local providers, use direct LM Studio connection
        if provider['is_local']:
            return await self._get_local_response(provider, prompt, **kwargs)
        else:
            return await self._get_api_response(provider, prompt, **kwargs)

    async def _get_local_response(self, provider: dict, prompt: str, **kwargs) -> str:
        """Get response from local provider (LM Studio)"""
        from .lm_studio_integration import LMStudioIntegration

        async with LMStudioIntegration(
            base_url=provider['url'],
            model=provider['model']
        ) as lm_studio:
            return await lm_studio.get_response(prompt, **kwargs)

    async def _get_api_response(self, provider: dict, prompt: str, **kwargs) -> str:
        """Get response from API provider"""
        url = f"{provider['url']}/chat/completions"

        payload = {
            "model": provider['model'],
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert AI assistant. Provide accurate, well-reasoned answers."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": kwargs.get('max_tokens', provider['max_tokens']),
            "temperature": kwargs.get('temperature', provider['temperature']),
            "top_p": kwargs.get('top_p', 0.9)
        }

        headers = {
            "Content-Type": "application/json"
        }

        # Add API key if available
        if provider['api_key']:
            headers["Authorization"] = f"Bearer {provider['api_key']}"

        timeout = aiohttp.ClientTimeout(total=provider.get('timeout', 60))

        try:
            async with self.session.post(url, json=payload, headers=headers, timeout=timeout) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    raise Exception(f"API error {response.status}: {error_text}")
        except Exception as e:
            raise Exception(f"Failed to get response: {str(e)}")

    def list_providers(self):
        """List available providers"""
        print("Available Providers:")
        for name, provider in self.providers.items():
            api_status = "API Key Set" if provider['api_key'] else "No API Key"
            local_status = "Local" if provider['is_local'] else "Remote"
            print(f"  {name}: {provider['model']} ({local_status}, {api_status})")

# Convenience functions for benchmarking
async def gpt_oss_20b_direct(prompt: str) -> str:
    """GPT-OSS-20B via config"""
    async with ConfigAwareIntegration() as integration:
        return await integration.get_response("gpt-oss-20b", prompt)

async def granite_tiny_direct(prompt: str) -> str:
    """GraniteTiny via config"""
    async with ConfigAwareIntegration() as integration:
        return await integration.get_response("GraniteTiny", prompt)

async def glm_46_direct(prompt: str) -> str:
    """GLM-4.6 via config"""
    async with ConfigAwareIntegration() as integration:
        return await integration.get_response("glm-4.6", prompt)

# Test function
async def test_providers():
    """Test all configured providers"""
    integration = ConfigAwareIntegration()
    integration.list_providers()

    print("\nTesting providers...")

    # Test GraniteTiny (should work - local)
    try:
        response = await integration.get_response("GraniteTiny", "What is 2+2?", max_tokens=50)
        print(f"GraniteTiny: SUCCESS - {response[:50]}...")
    except Exception as e:
        print(f"GraniteTiny: FAILED - {e}")

    # Test GPT-OSS-20B (should work with API key)
    try:
        response = await integration.get_response("gpt-oss-20b", "What is 2+2?", max_tokens=50)
        print(f"GPT-OSS-20B: SUCCESS - {response[:50]}...")
    except Exception as e:
        print(f"GPT-OSS-20B: FAILED - {e}")

if __name__ == "__main__":
    asyncio.run(test_providers())