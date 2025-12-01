"""
LLM Provider with Failover Support
Simplified provider system that supports multiple LLM providers with automatic failover
"""

import json
import os
import asyncio
import time
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from rich.console import Console
from rich.progress import Progress, TextColumn

console = Console()
error_console = Console(stderr=True)


class ProviderStatus(Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"


@dataclass
class LLMProvider:
    """LLM Provider configuration"""

    url: str
    api_key: str
    model: str
    name: str = ""
    status: ProviderStatus = ProviderStatus.AVAILABLE
    last_error: Optional[str] = None
    last_check: float = 0
    retry_after: float = 0

    def __post_init__(self):
        if not self.name:
            # Generate name from URL
            if "openai.com" in self.url:
                self.name = "openai"
            elif "anthropic.com" in self.url:
                self.name = "anthropic"
            elif "openrouter.ai" in self.url:
                self.name = "openrouter"
            elif "localhost:11434" in self.url:
                self.name = "ollama"
            elif "localhost:1234" in self.url:
                self.name = "lm_studio"
            else:
                self.name = "custom"


class LLMProviderManager:
    """Manages multiple LLM providers with failover support"""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.providers: List[LLMProvider] = []
        self.current_provider_index = 0
        self.config_path = config_path or self._get_default_config_path()
        self._load_providers()

    def _get_default_config_path(self) -> Path:
        """Get default config path (workspace or home)"""
        # Check for workspace config first
        workspace_config = Path.cwd() / ".conjecture" / "config.json"
        if workspace_config.exists():
            return workspace_config

        # Fall back to home config
        home_config = Path.home() / ".conjecture" / "config.json"
        return home_config

    def _load_providers(self):
        """Load providers from config file"""
        try:
            if not self.config_path.exists():
                console.print(
                    f"[yellow]Config file not found: {self.config_path}[/yellow]"
                )
                console.print("[yellow]Using default provider configuration[/yellow]")
                self._create_default_config()
                return

            with open(self.config_path, "r") as f:
                config = json.load(f)

            providers_config = config.get("providers", [])
            if not providers_config:
                console.print("[yellow]No providers configured[/yellow]")
                return

            self.providers = []
            for provider_config in providers_config:
                provider = LLMProvider(
                    url=provider_config.get("url", ""),
                    api_key=provider_config.get("api", ""),
                    model=provider_config.get("model", ""),
                    name=provider_config.get("name", ""),
                )
                self.providers.append(provider)

            console.print(
                f"[green]Loaded {len(self.providers)} providers from {self.config_path}[/green]"
            )

        except Exception as e:
            error_console.print(f"[red]Error loading providers: {e}[/red]")
            self.providers = []

    def _create_default_config(self):
        """Create a default config file"""
        default_config = {
            "providers": [
                {"url": "http://localhost:11434", "api": "", "model": "llama2"}
            ]
        }

        # Create config directory if it doesn't exist
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.config_path, "w") as f:
            json.dump(default_config, f, indent=2)

        console.print(f"[green]Created default config at: {self.config_path}[/green]")
        console.print(
            "[yellow]Please edit the config file to add your LLM providers[/yellow]"
        )

    def get_available_provider(self) -> Optional[LLMProvider]:
        """Get the next available provider"""
        if not self.providers:
            return None

        # Check current provider first
        current_provider = self.providers[self.current_provider_index]
        if self._is_provider_available(current_provider):
            return current_provider

        # Try other providers
        for i, provider in enumerate(self.providers):
            if i != self.current_provider_index and self._is_provider_available(
                provider
            ):
                self.current_provider_index = i
                console.print(f"[yellow]Switched to provider: {provider.name}[/yellow]")
                return provider

        return None

    def _is_provider_available(self, provider: LLMProvider) -> bool:
        """Check if provider is available (basic health check)"""
        current_time = time.time()

        # Respect retry_after for rate limited providers
        if current_time < provider.retry_after:
            return False

        # Simple availability check - in real implementation, this would
        # make a lightweight API call to verify the provider is reachable
        if provider.url.startswith("http://localhost") or provider.url.startswith(
            "https://localhost"
        ):
            # For local providers, just check if the URL seems reasonable
            return bool(provider.url and provider.model)
        else:
            # For cloud providers, require API key
            return bool(provider.url and provider.api_key and provider.model)

    def mark_provider_error(self, provider: LLMProvider, error: str):
        """Mark a provider as having an error"""
        provider.status = ProviderStatus.ERROR
        provider.last_error = error
        provider.last_check = time.time()

        # If it's a rate limit error, set retry_after
        if "rate limit" in error.lower() or "429" in error:
            provider.status = ProviderStatus.RATE_LIMITED
            provider.retry_after = time.time() + 60  # Retry after 1 minute

        error_console.print(f"[red]Provider {provider.name} error: {error}[/red]")

    def mark_provider_success(self, provider: LLMProvider):
        """Mark a provider as successful"""
        provider.status = ProviderStatus.AVAILABLE
        provider.last_error = None
        provider.last_check = time.time()

    def get_providers(self) -> List[Dict[str, Any]]:
        """Get list of all providers"""
        return [
            {
                "name": p.name,
                "url": p.url,
                "model": p.model,
                "api": p.api_key[:8] + "..." if p.api_key else "",
                "status": p.status.value,
                "last_error": p.last_error,
            }
            for p in self.providers
        ]

    def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all providers"""
        return {
            "total_providers": len(self.providers),
            "current_provider": self.providers[self.current_provider_index].name
            if self.providers
            else None,
            "providers": self.get_providers(),
        }

    async def process_with_failover(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Process prompt with automatic failover"""
        provider = self.get_available_provider()
        if not provider:
            raise Exception("No available LLM providers")

        max_retries = len(self.providers)
        for attempt in range(max_retries):
            try:
                # In a real implementation, this would call the actual LLM API
                result = await self._call_provider(provider, prompt, **kwargs)
                self.mark_provider_success(provider)
                return result

            except Exception as e:
                self.mark_provider_error(provider, str(e))

                # Try next provider
                provider = self.get_available_provider()
                if not provider:
                    break

        raise Exception("All providers failed")

    async def _call_provider(
        self, provider: LLMProvider, prompt: str, **kwargs
    ) -> Dict[str, Any]:
        """Call a specific provider (mock implementation)"""
        # Mock implementation - in real code, this would make actual API calls
        await asyncio.sleep(0.1)  # Simulate API call

        return {
            "provider": provider.name,
            "model": provider.model,
            "response": f"Mock response from {provider.name} for: {prompt[:50]}...",
            "tokens_used": 100,
            "status": "success",
        }


# Global instance for easy access
_provider_manager = None


def get_provider_manager() -> LLMProviderManager:
    """Get the global provider manager instance"""
    global _provider_manager
    if _provider_manager is None:
        _provider_manager = LLMProviderManager()
    return _provider_manager


def get_provider_status() -> Dict[str, Any]:
    """Get status of all providers (convenience function)"""
    return get_provider_manager().get_provider_status()


async def process_prompt_with_failover(prompt: str, **kwargs) -> Dict[str, Any]:
    """Process prompt with automatic failover (convenience function)"""
    return await get_provider_manager().process_with_failover(prompt, **kwargs)
