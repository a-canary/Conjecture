"""
Individual Environment Variable Provider Validator
Handles format: PROVIDER_API_URL, PROVIDER_API_KEY, PROVIDER_MODELS
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from typing import Dict, List, Optional
from .adapters.base_adapter import BaseAdapter, ValidationResult
from .common import ProviderConfig

class IndividualEnvValidator:
    """Validator for individual environment variable format"""

    def __init__(self, env_file: str = ".env"):
        self.env_file = env_file
        self.providers = self._load_providers()

    def _load_providers(self) -> Dict[str, ProviderConfig]:
        """Load provider configurations from environment variables"""
        providers = {}

        # Provider definitions with priorities
        provider_configs = {
            "ollama": {"priority": 1, "is_local": True},
            "lm_studio": {"priority": 2, "is_local": True},
            "chutes": {"priority": 3, "is_local": False},
            "openrouter": {"priority": 4, "is_local": False},
            "groq": {"priority": 5, "is_local": False},
            "openai": {"priority": 6, "is_local": False},
            "anthropic": {"priority": 7, "is_local": False},
            "google": {"priority": 8, "is_local": False},
            "cohere": {"priority": 9, "is_local": False},
        }

        for provider_name, config in provider_configs.items():
            # Get environment variables for this provider
            url_var = f"{provider_name.upper()}_API_URL"
            key_var = f"{provider_name.upper()}_API_KEY"
            models_var = f"{provider_name.upper()}_MODELS"

            api_url = os.getenv(url_var, "")
            api_key = os.getenv(key_var, "")
            models_str = os.getenv(models_var, "[]")

            # Only include provider if URL is configured
            if api_url:
                try:
                    # Parse models JSON array
                    if models_str.startswith("[") and models_str.endswith("]"):
                        try:
                            models = json.loads(models_str)
                        except json.JSONDecodeError:
                            # Fallback: parse manually quoted array
                            models_str = models_str.strip("[]")
                            models = [
                                m.strip().strip('"').strip("'")
                                for m in models_str.split(",")
                                if m.strip()
                            ]
                    else:
                        models = [models_str] if models_str else []

                    providers[provider_name] = ProviderConfig(
                        name=provider_name.replace("_", " ").title(),
                        api_url=api_url,
                        api_key=api_key,
                        models=models,
                        priority=config["priority"],
                        is_local=config["is_local"],
                    )
                except json.JSONDecodeError:
                    console.print(
                        f"[red]Error parsing models for {provider_name}: {models_str}[/red]"
                    )
                    # Still include provider with empty models
                    providers[provider_name] = ProviderConfig(
                        name=provider_name.replace("_", " ").title(),
                        api_url=api_url,
                        api_key=api_key,
                        models=[],
                        priority=config["priority"],
                        is_local=config["is_local"],
                    )

        return providers

    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """Validate current configuration"""
        if not self.providers:
            return False, [
                "No providers configured. Please configure at least one provider."
            ]

        errors = []

        for name, provider in self.providers.items():
            # Validate API URL
            if not provider.api_url:
                errors.append(f"{name}: API URL is required")
            elif not (
                provider.api_url.startswith("http://")
                or provider.api_url.startswith("https://")
            ):
                errors.append(f"{name}: API URL must start with http:// or https://")

            # Validate API key (not required for local services)
            if not provider.is_local and not provider.api_key:
                errors.append(f"{name}: API key is required for cloud services")

            # Validate models
            if not provider.models:
                errors.append(f"{name}: At least one model is required")

        return len(errors) == 0, errors

    def get_primary_provider(self) -> Optional[ProviderConfig]:
        """Get the primary provider (lowest priority number)"""
        if not self.providers:
            return None

        return min(self.providers.values(), key=lambda p: p.priority)

    def show_configuration_status(self):
        """Display current configuration status"""
        console.print("[bold blue]Provider Configuration Status[/bold blue]")
        console.print("=" * 60)

        if not self.providers:
            console.print("[red]No providers configured[/red]")
            console.print("\n[yellow]Quick setup:[/yellow]")
            console.print("1. Copy template: copy .env.example .env")
            console.print("2. Edit .env and configure your preferred provider")
            console.print("3. Replace placeholder API keys with actual keys")
            return

        # Create table
        table = Table(title="Configured Providers")
        table.add_column("Provider", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Priority", style="yellow")
        table.add_column("Status", style="blue")
        table.add_column("Models", style="white")
        table.add_column("API URL", style="white")

        # Sort by priority
        sorted_providers = sorted(self.providers.items(), key=lambda x: x[1].priority)

        for name, provider in sorted_providers:
            provider_type = "Local" if provider.is_local else "Cloud"
            status = (
                "[green]Ready[/green]"
                if self._is_provider_ready(provider)
                else "[red]Incomplete[/red]"
            )

            # Truncate for display
            models_display = f"{len(provider.models)} models"
            url_display = (
                provider.api_url[:25] + "..."
                if len(provider.api_url) > 25
                else provider.api_url
            )

            table.add_row(
                provider.name,
                provider_type,
                str(provider.priority),
                status,
                models_display,
                url_display,
            )

        console.print(table)

        # Show primary provider
        primary = self.get_primary_provider()
        if primary:
            console.print(f"\n[green]Primary provider: {primary.name}[/green]")
            console.print(
                f"Models: {', '.join(primary.models[:3])}{'...' if len(primary.models) > 3 else ''}"
            )

        # Validate configuration
        is_valid, errors = self.validate_configuration()
        if is_valid:
            console.print("\n[green]Configuration is valid and ready to use![/green]")
        else:
            console.print("\n[red]Configuration errors:[/red]")
            for error in errors:
                console.print(f"  • {error}")

    def _is_provider_ready(self, provider: ProviderConfig) -> bool:
        """Check if a provider is properly configured"""
        if not provider.api_url or not provider.models:
            return False

        if not provider.is_local and not provider.api_key:
            return False

        return True

    def show_provider_details(self):
        """Show detailed configuration for all providers"""
        console.print("[bold blue]Detailed Provider Configuration[/bold blue]")
        console.print("=" * 50)

        if not self.providers:
            console.print("[red]No providers configured[/red]")
            return

        for name, provider in self.providers.items():
            console.print(
                f"\n[cyan]{provider.name}[/cyan] (Priority: {provider.priority})"
            )
            console.print(f"  Type: {'Local' if provider.is_local else 'Cloud'}")
            console.print(f"  API URL: {provider.api_url}")
            console.print(
                f"  API Key: {'[green]Configured[/green]' if provider.api_key else '[yellow]Not needed (local)[/yellow]' if provider.is_local else '[red]Missing[/red]'}"
            )
            console.print(f"  Models: {', '.join(provider.models)}")
            console.print(
                f"  Status: {'[green]Ready[/green]' if self._is_provider_ready(provider) else '[red]Incomplete[/red]'}"
            )

def main():
    """Test the validator"""
    validator = IndividualEnvValidator()

    console.print(
        "[bold green]Individual Environment Variable Configuration Test[/bold green]"
    )
    console.print("=" * 60)

    validator.show_configuration_status()
    console.print("\n")
    validator.show_provider_details()

if __name__ == "__main__":
    main()
