"""
Simplified Provider Configuration Validator
Handles PROVIDER_[NAME]=[BASE_URL],[API_KEY],[MODEL],[PROTOCOL] format
"""

import os
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from typing import Dict, List, Optional
from .adapters.base_adapter import BaseAdapter, ValidationResult
from .common import ProviderConfig

class SimpleProviderValidator:
    """Simple validator for the new provider format"""

    def __init__(self, env_file: str = ".env"):
        self.env_file = env_file
        self.providers = self._load_providers()

    def _load_providers(self) -> Dict[str, ProviderConfig]:
        """Load provider configurations from environment"""
        providers = {}

        # Provider definitions with priorities
        provider_configs = {
            "ollama": {"priority": 1, "is_local": True},
            "lm_studio": {"priority": 2, "is_local": True},
            "chutes": {"priority": 3, "is_local": False},
            "openrouter": {"priority": 4, "is_local": False},
            "openai": {"priority": 5, "is_local": False},
            "anthropic": {"priority": 6, "is_local": False},
            "google": {"priority": 7, "is_local": False},
            "groq": {"priority": 8, "is_local": False},
            "cohere": {"priority": 9, "is_local": False},
        }

        # Load from environment variables
        for key, value in os.environ.items():
            if key.startswith("PROVIDER_") and value:
                provider_name = key[9:].lower()  # Remove "PROVIDER_" prefix

                if provider_name in provider_configs:
                    try:
                        # Parse: BASE_URL,API_KEY,MODEL,PROTOCOL
                        parts = [part.strip() for part in value.split(",")]

                        if len(parts) >= 4:
                            base_url, api_key, model, protocol = parts[:4]

                            providers[provider_name] = ProviderConfig(
                                name=provider_name.replace("_", " ").title(),
                                base_url=base_url,
                                api_key=api_key,
                                model=model,
                                protocol=protocol,
                                priority=provider_configs[provider_name]["priority"],
                                is_local=provider_configs[provider_name]["is_local"],
                            )
                    except Exception as e:
                        console.print(f"[red]Error parsing {key}: {e}[/red]")

        return providers

    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """Validate current configuration"""
        if not self.providers:
            return False, [
                "No providers configured. Please configure at least one provider."
            ]

        errors = []

        for name, provider in self.providers.items():
            # Validate base URL
            if not provider.base_url:
                errors.append(f"{name}: Base URL is required")
            elif not (
                provider.base_url.startswith("http://")
                or provider.base_url.startswith("https://")
            ):
                errors.append(f"{name}: Base URL must start with http:// or https://")

            # Validate API key (not required for local services)
            if not provider.is_local and not provider.api_key:
                errors.append(f"{name}: API key is required for cloud services")

            # Validate model
            if not provider.model:
                errors.append(f"{name}: Model name is required")

            # Validate protocol
            valid_protocols = ["openai", "anthropic", "google", "cohere", "ollama"]
            if provider.protocol not in valid_protocols:
                errors.append(f"{name}: Protocol must be one of {valid_protocols}")

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
            console.print("2. Edit .env and uncomment ONE provider")
            console.print("3. Configure the provider with your details")
            return

        # Create table
        table = Table(title="Configured Providers")
        table.add_column("Provider", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Priority", style="yellow")
        table.add_column("Status", style="blue")
        table.add_column("Base URL", style="white")

        # Sort by priority
        sorted_providers = sorted(self.providers.items(), key=lambda x: x[1].priority)

        for name, provider in sorted_providers:
            provider_type = "Local" if provider.is_local else "Cloud"
            status = (
                "[green]Ready[/green]"
                if self._is_provider_ready(provider)
                else "[red]Incomplete[/red]"
            )

            # Truncate base URL for display
            display_url = (
                provider.base_url[:30] + "..."
                if len(provider.base_url) > 30
                else provider.base_url
            )

            table.add_row(
                provider.name,
                provider_type,
                str(provider.priority),
                status,
                display_url,
            )

        console.print(table)

        # Show primary provider
        primary = self.get_primary_provider()
        if primary:
            console.print(f"\n[green]Primary provider: {primary.name}[/green]")

        # Validate configuration
        is_valid, errors = self.validate_configuration()
        if is_valid:
            console.print("\n[green]✓ Configuration is valid and ready to use![/green]")
        else:
            console.print("\n[red]Configuration errors:[/red]")
            for error in errors:
                console.print(f"  • {error}")

    def _is_provider_ready(self, provider: ProviderConfig) -> bool:
        """Check if a provider is properly configured"""
        if not provider.base_url or not provider.model:
            return False

        if not provider.is_local and not provider.api_key:
            return False

        return True

    def show_provider_examples(self):
        """Show example configurations for all providers"""
        console.print("[bold blue]Provider Configuration Examples[/bold blue]")
        console.print("=" * 50)

        examples = {
            "Ollama (Local)": "PROVIDER_OLLAMA=http://localhost:11434,,llama2,ollama",
            "LM Studio (Local)": "PROVIDER_LM_STUDIO=http://localhost:1234/v1,,microsoft/DialoGPT-medium,openai",
            "Chutes.ai": "PROVIDER_CHUTES=https://api.chutes.ai/v1,your-chutes-key,chutes-gpt-3.5-turbo,openai",
            "OpenRouter": "PROVIDER_OPENROUTER=https://openrouter.ai/api/v1,your-openrouter-key,openai/gpt-3.5-turbo,openai",
            "OpenAI": "PROVIDER_OPENAI=https://api.openai.com/v1,your-openai-key,gpt-3.5-turbo,openai",
            "Anthropic": "PROVIDER_ANTHROPIC=https://api.anthropic.com,your-anthropic-key,claude-3-haiku-20240307,anthropic",
            "Google": "PROVIDER_GOOGLE=https://generativelanguage.googleapis.com,your-google-key,gemini-pro,google",
            "Groq": "PROVIDER_GROQ=https://api.groq.com/openai/v1,your-groq-key,llama3-8b-8192,openai",
            "Cohere": "PROVIDER_COHERE=https://api.cohere.ai/v1,your-cohere-key,command,cohere",
        }

        for name, example in examples.items():
            console.print(f"\n[cyan]{name}:[/cyan]")
            console.print(f"  {example}")

        console.print(
            f"\n[yellow]Format:[/yellow] PROVIDER_[NAME]=[BASE_URL],[API_KEY],[MODEL],[PROTOCOL]"
        )
        console.print(f"[yellow]Note:[/yellow] Empty API_KEY for local services")

def main():
    """Test the validator"""
    validator = SimpleProviderValidator()

    console.print("[bold green]Simple Provider Configuration Test[/bold green]")
    console.print("=" * 50)

    validator.show_configuration_status()

    console.print("\n")
    validator.show_provider_examples()

if __name__ == "__main__":
    main()
