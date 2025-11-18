"""
Unified Provider Configuration Validator
Handles format: PROVIDER_API_URL, PROVIDER_API_KEY, PROVIDER_MODEL
"""

import os
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table

console = Console()

@dataclass
class ProviderConfig:
    """Configuration for a single provider"""
    name: str
    api_url: str
    api_key: str
    model: str
    priority: int
    is_local: bool = False

class UnifiedProviderValidator:
    """Validator for unified PROVIDER_* format"""
    
    def __init__(self, env_file: str = ".env"):
        self.env_file = env_file
        self.provider = self._load_provider()
    
    def _load_provider(self) -> Optional[ProviderConfig]:
        """Load the active provider configuration"""
        # Get the unified provider variables
        api_url = os.getenv("PROVIDER_API_URL", "")
        api_key = os.getenv("PROVIDER_API_KEY", "")
        model = os.getenv("PROVIDER_MODEL", "")
        
        if not api_url:
            return None
        
        # Detect provider type and priority based on URL
        provider_info = self._detect_provider(api_url)
        
        return ProviderConfig(
            name=provider_info["name"],
            api_url=api_url,
            api_key=api_key,
            model=model,
            priority=provider_info["priority"],
            is_local=provider_info["is_local"]
        )
    
    def _detect_provider(self, api_url: str) -> Dict[str, any]:
        """Detect provider type and priority from URL"""
        url_lower = api_url.lower()
        
        # Provider detection patterns
        providers = {
            "ollama": {
                "pattern": "localhost:11434|ollama",
                "name": "Ollama",
                "priority": 1,
                "is_local": True
            },
            "lm_studio": {
                "pattern": "localhost:1234|lmstudio",
                "name": "LM Studio", 
                "priority": 2,
                "is_local": True
            },
            "chutes": {
                "pattern": "chutes.ai|llm.chutes.ai",
                "name": "Chutes.ai",
                "priority": 3,
                "is_local": False
            },
            "openrouter": {
                "pattern": "openrouter.ai",
                "name": "OpenRouter",
                "priority": 4,
                "is_local": False
            },
            "groq": {
                "pattern": "api.groq.com|groq",
                "name": "Groq",
                "priority": 5,
                "is_local": False
            },
            "openai": {
                "pattern": "api.openai.com|openai",
                "name": "OpenAI",
                "priority": 6,
                "is_local": False
            },
            "anthropic": {
                "pattern": "api.anthropic.com|anthropic",
                "name": "Anthropic",
                "priority": 7,
                "is_local": False
            },
            "google": {
                "pattern": "generativelanguage.googleapis.com|google",
                "name": "Google",
                "priority": 8,
                "is_local": False
            },
            "cohere": {
                "pattern": "api.cohere.ai|cohere",
                "name": "Cohere",
                "priority": 9,
                "is_local": False
            }
        }
        
        # Find matching provider
        for provider_key, info in providers.items():
            if re.search(info["pattern"], url_lower):
                return info
        
        # Default fallback
        return {
            "name": "Unknown",
            "priority": 99,
            "is_local": False
        }
    
    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """Validate current configuration"""
        if not self.provider:
            return False, ["No provider configured. Please uncomment and configure PROVIDER_* variables."]
        
        errors = []
        
        # Validate API URL
        if not self.provider.api_url:
            errors.append("PROVIDER_API_URL is required")
        elif not (self.provider.api_url.startswith('http://') or self.provider.api_url.startswith('https://')):
            errors.append("PROVIDER_API_URL must start with http:// or https://")
        
        # Validate API key (not required for local services)
        if not self.provider.is_local and not self.provider.api_key:
            errors.append("PROVIDER_API_KEY is required for cloud services")
        
        # Validate model
        if not self.provider.model:
            errors.append("PROVIDER_MODEL is required")
        
        return len(errors) == 0, errors
    
    def get_provider_info(self) -> Dict[str, any]:
        """Get information about the current provider"""
        if not self.provider:
            return {"configured": False}
        
        return {
            "configured": True,
            "name": self.provider.name,
            "api_url": self.provider.api_url,
            "api_key_configured": bool(self.provider.api_key),
            "model": self.provider.model,
            "priority": self.provider.priority,
            "is_local": self.provider.is_local,
            "ready": self._is_provider_ready()
        }
    
    def _is_provider_ready(self) -> bool:
        """Check if the provider is properly configured"""
        if not self.provider:
            return False
        
        if not self.provider.api_url or not self.provider.model:
            return False
        
        if not self.provider.is_local and not self.provider.api_key:
            return False
        
        return True
    
    def show_configuration_status(self):
        """Display current configuration status"""
        console.print("[bold blue]Unified Provider Configuration Status[/bold blue]")
        console.print("=" * 60)
        
        if not self.provider:
            console.print("[red]No provider configured[/red]")
            console.print("\n[yellow]Quick setup:[/yellow]")
            console.print("1. Copy template: copy .env.example .env")
            console.print("2. Edit .env and uncomment ONE provider section")
            console.print("3. Configure PROVIDER_API_URL, PROVIDER_API_KEY, PROVIDER_MODEL")
            console.print("4. Save this file")
            return
        
        # Create table
        table = Table(title="Active Provider Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")
        table.add_column("Status", style="green")
        
        # Add provider info
        table.add_row("Provider", self.provider.name, "[green]Active[/green]")
        table.add_row("Type", "Local" if self.provider.is_local else "Cloud", "[blue]Info[/blue]")
        table.add_row("Priority", str(self.provider.priority), "[yellow]Info[/yellow]")
        table.add_row("API URL", self.provider.api_url, "[green]Configured[/green]")
        
        key_status = "[green]Configured[/green]" if self.provider.api_key else "[yellow]Not needed[/yellow]" if self.provider.is_local else "[red]Missing[/red]"
        table.add_row("API Key", "***" if self.provider.api_key else "Empty", key_status)
        
        table.add_row("Model", self.provider.model, "[green]Configured[/green]")
        
        console.print(table)
        
        # Validate configuration
        is_valid, errors = self.validate_configuration()
        if is_valid:
            console.print(f"\n[green]Configuration is valid and ready to use![/green]")
            console.print(f"[green]Provider: {self.provider.name} | Model: {self.provider.model}[/green]")
        else:
            console.print(f"\n[red]Configuration errors:[/red]")
            for error in errors:
                console.print(f"  • {error}")
    
    def show_available_providers(self):
        """Show all available provider options"""
        console.print("[bold blue]Available Provider Options[/bold blue]")
        console.print("=" * 40)
        
        providers = [
            {
                "name": "Ollama (Local - Priority 1)",
                "url": "http://localhost:11434",
                "key_needed": False,
                "example_models": ["llama2", "mistral", "codellama"]
            },
            {
                "name": "LM Studio (Local - Priority 2)",
                "url": "http://localhost:1234/v1", 
                "key_needed": False,
                "example_models": ["microsoft/DialoGPT-medium", "microsoft/DialoGPT-large"]
            },
            {
                "name": "Chutes.ai (Priority 3)",
                "url": "https://llm.chutes.ai/v1",
                "key_needed": True,
                "example_models": ["openai/gpt-oss-20b", "zai-org/GLM-4.5-Air", "zai-org/GLM-4.6-FP8"]
            },
            {
                "name": "OpenRouter (Priority 4)",
                "url": "https://openrouter.ai/api/v1",
                "key_needed": True,
                "example_models": ["openai/gpt-3.5-turbo", "openai/gpt-4", "anthropic/claude-3-haiku"]
            },
            {
                "name": "Groq (Priority 5)",
                "url": "https://api.groq.com/openai/v1",
                "key_needed": True,
                "example_models": ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"]
            },
            {
                "name": "OpenAI (Priority 6)",
                "url": "https://api.openai.com/v1",
                "key_needed": True,
                "example_models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
            }
        ]
        
        for provider in providers:
            console.print(f"\n[cyan]{provider['name']}[/cyan]")
            console.print(f"  URL: {provider['url']}")
            console.print(f"  API Key: {'Required' if provider['key_needed'] else 'Not needed'}")
            console.print(f"  Example Models: {', '.join(provider['example_models'][:2])}{'...' if len(provider['example_models']) > 2 else ''}")

def main():
    """Test the validator"""
    validator = UnifiedProviderValidator()
    
    console.print("[bold green]Unified Provider Configuration Test[/bold green]")
    console.print("=" * 50)
    
    validator.show_configuration_status()
    console.print("\n")
    validator.show_available_providers()

if __name__ == "__main__":
    main()