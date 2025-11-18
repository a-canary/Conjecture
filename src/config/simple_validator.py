"""
Simple Configuration Validator for Conjecture
Replaces complex discovery with clean, documented configuration approach
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Rich console for beautiful output
console = Console()
error_console = Console(stderr=True)


@dataclass
class ProviderConfig:
    """Configuration for a single provider"""
    name: str
    type: str  # 'local' or 'cloud'
    priority: int  # Lower number = higher priority
    required_vars: List[str]
    optional_vars: List[str] = None
    setup_instructions: str = ""
    example_config: str = ""

    def __post_init__(self):
        if self.optional_vars is None:
            self.optional_vars = []


@dataclass
class ValidationResult:
    """Result of configuration validation"""
    success: bool
    primary_provider: Optional[str] = None
    available_providers: List[ProviderConfig] = None
    missing_vars: Dict[str, List[str]] = None
    errors: List[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.available_providers is None:
            self.available_providers = []
        if self.missing_vars is None:
            self.missing_vars = {}
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class SimpleValidator:
    """
    Simple configuration validator with smart defaults and clear messaging
    Focuses on user-friendly configuration rather than complex discovery
    """

    def __init__(self):
        self.env_file = Path(".env")
        self.env_example = Path(".env.example")
        self.project_root = Path.cwd()
        
        # Define providers in priority order
        self.providers = {
            "ollama": ProviderConfig(
                name="Ollama",
                type="local",
                priority=1,
                required_vars=["OLLAMA_ENDPOINT"],
                optional_vars=["OLLAMA_MODEL"],
                setup_instructions="""
1. Install Ollama: https://ollama.ai/
2. Start service: ollama serve
3. Pull a model: ollama pull llama2
4. Configure in .env: OLLAMA_ENDPOINT=http://localhost:11434
                """.strip(),
                example_config="""
OLLAMA_ENDPOINT=http://localhost:11434
OLLAMA_MODEL=llama2
                """.strip()
            ),
            "lm_studio": ProviderConfig(
                name="LM Studio",
                type="local", 
                priority=2,
                required_vars=["LM_STUDIO_ENDPOINT"],
                optional_vars=["LM_STUDIO_MODEL"],
                setup_instructions="""
1. Install LM Studio: https://lmstudio.ai/
2. Launch LM Studio application
3. Start server from "Server" tab
4. Configure in .env: LM_STUDIO_ENDPOINT=http://localhost:1234/v1
                """.strip(),
                example_config="""
LM_STUDIO_ENDPOINT=http://localhost:1234/v1
LM_STUDIO_MODEL=microsoft/DialoGPT-medium
                """.strip()
            ),
            "openai": ProviderConfig(
                name="OpenAI",
                type="cloud",
                priority=3,
                required_vars=["OPENAI_API_KEY"],
                optional_vars=["OPENAI_MODEL"],
                setup_instructions="""
1. Get API key: https://platform.openai.com/api-keys
2. Add credits to your account
3. Configure in .env: OPENAI_API_KEY=sk-your-key-here
                """.strip(),
                example_config="""
OPENAI_API_KEY=sk-1234567890abcdef1234567890abcdef12345678
OPENAI_MODEL=gpt-3.5-turbo
                """.strip()
            ),
            "anthropic": ProviderConfig(
                name="Anthropic Claude",
                type="cloud",
                priority=4,
                required_vars=["ANTHROPIC_API_KEY"],
                optional_vars=["ANTHROPIC_MODEL"],
                setup_instructions="""
1. Get API key: https://console.anthropic.com/
2. Add credits to your account
3. Configure in .env: ANTHROPIC_API_KEY=sk-ant-your-key-here
                """.strip(),
                example_config="""
ANTHROPIC_API_KEY=sk-ant-1234567890abcdef1234567890abcdef12345678
ANTHROPIC_MODEL=claude-3-haiku-20240307
                """.strip()
            ),
            "google": ProviderConfig(
                name="Google Gemini",
                type="cloud",
                priority=5,
                required_vars=["GOOGLE_API_KEY"],
                optional_vars=["GOOGLE_MODEL"],
                setup_instructions="""
1. Get API key: https://makersuite.google.com/app/apikey
2. Enable Gemini API in your Google Cloud project
3. Configure in .env: GOOGLE_API_KEY=your-key-here
                """.strip(),
                example_config="""
GOOGLE_API_KEY=AIza1234567890abcdef1234567890abcdef12345678
GOOGLE_MODEL=gemini-pro
                """.strip()
            ),
            "cohere": ProviderConfig(
                name="Cohere",
                type="cloud",
                priority=6,
                required_vars=["COHERE_API_KEY"],
                optional_vars=["COHERE_MODEL"],
                setup_instructions="""
1. Get API key: https://dashboard.cohere.ai/api-keys
2. Add credits to your account
3. Configure in .env: COHERE_API_KEY=your-key-here
                """.strip(),
                example_config="""
COHERE_API_KEY=1234567890abcdef1234567890abcdef12345678
COHERE_MODEL=command
                """.strip()
            ),
            "chutes": ProviderConfig(
                name="Chutes.ai",
                type="cloud",
                priority=3,
                required_vars=["CHUTES_API_KEY"],
                optional_vars=["CHUTES_BASE_URL", "CHUTES_MODEL"],
                setup_instructions="""
1. Get API key: https://chutes.ai/
2. Sign up for optimized AI service
3. Configure in .env: CHUTES_API_KEY=your-key-here
4. Optional: Custom endpoint URL
                """.strip(),
                example_config="""
CHUTES_API_KEY=sk-chutes-your-key-here
CHUTES_BASE_URL=https://api.chutes.ai/v1
CHUTES_MODEL=chutes-gpt-3.5-turbo
                """.strip()
            ),
            "openrouter": ProviderConfig(
                name="OpenRouter",
                type="cloud",
                priority=4,
                required_vars=["OPENROUTER_API_KEY"],
                optional_vars=["OPENROUTER_BASE_URL", "OPENROUTER_MODEL"],
                setup_instructions="""
1. Get API key: https://openrouter.ai/keys
2. Access 100+ models from various providers
3. Configure in .env: OPENROUTER_API_KEY=your-key-here
4. Optional: Custom endpoint and model selection
                """.strip(),
                example_config="""
OPENROUTER_API_KEY=sk-or-your-key-here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=openai/gpt-3.5-turbo
                """.strip()
            )
        }

    def load_env_file(self) -> bool:
        """Load environment variables from .env file"""
        try:
            from dotenv import load_dotenv
            load_dotenv()
            return True
        except ImportError:
            # dotenv not available, try manual parsing
            if self.env_file.exists():
                try:
                    with open(self.env_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#') and '=' in line:
                                key, value = line.split('=', 1)
                                os.environ[key.strip()] = value.strip()
                    return True
                except Exception:
                    pass
            return False

    def validate_provider(self, provider_name: str) -> Tuple[bool, List[str]]:
        """
        Validate a specific provider configuration
        Returns: (is_configured, missing_variables)
        """
        if provider_name not in self.providers:
            return False, [f"Unknown provider: {provider_name}"]

        provider = self.providers[provider_name]
        missing_vars = []

        # Check required variables
        for var in provider.required_vars:
            value = os.getenv(var)
            if not value or value.strip() == "":
                missing_vars.append(var)

        # For local providers, also validate the endpoint is reachable
        if provider.type == "local" and not missing_vars:
            endpoint_var = provider.required_vars[0]  # First required var is usually the endpoint
            endpoint = os.getenv(endpoint_var)
            if endpoint:
                if not self._is_endpoint_valid(endpoint):
                    missing_vars.append(f"{endpoint_var} (endpoint not reachable)")

        is_configured = len(missing_vars) == 0
        return is_configured, missing_vars

    def _is_endpoint_valid(self, endpoint: str) -> bool:
        """Basic validation of endpoint URL format"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(endpoint)
            return parsed.scheme in ['http', 'https'] and parsed.netloc != ""
        except Exception:
            return False

    def validate_configuration(self) -> ValidationResult:
        """
        Validate the complete configuration
        Returns detailed validation result
        """
        result = ValidationResult(success=False)
        
        # Load environment variables
        env_loaded = self.load_env_file()
        if not env_loaded and not self.env_file.exists():
            result.errors.append("No .env file found")
            result.warnings.append("Copy .env.example to .env and configure at least one provider")
        
        # Check each provider
        available_providers = []
        all_missing_vars = {}
        
        for provider_name, provider in self.providers.items():
            is_configured, missing_vars = self.validate_provider(provider_name)
            
            if is_configured:
                available_providers.append(provider)
            else:
                if missing_vars:
                    all_missing_vars[provider_name] = missing_vars
        
        result.available_providers = available_providers
        result.missing_vars = all_missing_vars
        
        # Determine success and primary provider
        if available_providers:
            result.success = True
            # Sort by priority (lower number = higher priority)
            available_providers.sort(key=lambda p: p.priority)
            result.primary_provider = available_providers[0].name
            
            # Add warnings for partially configured providers
            for provider_name, missing_vars in all_missing_vars.items():
                if len(missing_vars) < len(self.providers[provider_name].required_vars):
                    result.warnings.append(f"{provider_name} partially configured: missing {', '.join(missing_vars)}")
        else:
            result.success = False
            result.errors.append("No providers configured. Please configure at least one provider.")
        
        # Add security warnings
        if self._check_env_in_gitignore():
            result.warnings.append(".env file is protected by .gitignore ✓")
        else:
            result.errors.append(".env file is NOT protected by .gitignore! Add .env to .gitignore immediately.")
        
        return result

    def _check_env_in_gitignore(self) -> bool:
        """Check if .env is protected by .gitignore"""
        gitignore_path = Path(".gitignore")
        if not gitignore_path.exists():
            return False
        
        try:
            with open(gitignore_path, 'r') as f:
                content = f.read()
                return ".env" in content.splitlines()
        except Exception:
            return False

    def print_validation_result(self, result: ValidationResult):
        """Print validation result in a user-friendly format"""
        if result.success:
            console.print("[bold green]✅ Configuration Validation: PASSED[/bold green]")
            
            if result.primary_provider:
                primary = next((p for p in result.available_providers if p.name == result.primary_provider), None)
                if primary:
                    panel = Panel(
                        f"[bold]Primary Provider:[/bold] {primary.name}\n"
                        f"[bold]Type:[/bold] {primary.type.capitalize()}\n"
                        f"[bold]Priority:[/bold] {primary.priority}",
                        title="[bold]Configuration Summary[/bold]",
                        border_style="green"
                    )
                    console.print(panel)
            
            if len(result.available_providers) > 1:
                console.print(f"\n[bold]Additional Available Providers:[/bold]")
                for provider in result.available_providers[1:]:
                    console.print(f"• {provider.name} ({provider.type})")
        
        else:
            console.print("[bold red]❌ Configuration Validation: FAILED[/bold red]")
            
            # Print errors
            if result.errors:
                console.print("\n[bold red]Errors:[/bold red]")
                for error in result.errors:
                    console.print(f"  • {error}")
            
            # Print missing configuration
            if result.missing_vars:
                console.print("\n[bold yellow]Missing Configuration:[/bold yellow]")
                for provider_name, missing_vars in result.missing_vars.items():
                    provider = self.providers[provider_name]
                    console.print(f"\n[bold]{provider.name} ({provider.type}):[/bold]")
                    for var in missing_vars:
                        if "endpoint not reachable" in var.lower():
                            console.print(f"  • {var}")
                        else:
                            console.print(f"  • {var}=<your-value-here>")
        
        # Print warnings
        if result.warnings:
            console.print("\n[bold yellow]Warnings:[/bold yellow]")
            for warning in result.warnings:
                console.print(f"  • {warning}")

    def print_setup_instructions(self, provider_name: str = None):
        """Print detailed setup instructions"""
        if provider_name and provider_name in self.providers:
            # Instructions for specific provider
            provider = self.providers[provider_name]
            self._print_provider_instructions(provider)
        else:
            # Instructions for all providers, organized by type
            console.print("[bold]Setup Instructions[/bold]")
            console.print("=" * 50)
            
            console.print("\n[bold blue]🏠 Local Services (Recommended for Privacy)[/bold blue]")
            console.print("These work offline and keep your data private.\n")
            
            local_providers = [p for p in self.providers.values() if p.type == "local"]
            for provider in local_providers:
                self._print_provider_instructions(provider, brief=True)
            
            console.print("\n[bold blue]☁️ Cloud Services (Internet Required)[/bold blue]")
            console.print("These require API keys and incur costs.\n")
            
            cloud_providers = [p for p in self.providers.values() if p.type == "cloud"]
            for provider in cloud_providers:
                self._print_provider_instructions(provider, brief=True)
            
            # Quick start
            console.print("\n[bold green]🚀 Quick Start:[/bold green]")
            console.print("1. Copy template: [cyan]cp .env.example .env[/cyan]")
            console.print("2. Edit [cyan].env[/cyan] with your preferred provider")
            console.print("3. Test: [cyan]python simple_local_cli.py config-status[/cyan]")

    def _print_provider_instructions(self, provider: ProviderConfig, brief: bool = False):
        """Print instructions for a single provider"""
        console.print(f"\n[bold]📋 {provider.name}[/bold] ({provider.type})")
        
        if brief:
            console.print(f"   Required: {', '.join(provider.required_vars)}")
        else:
            console.print(provider.setup_instructions)
            console.print(f"\n[bold]Required Variables:[/bold]")
            for var in provider.required_vars:
                console.print(f"  • {var}")
            
            if provider.optional_vars:
                console.print(f"\n[bold]Optional Variables:[/bold]")
                for var in provider.optional_vars:
                    console.print(f"  • {var} (default will be used if not set)")
            
            console.print(f"\n[bold]Example Configuration:[/bold]")
            console.print(f"[cyan]{provider.example_config}[/cyan]")

    def print_configuration_status(self):
        """Print current configuration status table"""
        table = Table(title="Provider Configuration Status")
        table.add_column("Provider", style="bold white")
        table.add_column("Type", style="cyan")
        table.add_column("Priority", style="yellow")
        table.add_column("Status", style="bold")
        table.add_column("Notes", style="dim")
        
        # Load current configuration
        self.load_env_file()
        
        for provider_name, provider in self.providers.items():
            is_configured, missing_vars = self.validate_provider(provider_name)
            
            if is_configured:
                status = "[bold green]✅ Configured[/bold green]"
                notes = "Ready to use"
            else:
                status = "[bold red]❌ Missing[/bold red]"
                if missing_vars:
                    notes = f"Needs: {', '.join(missing_vars)}"
                else:
                    notes = "Unknown error"
            
            table.add_row(
                provider.name,
                provider.type.capitalize(),
                str(provider.priority),
                status,
                notes
            )
        
        console.print(table)
        
        # Environment file status
        console.print("\n[bold]Environment Files:[/bold]")
        if self.env_file.exists():
            console.print(f"  ✅ .env file exists: {self.env_file.absolute()}")
        else:
            console.print(f"  ❌ .env file not found: {self.env_file.absolute()}")
            console.print(f"     Create with: cp .env.example .env")
        
        if self.env_example.exists():
            console.print(f"  ✅ .env.example exists: {self.env_example.absolute()}")
        else:
            console.print(f"  ❌ .env.example not found: {self.env_example.absolute()}")

    def get_configured_provider(self) -> Optional[Dict]:
        """Get the best configured provider info"""
        result = self.validate_configuration()
        
        if not result.success or not result.primary_provider:
            return None
        
        primary = next((p for p in result.available_providers if p.name == result.primary_provider), None)
        if not primary:
            return None
        
        # Build provider config dict
        config = {
            "name": primary.name,
            "type": primary.type,
            "priority": primary.priority,
            "required_vars": {},
            "optional_vars": {}
        }
        
        # Add configured variables
        for var in primary.required_vars:
            value = os.getenv(var)
            if value:
                config["required_vars"][var] = value
        
        for var in primary.optional_vars:
            value = os.getenv(var)
            if value:
                config["optional_vars"][var] = value
        
        return config


# Global validator instance
validator = SimpleValidator()


def validate_config() -> ValidationResult:
    """Validate the configuration using the global validator"""
    return validator.validate_configuration()


def print_validation_result(result: ValidationResult):
    """Print validation result using the global validator"""
    validator.print_validation_result(result)


def print_setup_instructions(provider_name: str = None):
    """Print setup instructions using the global validator"""
    validator.print_setup_instructions(provider_name)


def print_configuration_status():
    """Print configuration status using the global validator"""
    validator.print_configuration_status()


def get_configured_provider() -> Optional[Dict]:
    """Get the best configured provider using the global validator"""
    return validator.get_configured_provider()


if __name__ == "__main__":
    """Test the validator when run directly"""
    print("Testing Simple Configuration Validator")
    print("=" * 50)
    
    result = validate_config()
    print_validation_result(result)
    
    if not result.success:
        print("\n" + "=" * 50)
        print_setup_instructions()