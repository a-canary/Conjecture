"""
Unified Configuration Validator
Consolidates 4 overlapping configuration validators into a single unified validator

Supported Formats:
1. Simple Provider Format: PROVIDER_[NAME]=[BASE_URL],[API_KEY],[MODEL],[PROTOCOL]
2. Individual Environment Format: [PROVIDER]_API_URL, [PROVIDER]_API_KEY, [PROVIDER]_MODELS
3. Unified Provider Format: PROVIDER_API_URL, PROVIDER_API_KEY, PROVIDER_MODEL
4. Simple Validator Format: Individual vars per provider (OLLAMA_ENDPOINT, OPENAI_API_KEY, etc.)

Priority Order (highest to lowest):
1. Unified Provider Format (PROVIDER_*) - Recommended format
2. Simple Provider Format (PROVIDER_[NAME])
3. Individual Environment Format ([PROVIDER]_API_URL)
4. Simple Validator Format (OLLAMA_ENDPOINT, etc.)
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from enum import Enum

# Import adapters
from .adapters import (
    BaseAdapter, 
    ProviderConfig, 
    ValidationResult,
    SimpleProviderAdapter,
    IndividualEnvAdapter,
    UnifiedProviderAdapter,
    SimpleValidatorAdapter
)

console = Console()

class ConfigFormat(Enum):
    """Configuration format types"""
    UNIFIED_PROVIDER = "unified_provider"
    SIMPLE_PROVIDER = "simple_provider"
    INDIVIDUAL_ENV = "individual_env"
    SIMPLE_VALIDATOR = "simple_validator"
    UNKNOWN = "unknown"

@dataclass
class UnifiedValidationResult:
    """Comprehensive validation result for all formats"""
    success: bool
    active_format: ConfigFormat
    active_providers: List[ProviderConfig]
    all_detected_formats: List[ConfigFormat]
    errors: List[str]
    warnings: List[str]
    migration_suggestions: Dict[str, Dict[str, List[str]]]
    format_conflicts: List[str]

    def __post_init__(self):
        if self.active_providers is None:
            self.active_providers = []
        if self.all_detected_formats is None:
            self.all_detected_formats = []
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.migration_suggestions is None:
            self.migration_suggestions = {}
        if self.format_conflicts is None:
            self.format_conflicts = []

class UnifiedConfigValidator:
    """
    Unified configuration validator that supports all formats
    with automatic format detection and priority-based selection
    """

    def __init__(self, env_file: str = ".env"):
        self.env_file = Path(env_file)
        self.env_vars = self._load_environment_variables()
        
        # Initialize adapters in priority order
        self.adapters = [
            UnifiedProviderAdapter(),    # Highest priority
            SimpleProviderAdapter(),     # High priority
            IndividualEnvAdapter(),      # Medium priority
            SimpleValidatorAdapter()     # Lowest priority (legacy)
        ]
        
        # Cache for validation results
        self._validation_cache = None
        self._primary_provider_cache = None

    def _load_environment_variables(self) -> Dict[str, str]:
        """Load environment variables from .env file and system environment"""
        env_vars = dict(os.environ)
        
        # Load from .env file if it exists
        if self.env_file.exists():
            try:
                with open(self.env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            env_vars[key.strip()] = value.strip()
            except Exception as e:
                console.print(f"[red]Warning: Could not load {self.env_file}: {e}[/red]")
        
        return env_vars

    def detect_formats(self) -> List[ConfigFormat]:
        """
        Detect all configured formats in the environment
        """
        detected_formats = []
        
        for adapter in self.adapters:
            if adapter.detect_format(self.env_vars):
                format_name = adapter.format_type
                try:
                    detected_formats.append(ConfigFormat(format_name))
                except ValueError:
                    detected_formats.append(ConfigFormat.UNKNOWN)
        
        return detected_formats

    def get_active_format(self) -> ConfigFormat:
        """
        Get the active format based on priority
        """
        detected_formats = self.detect_formats()
        
        if not detected_formats:
            return ConfigFormat.UNKNOWN
        
        # Return the highest priority format
        format_priorities = {
            ConfigFormat.UNIFIED_PROVIDER: 1,
            ConfigFormat.SIMPLE_PROVIDER: 2,
            ConfigFormat.INDIVIDUAL_ENV: 3,
            ConfigFormat.SIMPLE_VALIDATOR: 4
        }
        
        detected_formats.sort(key=lambda f: format_priorities.get(f, 999))
        return detected_formats[0]

    def validate_configuration(self, force_refresh: bool = False) -> UnifiedValidationResult:
        """
        Validate the complete configuration across all formats
        """
        if self._validation_cache and not force_refresh:
            return self._validation_cache
        
        result = UnifiedValidationResult(
            success=False,
            active_format=ConfigFormat.UNKNOWN,
            active_providers=[],
            all_detected_formats=[],
            errors=[],
            warnings=[],
            migration_suggestions={},
            format_conflicts=[]
        )
        
        # Detect all formats
        detected_formats = self.detect_formats()
        result.all_detected_formats = detected_formats
        
        if not detected_formats:
            result.errors.append("No configuration format detected")
            result.warnings.append("Please configure at least one provider using any supported format")
            self._validation_cache = result
            return result
        
        # Check for format conflicts
        if len(detected_formats) > 1:
            format_names = [f.value for f in detected_formats]
            result.format_conflicts.append(f"Multiple formats detected: {', '.join(format_names)}")
            result.warnings.append("Multiple configuration formats detected. Using highest priority format.")
        
        # Get active format and validate with its adapter
        active_format = self.get_active_format()
        result.active_format = active_format
        
        # Find the corresponding adapter
        active_adapter = None
        for adapter in self.adapters:
            if adapter.format_type == active_format.value:
                active_adapter = adapter
                break
        
        if not active_adapter:
            result.errors.append(f"No adapter found for format: {active_format.value}")
            self._validation_cache = result
            return result
        
        # Load providers using the active adapter
        success, providers, errors = active_adapter.load_providers(self.env_vars)
        result.active_providers = providers
        result.errors.extend(errors)
        
        # Validate the format itself
        is_valid, format_errors = active_adapter.validate_format(self.env_vars)
        if not is_valid:
            result.errors.extend(format_errors)
        
        result.success = success and len(result.errors) == 0
        
        # Generate migration suggestions for non-unified formats
        if active_format != ConfigFormat.UNIFIED_PROVIDER:
            try:
                migration_result = active_adapter.migrate_to_unified_format(self.env_vars)
                result.migration_suggestions[active_format.value] = migration_result
            except Exception as e:
                result.warnings.append(f"Could not generate migration suggestions: {e}")
        
        # Add warnings based on configuration state
        self._add_configuration_warnings(result)
        
        self._validation_cache = result
        return result

    def _add_configuration_warnings(self, result: UnifiedValidationResult):
        """Add contextual warnings based on configuration state"""
        # Security warnings
        if self.env_file.exists():
            gitignore_path = Path(".gitignore")
            if not gitignore_path.exists() or ".env" not in gitignore_path.read_text():
                result.warnings.append(".env file is not protected by .gitignore! Add .env to .gitignore immediately.")
        
        # Provider-specific warnings
        for provider in result.active_providers:
            if not provider.is_local and not provider.api_key:
                result.warnings.append(f"{provider.name}: API key is missing for cloud service")
            
            if provider.is_local and not provider.base_url.startswith('http://localhost'):
                result.warnings.append(f"{provider.name}: Local service URL should typically use localhost")

    def get_primary_provider(self, force_refresh: bool = False) -> Optional[ProviderConfig]:
        """
        Get the primary provider based on active format and priority
        """
        if self._primary_provider_cache and not force_refresh:
            return self._primary_provider_cache
        
        validation_result = self.validate_configuration(force_refresh)
        
        if not validation_result.success or not validation_result.active_providers:
            self._primary_provider_cache = None
            return None
        
        # Providers are already sorted by priority in adapters
        primary = validation_result.active_providers[0]
        self._primary_provider_cache = primary
        return primary

    def get_all_providers(self) -> Dict[ConfigFormat, List[ProviderConfig]]:
        """
        Get all configured providers across all formats
        """
        all_providers = {}
        
        for adapter in self.adapters:
            if adapter.detect_format(self.env_vars):
                success, providers, _ = adapter.load_providers(self.env_vars)
                if success:
                    try:
                        format_key = ConfigFormat(adapter.format_type)
                        all_providers[format_key] = providers
                    except ValueError:
                        all_providers[ConfigFormat.UNKNOWN] = providers
        
        return all_providers

    def migrate_to_unified_format(self, source_format: Optional[ConfigFormat] = None) -> Dict[str, List[str]]:
        """
        Generate migration commands to convert to unified format
        """
        if source_format is None:
            source_format = self.get_active_format()
        
        if source_format == ConfigFormat.UNIFIED_PROVIDER:
            return {"success": ["Already using unified format"], "errors": []}
        
        # Find the adapter for the source format
        source_adapter = None
        for adapter in self.adapters:
            if adapter.format_type == source_format.value:
                source_adapter = adapter
                break
        
        if not source_adapter:
            return {"success": [], "errors": [f"No adapter found for format: {source_format.value}"]}
        
        try:
            return source_adapter.migrate_to_unified_format(self.env_vars)
        except Exception as e:
            return {"success": [], "errors": [f"Migration failed: {str(e)}"]}

    def show_configuration_status(self, detailed: bool = False):
        """
        Display current configuration status
        """
        validation_result = self.validate_configuration()
        
        console.print("[bold blue]Unified Configuration Validator Status[/bold blue]")
        console.print("=" * 60)
        
        # Show format information
        formats_table = Table(title="Detected Formats (Priority Order)")
        formats_table.add_column("Format", style="cyan")
        formats_table.add_column("Status", style="bold")
        formats_table.add_column("Priority", style="yellow")
        formats_table.add_column("Providers", style="green")
        
        format_priorities = {
            ConfigFormat.UNIFIED_PROVIDER: 1,
            ConfigFormat.SIMPLE_PROVIDER: 2,
            ConfigFormat.INDIVIDUAL_ENV: 3,
            ConfigFormat.SIMPLE_VALIDATOR: 4
        }
        
        for detected_format in validation_result.all_detected_formats:
            is_active = detected_format == validation_result.active_format
            status = "[bold green]Active[/bold green]" if is_active else "[dim]Inactive[/dim]"
            priority = format_priorities.get(detected_format, 999)
            
            # Count providers for this format
            all_providers = self.get_all_providers()
            provider_count = len(all_providers.get(detected_format, []))
            
            formats_table.add_row(
                detected_format.value.replace('_', ' ').title(),
                status,
                str(priority),
                str(provider_count)
            )
        
        console.print(formats_table)
        
        # Show active providers
        if validation_result.active_providers:
            console.print(f"\n[bold green]Active Providers ({validation_result.active_format.value.replace('_', ' ').title()}):[/bold green]")
            
            providers_table = Table()
            providers_table.add_column("Provider", style="cyan")
            providers_table.add_column("Type", style="green")
            providers_table.add_column("Priority", style="yellow")
            providers_table.add_column("Status", style="bold")
            providers_table.add_column("Base URL", style="white")
            providers_table.add_column("Model", style="blue")
            
            for provider in validation_result.active_providers:
                provider_type = "Local" if provider.is_local else "Cloud"
                status = "[green]Ready[/green]" if self._is_provider_ready(provider) else "[red]Incomplete[/red]"
                
                # Truncate for display
                url_display = provider.base_url[:30] + "..." if len(provider.base_url) > 30 else provider.base_url
                model_display = provider.model or provider.models[0] if provider.models else "None"
                
                providers_table.add_row(
                    provider.name,
                    provider_type,
                    str(provider.priority),
                    status,
                    url_display,
                    model_display
                )
            
            console.print(providers_table)
            
            # Show primary provider
            primary = self.get_primary_provider()
            if primary:
                console.print(f"\n[green]Primary provider: {primary.name} ({primary.model})[/green]")
        
        # Show errors and warnings
        if validation_result.errors:
            console.print(f"\n[bold red]Errors:[/bold red]")
            for error in validation_result.errors:
                console.print(f"  • {error}")
        
        if validation_result.warnings:
            console.print(f"\n[bold yellow]Warnings:[/bold red]")
            for warning in validation_result.warnings:
                console.print(f"  • {warning}")
        
        if validation_result.format_conflicts:
            console.print(f"\n[bold yellow]Format Conflicts:[/bold red]")
            for conflict in validation_result.format_conflicts:
                console.print(f"  • {conflict}")
        
        # Show migration suggestions
        if detailed and validation_result.migration_suggestions:
            console.print(f"\n[bold blue]Migration Suggestions:[/bold blue]")
            for format_name, migration in validation_result.migration_suggestions.items():
                if migration.get("success"):
                    console.print(f"\n[cyan]Migration from {format_name.replace('_', ' ').title()}:[/cyan]")
                    for line in migration["success"][:5]:  # Show first 5 lines
                        console.print(f"  {line}")
                    if len(migration["success"]) > 5:
                        console.print(f"  ... and {len(migration['success']) - 5} more lines")
        
        # Overall status
        if validation_result.success:
            console.print(f"\n[bold green]✅ Configuration is valid and ready to use![/bold green]")
        else:
            console.print(f"\n[bold red]❌ Configuration has issues that need to be resolved[/bold red]")

    def _is_provider_ready(self, provider: ProviderConfig) -> bool:
        """Check if a provider is properly configured"""
        if not provider.base_url or not (provider.model or provider.models):
            return False
        
        if not provider.is_local and not provider.api_key:
            return False
        
        return True

    def show_migration_guide(self, target_format: ConfigFormat = ConfigFormat.UNIFIED_PROVIDER):
        """
        Show detailed migration guide
        """
        current_format = self.get_active_format()
        
        console.print(f"[bold blue]Migration Guide: {current_format.value.replace('_', ' ').title()} → {target_format.value.replace('_', ' ').title()}[/bold blue]")
        console.print("=" * 60)
        
        if current_format == target_format:
            console.print("[green]You're already using the target format![/green]")
            return
        
        migration_commands = self.migrate_to_unified_format(current_format)
        
        if migration_commands.get("errors"):
            console.print("[bold red]Migration Errors:[/bold red]")
            for error in migration_commands["errors"]:
                console.print(f"  • {error}")
            return
        
        console.print("[bold yellow]Steps to migrate:[/bold yellow]")
        console.print("1. Backup your current .env file")
        console.print("2. Remove the old configuration variables")
        console.print("3. Add the new unified variables")
        console.print("4. Test your configuration")
        
        console.print(f"\n[bold cyan]Migration Commands:[/bold cyan]")
        for line in migration_commands["success"]:
            console.print(f"  {line}")
        
        console.print(f"\n[bold green]After migration:[/bold green]")
        console.print("1. Run: python -m conjecture.config.unified_validator test")
        console.print("2. Verify your configuration with: python simple_cli.py config-status")

    def get_format_examples(self, format_type: Optional[ConfigFormat] = None) -> Dict[str, Any]:
        """
        Get example configurations for a specific format or all formats
        """
        if format_type is None:
            examples = {}
            for adapter in self.adapters:
                try:
                    format_key = ConfigFormat(adapter.format_type)
                    examples[format_key] = adapter.get_format_examples()
                except ValueError:
                    examples[ConfigFormat.UNKNOWN] = adapter.get_format_examples()
            return examples
        else:
            # Find the adapter for the requested format
            for adapter in self.adapters:
                if adapter.format_type == format_type.value:
                    return adapter.get_format_examples()
            return {}

    def export_configuration(self, format_type: ConfigFormat = ConfigFormat.UNIFIED_PROVIDER) -> Dict[str, str]:
        """
        Export current configuration in the specified format
        """
        primary_provider = self.get_primary_provider()
        if not primary_provider:
            return {"error": "No provider configured"}
        
        exported_config = {}
        
        if format_type == ConfigFormat.UNIFIED_PROVIDER:
            exported_config = {
                "PROVIDER_API_URL": primary_provider.base_url,
                "PROVIDER_API_KEY": primary_provider.api_key,
                "PROVIDER_MODEL": primary.model or (primary.models[0] if primary.models else "")
            }
        elif format_type == ConfigFormat.SIMPLE_PROVIDER:
            provider_key = primary_provider.name.lower().replace(' ', '_')
            exported_config[f"PROVIDER_{provider_key.upper()}"] = (
                f"{primary_provider.base_url},{primary_provider.api_key},"
                f"{primary.model or primary.models[0] if primary.models else ''},"
                f"{primary.protocol or 'openai'}"
            )
        else:
            exported_config = {"error": f"Export to {format_type.value} not yet implemented"}
        
        return exported_config

    def clear_cache(self):
        """Clear internal caches"""
        self._validation_cache = None
        self._primary_provider_cache = None

# Global instance for backward compatibility
_global_validator = None

def get_unified_validator(env_file: str = ".env") -> UnifiedConfigValidator:
    """Get the global unified validator instance"""
    global _global_validator
    if _global_validator is None:
        _global_validator = UnifiedConfigValidator(env_file)
    return _global_validator

def validate_config() -> UnifiedValidationResult:
    """Validate configuration using the global validator"""
    return get_unified_validator().validate_configuration()

def get_primary_provider() -> Optional[ProviderConfig]:
    """Get primary provider using the global validator"""
    return get_unified_validator().get_primary_provider()

def show_configuration_status(detailed: bool = False):
    """Show configuration status using the global validator"""
    return get_unified_validator().show_configuration_status(detailed)

def main():
    """Test the unified validator when run directly"""
    validator = UnifiedConfigValidator()
    
    console.print("[bold green]Unified Configuration Validator Test[/bold green]")
    console.print("=" * 60)
    
    validator.show_configuration_status(detailed=True)

if __name__ == "__main__":
    main()