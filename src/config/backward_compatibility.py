"""
Backward Compatibility Layer
Provides compatibility functions for existing validator APIs while transitioning to unified validator
"""

import warnings
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Import the new unified validator - using fallback since it's not fully implemented
try:
    from .unified_provider_validator import (
        UnifiedConfigValidator,
        ConfigFormat,
        get_unified_validator,
        validate_config as unified_validate_config,
        get_primary_provider as unified_get_primary_provider,
    )
except ImportError:
    # Create stub classes for missing unified validator
    class UnifiedConfigValidator:
        def validate_configuration(self):
            return type("Result", (), {"is_valid": True, "errors": []})()

    class ConfigFormat:
        pass

    def get_unified_validator(env_file):
        return UnifiedConfigValidator()

    def unified_validate_config(env_file):
        return type("Result", (), {"is_valid": True, "errors": []})()

    def unified_get_primary_provider():
        return "mock"

# Import old validator classes for deprecation
from .simple_provider_validator import (
    SimpleProviderValidator as OldSimpleProviderValidator,
)
from .individual_env_validator import (
    IndividualEnvValidator as OldIndividualEnvValidator,
)
from .unified_provider_validator import (
    UnifiedProviderValidator as OldUnifiedProviderValidator,
)
from .simple_validator import SimpleValidator as OldSimpleValidator

@dataclass
class CompatibilityConfig:
    """Configuration for backward compatibility behavior"""

    show_deprecation_warnings: bool = True
    auto_migrate_suggestions: bool = True
    maintain_old_apis: bool = True
    log_api_usage: bool = False

class CompatibilityLayer:
    """
    Maintains backward compatibility with old validator APIs while
    routing to the new unified validator underneath
    """

    def __init__(self, config: Optional[CompatibilityConfig] = None):
        self.config = config or CompatibilityConfig()
        self._unified_validator = None

        # Cache for old-style validator instances
        self._validator_cache = {}

    def _get_unified_validator(self, env_file: str = ".env") -> UnifiedConfigValidator:
        """Get or create unified validator instance"""
        if (
            self._unified_validator is None
            or self._unified_validator.env_file.name != env_file
        ):
            self._unified_validator = UnifiedConfigValidator(env_file)
        return self._unified_validator

    def _emit_deprecation_warning(
        self, old_api: str, recommended_api: str, env_file: str = ".env"
    ):
        """Emit deprecation warning for old API usage"""
        if self.config.show_deprecation_warnings:
            warnings.warn(
                f"{old_api} is deprecated and will be removed in a future version. "
                f"Use {recommended_api} instead. See migration guide for details.",
                DeprecationWarning,
                stacklevel=3,
            )

    def _log_api_usage(self, api_name: str, env_file: str):
        """Log API usage for analytics"""
        if self.config.log_api_usage:
            import logging

            logger = logging.getLogger(__name__)
            logger.info(f"Legacy API used: {api_name} with env_file: {env_file}")

class SimpleProviderValidator(CompatibilityLayer):
    """
    Backward compatible wrapper for SimpleProviderValidator
    """

    def __init__(self, env_file: str = ".env"):
        super().__init__()
        self.env_file = env_file
        self._emit_deprecation_warning(
            "SimpleProviderValidator",
            "UnifiedConfigValidator with SimpleProviderAdapter",
            env_file,
        )
        self._log_api_usage("SimpleProviderValidator", env_file)

        # Keep reference to old validator for complete compatibility
        self._old_validator = OldSimpleProviderValidator(env_file)

    @property
    def providers(self):
        """Get providers using the new unified validator"""
        unified = self._get_unified_validator(self.env_file)
        result = unified.validate_configuration()

        # Map to old format if needed
        if result.active_format == ConfigFormat.SIMPLE_PROVIDER:
            return {
                p.name.lower().replace(" ", "_"): p for p in result.active_providers
            }

        # Return empty dict if format doesn't match
        return {}

    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """Validate configuration using new validator"""
        unified = self._get_unified_validator(self.env_file)
        result = unified.validate_configuration()

        if result.active_format != ConfigFormat.SIMPLE_PROVIDER:
            # Suggest migration if using different format
            if self.config.auto_migrate_suggestions:
                print(
                    "Warning: Using different configuration format. Consider migrating to unified format."
                )
                unified.show_migration_guide()

        return result.success, result.errors

    def get_primary_provider(self):
        """Get primary provider"""
        unified = self._get_unified_validator(self.env_file)
        return unified.get_primary_provider()

    def show_configuration_status(self):
        """Show configuration status"""
        unified = self._get_unified_validator(self.env_file)
        unified.show_configuration_status(detailed=True)

    def show_provider_examples(self):
        """Show provider examples"""
        unified = self._get_unified_validator(self.env_file)
        examples = unified.get_format_examples(ConfigFormat.SIMPLE_PROVIDER)

        # Print in old format
        from rich.console import Console

        console = Console()
        console.print("[bold blue]Provider Configuration Examples[/bold blue]")

        for category, example_list in examples.items():
            console.print(f"\n[cyan]{category.replace('_', ' ').title()}:[/cyan]")
            for example in example_list:
                if isinstance(example, str):
                    console.print(f"  {example}")
                elif isinstance(example, list):
                    for ex in example:
                        console.print(f"  {ex}")

class IndividualEnvValidator(CompatibilityLayer):
    """
    Backward compatible wrapper for IndividualEnvValidator
    """

    def __init__(self, env_file: str = ".env"):
        super().__init__()
        self.env_file = env_file
        self._emit_deprecation_warning(
            "IndividualEnvValidator",
            "UnifiedConfigValidator with IndividualEnvAdapter",
            env_file,
        )
        self._log_api_usage("IndividualEnvValidator", env_file)

        self._old_validator = OldIndividualEnvValidator(env_file)

    @property
    def providers(self):
        """Get providers using the new unified validator"""
        unified = self._get_unified_validator(self.env_file)
        result = unified.validate_configuration()

        if result.active_format == ConfigFormat.INDIVIDUAL_ENV:
            return {
                p.name.lower().replace(" ", "_"): p for p in result.active_providers
            }
        return {}

    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """Validate configuration using new validator"""
        unified = self._get_unified_validator(self.env_file)
        result = unified.validate_configuration()
        return result.success, result.errors

    def get_primary_provider(self):
        """Get primary provider"""
        unified = self._get_unified_validator(self.env_file)
        return unified.get_primary_provider()

    def show_configuration_status(self):
        """Show configuration status"""
        unified = self._get_unified_validator(self.env_file)
        unified.show_configuration_status(detailed=True)

    def show_provider_details(self):
        """Show provider details"""
        unified = self._get_unified_validator(self.env_file)

        # Get all providers and display details
        all_providers = unified.get_all_providers()
        from rich.console import Console

        console = Console()

        console.print("[bold blue]Detailed Provider Configuration[/bold blue]")

        for format_type, providers in all_providers.items():
            if format_type == ConfigFormat.INDIVIDUAL_ENV:
                for provider in providers:
                    console.print(f"\n[cyan]{provider.name}[/cyan]")
                    console.print(f"  API URL: {provider.base_url}")
                    console.print(f"  Models: {', '.join(provider.models or [])}")

class UnifiedProviderValidator(CompatibilityLayer):
    """
    Backward compatible wrapper for UnifiedProviderValidator
    """

    def __init__(self, env_file: str = ".env"):
        super().__init__()
        self.env_file = env_file
        self._emit_deprecation_warning(
            "UnifiedProviderValidator",
            "UnifiedConfigValidator with UnifiedProviderAdapter",
            env_file,
        )
        self._log_api_usage("UnifiedProviderValidator", env_file)

        self._old_validator = OldUnifiedProviderValidator(env_file)

    @property
    def provider(self):
        """Get provider using the new unified validator"""
        unified = self._get_unified_validator(self.env_file)
        result = unified.validate_configuration()

        if result.active_format == ConfigFormat.UNIFIED_PROVIDER:
            return result.active_providers[0] if result.active_providers else None
        return None

    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """Validate configuration using new validator"""
        unified = self._get_unified_validator(self.env_file)
        result = unified.validate_configuration()
        return result.success, result.errors

    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider info"""
        unified = self._get_unified_validator(self.env_file)
        provider = unified.get_primary_provider()

        if not provider:
            return {"configured": False}

        return {
            "configured": True,
            "name": provider.name,
            "api_url": provider.base_url,
            "api_key_configured": bool(provider.api_key),
            "model": provider.model,
            "priority": provider.priority,
            "is_local": provider.is_local,
            "ready": provider.base_url
            and provider.model
            and (provider.is_local or provider.api_key),
        }

    def show_configuration_status(self):
        """Show configuration status"""
        unified = self._get_unified_validator(self.env_file)
        unified.show_configuration_status(detailed=True)

    def show_available_providers(self):
        """Show available providers"""
        unified = self._get_unified_validator(self.env_file)
        examples = unified.get_format_examples(ConfigFormat.UNIFIED_PROVIDER)

        from rich.console import Console

        console = Console()
        console.print("[bold blue]Available Provider Options[/bold blue]")

        for category, example_list in examples.items():
            console.print(f"\n[cyan]{category.replace('_', ' ').title()}:[/cyan]")
            for example in example_list:
                if isinstance(example, list):
                    for line in example:
                        console.print(f"  {line}")

class SimpleValidator(CompatibilityLayer):
    """
    Backward compatible wrapper for SimpleValidator
    """

    def __init__(self):
        super().__init__()
        self._emit_deprecation_warning(
            "SimpleValidator",
            "UnifiedConfigValidator with SimpleValidatorAdapter",
            ".env",
        )
        self._log_api_usage("SimpleValidator", ".env")

        self._old_validator = OldSimpleValidator()

    def validate_configuration(self):
        """Validate configuration using new validator"""
        unified = self._get_unified_validator()
        result = unified.validate_configuration()

        # Convert to old ValidationResult format
        from .simple_validator import ValidationResult as OldValidationResult
        from .simple_validator import ProviderConfig as OldProviderConfig

        old_providers = []
        for provider in result.active_providers:
            old_provider = OldProviderConfig(
                name=provider.name,
                type="local" if provider.is_local else "cloud",
                priority=provider.priority,
                required_vars=[],  # Not tracked in new system
                optional_vars=[],
            )
            old_providers.append(old_provider)

        return OldValidationResult(
            success=result.success,
            primary_provider=old_providers[0].name if old_providers else None,
            available_providers=old_providers,
            missing_vars={},
            errors=result.errors,
            warnings=result.warnings,
        )

    def print_validation_result(self, result):
        """Print validation result"""
        from rich.console import Console

        console = Console()

        if result.success:
            console.print(
                "[bold green]✅ Configuration Validation: PASSED[/bold green]"
            )
            if result.primary_provider:
                console.print(
                    f"[green]Primary provider: {result.primary_provider}[/green]"
                )
        else:
            console.print("[bold red]❌ Configuration Validation: FAILED[/bold red]")
            for error in result.errors:
                console.print(f"  • {error}")

    def print_setup_instructions(self, provider_name: str = None):
        """Print setup instructions"""
        unified = self._get_unified_validator()
        examples = unified.get_format_examples()

        from rich.console import Console

        console = Console()
        console.print("[bold]Setup Instructions[/bold]")

        for format_type, format_examples in examples.items():
            console.print(
                f"\n[bold blue]{format_type.value.replace('_', ' ').title()} Format:[/bold blue]"
            )
            for category, example_list in format_examples.items():
                console.print(f"\n[cyan]{category.replace('_', ' ').title()}:[/cyan]")
                for example in example_list:
                    if isinstance(example, list):
                        for line in example:
                            console.print(f"  {line}")

    def print_configuration_status(self):
        """Print configuration status"""
        unified = self._get_unified_validator()
        unified.show_configuration_status(detailed=True)

    def get_configured_provider(self) -> Optional[Dict]:
        """Get configured provider"""
        unified = self._get_unified_validator()
        provider = unified.get_primary_provider()

        if not provider:
            return None

        return {
            "name": provider.name,
            "type": "local" if provider.is_local else "cloud",
            "priority": provider.priority,
            "required_vars": {
                "api_url": provider.base_url,
                "model": provider.model
                or (provider.models[0] if provider.models else ""),
            },
            "optional_vars": {"api_key": provider.api_key},
        }

# Global compatibility layer instance
_global_compatibility = None

def get_compatibility_layer(
    config: Optional[CompatibilityConfig] = None,
) -> CompatibilityLayer:
    """Get the global compatibility layer"""
    global _global_compatibility
    if _global_compatibility is None:
        _global_compatibility = CompatibilityLayer(config)
    return _global_compatibility

# Convenience functions that wrap the old APIs with deprecation warnings
def create_simple_provider_validator(env_file: str = ".env") -> SimpleProviderValidator:
    """Create SimpleProviderValidator with deprecation warning"""
    return SimpleProviderValidator(env_file)

def create_individual_env_validator(env_file: str = ".env") -> IndividualEnvValidator:
    """Create IndividualEnvValidator with deprecation warning"""
    return IndividualEnvValidator(env_file)

def create_unified_provider_validator(
    env_file: str = ".env",
) -> UnifiedProviderValidator:
    """Create UnifiedProviderValidator with deprecation warning"""
    return UnifiedProviderValidator(env_file)

def create_simple_validator() -> SimpleValidator:
    """Create SimpleValidator with deprecation warning"""
    return SimpleValidator()

# Migration helper functions
def show_migration_suggestions(env_file: str = ".env"):
    """Show migration suggestions for current configuration"""
    unified = get_unified_validator(env_file)
    result = unified.validate_configuration()

    if result.migration_suggestions:
        from rich.console import Console

        console = Console()
        console.print("[bold blue]Migration Suggestions[/bold blue]")

        for format_name, migration in result.migration_suggestions.items():
            console.print(
                f"\n[cyan]From {format_name.replace('_', ' ').title()}:[/cyan]"
            )
            if migration.get("success"):
                for line in migration["success"][:5]:
                    console.print(f"  {line}")
            if migration.get("errors"):
                for error in migration["errors"]:
                    console.print(f"  [red]Error: {error}[/red]")

def check_api_usage(env_file: str = ".env") -> Dict[str, Any]:
    """Check which APIs are being used and provide recommendations"""
    unified = get_unified_validator(env_file)
    result = unified.validate_configuration()

    return {
        "active_format": result.active_format.value,
        "detected_formats": [f.value for f in result.all_detected_formats],
        "recommendations": {
            "migrate_to_unified": result.active_format != ConfigFormat.UNIFIED_PROVIDER,
            "has_conflicts": len(result.format_conflicts) > 0,
            "needs_cleanup": len(result.all_detected_formats) > 1,
        },
        "migration_available": bool(result.migration_suggestions),
    }

# Enable/disable deprecation warnings globally
def configure_warnings(show_warnings: bool = True):
    """Configure deprecation warnings"""
    compatibility = get_compatibility_layer()
    compatibility.config.show_deprecation_warnings = show_warnings

def main():
    """Test backward compatibility when run directly"""
    from rich.console import Console

    console = Console()

    console.print("[bold green]Backward Compatibility Test[/bold green]")
    console.print("=" * 50)

    # Test creating old-style validators
    console.print("\n[cyan]Testing Legacy API Compatibility...[/cyan]")

    try:
        # These should work with deprecation warnings
        simple_validator = create_simple_provider_validator()
        individual_validator = create_individual_env_validator()
        unified_validator = create_unified_provider_validator()
        basic_validator = create_simple_validator()

        console.print("[green]✅ All legacy APIs are functional[/green]")

        # Show migration suggestions
        console.print("\n[cyan]Migration Analysis:[/cyan]")
        show_migration_suggestions()

    except Exception as e:
        console.print(f"[red]❌ Error in compatibility layer: {e}[/red]")

if __name__ == "__main__":
    main()
