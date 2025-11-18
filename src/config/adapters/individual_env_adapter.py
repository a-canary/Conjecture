"""
Individual Environment Adapter
Handles format: [PROVIDER]_API_URL, [PROVIDER]_API_KEY, [PROVIDER]_MODELS
Example: OLLAMA_API_URL=http://localhost:11434, OLLAMA_API_KEY=, OLLAMA_MODELS=["llama2","mistral"]
"""

import json
import os
from typing import Dict, List, Tuple
from .base_adapter import BaseAdapter, ProviderConfig, ValidationResult, FormatPriority

class IndividualEnvAdapter(BaseAdapter):
    """
    Adapter for individual environment variable format:
    [PROVIDER]_API_URL, [PROVIDER]_API_KEY, [PROVIDER]_MODELS
    
    This format allows configuring each provider with separate variables
    """

    def __init__(self):
        super().__init__(
            format_type="individual_env",
            priority=FormatPriority.MEDIUM  # Medium priority
        )

    def detect_format(self, env_vars: Dict[str, str]) -> bool:
        """
        Detect if environment contains individual provider variables
        """
        url_vars = [k for k in env_vars.keys() if k.endswith('_API_URL')]
        return len(url_vars) > 0

    def validate_format(self, env_vars: Dict[str, str]) -> Tuple[bool, List[str]]:
        """
        Validate the format structure
        """
        errors = []

        # Find all providers by looking for _API_URL variables
        provider_names = set()
        for var in env_vars.keys():
            if var.endswith('_API_URL'):
                provider_name = var[:-9]  # Remove '_API_URL' suffix
                provider_names.add(provider_name.lower())

        if not provider_names:
            errors.append("No [PROVIDER]_API_URL variables found")
            return False, errors

        for provider_name in provider_names:
            # Check required variables
            url_var = f"{provider_name.upper()}_API_URL"
            key_var = f"{provider_name.upper()}_API_KEY"
            models_var = f"{provider_name.upper()}_MODELS"

            api_url = env_vars.get(url_var, "")
            api_key = env_vars.get(key_var, "")
            models_str = env_vars.get(models_var, "")

            # Validate API URL
            if not api_url:
                errors.append(f"{url_var}: API URL is required")
            elif not api_url.startswith(('http://', 'https://')):
                errors.append(f"{url_var}: API URL must start with http:// or https://")

            # Validate models
            if models_str:
                try:
                    # Try to parse as JSON array
                    if models_str.startswith('[') and models_str.endswith(']'):
                        json.loads(models_str)
                    # Handle manually quoted array: ["model1","model2"]
                    elif '"' in models_str or "'" in models_str:
                        models_str = models_str.strip('[]')
                        for model in models_str.split(','):
                            model = model.strip().strip('"').strip("'")
                            if not model:
                                errors.append(f"{models_var}: Empty model name found")
                except json.JSONDecodeError:
                    errors.append(f"{models_var}: Invalid JSON array format")

        return len(errors) == 0, errors

    def load_providers(self, env_vars: Dict[str, str]) -> Tuple[bool, List[ProviderConfig], List[str]]:
        """
        Load providers from environment variables
        """
        providers = []
        errors = []

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

            api_url = env_vars.get(url_var, "")
            api_key = env_vars.get(key_var, "")
            models_str = env_vars.get(models_var, "")

            # Only include provider if URL is configured
            if api_url:
                try:
                    # Parse models
                    models = self._parse_models(models_str, provider_name)

                    # Normalize base URL
                    api_url = self.normalize_url(api_url)

                    # Get provider info from URL
                    provider_info = self.detect_provider_type(api_url)

                    # Create provider config
                    provider_config = self.create_provider_config(
                        name=provider_info["name"],
                        base_url=api_url,
                        api_key=api_key,
                        models=models,
                        priority=config["priority"],
                        is_local=config["is_local"],
                        protocol=provider_info["protocol"],
                        model=models[0] if models else None
                    )

                    providers.append(provider_config)

                except Exception as e:
                    errors.append(f"Error parsing {provider_name} configuration: {str(e)}")

        # Sort by priority
        providers.sort(key=lambda p: p.priority)

        return len(providers) > 0, providers, errors

    def _parse_models(self, models_str: str, provider_name: str) -> List[str]:
        """
        Parse models string in various formats
        """
        if not models_str:
            return []

        if models_str.startswith('[') and models_str.startswith(']'):
            try:
                # Parse as JSON array
                return json.loads(models_str)
            except json.JSONDecodeError:
                pass

        # Handle manually quoted array: ["model1","model2"]
        if '"' in models_str or "'" in models_str:
            try:
                models_str = models_str.strip('[]')
                models = []
                for model in models_str.split(','):
                    model = model.strip().strip('"').strip("'")
                    if model:
                        models.append(model)
                return models
            except Exception:
                pass

        # Handle comma-separated list
        if ',' in models_str:
            return [m.strip() for m in models_str.split(',') if m.strip()]

        # Single model
        return [models_str] if models_str.strip() else []

    def get_primary_provider(self, env_vars: Dict[str, str]) -> Tuple[bool, ProviderConfig, List[str]]:
        """
        Get the primary provider (lowest priority number)
        """
        success, providers, errors = self.load_providers(env_vars)
        
        if not success or not providers:
            return False, None, errors
        
        return True, providers[0], errors

    def migrate_to_unified_format(self, env_vars: Dict[str, str]) -> Dict[str, List[str]]:
        """
        Generate migration commands to convert to unified format
        """
        migration_commands = []
        success, providers, errors = self.load_providers(env_vars)
        
        if not success or not providers:
            return {"success": [], "errors": ["No providers to migrate"] + errors}

        # Get the primary provider
        primary_provider = providers[0]
        provider_key = primary_provider.name.lower().replace(' ', '_')
        
        migration_commands.append("# Migration from Individual Environment format to Unified format")
        migration_commands.append("# Remove old individual variables:")
        migration_commands.append("")
        
        # Add commands to remove old variables
        url_var = f"{provider_key.upper()}_API_URL"
        key_var = f"{provider_key.upper()}_API_KEY"
        models_var = f"{provider_key.upper()}_MODELS"
        
        for var in [url_var, key_var, models_var]:
            if var in env_vars:
                migration_commands.append(f"# {var}={env_vars[var]}")
        
        migration_commands.append("")
        migration_commands.append("# Add new unified variables:")
        migration_commands.append(f"PROVIDER_API_URL={primary_provider.base_url}")
        migration_commands.append(f"PROVIDER_API_KEY={primary_provider.api_key}")
        
        if primary_provider.models:
            migration_commands.append(f"PROVIDER_MODEL={primary_provider.models[0]}")
        elif primary_provider.model:
            migration_commands.append(f"PROVIDER_MODEL={primary_provider.model}")
        
        if len(primary_provider.models) > 1:
            migration_commands.append(f"# Additional models available: {', '.join(primary_provider.models[1:])}")
        
        return {"success": migration_commands, "errors": []}

    def get_format_examples(self) -> Dict[str, List[str]]:
        """
        Get example configurations for this format
        """
        return {
            "local_providers": [
                [
                    "OLLAMA_API_URL=http://localhost:11434",
                    "OLLAMA_API_KEY=",
                    "OLLAMA_MODELS=[\"llama2\", \"mistral\", \"codellama\"]"
                ],
                [
                    "LM_STUDIO_API_URL=http://localhost:1234/v1",
                    "LM_STUDIO_API_KEY=",
                    "LM_STUDIO_MODELS=[\"microsoft/DialoGPT-medium\", \"microsoft/DialoGPT-large\"]"
                ]
            ],
            "cloud_providers": [
                [
                    "OPENROUTER_API_URL=https://openrouter.ai/api/v1",
                    "OPENROUTER_API_KEY=your-openrouter-key",
                    "OPENROUTER_MODELS=[\"openai/gpt-3.5-turbo\", \"anthropic/claude-3-haiku\"]"
                ],
                [
                    "OPENAI_API_URL=https://api.openai.com/v1",
                    "OPENAI_API_KEY=your-openai-key",
                    "OPENAI_MODELS=[\"gpt-3.5-turbo\", \"gpt-4\"]"
                ],
                [
                    "ANTHROPIC_API_URL=https://api.anthropic.com",
                    "ANTHROPIC_API_KEY=your-anthropic-key",
                    "ANTHROPIC_MODELS=[\"claude-3-haiku-20240307\", \"claude-3-sonnet-20240229\"]"
                ]
            ]
        }

    def validate_provider(self, provider_config: ProviderConfig) -> Tuple[bool, List[str]]:
        """
        Validate a specific provider configuration
        """
        errors = []

        # Validate base URL
        if not provider_config.base_url:
            errors.append("API URL is required")
        elif not (provider_config.base_url.startswith('http://') or provider_config.base_url.startswith('https://')):
            errors.append("API URL must start with http:// or https://")

        # Validate API key (not required for local services)
        if not provider_config.is_local and not provider_config.api_key:
            errors.append("API key is required for cloud services")

        # Validate models
        if not provider_config.models and not provider_config.model:
            errors.append("At least one model is required")

        return len(errors) == 0, errors

    def get_provider_config_summary(self, env_vars: Dict[str, str]) -> Dict[str, Dict]:
        """
        Get a summary of all provider configurations
        """
        summary = {}

        for var_name, value in env_vars.items():
            if var_name.endswith('_API_URL'):
                provider_name = var_name[:-9].lower()
                
                # Get all variables for this provider
                url_var = f"{provider_name.upper()}_API_URL"
                key_var = f"{provider_name.upper()}_API_KEY"
                models_var = f"{provider_name.upper()}_MODELS"

                summary[provider_name] = {
                    "api_url": env_vars.get(url_var, ""),
                    "api_key": env_vars.get(key_var, ""),
                    "models_str": env_vars.get(models_var, ""),
                    "models": self._parse_models(env_vars.get(models_var, ""), provider_name),
                    "configured": bool(env_vars.get(url_var, ""))
                }

        return summary