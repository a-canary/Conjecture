"""
Simple Provider Adapter
Handles PROVIDER_[NAME]=[BASE_URL],[API_KEY],[MODEL],[PROTOCOL] format
Example: PROVIDER_OLLAMA=http://localhost:11434,,llama2,ollama
"""

import os
import json
from typing import Dict, List, Tuple
from .base_adapter import BaseAdapter, ProviderConfig, ValidationResult, FormatPriority

class SimpleProviderAdapter(BaseAdapter):
    """
    Adapter for the simple provider format:
    PROVIDER_[NAME]=[BASE_URL],[API_KEY],[MODEL],[PROTOCOL]
    
    This format allows configuring multiple providers with priority-based selection
    """

    def __init__(self):
        super().__init__(
            format_type="simple_provider",
            priority=FormatPriority.HIGH  # High priority for backward compatibility
        )

    def detect_format(self, env_vars: Dict[str, str]) -> bool:
        """
        Detect if environment contains PROVIDER_[NAME] variables
        """
        provider_vars = [k for k in env_vars.keys() if k.startswith("PROVIDER_") and k.count('_') == 1]
        return len(provider_vars) > 0

    def validate_format(self, env_vars: Dict[str, str]) -> Tuple[bool, List[str]]:
        """
        Validate the format structure
        """
        errors = []
        provider_vars = [k for k in env_vars.keys() if k.startswith("PROVIDER_") and k.count('_') == 1]
        
        if not provider_vars:
            errors.append("No PROVIDER_[NAME] variables found")
            return False, errors

        for var in provider_vars:
            value = env_vars[var]
            if not value:
                errors.append(f"{var} is empty")
                continue

            # Parse the value
            parts = [part.strip() for part in value.split(',')]
            if len(parts) < 4:
                errors.append(f"{var} should have 4 parts: BASE_URL,API_KEY,MODEL,PROTOCOL")
                continue

            base_url, api_key, model, protocol = parts[:4]

            # Validate base URL
            if not base_url:
                errors.append(f"{var}: BASE_URL is required")
            elif not base_url.startswith(('http://', 'https://')):
                errors.append(f"{var}: BASE_URL must start with http:// or https://")

            # Validate protocol
            valid_protocols = ['openai', 'anthropic', 'google', 'cohere', 'ollama']
            if protocol not in valid_protocols:
                errors.append(f"{var}: PROTOCOL must be one of {valid_protocols}")

            # Validate model
            if not model:
                errors.append(f"{var}: MODEL is required")

        return len(errors) == 0, errors

    def load_providers(self, env_vars: Dict[str, str]) -> Tuple[bool, List[ProviderConfig], List[str]]:
        """
        Load providers from environment variables
        """
        providers = []
        errors = []

        # Provider definitions with priorities and properties
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

        for key, value in env_vars.items():
            if key.startswith("PROVIDER_") and value:
                provider_name = key[9:].lower()  # Remove "PROVIDER_" prefix

                if provider_name in provider_configs:
                    try:
                        # Parse: BASE_URL,API_KEY,MODEL,PROTOCOL
                        parts = [part.strip() for part in value.split(',')]

                        if len(parts) >= 4:
                            base_url, api_key, model, protocol = parts[:4]

                            # Normalize base URL
                            base_url = self.normalize_url(base_url)

                            # Get provider info from URL or use predefined
                            provider_info = self.detect_provider_type(base_url)
                            
                            # Create provider config
                            provider_config = self.create_provider_config(
                                name=provider_info["name"],
                                base_url=base_url,
                                api_key=api_key,
                                model=model,
                                protocol=protocol,
                                priority=provider_configs[provider_name]["priority"],
                                is_local=provider_configs[provider_name]["is_local"]
                            )
                            
                            providers.append(provider_config)

                        else:
                            errors.append(f"{key}: Invalid format - expected 4 comma-separated values")

                    except Exception as e:
                        errors.append(f"Error parsing {key}: {str(e)}")

        # Sort by priority
        providers.sort(key=lambda p: p.priority)

        return len(providers) > 0, providers, errors

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
        
        migration_commands.append("# Migration from Simple Provider format to Unified format")
        migration_commands.append("# Remove old PROVIDER_[NAME] variables:")
        migration_commands.append("")
        
        # Add commands to remove old variables
        for key in env_vars.keys():
            if key.startswith("PROVIDER_"):
                migration_commands.append(f"# {key}={env_vars[key]}")
        
        migration_commands.append("")
        migration_commands.append("# Add new unified variables:")
        migration_commands.append(f"PROVIDER_API_URL={primary_provider.base_url}")
        migration_commands.append(f"PROVIDER_API_KEY={primary_provider.api_key}")
        migration_commands.append(f"PROVIDER_MODEL={primary_provider.model}")
        
        if primary_provider.protocol:
            migration_commands.append(f"# Original protocol: {primary_provider.protocol}")
        
        return {"success": migration_commands, "errors": []}

    def get_format_examples(self) -> Dict[str, List[str]]:
        """
        Get example configurations for this format
        """
        return {
            "local_providers": [
                "PROVIDER_OLLAMA=http://localhost:11434,,llama2,ollama",
                "PROVIDER_LM_STUDIO=http://localhost:1234/v1,,microsoft/DialoGPT-medium,openai"
            ],
            "cloud_providers": [
                "PROVIDER_OPENROUTER=https://openrouter.ai/api/v1,your-openrouter-key,openai/gpt-3.5-turbo,openai",
                "PROVIDER_OPENAI=https://api.openai.com/v1,your-openai-key,gpt-3.5-turbo,openai",
                "PROVIDER_ANTHROPIC=https://api.anthropic.com,your-anthropic-key,claude-3-haiku-20240307,anthropic",
                "PROVIDER_GOOGLE=https://generativelanguage.googleapis.com,your-google-key,gemini-pro,google",
                "PROVIDER_GROQ=https://api.groq.com/openai/v1,your-groq-key,llama3-8b-8192,openai",
                "PROVIDER_COHERE=https://api.cohere.ai/v1,your-cohere-key,command,cohere"
            ],
            "specialized": [
                "PROVIDER_CHUTES=https://api.chutes.ai/v1,your-chutes-key,chutes-gpt-3.5-turbo,openai"
            ]
        }

    def validate_provider(self, provider_config: ProviderConfig) -> Tuple[bool, List[str]]:
        """
        Validate a specific provider configuration
        """
        errors = []

        # Validate base URL
        if not provider_config.base_url:
            errors.append("Base URL is required")
        elif not (provider_config.base_url.startswith('http://') or provider_config.base_url.startswith('https://')):
            errors.append("Base URL must start with http:// or https://")

        # Validate API key (not required for local services)
        if not provider_config.is_local and not provider_config.api_key:
            errors.append("API key is required for cloud services")

        # Validate model
        if not provider_config.model:
            errors.append("Model name is required")

        # Validate protocol
        valid_protocols = ['openai', 'anthropic', 'google', 'cohere', 'ollama']
        if provider_config.protocol and provider_config.protocol not in valid_protocols:
            errors.append(f"Protocol must be one of {valid_protocols}")

        return len(errors) == 0, errors