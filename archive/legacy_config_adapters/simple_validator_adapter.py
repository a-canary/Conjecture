"""
Simple Validator Adapter
Handles format: individual vars per provider like OLLAMA_ENDPOINT
Example: OLLAMA_ENDPOINT=http://localhost:11434, OLLAMA_MODEL=llama2
"""

import os
from typing import Dict, List, Tuple
from .base_adapter import BaseAdapter, ProviderConfig, ValidationResult, FormatPriority

class SimpleValidatorAdapter(BaseAdapter):
    """
    Adapter for the simple validator format:
    Individual variables per provider (OLLAMA_ENDPOINT, OPENAI_API_KEY, etc.)
    
    This format uses specific variable names for each provider
    """

    def __init__(self):
        super().__init__(
            format_type="simple_validator",
            priority=FormatPriority.LOW  # Lowest priority (legacy format)
        )

        # Define provider configurations with variable mappings
        self.provider_configs = {
            "ollama": {
                "priority": 1,
                "is_local": True,
                "vars": {
                    "endpoint": "OLLAMA_ENDPOINT",
                    "model": "OLLAMA_MODEL"
                },
                "default_model": "llama2",
                "name": "Ollama"
            },
            "lm_studio": {
                "priority": 2,
                "is_local": True,
                "vars": {
                    "endpoint": "LM_STUDIO_ENDPOINT",
                    "model": "LM_STUDIO_MODEL"
                },
                "default_model": "microsoft/DialoGPT-medium",
                "name": "LM Studio"
            },
            "openai": {
                "priority": 3,
                "is_local": False,
                "vars": {
                    "api_key": "OPENAI_API_KEY",
                    "model": "OPENAI_MODEL"
                },
                "default_model": "gpt-3.5-turbo",
                "name": "OpenAI",
                "default_endpoint": "https://api.openai.com/v1"
            },
            "anthropic": {
                "priority": 4,
                "is_local": False,
                "vars": {
                    "api_key": "ANTHROPIC_API_KEY",
                    "model": "ANTHROPIC_MODEL"
                },
                "default_model": "claude-3-haiku-20240307",
                "name": "Anthropic Claude",
                "default_endpoint": "https://api.anthropic.com"
            },
            "google": {
                "priority": 5,
                "is_local": False,
                "vars": {
                    "api_key": "GOOGLE_API_KEY",
                    "model": "GOOGLE_MODEL"
                },
                "default_model": "gemini-pro",
                "name": "Google Gemini",
                "default_endpoint": "https://generativelanguage.googleapis.com"
            },
            "cohere": {
                "priority": 6,
                "is_local": False,
                "vars": {
                    "api_key": "COHERE_API_KEY",
                    "model": "COHERE_MODEL"
                },
                "default_model": "command",
                "name": "Cohere",
                "default_endpoint": "https://api.cohere.ai/v1"
            },
            "chutes": {
                "priority": 2,
                "is_local": False,
                "vars": {
                    "api_key": "CHUTES_API_KEY",
                    "base_url": "CHUTES_BASE_URL",
                    "model": "CHUTES_MODEL"
                },
                "default_model": "chutes-gpt-3.5-turbo",
                "name": "Chutes.ai",
                "default_endpoint": "https://api.chutes.ai/v1"
            },
            "openrouter": {
                "priority": 3,
                "is_local": False,
                "vars": {
                    "api_key": "OPENROUTER_API_KEY",
                    "base_url": "OPENROUTER_BASE_URL",
                    "model": "OPENROUTER_MODEL"
                },
                "default_model": "openai/gpt-3.5-turbo",
                "name": "OpenRouter",
                "default_endpoint": "https://openrouter.ai/api/v1"
            },
            "groq": {
                "priority": 4,
                "is_local": False,
                "vars": {
                    "api_key": "GROQ_API_KEY",
                    "base_url": "GROQ_BASE_URL",
                    "model": "GROQ_MODEL"
                },
                "default_model": "llama3-8b-8192",
                "name": "Groq",
                "default_endpoint": "https://api.groq.com/openai/v1"
            }
        }

    def detect_format(self, env_vars: Dict[str, str]) -> bool:
        """
        Detect if environment contains simple validator format variables
        """
        # Look for any of the known provider variables
        for provider_config in self.provider_configs.values():
            for var_name in provider_config["vars"].values():
                if var_name in env_vars and env_vars[var_name]:
                    return True
        return False

    def validate_format(self, env_vars: Dict[str, str]) -> Tuple[bool, List[str]]:
        """
        Validate the format structure
        """
        errors = []

        for provider_key, config in self.provider_configs.items():
            provider_errors = []
            
            # Check endpoint/API key requirement
            if "endpoint" in config["vars"]:
                endpoint_var = config["vars"]["endpoint"]
                endpoint_value = env_vars.get(endpoint_var, "")
                
                if not endpoint_value:
                    # Skip if other variables not set
                    if not any(env_vars.get(var, "") for var in config["vars"].values() if var != endpoint_var):
                        continue
                    
                    provider_errors.append(f"{endpoint_var} is required")
                elif not endpoint_value.startswith(('http://', 'https://')):
                    provider_errors.append(f"{endpoint_var} must start with http:// or https://")

            # Check API key requirement for non-local providers
            if not config["is_local"] and "api_key" in config["vars"]:
                api_key_var = config["vars"]["api_key"]
                api_key_value = env_vars.get(api_key_var, "")
                
                # Only validate API key if other provider vars are set
                other_vars_set = any(
                    env_vars.get(var, "") 
                    for key, var in config["vars"].items() 
                    if key != "api_key" and env_vars.get(var, "")
                )
                
                if other_vars_set and not api_key_value:
                    provider_errors.append(f"{api_key_var} is required for cloud services")

            # Add provider-specific errors
            errors.extend(provider_errors)

        return len(errors) == 0, errors

    def load_providers(self, env_vars: Dict[str, str]) -> Tuple[bool, List[ProviderConfig], List[str]]:
        """
        Load providers from environment variables
        """
        providers = []
        errors = []

        for provider_key, config in self.provider_configs.items():
            provider_vars = {}
            
            # Collect all variables for this provider
            for var_key, env_var_name in config["vars"].items():
                value = env_vars.get(env_var_name, "")
                if value:
                    provider_vars[var_key] = value

            # Skip if no variables are set for this provider
            if not provider_vars:
                continue

            try:
                # Get base URL
                if "endpoint" in provider_vars:
                    base_url = self.normalize_url(provider_vars["endpoint"])
                elif "base_url" in provider_vars:
                    base_url = self.normalize_url(provider_vars["base_url"])
                else:
                    base_url = self.normalize_url(config.get("default_endpoint", ""))

                if not base_url:
                    errors.append(f"{config['name']}: No endpoint or base URL configured")
                    continue

                # Get API key
                api_key = provider_vars.get("api_key", "")

                # Get model
                model = provider_vars.get("model", config["default_model"])

                # Detect provider type or use known type
                provider_info = self.detect_provider_type(base_url)
                if provider_info["key"] == "unknown":
                    provider_info = {
                        "name": config["name"],
                        "key": provider_key,
                        "is_local": config["is_local"],
                        "protocol": "openai",  # Default protocol
                        "default_model": config["default_model"]
                    }

                # Create provider config
                provider_config = self.create_provider_config(
                    name=provider_info["name"],
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    protocol=provider_info["protocol"],
                    priority=config["priority"],
                    is_local=config["is_local"]
                )

                providers.append(provider_config)

            except Exception as e:
                errors.append(f"Error parsing {config['name']} configuration: {str(e)}")

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
        
        migration_commands.append("# Migration from Simple Validator format to Unified format")
        migration_commands.append("# Remove old provider-specific variables:")
        migration_commands.append("")
        
        # Find which provider we're migrating from
        for provider_key, config in self.provider_configs.items():
            if config["name"] == primary_provider.name:
                # Add commands to remove old variables
                for var_key, env_var_name in config["vars"].items():
                    if env_var_name in env_vars:
                        migration_commands.append(f"# {env_var_name}={env_vars[env_var_name]}")
                break
        
        migration_commands.append("")
        migration_commands.append("# Add new unified variables:")
        migration_commands.append(f"PROVIDER_API_URL={primary_provider.base_url}")
        migration_commands.append(f"PROVIDER_API_KEY={primary_provider.api_key}")
        migration_commands.append(f"PROVIDER_MODEL={primary_provider.model}")
        
        return {"success": migration_commands, "errors": []}

    def get_format_examples(self) -> Dict[str, List[str]]:
        """
        Get example configurations for this format
        """
        return {
            "local_providers": [
                [
                    "# Ollama local service",
                    "OLLAMA_ENDPOINT=http://localhost:11434",
                    "OLLAMA_MODEL=llama2"
                ],
                [
                    "# LM Studio local service",
                    "LM_STUDIO_ENDPOINT=http://localhost:1234/v1",
                    "LM_STUDIO_MODEL=microsoft/DialoGPT-medium"
                ]
            ],
            "cloud_providers": [
                [
                    "# OpenAI cloud service",
                    "OPENAI_API_KEY=sk-your-openai-key-here",
                    "OPENAI_MODEL=gpt-3.5-turbo"
                ],
                [
                    "# Anthropic Claude",
                    "ANTHROPIC_API_KEY=sk-ant-your-key-here",
                    "ANTHROPIC_MODEL=claude-3-haiku-20240307"
                ],
                [
                    "# Google Gemini",
                    "GOOGLE_API_KEY=your-google-key-here",
                    "GOOGLE_MODEL=gemini-pro"
                ],
                [
                    "# Cohere",
                    "COHERE_API_KEY=your-cohere-key-here",
                    "COHERE_MODEL=command"
                ]
            ],
            "multi_platform": [
                [
                    "# OpenRouter (access multiple models)",
                    "OPENROUTER_API_KEY=sk-or-your-key-here",
                    "OPENROUTER_BASE_URL=https://openrouter.ai/api/v1",
                    "OPENROUTER_MODEL=openai/gpt-3.5-turbo"
                ],
                [
                    "# Chutes.ai (optimized service)",
                    "CHUTES_API_KEY=sk-chutes-your-key-here",
                    "CHUTES_BASE_URL=https://api.chutes.ai/v1",
                    "CHUTES_MODEL=chutes-gpt-3.5-turbo"
                ],
                [
                    "# Groq (ultra-fast)",
                    "GROQ_API_KEY=your-groq-key-here",
                    "GROQ_BASE_URL=https://api.groq.com/openai/v1",
                    "GROQ_MODEL=llama3-8b-8192"
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
            errors.append("Endpoint or base URL is required")
        elif not (provider_config.base_url.startswith('http://') or provider_config.base_url.startswith('https://')):
            errors.append("Endpoint or base URL must start with http:// or https://")

        # Validate API key (not required for local services)
        if not provider_config.is_local and not provider_config.api_key:
            errors.append("API key is required for cloud services")

        # Validate model
        if not provider_config.model:
            errors.append("Model name is required")

        return len(errors) == 0, errors

    def get_provider_variables_mapping(self) -> Dict[str, Dict[str, str]]:
        """
        Get the mapping of provider names to their environment variables
        """
        mapping = {}
        
        for provider_key, config in self.provider_configs.items():
            mapping[provider_key] = {
                "name": config["name"],
                "type": "local" if config["is_local"] else "cloud",
                "priority": config["priority"],
                "variables": config["vars"],
                "default_model": config["default_model"],
                "default_endpoint": config.get("default_endpoint", "")
            }
        
        return mapping

    def detect_configured_providers(self, env_vars: Dict[str, str]) -> List[Dict]:
        """
        Detect which providers are configured in the environment
        """
        configured_providers = []

        for provider_key, config in self.provider_configs.items():
            # Check if any variables for this provider are set
            configured_vars = {
                var_key: env_vars.get(env_var_name, "")
                for var_key, env_var_name in config["vars"].items()
                if env_vars.get(env_var_name, "")
            }

            if configured_vars:
                # Determine configuration completeness
                required_vars = []
                if config["is_local"]:
                    required_vars = ["endpoint"]
                else:
                    required_vars = ["api_key"]

                is_complete = all(
                    any(configured_vars.get(var) for var in required_vars)
                )

                configured_providers.append({
                    "key": provider_key,
                    "name": config["name"],
                    "type": "local" if config["is_local"] else "cloud",
                    "priority": config["priority"],
                    "is_complete": is_complete,
                    "configured_vars": configured_vars,
                    "priority": config["priority"]
                })

        # Sort by priority
        configured_providers.sort(key=lambda p: p["priority"])

        return configured_providers