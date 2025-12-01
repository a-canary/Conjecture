"""
Unified Provider Adapter
Handles format: PROVIDER_API_URL, PROVIDER_API_KEY, PROVIDER_MODEL
Example: PROVIDER_API_URL=http://localhost:11434, PROVIDER_API_KEY=, PROVIDER_MODEL=llama2
"""

import os
from typing import Dict, List, Tuple
from .base_adapter import BaseAdapter, ProviderConfig, ValidationResult, FormatPriority

class UnifiedProviderAdapter(BaseAdapter):
    """
    Adapter for the unified provider format:
    PROVIDER_API_URL, PROVIDER_API_KEY, PROVIDER_MODEL
    
    This is the simplest format with a single active provider
    """

    def __init__(self):
        super().__init__(
            format_type="unified_provider",
            priority=FormatPriority.HIGHEST  # Highest priority for the recommended format
        )

    def detect_format(self, env_vars: Dict[str, str]) -> bool:
        """
        Detect if environment contains PROVIDER_* variables
        """
        url_set = "PROVIDER_API_URL" in env_vars and env_vars["PROVIDER_API_URL"]
        return url_set

    def validate_format(self, env_vars: Dict[str, str]) -> Tuple[bool, List[str]]:
        """
        Validate the format structure
        """
        errors = []

        api_url = env_vars.get("PROVIDER_API_URL", "")
        api_key = env_vars.get("PROVIDER_API_KEY", "")
        model = env_vars.get("PROVIDER_MODEL", "")

        # Validate API URL
        if not api_url:
            errors.append("PROVIDER_API_URL is required")
        elif not api_url.startswith(('http://', 'https://')):
            errors.append("PROVIDER_API_URL must start with http:// or https://")

        # Validate model
        if not model:
            errors.append("PROVIDER_MODEL is required")

        # Get provider info to check if API key is needed
        if api_url:
            provider_info = self.detect_provider_type(api_url)
            if not provider_info["is_local"] and not api_key:
                errors.append("PROVIDER_API_KEY is required for cloud services")

        return len(errors) == 0, errors

    def load_providers(self, env_vars: Dict[str, str]) -> Tuple[bool, List[ProviderConfig], List[str]]:
        """
        Load the active provider from environment variables
        """
        providers = []
        errors = []

        api_url = env_vars.get("PROVIDER_API_URL", "")
        api_key = env_vars.get("PROVIDER_API_KEY", "")
        model = env_vars.get("PROVIDER_MODEL", "")

        if not api_url:
            errors.append("PROVIDER_API_URL is not set")
            return False, providers, errors

        try:
            # Normalize base URL
            api_url = self.normalize_url(api_url)

            # Detect provider type from URL
            provider_info = self.detect_provider_type(api_url)

            # Create provider config
            provider_config = self.create_provider_config(
                name=provider_info["name"],
                base_url=api_url,
                api_key=api_key,
                model=model,
                protocol=provider_info["protocol"],
                priority=provider_info.get("priority", self.get_priority_for_provider(provider_info["key"])),
                is_local=provider_info["is_local"]
            )

            providers.append(provider_config)

        except Exception as e:
            errors.append(f"Error parsing unified provider configuration: {str(e)}")

        return len(providers) > 0, providers, errors

    def get_primary_provider(self, env_vars: Dict[str, str]) -> Tuple[bool, ProviderConfig, List[str]]:
        """
        Get the configured provider (there's only one in this format)
        """
        success, providers, errors = self.load_providers(env_vars)
        
        if not success or not providers:
            return False, None, errors
        
        # In unified format, there's only one provider
        return True, providers[0], errors

    def migrate_from_format(self, target_format: str, env_vars: Dict[str, str]) -> Dict[str, List[str]]:
        """
        Generate migration commands to convert from other formats to unified format
        """
        migration_commands = []
        
        if target_format == "simple_provider":
            from .simple_provider_adapter import SimpleProviderAdapter
            adapter = SimpleProviderAdapter()
            return adapter.migrate_to_unified_format(env_vars)
        elif target_format == "individual_env":
            from .individual_env_adapter import IndividualEnvAdapter
            adapter = IndividualEnvAdapter()
            return adapter.migrate_to_unified_format(env_vars)
        elif target_format == "simple_validator":
            from .simple_validator_adapter import SimpleValidatorAdapter
            adapter = SimpleValidatorAdapter()
            return adapter.migrate_to_unified_format(env_vars)
        else:
            return {"success": [], "errors": [f"Migration from {target_format} format not supported"]}

    def get_format_examples(self) -> Dict[str, List[str]]:
        """
        Get example configurations for this format
        """
        return {
            "local_providers": [
                [
                    "PROVIDER_API_URL=http://localhost:11434",
                    "PROVIDER_API_KEY=",
                    "PROVIDER_MODEL=llama2"
                ],
                [
                    "PROVIDER_API_URL=http://localhost:1234/v1",
                    "PROVIDER_API_KEY=",
                    "PROVIDER_MODEL=microsoft/DialoGPT-medium"
                ]
            ],
            "cloud_providers": [
                [
                    "PROVIDER_API_URL=https://openrouter.ai/api/v1",
                    "PROVIDER_API_KEY=your-openrouter-key",
                    "PROVIDER_MODEL=openai/gpt-3.5-turbo"
                ],
                [
                    "PROVIDER_API_URL=https://api.openai.com/v1",
                    "PROVIDER_API_KEY=your-openai-key",
                    "PROVIDER_MODEL=gpt-3.5-turbo"
                ],
                [
                    "PROVIDER_API_URL=https://api.anthropic.com",
                    "PROVIDER_API_KEY=your-anthropic-key",
                    "PROVIDER_MODEL=claude-3-haiku-20240307"
                ],
                [
                    "PROVIDER_API_URL=https://generativelanguage.googleapis.com",
                    "PROVIDER_API_KEY=your-google-key",
                    "PROVIDER_MODEL=gemini-pro"
                ],
                [
                    "PROVIDER_API_URL=https://api.groq.com/openai/v1",
                    "PROVIDER_API_KEY=your-groq-key",
                    "PROVIDER_MODEL=llama3-8b-8192"
                ],
                [
                    "PROVIDER_API_URL=https://api.cohere.ai/v1",
                    "PROVIDER_API_KEY=your-cohere-key",
                    "PROVIDER_MODEL=command"
                ]
            ],
            "specialized": [
                [
                    "PROVIDER_API_URL=https://api.chutes.ai/v1",
                    "PROVIDER_API_KEY=your-chutes-key",
                    "PROVIDER_MODEL=chutes-gpt-3.5-turbo"
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
            errors.append("PROVIDER_API_URL is required")
        elif not (provider_config.base_url.startswith('http://') or provider_config.base_url.startswith('https://')):
            errors.append("PROVIDER_API_URL must start with http:// or https://")

        # Validate API key (not required for local services)
        if not provider_config.is_local and not provider_config.api_key:
            errors.append("PROVIDER_API_KEY is required for cloud services")

        # Validate model
        if not provider_config.model:
            errors.append("PROVIDER_MODEL is required")

        return len(errors) == 0, errors

    def get_provider_recommendations(self, api_url: str) -> List[Dict[str, str]]:
        """
        Get model recommendations for a provider
        """
        provider_info = self.detect_provider_type(self.normalize_url(api_url))
        provider_key = provider_info["key"]
        
        recommendations = {
            "ollama": [
                {"model": "llama2", "description": "Good general purpose model"},
                {"model": "mistral", "description": "Fast and efficient"},
                {"model": "codellama", "description": "Specialized for code"},
                {"model": "llama3:8b", "description": "Latest Llama 3 8B model"}
            ],
            "lm_studio": [
                {"model": "microsoft/DialoGPT-medium", "description": "Good for conversations"},
                {"model": "microsoft/DialoGPT-large", "description": "Better conversations, slower"},
                {"model": "TheBloke/Llama-2-7B-Chat-GGUF", "description": "Popular Llama 2 model"}
            ],
            "openrouter": [
                {"model": "openai/gpt-3.5-turbo", "description": "Fast and reliable"},
                {"model": "openai/gpt-4", "description": "More capable, slower"},
                {"model": "anthropic/claude-3-haiku", "description": "Anthropic's fast model"},
                {"model": "anthropic/claude-3-sonnet", "description": "Good balance of speed/capability"},
                {"model": "meta-llama/llama-3-8b-instruct", "description": "Latest Llama 3"}
            ],
            "openai": [
                {"model": "gpt-3.5-turbo", "description": "Fast and cost-effective"},
                {"model": "gpt-4", "description": "Most capable"},
                {"model": "gpt-4-turbo", "description": "Faster GPT-4"}
            ],
            "anthropic": [
                {"model": "claude-3-haiku-20240307", "description": "Fast and affordable"},
                {"model": "claude-3-sonnet-20240229", "description": "Good balance"},
                {"model": "claude-3-opus-20240229", "description": "Most capable"}
            ],
            "google": [
                {"model": "gemini-pro", "description": "Google's capable model"},
                {"model": "gemini-pro-vision", "description": "Supports images"}
            ],
            "groq": [
                {"model": "llama3-8b-8192", "description": "Very fast Llama 3"},
                {"model": "llama3-70b-8192", "description": "Fast large model"},
                {"model": "mixtral-8x7b-32768", "description": "Fast mixture of experts"}
            ],
            "cohere": [
                {"model": "command", "description": "General purpose"},
                {"model": "command-nightly", "description": "Latest features"},
                {"model": "command-light", "description": "Fast and lightweight"}
            ],
            "chutes": [
                {"model": "chutes-gpt-3.5-turbo", "description": "Optimized GPT-3.5"},
                {"model": "openai/gpt-oss-20b", "description": "Open source large model"},
                {"model": "zai-org/GLM-4.5-Air", "description": "Chinese language model"}
            ]
        }
        
        return recommendations.get(provider_key, [
            {"model": "default", "description": "Default model for this provider"}
        ])

    def setup_wizard_guidance(self) -> Dict[str, List[str]]:
        """
        Get setup wizard guidance for this format
        """
        return {
            "steps": [
                "1. Choose your preferred AI provider from the examples below",
                "2. Copy the example configuration to your .env file",
                "3. Replace placeholder API keys with your actual keys",
                "4. Update the model name if desired",
                "5. Save the .env file and test your configuration"
            ],
            "next_steps": [
                "Run: python -m conjecture.config.unified_validator test",
                "Or use your CLI: python simple_cli.py config-status"
            ]
        }