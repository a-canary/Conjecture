"""
Enhanced Provider Configuration Management
Provides utilities for managing enhanced LLM provider configurations
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from .unified_config import get_config
from ..processing.enhanced_llm_router import ProviderConfig, RoutingStrategy


class EnhancedProviderConfig:
    """Enhanced provider configuration management"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize enhanced provider configuration manager
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = Path(config_path) if config_path else None
        self.providers: Dict[str, ProviderConfig] = {}
        self.routing_strategy = RoutingStrategy.PRIORITY
        self.load_configuration()
    
    def load_configuration(self):
        """Load configuration from file or unified config"""
        try:
            # Try to load from specified path first
            if self.config_path and self.config_path.exists():
                self._load_from_file(self.config_path)
            else:
                # Load from unified config and enhance it
                self._load_from_unified_config()
            
            print(f"[Config] Loaded {len(self.providers)} enhanced provider configurations")
            
        except Exception as e:
            print(f"[Config] Failed to load configuration: {e}")
            self.providers = {}
    
    def _load_from_file(self, file_path: Path):
        """Load configuration from JSON file"""
        with open(file_path, 'r') as f:
            config_data = json.load(f)
        
        self._parse_config_data(config_data)
    
    def _load_from_unified_config(self):
        """Load and enhance configuration from unified config"""
        try:
            unified_config = get_config()
            config_data = {
                "providers": {},
                "routing_strategy": unified_config.get("routing_strategy", "priority"),
                "global_settings": {
                    "max_retries": unified_config.get("max_retries", 3),
                    "timeout": unified_config.get("timeout", 60),
                    "health_check_interval": unified_config.get("health_check_interval", 300),
                    "health_check_timeout": unified_config.get("health_check_timeout", 10)
                }
            }
            
            # Convert providers to enhanced format
            if isinstance(unified_config.providers, list):
                for provider in unified_config.providers:
                    config_data["providers"][provider["name"]] = self._enhance_provider_config(provider)
            else:
                for name, provider in unified_config.providers.items():
                    enhanced_provider = {
                        "name": name,
                        "url": provider.get("url", ""),
                        "api_key": provider.get("key", provider.get("api", "")),
                        "model": provider.get("model", ""),
                        "priority": provider.get("priority", 999)
                    }
                    config_data["providers"][name] = self._enhance_provider_config(enhanced_provider)
            
            self._parse_config_data(config_data)
            
        except Exception as e:
            print(f"[Config] Failed to load from unified config: {e}")
    
    def _enhance_provider_config(self, provider: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance provider configuration with default settings"""
        enhanced = provider.copy()
        
        # Add enhanced defaults
        enhanced.setdefault("max_retries", 3)
        enhanced.setdefault("timeout", 60)
        enhanced.setdefault("enabled", True)
        enhanced.setdefault("max_concurrent_requests", 10)
        enhanced.setdefault("health_check_interval", 300)
        enhanced.setdefault("health_check_timeout", 10)
        
        # Detect local providers
        url = enhanced.get("url", "")
        local_indicators = ["localhost", "127.0.0.1", "0.0.0.0"]
        enhanced["is_local"] = any(indicator in url.lower() for indicator in local_indicators)
        
        # Set provider-specific defaults
        provider_name = enhanced.get("name", "").lower()
        if "openrouter" in provider_name:
            enhanced.setdefault("timeout", 120)  # Longer timeout for OpenRouter
        elif "chutes" in provider_name or "z.ai" in provider_name:
            enhanced.setdefault("temperature", 0.7)  # Default temperature for z.ai
        elif "ollama" in provider_name or "lm_studio" in provider_name:
            enhanced.setdefault("timeout", 30)  # Shorter timeout for local providers
            enhanced.setdefault("max_retries", 2)  # Fewer retries for local
        
        return enhanced
    
    def _parse_config_data(self, config_data: Dict[str, Any]):
        """Parse configuration data into provider configs"""
        # Parse routing strategy
        strategy_str = config_data.get("routing_strategy", "priority")
        try:
            self.routing_strategy = RoutingStrategy(strategy_str)
        except ValueError:
            print(f"[Config] Invalid routing strategy: {strategy_str}, using priority")
            self.routing_strategy = RoutingStrategy.PRIORITY
        
        # Parse global settings
        global_settings = config_data.get("global_settings", {})
        
        # Parse providers
        providers_data = config_data.get("providers", {})
        for name, provider_data in providers_data.items():
            try:
                # Apply global settings as defaults
                enhanced_data = global_settings.copy()
                enhanced_data.update(provider_data)
                enhanced_data["name"] = name
                
                # Create ProviderConfig
                provider_config = ProviderConfig(
                    name=enhanced_data.get("name", name),
                    url=enhanced_data.get("url", ""),
                    api_key=enhanced_data.get("api_key", ""),
                    model=enhanced_data.get("model", ""),
                    priority=enhanced_data.get("priority", 999),
                    max_retries=enhanced_data.get("max_retries", 3),
                    timeout=enhanced_data.get("timeout", 60),
                    is_local=enhanced_data.get("is_local", False),
                    enabled=enhanced_data.get("enabled", True),
                    max_concurrent_requests=enhanced_data.get("max_concurrent_requests", 10),
                    health_check_interval=enhanced_data.get("health_check_interval", 300),
                    health_check_timeout=enhanced_data.get("health_check_timeout", 10),
                    temperature=enhanced_data.get("temperature"),
                    max_tokens=enhanced_data.get("max_tokens"),
                    top_p=enhanced_data.get("top_p")
                )
                
                self.providers[name] = provider_config
                
            except Exception as e:
                print(f"[Config] Failed to parse provider {name}: {e}")
    
    def save_configuration(self, file_path: Optional[Union[str, Path]] = None):
        """Save current configuration to file"""
        save_path = Path(file_path) if file_path else self.config_path
        if not save_path:
            save_path = Path("enhanced_provider_config.json")
        
        try:
            config_data = {
                "routing_strategy": self.routing_strategy.value,
                "global_settings": {
                    "max_retries": 3,
                    "timeout": 60,
                    "health_check_interval": 300,
                    "health_check_timeout": 10
                },
                "providers": {}
            }
            
            # Add providers
            for name, provider_config in self.providers.items():
                config_data["providers"][name] = provider_config.to_dict()
            
            # Save to file
            with open(save_path, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            
            print(f"[Config] Saved configuration to {save_path}")
            
        except Exception as e:
            print(f"[Config] Failed to save configuration: {e}")
    
    def add_provider(self, provider_config: ProviderConfig) -> bool:
        """Add or update a provider configuration"""
        try:
            self.providers[provider_config.name] = provider_config
            print(f"[Config] Added/updated provider: {provider_config.name}")
            return True
        except Exception as e:
            print(f"[Config] Failed to add provider {provider_config.name}: {e}")
            return False
    
    def remove_provider(self, provider_name: str) -> bool:
        """Remove a provider configuration"""
        if provider_name in self.providers:
            del self.providers[provider_name]
            print(f"[Config] Removed provider: {provider_name}")
            return True
        return False
    
    def get_provider(self, provider_name: str) -> Optional[ProviderConfig]:
        """Get a specific provider configuration"""
        return self.providers.get(provider_name)
    
    def get_all_providers(self) -> Dict[str, ProviderConfig]:
        """Get all provider configurations"""
        return self.providers.copy()
    
    def get_enabled_providers(self) -> Dict[str, ProviderConfig]:
        """Get only enabled provider configurations"""
        return {
            name: config for name, config in self.providers.items()
            if config.enabled
        }
    
    def get_providers_by_priority(self) -> List[ProviderConfig]:
        """Get providers sorted by priority"""
        enabled_providers = self.get_enabled_providers()
        return sorted(
            enabled_providers.values(),
            key=lambda p: p.priority
        )
    
    def set_routing_strategy(self, strategy: RoutingStrategy):
        """Set the routing strategy"""
        self.routing_strategy = strategy
        print(f"[Config] Routing strategy set to: {strategy.value}")
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate the current configuration"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "summary": {}
        }
        
        # Check if we have any providers
        if not self.providers:
            validation_result["valid"] = False
            validation_result["errors"].append("No providers configured")
            return validation_result
        
        # Check if we have any enabled providers
        enabled_providers = self.get_enabled_providers()
        if not enabled_providers:
            validation_result["valid"] = False
            validation_result["errors"].append("No enabled providers")
            return validation_result
        
        # Validate each provider
        for name, config in self.providers.items():
            provider_errors = []
            provider_warnings = []
            
            # Check required fields
            if not config.url:
                provider_errors.append("URL is required")
            if not config.model:
                provider_errors.append("Model is required")
            
            # Check URL format
            if config.url and not (config.url.startswith("http://") or config.url.startswith("https://")):
                provider_errors.append("URL must start with http:// or https://")
            
            # Check API key for remote providers
            if (not config.is_local and not config.api_key and config.enabled):
                provider_warnings.append("Remote provider without API key may not work")
            
            # Check priority
            if config.priority < 0:
                provider_errors.append("Priority must be non-negative")
            
            # Add to validation result
            if provider_errors:
                validation_result["errors"].append(f"Provider {name}: {'; '.join(provider_errors)}")
                validation_result["valid"] = False
            
            if provider_warnings:
                validation_result["warnings"].append(f"Provider {name}: {'; '.join(provider_warnings)}")
        
        # Add summary
        validation_result["summary"] = {
            "total_providers": len(self.providers),
            "enabled_providers": len(enabled_providers),
            "local_providers": len([p for p in self.providers.values() if p.is_local]),
            "remote_providers": len([p for p in self.providers.values() if not p.is_local]),
            "routing_strategy": self.routing_strategy.value
        }
        
        return validation_result
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration"""
        enabled_providers = self.get_enabled_providers()
        local_providers = [p for p in enabled_providers.values() if p.is_local]
        remote_providers = [p for p in enabled_providers.values() if not p.is_local]
        
        return {
            "total_providers": len(self.providers),
            "enabled_providers": len(enabled_providers),
            "local_providers": len(local_providers),
            "remote_providers": len(remote_providers),
            "routing_strategy": self.routing_strategy.value,
            "provider_names": list(self.providers.keys()),
            "enabled_provider_names": list(enabled_providers.keys()),
            "local_provider_names": [p.name for p in local_providers],
            "remote_provider_names": [p.name for p in remote_providers]
        }
    
    def create_sample_config(self) -> Dict[str, Any]:
        """Create a sample configuration with common providers"""
        return {
            "routing_strategy": "priority",
            "global_settings": {
                "max_retries": 3,
                "timeout": 60,
                "health_check_interval": 300,
                "health_check_timeout": 10
            },
            "providers": {
                "lm_studio": {
                    "name": "lm_studio",
                    "url": "http://localhost:1234",
                    "api_key": "",
                    "model": "llama-2-7b-chat",
                    "priority": 1,
                    "max_retries": 2,
                    "timeout": 30,
                    "enabled": True,
                    "max_concurrent_requests": 5,
                    "health_check_interval": 60,
                    "health_check_timeout": 5,
                    "temperature": 0.7,
                    "max_tokens": 2048
                },
                "openrouter": {
                    "name": "openrouter",
                    "url": "https://openrouter.ai/api/v1",
                    "api_key": "your-openrouter-api-key",
                    "model": "meta-llama/llama-3-8b-instruct",
                    "priority": 2,
                    "max_retries": 3,
                    "timeout": 120,
                    "enabled": True,
                    "max_concurrent_requests": 10,
                    "health_check_interval": 300,
                    "health_check_timeout": 10,
                    "temperature": 0.7,
                    "max_tokens": 4096
                },
                "z_ai": {
                    "name": "z_ai",
                    "url": "https://api.z.ai/api/coding/paas/v4",
                    "api_key": "your-z-ai-api-key",
                    "model": "glm-4.6",
                    "priority": 3,
                    "max_retries": 3,
                    "timeout": 90,
                    "enabled": True,
                    "max_concurrent_requests": 8,
                    "health_check_interval": 300,
                    "health_check_timeout": 15,
                    "temperature": 0.7,
                    "max_tokens": 8192
                }
            }
        }


# Global instance for easy access
_enhanced_provider_config = None


def get_enhanced_provider_config() -> EnhancedProviderConfig:
    """Get global enhanced provider configuration instance"""
    global _enhanced_provider_config
    if _enhanced_provider_config is None:
        _enhanced_provider_config = EnhancedProviderConfig()
    return _enhanced_provider_config


def create_sample_config_file(file_path: Union[str, Path]):
    """Create a sample configuration file"""
    config_manager = EnhancedProviderConfig()
    sample_config = config_manager.create_sample_config()
    
    save_path = Path(file_path)
    with open(save_path, 'w') as f:
        json.dump(sample_config, f, indent=2)
    
    print(f"[Config] Sample configuration saved to {save_path}")


if __name__ == "__main__":
    # Create sample configuration
    create_sample_config_file("enhanced_provider_config.sample.json")
    
    # Load and validate current configuration
    config_manager = get_enhanced_provider_config()
    validation = config_manager.validate_configuration()
    
    print("\n=== Configuration Validation ===")
    print(f"Valid: {validation['valid']}")
    if validation['errors']:
        print("Errors:")
        for error in validation['errors']:
            print(f"  - {error}")
    if validation['warnings']:
        print("Warnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    
    print(f"\n=== Configuration Summary ===")
    summary = validation['summary']
    for key, value in summary.items():
        print(f"{key}: {value}")