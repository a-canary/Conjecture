"""
Simplified Configuration Wizard for Conjecture
Guides users through setting up OpenAI-compatible providers
"""

import os
import json
import sys
from typing import Dict, List, Optional, Any
from pathlib import Path

from .simplified_config import (
    SimplifiedConfigManager, 
    ProviderConfig,
    get_config_manager
)


class SimplifiedConfigWizard:
    """Simplified configuration wizard for OpenAI-compatible providers"""

    def __init__(self):
        self.config_manager = get_config_manager()
        self.config = self.config_manager.config

    def run_interactive_setup(self) -> bool:
        """Run interactive configuration setup"""
        print("🚀 Conjecture Simplified Configuration Wizard")
        print("=" * 50)
        print("This wizard helps you configure OpenAI-compatible providers.")
        print("Supported providers: OpenAI, OpenRouter, Chutes.ai, LM Studio")
        print()

        # Get user preferences
        setup_type = self._get_setup_type()
        
        if setup_type == "auto":
            return self._auto_setup()
        elif setup_type == "manual":
            return self._manual_setup()
        elif setup_type == "quick":
            return self._quick_setup()
        else:
            print("❌ Invalid setup type selected")
            return False

    def _get_setup_type(self) -> str:
        """Get setup type from user"""
        print("Choose setup type:")
        print("1. Auto-detect local providers")
        print("2. Quick setup (common providers)")
        print("3. Manual setup (custom configuration)")
        print()

        while True:
            choice = input("Enter choice (1-3): ").strip()
            if choice == "1":
                return "auto"
            elif choice == "2":
                return "quick"
            elif choice == "3":
                return "manual"
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")

    def _auto_setup(self) -> bool:
        """Auto-detect and setup local providers"""
        print("\n🔍 Auto-detecting local providers...")
        
        detected_providers = []
        
        # Check for LM Studio
        lm_studio_url = "http://localhost:1234"
        if self._test_provider_connection(lm_studio_url):
            detected_providers.append(ProviderConfig(
                name="lm_studio",
                url=lm_studio_url,
                api="",
                model="ibm/granite-4-h-tiny",
                priority=1
            ))
            print(f"✅ LM Studio detected at {lm_studio_url}")
        
        # Check for Ollama (if it supports OpenAI-compatible endpoint)
        ollama_url = "http://localhost:11434"
        if self._test_provider_connection(f"{ollama_url}/v1"):
            detected_providers.append(ProviderConfig(
                name="ollama",
                url=f"{ollama_url}/v1",
                api="",
                model="llama2",
                priority=2
            ))
            print(f"✅ Ollama detected at {ollama_url}")
        
        if not detected_providers:
            print("❌ No local providers detected")
            print("Please ensure LM Studio or Ollama is running locally")
            return False
        
        # Add detected providers to config
        for provider in detected_providers:
            self.config_manager.add_provider(provider)
        
        # Save configuration
        self.config_manager.save_config()
        print(f"✅ Configuration saved with {len(detected_providers)} local providers")
        return True

    def _quick_setup(self) -> bool:
        """Quick setup with common cloud providers"""
        print("\n⚡ Quick setup - Common cloud providers")
        print("We'll configure popular cloud providers with placeholder API keys.")
        print("You'll need to edit the config file to add your actual API keys.")
        print()

        quick_providers = [
            ProviderConfig(
                name="openai",
                url="https://api.openai.com/v1",
                api="your-openai-api-key-here",
                model="gpt-3.5-turbo",
                priority=2
            ),
            ProviderConfig(
                name="openrouter",
                url="https://openrouter.ai/api/v1",
                api="your-openrouter-api-key-here",
                model="openai/gpt-3.5-turbo",
                priority=3
            ),
            ProviderConfig(
                name="chutes",
                url="https://llm.chutes.ai/v1",
                api="your-chutes-api-key-here",
                model="zai-org/GLM-4.6",
                priority=4
            )
        ]
        
        # Add providers to config
        for provider in quick_providers:
            self.config_manager.add_provider(provider)
        
        # Save configuration
        self.config_manager.save_config()
        print(f"✅ Configuration saved with {len(quick_providers)} cloud providers")
        print("\n📝️  Next steps:")
        print("1. Edit ~/.conjecture/config.json")
        print("2. Replace 'your-*-api-key-here' with actual API keys")
        print("3. Run 'conjecture validate' to test configuration")
        return True

    def _manual_setup(self) -> bool:
        """Manual setup with custom provider configuration"""
        print("\n⚙️  Manual provider configuration")
        print("Configure custom OpenAI-compatible providers.")
        print()

        providers = []
        
        while True:
            print(f"\nCurrent providers: {len(providers)}")
            action = input("Add provider (a) or finish (f): ").strip().lower()
            
            if action == 'f':
                if not providers:
                    print("❌ At least one provider is required")
                    continue
                break
            elif action == 'a':
                provider = self._get_manual_provider()
                if provider:
                    providers.append(provider)
                    print(f"✅ Added provider: {provider.name}")
            else:
                print("❌ Invalid action. Please enter 'a' or 'f'.")

        # Add providers to config
        for provider in providers:
            self.config_manager.add_provider(provider)
        
        # Save configuration
        self.config_manager.save_config()
        print(f"✅ Configuration saved with {len(providers)} custom providers")
        return True

    def _get_manual_provider(self) -> Optional[ProviderConfig]:
        """Get manual provider configuration from user"""
        try:
            print("\nEnter provider details:")
            
            name = input("Provider name (e.g., 'my_provider'): ").strip()
            if not name:
                print("❌ Provider name is required")
                return None
            
            url = input("API URL (e.g., 'https://api.example.com/v1'): ").strip()
            if not url or not url.startswith(('http://', 'https://')):
                print("❌ Valid URL is required")
                return None
            
            api_key = input("API key (press Enter for local providers): ").strip()
            model = input("Model name (e.g., 'gpt-3.5-turbo'): ").strip()
            if not model:
                print("❌ Model name is required")
                return None
            
            priority_str = input("Priority (1=highest, default=999): ").strip()
            try:
                priority = int(priority_str) if priority_str else 999
            except ValueError:
                priority = 999
            
            return ProviderConfig(
                name=name,
                url=url,
                api=api_key,
                model=model,
                priority=priority
            )
            
        except KeyboardInterrupt:
            print("\n❌ Provider setup cancelled")
            return None
        except Exception as e:
            print(f"❌ Error getting provider details: {e}")
            return None

    def _test_provider_connection(self, url: str) -> bool:
        """Test connection to a provider URL"""
        try:
            import requests
            response = requests.get(f"{url}/models", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def show_current_config(self):
        """Display current configuration"""
        print("\n📋 Current Configuration")
        print("=" * 50)
        
        if not self.config:
            print("❌ No configuration loaded")
            return
        
        summary = self.config_manager.get_config_summary()
        
        print(f"Total providers: {summary['total_providers']}")
        print(f"Local providers: {summary['local_providers']}")
        print(f"Cloud providers: {summary['cloud_providers']}")
        print(f"Config file: {summary['config_file']}")
        print()
        
        for provider in self.config_manager.get_available_providers():
            is_local = self.config_manager._is_provider_local(provider)
            status = "🟢 Local" if is_local else "🔵 Cloud"
            print(f"{status} {provider.name}")
            print(f"  URL: {provider.url}")
            print(f"  Model: {provider.model}")
            print(f"  API Key: {'Set' if provider.api else 'None'}")
            print(f"  Priority: {provider.priority}")
            print()

    def validate_current_config(self) -> bool:
        """Validate current configuration"""
        print("\n🔍 Validating Configuration")
        print("=" * 50)
        
        if not self.config:
            print("❌ No configuration loaded")
            return False
        
        validation_results = self.config_manager.validate_providers()
        
        all_valid = True
        for provider_name, result in validation_results.items():
            if result['valid']:
                status = "✅"
                issues = "None"
            else:
                status = "❌"
                issues = ", ".join(result['issues'])
                all_valid = False
            
            provider_type = "Local" if result['is_local'] else "Cloud"
            print(f"{status} {provider_name} ({provider_type})")
            print(f"  Issues: {issues}")
        
        if all_valid:
            print("\n✅ All providers are valid!")
        else:
            print("\n❌ Some providers have configuration issues")
            print("Please fix the issues above and re-run validation")
        
        return all_valid

    def test_providers(self) -> bool:
        """Test all configured providers"""
        print("\n🧪 Testing Providers")
        print("=" * 50)
        
        if not self.config:
            print("❌ No configuration loaded")
            return False
        
        from ..processing.simplified_llm_manager import SimplifiedLLMManager
        
        # Create manager and test providers
        manager = SimplifiedLLMManager([
            {
                "name": p.name,
                "url": p.url,
                "api": p.api,
                "model": p.model,
                "priority": p.priority
            }
            for p in self.config.providers
        ])
        
        health_status = manager.health_check()
        
        print(f"Total providers: {health_status['total_providers']}")
        print(f"Available providers: {health_status['available_providers']}")
        print(f"Failed providers: {len(health_status['failed_providers'])}")
        print()
        
        for provider_name, provider_health in health_status['providers'].items():
            if provider_health['status'] == 'healthy':
                status = "✅"
                error_msg = "None"
            else:
                status = "❌"
                error_msg = provider_health.get('error', 'Unknown error')
            
            print(f"{status} {provider_name}")
            print(f"  Model: {provider_health.get('model', 'Unknown')}")
            print(f"  Error: {error_msg}")
            print(f"  Last check: {provider_health.get('last_check', 'Unknown')}")
            print()
        
        return health_status['overall_status'] == 'healthy'

    def create_config_directory(self):
        """Create configuration directory"""
        config_dir = Path.home() / ".conjecture"
        config_dir.mkdir(exist_ok=True)
        
        # Create .gitignore if it doesn't exist
        gitignore_path = config_dir / ".gitignore"
        if not gitignore_path.exists():
            gitignore_path.write_text("config.json\n")
        
        print(f"✅ Configuration directory created: {config_dir}")


def main():
    """Main entry point for the configuration wizard"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Conjecture Simplified Configuration Wizard")
    parser.add_argument("action", choices=["setup", "show", "validate", "test", "init"], 
                       help="Action to perform")
    parser.add_argument("--config", help="Path to config file")
    
    args = parser.parse_args()
    
    # Override config path if provided
    if args.config:
        os.environ["CONJECTURE_CONFIG_PATH"] = args.config
    
    wizard = SimplifiedConfigWizard()
    
    try:
        if args.action == "init":
            wizard.create_config_directory()
            wizard.run_interactive_setup()
        elif args.action == "setup":
            wizard.run_interactive_setup()
        elif args.action == "show":
            wizard.show_current_config()
        elif args.action == "validate":
            wizard.validate_current_config()
        elif args.action == "test":
            wizard.test_providers()
        else:
            print(f"❌ Unknown action: {args.action}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n👋 Configuration cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Configuration wizard error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()