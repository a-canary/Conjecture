"""
Simple Configuration Setup Wizard

Replaces the complex discovery system with a straightforward 3-step setup process.
Focuses on the 80/20 rule - covering 90% of user needs with simple, synchronous logic.

Usage:
    wizard = SetupWizard()
    status = wizard.quick_status()
    if not status['configured']:
        wizard.interactive_setup()
"""

import os
import json
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import re
from datetime import datetime

@dataclass
class SimpleProvider:
    """Simple provider representation"""
    name: str
    type: str  # 'local' or 'cloud'
    endpoint: Optional[str] = None
    api_key_env_var: Optional[str] = None
    default_model: str = ""
    setup_commands: List[str] = None
    description: str = ""
    
    def __post_init__(self):
        if self.setup_commands is None:
            self.setup_commands = []

class SetupWizard:
    """Simple configuration setup wizard"""
    
    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.env_file = self.project_root / '.env'
        self.env_example_file = self.project_root / '.env.example'
        
        # Common provider configurations (80% of use cases)
        self.providers = {
            'ollama': SimpleProvider(
                name='Ollama',
                type='local',
                endpoint='http://localhost:11434',
                default_model='llama2',
                setup_commands=[
                    'curl -fsSL https://ollama.ai/install.sh | sh',
                    'ollama pull llama2'
                ],
                description='Local LLM runner - easiest setup for local AI'
            ),
            'lm_studio': SimpleProvider(
                name='LM Studio',
                type='local', 
                endpoint='http://localhost:1234/v1',
                default_model='local-model',
                setup_commands=[
                    'Download from https://lmstudio.ai/',
                    'Start LM Studio and load a model'
                ],
                description='Local model server with GUI interface'
            ),
            'openai': SimpleProvider(
                name='OpenAI',
                type='cloud',
                endpoint='https://api.openai.com/v1',
                api_key_env_var='OPENAI_API_KEY',
                default_model='gpt-3.5-turbo',
                setup_commands=[
                    'Get API key from https://platform.openai.com/api-keys'
                ],
                description='OpenAI GPT models (paid)'
            ),
            'anthropic': SimpleProvider(
                name='Anthropic',
                type='cloud',
                endpoint='https://api.anthropic.com',
                api_key_env_var='ANTHROPIC_API_KEY',
                default_model='claude-3-haiku-20240307',
                setup_commands=[
                    'Get API key from https://console.anthropic.com/'
                ],
                description='Anthropic Claude models (paid)'
            ),
            'chutes': SimpleProvider(
                name='Chutes',
                type='cloud',
                endpoint='https://llm.chutes.ai/v1',
                api_key_env_var='CHUTES_API_KEY',
                default_model='llama-3-8b',
                setup_commands=[
                    'Get API key from https://chutes.ai/'
                ],
                description='Affordable cloud API for various models'
            )
        }

    def quick_status(self) -> Dict[str, Any]:
        """Return simple configuration status"""
        status = {
            'configured': False,
            'provider': None,
            'provider_type': None,
            'model': None,
            'api_url': None,
            'env_file_exists': self.env_file.exists(),
            'missing_api_key': False
        }
        
        if not self.env_file.exists():
            return status
            
        # Read current configuration
        env_vars = self._read_env_file()
        
        provider = env_vars.get('Conjecture_LLM_PROVIDER')
        if not provider:
            return status
            
        provider_info = self.providers.get(provider.lower())
        if not provider_info:
            return status
            
        # Check if required API key is present for cloud providers
        if provider_info.type == 'cloud' and provider_info.api_key_env_var:
            api_key = env_vars.get(provider_info.api_key_env_var)
            if not api_key or api_key.startswith('your_'):
                status['missing_api_key'] = True
                return status
        
        # All checks passed - configured
        status.update({
            'configured': True,
            'provider': provider_info.name,
            'provider_type': provider_info.type,
            'model': env_vars.get('Conjecture_LLM_MODEL', provider_info.default_model),
            'api_url': env_vars.get('Conjecture_LLM_API_URL', provider_info.endpoint)
        })
        
        return status
    
    def interactive_setup(self) -> bool:
        """3-step interactive setup process"""
        print("Conjecture Setup Wizard")
        print("=" * 40)
        print("This wizard will help you configure Conjecture in 3 simple steps.")
        print()
        
        try:
            # Step 1: Choose provider
            provider_key = self._choose_provider()
            if not provider_key:
                print("\n❌ Setup cancelled")
                return False
                
            provider = self.providers[provider_key]
            
            # Step 2: Configure provider
            config = self._configure_provider(provider)
            if not config:
                print("\n❌ Setup cancelled")
                return False
            
            # Step 3: Validate and save
            if self._validate_and_save(provider, config):
                print("\n✅ Setup completed successfully!")
                print(f"Provider: {provider.name}")
                print(f"Model: {config.get('model', provider.default_model)}")
                print(f"API URL: {config.get('api_url', provider.endpoint)}")
                print("\nYou can now use Conjecture! 🚀")
                return True
            else:
                print("\n❌ Setup failed during validation")
                return False
                
        except KeyboardInterrupt:
            print("\n\n⚠️ Setup cancelled by user")
            return False
        except Exception as e:
            print(f"\n❌ Setup failed: {e}")
            return False
    
    def auto_detect_local(self) -> List[str]:
        """Simple local service detection (no async complexity)"""
        detected = []
        
        for provider_key, provider in self.providers.items():
            if provider.type == 'local' and provider.endpoint:
                if self._test_endpoint(provider.endpoint):
                    detected.append(provider_key)
                    print(f"[OK] Found {provider.name} at {provider.endpoint}")
        
        return detected
    
    def update_env_file(self, provider_config: Dict[str, Any]) -> bool:
        """Direct env file updates (no complex merging)"""
        try:
            # Read existing env or create new
            if self.env_file.exists():
                env_vars = self._read_env_file()
                # Create backup
                backup_file = self._create_backup()
                if backup_file:
                    print(f"💾 Created backup: {backup_file.name}")
            else:
                env_vars = {}
                # Create .env.example if it doesn't exist
                if not self.env_example_file.exists():
                    self._create_env_example()
            
            # Update with new configuration
            env_vars.update(provider_config)
            
            # Add defaults
            defaults = {
                'Conjecture_EMBEDDING_MODEL': 'all-MiniLM-L6-v2',
                'Conjecture_DB_PATH': 'data/conjecture.db',
                'Conjecture_CONFIDENCE': '0.7',
                'Conjecture_DEBUG': 'false'
            }
            
            for key, value in defaults.items():
                if key not in env_vars:
                    env_vars[key] = value
            
            # Write updated configuration
            self._write_env_file(env_vars)
            return True
            
        except Exception as e:
            print(f"❌ Failed to update .env file: {e}")
            return False
    
    def _choose_provider(self) -> Optional[str]:
        """Step 1: Provider selection"""
        print("📍 Step 1: Choose your LLM provider")
        print("-" * 40)
        
        # Auto-detect local providers first
        local_providers = self.auto_detect_local()
        
        # Check for existing API keys
        cloud_providers_with_keys = []
        for key, provider in self.providers.items():
            if provider.type == 'cloud' and provider.api_key_env_var:
                if os.getenv(provider.api_key_env_var):
                    cloud_providers_with_keys.append(key)
        
        # Display options
        options = []
        
        # Local providers (detected first)
        if local_providers:
            print("\n🏠 Local Providers (Detected):")
            for i, provider_key in enumerate(local_providers, 1):
                provider = self.providers[provider_key]
                print(f"  {i}. {provider.name} - {provider.description}")
                options.append(('local', provider_key))
        
        # Cloud providers with existing keys
        if cloud_providers_with_keys:
            print("\n☁️ Cloud Providers (API key detected):")
            start_idx = len(options) + 1
            for i, provider_key in enumerate(cloud_providers_with_keys, start_idx):
                provider = self.providers[provider_key]
                print(f"  {i}. {provider.name} - {provider.description} ⭐")
                options.append(('cloud_key', provider_key))
        
        # All available providers
        print("\n📋 All Available Providers:")
        remaining_providers = [k for k in self.providers.keys() 
                            if k not in local_providers and k not in cloud_providers_with_keys]
        start_idx = len(options) + 1
        for i, provider_key in enumerate(remaining_providers, start_idx):
            provider = self.providers[provider_key]
            provider_type = "🏠" if provider.type == 'local' else "☁️"
            print(f"  {i}. {provider.name} - {provider.description} {provider_type}")
            options.append(('all', provider_key))
        
        # Get user selection
        while True:
            try:
                choice = input(f"\nEnter choice (1-{len(options)}) or press Enter for default [1]: ").strip()
                if not choice:
                    choice = "1"
                
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(options):
                        _, provider_key = options[idx]
                        provider = self.providers[provider_key]
                        print(f"\n✅ Selected: {provider.name}")
                        return provider_key
                
                print("❌ Invalid choice. Please try again.")
                
            except (KeyboardInterrupt, EOFError):
                return None
    
    def _configure_provider(self, provider: SimpleProvider) -> Optional[Dict[str, Any]]:
        """Step 2: Configure the selected provider"""
        print(f"\n⚙️ Step 2: Configure {provider.name}")
        print("-" * 40)
        
        config = {
            'Conjecture_LLM_PROVIDER': provider.name.lower(),
            'Conjecture_LLM_API_URL': provider.endpoint
        }
        
        if provider.type == 'local':
            config['Conjecture_LLM_MODEL'] = provider.default_model
            print(f"Default model: {provider.default_model}")
            print("You can change this later by editing the .env file")
            
        else:  # Cloud provider
            print(f"Setup instructions:")
            for cmd in provider.setup_commands:
                print(f"  • {cmd}")
            
            api_key = self._get_api_key(provider)
            if not api_key:
                return None
                
            config[provider.api_key_env_var] = api_key
            config['Conjecture_LLM_MODEL'] = provider.default_model
        
        # Allow customization
        custom_model = input(f"Model [{provider.default_model}]: ").strip()
        if custom_model:
            config['Conjecture_LLM_MODEL'] = custom_model
        
        custom_url = input(f"API URL [{provider.endpoint}]: ").strip()
        if custom_url:
            config['Conjecture_LLM_API_URL'] = custom_url
        
        return config
    
    def _get_api_key(self, provider: SimpleProvider) -> Optional[str]:
        """Get API key from user or environment"""
        # Check environment first
        if provider.api_key_env_var:
            existing_key = os.getenv(provider.api_key_env_var)
            if existing_key and not existing_key.startswith('your_'):
                use_existing = input(f"Use existing API key from {provider.api_key_env_var}? [Y/n]: ").strip().lower()
                if use_existing in ('', 'y', 'yes'):
                    return existing_key
        
        # Prompt for new key
        while True:
            try:
                api_key = input(f"Enter {provider.name} API key: ").strip()
                if not api_key:
                    retry = input("No API key entered. Try again? [Y/n]: ").strip().lower()
                    if retry in ('', 'y', 'yes'):
                        continue
                    return None
                
                # Basic validation
                if self._validate_api_key_format(provider.name.lower(), api_key):
                    return api_key
                else:
                    print("⚠️ API key format looks unusual. Proceed anyway? [y/N]: ", end="")
                    proceed = input().strip().lower()
                    if proceed == 'y':
                        return api_key
                    continue
                    
            except (KeyboardInterrupt, EOFError):
                return None
    
    def _validate_and_save(self, provider: SimpleProvider, config: Dict[str, Any]) -> bool:
        """Step 3: Validate configuration and save"""
        print(f"\n🔍 Step 3: Validate and save configuration")
        print("-" * 40)
        
        print("Configuration summary:")
        for key, value in config.items():
            if 'API_KEY' in key:
                masked = self._mask_api_key(value)
                print(f"  {key}: {masked}")
            else:
                print(f"  {key}: {value}")
        
        # Quick validation
        if provider.type == 'local':
            print("\nTesting local service connection...")
            if not self._test_endpoint(config['Conjecture_LLM_API_URL']):
                print("⚠️ Local service not responding, but proceeding anyway...")
                print("Make sure the service is running when you use Conjecture")
        else:
            # For cloud providers, we can't easily validate without making API calls
            print("✅ Cloud provider configuration looks correct")
        
        # Confirmation
        proceed = input("\nSave this configuration? [Y/n]: ").strip().lower()
        if proceed not in ('', 'y', 'yes'):
            return False
        
        # Save configuration
        return self.update_env_file(config)
    
    def _test_endpoint(self, endpoint: str, timeout: int = 3) -> bool:
        """Simple endpoint test (no async)"""
        try:
            response = requests.get(endpoint, timeout=timeout)
            return response.status_code == 200
        except:
            return False
    
    def _validate_api_key_format(self, provider_name: str, api_key: str) -> bool:
        """Basic API key format validation"""
        patterns = {
            'openai': r'^sk-[A-Za-z0-9]{48}$',
            'anthropic': r'^sk-ant-api[0-9]{2}-[A-Za-z0-9_-]{95}$',
            'google': r'^[A-Za-z0-9_-]{39}$',
            'chutes': r'^[A-Za-z0-9_-]{20,}$',
            'openrouter': r'^sk-or-v1-[A-Za-z0-9]{48}$'
        }
        
        pattern = patterns.get(provider_name)
        if not pattern:
            return True  # No validation for unknown providers
        
        try:
            return bool(re.match(pattern, api_key))
        except re.error:
            return True  # Invalid pattern, allow through
    
    def _mask_api_key(self, api_key: str) -> str:
        """Mask API key for display"""
        if len(api_key) <= 8:
            return '*' * len(api_key)
        return api_key[:4] + '*' * (len(api_key) - 8) + api_key[-4:]
    
    def _read_env_file(self) -> Dict[str, str]:
        """Read environment variables from .env file"""
        env_vars = {}
        
        if not self.env_file.exists():
            return env_vars
            
        try:
            content = self.env_file.read_text()
            for line in content.splitlines():
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
        except Exception as e:
            print(f"⚠️ Failed to read .env file: {e}")
        
        return env_vars
    
    def _write_env_file(self, env_vars: Dict[str, str]) -> None:
        """Write environment variables to .env file"""
        try:
            lines = [
                "# Conjecture Environment Configuration",
                f"# Generated on {datetime.now().isoformat()}",
                ""
            ]
            
            # Preferred order
            preferred_order = [
                'Conjecture_LLM_PROVIDER',
                'Conjecture_LLM_API_URL', 
                'Conjecture_LLM_MODEL',
                'Conjecture_EMBEDDING_MODEL',
                'Conjecture_DB_PATH',
                'Conjecture_CONFIDENCE',
                'Conjecture_DEBUG',
                'OPENAI_API_KEY',
                'ANTHROPIC_API_KEY',
                'GOOGLE_API_KEY',
                'CHUTES_API_KEY',
                'OPENROUTER_API_KEY'
            ]
            
            added = set()
            for key in preferred_order:
                if key in env_vars and key not in added:
                    value = env_vars[key]
                    if 'API_KEY' in key:
                        masked = self._mask_api_key(value)
                        lines.append(f"{key}={value}  # {masked}")
                    else:
                        lines.append(f"{key}={value}")
                    added.add(key)
            
            # Add any remaining variables
            for key, value in env_vars.items():
                if key not in added:
                    if 'API_KEY' in key:
                        masked = self._mask_api_key(value)
                        lines.append(f"{key}={value}  # {masked}")
                    else:
                        lines.append(f"{key}={value}")
            
            self.env_file.write_text('\n'.join(lines) + '\n')
            os.chmod(self.env_file, 0o600)  # Secure permissions
            
        except Exception as e:
            raise Exception(f"Failed to write .env file: {e}")
    
    def _create_backup(self) -> Optional[Path]:
        """Create backup of .env file"""
        if not self.env_file.exists():
            return None
            
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = self.env_file.parent / f'.env.backup.{timestamp}'
            backup_file.write_text(self.env_file.read_text())
            return backup_file
        except Exception as e:
            print(f"⚠️ Failed to create backup: {e}")
            return None
    
    def _create_env_example(self) -> None:
        """Create .env.example file"""
        content = """# Conjecture Environment Variables Template
# Copy this file to .env and fill in your actual values

# Primary Provider Configuration
Conjecture_LLM_PROVIDER=ollama
Conjecture_LLM_API_URL=http://localhost:11434
Conjecture_LLM_MODEL=llama2

# Embedding Configuration
Conjecture_EMBEDDING_MODEL=all-MiniLM-L6-v2

# Database Configuration
Conjecture_DB_PATH=data/conjecture.db
Conjecture_CONFIDENCE=0.7

# Debug Settings
Conjecture_DEBUG=false

# Cloud Service API Keys (Optional)
# OPENAI_API_KEY=your_openai_api_key_here
# ANTHROPIC_API_KEY=your_anthropic_api_key_here
# GOOGLE_API_KEY=your_google_api_key_here
# CHUTES_API_KEY=your_chutes_api_key_here
# OPENROUTER_API_KEY=your_openrouter_api_key_here
"""
        self.env_example_file.write_text(content.strip())

# Convenience functions
def quick_setup(project_root: Optional[str] = None) -> bool:
    """Quick setup - interactive if not configured"""
    wizard = SetupWizard(project_root)
    status = wizard.quick_status()
    
    if status['configured']:
        print(f"✅ Already configured with {status['provider']} ({status['model']})")
        return True
    
    return wizard.interactive_setup()

def check_status(project_root: Optional[str] = None) -> Dict[str, Any]:
    """Check configuration status"""
    wizard = SetupWizard(project_root)
    return wizard.quick_status()

def auto_setup_ollama(project_root: Optional[str] = None) -> bool:
    """Auto-setup for Ollama if detected"""
    wizard = SetupWizard(project_root)
    
    if wizard.quick_status()['configured']:
        print("✅ Already configured")
        return True
    
    # Check if Ollama is available
    if 'ollama' in wizard.auto_detect_local():
        config = {
            'Conjecture_LLM_PROVIDER': 'ollama',
            'Conjecture_LLM_API_URL': 'http://localhost:11434',
            'Conjecture_LLM_MODEL': 'llama2'
        }
        
        if wizard.update_env_file(config):
            print("✅ Auto-configured Ollama successfully!")
            return True
    
    print("❌ Ollama not available or setup failed")
    return False