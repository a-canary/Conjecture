"""
Streamlined Configuration Wizard for Conjecture

A modern, user-friendly setup wizard with diagnostics, step-by-step guidance,
and uv integration for dependency management.
"""

import os
import sys
import json
import subprocess
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import re
import shutil

from .diagnostics import SystemDiagnostics, DiagnosticResult, run_diagnostics

@dataclass
class ProviderConfig:
    """Provider configuration data"""
    name: str
    type: str  # 'local' or 'cloud'
    endpoint: str
    api_key_env_var: Optional[str] = None
    default_model: str = ""
    description: str = ""
    setup_url: str = ""
    api_key_pattern: Optional[str] = None

class StreamlinedConfigWizard:
    """Modern configuration wizard with diagnostics and uv support"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.env_file = self.project_root / '.env'
        self.env_example = self.project_root / '.env.example'
        
        # Provider configurations
        self.providers = {
            'ollama': ProviderConfig(
                name='Ollama',
                type='local',
                endpoint='http://localhost:11434',
                default_model='llama2',
                description='Local LLM runner - easiest setup for local AI',
                setup_url='https://ollama.ai/'
            ),
            'lm_studio': ProviderConfig(
                name='LM Studio',
                type='local',
                endpoint='http://localhost:1234/v1',
                default_model='local-model',
                description='Local model server with GUI interface',
                setup_url='https://lmstudio.ai/'
            ),
            'openai': ProviderConfig(
                name='OpenAI',
                type='cloud',
                endpoint='https://api.openai.com/v1',
                api_key_env_var='OPENAI_API_KEY',
                default_model='gpt-3.5-turbo',
                description='OpenAI GPT models (paid)',
                setup_url='https://platform.openai.com/api-keys',
                api_key_pattern=r'^sk-[A-Za-z0-9]{48}$'
            ),
            'anthropic': ProviderConfig(
                name='Anthropic',
                type='cloud',
                endpoint='https://api.anthropic.com',
                api_key_env_var='ANTHROPIC_API_KEY',
                default_model='claude-3-haiku-20240307',
                description='Anthropic Claude models (paid)',
                setup_url='https://console.anthropic.com/',
                api_key_pattern=r'^sk-ant-api[0-9]{2}-[A-Za-z0-9_-]{95}$'
            ),
            'chutes': ProviderConfig(
                name='Chutes.ai',
                type='cloud',
                endpoint='https://llm.chutes.ai/v1',
                api_key_env_var='CHUTES_API_KEY',
                default_model='zai-org/GLM-4.6-FP8',
                description='Affordable cloud API for various models',
                setup_url='https://chutes.ai/',
                api_key_pattern=r'^[A-Za-z0-9_-]{20,}$'
            ),
            'openrouter': ProviderConfig(
                name='OpenRouter',
                type='cloud',
                endpoint='https://openrouter.ai/api/v1',
                api_key_env_var='OPENROUTER_API_KEY',
                default_model='openai/gpt-3.5-turbo',
                description='Access to 100+ models (paid)',
                setup_url='https://openrouter.ai/keys',
                api_key_pattern=r'^sk-or-v1-[A-Za-z0-9]{48}$'
            )
        }
    
    def run_wizard(self) -> bool:
        """Run the complete setup wizard"""
        self._print_header()
        
        # Phase 1: Diagnostics
        print("\\n🔍 Phase 1: System Diagnostics")
        print("=" * 50)
        
        diagnostics = SystemDiagnostics(self.project_root)
        diagnostic_results = diagnostics.run_all_diagnostics()
        
        self._display_diagnostics(diagnostic_results)
        
        if not diagnostic_results['summary']['ready_for_setup']:
            print("\\n❌ Critical issues found. Please resolve them before continuing.")
            if not self._ask_continue("Continue anyway?"):
                return False
        
        # Phase 2: Setup
        print("\\n⚙️ Phase 2: Configuration Setup")
        print("=" * 50)
        
        # Check if already configured
        if self._is_configured():
            print("✅ Conjecture is already configured!")
            if not self._ask_continue("Would you like to reconfigure?"):
                return True
        
        # Step 1: Install dependencies
        if not self._handle_dependencies():
            print("❌ Dependency installation failed")
            return False
        
        # Step 2: Configure provider
        provider_config = self._configure_provider()
        if not provider_config:
            print("❌ Provider configuration cancelled")
            return False
        
        # Step 3: Configure user settings
        user_config = self._configure_user_settings()
        if not user_config:
            print("❌ User configuration cancelled")
            return False
        
        # Phase 3: Validation and Save
        print("\\n💾 Phase 3: Validation and Save")
        print("=" * 50)
        
        if self._validate_and_save(provider_config, user_config):
            self._print_completion_summary(provider_config, user_config)
            return True
        else:
            print("❌ Failed to save configuration")
            return False
    
    def _print_header(self) -> None:
        """Print wizard header"""
        print("🚀 Conjecture Configuration Wizard")
        print("=" * 50)
        print("This wizard will help you set up Conjecture with:")
        print("• System diagnostics and health checks")
        print("• Dependency installation with uv")
        print("• LLM provider configuration")
        print("• Personalized settings")
        print("• Configuration validation")
        print()
    
    def _display_diagnostics(self, results: Dict[str, Any]) -> None:
        """Display diagnostic results"""
        # System info
        sys_info = results['system_info']
        print(f"📊 System: {sys_info['platform']} | Python {sys_info['python_version'].split()[0]}")
        print(f"💾 Memory: {sys_info['memory_gb']}GB | 💿 Disk: {sys_info['available_disk_gb']}GB free")
        print()
        
        # Results
        for result in results['results']:
            icon = self._get_status_icon(result['status'])
            print(f"{icon} {result['name']}: {result['message']}")
            
            if result.get('suggestion'):
                print(f"   💡 {result['suggestion']}")
        
        # Summary
        summary = results['summary']
        print(f"\\n📋 Summary: {summary['message']}")
        print(f"   Checks: {summary['total_checks']} | ✅ {summary['status_counts']['pass']} | ⚠️ {summary['status_counts']['warn']} | ❌ {summary['status_counts']['fail']}")
    
    def _get_status_icon(self, status: str) -> str:
        """Get icon for status"""
        icons = {
            'pass': '✅',
            'warn': '⚠️',
            'fail': '❌',
            'info': 'ℹ️'
        }
        return icons.get(status, '❓')
    
    def _ask_continue(self, message: str) -> bool:
        """Ask user for confirmation"""
        while True:
            response = input(f"\\n{message} [Y/n]: ").strip().lower()
            if response in ('', 'y', 'yes'):
                return True
            elif response in ('n', 'no'):
                return False
            else:
                print("Please enter 'y' or 'n'")
    
    def _is_configured(self) -> bool:
        """Check if already configured"""
        if not self.env_file.exists():
            return False
        
        try:
            content = self.env_file.read_text()
            # Check for placeholder values
            if 'your-api-key-here' in content or 'your_' in content:
                return False
            
            # Check for required variables
            required_vars = ['Conjecture_LLM_PROVIDER', 'Conjecture_LLM_API_URL']
            for var in required_vars:
                if var not in content:
                    return False
            
            return True
        except Exception:
            return False
    
    def _handle_dependencies(self) -> bool:
        """Handle dependency installation with uv"""
        print("\\n📦 Step 1: Dependencies")
        print("-" * 30)
        
        # Check if uv is available
        uv_available = shutil.which('uv') is not None
        
        if uv_available:
            print("✅ uv detected - using fast dependency installation")
            use_uv = self._ask_continue("Use uv for dependency installation?")
        else:
            print("ℹ️ uv not available - will use pip")
            print("💡 Install uv for faster installs: pip install uv")
            use_uv = False
        
        # Install dependencies
        requirements_file = self.project_root / 'requirements.txt'
        if not requirements_file.exists():
            print("❌ requirements.txt not found")
            return False
        
        print(f"📥 Installing dependencies from requirements.txt...")
        
        try:
            if use_uv:
                cmd = ['uv', 'pip', 'install', '-r', str(requirements_file)]
            else:
                cmd = [sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)]
            
            # Run installation
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Dependencies installed successfully")
                return True
            else:
                print(f"❌ Installation failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Installation error: {e}")
            return False
    
    def _configure_provider(self) -> Optional[Dict[str, str]]:
        """Configure LLM provider"""
        print("\\n🤖 Step 2: LLM Provider Configuration")
        print("-" * 40)
        
        # Detect available local providers
        available_local = []
        for key, provider in self.providers.items():
            if provider.type == 'local':
                try:
                    response = requests.get(provider.endpoint, timeout=2)
                    if response.status_code == 200:
                        available_local.append(key)
                except:
                    pass
        
        # Check for existing API keys
        available_cloud = []
        for key, provider in self.providers.items():
            if provider.type == 'cloud' and provider.api_key_env_var:
                if os.getenv(provider.api_key_env_var):
                    available_cloud.append(key)
        
        # Display options
        print("Available providers:")
        options = []
        
        # Local providers (detected first)
        if available_local:
            print("\\n🏠 Local Providers (Detected):")
            for i, key in enumerate(available_local, 1):
                provider = self.providers[key]
                print(f"  {i}. {provider.name} - {provider.description} ✅")
                options.append(('local_detected', key))
        
        # Cloud providers with keys
        if available_cloud:
            print("\\n☁️ Cloud Providers (API key detected):")
            start_idx = len(options) + 1
            for i, key in enumerate(available_cloud, start_idx):
                provider = self.providers[key]
                print(f"  {i}. {provider.name} - {provider.description} 🔑")
                options.append(('cloud_detected', key))
        
        # All providers
        print("\\n📋 All Providers:")
        remaining = [k for k in self.providers.keys() 
                    if k not in available_local and k not in available_cloud]
        start_idx = len(options) + 1
        for i, key in enumerate(remaining, start_idx):
            provider = self.providers[key]
            icon = "🏠" if provider.type == 'local' else "☁️"
            print(f"  {i}. {provider.name} - {provider.description} {icon}")
            options.append(('all', key))
        
        # Get selection
        while True:
            try:
                choice = input(f"\\nSelect provider [1-{len(options)}]: ").strip()
                if not choice:
                    continue
                
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    _, provider_key = options[idx]
                    break
                else:
                    print("❌ Invalid choice")
            except ValueError:
                print("❌ Please enter a number")
            except KeyboardInterrupt:
                return None
        
        provider = self.providers[provider_key]
        print(f"\\n✅ Selected: {provider.name}")
        
        # Configure provider
        config = {
            'Conjecture_LLM_PROVIDER': provider.name.lower(),
            'Conjecture_LLM_API_URL': provider.endpoint
        }
        
        if provider.type == 'local':
            config['Conjecture_LLM_MODEL'] = provider.default_model
            print(f"📝 Default model: {provider.default_model}")
        else:
            # Cloud provider - get API key
            api_key = self._get_api_key(provider)
            if not api_key:
                return None
            
            config[provider.api_key_env_var] = api_key
            config['Conjecture_LLM_MODEL'] = provider.default_model
        
        # Custom model
        custom_model = input(f"Custom model [{provider.default_model}]: ").strip()
        if custom_model:
            config['Conjecture_LLM_MODEL'] = custom_model
        
        return config
    
    def _get_api_key(self, provider: ProviderConfig) -> Optional[str]:
        """Get API key from user"""
        print(f"\\n🔑 {provider.name} API Key Required")
        print(f"📖 Get your key at: {provider.setup_url}")
        
        # Check environment first
        if provider.api_key_env_var and os.getenv(provider.api_key_env_var):
            use_existing = self._ask_continue(f"Use existing {provider.api_key_env_var}?")
            if use_existing:
                return os.getenv(provider.api_key_env_var)
        
        # Prompt for key
        while True:
            try:
                api_key = input("Enter API key: ").strip()
                if not api_key:
                    retry = self._ask_continue("No key entered. Try again?")
                    if not retry:
                        return None
                    continue
                
                # Validate format if pattern exists
                if provider.api_key_pattern:
                    if not re.match(provider.api_key_pattern, api_key):
                        print("⚠️ API key format looks unusual")
                        if not self._ask_continue("Use this key anyway?"):
                            continue
                
                return api_key
                
            except KeyboardInterrupt:
                return None
    
    def _configure_user_settings(self) -> Dict[str, str]:
        """Configure user-specific settings"""
        print("\\n👤 Step 3: User Settings")
        print("-" * 25)
        
        config = {}
        
        # Username
        current_user = os.getenv('USER') or os.getenv('USERNAME') or 'user'
        username = input(f"Username [{current_user}]: ").strip()
        config['CONJECTURE_USER'] = username if username else current_user
        
        # Workspace
        workspace = input("Workspace name [my-project]: ").strip()
        config['CONJECTURE_WORKSPACE'] = workspace if workspace else 'my-project'
        
        # Team (optional)
        team = input("Team name (optional): ").strip()
        if team:
            config['CONJECTURE_TEAM'] = team
        
        # Database path
        db_path = input("Database path [data/conjecture.db]: ").strip()
        config['DB_PATH'] = db_path if db_path else 'data/conjecture.db'
        
        # Create data directory if needed
        data_dir = self.project_root / Path(db_path).parent
        data_dir.mkdir(exist_ok=True)
        
        return config
    
    def _validate_and_save(self, provider_config: Dict[str, str], user_config: Dict[str, str]) -> bool:
        """Validate and save configuration"""
        print("\\n🔍 Configuration Summary")
        print("-" * 25)
        
        # Combine all config
        all_config = {**provider_config, **user_config}
        
        # Add defaults
        defaults = {
            'Conjecture_EMBEDDING_MODEL': 'all-MiniLM-L6-v2',
            'CONFIDENCE_THRESHOLD': '0.95',
            'MAX_CONTEXT_SIZE': '10',
            'DEBUG': 'false'
        }
        
        for key, value in defaults.items():
            if key not in all_config:
                all_config[key] = value
        
        # Display summary
        for key, value in all_config.items():
            if 'API_KEY' in key:
                masked = value[:4] + '*' * (len(value) - 8) + value[-4:] if len(value) > 8 else '*' * len(value)
                print(f"  {key}: {masked}")
            else:
                print(f"  {key}: {value}")
        
        # Confirm
        if not self._ask_continue("\\nSave this configuration?"):
            return False
        
        # Backup existing .env
        if self.env_file.exists():
            backup_path = self.env_file.with_suffix(f'.backup.{int(time.time())}')
            shutil.copy2(self.env_file, backup_path)
            print(f"💾 Backed up existing config to {backup_path.name}")
        
        # Write new configuration
        try:
            lines = [
                "# Conjecture Configuration",
                f"# Generated by wizard on {datetime.now().isoformat()}",
                ""
            ]
            
            # Write in logical order
            order = [
                'Conjecture_LLM_PROVIDER', 'Conjecture_LLM_API_URL', 'Conjecture_LLM_MODEL',
                'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'CHUTES_API_KEY', 'OPENROUTER_API_KEY',
                'CONJECTURE_USER', 'CONJECTURE_WORKSPACE', 'CONJECTURE_TEAM',
                'DB_PATH', 'Conjecture_EMBEDDING_MODEL', 'CONFIDENCE_THRESHOLD',
                'MAX_CONTEXT_SIZE', 'DEBUG'
            ]
            
            for key in order:
                if key in all_config:
                    lines.append(f"{key}={all_config[key]}")
            
            self.env_file.write_text('\\n'.join(lines) + '\\n')
            self.env_file.chmod(0o600)  # Secure permissions
            
            print("✅ Configuration saved successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save configuration: {e}")
            return False
    
    def _print_completion_summary(self, provider_config: Dict[str, str], user_config: Dict[str, str]) -> None:
        """Print completion summary"""
        print("\\n🎉 Setup Complete!")
        print("=" * 30)
        
        provider_name = provider_config.get('Conjecture_LLM_PROVIDER', 'unknown')
        model = provider_config.get('Conjecture_LLM_MODEL', 'unknown')
        username = user_config.get('CONJECTURE_USER', 'user')
        
        print(f"✅ Provider: {provider_name}")
        print(f"✅ Model: {model}")
        print(f"✅ User: {username}")
        print(f"✅ Config: {self.env_file}")
        
        print("\\n🚀 Next Steps:")
        print("1. Test your configuration:")
        print(f"   python {self.project_root.name} validate")
        print("\\n2. Create your first claim:")
        print(f"   python {self.project_root.name} create 'The sky is blue' --confidence 0.95")
        print("\\n3. Search claims:")
        print(f"   python {self.project_root.name} search 'sky'")
        
        print("\\n💡 Need help? Check the README.md file")

def run_wizard(project_root: Optional[Path] = None) -> bool:
    """Convenience function to run the wizard"""
    wizard = StreamlinedConfigWizard(project_root)
    return wizard.run_wizard()