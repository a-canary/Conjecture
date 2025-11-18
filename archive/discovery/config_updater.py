"""
Configuration Updater Module

Safe configuration file management with:
- .env file creation and updates
- Environment variable management
- Security validation
- Backup and rollback capabilities
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import re
import hashlib
import logging
from datetime import datetime

from .service_detector import DetectedProvider

# Configure logging
logger = logging.getLogger(__name__)

class ConfigUpdater:
    """Safe configuration file management"""
    
    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.env_file = self.project_root / '.env'
        self.env_example_file = self.project_root / '.env.example'
        self.gitignore_file = self.project_root / '.gitignore'
        
        # Security patterns for API keys
        self.api_key_patterns = {
            'openai': r'^sk-[A-Za-z0-9]{48}$',
            'anthropic': r'^sk-ant-api[0-9]{2}-[A-Za-z0-9_-]{95}$',
            'google': r'^[A-Za-z0-9_-]{39}$',
            'chutes': r'^[A-Za-z0-9_-]{20,}$',
            'openrouter': r'^sk-or-v1-[A-Za-z0-9]{48}$'
        }
        
        # Environment variable mappings
        self.env_mappings = {
            'Ollama': {
                'Conjecture_LLM_PROVIDER': 'ollama',
                'Conjecture_LLM_API_URL': 'http://localhost:11434',
                'Conjecture_LLM_MODEL': 'llama2'
            },
            'Lm_Studio': {
                'Conjecture_LLM_PROVIDER': 'lm_studio',
                'Conjecture_LLM_API_URL': 'http://localhost:1234/v1',
                'Conjecture_LLM_MODEL': 'local-model'
            },
            'Openai': {
                'Conjecture_LLM_PROVIDER': 'openai',
                'Conjecture_LLM_API_URL': 'https://api.openai.com/v1',
                'OPENAI_API_KEY': '{api_key}'
            },
            'Anthropic': {
                'Conjecture_LLM_PROVIDER': 'anthropic',
                'Conjecture_LLM_API_URL': 'https://api.anthropic.com',
                'ANTHROPIC_API_KEY': '{api_key}'
            },
            'Google': {
                'Conjecture_LLM_PROVIDER': 'google',
                'Conjecture_LLM_API_URL': 'https://generativelanguage.googleapis.com/v1',
                'GOOGLE_API_KEY': '{api_key}'
            },
            'Chutes': {
                'Conjecture_LLM_PROVIDER': 'chutes',
                'Conjecture_LLM_API_URL': 'https://llm.chutes.ai/v1',
                'CHUTES_API_KEY': '{api_key}'
            },
            'Openrouter': {
                'Conjecture_LLM_PROVIDER': 'openrouter',
                'Conjecture_LLM_API_URL': 'https://openrouter.ai/api/v1',
                'OPENROUTER_API_KEY': '{api_key}'
            }
        }

    def ensure_gitignore(self) -> bool:
        """Ensure .env is in .gitignore"""
        try:
            if not self.gitignore_file.exists():
                # Create basic .gitignore
                content = """
# Environment variables
.env
.env.local
.env.production

# API keys and secrets
*api_key*
*secret*
*credential*
*token*

# Conjecture specific
*.db
*.sqlite
data/
logs/
"""
                self.gitignore_file.write_text(content.strip())
                return True
            
            # Check if .env is already ignored
            content = self.gitignore_file.read_text()
            if '.env' not in content:
                content += '\n# Environment variables\n.env\n.env.local\n.env.production\n'
                self.gitignore_file.write_text(content)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to update .gitignore: {e}")
            return False

    def backup_env_file(self) -> Optional[Path]:
        """Create backup of existing .env file"""
        if not self.env_file.exists():
            return None
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = self.env_file.parent / f'.env.backup.{timestamp}'
            shutil.copy2(self.env_file, backup_file)
            logger.info(f"Created backup: {backup_file}")
            return backup_file
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return None

    def validate_api_key(self, provider_name: str, api_key: str) -> bool:
        """Validate API key format"""
        pattern = self.api_key_patterns.get(provider_name.lower())
        if not pattern:
            return True  # No validation pattern
        
        try:
            return bool(re.match(pattern, api_key))
        except re.error:
            logger.warning(f"Invalid regex pattern for {provider_name}")
            return True

    def mask_api_key(self, api_key: str) -> str:
        """Mask API key for safe display"""
        if len(api_key) <= 8:
            return '*' * len(api_key)
        return api_key[:4] + '*' * (len(api_key) - 8) + api_key[-4:]

    def read_existing_env(self) -> Dict[str, str]:
        """Read existing .env file"""
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
            logger.error(f"Failed to read .env file: {e}")
        
        return env_vars

    def write_env_file(self, env_vars: Dict[str, str], create_backup: bool = True) -> bool:
        """Write environment variables to .env file"""
        try:
            # Create backup if file exists
            if create_backup and self.env_file.exists():
                self.backup_env_file()
            
            # Build content
            lines = ["# Conjecture Environment Configuration", f"# Generated on {datetime.now().isoformat()}"]
            
            # Add environment variables in a logical order
            preferred_order = [
                'Conjecture_LLM_PROVIDER',
                'Conjecture_LLM_API_URL', 
                'Conjecture_LLM_MODEL',
                'Conjecture_EMBEDDING_MODEL',
                'Conjecture_DB_PATH',
                'Conjecture_CONFIDENCE',
                'OPENAI_API_KEY',
                'ANTHROPIC_API_KEY',
                'GOOGLE_API_KEY',
                'CHUTES_API_KEY',
                'OPENROUTER_API_KEY'
            ]
            
            added_vars = set()
            
            # Add variables in preferred order
            for key in preferred_order:
                if key in env_vars and key not in added_vars:
                    value = env_vars[key]
                    # Mask API keys in comments
                    if 'API_KEY' in key:
                        masked = self.mask_api_key(value)
                        lines.append(f"{key}={value}  # {masked}")
                    else:
                        lines.append(f"{key}={value}")
                    added_vars.add(key)
            
            # Add any remaining variables
            for key, value in env_vars.items():
                if key not in added_vars and key not in added_vars:
                    if 'API_KEY' in key:
                        masked = self.mask_api_key(value)
                        lines.append(f"{key}={value}  # {masked}")
                    else:
                        lines.append(f"{key}={value}")
            
            # Write file
            self.env_file.write_text('\n'.join(lines) + '\n')
            
            # Ensure proper permissions
            os.chmod(self.env_file, 0o600)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to write .env file: {e}")
            return False

    def update_config_with_providers(
        self, 
        providers: List[DetectedProvider], 
        primary_provider: Optional[str] = None,
        auto_confirm: bool = False
    ) -> Tuple[bool, Dict[str, Any]]:
        """Update configuration with detected providers"""
        
        # Ensure .env is ignored by git
        self.ensure_gitignore()
        
        if not providers:
            return False, {'error': 'No providers detected'}
        
        # Select primary provider
        if not primary_provider:
            # Prefer local services, then first available
            local_providers = [p for p in providers if p.type == 'local']
            primary_provider = local_providers[0].name if local_providers else providers[0].name
        
        primary_provider_obj = next((p for p in providers if p.name == primary_provider), providers[0])
        
        # Read existing configuration
        existing_env = self.read_existing_env()
        
        # Prepare new environment variables
        new_env = existing_env.copy()
        updates = {}
        
        # Add configuration for primary provider
        provider_config = self.env_mappings.get(primary_provider_obj.name, {})
        
        for key, value in provider_config.items():
            if '{api_key}' in value and primary_provider_obj.api_key_env_var:
                api_key = os.getenv(primary_provider_obj.api_key_env_var)
                if api_key and self.validate_api_key(primary_provider_obj.name.lower(), api_key):
                    new_env[key] = api_key
                    updates[key] = self.mask_api_key(api_key)
            else:
                new_env[key] = value
                updates[key] = value
        
        # Set preferred model if available
        if primary_provider_obj.models:
            model_key = 'Conjecture_LLM_MODEL'
            preferred_model = primary_provider_obj.models[0]
            if preferred_model:
                new_env[model_key] = preferred_model
                updates[model_key] = preferred_model
        
        # Add other detected providers as backup (API keys only)
        for provider in providers:
            if provider.name != primary_provider_obj.name and provider.api_key_env_var:
                api_key = os.getenv(provider.api_key_env_var)
                if api_key and self.validate_api_key(provider.name.lower(), api_key):
                    new_env[provider.api_key_env_var] = api_key
                    updates[provider.api_key_env_var] = self.mask_api_key(api_key)
        
        # Add default configurations
        defaults = {
            'Conjecture_EMBEDDING_MODEL': 'all-MiniLM-L6-v2',
            'Conjecture_DB_PATH': 'data/conjecture.db',
            'Conjecture_CONFIDENCE': '0.7',
            'Conjecture_DEBUG': 'false'
        }
        
        for key, value in defaults.items():
            if key not in new_env:
                new_env[key] = value
                updates[key] = value
        
        # Summary of changes
        summary = {
            'primary_provider': primary_provider_obj.name,
            'provider_type': primary_provider_obj.type,
            'updates': updates,
            'total_providers': len(providers),
            'backup_created': False
        }
        
        # Write configuration
        if auto_confirm:
            success = self.write_env_file(new_env)
            summary['auto_confirmed'] = True
        else:
            # For manual confirmation, we just return the summary
            summary['pending_env'] = new_env
            success = True
            summary['awaiting_confirmation'] = True
        
        return success, summary

    def create_env_example(self) -> bool:
        """Create or update .env.example file"""
        try:
            content = """# Conjecture Environment Variables Template
# Copy this file to .env and fill in your actual API keys

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
# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic Claude
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Google Gemini
GOOGLE_API_KEY=your_google_api_key_here

# Chutes.ai
CHUTES_API_KEY=your_chutes_api_key_here

# OpenRouter
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Alternative Provider URLs
# LM Studio (alternative to Ollama)
# Conjecture_LLM_PROVIDER=lm_studio
# Conjecture_LLM_API_URL=http://localhost:1234/v1

# OpenAI (cloud)
# Conjecture_LLM_PROVIDER=openai
# Conjecture_LLM_API_URL=https://api.openai.com/v1

# Anthropic (cloud)
# Conjecture_LLM_PROVIDER=anthropic
# Conjecture_LLM_API_URL=https://api.anthropic.com
"""
            
            self.env_example_file.write_text(content.strip())
            return True
            
        except Exception as e:
            logger.error(f"Failed to create .env.example: {e}")
            return False

    def get_config_status(self) -> Dict[str, Any]:
        """Get current configuration status"""
        status = {
            'env_file_exists': self.env_file.exists(),
            'env_example_exists': self.env_example_file.exists(),
            'gitignore_protected': False,
            'configured_providers': [],
            'missing_providers': []
        }
        
        # Check gitignore protection
        if self.gitignore_file.exists():
            content = self.gitignore_file.read_text()
            status['gitignore_protected'] = '.env' in content
        
        # Check configured providers
        if self.env_file.exists():
            env_vars = self.read_existing_env()
            
            provider = env_vars.get('Conjecture_LLM_PROVIDER')
            if provider:
                status['configured_providers'].append(provider)
            
            # Check for API keys
            api_key_vars = [
                'OPENAI_API_KEY',
                'ANTHROPIC_API_KEY', 
                'GOOGLE_API_KEY',
                'CHUTES_API_KEY',
                'OPENROUTER_API_KEY'
            ]
            
            for var in api_key_vars:
                if var in env_vars and env_vars[var] and env_vars[var] != f'your_{var.lower()}_here':
                    provider_name = var.replace('_API_KEY', '').lower()
                    if provider_name not in status['configured_providers']:
                        status['configured_providers'].append(provider_name)
        
        return status