#!/usr/bin/env python3
"""
Migration script to convert environment variables to config files
"""
import os
import json
from pathlib import Path

def migrate_from_env():
    """Migrate from environment variables to config files"""
    print("Migrating from environment variables to config files...")
    
    # Get environment variables
    env_vars = {
        'ollama': {
            'url': os.getenv('OLLAMA_API_URL', 'http://localhost:11434'),
            'api': os.getenv('OLLAMA_API_KEY', ''),
            'model': os.getenv('OLLAMA_MODEL', 'llama2'),
        },
        'lm_studio': {
            'url': os.getenv('LM_STUDIO_API_URL', 'http://localhost:1234'),
            'api': os.getenv('LM_STUDIO_API_KEY', ''),
            'model': os.getenv('LM_STUDIO_MODEL', 'ibm/granite-4-h-tiny'),
        },
        'chutes': {
            'url': os.getenv('CHUTES_API_URL', os.getenv('PROVIDER_API_URL', 'https://llm.chutes.ai/v1')),
            'api': os.getenv('CHUTES_API_KEY', os.getenv('PROVIDER_API_KEY', 'cpk_your-api-key-here')),
            'model': os.getenv('CHUTES_MODEL', os.getenv('PROVIDER_MODEL', 'zai-org/GLM-4.6-FP8')),
        },
        'openrouter': {
            'url': os.getenv('OPENROUTER_API_URL', 'https://openrouter.ai/api/v1'),
            'api': os.getenv('OPENROUTER_API_KEY', ''),
            'model': os.getenv('OPENROUTER_MODEL', 'openai/gpt-3.5-turbo'),
        },
        'openai': {
            'url': os.getenv('OPENAI_API_URL', 'https://api.openai.com/v1'),
            'api': os.getenv('OPENAI_API_KEY', ''),
            'model': os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
        }
    }
    
    # Build providers list
    providers = []
    for name, config in env_vars.items():
        # Only include provider if it has an API key (except for local providers)
        if config['api'] or name in ['ollama', 'lm_studio']:
            providers.append({
                'url': config['url'],
                'api': config['api'],
                'model': config['model'],
                'name': name
            })
    
    # Create config object
    config_data = {
        'providers': providers,
        'confidence_threshold': float(os.getenv('CONFIDENCE_THRESHOLD', 0.95)),
        'confident_threshold': float(os.getenv('CONFIDENT_THRESHOLD', 0.8)),
        'max_context_size': int(os.getenv('MAX_CONTEXT_SIZE', 10)),
        'batch_size': int(os.getenv('BATCH_SIZE', 10)),
        'debug': os.getenv('DEBUG', 'false').lower() == 'true',
        'database_path': os.getenv('DB_PATH', os.getenv('DATABASE_PATH', 'data/conjecture.db')),
        'user': os.getenv('CONJECTURE_USER', 'user'),
        'team': os.getenv('CONJECTURE_TEAM', 'default'),
    }
    
    # Create user config
    user_config_path = Path.home() / '.conjecture' / 'config.json'
    user_config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(user_config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"Created user config at: {user_config_path}")
    print(f"Added {len(providers)} providers")
    
    # Show what was migrated
    print("\nMigrated Configuration:")
    for provider in providers:
        print(f"  * {provider['name']}: {provider['model']} at {provider['url']}")
    
    print("\nNext steps:")
    print("  1. Review the config file at ~/.conjecture/config.json")
    print("  2. Update any placeholder API keys with real values")
    print("  3. Remove or comment out environment variables from .env file")
    print("  4. Test with: python conjecture validate")

def backup_existing_config():
    """Backup existing config file if it exists"""
    user_config_path = Path.home() / '.conjecture' / 'config.json'
    
    if user_config_path.exists():
        backup_path = user_config_path.with_suffix('.json.backup')
        shutil.copy2(user_config_path, backup_path)
        print(f"Backed up existing config to: {backup_path}")
        return True
    return False

if __name__ == "__main__":
    import shutil
    
    backup_existing_config()
    migrate_from_env()