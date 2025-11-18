# Conjecture Provider Discovery System

A comprehensive automatic provider discovery system for the Conjecture CLI that detects and configures available LLM providers.

## Features

### 🔍 **Automatic Detection**
- **Local Services**: Ollama, LM Studio
- **Cloud Services**: OpenAI, Anthropic, Google, Chutes.ai, OpenRouter
- **Priority-based Selection**: Local services preferred for privacy

### ⚙️ **Smart Configuration**
- **Safe .env Management**: Automatic backup and rollback
- **Security-First**: API key masking, gitignore protection
- **Validation**: API key format validation before storage

### 🔄 **Flexible Usage**
- **Auto Mode**: One-click setup
- **Manual Mode**: Interactive provider selection
- **Check Mode**: Quick availability scan
- **Integration**: Seamless CLI integration

## Installation

The discovery system requires `aiohttp` for async HTTP requests:

```bash
pip install aiohttp
```

## Usage

### Quick Start

```bash
# Auto-discover and configure best provider
python simple_local_cli.py discover --auto

# Check available providers without configuring
python simple_local_cli.py discover --check

# Interactive provider selection
python simple_local_cli.py discover

# Check current configuration status
python simple_local_cli.py config-status

# Full system health check (includes discovery)
python simple_local_cli.py health
```

### Command Options

#### `discover` Command
```bash
python simple_local_cli.py discover [OPTIONS]

Options:
  --auto, -a     Automatically configure best provider
  --provider, -p PREFERRED_PROVIDER
                 Preferred provider (ollama, lm_studio, openai, etc.)
  --check, -c    Quick check only, don't configure
```

#### `config-status` Command
Shows current configuration status:
- `.env` file existence
- `.gitignore` protection status
- Configured providers
- Missing components

#### `health` Command
System health check including:
- Provider discovery system status
- Available LLM providers
- Component availability

## Provider Detection Priority

1. **Local Services** (Preferred)
   - Ollama: `http://localhost:11434`
   - LM Studio: `http://localhost:1234`

2. **Cloud Services** (Backup)
   - Chutes.ai (priority 5)
   - OpenAI (priority 10)
   - Anthropic (priority 11)
   - Google (priority 12)
   - OpenRouter (priority 13)

## Supported Providers

### Local Services

| Provider | Endpoint | Models | Notes |
|----------|----------|--------|-------|
| Ollama | `http://localhost:11434` | Auto-detected | Privacy-focused |
| LM Studio | `http://localhost:1234` | Auto-detected | Local GUI |

### Cloud Services

| Provider | API Key Variable | Models | Status |
|----------|------------------|--------|--------|
| OpenAI | `OPENAI_API_KEY` | Public API list | ✅ Tested |
| Anthropic | `ANTHROPIC_API_KEY` | Private models | ✅ Tested |
| Google | `GOOGLE_API_KEY` | Private models | ✅ Tested |
| Chutes.ai | `CHUTES_API_KEY` | Pre-configured | ✅ Tested |
| OpenRouter | `OPENROUTER_API_KEY` | Public API list | ✅ Tested |

## Configuration Files

### `.env` (Auto-generated)
```bash
# Primary Provider Configuration
Conjecture_LLM_PROVIDER=ollama
Conjecture_LLM_API_URL=http://localhost:11434
Conjecture_LLM_MODEL=llama2

# Security: API keys are masked in comments
OPENAI_API_KEY=sk-1234...5678  # sk-1234...5678

# System Configuration
Conjecture_EMBEDDING_MODEL=all-MiniLM-L6-v2
Conjecture_DB_PATH=data/conjecture.db
Conjecture_CONFIDENCE=0.7
```

### `.env.example` (Template)
Comprehensive template with all supported providers and configuration examples.

## Security Features

### 🔒 **API Key Protection**
- Format validation before storage
- Automatic masking in logs and comments
- Secure file permissions (600)
- Never logged in plaintext

### 🛡️ **Git Protection**
- Automatic `.gitignore` updates
- Patterns for sensitive files
- Prevention of accidental commits
- Backup creation before changes

### 🔐 **Safe Configuration**
- Rollback capabilities
- Validation before writes
- Temporary file atomic writes
- Error handling and recovery

## API Reference

### ProviderDiscovery

```python
from discovery.provider_discovery import ProviderDiscovery

discovery = ProviderDiscovery()

# Automatic discovery and configuration
result = await discovery.run_automatic_discovery(
    auto_configure=True,
    preferred_provider="ollama"
)

# Manual interactive discovery
result = await discovery.run_manual_discovery()

# Quick availability check
result = await discovery.quick_check()

# Configuration status
status = discovery.get_configuration_status()
```

### ServiceDetector

```python
from discovery.service_detector import ServiceDetector

async with ServiceDetector(timeout=3) as detector:
    providers = await detector.detect_all()
    
    # Validate API key format
    is_valid = detector.validate_api_key_format("openai", api_key)
    
    # Mask API key for display
    masked = detector.mask_api_key(api_key)
```

### ConfigUpdater

```python
from discovery.config_updater import ConfigUpdater

updater = ConfigUpdater()

# Update configuration with detected providers
success, result = updater.update_config_with_providers(
    providers, 
    primary_provider="ollama",
    auto_confirm=True
)

# Check configuration status
status = updater.get_config_status()

# Ensure gitignore protection
updater.ensure_gitignore()
```

## Examples

### Basic Usage

```python
import asyncio
from discovery.provider_discovery import discover_and_configure

async def setup_conjecture():
    result = await discover_and_configure(
        auto_configure=True,
        preferred_provider="ollama"
    )
    
    if result['success']:
        print("✅ Setup complete!")
        print(f"Provider: {result['primary_provider']['name']}")
    else:
        print(f"❌ Setup failed: {result['message']}")

asyncio.run(setup_conjecture())
```

### Check Providers Only

```python
import asyncio
from discovery.provider_discovery import quick_check_providers

async def check_available():
    result = await quick_check_providers()
    
    if result['success']:
        providers = result['providers']
        print(f"Found {len(providers)} providers:")
        for p in providers:
            print(f"  • {p['name']} ({p['type']}) - {p['models_count']} models")

asyncio.run(check_available())
```

### Manual Configuration

```python
import asyncio
from discovery.provider_discovery import manual_discovery

async def interactive_setup():
    # This will prompt the user interactively
    result = await manual_discovery()
    return result['success']

asyncio.run(interactive_setup())
```

## Error Handling

### Common Issues

1. **No aiohttp installed**
   ```
   Error: Discovery system not available. Install aiohttp: pip install aiohttp
   ```
   Solution: Install aiohttp

2. **No providers detected**
   - Install Ollama: https://ollama.ai/
   - Install LM Studio: https://lmstudio.ai/
   - Set environment variables for cloud services

3. **Configuration permission errors**
   - Check directory write permissions
   - Ensure `.env` is not locked by another process

4. **Git ignore not working**
   - Run `config-status` to check protection
   - Manually add `.env` to `.gitignore` if needed

### Troubleshooting

```bash
# Check system health
python simple_local_cli.py health

# Verify configuration status
python simple_local_cli.py config-status

# Debug discovery process
python simple_local_cli.py discover --check

# Test discovery system
python quick_discovery_test.py
```

## Development

### Project Structure

```
src/discovery/
├── __init__.py              # Module exports
├── provider_discovery.py    # Main discovery engine
├── service_detector.py      # Individual service detectors
└── config_updater.py        # Safe configuration management
```

### Adding New Providers

1. **Update `service_detector.py`**:
   ```python
   self.cloud_providers['new_provider'] = {
       'env_vars': ['NEW_PROVIDER_API_KEY'],
       'key_pattern': r'^pattern_here$',
       'models_endpoint': 'https://api.newprovider.com/v1/models',
       'priority': 15
   }
   ```

2. **Update `config_updater.py`**:
   ```python
   self.env_mappings['New_Provider'] = {
       'Conjecture_LLM_PROVIDER': 'new_provider',
       'Conjecture_LLM_API_URL': 'https://api.newprovider.com/v1',
       'NEW_PROVIDER_API_KEY': '{api_key}'
   }
   ```

3. **Update `.env.example`** with new provider documentation

### Testing

```bash
# Run basic functionality test
python quick_discovery_test.py

# Run comprehensive test suite
python test_discovery_system.py

# Test CLI integration
python simple_local_cli.py discover --check
```

## Performance

### Discovery Speed
- **Local services**: ~2 seconds
- **Cloud services**: ~3 seconds  
- **Total discovery**: ~5 seconds maximum

### Memory Usage
- **Discovery system**: ~10MB additional memory
- **Async operations**: Non-blocking, efficient

### Network Requirements
- **Local detection**: No internet required
- **Cloud detection**: Internet for model lists
- **Timeout**: Configurable (default 3 seconds)

## License

This discovery system is part of the Conjecture project and follows the same license terms.

## Contributing

When contributing to the discovery system:

1. Maintain security-first approach
2. Add comprehensive test coverage
3. Update documentation
4. Follow existing code patterns
5. Test with multiple provider types

## Support

For issues with the discovery system:

1. Check the troubleshooting section
2. Run `python simple_local_cli.py health`
3. Review system logs
4. Create an issue with diagnostic output