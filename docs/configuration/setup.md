# Setup Wizard - Usage Guide

## Quick Start

The Setup Wizard makes configuring Conjecture simple and intuitive. Here's everything you need to know.

## Installation Check

After installing Conjecture, check if you're configured:

```python
from config.setup_wizard import check_status

status = check_status()
if status['configured']:
    print(f"✅ Ready! Using {status['provider']} ({status['model']})")
else:
    print("❌ Configuration required")
```

## Setup Methods

### 1. Automatic Setup (Recommended)

```python
from config.setup_wizard import quick_setup

quick_setup()  # Handles everything automatically
```

### 2. Ollama Auto-Setup

If you have Ollama installed:

```python
from config.setup_wizard import auto_setup_ollama

if auto_setup_ollama():
    print("🎉 Ollama configured!")
else:
    print("Install Ollama first: https://ollama.ai/")
```

### 3. Interactive Setup

For full control:

```python
from config.setup_wizard import SetupWizard

wizard = SetupWizard()
wizard.interactive_setup()
```

## Supported Providers

### Local Services (Easiest)

#### Ollama
```bash
# Install
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model  
ollama pull llama2

# Auto-configure with Conjecture
python -c "from config.setup_wizard import auto_setup_ollama; auto_setup_ollama()"
```

#### LM Studio
```bash
# Download and run LM Studio
# Load a model in the UI
# Configure with interactive setup
python -c "from config.setup_wizard import SetupWizard; SetupWizard().interactive_setup()"
```

### Cloud Services

#### OpenAI
```bash
# Get API key from: https://platform.openai.com/api-keys
export OPENAI_API_KEY="your-key-here"

# Configure interactively
python -c "from config.setup_wizard import quick_setup; quick_setup()"
```

#### Anthropic
```bash
# Get API key from: https://console.anthropic.com/
export ANTHROPIC_API_KEY="your-key-here"

# Configure interactively  
python -c "from config.setup_wizard import quick_setup; quick_setup()"
```

#### Chutes (Affordable Alternative)
```bash
# Get API key from: https://chutes.ai/
export CHUTES_API_KEY="your-key-here"

# Configure interactively
python -c "from config.setup_wizard import quick_setup; quick_setup()"
```

## Manual Configuration

Edit `.env` file directly:

```bash
# For Ollama
PROVIDER_API_URL=http://localhost:11434
PROVIDER_API_KEY=
PROVIDER_MODEL=llama2

# For OpenAI
PROVIDER_API_URL=https://api.openai.com/v1
PROVIDER_API_KEY=sk-your-api-key-here
PROVIDER_MODEL=gpt-3.5-turbo

# For Anthropic
PROVIDER_API_URL=https://api.anthropic.com
PROVIDER_API_KEY=sk-ant-api03-your-anthropic-key-here
PROVIDER_MODEL=claude-3-haiku-20240307

# For Chutes.ai
PROVIDER_API_URL=https://llm.chutes.ai/v1
PROVIDER_API_KEY=cpk_your-api-key-here
PROVIDER_MODEL=zai-org/GLM-4.6-FP8
```

## Common Workflows

### First Time Setup
```python
from config.setup_wizard import check_status, quick_setup

status = check_status()
if not status['configured']:
    print("Welcome to Conjecture! Let's set you up...")
    if quick_setup():
        print("🎉 You're ready to go!")
    else:
        print("See setup guide for manual configuration")
```

### Switching Providers
```python
from config.setup_wizard import SetupWizard

wizard = SetupWizard()
wizard.interactive_setup()  # Will guide you through switching
```

### Checking Configuration
```python
from config.setup_wizard import check_status

status = check_status()
print(f"Provider: {status.get('provider', 'Not configured')}")
print(f"Model: {status.get('model', 'N/A')}")
print(f"API URL: {status.get('api_url', 'N/A')}")
print(f"Type: {status.get('provider_type', 'N/A')}")
```

### Detecting Local Services
```python
from config.setup_wizard import SetupWizard

wizard = SetupWizard()
services = wizard.auto_detect_local()

if services:
    print(f"Found local services: {', '.join(services).title()}")
else:
    print("No local services detected")
    print("Install Ollama: https://ollama.ai/")
    print("Install LM Studio: https://lmstudio.ai/")
```

## Troubleshooting

### Common Issues

**"No local providers detected"**
```bash
# Make sure Ollama is running
ollama list

# Or start LM Studio and load a model
```

**"Missing API key"**
```bash
# Set environment variable
export OPENAI_API_KEY="your-key-here"

# Or run interactive setup
python -c "from config.setup_wizard import SetupWizard; SetupWizard().interactive_setup()"
```

**"Connection failed"**
```bash
# Check if service is running
curl http://localhost:11434/api/tags  # Ollama
curl http://localhost:1234/v1/models  # LM Studio

# Verify service is accessible
```

### Debug Information
```python
from config.setup_wizard import check_status

status = check_status()
print("Debug info:")
for key, value in status.items():
    print(f"  {key}: {value}")
```

## Environment Variables

### Required
- `PROVIDER_API_URL` - API endpoint
- `PROVIDER_API_KEY` - API key (for cloud services)
- `PROVIDER_MODEL` - Model name

### Optional (for cloud providers)
- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key  
- `GOOGLE_API_KEY` - Google API key
- `CHUTES_API_KEY` - Chutes API key

### Defaults (auto-added)
- `DB_PATH=data/conjecture.db`
- `CONFIDENCE_THRESHOLD=0.95`
- `MAX_CONTEXT_SIZE=10`
- `BATCH_SIZE=10`
- `DEBUG=false`

## Best Practices

1. **Start Local**: Use Ollama for free, private AI
2. **Secure Keys**: Never commit API keys to git
3. **Backups**: The wizard automatically backs up your .env file
4. **Validation**: Always test after configuration
5. **Environment**: Use environment variables for API keys

## Integration Examples

### For Scripts
```python
#!/usr/bin/env python3
import sys
from config.setup_wizard import check_status

status = check_status()
if not status['configured']:
    print("Conjecture not configured. Run setup first!")
    sys.exit(1)

# Your script logic here
```

### For Applications
```python
def ensure_configured():
    from config.setup_wizard import check_status, quick_setup
    
    status = check_status()
    if not status['configured']:
        if not quick_setup():
            raise Exception("Failed to configure Conjecture")
    
    return status

# Use in your app
config = ensure_configured()
print(f"Configured with {config['provider']}")
```

## Security

- API keys are masked in files and display
- .env files get secure permissions (0o600)
- Backups preserve history without exposing keys
- Git protection via .gitignore updates

## Need Help?

1. Run the demo: `python demo_setup_wizard.py`
2. Check the migration guide: `SETUP_WIZARD_MIGRATION_GUIDE.md`
3. Run tests: `python test_setup_wizard_simple.py`
4. Check archived system for advanced features: `archive/discovery/`

The Setup Wizard makes Conjecture configuration painless while preserving all the essential functionality! 🎉