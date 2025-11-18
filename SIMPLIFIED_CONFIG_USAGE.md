# Simplified Configuration System Usage Guide

## Overview

The simplified configuration system replaces complex auto-discovery with a clean, documented approach that gives users full control while providing helpful guidance.

## Quick Start

### 1. Setup Configuration

```bash
# Copy the comprehensive template
cp .env.example .env

# Edit the configuration file
nano .env  # or use your preferred editor
```

### 2. Configure at Least ONE Provider

**For Local Privacy (Recommended):**
```bash
# Ollama (easiest local option)
OLLAMA_ENDPOINT=http://localhost:11434
OLLAMA_MODEL=llama2
```

**For Cloud Access:**
```bash
# OpenAI
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-3.5-turbo
```

### 3. Test Configuration

```bash
# Validate your setup
python simple_conjecture_cli.py validate

# Check what's configured
python simple_conjecture_cli.py config

# If validation passes, you're ready!
python simple_conjecture_cli.py create "Test claim" --confidence 0.8
```

## Available Commands

### Primary CLI: `simple_conjecture_cli.py`

```bash
# Configuration Commands
python simple_conjecture_cli.py config        # Show configuration status
python simple_conjecture_cli.py validate      # Validate configuration
python simple_conjecture_cli.py setup         # Interactive setup help
python simple_conjecture_cli.py providers      # Show all providers and status

# Core Functionality
python simple_conjecture_cli.py create "Your claim here" --confidence 0.8
python simple_conjecture_cli.py get <claim-id>
python simple_conjecture_cli.py search "query terms"
python simple_conjecture_cli.py stats         # Database statistics

# Help and Getting Started
python simple_conjecture_cli.py quickstart    # New user guide
python simple_conjecture_cli.py --help        # All available commands
```

### Existing CLI: `simple_local_cli.py` (Updated)

```bash
# New simplified configuration commands
python simple_local_cli.py simple_config     # Show configuration status
python simple_local_cli.py simple_setup      # Setup instructions
python simple_local_cli.py simple_validate   # Validate configuration

# All existing commands still work
python simple_local_cli.py create "Your claim"
python simple_local_cli.py search "query"
python simple_local_cli.py health
```

## Provider Configuration

### Priority Order

The system automatically selects the best available provider:

1. **Ollama** (Local, Private, Offline)
2. **LM Studio** (Local, Private, Offline)
3. **OpenAI** (Cloud, API Key Required)
4. **Anthropic Claude** (Cloud, API Key Required)
5. **Google Gemini** (Cloud, API Key Required)
6. **Cohere** (Cloud, API Key Required)

### Local Providers Setup

#### Ollama (Recommended)

```bash
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Start the service
ollama serve

# 3. Pull a model
ollama pull llama2

# 4. Configure in .env
echo "OLLAMA_ENDPOINT=http://localhost:11434" >> .env
echo "OLLAMA_MODEL=llama2" >> .env
```

#### LM Studio

```bash
# 1. Download and install LM Studio
# https://lmstudio.ai/

# 2. Launch the application
# 3. Go to "Server" tab
# 4. Start the server
# 5. Configure in .env
echo "LM_STUDIO_ENDPOINT=http://localhost:1234/v1" >> .env
```

### Cloud Providers Setup

#### OpenAI

```bash
# 1. Get API key
# Visit: https://platform.openai.com/api-keys

# 2. Configure in .env
echo "OPENAI_API_KEY=sk-your-api-key-here" >> .env
echo "OPENAI_MODEL=gpt-3.5-turbo" >> .env
```

#### Anthropic Claude

```bash
# 1. Get API key
# Visit: https://console.anthropic.com/

# 2. Configure in .env
echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" >> .env
echo "ANTHROPIC_MODEL=claude-3-haiku-20240307" >> .env
```

## Configuration File Structure

The `.env.example` file is comprehensively documented with:

- **Clear sections** for different provider types
- **Setup instructions** for each provider
- **Example configurations** with realistic values
- **Security guidelines** and best practices
- **Troubleshooting tips** for common issues
- **Advanced settings** for power users

## Error Messages and Guidance

### No Configuration Found

```
❌ Configuration Required
No providers configured. Please configure at least one provider.

📋 Quick Setup:
1. Copy template: cp .env.example .env
2. Edit .env with your preferred provider
3. Try again: python simple_conjecture_cli.py

💡 Need help?
Run: python simple_conjecture_cli.py setup
Run: python simple_conjecture_cli.py providers
```

### Partial Configuration

```
⚠️ Warnings:
• Ollama partially configured: missing OLLAMA_MODEL
• .env file is protected by .gitignore ✓
```

### Successful Configuration

```
✅ Configuration Validation: PASSED

Configuration Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Primary Provider: Ollama
Type: Local
Priority: 1
```

## Testing Your Setup

### Run the Test Suite

```bash
python test_simple_config.py
```

This tests:
- ✅ Configuration validation logic
- ✅ .env.example file completeness
- ✅ CLI module import and structure
- ✅ Error handling and messaging

### Manual Validation

```bash
# 1. Check configuration status
python simple_conjecture_cli.py config

# 2. Validate setup
python simple_conjecture_cli.py validate

# 3. Test basic functionality
python simple_conjecture_cli.py create "Test claim" --confidence 0.8

# 4. Test search
python simple_conjecture_cli.py search "Test"

# 5. Check database
python simple_conjecture_cli.py stats
```

## Migration from Discovery System

If you were using the old discovery system:

```bash
# 1. Backup existing configuration
cp .env .env.backup

# 2. Get the new template
cp .env.example .env

# 3. Copy your existing API keys to the new format
# OLD: Conjecture_LLM_API_KEY=sk-xxx
# NEW: OPENAI_API_KEY=sk-xxx

# 4. Test the new system
python simple_conjecture_cli.py validate
```

## Benefits of the Simplified System

### ✅ User Control
- Users explicitly choose their provider
- No silent auto-configuration
- Clear visibility of what's configured

### ✅ Better Documentation
- Comprehensive inline documentation
- Step-by-step setup instructions
- Example configurations for each provider

### ✅ Simplified Codebase
- Replaces 1000+ lines of discovery code
- Less complexity, more maintainable
- Easier to debug and extend

### ✅ Clear Error Messages
- Specific guidance for each issue
- Next steps clearly outlined
- Consistent formatting and help

### ✅ Security Best Practices
- .env protection validation
- API key handling guidance
- No hardcoded credentials

## Troubleshooting

### Common Issues

**Problem:** "No providers configured"
**Solution:** Configure at least one provider in .env

**Problem:** "Ollama connection failed"
**Solution:** 
```bash
ollama serve
ollama pull llama2
```

**Problem:** "sentence-transformers not installed"
**Solution:** `pip install sentence-transformers`

**Problem:** "Permission denied"
**Solution:** Check file permissions and .gitignore

### Getting Help

```bash
# Quick start guide
python simple_conjecture_cli.py quickstart

# Setup help
python simple_conjecture_cli.py setup

# Provider-specific help
python simple_conjecture_cli.py setup --provider ollama

# Configuration status
python simple_conjecture_cli.py config
```

## File Structure

```
D:\projects\Conjecture\
├── .env.example              # Comprehensive configuration template
├── .env                      # Your actual configuration (create this)
├── simple_conjecture_cli.py  # New simplified CLI
├── simple_local_cli.py       # Updated existing CLI
├── src/config/
│   └── simple_validator.py   # Configuration validation logic
├── test_simple_config.py     # Test suite
└── SIMPLIFIED_CONFIG_USAGE.md # This guide
```

## Next Steps

1. **Try the quickstart:**
   ```bash
   python simple_conjecture_cli.py quickstart
   ```

2. **Configure your preferred provider**
3. **Test with a claim:**
   ```bash
   python simple_conjecture_cli.py create "The sky appears blue due to Rayleigh scattering" --confidence 0.9
   ```

4. **Explore the features:**
   ```bash
   python simple_conjecture_cli.py --help
   ```

The simplified configuration system is designed to be intuitive, well-documented, and easy to troubleshoot. Happy coding!