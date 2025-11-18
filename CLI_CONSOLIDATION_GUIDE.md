# CLI Consolidation Migration Guide

## Overview

The Conjecture project has been successfully consolidated from 9 overlapping CLI implementations into a single modular CLI with pluggable backends. This guide helps you migrate from the old CLI files to the new unified system.

## What Changed

### Before (9 separate CLI files)
- `simple_conjecture_cli.py` - Cloud services focused
- `simple_local_cli.py` - Local services focused  
- `src/cli.py` - Data layer focused
- `src/enhanced_cli.py` - LLM integrations
- `src/full_cli.py` - Full feature set
- `src/local_cli.py` - Local services
- `src/simple_cli.py` - Basic functionality
- Plus other variants and duplicates

### After (1 unified CLI with 4 backends)
- **Single entry point**: `conjecture` command
- **Modular backends**: auto, local, cloud, hybrid
- **Consistent interface**: Same commands across all backends
- **Auto-detection**: Intelligent backend selection
- **Rich UI**: Beautiful console output with progress indicators

## Quick Start

### 1. Install and Configure
```bash
# Copy configuration template
cp .env.example .env

# Edit .env with your preferred provider
# For local services:
OLLAMA_ENDPOINT=http://localhost:11434

# For cloud services:
OPENAI_API_KEY=sk-your-key-here

# Or unified format:
PROVIDER_API_URL=http://localhost:11434
PROVIDER_MODEL=llama2
PROVIDER_API_KEY=optional-if-local
```

### 2. Validate Configuration
```bash
conjecture config
```

### 3. Create Your First Claim
```bash
conjecture create "The sky is blue" --confidence 0.9
```

### 4. Search Claims
```bash
conjecture search "blue"
```

## Command Mapping

### Basic Commands

| Old Command | New Command | Notes |
|-------------|-------------|-------|
| `python simple_conjecture_cli.py create "text"` | `conjecture create "text"` | Same functionality |
| `python simple_local_cli.py search "text"` | `conjecture search "text"` | Auto-detects optimal backend |
| `python src/cli.py get c1234567` | `conjecture get c1234567` | Uses unified database |
| `python src/simple_cli.py stats` | `conjecture stats` | Enhanced statistics |

### Backend-Specific Commands

| Use Case | Old Approach | New Approach |
|----------|--------------|--------------|
| **Local Only** | `python simple_local_cli.py` | `conjecture --backend local` |
| **Cloud Only** | `python simple_conjecture_cli.py` | `conjecture --backend cloud` |
| **Both Services** | No equivalent | `conjecture --backend hybrid` |
| **Auto-Detection** | No equivalent | `conjecture --backend auto` (default) |

### Enhanced Features

| Feature | Old Command | New Command |
|---------|-------------|-------------|
| **Health Check** | No equivalent | `conjecture health` |
| **Backend Status** | No equivalent | `conjecture backends` |
| **Provider Info** | No equivalent | `conjecture providers` |
| **Setup Guide** | `python simple_conjecture_cli.py setup` | `conjecture setup` |
| **Quick Start** | No equivalent | `conjecture quickstart` |

## Backend Selection

### Auto Backend (Default)
```bash
conjecture create "test"  # Automatically selects best backend
```
- **Intelligent selection** based on configuration
- **Local first** for privacy when available
- **Cloud fallback** for advanced features
- **Hybrid mode** when both are configured

### Local Backend
```bash
conjecture --backend local create "test"
```
- **Privacy first**: Complete data privacy
- **Offline capable**: Works without internet
- **Fast processing**: Local embeddings and search
- **No API costs**: Free to use
- **Supported providers**: Ollama, LM Studio, LocalAI

### Cloud Backend
```bash
conjecture --backend cloud create "test" --analyze
```
- **Advanced models**: GPT-4, Claude, Gemini, etc.
- **Powerful analysis**: Fact-checking and web search
- **Regular updates**: Always latest models
- **API costs**: Pay-per-use pricing
- **Requires internet**: Cloud connectivity needed

### Hybrid Backend
```bash
conjecture --backend hybrid create "test" --analyze
```
- **Best of both**: Local + cloud capabilities
- **Optimal performance**: Right tool for each task
- **Automatic fallback**: If one fails, try the other
- **Cost optimization**: Use local when possible
- **Flexible modes**: Can prioritize backend types

## Configuration Examples

### Local Provider Setup
```bash
# Option 1: Ollama
OLLAMA_ENDPOINT=http://localhost:11434

# Option 2: LM Studio  
LM_STUDIO_ENDPOINT=http://localhost:1234

# Option 3: Unified format
PROVIDER_API_URL=http://localhost:11434
PROVIDER_MODEL=llama2
```

### Cloud Provider Setup
```bash
# Option 1: OpenAI
OPENAI_API_KEY=sk-your-key-here

# Option 2: Anthropic
ANTHROPIC_API_KEY=your-key-here

# Option 3: Unified format
PROVIDER_API_URL=https://api.openai.com/v1
PROVIDER_API_KEY=sk-your-key-here
PROVIDER_MODEL=gpt-4
```

### Multiple Providers (Hybrid)
```bash
# Primary cloud provider
PROVIDER_API_URL=https://api.openai.com/v1
PROVIDER_API_KEY=sk-your-key-here
PROVIDER_MODEL=gpt-4

# Fallback local provider
OLLAMA_ENDPOINT=http://localhost:11434
```

## Advanced Usage

### Backend Override Per Command
```bash
# Use auto for most operations but force local for search
conjecture create "test" --analyze  # Uses auto (cloud preferred)
conjecture search "python" --backend local  # Forces local search
```

### Setting Preferred Mode in Hybrid
```python
from conjecture.cli.backends import HybridBackend

backend = HybridBackend()
backend.set_preferred_mode("local")  # or "cloud", "auto"
```

### Cross-Backend Analysis
```bash
# Compare results from multiple backends
conjecture --backend hybrid analyze c1234567 --cross-compare
```

## Migration Script Examples

### Existing Script Migration
**Before:**
```bash
#!/bin/bash
python simple_conjecture_cli.py create "$1" --confidence "$2"
python simple_local_cli.py search "$3"
```

**After:**
```bash
#!/bin/bash
conjecture create "$1" --confidence "$2"
conjecture search "$3"
```

### Docker Integration
**Before:**
```dockerfile
RUN python simple_conjecture_cli.py create "test"
```

**After:**
```dockerfile
RUN conjecture create "test"
```

### CI/CD Pipeline
**Before:**
```yaml
- name: Test CLI
  run: python src/cli.py create "test" --confidence 0.8
```

**After:**
```yaml
- name: Test CLI  
  run: conjecture create "test" --confidence 0.8
```

## Troubleshooting

### Import Errors
```
Error: No module named 'cli.backends'
```
**Solution:** Ensure you're running from project root or use:
```bash
python -m src.cli.modular_cli
```

### Backend Not Available
```
Error: Local backend is not properly configured
```
**Solution:** Check configuration:
```bash
conjecture config
conjecture setup
```

### Auto-Detection Issues
```bash
# Force specific backend
conjecture --backend local create "test"

# Check what's available
conjecture backends
conjecture health
```

### Database Issues
```bash
# Check database location
conjecture stats

# Fresh start (optional)
rm data/conjecture.db
conjecture create "test" --confidence 0.8
```

## Performance Tips

### For Privacy/Offline
```bash
# Use local backend exclusively
conjecture --backend local create "test"
conjecture --backend local search "python"
```

### For Advanced Analysis
```bash
# Use cloud for powerful analysis
conjecture --backend cloud create "test" --analyze
conjecture --backend cloud analyze c1234567
```

### For Cost Optimization
```bash
# Use hybrid to minimize costs
conjecture --backend hybrid search "python"  # Local search
conjecture --backend hybrid analyze c1234567  # Cloud analysis
```

### For Best Performance
```bash
# Let auto select optimal backend
conjecture create "test"  # Auto-detects
conjecture search "test"  # Uses local for speed
```

## File Structure

### New CLI Structure
```
src/cli/
├── __init__.py              # Package initialization
├── base_cli.py              # Base functionality
├── modular_cli.py           # Main CLI entry point
└── backends/                # Backend implementations
    ├── __init__.py         # Backend registry
    ├── local_backend.py    # Local services
    ├── cloud_backend.py    # Cloud services  
    ├── hybrid_backend.py   # Combined services
    └── auto_backend.py     # Auto-detection

Entry Points:
├── conjecture                # Main executable
├── simple_conjecture_cli.py # Redirector
├── simple_local_cli.py     # Redirector
└── src/*.py (old files)    # All redirect to new CLI
```

## Testing

### System Health Check
```bash
conjecture health
```

### Configuration Validation
```bash
conjecture config
conjecture providers
```

### Backend Testing
```bash
conjecture backends
conjecture --backend local stats
conjecture --backend cloud stats
```

### Functional Testing
```bash
# Test all basic operations
conjecture create "test claim for validation" --confidence 0.9 --user test
conjecture get c1234567
conjecture search "validation"
conjecture stats
```

## Support

### Getting Help
```bash
conjecture --help
conjecture quickstart
conjecture setup
```

### Checking Status
```bash
conjecture health    # Overall system status
conjecture backends  # Available backends
conjecture config    # Configuration status
```

### Migration Assistance
```bash
# Show equivalent commands
python simple_conjecture_cli.py

# Auto-redirect with hints  
python simple_local_cli.py

# Direct new CLI usage
conjecture --help
```

## Benefits Achieved

✅ **Reduced Complexity**: From 9 CLI files to 1 unified system  
✅ **Consistent Interface**: Same commands work across all backends  
✅ **Auto-Detection**: Intelligent backend selection  
✅ **Enhanced UX**: Rich console output and progress indicators  
✅ **Better Error Handling**: Comprehensive error messages and recovery  
✅ **Flexible Architecture**: Easy to add new backends  
✅ **Backward Compatibility**: Old CLI files redirect with guidance  
✅ **Comprehensive Testing**: Full test coverage for all components  
✅ **Clear Documentation**: Migration guides and help systems  

## Future Enhancements

- 🔄 **Cross-backend synchronization**  
- 🔌 **Plugin system for custom backends**  
- 📊 **Performance monitoring and analytics**  
- 🌐 **Web interface integration**  
- 📱 **Mobile app connectivity**  
- 🔧 **Configuration management GUI**  

---

This consolidation represents a significant improvement in maintainability, usability, and extensibility while preserving all existing functionality and providing clear migration paths for users.