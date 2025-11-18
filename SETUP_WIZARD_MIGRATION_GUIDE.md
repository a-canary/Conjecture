# Setup Wizard Migration Guide

## Overview

The complex provider discovery system has been replaced with a simple, intuitive Setup Wizard that follows the 80/20 rule - covering 90% of user needs with 20% of the complexity.

## What Changed

### Old System (Archived)
- **Location**: `src/discovery/` → moved to `archive/discovery/`
- **Complexity**: ~1000 lines of async code
- **Features**: Multiple discovery modes, complex orchestration
- **Files**: 
  - `provider_discovery.py` (485 lines)
  - `service_detector.py` (~300 lines) 
  - `config_updater.py` (~250 lines)

### New System
- **Location**: `src/config/setup_wizard.py`
- **Complexity**: ~200 lines of synchronous code
- **Features**: Simple 3-step interactive setup
- **Focus**: Common use cases (Ollama, OpenAI, Anthropic, etc.)

## Quick Start

### For Users

```python
# Check if configured
from config.setup_wizard import check_status
status = check_status()
if not status['configured']:
    print("Not configured yet")

# Quick setup (interactive if needed)
from config.setup_wizard import quick_setup
quick_setup()  # Returns True if successful

# Auto-setup Ollama if detected
from config.setup_wizard import auto_setup_ollama
if auto_setup_ollama():
    print("✅ Ollama configured!")
```

### Interactive Setup

```python
from config.setup_wizard import SetupWizard

wizard = SetupWizard()
wizard.interactive_setup()  # 3-step wizard
```

## Migration Reference

| Old System | New System | Purpose |
|------------|------------|---------|
| `ProviderDiscovery.run_automatic_discovery()` | `quick_setup()` | Auto-configure |
| `ProviderDiscovery.run_manual_discovery()` | `SetupWizard().interactive_setup()` | Interactive setup |
| `ProviderDiscovery.quick_check()` | `SetupWizard().quick_status()` | Check status |
| `ServiceDetector.detect_all()` | `SetupWizard().auto_detect_local()` | Detect local providers |
| `ConfigUpdater.update_config_with_providers()` | `SetupWizard().update_env_file()` | Update configuration |

## Key Improvements

### 1. Simplicity
- **Old**: Complex async patterns, multiple discovery modes
- **New**: Simple synchronous methods, clear single-purpose functions

### 2. User Experience
- **Old**: Technical output, complex confirmation flows
- **New**: Friendly 3-step wizard, clear guidance

### 3. Maintenance
- **Old**: 1000+ lines, hard to debug
- **New**: 200 lines, easy to understand and modify

### 4. Reliability
- **Old**: Complex error handling in async context
- **New**: Simple try/catch with clear error messages

## Usage Examples

### Basic Configuration Check
```python
from config.setup_wizard import check_status

status = check_status()
if status['configured']:
    print(f"Using {status['provider']} with {status['model']}")
else:
    print("Setup required")
```

### Auto-Configure Ollama
```python
from config.setup_wizard import auto_setup_ollama

success = auto_setup_ollama()
if success:
    print("Ollama configured automatically!")
else:
    print("Ollama not available or setup failed")
```

### Manual Provider Setup
```python
from config.setup_wizard import SetupWizard

wizard = SetupWizard()

# Check for local providers
local = wizard.auto_detect_local()
print(f"Local services: {local}")

# Configure manually
config = {
    'Conjecture_LLM_PROVIDER': 'openai',
    'Conjecture_LLM_API_URL': 'https://api.openai.com/v1',
    'Conjecture_LLM_MODEL': 'gpt-3.5-turbo',
    'OPENAI_API_KEY': 'your-api-key-here'
}

success = wizard.update_env_file(config)
```

## File Changes

### New Files
- `src/config/setup_wizard.py` - Main wizard implementation
- `tests/test_setup_wizard.py` - Comprehensive tests
- `demo_setup_wizard.py` - Demo script
- `test_setup_wizard_simple.py` - Simple test runner

### Moved Files
- `src/discovery/*` → `archive/discovery/` (preserved for reference)

### Updated Files
- Old demo files preserved with original content
- New demo files created with updated examples

## Testing

### Run Comprehensive Tests
```bash
python tests/test_setup_wizard.py
```

### Run Simple Tests
```bash
python test_setup_wizard_simple.py
```

### Demo the Wizard
```bash
python demo_setup_wizard.py
python demo_simple_auto_configure.py
```

## Recovery

If you need to restore the old discovery system:

1. Move files back:
   ```bash
   mv archive/discovery/* src/discovery/
   ```

2. Update imports in your code

3. Restore any CLI integration

## Design Philosophy

### 80/20 Rule Focus
The wizard covers the 90% of common use cases:
- Ollama setup
- OpenAI/Anthropic configuration
- Basic local service detection
- Simple .env file management

### What Was Simplified
- Removed async complexity (not needed for simple service checks)
- Eliminated multiple discovery modes (confusing for users)
- Simplified configuration merging (direct updates)
- Removed complex confirmation flows (wizard handles this)

### What Was Preserved
- All essential functionality
- Security features (API key masking, secure files)
- Backup protection
- Error handling
- Provider validation

## Common Questions

### Q: Can I still use async features?
A: No, the wizard is synchronous by design for simplicity. If you need async, use the archived system.

### Q: What about advanced discovery features?
A: Advanced features are in `archive/discovery/` for power users. Most users don't need them.

### Q: How do I add a new provider?
A: Add a `SimpleProvider` to the `providers` dict in `setup_wizard.py`. Much simpler than before!

### Q: Is my old configuration compatible?
A: Yes, the wizard reads existing .env files perfectly.

## Success Metrics

- **Complexity**: 1000+ lines → 200 lines (80% reduction)
- **Setup time**: Complex → 3 simple steps
- **Maintenance**: Hard → Easy
- **User experience**: Technical → Friendly
- **Reliability**: Complex error handling → Simple, clear

The new system provides the same essential functionality with dramatically improved maintainability and user experience.