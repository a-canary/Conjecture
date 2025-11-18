# Unified Configuration Validator Documentation

## Overview

The Unified Configuration Validator consolidates 4 overlapping configuration validators into a single, intelligent system that automatically detects and handles multiple configuration formats while maintaining full backward compatibility.

## Supported Configuration Formats

### 1. Unified Provider Format (Recommended - Highest Priority)
```bash
PROVIDER_API_URL=http://localhost:11434
PROVIDER_API_KEY=
PROVIDER_MODEL=llama2
```

**Advantages:**
- Simple and clean
- Single active provider
- Easy to understand
- Recommended for new installations

### 2. Simple Provider Format (High Priority)
```bash
PROVIDER_OLLAMA=http://localhost:11434,,llama2,ollama
PROVIDER_OPENROUTER=https://openrouter.ai/api/v1,sk-key,openai/gpt-3.5-turbo,openai
```

**Format:** `PROVIDER_[NAME]=[BASE_URL],[API_KEY],[MODEL],[PROTOCOL]`

**Advantages:**
- Support for multiple providers
- Priority-based selection
- Good for testing multiple services

### 3. Individual Environment Format (Medium Priority)
```bash
OLLAMA_API_URL=http://localhost:11434
OLLAMA_API_KEY=
OLLAMA_MODELS=["llama2", "mistral", "codellama"]
```

**Format:** `[PROVIDER]_API_URL`, `[PROVIDER]_API_KEY`, `[PROVIDER]_MODELS`

**Advantages:**
- Multiple models per provider
- Explicit variable names
- Good for complex setups

### 4. Simple Validator Format (Low Priority - Legacy)
```bash
OLLAMA_ENDPOINT=http://localhost:11434
OLLAMA_MODEL=llama2
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-3.5-turbo
```

**Format:** Provider-specific variables

**Advantages:**
- Most explicit
- Individual variables per provider
- Legacy compatibility

## Priority-Based Selection

When multiple formats are detected, the system uses this priority order:

1. **Unified Provider Format** (Priority: 1) - Recommended
2. **Simple Provider Format** (Priority: 2)
3. **Individual Environment Format** (Priority: 3)
4. **Simple Validator Format** (Priority: 4) - Legacy

## Usage Examples

### Basic Validation
```python
from src.config import validate_config, show_configuration_status

# Validate current configuration
result = validate_config()
if result.success:
    print("Configuration is valid!")
else:
    print("Configuration errors:", result.errors)

# Show detailed status
show_configuration_status(detailed=True)
```

### Getting Primary Provider
```python
from src.config import get_primary_provider

provider = get_primary_provider()
if provider:
    print(f"Using: {provider.name} ({provider.model})")
    print(f"URL: {provider.base_url}")
    print(f"Type: {'Local' if provider.is_local else 'Cloud'}")
```

### Working with Specific Formats
```python
from src.config.unified_validator import UnifiedConfigValidator
from src.config import ConfigFormat

validator = UnifiedConfigValidator()

# Detect which formats are present
formats = validator.detect_formats()
print(f"Detected formats: {[f.value for f in formats]}")

# Get active format
active = validator.get_active_format()
print(f"Active format: {active.value}")

# Access all providers across all formats
all_providers = validator.get_all_providers()
for format_type, providers in all_providers.items():
    print(f"{format_type.value}: {len(providers)} providers")
```

## Migration Guide

### Automatic Migration Analysis
```python
from src.config.migration_utils import analyze_migration, show_migration_analysis

# Analyze current configuration
analysis = analyze_migration()
print(f"Current format: {analysis['current_format'].value}")
print(f"Migration difficulty: {analysis['migration_difficulty']}")

# Show detailed analysis
show_migration_analysis()
```

### Dry Run Migration
```python
from src.config.migration_utils import execute_migration

# Test migration without making changes
result = execute_migration(dry_run=True)
if result['success']:
    print("Migration would succeed:")
    for change in result['changes']:
        print(f"  {change}")
```

### Executing Migration
```python
# Execute actual migration (with backup)
result = execute_migration(dry_run=False)
if result['success']:
    print("Migration completed successfully!")
    print(f"Backup created: {result['backup_created']}")
else:
    print("Migration failed:", result['errors'])
```

## Provider Configuration Examples

### Local Providers

#### Ollama
```bash
# Unified Format (Recommended)
PROVIDER_API_URL=http://localhost:11434
PROVIDER_API_KEY=
PROVIDER_MODEL=llama2

# Alternative Formats
PROVIDER_OLLAMA=http://localhost:11434,,llama2,ollama
OLLAMA_API_URL=http://localhost:11434
OLLAMA_API_KEY=
OLLAMA_MODELS=["llama2", "mistral", "codellama"]
OLLAMA_ENDPOINT=http://localhost:11434
OLLAMA_MODEL=llama2
```

#### LM Studio
```bash
# Unified Format
PROVIDER_API_URL=http://localhost:1234/v1
PROVIDER_API_KEY=
PROVIDER_MODEL=microsoft/DialoGPT-medium

# Alternative Formats
PROVIDER_LM_STUDIO=http://localhost:1234/v1,,microsoft/DialoGPT-medium,openai
LM_STUDIO_API_URL=http://localhost:1234/v1
LM_STUDIO_API_KEY=
LM_STUDIO_MODELS=["microsoft/DialoGPT-medium", "microsoft/DialoGPT-large"]
LM_STUDIO_ENDPOINT=http://localhost:1234/v1
LM_STUDIO_MODEL=microsoft/DialoGPT-medium
```

### Cloud Providers

#### OpenAI
```bash
# Unified Format
PROVIDER_API_URL=https://api.openai.com/v1
PROVIDER_API_KEY=sk-your-openai-key-here
PROVIDER_MODEL=gpt-3.5-turbo

# Alternative Formats
PROVIDER_OPENAI=https://api.openai.com/v1,sk-your-openai-key-here,gpt-3.5-turbo,openai
OPENAI_API_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-your-openai-key-here
OPENAI_MODELS=["gpt-3.5-turbo", "gpt-4"]
OPENAI_API_KEY=sk-your-openai-key-here
OPENAI_MODEL=gpt-3.5-turbo
```

#### Anthropic Claude
```bash
# Unified Format
PROVIDER_API_URL=https://api.anthropic.com
PROVIDER_API_KEY=sk-ant-your-key-here
PROVIDER_MODEL=claude-3-haiku-20240307

# Alternative Formats
PROVIDER_ANTHROPIC=https://api.anthropic.com,sk-ant-your-key-here,claude-3-haiku-20240307,anthropic
ANTHROPIC_API_URL=https://api.anthropic.com
ANTHROPIC_API_KEY=sk-ant-your-key-here
ANTHROPIC_MODELS=["claude-3-haiku-20240307", "claude-3-sonnet-20240229"]
ANTHROPIC_API_KEY=sk-ant-your-key-here
ANTHROPIC_MODEL=claude-3-haiku-20240307
```

#### OpenRouter
```bash
# Unified Format
PROVIDER_API_URL=https://openrouter.ai/api/v1
PROVIDER_API_KEY=sk-or-your-key-here
PROVIDER_MODEL=openai/gpt-3.5-turbo

# Alternative Formats
PROVIDER_OPENROUTER=https://openrouter.ai/api/v1,sk-or-your-key-here,openai/gpt-3.5-turbo,openai
OPENROUTER_API_URL=https://openrouter.ai/api/v1
OPENROUTER_API_KEY=sk-or-your-key-here
OPENROUTER_MODELS=["openai/gpt-3.5-turbo", "anthropic/claude-3-haiku"]
OPENROUTER_API_KEY=sk-or-your-key-here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=openai/gpt-3.5-turbo
```

## Advanced Usage

### Custom Adapter Development
```python
from src.config.adapters import BaseAdapter, ProviderConfig, ValidationResult, FormatPriority

class CustomAdapter(BaseAdapter):
    def __init__(self):
        super().__init__(format_type="custom", priority=FormatPriority.MEDIUM)
    
    def detect_format(self, env_vars):
        return "CUSTOM_VAR" in env_vars
    
    def load_providers(self, env_vars):
        # Custom provider loading logic
        pass
    
    def validate_format(self, env_vars):
        # Custom validation logic
        pass

# Register with unified validator
validator = UnifiedConfigValidator()
validator.adapters.append(CustomAdapter())
```

### Validation Notifications
```python
from src.config.unified_validator import UnifiedConfigValidator

validator = UnifiedConfigValidator()

# Enable deprecation warnings
validator.config.show_deprecation_warnings = True

# Monitor API usage
validator.config.log_api_usage = True

# Custom error handling
result = validator.validate_configuration()
if not result.success:
    for error in result.errors:
        # Custom error processing
        pass
```

### Configuration Export
```python
from src.config.unified_validator import UnifiedConfigValidator, ConfigFormat

validator = UnifiedConfigValidator()

# Export to different formats
unified_config = validator.export_configuration(ConfigFormat.UNIFIED_PROVIDER)
simple_config = validator.export_configuration(ConfigFormat.SIMPLE_PROVIDER)

print("Unified format export:")
for key, value in unified_config.items():
    print(f"  {key}={value}")
```

## Troubleshooting

### Common Issues

#### Format Conflicts
```
Issue: Multiple configuration formats detected
Solution: The system automatically uses the highest priority format. Consider migrating to unified format.
```

#### Missing Provider
```
Issue: No providers configured
Solution: Configure at least one provider in any supported format
```

#### Invalid URL
```
Issue: URL must start with http:// or https://
Solution: Ensure all URLs include the protocol
```

#### API Key Required
```
Issue: API key is required for cloud services
Solution: Add API key for cloud providers or use local services
```

### Debug Information
```python
from src.config.unified_validator import UnifiedConfigValidator

validator = UnifiedConfigValidator()

# Show detailed status
validator.show_configuration_status(detailed=True)

# Check API usage
from src.config.backward_compatibility import check_api_usage
usage_info = check_api_usage()
print("API usage:", usage_info)

# Migration suggestions
from src.config.backward_compatibility import show_migration_suggestions
show_migration_suggestions()
```

## Testing

### Running Tests
```bash
# Run comprehensive test suite
python -m pytest tests/test_unified_validator.py -v

# Run specific test groups
python -m pytest tests/test_unified_validator.py::TestUnifiedConfigValidator -v
python -m pytest tests/test_unified_validator.py::TestMigration -v
```

### Test Coverage
The test suite covers:
- All 4 configuration formats
- Format detection and priority handling
- Provider loading and validation
- Migration utilities
- Backward compatibility
- Error handling and edge cases
- Integration scenarios

## Backward Compatibility

The unified validator maintains full backward compatibility:

```python
# Old validators still work (with deprecation warnings)
from src.config import SimpleProviderValidator, IndividualEnvValidator

# Old function names are preserved
from src.config.simple_validator import validate_config, print_setup_instructions

# Automatic mapping to new system
validator = SimpleProviderValidator()  # Now uses unified validator internally
```

### Migration Path
1. **Phase 1**: Use old validators with deprecation warnings
2. **Phase 2**: Update imports to use unified validator
3. **Phase 3**: Migrate configuration to unified format
4. **Phase 4**: Remove old validators (future release)

## Security Considerations

### API Key Protection
```python
from pathlib import Path

# Check if .env is protected
gitignore_path = Path(".gitignore")
if not gitignore_path.exists() or ".env" not in gitignore_path.read_text():
    print("⚠️  Add .env to .gitignore for security!")
```

### Local vs Cloud
- Local services (Ollama, LM Studio): No API key required
- Cloud services (OpenAI, Anthropic): API key required
- Mixed environments supported with proper validation

## Performance Optimization

### Caching
```python
validator = UnifiedConfigValidator()

# Results are cached automatically
result1 = validator.validate_configuration()
result2 = validator.validate_configuration()  # Returns cached result

# Clear cache when needed
validator.clear_cache()
```

### Memory Usage
- Adapters are created once and reused
- Environment variables loaded once per validator instance
- Validation results cached until cache is cleared

## Best Practices

### Recommended Configuration
1. Use unified provider format for new installations
2. Configure only one provider for simplicity
3. Use local services when possible for privacy
4. Always test configuration after changes

### Development Workflows
1. Use `validate_config()` before operations
2. Check `get_primary_provider()` for current selection
3. Monitor format conflicts during development
4. Use migration tools for format changes

### Production Deployment
1. Validate configuration at startup
2. Use backup and rollback procedures
3. Monitor for format conflicts
4. Plan migration path for legacy configurations

---

## API Reference

### Core Classes

- `UnifiedConfigValidator`: Main validator class
- `ProviderConfig`: Provider configuration data structure
- `UnifiedValidationResult`: Comprehensive validation result
- `ConfigFormat`: Enum for supported formats
- `ConfigMigrator`: Migration utilities

### Key Functions

- `validate_config()`: Validate all configurations
- `get_primary_provider()`: Get active provider
- `show_configuration_status()`: Display configuration status
- `analyze_migration()`: Analyze migration options
- `execute_migration()`: Execute migration with backup

### Compatibility Functions

- `SimpleProviderValidator()`: Backward compatible wrapper
- `IndividualEnvValidator()`: Backward compatible wrapper
- `UnifiedProviderValidator()`: Backward compatible wrapper
- `SimpleValidator()`: Backward compatible wrapper

For detailed API documentation, see the inline docstrings in the source code.