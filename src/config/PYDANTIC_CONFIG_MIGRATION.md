# Pydantic Configuration System Migration

This document describes the migration from the old configuration system to the new Pydantic-based configuration system.

## Overview

The Conjecture configuration system has been successfully migrated to use Pydantic models for type safety, validation, and better structure. This migration removes environment variable support as requested and focuses on pure JSON-based configuration with a workspace → user → default hierarchy.

## What Changed

### 1. New Pydantic Settings Models

**File: `src/config/settings_models.py`**

- **ProviderConfig**: Converted from dataclass to Pydantic BaseModel with validation
- **ConjectureSettings**: Main settings class using Pydantic BaseModel
- **Specialized Settings Classes**: DatabaseSettings, LLMSettings, ProcessingSettings, DirtyFlagSettings, LoggingSettings, WorkspaceSettings

### 2. New Configuration Loader

**File: `src/config/pydantic_config.py`**

- **PydanticConfig**: New configuration loader that handles workspace → user → default config hierarchy
- **ConfigHierarchy**: Manages configuration file loading and merging
- Removed all environment variable support per requirements

### 3. Updated Main Config Class

**File: `src/config/config.py`**

- Updated to use PydanticConfig internally
- Maintains backward compatibility with existing property interface
- All existing methods preserved for compatibility

### 4. Key Features

#### Type Safety
- All configuration now uses Pydantic models with automatic validation
- Type hints throughout the codebase
- Runtime validation of configuration values

#### Validation
- Provider URLs, names, and required fields validated
- Confidence thresholds and numeric ranges validated
- Log levels and file paths validated

#### Structure
- Modular settings classes for different concerns (database, LLM, processing, etc.)
- Clear separation of concerns with dedicated settings classes
- Hierarchical configuration with proper merging

#### No Environment Variables
- Removed all environment variable support as requested
- Pure JSON-based configuration only
- Cleaner, more predictable configuration behavior

## Usage

### Basic Usage

```python
from src.config.config import Config

# Load configuration with full Pydantic validation
config = Config()

# Access settings with type safety
print(f"Confidence threshold: {config.confidence_threshold}")
print(f"Primary provider: {config.get_primary_provider()['name']}")
print(f"Database path: {config.database_path}")
```

### Advanced Usage

```python
from src.config.pydantic_config import PydanticConfig
from src.config.settings_models import ConjectureSettings, ProviderConfig

# Direct access to Pydantic settings
pydantic_config = PydanticConfig()
settings = pydantic_config.settings

# Create custom provider
provider = ProviderConfig(
    name="custom",
    url="http://localhost:8080",
    api="custom_key",
    model="custom_model",
    priority=1
)

settings.add_provider(provider)
pydantic_config.save_settings('user')
```

## Migration Guide

### For Existing Code

Existing code should continue to work without changes:

```python
from src.config.config import Config

config = Config()
# All existing code continues to work
```

### For New Code

New code can take advantage of Pydantic features:

```python
from src.config.pydantic_config import PydanticConfig
from src.config.settings_models import ConjectureSettings

config = PydanticConfig()
settings = config.settings

# Type-safe access to all settings
print(f"Debug mode: {settings.debug}")
print(f"LLM temperature: {settings.llm.temperature}")
```

## Configuration File Format

The new system maintains the same JSON configuration file format:

```json
{
  "providers": [
    {
      "url": "http://localhost:11434",
      "api": "",
      "model": "llama2",
      "name": "ollama"
    }
  ],
  "confidence_threshold": 0.95,
  "confident_threshold": 0.8,
  "max_context_size": 10,
  "batch_size": 10,
  "debug": false,
  "database_path": "data/conjecture.db",
  "user": "user",
  "team": "default"
}
```

## Validation

The new system provides comprehensive validation:

- Provider configuration validation (URLs, names, required fields)
- Numeric range validation (confidence thresholds, batch sizes)
- String validation (log levels, file paths)
- Type safety with Pydantic models

## Backward Compatibility

The migration maintains 100% backward compatibility:

- All existing `Config` class methods work unchanged
- Property-based access to configuration values
- Same return types and interfaces

## Testing

Run the test script to verify the migration:

```bash
python simple_test.py
```

## Benefits

1. **Type Safety**: Compile-time validation prevents configuration errors
2. **Better Structure**: Modular design with clear separation of concerns
3. **Validation**: Automatic validation ensures configuration integrity
4. **Documentation**: Self-documenting models with clear field descriptions
5. **IDE Support**: Better autocompletion and type hints
6. **Maintainability**: Easier to modify and extend configuration

## Notes

- Environment variable support has been completely removed as requested
- The system now relies purely on JSON configuration files
- All validation happens at startup through Pydantic models
- Configuration is immutable after creation (frozen=False allows modification)