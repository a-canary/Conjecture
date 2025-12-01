# Simple Architecture Specification

## Overview

Conjecture uses a simple, elegant architecture based on a single unified API. After a major refactoring in November 2025, the codebase achieved an 87% reduction in duplicate data classes while maintaining all functionality. No over-engineering, no complex service layers - just clean, direct functionality.

## Core Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Interfaces Layer                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │
│  │   CLI   │  │  Agent  │  │  LLM    │  │ Local   │      │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                 Processing Layer                            │
│         ┌─────────────────────────────────────┐           │
│         │  processing/ (core logic)           │           │
│         └─────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Data Layer                                │
│        ┌─────────────────────────────────────┐           │
│        │    core/ │ config/ │ agent/ │ llm/ │        │
│        └─────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

## Key Principles

### 1. Single Unified API
- One `Conjecture` class provides all functionality
- All interfaces use the same API
- No interface-specific business logic

### 2. No Over-Engineering
- No service layers, no dependency injection
- Direct API calls from all interfaces
- Simple, maintainable code

### 3. Clean Separation
- Data models: `src/core/models.py` (single source of truth)
- Unified configs: `src/config/common.py`, `src/processing/llm/common.py`
- Processing: `src/conjecture.py` (single class)
- Interfaces: `src/cli/`, `src/interfaces/`

## Implementation Pattern

### All Interfaces Follow This Pattern:
```python
from contextflow import Conjecture

class YourInterface:
    def __init__(self):
        self.cf = Conjecture()  # Single API instance
    
    def your_method(self):
        # Direct API usage - no abstraction needed
        results = self.cf.explore("your query")
        claim = self.cf.add_claim("content", 0.8, "concept")
        return results, claim
```

### CLI Example:
```python
# src/cli/modular_cli.py
from contextflow import Conjecture

def main():
    cf = Conjecture()
    result = cf.explore(args.query, max_claims=args.limit)
    print(result.summary())
```

### TUI Example:
```python
# src/interfaces/simple_tui.py
from contextflow import Conjecture

class TUIApp:
    def __init__(self):
        self.cf = Conjecture()
    
    def search_screen(self):
        query = self.get_input("Search: ")
        results = self.cf.explore(query)
        self.display_results(results)
```

### GUI Example:
```python
# src/interfaces/simple_gui.py
from contextflow import Conjecture

class GUIApp:
    def __init__(self):
        self.cf = Conjecture()
    
    def on_search_button(self):
        results = self.cf.explore(self.search_input.get())
        self.populate_results(results)
```

## Benefits

### 1. Simplicity
- Easy to understand and maintain
- No complex abstractions to learn
- Clear responsibility boundaries

### 2. Consistency
- All interfaces work the same way
- Single source of truth for functionality
- No duplication of business logic

### 3. Flexibility
- Easy to add new interfaces
- Direct data access when needed
- Minimal coupling between layers

### 4. Performance
- No unnecessary abstraction overhead
- Direct API calls
- Efficient resource usage

## When to Break the Pattern

### Direct Data Access (Rare Cases):
```python
# For performance-critical bulk operations
from core.unified_models import Claim
from data.storage import load_claims_batch

claims = load_claims_batch(claim_ids)  # Bypass API when needed
```

### Interface-Specific Logic:
```python
# Only in interface layer, never in business logic
class CLI:
    def format_output(self, results):
        # CLI-specific formatting
        pass

class GUI:
    def format_output(self, results):
        # GUI-specific formatting
        pass
```

## Testing Strategy

### Unit Tests:
- Test `Conjecture` class independently
- Mock data layer for business logic tests
- Test each interface with mocked `Conjecture`

### Integration Tests:
- Test full flow from interface to data
- Validate API contracts
- Ensure consistent behavior across interfaces

## File Structure

```
src/
├── core/
│   ├── models.py             # Single source for Claim models
│   └── common_results.py     # Unified ProcessingResult
├── config/
│   └── common.py             # Unified ProviderConfig
├── processing/
│   ├── common_context.py     # Simplified context models
│   └── llm/
│       └── common.py         # Unified GenerationConfig
├── conjecture.py             # Single Conjecture class
├── cli/
│   └── modular_cli.py        # CLI implementation
├── agent/
│   └── ...                   # Agent components
├── local/
│   └── ...                   # Local service integrations
├── interfaces/
│   ├── simple_gui.py         # GUI implementation
│   └── simple_tui.py         # TUI implementation
└── utils/
    └── ...                   # Utility functions
```

## Migration Guide

### From Complex Architecture:
1. Identify business logic in service layers
2. Move to `Conjecture` class methods
3. Update interfaces to use `Conjecture` directly
4. Remove unnecessary abstractions
5. Update tests to reflect simple structure

### Adding New Interface:
1. Create interface directory
2. Implement interface using `Conjecture` class
3. Add interface-specific logic only
4. Test with mocked `Conjecture` class

## Conclusion

This simple architecture provides maximum power with minimum complexity. It's easy to understand, maintain, and extend while avoiding the pitfalls of over-engineering.

**Remember**: One `Conjecture` class, unified API, multiple interfaces.