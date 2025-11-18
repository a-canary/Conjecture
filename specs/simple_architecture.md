# Simple Architecture Specification

## Overview

Conjecture uses a simple, elegant architecture based on a single unified API. No over-engineering, no complex service layers - just clean, direct functionality.

## Core Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Interfaces Layer                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │
│  │   CLI   │  │   TUI   │  │   GUI   │  │  Future │      │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘      │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 Processing Layer                            │
│              Single Conjecture Class                        │
│         ┌─────────────────────────────────────┐           │
│         │  explore() │ add_claim() │ stats()   │           │
│         └─────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Data Layer                                │
│        ┌─────────────────────────────────────┐           │
│        │    Claim Model │ Validation │ Storage │        │
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
- Data models: `src/core/unified_models.py`
- Processing: `src/contextflow.py` (single class)
- Interfaces: `src/cli/`, `src/tui/`, `src/gui/`

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
# src/cli/main.py
from contextflow import Conjecture

def main():
    cf = Conjecture()
    result = cf.explore(args.query, max_claims=args.limit)
    print(result.summary())
```

### TUI Example:
```python
# src/tui/app.py
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
# src/gui/app.py
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
│   └── unified_models.py     # Data models only
├── contextflow.py            # Single Conjecture class
├── cli/
│   └── main.py               # CLI implementation
├── tui/
│   └── app.py                # TUI implementation
├── gui/
│   └── app.py                # GUI implementation
└── config/
    └── simple_config.py      # Configuration only
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