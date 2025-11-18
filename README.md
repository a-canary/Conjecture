# Simple Architecture Summary

## 🎯 The Core Idea

Conjecture uses a **simple, elegant architecture** based on a **single unified API**. No over-engineering, no complex service layers - just clean, direct functionality.

## 📐 Architecture Diagram

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

## 🔑 Key Principles

### 1. Single Unified API
```python
from contextflow import Conjecture

# One class for all functionality
cf = Conjecture()
result = cf.explore("machine learning")
claim = cf.add_claim("content", 0.85, "concept")
stats = cf.get_statistics()
```

### 2. No Over-Engineering
- ❌ No service layers
- ❌ No dependency injection frameworks
- ❌ No complex abstractions
- ✅ Direct API usage
- ✅ Simple, maintainable code

### 3. All Interfaces Follow Same Pattern
```python
# CLI, TUI, GUI - all the same pattern
class YourInterface:
    def __init__(self):
        self.cf = Conjecture()  # Single API instance
    
    def your_method(self):
        results = self.cf.explore("query")
        claim = self.cf.add_claim("content", 0.8, "concept")
        return results, claim
```

## 📁 File Structure

```
src/
├── core/
│   └── unified_models.py     # Data models only
├── contextflow.py            # Single Conjecture class
├── cli/
│   ├── simple_cli.py         # CLI example
│   └── base_cli.py           # Base CLI functionality
├── tui/
│   └── simple_tui.py         # TUI example
├── gui/
│   └── simple_gui.py         # GUI example
└── config/
    └── simple_config.py      # Configuration only

specs/
├── simple_architecture.md           # Architecture specification
└── interface_implementation_guide.md # Implementation guide

demo/
└── unified_api_demo.py              # Live demonstration
```

## 🚀 Benefits

### Simplicity
- Easy to understand and maintain
- No complex abstractions to learn
- Clear responsibility boundaries

### Consistency
- All interfaces work the same way
- Single source of truth for functionality
- No duplication of business logic

### Flexibility
- Easy to add new interfaces
- Direct data access when needed
- Minimal coupling between layers

### LLM Provider Support
- Multiple LLM providers supported: Chutes.ai, LM Studio, OpenAI, Anthropic, and more
- Local model support through LM Studio with models like ibm/granite-4-h-tiny
- Easy configuration switching between providers
- Robust fallback mechanisms

### Performance
- No unnecessary abstraction overhead
- Direct API calls
- Efficient resource usage

## 🎭 Interface Examples

### CLI Example
```python
from contextflow import Conjecture
from rich.console import Console

class CLI:
    def __init__(self):
        self.cf = Conjecture()
        self.console = Console()
    
    def search(self, query):
        result = self.cf.explore(query)
        self.console.print(f"Found {len(result.claims)} claims")
```

### TUI Example
```python
import curses
from contextflow import Conjecture

class TUI:
    def __init__(self):
        self.cf = Conjecture()
    
    def search_screen(self, stdscr):
        query = self.get_input(stdscr, "Search: ")
        result = self.cf.explore(query)
        self.display_results(stdscr, result)
```

### GUI Example
```python
import tkinter as tk
from contextflow import Conjecture

class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.cf = Conjecture()
    
    def on_search(self):
        results = self.cf.explore(self.search_entry.get())
        self.populate_results(results)
```

## 🧪 Testing Strategy

### Unit Tests
- Test `Conjecture` class independently
- Mock data layer for business logic tests
- Test each interface with mocked `Conjecture`

### Integration Tests
- Test full flow from interface to data
- Validate API contracts
- Ensure consistent behavior across interfaces

## 📖 Available Documentation

1. **[Simple Architecture Specification](specs/simple_architecture.md)** - Complete architecture details
2. **[Interface Implementation Guide](specs/interface_implementation_guide.md)** - How to implement interfaces
3. **[Live Demo](demo/unified_api_demo.py)** - Working demonstration
4. **[QWEN Context](QWEN.md)** - Project overview and status

## 🎯 Key Takeaway

**One `Conjecture` class, unified API, multiple interfaces.**

This simple architecture provides maximum power with minimum complexity. It's easy to understand, maintain, and extend while avoiding the pitfalls of over-engineering.

---

*Last updated: November 12, 2025*