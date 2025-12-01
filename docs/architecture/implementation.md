# Interface Implementation Guide

## Overview

All Conjecture interfaces (CLI, TUI, GUI) follow the same simple pattern using the unified `Conjecture` API. After the major refactoring in November 2025, the system now uses unified data models that reduce complexity by 87% while ensuring consistency, maintainability, and easy development.

## The Simple Pattern

### Core Principle
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

## Implementation Examples

### 1. CLI Implementation
```python
# src/cli/modular_cli.py
from contextflow import Conjecture
from rich.console import Console

class YourCLI:
    def __init__(self):
        self.cf = Conjecture()
        self.console = Console()
    
    def search_command(self, query: str, limit: int = 10):
        """Handle search command."""
        try:
            result = self.cf.explore(query, max_claims=limit)
            self.console.print(f"Found {len(result.claims)} claims")
            for claim in result.claims:
                self.console.print(f"- {claim.id}: {claim.content[:50]}...")
        except Exception as e:
            self.console.print(f"Error: {e}")
    
    def add_command(self, content: str, confidence: float, claim_type: str):
        """Handle add claim command."""
        try:
            claim = self.cf.add_claim(content, confidence, claim_type)
            self.console.print(f"✅ Created claim {claim.id}")
        except Exception as e:
            self.console.print(f"Error: {e}")
```

Note: The Conjecture class integrates with agent/, processing/, llm/, and local/ components internally using unified data models (GenerationConfig, ProviderConfig, ProcessingResult), but interfaces only interact with the unified API.

### 2. TUI Implementation
```python
# src/interfaces/simple_tui.py
import curses
from contextflow import Conjecture

class YourTUI:
    def __init__(self):
        self.cf = Conjecture()
    
    def search_screen(self, stdscr):
        """Handle search screen."""
        stdscr.addstr(0, 0, "Enter search query: ")
        curses.echo()
        query = stdscr.getstr(1, 0).decode('utf-8')
        curses.noecho()
        
        try:
            result = self.cf.explore(query, max_claims=10)
            self.display_results(stdscr, result)
        except Exception as e:
            stdscr.addstr(3, 0, f"Error: {e}")
    
    def display_results(self, stdscr, result):
        """Display search results."""
        y = 3
        for claim in result.claims:
            stdscr.addstr(y, 0, f"{claim.id}: {claim.content[:40]}...")
            y += 1
```

### 3. GUI Implementation
```python
# src/interfaces/simple_gui.py
import tkinter as tk
from tkinter import messagebox
from contextflow import Conjecture

class YourGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.cf = Conjecture()
        self.setup_ui()
    
    def setup_ui(self):
        """Setup GUI components."""
        tk.Label(self.root, text="Search:").pack()
        self.search_entry = tk.Entry(self.root)
        self.search_entry.pack()
        tk.Button(self.root, text="Search", command=self.on_search).pack()
    
    def on_search(self):
        """Handle search button."""
        query = self.search_entry.get()
        try:
            result = self.cf.explore(query, max_claims=10)
            messagebox.showinfo("Results", f"Found {len(result.claims)} claims")
        except Exception as e:
            messagebox.showerror("Error", str(e))
```

## Available API Methods

### Core Methods
```python
# Exploration
result = cf.explore(query, max_claims=10, claim_types=None, confidence_threshold=None)

# Claim Management
claim = cf.add_claim(content, confidence, claim_type, tags=None)

# System Information
stats = cf.get_statistics()
```

### Result Objects
```python
# Exploration Result
result.query          # Original query
result.claims         # List of Claim objects
result.total_found    # Total claims found
result.search_time    # Search duration
result.summary()      # Human-readable summary

# Claim Object
claim.id              # Unique identifier
claim.content         # Claim content
claim.confidence      # Confidence score (0.0-1.0)
claim.type            # List of claim types
claim.tags            # List of tags
claim.state           # Claim state
claim.created         # Creation timestamp
```

## Error Handling Pattern

### Consistent Error Handling
```python
def your_method(self):
    try:
        result = self.cf.explore("query")
        # Process successful result
        return {"success": True, "data": result}
    except ValueError as e:
        # Handle validation errors
        return {"success": False, "error": f"Validation error: {e}"}
    except Exception as e:
        # Handle other errors
        return {"success": False, "error": f"Error: {e}"}
```

### Common Validation Errors
- Query too short (< 5 characters)
- Invalid confidence range (must be 0.0-1.0)
- Invalid claim type
- Content too short (< 10 characters)

## Interface-Specific Considerations

### CLI Considerations
- Use rich formatting for output
- Handle command-line arguments
- Provide clear error messages
- Support batch operations

### TUI Considerations
- Handle keyboard input properly
- Manage screen layout
- Provide navigation
- Handle screen resizing

### GUI Considerations
- Handle user interactions
- Update UI asynchronously
- Provide progress indicators
- Handle window events

## Testing Your Interface

### Unit Testing Pattern
```python
import unittest
from unittest.mock import Mock, patch
from your_interface import YourInterface

class TestYourInterface(unittest.TestCase):
    def setUp(self):
        # Mock the Conjecture API
        with patch('your_interface.Conjecture') as mock_cf:
            self.mock_cf = Mock()
            mock_cf.return_value = self.mock_cf
            self.interface = YourInterface()
    
    def test_search_success(self):
        # Mock successful search
        mock_result = Mock()
        mock_result.claims = [Mock(id="test", content="test")]
        self.mock_cf.explore.return_value = mock_result
        
        result = self.interface.search_command("test")
        self.assertTrue(result["success"])
    
    def test_search_error(self):
        # Mock search error
        self.mock_cf.explore.side_effect = Exception("Search failed")
        
        result = self.interface.search_command("test")
        self.assertFalse(result["success"])
```

## Best Practices

### 1. Keep It Simple
- Use the `Conjecture` API directly
- Don't add unnecessary abstraction layers
- Focus on interface-specific logic

### 2. Handle Errors Gracefully
- Validate inputs before calling API
- Provide clear error messages
- Handle API exceptions properly

### 3. Follow Consistent Patterns
- Use the same initialization pattern
- Handle results consistently
- Follow naming conventions

### 4. Optimize for User Experience
- Provide progress feedback
- Handle long operations asynchronously
- Design intuitive interfaces

## Migration from Complex Architecture

### If You Have Service Layers
```python
# Before (complex)
class YourInterface:
    def __init__(self):
        self.service = ClaimService()
        self.repository = ClaimRepository()
    
    def search(self, query):
        results = self.service.search_claims(query)
        return self.format_results(results)

# After (simple)
class YourInterface:
    def __init__(self):
        self.cf = Conjecture()
    
    def search(self, query):
        result = self.cf.explore(query)
        return self.format_results(result)
```

### Benefits of Migration
- Less code to maintain
- Fewer bugs
- Easier testing
- Better performance

## Conclusion

The simple unified API pattern provides maximum power with minimum complexity. By using the `Conjecture` class directly across all interfaces, you ensure consistency, maintainability, and rapid development.

**Remember**: One `Conjecture` class, unified API, multiple interfaces.