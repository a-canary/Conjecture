# Conjecture Project - QWEN Context File

## Project Overview

Conjecture is an evidence-based AI reasoning system that enables exploration and validation of knowledge claims through a simple, elegant unified API. The project provides maximum power through minimum complexity using a single `Conjecture` class that serves all interfaces (CLI, TUI, GUI).

### Key Features:
- **Unified API**: Single `Conjecture` class provides all functionality across all interfaces
- **Simple Architecture**: Clean separation of data models, processing, and interfaces with no over-engineering
- **Easy Interface Swapping**: CLI, TUI, and GUI all use the same `Conjecture` API
- **Direct Data Access**: When needed, direct access to underlying models and data
- **Claim Management**: Pydantic-based models with confidence scores, types, states, and relationships
- **Vector Database Integration**: Support for ChromaDB and FAISS for semantic similarity search
- **LLM Integration**: Flexible interface for AI processing with multiple provider support
- **Configurable Architecture**: Environment variable-based configuration system

### Core Architecture:
1. **Data Models** (`src/core/unified_models.py`): Clean, unified Pydantic models
2. **Processing** (`src/contextflow.py`): Single `Conjecture` class providing unified API
3. **Interfaces** (`src/cli/`, `src/tui/`, `src/gui/`): CLI, TUI, GUI - all using the same `Conjecture` class
4. **Configuration** (`src/config/simple_config.py`): Smart defaults with minimal setup

### Simple Interface Pattern
All interfaces follow the same simple pattern:

```python
# CLI Interface
from contextflow import Conjecture
cf = Conjecture()
result = cf.explore("machine learning", max_claims=5)
claim = cf.add_claim("content", 0.85, "concept")

# TUI Interface  
from contextflow import Conjecture
cf = Conjecture()
# Same API usage

# GUI Interface
from contextflow import Conjecture
cf = Conjecture()
# Same API usage
```

**No over-engineering**: Direct API usage across all interfaces with clean separation of concerns.

## Project Structure

```
Conjecture/
├── src/
│   ├── config/           # Configuration system
│   ├── core/             # Core models (unified_models.py)
│   ├── contextflow.py    # Main unified API - single Conjecture class
│   ├── ui/               # Interface implementations (CLI, TUI, GUI)
│   └── utils/            # Utility functions
├── tests/                # Test suite
├── demo/                 # Example implementations
├── specs/                # Specification documents
├── requirements.txt      # Dependencies
└── documentation files
```

## Building and Running

### Prerequisites
- Python 3.8+
- Required packages per `requirements.txt`

### Installation
```bash
pip install -r requirements.txt
```

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test files
python tests/test_models.py
python tests/test_data_layer.py
python tests/test_processing_layer.py
python tests/test_data_layer_complete.py
```

### Basic Usage
```python
from src.contextflow import Conjecture

# Initialize with default configuration
cf = Conjecture()

# Explore claims related to a topic
result = cf.explore("machine learning", max_claims=5)
print(result.summary())

# Add a new claim
claim = cf.add_claim(
    content="Machine learning algorithms require substantial training data",
    confidence=0.85,
    claim_type="concept",
    tags=["ml", "data"]
)

# Same API works in CLI, TUI, and GUI applications
# No service layers or complex abstractions needed
```

### Interface Implementation Pattern
All interfaces (CLI, TUI, GUI) use the same simple pattern:

```python
# CLI Interface Example
from contextflow import Conjecture

def main():
    cf = Conjecture()
    result = cf.explore("machine learning", max_claims=5)
    claim = cf.add_claim("content", 0.85, "concept")
    print(result.summary())

# TUI Interface Example  
from contextflow import Conjecture

class TUIApp:
    def __init__(self):
        self.cf = Conjecture()  # Single API instance
    
    def search_screen(self):
        results = self.cf.explore(self.get_query())
        self.display_results(results)

# GUI Interface Example
from contextflow import Conjecture

class GUIApp:
    def __init__(self):
        self.cf = Conjecture()  # Same API
    
    def on_search_button(self):
        results = self.cf.explore(self.search_input.get())
        self.populate_results_list(results)
```

**Key Principle**: One `Conjecture` class, unified API, multiple interfaces.
```

### Environment Variables
The system uses environment variables for configuration:
- `Conjecture_DB_PATH`: Database file path
- `Conjecture_DB_TYPE`: Database type (file, chroma, mock)
- `Conjecture_CONFIDENCE`: Confidence threshold
- `Conjecture_MAX_CONTEXT`: Maximum context size
- `Conjecture_BATCH_SIZE`: Exploration batch size
- `Conjecture_EMBEDDING_MODEL`: Embedding model name
- `Conjecture_LLM_API_KEY`: LLM API key (enables LLM features)
- `Conjecture_LLM_MODEL`: LLM model name
- `Conjecture_DEBUG`: Debug mode

## Development Conventions

### Code Style
- Follow PEP 8 Python style guidelines
- Use Pydantic for data validation and modeling
- Write comprehensive docstrings using Google-style format
- Include type hints for all public functions and methods

### Simple Architecture Principles
- **Single API**: All interfaces use the same `Conjecture` class
- **No Over-Engineering**: Avoid unnecessary abstraction layers
- **Direct Usage**: Interfaces call `Conjecture` methods directly
- **Clean Separation**: Data models, processing, and interfaces are separate layers
- **Easy Testing**: Each layer can be tested independently

### Testing
- All core functionality must have 100% test coverage
- Use pytest for test execution
- Follow the pattern of creating separate test files for different components
- Include both positive and negative test cases
- Validate all model constraints and business logic

### Claim Model Design
- Claims must have an ID, content (min 10 chars), confidence (0.0-1.0), and at least one type
- Supported claim types: concept, reference, thesis, skill, example, goal
- Claim states: Explore, Validated, Orphaned, Queued
- Use validation methods to ensure data integrity
- Implement proper timestamp management (created, updated)

### Architecture Principles
- **Maximum power through minimum complexity**
- **Single unified API** - `Conjecture` class serves all interfaces
- **No over-engineering** - Direct API usage across all interfaces
- **Clean separation** - Data models + Processing + Interfaces
- **Easy interface swapping** - All use same `Conjecture` API
- **Direct data access** - Available when needed
- Smart defaults with minimal configuration
- Comprehensive error handling and validation

### Git Workflow
- Feature branches for new development
- Descriptive commit messages
- Pull requests with comprehensive descriptions
- All tests must pass before merging
- Update documentation when making significant changes

## Key Files and Modules

### Core Models (`src/core/unified_models.py`)
- Implements the `Claim`, `ClaimBatch`, and `ProcessingResult` models
- Defines `ClaimState` and `ClaimType` enumerations
- Includes validation rules and data transformation methods

### Main API (`src/contextflow.py`)
- **Single `Conjecture` class** provides unified API for all interfaces
- Implements `explore()` and `add_claim()` methods
- Handles configuration and backend initialization
- **Same API used by CLI, TUI, and GUI** - no duplication needed
- Direct, simple interface with no service layer complexity

### Interface Implementation
- **CLI**: Uses `Conjecture` class directly in `src/cli/`
- **TUI**: Uses `Conjecture` class directly in `src/tui/` (planned)
- **GUI**: Uses `Conjecture` class directly in `src/gui/` (planned)
- **Pattern**: All interfaces follow `cf = Conjecture(); cf.method()` pattern

### Configuration (`src/config/simple_config.py`)
- Single `Config` class with smart defaults
- Environment variable-based configuration
- Property methods for specific settings (LLM, ChromaDB, etc.)

## Current Status

### ✅ Completed Features
- **Simple Unified Architecture**: Single `Conjecture` class provides unified API for all interfaces
- **Core Models**: Clean Pydantic models with validation (`Claim`, `ClaimBatch`, `ProcessingResult`)
- **Configuration System**: Smart defaults with environment variable support
- **Interface Examples**: Working examples for CLI, TUI, and GUI using unified API
- **Documentation**: Complete architecture specifications and implementation guides

### 🔄 In Progress
- **CLI Refactoring**: Migrating existing CLI to use unified API pattern
- **TUI Implementation**: Basic terminal interface using unified API
- **GUI Implementation**: Simple tkinter interface using unified API

### 📋 Planned Features
- **Vector Database Integration**: ChromaDB and FAISS support for semantic search
- **LLM Integration**: Multiple provider support for claim processing
- **Advanced Relationships**: Claim relationship management
- **Performance Optimization**: Caching and batch operations

### 🎯 Architecture Status
- **Phase 1 (Simple Architecture)**: ✅ COMPLETE - Unified API implemented
- **Phase 2 (Interface Examples)**: ✅ COMPLETE - CLI/TUI/GUI examples created
- **Phase 3 (Documentation)**: ✅ COMPLETE - Architecture specs and guides written
- **Phase 4 (Production Readiness)**: 🔄 IN PROGRESS - Testing and optimization

**Key Achievement**: Successfully implemented simple, elegant architecture with no over-engineering. All interfaces use the same `Conjecture` API directly.

## Simple Unified Architecture

Conjecture follows a simple, elegant architecture based on a unified API design:

### Core Concept: Single `Conjecture` Class
- **One API to rule them all**: The `Conjecture` class in `src/contextflow.py` provides all functionality
- **No service layers**: Direct API usage across all interfaces
- **Easy interface swapping**: CLI, TUI, and GUI all use the same `Conjecture` class
- **Clean separation**: Data models + Processing + Interfaces

### Architecture Overview
The system follows a simple 3-layer architecture:

1. **Interface Layer**: CLI, TUI, GUI applications
   - All interfaces **directly instantiate and use** the `Conjecture` class
   - **No abstractions** or service layers needed
   - **Simple pattern**: `cf = Conjecture()` -> `cf.explore()` / `cf.add_claim()`

2. **Processing Layer**: The `Conjecture` class
   - **Single unified API** for all functionality
   - Handles configuration, backend initialization, and all operations
   - **Direct, simple interface** with no complexity

3. **Data Layer**: Core models and backends
   - **Clean Pydantic models** in `src/core/unified_models.py`
   - Vector databases (ChromaDB, FAISS) for similarity search
   - File-based or mock backends for different use cases

### Key Simplicity Advantages
- **No over-engineering**: Direct API usage maximizes simplicity
- **Fast development**: New features added to one place
- **Easy maintenance**: Single source of truth for all logic
- **Consistent behavior**: All interfaces work identically
- **Flexible interfaces**: Switch between CLI/TUI/GUI effortlessly

### Key Features
- **Unified API**: Single `Conjecture` class serves all interfaces (CLI, TUI, GUI)
- **Simple Integration**: Direct API usage with no service layer complexity
- **Claim Management**: Create, explore, and manage claims with confidence scoring
- **Vector Similarity**: Semantic search capabilities using embedding-based similarity
- **Multiple Backends**: Support for ChromaDB, FAISS, or mock implementations
- **Flexible Interfaces**: Easy switching between CLI, TUI, and GUI applications
- **Smart Configuration**: Environment-based configuration with sensible defaults
- **Extensible Design**: Easy to add new features to the unified API

### Implementation Approach
The project follows a **simple, pragmatic development approach**:

1. **Core Foundation**: Unified models and `Conjecture` API
2. **Interface Development**: CLI, TUI, GUI using the same API
3. **Backend Integration**: Vector databases and LLM providers
4. **Enhancement**: Add features to the unified API as needed

**Key Principle**: Add features to the `Conjecture` class once, and all interfaces automatically benefit.

### Interface Design
All interfaces use the **exact same simple pattern**:

```python
# CLI, TUI, and GUI all follow this pattern
from src.contextflow import Conjecture

cf = Conjecture()
result = cf.explore("query")
claim = cf.add_claim(content, confidence, claim_type)
```

**No shared components needed** - the `Conjecture` class provides everything:
- **Unified API**: `explore()`, `add_claim()`, `get_statistics()`
- **Direct Access**: When needed, direct access to data models
- **Same Behavior**: Identical functionality across all interfaces
- **Simple Integration**: Two lines of code to get full functionality

### Recommended Claim Tag Definitions
The following tag definitions provide structured semantics that guide how the LLM interprets claims for inference:

- **type.concept**: an idea, process or thing explained such that a 4th grader could understand
- **type.thesis**: a transferable expectation of outcome based on conditional observation  
- **type.principle**: foundational belief
- **type.sample**: specific observation, condition and outcome
- **type.goal**: a desired result
- **domain.{name}**: a domain of knowledge, like [SWE, medical, legal]

These tags all use the same Claim structure and process, but convey how the LLM should interpret them for inference. When building LLM prompt context, the system will use the top N from each type and over-weight results with matching domains.

## Dependencies
- `chromadb==0.4.15`: Vector database
- `pydantic==2.5.2`: Data validation
- `faiss-cpu`: Alternative vector search
- `sentence-transformers`: Embedding models
- `numpy`: Mathematical operations