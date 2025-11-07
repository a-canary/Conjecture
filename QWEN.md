# Conjecture Project - QWEN Context File

## Project Overview

Conjecture is an evidence-based AI reasoning system that enables exploration and validation of knowledge claims through vector similarity search and LLM processing. The project provides a sophisticated yet elegant architecture for managing claims, their relationships, and their validation through semantic search and AI processing.

### Key Features:
- **Claim Management**: Robust Pydantic-based models for representing knowledge claims with confidence scores, types, states, and relationships
- **Vector Database Integration**: Support for ChromaDB and FAISS for semantic similarity search
- **LLM Integration**: Flexible interface for AI processing with Gemini API support
- **Terminal User Interface**: Rich TUI for interactive claim exploration (in development)
- **Configurable Architecture**: Environment variable-based configuration system

### Core Components:
1. **Claim Model**: Central data structure with validation, state management, and relationship tracking
2. **Vector Search**: Semantic similarity matching using embeddings
3. **Processing Layer**: LLM integration for claim validation and generation
4. **Configuration System**: Simplified, smart-default configuration with essential settings

## Project Structure

```
Conjecture/
├── src/
│   ├── config/           # Configuration system
│   ├── core/             # Core models and embedding methods
│   ├── data/             # Data handling
│   ├── processing/       # LLM processing interfaces
│   ├── ui/               # Terminal user interface
│   ├── utils/            # Utility functions
│   └── contextflow.py    # Main API interface
├── tests/                # Test suite
├── demo/                 # Example implementations
├── specs/                # Specification documents
├── EmbeddingMethods.py   # Standalone embedding implementation
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
- Maximum power through minimum complexity
- Clean interfaces between components
- Mock implementations for parallel development
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
- Provides the `Conjecture` class as the main interface
- Implements `explore()` and `add_claim()` methods
- Handles configuration and backend initialization

### Configuration (`src/config/simple_config.py`)
- Single `Config` class with smart defaults
- Environment variable-based configuration
- Property methods for specific settings (LLM, ChromaDB, etc.)

## Current Status
- Phase 1 (Core Foundation): In progress - Basic claim models and configuration system implemented
- Phase 2 (Claim Relationships): Planning stage - Relationship system design complete, implementation pending
- Phase 3 (Vector Similarity Integration): Planning stage - Vector database integration planned
- Phase 4 (Enhanced TUI Experience): Planning stage - TUI design specifications complete
- Phase 5 (CLI Interface Development): Planning stage - CLI command structure designed
- Phase 6 (Model Context Protocol): Planning stage - MCP protocol specifications defined
- Phase 7 (Web Interface Development): Planning stage - WebUI architecture planned
- Phase 8 (Claim-Based Goal Management): Planning stage - Goal management system designed
- Phase 9 (Dirty Flag Optimization): Planning stage - Dirty flag evaluation system designed
- Phase 10 (Production Deployment and Evaluation): Planning stage - Deployment strategy outlined
- Project is in early development/planning phase, with core models and basic infrastructure implemented

## Complete Project Vision

Based on the specification documents, Conjecture is a comprehensive evidence-based AI reasoning system with the following key components:

### Core Concepts
- **Knowledge Claims**: Assertions about the world with unique IDs in format "c#######" that can be validated, connected, and reasoned about
- **Claim Relationships**: Parent-child connections between claims stored in a junction table representing support relationships
- **Vector-Based Similarity**: Numerical representations enabling semantic searching and clustering
- **Multi-Modal Interaction**: TUI, CLI, MCP, and WebUI interfaces providing different interaction methods
- **Dirty Flag Evaluation**: Claims marked for re-evaluation when new information becomes available, with prioritized processing based on confidence and relevance

### Architecture Overview
The system follows a multi-layer architecture:
1. **Interface Layer**: TUI, CLI, MCP, and WebUI with shared business logic
2. **Application Layer**: Claim Processor, Evidence Manager, Goal Tracker
3. **Domain Layer**: Claim Management, Evidence Management, Vector Similarity, Query Engine
4. **Infrastructure Layer**: Database Manager, Storage Manager, Embedding Service, Auth Manager
5. **External Services**: Vector DB, LLM Service, External Sources, File System

### Key Features
- **Claim Management**: Create, edit, delete claims with confidence scoring and metadata
- **Relationship Management**: Support relationships between claims with junction table implementation
- **Knowledge Exploration**: Navigate knowledge graph through parent-child relationships
- **Goal Management**: Goals implemented as specialized claims with "goal" tags and evidence-based confidence tracking
- **Dirty Flag System**: Automated evaluation of claims needing re-assessment with confidence-based prioritization
- **Multi-Modal Interface**: Consistent experience across terminal, command-line, AI assistant protocols, and web interfaces
- **Vector Similarity**: Semantic search capabilities using embedding-based similarity matching
- **Structured Tagging System**: Standardized tags that convey how the LLM should interpret claims for inference

### Implementation Approach
The project follows a 10-phase development approach:
1. Core Foundation (Weeks 1-2)
2. Claim Relationships (Weeks 3-4)
3. Vector Similarity Integration (Weeks 5-6)
4. Enhanced TUI Experience (Weeks 7-8)
5. CLI Interface Development (Weeks 9-10)
6. Model Context Protocol (Weeks 11-12)
7. Web Interface Development (Weeks 13-14)
8. Claim-Based Goal Management (Weeks 15-16)
9. Dirty Flag Optimization (Weeks 17-18)
10. Production Deployment and Evaluation (Weeks 19-20)

### Dirty Flag Evaluation System
A sophisticated system that:
- Marks claims as dirty when new information becomes available
- Selects dirty claims for evaluation based on relevance and confidence
- Prioritizes claims with confidence < 0.90 for evaluation with confidence boost
- Processes dirty claims in parallel batches
- Cascades dirty status to supported claims
- Processes LLM responses with two-pass system (claims first, then relationships)

### Goal Management
Goals are implemented as specialized claims with the "goal" tag:
- Progress tracked through evidence-based confidence (not completion percentage)
- Status maintained through additional tags (active, paused, completed)
- No special handling - just regular claims evaluated by LLM based on evidence
- Child tasks connected through supports relationships

### Interface Design
All interfaces share common components:
- **Claim Manager**: Centralized claim operations and state management
- **Confidence Manager**: Confidence evaluation and tracking
- **Relationship Manager**: Relationship management across claims
- **Real-Time Updates**: Event-driven architecture for immediate response to changes

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