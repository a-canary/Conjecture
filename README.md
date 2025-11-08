# Conjecture - Evidence-Based AI Reasoning System

## Overview

Conjecture is an evidence-based AI reasoning system that enables exploration and validation of knowledge claims through vector similarity search and LLM processing. The project provides a sophisticated yet elegant architecture for managing claims, their relationships, and their validation through semantic search and AI processing.

## Quick Start

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

## Project Structure

```
Conjecture/
├── specs/                 # 📋 Finalized specifications (authoritative)
│   ├── design.md         # Complete system architecture
│   ├── requirements.md   # Comprehensive requirements
│   ├── phases.md         # Development roadmap
│   └── interface_design.md # Interface specifications
├── src/                  # 💻 Source code implementation
│   ├── config/          # Configuration system
│   ├── core/            # Core models and business logic
│   ├── data/            # Data handling and storage
│   ├── processing/      # LLM processing interfaces
│   ├── ui/              # User interface components
│   ├── utils/           # Utility functions
│   └── contextflow.py   # Main API interface
├── tests/               # 🧪 Test suite
├── demo/                # 🎯 Example implementations
├── data/                # 📊 Data storage (git-ignored)
├── archive/             # 📦 Archived documentation
├── requirements.txt     # 📦 Dependencies
└── README.md           # 📖 This file
```

## Key Features

- **Claim Management**: Robust Pydantic-based models for representing knowledge claims
- **Vector Database Integration**: Support for ChromaDB and FAISS for semantic similarity search
- **LLM Integration**: Flexible interface for AI processing with Gemini API support
- **Multi-Modal Interface**: TUI, CLI, MCP, and WebUI support planned
- **Dirty Flag Evaluation**: Automated claim re-evaluation with confidence-based prioritization

## Development Status

**Phase**: Core Foundation (Weeks 1-2)  
**Status**: Ready for implementation  
**Specifications**: Complete and finalized in `specs/` folder

## Environment Variables

Configure the system using environment variables:
- `Conjecture_DB_PATH`: Database file path
- `Conjecture_DB_TYPE`: Database type (file, chroma, mock)
- `Conjecture_CONFIDENCE`: Confidence threshold
- `Conjecture_MAX_CONTEXT`: Maximum context size
- `Conjecture_BATCH_SIZE`: Exploration batch size
- `Conjecture_EMBEDDING_MODEL`: Embedding model name
- `Conjecture_LLM_API_KEY`: LLM API key (enables LLM features)
- `Conjecture_LLM_MODEL`: LLM model name
- `Conjecture_DEBUG`: Debug mode

## Documentation

### Current Specifications (Authoritative)
- **[System Design](specs/design.md)** - Complete architecture and component design
- **[Requirements](specs/requirements.md)** - Comprehensive requirements and use cases
- **[Development Phases](specs/phases.md)** - Implementation roadmap and milestones
- **[Interface Design](specs/interface_design.md)** - Multi-modal interface specifications

### Archived Documentation
Historical documentation and research materials are available in the `archive/` folder. See `archive/stored-2025-01-07.md` for a detailed inventory of archived content and its potential value.

## Contributing

1. Review the specifications in `specs/` folder
2. Follow the development phases outlined in `specs/phases.md`
3. Ensure all tests pass before submitting changes
4. Update documentation as needed

## License

See [LICENSE](LICENSE) file for details.