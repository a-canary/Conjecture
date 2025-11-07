# Conjecture

**Evidence-Based AI Reasoning System**

> *"Maximum power through minimum complexity"* - Richard Feynman

Conjecture is a sophisticated evidence-based AI reasoning system that validates knowledge claims through iterative research and confidence scoring. Unlike traditional task-completion agents, Conjecture focuses on epistemic rigor and evidence validation.

## 🎯 Core Philosophy

Conjecture embodies a fundamentally different paradigm for knowledge-based AI systems:

- **Validation-Centric**: Focus on evidence validation rather than task completion
- **Evidence-External**: Confidence from external evidence, not internal model certainty
- **State-Driven**: Processing behavior determined by claim states and confidence thresholds
- **Relationship-First**: Knowledge graph structure is primary, not secondary
- **Anti-Complexity**: Actively resists unnecessary complexity
- **Unified Model**: Everything is a claim - no separate data structures

## 🏗️ Architecture Overview

### Core Components

1. **Claim Management**: Unified claim model with confidence scoring
2. **Skill-Based Agency**: LLM instruction through `type.skill` and `type.example` claims
3. **Multi-Session Architecture**: Isolated sessions with shared persistent storage
4. **Trustworthiness Validation**: Web content author trust assessment system
5. **Contradiction Detection**: Automated conflict detection and intelligent merging

### Multi-Modal Interfaces

- **TUI**: Rich terminal interface for keyboard-driven interaction
- **CLI**: Command-line interface for automation and scripting
- **MCP**: Model Context Protocol for AI assistant integration
- **WebUI**: Browser-based interface for collaborative features

## 🚀 Key Features

### Evidence-Based Reasoning
- All claims start "dirty" and must earn confidence through evidence
- Multi-cycle processing until confidence threshold (≥95%) achieved
- Structured evidence hierarchies with bidirectional support relationships

### Skill-Based Agency
- `type.skill` claims instruct LLM how to perform specific actions
- `type.example` claims show proper tool response formatting
- Tool call reflection and example claim creation
- Custom-tailored agent prompts for each evaluation

### Advanced Knowledge Management
- Confidence propagation through support relationships
- Dirty flag cascading to related claims
- Intelligent claim merging and contradiction resolution
- Persistent trustworthiness validation

## 📁 Project Structure

```
Conjecture/
├── specs/                    # Complete specification documents
│   ├── design.md            # Core architecture and system design
│   ├── interface_design.md  # Multi-modal interface patterns
│   ├── phases.md            # Development roadmap
│   └── requirements.md      # Functional and non-functional requirements
├── src/                     # Source code implementation
│   ├── core/                # Core models and embedding methods
│   ├── config/              # Configuration system
│   ├── processing/          # LLM processing interfaces
│   └── ui/                  # Terminal user interface
├── tests/                   # Comprehensive test suite
├── docs/                    # Documentation and guides
└── archive/                 # Archived documentation
```

## 🛠️ Development Status

### Current Phase: Specification Complete ✅
- ✅ Complete specification documents with skill-based agency architecture
- ✅ Multi-modal interface design (TUI, CLI, MCP, WebUI)
- ✅ Core claim models and processing frameworks
- ✅ Comprehensive evaluation and testing frameworks

### Next Steps
1. **Phase 1**: Core Foundation - Unified claim model and database layer
2. **Phase 2**: Skill-Based Agency - LLM instruction and tool execution
3. **Phase 3**: Interface Implementation - Multi-modal interface development
4. **Phase 4**: Trustworthiness System - Source validation and confidence scoring

## 📋 Requirements

### System Requirements
- Python 3.8+
- Git for version control

### Dependencies
See `Conjecture/requirements.txt` for complete dependency list:
- Pydantic v2.5.2+ for data validation
- ChromaDB v0.4.15+ for vector database
- Sentence Transformers for embeddings
- Google Generative AI for LLM integration

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd Conjecture

# Install dependencies
pip install -r Conjecture/requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and configuration
```

### Basic Usage
```python
from src.contextflow import Conjecture

# Initialize with default configuration
cf = Conjecture()

# Start a new session with root claim
session = cf.start_session("US policies based on the Books of Acts from the NSRV Bible")

# Let the system evaluate and build evidence
result = session.run_until_confident(threshold=0.95)
print(result.summary())
```

## 📚 Documentation

- **[Specifications](Conjecture/specs/)**: Complete technical specifications
- **[Architecture](docs/SystemArchitecture.md)**: System architecture overview
- **[Claim Processing](docs/ClaimProcessing.md)**: Claim evaluation workflow
- **[Research Reports](ContextFlow_Research_Recommendations_2025.md)**: Latest AI research integration

## 🧪 Testing

Run the complete test suite:
```bash
# Run all tests
python -m pytest tests/

# Run specific test files
python tests/test_models.py
python tests/test_data_layer.py
python tests/test_processing_layer.py
```

## 🤝 Contributing

Conjecture follows evidence-based development practices:

1. **All changes start as claims** with confidence scores
2. **Changes must earn confidence** through testing and validation
3. **Documentation is evidence** for architectural decisions
4. **Simplicity is valued** over complexity (Feynman principle)

### Development Workflow
1. Create feature branch from main
2. Implement changes with comprehensive tests
3. Update documentation and specifications
4. Submit pull request with evidence-based justification

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Richard Feynman**: For the principle of "maximum power through minimum complexity"
- **Evidence-Based AI Community**: For research and insights into validation systems
- **Open Source Contributors**: For tools and frameworks that make this project possible

## 📞 Contact

- **Project Issues**: [GitHub Issues](https://github.com/your-org/conjecture/issues)
- **Documentation**: [Project Wiki](https://github.com/your-org/conjecture/wiki)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/conjecture/discussions)

---

**Conjecture**: Where evidence meets intelligence, and simplicity enables sophistication.