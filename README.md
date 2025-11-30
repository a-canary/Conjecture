# Conjecture: AI-Powered Evidence-Based Reasoning System

Conjecture is a lightweight, modular AI reasoning system that helps you create, search, and analyze evidence-based claims. Built with simplicity in mind, it provides powerful functionality with minimal complexity.

## 🎯 What It Does

Conjecture enables you to:
- **Create Claims**: Store knowledge claims with confidence scores and evidence
- **Search Claims**: Find relevant information using semantic search
- **Analyze Evidence**: Evaluate and validate claims using AI-powered tools
- **Web Research**: Automatically gather evidence from the web
- **Multiple Backends**: Use local models (Ollama, LM Studio) or cloud providers (OpenAI, Anthropic, etc.)

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone <repository-url>
cd Conjecture

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
```bash
# Copy the configuration template
cp .env.example .env

# Edit .env with your preferred provider
# For local use (recommended):
#   - Install Ollama from https://ollama.ai/
#   - Set PROVIDER_API_URL=http://localhost:11434
#   - Set PROVIDER_MODEL=llama2

# For cloud use:
#   - Get API key from your provider
#   - Set PROVIDER_API_URL, PROVIDER_API_KEY, and PROVIDER_MODEL
```

### 3. Run Conjecture
```bash
# Make the main script executable (Unix/Linux/macOS)
chmod +x conjecture

# Or run directly with Python
python conjecture

# Test your setup
python conjecture validate
```

## 📋 Usage Examples

### Basic Commands
```bash
# Create a claim
python conjecture create "The sky is blue due to Rayleigh scattering" --confidence 0.95

# Search for claims
python conjecture search "sky color"

# View statistics
python conjecture stats

# Analyze a specific claim
python conjecture analyze c0000001
```

### Backend Selection
```bash
# Use local models (offline, private)
python conjecture --backend local create "Local claim"

# Use cloud models (more powerful)
python conjecture --backend cloud search "AI research"

# Auto-detect best backend
python conjecture --backend auto analyze c0000001
```

### Advanced Features
```bash
# Web search integration
python conjecture research "quantum computing applications"

# Batch operations
python conjecture create --file claims.json

# Export results
python conjecture export --format json --output results.json
```

## 🛠️ Available Tools

| Tool | Description | Example |
|------|-------------|---------|
| **WebSearch** | Search the web for current information | `research "AI trends 2024"` |
| **CreateClaim** | Store knowledge claims with evidence | `create "Python is popular" --confidence 0.9` |
| **ReadFiles** | Analyze content from local files | `analyze --file document.pdf` |
| **WriteCodeFile** | Generate and save code | `generate --language python "sorting algorithm"` |

## 🏗️ Architecture

Conjecture uses a clean, modular architecture:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLI Interface │    │  Core Engine    │    │   Data Layer    │
│                 │    │                 │    │                 │
│ • Commands      │───▶│ • Processing    │───▶│ • SQLite DB     │
│ • Backends      │    │ • Tools         │    │ • Embeddings    │
│ • Validation    │    │ • Analysis      │    │ • Search        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Components
- **`conjecture`**: Main entry point script
- **`src/cli/modular_cli.py`**: Unified CLI with backend auto-detection
- **`src/conjecture.py`**: Core Conjecture class with async evaluation
- **`src/core.py`**: Core models and utilities
- **`data/conjecture.db`**: SQLite database for claim storage

## 🔧 Configuration

### Environment Variables (.env)
```bash
# Provider Configuration
PROVIDER_API_URL=https://llm.chutes.ai/v1  # Your chosen provider
PROVIDER_API_KEY=your_api_key_here
PROVIDER_MODEL=zai-org/GLM-4.6-FP8

# Local Provider Example
# PROVIDER_API_URL=http://localhost:11434  # Ollama
# PROVIDER_API_KEY=
# PROVIDER_MODEL=llama2

# Application Settings
DB_PATH=data/conjecture.db
CONFIDENCE_THRESHOLD=0.95
MAX_CONTEXT_SIZE=10
DEBUG=false
```

### Supported Providers

#### Local Providers (Privacy-focused)
- **Ollama**: `http://localhost:11434` - Install from https://ollama.ai/
- **LM Studio**: `http://localhost:1234` - Install from https://lmstudio.ai/

#### Cloud Providers (High performance)
- **Chutes.ai**: Fast and cost-effective
- **OpenRouter**: Access to 100+ models
- **OpenAI**: GPT models
- **Anthropic**: Claude models
- **Google**: Gemini models

## 🧪 Testing

Run the test suite to verify your installation:

```bash
# Run all tests
python -m pytest tests/

# Test specific components
python -m pytest tests/test_core_tools.py
python -m pytest tests/test_data_layer.py
python -m pytest tests/test_dirty_flag.py

# Test emoji support
python -m pytest tests/test_emoji.py
```

## 📊 Features

### Core Functionality
- ✅ **Claim Management**: Create, search, and analyze knowledge claims
- ✅ **Evidence-Based**: Attach evidence and confidence scores to claims
- ✅ **Web Integration**: Automatic web search for claim validation
- ✅ **Multiple Backends**: Support for local and cloud AI providers
- ✅ **Semantic Search**: Find relevant claims using natural language

### User Experience
- ✅ **Rich CLI**: Beautiful terminal output with progress indicators
- ✅ **Emoji Support**: Enhanced visual feedback (see [EMOJI_USAGE.md](./EMOJI_USAGE.md))
- ✅ **Cross-Platform**: Works on Windows, macOS, and Linux
- ✅ **Auto-Detection**: Intelligent backend selection

### Developer Features
- ✅ **Modular Design**: Easy to extend and customize
- ✅ **Clean Architecture**: Well-organized codebase
- ✅ **Comprehensive Tests**: High test coverage
- ✅ **Type Safety**: Full type annotations

## 📁 Project Structure

```
Conjecture/
├── conjecture                    # Main entry point
├── requirements.txt              # Python dependencies
├── .env.example                  # Configuration template
├── README.md                     # This file
├── CLAUDES_TODOLIST.md          # Development review and cleanup tasks
├── src/
│   ├── cli/
│   │   └── modular_cli.py       # Unified CLI interface
│   ├── conjecture.py            # Core Conjecture class
│   ├── core.py                  # Core models and utilities
│   ├── config/                  # Configuration management
│   ├── processing/              # LLM integration and evaluation
│   ├── core/                    # Data models and operations
│   ├── tools/                   # Tool registry and management
│   └── utils/                   # Utility functions
├── data/
│   └── conjecture.db            # SQLite database (auto-created)
├── tests/                       # Test files (consolidated)
├── docs/                        # Documentation
├── archive/                     # Archived files and documentation
└── EMOJI_USAGE.md               # Emoji feature documentation
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📚 Documentation

- [EMOJI_USAGE.md](./EMOJI_USAGE.md) - Complete emoji feature guide
- [CLAUDES_TODOLIST.md](./CLAUDES_TODOLIST.md) - Development review and cleanup tasks
- [docs/](./docs/) - Additional documentation and specifications
- [archive/](./archive/) - Archived documentation and historical files

## 🛡️ Security

- No API keys are stored in the repository
- All sensitive data is managed through environment variables
- Local providers keep your data completely private
- Regular security updates and dependency management

## 🐛 Troubleshooting

### Common Issues

**"Provider not found" error**
- Check your `.env` file configuration
- Verify the provider URL is accessible
- Ensure API key is valid (for cloud providers)

**"Database locked" error**
- Ensure only one instance of Conjecture is running
- Check file permissions on the `data/` directory

**"Module not found" error**
- Run `pip install -r requirements.txt` again
- Check your Python version (3.8+ recommended)

### Getting Help

```bash
# Check configuration
python conjecture config

# Validate setup
python conjecture validate

# See available commands
python conjecture --help

# Check backend status
python conjecture backends
```

## 📄 License

[Add your license information here]

---

**Conjecture** - Making evidence-based reasoning accessible and powerful.