# Conjecture Configuration Wizard

A streamlined, user-friendly setup wizard for Conjecture with comprehensive diagnostics and modern dependency management.

## Features

### 🔍 System Diagnostics
- **Health Checks**: Python version, dependencies, disk space, memory
- **Provider Detection**: Automatically finds local LLM providers (Ollama, LM Studio)
- **Network Validation**: Tests connectivity for cloud providers
- **Project Structure**: Verifies all required files and directories

### ⚙️ Step-by-Step Setup
- **Dependency Installation**: Uses `uv` for fast package management (falls back to pip)
- **Provider Configuration**: Supports local and cloud LLM providers
- **User Settings**: Personalized username, workspace, and team configuration
- **Validation**: Tests configuration before saving

### 🚀 Modern Experience
- **Rich CLI Output**: Clear status indicators and progress feedback
- **Smart Defaults**: Intelligent provider detection and configuration
- **Backup Protection**: Automatically backs up existing configurations
- **Error Recovery**: Graceful handling of setup failures

## Quick Start

### Run the Wizard
```bash
python setup_wizard.py
```

### Test the System
```bash
python test_wizard.py
```

## Supported Providers

### Local Providers (Privacy-focused)
- **Ollama**: `http://localhost:11434` - Install from https://ollama.ai/
- **LM Studio**: `http://localhost:1234/v1` - Install from https://lmstudio.ai/

### Cloud Providers (High performance)
- **OpenAI**: GPT models - Get keys from https://platform.openai.com/api-keys
- **Anthropic**: Claude models - Get keys from https://console.anthropic.com/
- **Chutes.ai**: Affordable models - Get keys from https://chutes.ai/
- **OpenRouter**: 100+ models - Get keys from https://openrouter.ai/keys

## Wizard Flow

### Phase 1: Diagnostics
1. System information collection
2. Dependency verification
3. Provider detection
4. Network connectivity testing
5. Resource availability checks

### Phase 2: Configuration
1. **Dependencies**: Install with `uv` (preferred) or `pip`
2. **Provider Selection**: Choose from detected or available providers
3. **API Configuration**: Enter keys for cloud providers
4. **User Settings**: Configure username, workspace, team

### Phase 3: Validation
1. Configuration summary display
2. Connection testing (where possible)
3. Backup creation
4. Secure file saving

## File Structure

```
Conjecture/
├── setup_wizard.py              # Main wizard entry point
├── test_wizard.py               # Test suite
├── src/
│   └── config/
│       ├── diagnostics.py       # System diagnostics module
│       ├── streamlined_wizard.py # Main wizard implementation
│       └── setup_wizard.py      # Original wizard (legacy)
└── .env                         # Generated configuration
```

## Configuration Output

The wizard generates a `.env` file with:

```bash
# Provider Configuration
Conjecture_LLM_PROVIDER=ollama
Conjecture_LLM_API_URL=http://localhost:11434
Conjecture_LLM_MODEL=llama2

# User Settings
CONJECTURE_USER=alice
CONJECTURE_WORKSPACE=my-project
CONJECTURE_TEAM=engineering

# Application Settings
DB_PATH=data/conjecture.db
Conjecture_EMBEDDING_MODEL=all-MiniLM-L6-v2
CONFIDENCE_THRESHOLD=0.95
DEBUG=false
```

## Error Handling

The wizard includes comprehensive error handling:

- **Import Errors**: Clear messages for missing dependencies
- **Network Issues**: Graceful fallbacks for connectivity problems
- **Permission Errors**: Helpful guidance for file access issues
- **Validation Failures**: Specific suggestions for configuration problems

## Testing

Run the test suite to verify functionality:

```bash
python test_wizard.py
```

Tests cover:
- ✅ Module imports
- ✅ Wizard initialization
- ✅ Provider configurations
- ✅ Environment file handling
- ✅ System diagnostics

## Next Steps After Setup

Once the wizard completes:

1. **Validate Configuration**:
   ```bash
   python conjecture validate
   ```

2. **Create First Claim**:
   ```bash
   python conjecture create "The sky is blue" --confidence 0.95
   ```

3. **Search Claims**:
   ```bash
   python conjecture search "sky"
   ```

## Troubleshooting

### Common Issues

**"uv not found"**
- The wizard will automatically fall back to pip
- Install uv for faster installs: `pip install uv`

**"Local provider not detected"**
- Ensure Ollama or LM Studio is running
- Check the service is on the expected port

**"API key format invalid"**
- The wizard provides format validation
- You can proceed with unusual formats if desired

**"Permission denied"**
- Check file permissions in the project directory
- Run from a user account with write access

### Debug Mode

For detailed troubleshooting, you can run the diagnostics directly:

```python
from src.config.diagnostics import run_diagnostics
results = run_diagnostics()
print(results)
```

## Architecture

The wizard uses a modular design:

- **Diagnostics Module**: `src/config/diagnostics.py`
- **Wizard Engine**: `src/config/streamlined_wizard.py`
- **Entry Point**: `setup_wizard.py`
- **Test Suite**: `test_wizard.py`

Each component is independently testable and replaceable.