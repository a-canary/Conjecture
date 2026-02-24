# Conjecture Simplified Architecture Guide

## Overview

Conjecture has been simplified to support **only OpenAI-compatible endpoints**, making the system more maintainable, reliable, and easier to use.

## 🎯 Key Changes

### Removed Complexity
- **Eliminated 9 different provider integrations** → **Single unified provider**
- **Simplified configuration** → Only OpenAI-compatible endpoints
- **Reduced codebase by ~87%** → Cleaner, more focused architecture
- **Standardized API format** → Consistent behavior across all providers

### What's Supported
✅ **OpenAI-Compatible Providers:**
- OpenAI (api.openai.com)
- OpenRouter (openrouter.ai) 
- Chutes.ai (llm.chutes.ai)
- LM Studio (localhost:1234)
- Ollama (localhost:11434/v1)
- Any other OpenAI-compatible endpoint

❌ **Removed Providers:**
- Anthropic (different API format)
- Google/Gemini (different API format)
- Cohere (different API format)
- Groq (different API format)

## 🏗️ New Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 Conjecture Core                    │
├─────────────────────────────────────────────────────────┤
│  SimplifiedLLMManager                          │
│  ├─ OpenAICompatibleProcessor                   │
│  └─ ProviderConfig (Pydantic)               │
├─────────────────────────────────────────────────────────┤
│  SimplifiedConfigManager                        │
│  └─ SimplifiedConfig (JSON)                   │
├─────────────────────────────────────────────────────────┤
│  CLI & Wizards                                   │
│  └─ simplified_wizard.py                      │
└─────────────────────────────────────────────────────────┘
```

## 📁 File Structure

### New Core Files
```
src/
├── processing/
│   ├── simplified_llm_manager.py          # 🆕 Unified LLM manager
│   ├── llm/
│   │   ├── openai_compatible_provider.py  # 🆕 Single provider for all
│   │   ├── common.py                    # Shared utilities
│   │   └── error_handling.py           # Error handling
│   └── simplified_config.py              # 🆕 Configuration management
├── config/
│   ├── simplified_config.py             # 🆕 Config models
│   ├── simplified_wizard.py             # 🆕 CLI wizard
│   └── default_simplified_config.json  # 🆕 Default config
└── conjecture.py                            # Updated to use simplified
```

### Removed Files
```
src/processing/llm/
├── anthropic_integration.py          # ❌ Removed
├── chutes_integration.py           # ❌ Removed  
├── openrouter_integration.py       # ❌ Removed
├── openai_integration.py          # ❌ Removed
├── groq_integration.py            # ❌ Removed
├── gemini_integration.py          # ❌ Removed
├── google_integration.py          # ❌ Removed
├── cohere_integration.py          # ❌ Removed
├── local_providers_adapter.py    # ❌ Removed
├── lm_studio_adapter.py         # ❌ Removed
├── llm_manager.py               # ❌ Removed
└── router_provider.py           # ❌ Removed
```

## ⚙️ Configuration

### Simplified Format
```json
{
  "providers": [
    {
      "name": "lm_studio",
      "url": "http://localhost:1234",
      "api": "",
      "model": "ibm/granite-4-h-tiny",
      "priority": 1
    },
    {
      "name": "openai", 
      "url": "https://api.openai.com/v1",
      "api": "your-openai-api-key-here",
      "model": "gpt-3.5-turbo",
      "priority": 2
    }
  ],
  "confidence_threshold": 0.95,
  "max_context_size": 10000,
  "batch_size": 10,
  "database_path": "data/conjecture.db",
  "user": "user",
  "team": "default",
  "debug": false
}
```

### Provider Priority
1. **Local providers** (priority 1-2) - Preferred for privacy
2. **Cloud providers** (priority 3+) - Fallback options

## 🚀 Usage

### Quick Setup
```bash
# Auto-detect local providers
python -m src.config.simplified_wizard init

# Quick setup with common cloud providers  
python -m src.config.simplified_wizard setup

# Manual configuration
python -m src.config.simplified_wizard setup --manual

# Test current configuration
python -m src.config.simplified_wizard test

# Show current config
python -m src.config.simplified_wizard show
```

### Programmatic Usage
```python
from src.config.simplified_config import SimplifiedConfigManager
from src.processing.simplified_llm_manager import SimplifiedLLMManager
from src.processing.llm.openai_compatible_provider import OpenAICompatibleProcessor

# Load configuration
config_manager = SimplifiedConfigManager()
config = config_manager.config

# Create LLM manager
llm_manager = SimplifiedLLMManager()

# Create provider directly
provider = OpenAICompatibleProcessor(
    api_key="your-key",
    api_url="https://api.openai.com/v1", 
    model_name="gpt-3.5-turbo",
    provider_name="openai"
)

# Use provider
result = provider.generate_response("Hello, world!")
```

## 🔧 Adding New Providers

### Adding OpenAI-Compatible Providers

Any service that uses the OpenAI `/chat/completions` endpoint format can be added:

1. **Add to configuration:**
```python
from src.config.simplified_config import ProviderConfig

new_provider = ProviderConfig(
    name="my_provider",
    url="https://api.myprovider.com/v1",
    api="your-api-key",
    model="my-model",
    priority=5
)

config_manager.add_provider(new_provider)
config_manager.save_config()
```

2. **Provider will work automatically** - No code changes needed!

### Provider Requirements
- **Endpoint**: Must support `/chat/completions` (or equivalent)
- **Request Format**: Standard OpenAI message format
- **Response Format**: Standard OpenAI choices format
- **Authentication**: Bearer token in Authorization header (optional for local)

## 🧪 Testing

### Run Tests
```bash
# Test simplified architecture
python -m pytest tests/test_simplified_architecture.py -v

# Test provider connectivity
python -m src.config.simplified_wizard test

# Validate configuration
python -m src.config.simplified_wizard validate
```

### Test Coverage
- ✅ Provider configuration validation
- ✅ OpenAI-compatible API communication  
- ✅ Provider fallback mechanism
- ✅ Priority-based provider selection
- ✅ Error handling and retry logic
- ✅ Health check functionality

## 🔄 Migration Guide

### From Old Architecture
```bash
# 1. Backup current config
cp ~/.conjecture/config.json ~/.conjecture/config.json.backup

# 2. Run migration wizard  
python -m src.config.simplified_wizard init

# 3. Verify migration
python -m src.config.simplified_wizard validate
```

### Manual Migration
If you have custom providers, update your `~/.conjecture/config.json`:

```json
{
  "providers": [
    {
      "name": "your_provider",
      "url": "https://api.yourprovider.com/v1", 
      "api": "your-api-key",
      "model": "your-model",
      "priority": 1
    }
  ]
}
```

## 🎁 Benefits

### For Users
- **Simpler setup** - Just configure URL, key, model
- **More reliable** - Single codebase, fewer bugs
- **Better performance** - Optimized for OpenAI format
- **Easier troubleshooting** - One provider type to debug
- **Consistent behavior** - Same features across all providers

### For Developers  
- **87% less code** - Easier to maintain
- **Single provider interface** - Add new providers easily
- **Standardized testing** - One test suite for all
- **Cleaner architecture** - Better separation of concerns
- **Future-proof** - Easy to extend and modify

## 🔍 Troubleshooting

### Common Issues

#### Provider Not Found
```bash
# Check configuration
python -m src.config.simplified_wizard show

# Test connectivity
python -m src.config.simplified_wizard test

# Verify URL format
curl -X GET "https://your-provider-url/v1/models"
```

#### API Key Issues
```bash
# Test API key manually
curl -X POST "https://api.openai.com/v1/chat/completions" \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"test"}],"max_tokens":10}'
```

#### Model Not Available
```bash
# List available models
curl -X GET "https://api.openai.com/v1/models" \
  -H "Authorization: Bearer YOUR_KEY"

# Check provider-specific models
curl -X GET "https://your-provider-url/v1/models"
```

### Debug Mode
Enable debug logging in configuration:
```json
{
  "debug": true
}
```

## 📚 API Reference

### OpenAICompatibleProcessor
```python
class OpenAICompatibleProcessor:
    def __init__(self, api_key: str, api_url: str, 
                 model_name: str, provider_name: str):
        """Initialize provider"""
    
    def generate_response(self, prompt: str, 
                       config: Optional[GenerationConfig] = None) -> LLMProcessingResult:
        """Generate response from provider"""
    
    def process_claims(self, claims: List[Claim], task: str = "analyze",
                       config: Optional[GenerationConfig] = None) -> LLMProcessingResult:
        """Process claims using provider"""
    
    def health_check(self) -> Dict[str, Any]:
        """Check provider health"""
```

### SimplifiedLLMManager
```python
class SimplifiedLLMManager:
    def __init__(self, providers: Optional[List[Dict]] = None):
        """Initialize with provider list"""
    
    def get_processor(self, provider: Optional[str] = None) -> OpenAICompatibleProcessor:
        """Get provider with fallback"""
    
    def process_claims(self, claims: List[Claim], **kwargs) -> LLMProcessingResult:
        """Process claims with automatic fallback"""
    
    def generate_response(self, prompt: str, **kwargs) -> LLMProcessingResult:
        """Generate response with automatic fallback"""
    
    def health_check(self) -> Dict[str, Any]:
        """Check all provider health"""
```

## 🚀 Future Enhancements

### Planned Features
- **Streaming support** - For real-time responses
- **Async provider support** - Better performance
- **Provider templates** - Quick setup for popular services
- **Automatic model detection** - Find best available model
- **Load balancing** - Distribute requests across providers
- **Cost tracking** - Monitor API usage and costs

### Extension Points
- **Custom providers** - Plugin architecture for special cases
- **Middleware support** - Request/response processing
- **Metrics collection** - Performance and usage analytics
- **Provider discovery** - Auto-detect new services

---

**This simplified architecture makes Conjecture more maintainable while supporting all major OpenAI-compatible LLM providers.**