# Conjecture Configuration

This directory contains your Conjecture configuration and data.

## config.json

The `config.json` file contains your LLM provider configurations with automatic failover support.

### Example Configuration

```json
{
  "providers": [
    {
      "url": "http://localhost:11434",
      "api": "",
      "model": "llama2"
    },
    {
      "url": "https://openrouter.ai/api/v1",
      "api": "sk-or-your-openrouter-key-here",
      "model": "openai/gpt-3.5-turbo"
    },
    {
      "url": "https://api.openai.com/v1",
      "api": "sk-your-openai-key-here",
      "model": "gpt-3.5-turbo"
    }
  ]
}
```

### Provider Configuration

Each provider requires:
- **url**: The API endpoint URL
- **api**: Your API key (empty for local providers)
- **model**: The model name to use

### Failover Behavior

Conjecture will automatically:
1. Try the first available provider
2. Fail over to the next provider on errors
3. Respect rate limits and retry after delays
4. Track provider health and availability

### Workspace vs Global Configuration

- **Global config**: `~/.conjecture/config.json` - Used when no workspace config exists
- **Workspace config**: `{workspace}/.conjecture/config.json` - Overrides global config

## Directory Structure

```
~/.conjecture/
├── config.json          # LLM provider configuration
├── tools/               # User/LLM-generated tools
├── db/                   # SQLite databases
├── logs/                 # Application logs
├── sessions/             # Session data
└── db_exports/           # Database exports
```