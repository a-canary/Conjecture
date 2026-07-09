# Conjecture Configuration

This directory contains your Conjecture configuration and data.

## config.json

The `config.json` file contains your LLM provider configurations with automatic failover support.

**This file is gitignored.** Copy `.conjecture/config.example.json` (or
`src/config/default_config.json`) to `.conjecture/config.json` and edit locally.
Never commit API keys — prefer loading them from environment variables (see
[API keys](#api-keys) below).

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
      "api": "",
      "env_var": "OPENROUTER_API_KEY",
      "model": "openai/gpt-3.5-turbo"
    },
    {
      "url": "https://api.openai.com/v1",
      "api": "",
      "env_var": "OPENAI_API_KEY",
      "model": "gpt-3.5-turbo"
    }
  ]
}
```

### Provider Configuration

Each provider requires:
- **url**: The API endpoint URL
- **api**: Your API key (empty for local providers, or set via `env_var`)
- **env_var**: Optional env var name to read the API key from at runtime
- **model**: The model name to use

### API keys

The loader resolves each provider's API credential in this order:

1. The provider's `env_var` (e.g. `OPENROUTER_API_KEY`, `CHUTES_API_KEY`)
   if set in the environment
2. The literal `api` field in `config.json` (for local providers, `""` is fine)

Recommended: leave `api: ""` in `config.json` and export the secret at runtime:

```sh
export OPENROUTER_API_KEY="sk-or-v1-..."
conjecture run ...
```

This keeps `config.json` safe to share and out of `.gitignore` accidents.

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