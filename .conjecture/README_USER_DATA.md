# Conjecture User Data Directory

This directory contains your personal Conjecture data and configuration.

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

## Configuration

Edit `config.json` to configure your LLM providers. See the README in this directory for more information.

## Tools Directory

Place custom tools in the `tools/` directory. These will be available alongside the core tools that ship with Conjecture.

## Data Storage

- **db/**: SQLite databases containing your claims and metadata
- **logs/**: Application logs for debugging and monitoring
- **sessions/**: Session data for multi-tasking support
- **db_exports/**: Exported database files for backup or sharing

## Privacy

All data in this directory is stored locally and never transmitted unless explicitly configured to do so in your provider settings.