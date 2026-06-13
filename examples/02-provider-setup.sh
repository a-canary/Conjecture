#!/bin/bash
# Conjecture Quickstart - Provider Setup
# Run this to configure LLM providers
#
# Prerequisites:
#   pip install -e .

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=========================================="
echo "Conjecture Quickstart: Provider Setup"
echo "=========================================="
echo ""

# Step 1: Show available providers
echo "1. Showing available providers..."
python3 conjecture providers
echo ""

# Step 2: Show current backends
echo "2. Checking available backends..."
python3 conjecture backends
echo ""

# Step 3: Configuration help
echo "3. Configuration location info..."
echo "   User config: ~/.conjecture/config.json"
echo "   Workspace config: .conjecture/config.json"
echo "   Default config: src/config/default_config.json"
echo ""

echo "=========================================="
echo "Provider Setup Tips:"
echo ""
echo "  Option A: Local LLM (Ollama)"
echo "    1. Install Ollama: https://ollama.ai/"
echo "    2. Start: ollama serve"
echo "    3. Pull model: ollama pull llama2"
echo "    4. Edit ~/.conjecture/config.json:"
echo '       {"providers": [{"name": "ollama", "url": "http://localhost:11434", "model": "llama2"}]}'
echo ""
echo "  Option B: Cloud Provider (e.g., Cerebras)"
echo "    1. Get API key: https://cloud.cerebras.ai/settings/api-keys"
echo "    2. Edit ~/.conjecture/config.json:"
echo '       {"providers": [{"name": "cerebras", "url": "https://api.cerebras.ai/v1", "api": "YOUR-KEY", "model": "llama3.1-8b"}]}'
echo ""
echo "  After setup, test with:"
echo "    python3 conjecture health"
echo "=========================================="