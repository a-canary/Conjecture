#!/bin/bash
# Conjecture Quickstart - Ledger Operations
# Run this to learn the core claim management commands
#
# Prerequisites:
#   pip install -e .

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=========================================="
echo "Conjecture Quickstart: Ledger Operations"
echo "=========================================="
echo ""

# Step 1: Check configuration
echo "1. Checking configuration..."
python3 conjecture config
echo ""

# Step 2: Create a test claim
echo "2. Creating a test claim..."
python3 conjecture create "The sky is blue during clear weather."
echo ""

# Step 3: List recent claims (using stats as a proxy - search doesn't have easy list mode)
echo "3. Checking database stats..."
python3 conjecture stats
echo ""

# Step 4: Health check
echo "4. Running health check..."
python3 conjecture health
echo ""

echo "=========================================="
echo "Quickstart complete!"
echo ""
echo "Next steps:"
echo "  - Run: python3 conjecture create 'Your claim here'"
echo "  - Run: python3 conjecture search 'blue sky'"
echo "  - Run: python3 conjecture analyze <claim-id>"
echo "  - Run: python3 conjecture quickstart  # Full guide"
echo "=========================================="