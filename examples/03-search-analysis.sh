#!/bin/bash
# Conjecture Quickstart - Search and Analysis
# Run this after creating some claims to search and analyze them
#
# Prerequisites:
#   pip install -e .
#   Some claims in your database (run 01-ledger-quickstart.sh first)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=========================================="
echo "Conjecture Quickstart: Search & Analysis"
echo "=========================================="
echo ""

# Step 1: Search for claims
echo "1. Searching for claims about 'sky'..."
python3 conjecture search "sky" || echo "  (No claims found - run 01-ledger-quickstart.sh first)"
echo ""

# Step 2: Search for claims about weather
echo "2. Searching for claims about 'weather'..."
python3 conjecture search "weather" || echo "  (No claims found)"
echo ""

# Step 3: Show stats
echo "3. Database statistics..."
python3 conjecture stats
echo ""

echo "=========================================="
echo "Search & Analysis Tips:"
echo ""
echo "  To analyze a claim (requires LLM provider):"
echo "    python3 conjecture analyze <claim-id>"
echo ""
echo "  To see claim tree:"
echo "    python3 conjecture tree <claim-id>"
echo ""
echo "  To trace claim provenance:"
echo "    python3 conjecture trace <claim-id>"
echo "=========================================="