#!/usr/bin/env python3
"""
Run improvement cycle with proper encoding handling
"""
import asyncio
import sys
import os
from pathlib import Path

# Set UTF-8 encoding for stdout
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "benchmarking"))

from improvement_cycle_agent import run_cycle_1

if __name__ == "__main__":
    asyncio.run(run_cycle_1())