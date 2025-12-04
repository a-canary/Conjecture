#!/usr/bin/env python3
"""Test script to validate all imports needed for research framework"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

print("Testing imports...")

try:
    from config.common import ProviderConfig
    print("[OK] ProviderConfig imported")
except Exception as e:
    print(f"[FAIL] ProviderConfig failed: {e}")

try:
    from core.models import Claim, ClaimState, ClaimType
    print("[OK] core.models imported")
except Exception as e:
    print(f"[FAIL] core.models failed: {e}")

try:
    from processing.llm.llm_manager import LLMManager
    print("[OK] LLMManager imported")
except Exception as e:
    print(f"[FAIL] LLMManager failed: {e}")

try:
    from utils.id_generator import generate_template_id, generate_context_id
    print("[OK] utils.id_generator imported")
except Exception as e:
    print(f"[FAIL] utils.id_generator failed: {e}")

print("Import testing complete.")