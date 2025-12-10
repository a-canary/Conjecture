#!/usr/bin/env python3
"""
Comprehensive Tests for Modular Conjecture CLI
Tests all backends, commands, and functionality
"""

import unittest
import tempfile
import os
import sys
import json
import sqlite3
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cli.base_cli import BaseCLI, ClaimValidationError, DatabaseError, BackendNotAvailableError
from cli.backends.local_backend import LocalBackend
from cli.backends.cloud_backend import CloudBackend
from cli.backends import BACKEND_REGISTRY
