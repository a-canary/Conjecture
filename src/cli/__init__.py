#!/usr/bin/env python3
"""
Conjecture CLI Package
Modular command-line interface with pluggable backends
"""

from .base_cli import BaseCLI, ClaimValidationError, DatabaseError, BackendNotAvailableError
from .backends import BACKEND_REGISTRY, LocalBackend, CloudBackend, HybridBackend, AutoBackend
from .modular_cli import app

__all__ = [
    'BaseCLI',
    'ClaimValidationError', 
    'DatabaseError',
    'BackendNotAvailableError',
    'BACKEND_REGISTRY',
    'LocalBackend',
    'CloudBackend', 
    'HybridBackend',
    'AutoBackend',
    'app'
]