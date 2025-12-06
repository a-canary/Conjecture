#!/usr/bin/env python3
"""
Conjecture CLI Package
Modular command-line interface with pluggable backends
"""

from .base_cli import (
    BaseCLI,
    ClaimValidationError,
    DatabaseError,
    BackendNotAvailableError,
)
from .backends import BACKEND_REGISTRY
from .backends.local_backend import LocalBackend
from .backends.cloud_backend import CloudBackend
# from .modular_cli import app

__all__ = [
    "BaseCLI",
    "ClaimValidationError",
    "DatabaseError",
    "BackendNotAvailableError",
    "BACKEND_REGISTRY",
    "LocalBackend",
    "CloudBackend",
    # 'HybridBackend',
    # 'AutoBackend',
    # 'app'
]
