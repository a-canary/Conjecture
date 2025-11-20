#!/usr/bin/env python3
"""
Conjecture CLI Backends
Pluggable backend implementations for different service configurations
"""

from .auto import AutoBackend
from .cloud import CloudBackend
from .hybrid import HybridBackend
from .local import LocalBackend

# Backend registry for easy access
BACKEND_REGISTRY = {
    "local": LocalBackend,
    "cloud": CloudBackend,
    "hybrid": HybridBackend,
    "auto": AutoBackend,
}

__all__ = [
    "LocalBackend",
    "CloudBackend",
    "HybridBackend",
    "AutoBackend",
    "BACKEND_REGISTRY",
]
