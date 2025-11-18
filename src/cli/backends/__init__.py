#!/usr/bin/env python3
"""
Conjecture CLI Backends
Pluggable backend implementations for different service configurations
"""

from .local_backend import LocalBackend
from .cloud_backend import CloudBackend
from .hybrid_backend import HybridBackend
from .auto_backend import AutoBackend

# Backend registry for easy access
BACKEND_REGISTRY = {
    'local': LocalBackend,
    'cloud': CloudBackend,
    'hybrid': HybridBackend,
    'auto': AutoBackend
}

__all__ = [
    'LocalBackend',
    'CloudBackend', 
    'HybridBackend',
    'AutoBackend',
    'BACKEND_REGISTRY'
]