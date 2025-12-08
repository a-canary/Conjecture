#!/usr/bin/env python3
"""
Backend registry and management for Conjecture CLI
Provides pluggable backend system for different LLM providers
"""

from typing import Dict, Type, Any, List
from ..base_cli import BaseCLI, BackendNotAvailableError


class BackendRegistry:
    """Registry for managing CLI backends"""
    
    def __init__(self):
        self._backends: Dict[str, Type[BaseCLI]] = {}
        self._instances: Dict[str, BaseCLI] = {}
    
    def register(self, name: str, backend_class: Type[BaseCLI]) -> None:
        """Register a backend class"""
        self._backends[name] = backend_class
    
    def get_backend_class(self, name: str) -> Type[BaseCLI]:
        """Get a backend class by name"""
        if name not in self._backends:
            raise BackendNotAvailableError(f"Backend '{name}' not registered")
        return self._backends[name]
    
    def get_backend(self, name: str, config: Dict[str, Any] = None) -> BaseCLI:
        """Get or create a backend instance"""
        if name not in self._instances:
            backend_class = self.get_backend_class(name)
            self._instances[name] = backend_class(config)
        return self._instances[name]
    
    def list_backends(self) -> List[str]:
        """List all registered backend names"""
        return list(self._backends.keys())
    
    def get_available_backends(self) -> List[str]:
        """Get list of backends that are available"""
        available = []
        for name in self.list_backends():
            try:
                backend = self.get_backend(name)
                if backend.is_available():
                    available.append(name)
            except Exception:
                pass
        return available


# Global backend registry instance
BACKEND_REGISTRY = BackendRegistry()


# Import and register backends
try:
    from .local_backend import LocalBackend
    BACKEND_REGISTRY.register("local", LocalBackend)
except ImportError:
    pass

try:
    from .cloud_backend import CloudBackend
    BACKEND_REGISTRY.register("cloud", CloudBackend)
except ImportError:
    pass

try:
    from .simple_backend import SimpleBackend
    BACKEND_REGISTRY.register("simple", SimpleBackend)
except ImportError:
    pass

# Export main components
__all__ = [
    "BACKEND_REGISTRY",
    "BackendRegistry",
    "BaseCLI",
    "BackendNotAvailableError",
]