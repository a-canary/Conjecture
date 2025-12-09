"""
Endpoint Layer - Public API Interface

This module provides the public API interface for the Conjecture system.
It acts as the entry point for external interactions, handling request
validation, routing, and response formatting.

The Endpoint layer is part of the 4-layer architecture:
Presentation → Endpoint → Process → Data
"""

from .conjecture_endpoint import ConjectureEndpoint

__all__ = ["ConjectureEndpoint"]