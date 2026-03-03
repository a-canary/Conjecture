"""
Endpoint Layer - Public API for Conjecture

The Endpoint layer is the single entry point for all external consumers.
Per A-0003, ConjectureEndpoint provides three core methods:
  - create_claim(): Create a new claim
  - get_claim(): Retrieve a claim by ID
  - evaluate(): Evaluate claims with LLM reasoning

Per A-0007, all methods return standardized APIResponse wrappers.
Per A-0013, MCP server exposes Conjecture as a reasoning backend.
"""

from src.endpoint.conjecture_endpoint import ConjectureEndpoint, APIResponse

# MCP server imported separately to avoid mandatory dependency
# Use: from src.endpoint.mcp_server import mcp, run_server

__all__ = ["ConjectureEndpoint", "APIResponse"]
