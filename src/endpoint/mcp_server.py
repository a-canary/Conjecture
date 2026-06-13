# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Conjecture MCP Server - Model Context Protocol Integration

Per A-0013: MCP Delivery Model
Expose Conjecture as an MCP server with tools:
  - build_context(query): Build claim context for a query
  - upsert_claim(claim, confidence, super_ids, sub_ids): Create or update a claim
  - explore_next(): Get the next claim to explore
  - get_claim_support(claim_or_query): Get supporting claims

Any MCP-compatible client (Claude Desktop, Cursor, custom) can use
Conjecture as a reasoning backend.
"""

import asyncio
import logging
from typing import List, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from src.endpoint.conjecture_endpoint import ConjectureEndpoint
from src.data.models import ClaimType, ClaimState

logger = logging.getLogger(__name__)

# Create the MCP server
mcp = FastMCP("Conjecture")

# Global endpoint instance (initialized on server start)
_endpoint: Optional[ConjectureEndpoint] = None


async def get_endpoint() -> ConjectureEndpoint:
    """Get or initialize the Conjecture endpoint."""
    global _endpoint
    if _endpoint is None:
        _endpoint = ConjectureEndpoint()
        await _endpoint.initialize()
    return _endpoint


@mcp.tool()
async def build_context(query: str, max_claims: int = 10) -> dict:
    """Build claim context for a natural language query.

    Retrieves relevant claims from the knowledge base and assembles
    them into a context blob suitable for LLM reasoning.

    Args:
        query: Natural language query to build context for
        max_claims: Maximum number of claims to include (default: 10)

    Returns:
        Context blob with relevant claims and metadata
    """
    endpoint = await get_endpoint()
    response = await endpoint.evaluate(
        query=query,
        max_claims=max_claims,
        include_reasoning=True
    )

    if response.success and response.data:
        return {
            "query": query,
            "claims": response.data.get("claims_found", []),
            "context": response.data.get("reasoning", []),
            "status": "success"
        }
    else:
        return {
            "query": query,
            "claims": [],
            "context": [],
            "status": "error",
            "errors": response.errors
        }


@mcp.tool()
async def upsert_claim(
    content: str,
    confidence: float = 0.5,
    super_ids: Optional[List[str]] = None,
    sub_ids: Optional[List[str]] = None,
    tags: Optional[List[str]] = None
) -> dict:
    """Create or update a claim in the knowledge base.

    If the claim content already exists, updates confidence and relationships.
    Otherwise, creates a new claim.

    Args:
        content: Claim content (10-5000 characters)
        confidence: Confidence score (0.0-1.0, default: 0.5)
        super_ids: IDs of claims this claim supports (provides evidence FOR)
        sub_ids: IDs of claims that support this claim
        tags: Optional tags for categorization

    Returns:
        Result with claim ID and operation status
    """
    endpoint = await get_endpoint()
    response = await endpoint.create_claim(
        content=content,
        confidence=confidence,
        supers=super_ids or [],
        subs=sub_ids or [],
        tags=tags or []
    )

    if response.success and response.data:
        return {
            "claim_id": response.data.get("id"),
            "content": content,
            "confidence": confidence,
            "status": "created",
            "message": response.message
        }
    else:
        return {
            "claim_id": None,
            "status": "error",
            "errors": response.errors
        }


@mcp.tool()
async def explore_next() -> dict:
    """Get the next claim to explore from the dirty queue.

    Returns the highest-priority claim that needs evaluation based on
    the [dirty, confidence, root_similarity] priority tuple (D-0002).

    Returns:
        Next claim to explore, or empty if no dirty claims
    """
    endpoint = await get_endpoint()

    # Search for dirty claims sorted by priority
    response = await endpoint.search_claims(
        min_confidence=0.0,
        max_confidence=1.0,
        limit=1
    )

    if response.success and response.data:
        claims = response.data.get("claims", [])
        if claims:
            claim = claims[0]
            return {
                "claim_id": claim.get("id"),
                "content": claim.get("content"),
                "confidence": claim.get("confidence"),
                "state": claim.get("state"),
                "status": "found"
            }
        else:
            return {
                "claim_id": None,
                "status": "empty",
                "message": "No claims to explore"
            }
    else:
        return {
            "claim_id": None,
            "status": "error",
            "errors": response.errors
        }


@mcp.tool()
async def get_claim_support(claim_id: str) -> dict:
    """Get the supporting claims (subs) for a claim.

    Retrieves all claims that provide evidence for the specified claim,
    forming the support tree beneath it.

    Args:
        claim_id: ID of the claim to get support for

    Returns:
        Support tree with sub-claims
    """
    endpoint = await get_endpoint()
    response = await endpoint.get_claim(claim_id)

    if response.success and response.data:
        claim_data = response.data
        sub_ids = claim_data.get("subs", [])

        # Fetch all sub-claims
        sub_claims = []
        for sub_id in sub_ids:
            sub_response = await endpoint.get_claim(sub_id)
            if sub_response.success and sub_response.data:
                sub_claims.append(sub_response.data)

        return {
            "claim_id": claim_id,
            "content": claim_data.get("content"),
            "confidence": claim_data.get("confidence"),
            "support_count": len(sub_claims),
            "supporting_claims": sub_claims,
            "status": "success"
        }
    else:
        return {
            "claim_id": claim_id,
            "supporting_claims": [],
            "status": "not_found",
            "errors": response.errors
        }


def run_server(host: str = "localhost", port: int = 3000):
    """Run the Conjecture MCP server.

    Args:
        host: Host to bind to (default: localhost)
        port: Port to listen on (default: 3000)
    """
    logger.info(f"Starting Conjecture MCP server on {host}:{port}")
    mcp.run()


def main():
    """Entry point for CLI: conjecture mcp"""
    import argparse
    parser = argparse.ArgumentParser(description="Conjecture MCP Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=3000, help="Port to listen on")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
