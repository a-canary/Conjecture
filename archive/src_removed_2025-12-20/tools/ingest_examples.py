"""
Tool Example Ingestor
Extracts examples from tool files and ingests them as standard Claims.
"""

import asyncio
import inspect
import logging
import sys
from typing import List

from src.core.models import Claim, create_claim
from src.data.data_manager import get_data_manager
from src.tools.registry import get_tool_registry

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def ingest_tool_examples():
    """
    Discover all tools, extract their examples, and save them as Claims.
    """
    logger.info("Starting tool example ingestion...")

    # Initialize DataManager
    data_manager = get_data_manager()
    await data_manager.initialize()

    # Get Tool Registry
    registry = get_tool_registry()

    examples_to_ingest = []

    # Iterate through all registered tools (Core + Optional)
    all_tools = {**registry.core_tools, **registry.optional_tools}

    for tool_name, tool_info in all_tools.items():
        logger.info(f"Processing tool: {tool_name}")

        # Try to find 'examples' function directly in the tool function's globals
        examples_func = tool_info.func.__globals__.get("examples")

        if examples_func and inspect.isfunction(examples_func):
            try:
                examples = examples_func()
                if isinstance(examples, list):
                    for example_text in examples:
                        # Create a Claim for each example
                        # We use tags to identify it as an example for a specific tool
                        claim = create_claim(
                            content=example_text,
                            confidence=1.0,  # Examples are authoritative
                            tag="tool_example",  # Primary tag
                            tags=["tool_example", tool_name, "example"],
                        )
                        examples_to_ingest.append(claim)
                        logger.info(f"  Found example: {example_text[:50]}...")
            except Exception as e:
                logger.error(f"Error extracting examples from {tool_name}: {e}")
        else:
            logger.info(f"  No examples() function found for {tool_name}")

    # Batch persist the claims
    if examples_to_ingest:
        logger.info(f"Persisting {len(examples_to_ingest)} example claims...")
        # Convert Claims to dicts for batch_create_claims
        claims_data = [
            claim.model_dump(exclude={"dirty"}) for claim in examples_to_ingest
        ]
        result = await data_manager.batch_create_claims(claims_data)

        if result.failure_count > 0:
            logger.error(
                f"Failed to persist {result.failure_count} claims. Errors: {result.errors}"
            )
        else:
            logger.info("All examples successfully ingested!")
    else:
        logger.info("No examples found to ingest.")

    await data_manager.close()

if __name__ == "__main__":
    asyncio.run(ingest_tool_examples())
