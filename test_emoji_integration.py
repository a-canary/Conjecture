#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.verbose_logger import VerboseLogger, VerboseLevel

print("Testing emoji package integration with verbose logger...")
logger = VerboseLogger(VerboseLevel.USER)

print("\n=== Testing claim assessment with emojis ===")
logger.claim_assessed_confident("c0000001", 0.9, 0.8)
logger.claim_assessed_confident("c0000002", 0.6, 0.8)
logger.claim_resolved("c0000001", 0.9)
logger.final_response("This is a test response with emoji package support")

print("\n=== Testing tool execution ===")
logger.tool_executed("WebSearch", {"query": "test"}, {"success": True})

print("\n=== Testing user communication ===")
logger.user_tool_executed("TellUser", "Hello from the system!")

print("\n=== Testing process logging ===")
logger.process_start("Test process")
logger.finish()

print("\n✅ Emoji package integration test completed!")