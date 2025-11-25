#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.verbose_logger import VerboseLogger, VerboseLevel

print("Testing emoji support...")
logger = VerboseLogger(VerboseLevel.USER)

print("Testing claim assessment with emojis...")
logger.claim_assessed_confident("c0000001", 0.9, 0.8)
logger.claim_assessed_confident("c0000002", 0.6, 0.8)
logger.claim_resolved("c0000001", 0.9)
logger.final_response("This is a test response with emoji support")

print("Testing tool execution...")
logger.tool_executed("WebSearch", {"query": "test"}, {"success": True})

print("Testing process logging...")
logger.process_start("Test process")
logger.finish()

print("Emoji support test completed!")