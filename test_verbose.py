#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.verbose_logger import VerboseLogger, VerboseLevel

print("Starting test...")
logger = VerboseLogger(VerboseLevel.USER)
print(f"Logger created with level: {logger.level}")
print(f"Level value: {logger.level.value}")
print(f"User level value: {VerboseLevel.USER.value}")
print(f"Should log: {logger.level.value >= VerboseLevel.USER.value}")

print("Testing direct print...")
print("[TEST] Test message")

print("Testing logger...")
logger._log(VerboseLevel.USER, "Test message from logger", "[TEST]")

print("Testing claim assessment...")
logger.claim_assessed_confident("c0000001", 0.9, 0.8)

print("Test completed")