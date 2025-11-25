"""
Core module for Conjecture - Internal components only
Main Conjecture class is now in src/conjecture.py
"""

import logging

# Export core models and utilities
from src.core.models import Claim, ClaimState, ClaimType

# Configure logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
