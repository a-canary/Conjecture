"""
Conjecture: Simplified configuration package
"""

from .simple_config import (
    Config,
    config,
    get_config,
    print_config_summary,
    validate_config,
)

__all__ = ["Config", "config", "get_config", "validate_config", "print_config_summary"]
