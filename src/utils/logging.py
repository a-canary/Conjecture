"""
Logging utilities for Conjecture
"""

import logging
import sys
from typing import Optional

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Configure console handler if not already configured
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger

def setup_logger(name: str, level: Optional[int] = None, handlers: Optional[list] = None) -> logging.Logger:
    """
    Setup a logger with custom configuration.
    
    Args:
        name: Logger name
        level: Logging level (optional)
        handlers: List of handlers (optional)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set level if provided
    if level is not None:
        logger.setLevel(level)
    else:
        logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Add custom handlers if provided
    if handlers:
        for handler in handlers:
            logger.addHandler(handler)
    else:
        # Add default console handler
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def setup_logging(level: str = "INFO", format_string: Optional[str] = None):
    """
    Setup global logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[logging.StreamHandler(sys.stdout)]
    )

class ClaimParseError(Exception):
    """Exception raised for claim parsing errors"""
    pass

class ClaimValidationError(Exception):
    """Exception raised for claim validation errors"""
    pass

class ClaimFormatError(Exception):
    """Exception raised for claim format errors"""
    pass