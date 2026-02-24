"""
Minimal file utilities for cross-platform file handling.
"""

import os
import tempfile
from pathlib import Path


class FileHandler:
    """Simple file handler for test fixtures."""

    def get_safe_temp_dir(self, prefix: str = "conjecture_") -> Path:
        """Create a safe temporary directory."""
        temp_dir = tempfile.mkdtemp(prefix=prefix)
        return Path(temp_dir)

    def safe_path(self, path: str) -> Path:
        """Return a safe cross-platform path."""
        return Path(path).resolve()


# Global file handler instance
file_handler = FileHandler()
