#!/usr/bin/env python3
"""
Base CLI functionality for Conjecture - Test Version
"""

import asyncio
import json
import os
import sqlite3
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config import get_config, validate_config


class BaseCLI(ABC):
    """Base class for all CLI implementations with common functionality."""

    def __init__(self, name: str = "conjecture", help_text: str = "Conjecture CLI"):
        self.name = name
        self.help_text = help_text
        self.console = Console()
        self.error_console = Console(stderr=True)
        self.embedding_model = None
        self.current_provider_config = None
        self.db_path = "data/conjecture.db"

    def _get_backend_type(self) -> str:
        """Get the backend type for this CLI."""
        return self.__class__.__name__.replace("CLI", "").lower()

    def _init_services(self):
        """Initialize services common to all backends."""
        # Validate configuration first
        result = validate_config()

        if not result.success:
            self.console.print("[bold red]❌ Configuration Required[/bold red]")
            self._print_validation_result(result)
            raise SystemExit(1)

        # Get the best configured provider
        self.current_provider_config = self._get_configured_provider()

        # Create data directory
        os.makedirs("data", exist_ok=True)

        # Initialize sentence transformer
        if self.embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.console.print("[green][OK][/green] Model loaded successfully")
            except ImportError:
                self.console.print(
                    "[red]Error: sentence-transformers not installed[/red]"
                )
                raise SystemExit(1)

    def _init_database(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS claims (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                confidence REAL NOT NULL,
                user_id TEXT NOT NULL,
                embedding BLOB,
                metadata TEXT,
                tags TEXT,
                is_dirty BOOLEAN DEFAULT 1,
                dirty_reason TEXT,
                dirty_priority INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        return self.embedding_model.encode(text)

    def _save_claim(
        self,
        content: str,
        confidence: float,
        user_id: str,
        metadata: dict = None,
        tags: list = None,
    ) -> str:
        """Save claim to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Generate embedding
        embedding = self._generate_embedding(content)
        embedding_bytes = embedding.tobytes()

        # Generate ID
        claim_id = f"c{int(time.time() * 1000) % 1000000000:07d}"

        # Save claim
        cursor.execute(
            """
            INSERT INTO claims (id, content, confidence, user_id, embedding, metadata, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                claim_id,
                content,
                confidence,
                user_id,
                embedding_bytes,
                json.dumps(metadata) if metadata else None,
                json.dumps(tags) if tags else None,
            ),
        )

        conn.commit()
        conn.close()
        return claim_id

    # Abstract methods that must be implemented by backends
    @abstractmethod
    def create_claim(
        self,
        content: str,
        confidence: float,
        user_id: str,
        analyze: bool = False,
        tags: list = None,
        **kwargs,
    ) -> str:
        """Create a new claim."""
        pass

    @abstractmethod
    def get_claim(self, claim_id: str) -> Optional[dict]:
        """Get a claim by ID."""
        pass

    @abstractmethod
    def search_claims(self, query: str, limit: int = 10, **kwargs) -> List[dict]:
        """Search claims by content."""
        pass

    @abstractmethod
    def analyze_claim(self, claim_id: str, **kwargs) -> Dict[str, Any]:
        """Analyze a claim using backend-specific services."""
        pass

    @abstractmethod
    def process_prompt(
        self, prompt_text: str, confidence: float = 0.8, verbose: int = 0, **kwargs
    ) -> Dict[str, Any]:
        """Process user prompt as claim with dirty evaluation."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available and properly configured."""
        pass


class ClaimValidationError(Exception):
    """Exception raised for claim validation errors."""

    pass


class DatabaseError(Exception):
    """Exception raised for database operations."""

    pass


class BackendNotAvailableError(Exception):
    """Exception raised when backend is not available."""

    pass
