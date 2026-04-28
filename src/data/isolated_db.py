"""
Isolated Database Factory for Worktree Testing

Ensures each branch/worktree test uses an isolated database
to prevent context contamination between experiments.
"""

import os
import sqlite3
import tempfile
import hashlib
from pathlib import Path
from typing import Optional
from contextlib import contextmanager


class IsolatedDBFactory:
    """
    Creates isolated database instances for testing.

    Each worktree/branch gets its own database to prevent:
    - Cross-experiment contamination
    - Stale claim pollution
    - False positive learning effects
    """

    def __init__(self, db_dir: Optional[Path] = None):
        # Allow override via constructor or ISOLATED_DB_DIR env var
        if db_dir:
            self.DB_DIR = db_dir
        elif os.environ.get("ISOLATED_DB_DIR"):
            self.DB_DIR = Path(os.environ["ISOLATED_DB_DIR"])
        else:
            # Fallback to project test directory or temp
            project_test_dir = Path(__file__).parent.parent.parent / ".test_dbs"
            self.DB_DIR = project_test_dir
        self.DB_DIR.mkdir(parents=True, exist_ok=True)

    def get_branch_name(self) -> str:
        """Get current git branch name."""
        try:
            result = os.popen("git rev-parse --abbrev-ref HEAD 2>/dev/null").read().strip()
            return result or "default"
        except:
            return "default"

    def get_worktree_id(self) -> str:
        """Get unique worktree identifier."""
        try:
            # Get worktree root path
            result = os.popen("git rev-parse --show-toplevel 2>/dev/null").read().strip()
            if result:
                # Hash the path to get unique ID
                return hashlib.md5(result.encode()).hexdigest()[:8]
        except:
            pass
        return "main"

    def get_experiment_db_path(self, experiment_name: str) -> Path:
        """
        Get isolated DB path for a specific experiment.

        Format: .test_dbs/{worktree_id}_{branch}_{experiment}.db
        """
        worktree_id = self.get_worktree_id()
        branch = self.get_branch_name().replace("/", "_")

        db_name = f"{worktree_id}_{branch}_{experiment_name}.db"
        return self.DB_DIR / db_name

    def create_isolated_db(self, experiment_name: str) -> sqlite3.Connection:
        """
        Create a fresh isolated database for an experiment.

        Deletes any existing DB to ensure clean state.
        """
        db_path = self.get_experiment_db_path(experiment_name)

        # Remove existing to ensure clean state
        if db_path.exists():
            db_path.unlink()

        conn = sqlite3.connect(str(db_path))
        self._init_schema(conn)

        return conn

    def _init_schema(self, conn: sqlite3.Connection):
        """Initialize claim storage schema."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS claims (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                question TEXT,
                confidence REAL DEFAULT 0.5,
                is_correct INTEGER DEFAULT 0,
                category TEXT DEFAULT 'general',
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                experiment_run TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_claims_correct ON claims(is_correct)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_claims_confidence ON claims(confidence)
        """)
        conn.commit()

    @contextmanager
    def isolated_session(self, experiment_name: str):
        """
        Context manager for isolated experiment session.

        Usage:
            factory = IsolatedDBFactory()
            with factory.isolated_session("primacy_test") as conn:
                # All claims stored in isolated DB
                cursor = conn.cursor()
                cursor.execute("INSERT INTO claims ...")
        """
        conn = self.create_isolated_db(experiment_name)
        try:
            yield conn
        finally:
            conn.close()

    def cleanup_experiment(self, experiment_name: str):
        """Remove experiment database after completion."""
        db_path = self.get_experiment_db_path(experiment_name)
        if db_path.exists():
            db_path.unlink()

    def cleanup_all(self):
        """Remove all test databases."""
        for db_file in self.DB_DIR.glob("*.db"):
            db_file.unlink()

    def list_experiment_dbs(self) -> list:
        """List all experiment databases."""
        return list(self.DB_DIR.glob("*.db"))


class IsolatedClaimMemory:
    """
    Claim memory with automatic DB isolation.

    Use this in experiments to ensure no cross-contamination.
    """

    def __init__(self, experiment_name: str, auto_cleanup: bool = True):
        self.experiment_name = experiment_name
        self.auto_cleanup = auto_cleanup
        self.factory = IsolatedDBFactory()
        self.conn = self.factory.create_isolated_db(experiment_name)
        self._run_id = hashlib.md5(os.urandom(16)).hexdigest()[:8]

    def add_claim(
        self,
        content: str,
        question: str = "",
        confidence: float = 0.5,
        is_correct: bool = False,
        category: str = "general"
    ):
        """Add claim to isolated storage."""
        self.conn.execute(
            """INSERT INTO claims
               (content, question, confidence, is_correct, category, experiment_run)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (content[:200], question[:100], confidence, int(is_correct), category, self._run_id)
        )
        self.conn.commit()

    def get_claims(
        self,
        correct_only: bool = True,
        min_confidence: float = 0.0,
        limit: int = 10
    ) -> list:
        """Retrieve claims from isolated storage."""
        query = "SELECT content, confidence, is_correct, category FROM claims WHERE 1=1"
        params = []

        if correct_only:
            query += " AND is_correct = 1"

        if min_confidence > 0:
            query += " AND confidence >= ?"
            params.append(min_confidence)

        query += " ORDER BY confidence DESC LIMIT ?"
        params.append(limit)

        cursor = self.conn.execute(query, params)
        return [
            {"content": row[0], "confidence": row[1], "is_correct": bool(row[2]), "category": row[3]}
            for row in cursor.fetchall()
        ]

    def get_stats(self) -> dict:
        """Get memory statistics."""
        cursor = self.conn.execute(
            "SELECT COUNT(*), SUM(is_correct), AVG(confidence) FROM claims"
        )
        row = cursor.fetchone()
        return {
            "total": row[0] or 0,
            "correct": row[1] or 0,
            "avg_confidence": row[2] or 0.0,
            "experiment": self.experiment_name,
            "run_id": self._run_id
        }

    def close(self):
        """Close connection and optionally cleanup."""
        self.conn.close()
        if self.auto_cleanup:
            self.factory.cleanup_experiment(self.experiment_name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Convenience function
def create_isolated_memory(experiment_name: str) -> IsolatedClaimMemory:
    """Create isolated claim memory for an experiment."""
    return IsolatedClaimMemory(experiment_name)
