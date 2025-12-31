#!/usr/bin/env python3
"""
Database Reset Utility for Standardized Benchmark Testing
Provides clean slate for consistent benchmark measurements
"""

import sqlite3
import shutil
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List

class DatabaseResetter:
    """Manages database state for standardized benchmark testing"""

    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path) if db_path else Path(__file__).parent.parent.parent / "data" / "conjecture.db"
        self.backup_dir = self.db_path.parent / "backups"
        self.backup_dir.mkdir(exist_ok=True)

    def create_backup(self) -> str:
        """Create backup of current database"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"conjecture_backup_{timestamp}.db"

        if self.db_path.exists():
            shutil.copy2(self.db_path, backup_path)
            print(f"Database backed up to: {backup_path}")
            return str(backup_path)
        else:
            print("No existing database to backup")
            return str(backup_path)

    def reset_database(self, create_fresh: bool = True) -> None:
        """Reset database to clean state"""
        # Create backup first
        self.create_backup()

        # Remove existing database
        if self.db_path.exists():
            os.remove(self.db_path)
            print(f"Removed existing database: {self.db_path}")

        # Create fresh database if requested
        if create_fresh:
            self.create_fresh_database()

    def create_fresh_database(self) -> None:
        """Create fresh database with standard schema"""
        # Import Conjecture database manager
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        try:
            from database.core.database_manager import ConjectureDatabaseManager

            # Create database with standard schema
            db_manager = ConjectureDatabaseManager(str(self.db_path))
            print(f"Created fresh database: {self.db_path}")

        except ImportError as e:
            # Fallback: create minimal database structure
            self._create_minimal_database()

    def _create_minimal_database(self) -> None:
        """Create minimal database structure for benchmarking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Basic tables needed for Conjecture operation
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS claims (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_uuid TEXT UNIQUE NOT NULL,
                claim_text TEXT NOT NULL,
                confidence_score REAL DEFAULT 0.0,
                evidence_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tools (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name TEXT UNIQUE NOT NULL,
                tool_description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_uuid TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()
        print(f"Created minimal database structure: {self.db_path}")

    def prime_with_examples(self, example_problems: List[dict] = None) -> None:
        """Prime database with example problems (not benchmark problems)"""
        if not example_problems:
            # Default warm-up examples
            example_problems = [
                {
                    "claim": "Basic arithmetic operations follow consistent mathematical rules",
                    "confidence": 0.95,
                    "evidence": "Mathematical axioms and centuries of verification"
                },
                {
                    "claim": "Logic puzzles require careful step-by-step reasoning",
                    "confidence": 0.90,
                    "evidence": "Formal logic principles and problem-solving methodologies"
                },
                {
                    "claim": "Mathematical proofs require rigorous logical deduction",
                    "confidence": 0.95,
                    "evidence": "Mathematical proof theory and formal verification"
                },
                {
                    "claim": "Word problems often contain hidden assumptions that need clarification",
                    "confidence": 0.85,
                    "evidence": "Educational research on mathematical problem-solving"
                },
                {
                    "claim": "Multiple solution approaches can lead to the same correct answer",
                    "confidence": 0.90,
                    "evidence": "Mathematical problem-solving theory and practice"
                }
            ]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for example in example_problems:
            cursor.execute('''
                INSERT OR REPLACE INTO claims (claim_uuid, claim_text, confidence_score, evidence_count)
                VALUES (?, ?, ?, ?)
            ''', (
                f"example_{hash(example['claim']) % 1000000}",
                example['claim'],
                example['confidence'],
                1  # Basic evidence count
            ))

        conn.commit()
        conn.close()
        print(f"Primed database with {len(example_problems)} example claims")

    def verify_clean_state(self) -> bool:
        """Verify database is in clean standardized state"""
        if not self.db_path.exists():
            return False

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        required_tables = ['claims', 'tools', 'sessions']
        if not all(table in tables for table in required_tables):
            conn.close()
            return False

        # Check claim count is reasonable (not too many from previous runs)
        cursor.execute("SELECT COUNT(*) FROM claims")
        claim_count = cursor.fetchone()[0]

        conn.close()

        # Allow reasonable number of primed examples (5-20)
        return 5 <= claim_count <= 20

# Convenience function for benchmark setup
def setup_benchmark_environment(
    db_path: str = None,
    prime_examples: bool = True,
    verify_state: bool = True
) -> DatabaseResetter:
    """Setup standardized benchmark environment"""

    resetter = DatabaseResetter(db_path)

    print("Setting up benchmark environment...")
    print(f"Database path: {resetter.db_path}")

    # Reset to clean state
    resetter.reset_database(create_fresh=True)

    # Prime with examples if requested
    if prime_examples:
        resetter.prime_with_examples()

    # Verify clean state
    if verify_state:
        if resetter.verify_clean_state():
            print("✓ Database verified in clean standardized state")
        else:
            print("✗ Database state verification failed")

    return resetter

# Command-line interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reset Conjecture database for benchmark testing")
    parser.add_argument("--db-path", help="Path to database file")
    parser.add_argument("--no-prime", action="store_true", help="Don't prime with examples")
    parser.add_argument("--verify", action="store_true", default=True, help="Verify clean state")
    parser.add_argument("--backup-only", action="store_true", help="Only create backup")
    parser.add_argument("--check-state", action="store_true", help="Check current database state")

    args = parser.parse_args()

    if args.backup_only:
        resetter = DatabaseResetter(args.db_path)
        resetter.create_backup()
    elif args.check_state:
        resetter = DatabaseResetter(args.db_path)
        if resetter.verify_clean_state():
            print("✓ Database is in clean standardized state")
        else:
            print("✗ Database is NOT in clean standardized state")
    else:
        resetter = setup_benchmark_environment(
            db_path=args.db_path,
            prime_examples=not args.no_prime,
            verify_state=args.verify
        )
        print("Benchmark environment setup complete!")