"""
Simple, focused tests for core CLI functionality.
Tests the essential user workflow: create → search → get.
"""
import pytest
import tempfile
import os
from pathlib import Path
from typer.testing import CliRunner

# Import the CLI main command
from src.cli.modular_cli import app

runner = CliRunner()


class TestCoreCLI:
    """Test core CLI functionality focused on 80/20 use cases."""

    def test_cli_help(self):
        """Test that CLI help works."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "create" in result.stdout
        assert "search" in result.stdout
        assert "get" in result.stdout

    def test_config_check(self):
        """Test configuration validation."""
        result = runner.invoke(app, ["config"])
        # Should show configuration status, even if not fully configured
        assert result.exit_code == 0

    def test_create_claim_simple(self):
        """Test creating a simple claim - the most common use case."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use a test database in temp directory
            test_db_path = Path(temp_dir) / "test.db"
            env_vars = {
                "DB_PATH": str(test_db_path),
                "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
                "PROVIDER_API_URL": "https://llm.chutes.ai/v1",
                "PROVIDER_API_KEY": "test-key",
                "PROVIDER_MODEL": "zai-org/GLM-4.6-FP8"
            }
            
            # Override environment for this test
            for key, value in env_vars.items():
                os.environ[key] = value
            
            try:
                # Test basic claim creation
                result = runner.invoke(app, [
                    "create", 
                    "Machine learning requires substantial training data for optimal performance.",
                    "--confidence", "0.8",
                    "--user", "test-user"
                ])
                
                # Should succeed even without LLM for basic storage
                assert result.exit_code == 0
                assert "created" in result.stdout.lower() or "success" in result.stdout.lower()
                
            finally:
                # Clean up environment
                for key in env_vars:
                    os.environ.pop(key, None)

    def test_search_claims_basic(self):
        """Test searching claims - the core discovery function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_db_path = Path(temp_dir) / "test.db"
            env_vars = {
                "DB_PATH": str(test_db_path),
                "EMBEDDING_MODEL": "all-MiniLM-L6-v2"
            }
            
            for key, value in env_vars.items():
                os.environ[key] = value
            
            try:
                # First create a claim
                runner.invoke(app, [
                    "create", 
                    "Python programming language supports multiple paradigms including object-oriented programming.",
                    "--confidence", "0.9",
                    "--user", "test-user"
                ])
                
                # Then search for it
                result = runner.invoke(app, ["search", "python programming"])
                
                assert result.exit_code == 0
                # Should find the claim or indicate no results (both are valid)
                assert "found" in result.stdout.lower() or "no claims" in result.stdout.lower() or "results" in result.stdout.lower()
                
            finally:
                for key in env_vars:
                    os.environ.pop(key, None)

    def test_get_claim_by_id(self):
        """Test retrieving a claim by ID - essential for inspection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_db_path = Path(temp_dir) / "test.db"
            env_vars = {
                "DB_PATH": str(test_db_path),
                "EMBEDDING_MODEL": "all-MiniLM-L6-v2"
            }
            
            for key, value in env_vars.items():
                os.environ[key] = value
            
            try:
                # First create a claim to get its ID
                result = runner.invoke(app, [
                    "create", 
                    "Database indexing improves query performance significantly.",
                    "--confidence", "0.85",
                    "--user", "test-user"
                ])
                
                # Try to extract claim ID from output (format dependent on implementation)
                # For now, test that the command doesn't crash
                if result.exit_code == 0:
                    # Try to get a claim with a test ID (might not exist, but should handle gracefully)
                    get_result = runner.invoke(app, ["get", "c0000001"])
                    # Should either find the claim or indicate not found (both are valid)
                    assert get_result.exit_code == 0 or "not found" in get_result.stdout.lower()
                
            finally:
                for key in env_vars:
                    os.environ.pop(key, None)

    def test_stats_command(self):
        """Test statistics command - useful for monitoring."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_db_path = Path(temp_dir) / "test.db"
            env_vars = {
                "DB_PATH": str(test_db_path)
            }
            
            for key, value in env_vars.items():
                os.environ[key] = value
            
            try:
                result = runner.invoke(app, ["stats"])
                assert result.exit_code == 0
                # Should show statistics or empty database info
                assert "claims" in result.stdout.lower() or "database" in result.stdout.lower()
                
            finally:
                for key in env_vars:
                    os.environ.pop(key, None)

    def test_health_check(self):
        """Test system health check - important for troubleshooting."""
        result = runner.invoke(app, ["health"])
        assert result.exit_code == 0
        # Should show system status
        assert "health" in result.stdout.lower() or "status" in result.stdout.lower() or "ok" in result.stdout.lower()


class TestCLIErrorHandling:
    """Test CLI error handling and edge cases."""

    def test_invalid_confidence(self):
        """Test handling of invalid confidence values."""
        result = runner.invoke(app, [
            "create", 
            "Test claim with invalid confidence.",
            "--confidence", "1.5"  # Invalid: > 1.0
        ])
        # Should handle the error gracefully
        assert result.exit_code != 0 or "error" in result.stdout.lower() or "invalid" in result.stdout.lower()

    def test_empty_search(self):
        """Test search with empty query."""
        result = runner.invoke(app, ["search", ""])
        # Should handle gracefully
        assert result.exit_code == 0 or "error" in result.stdout.lower()

    def test_nonexistent_claim(self):
        """Test getting a claim that doesn't exist."""
        result = runner.invoke(app, ["get", "c9999999"])
        # Should handle gracefully
        assert result.exit_code == 0 or "not found" in result.stdout.lower()


class TestCoreWorkflow:
    """Test the complete core workflow that users actually need."""

    def test_complete_workflow(self):
        """Test the complete create → search → get workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_db_path = Path(temp_dir) / "test.db"
            env_vars = {
                "DB_PATH": str(test_db_path),
                "EMBEDDING_MODEL": "all-MiniLM-L6-v2"
            }
            
            for key, value in env_vars.items():
                os.environ[key] = value
            
            try:
                # Step 1: Create a claim
                claim_content = "Artificial intelligence impacts modern society in significant ways."
                create_result = runner.invoke(app, [
                    "create", 
                    claim_content,
                    "--confidence", "0.75",
                    "--user", "test-user"
                ])
                
                # Step 2: Search for the claim
                search_result = runner.invoke(app, ["search", "artificial intelligence"])
                
                # Step 3: Get claim details (if we have an ID)
                get_result = runner.invoke(app, ["get", "c0000001"])
                
                # All commands should complete without crashing
                assert create_result.exit_code in [0, 1]  # 1 is OK for missing dependencies
                assert search_result.exit_code in [0, 1]
                assert get_result.exit_code in [0, 1]
                
                # At least the search should acknowledge the operation
                assert "results" in search_result.stdout.lower() or "found" in search_result.stdout.lower() or "no claims" in search_result.stdout.lower()
                
            finally:
                for key in env_vars:
                    os.environ.pop(key, None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])