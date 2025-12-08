"""
Tests for Enhanced Modular CLI
Tests the improved CLI with better UX and progress tracking
"""

import pytest
import asyncio
import time
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typer.testing import CliRunner
from typing import Dict, Any, List
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the enhanced CLI
from src.cli.enhanced_modular_cli import app, get_ui_enhancer
from src.cli.ui_enhancements import UIConfig, OutputMode, VerbosityLevel


class TestEnhancedCLI:
    """Test the enhanced modular CLI"""

    @pytest.fixture
    def mock_runner(self):
        """Create a mock Typer runner"""
        return CliRunner()

    @pytest.fixture
    def mock_backend(self):
        """Create a mock backend"""
        backend = Mock()
        backend.create_claim.return_value = "c0000001"
        backend.get_claim.return_value = {
            "id": "c0000001",
            "content": "Test claim",
            "confidence": 0.8,
            "tags": ["test"],
            "state": "validated",
            "created_at": "2023-01-01T00:00:00Z",
            "created_by": "test_user"
        }
        backend.search_claims.return_value = [
            {
                "id": "c0000001",
                "content": "Test claim about search",
                "confidence": 0.85,
                "tags": ["test", "search"],
                "state": "validated"
            }
        ]
        backend.analyze_claim.return_value = {
            "claim_id": "c0000001",
            "backend": "test",
            "analysis_type": "comprehensive",
            "confidence_score": 0.9,
            "sentiment": "positive",
            "topics": ["test", "analysis"],
            "verification_status": "verified"
        }
        backend.process_prompt.return_value = "Prompt processed successfully"
        backend.get_backend_info.return_value = {
            "name": "test_backend",
            "configured": True,
            "provider": "test_provider"
        }
        backend._get_backend_type.return_value = "test"
        backend.provider_manager = Mock()
        backend.provider_manager.get_providers.return_value = [
            {"name": "test_provider", "url": "http://test.com"}
        ]
        return backend

    def test_app_creation(self):
        """Test that the enhanced app is created properly"""
        assert app is not None
        assert app.info.name == "conjecture"
        assert "Enhanced user experience" in app.info.help

    @patch('src.cli.enhanced_modular_cli.get_backend')
    def test_create_command_success(self, mock_get_backend, mock_runner, mock_backend):
        """Test successful claim creation with enhanced CLI"""
        mock_get_backend.return_value = mock_backend

        result = mock_runner.invoke(app, [
            "create",
            "Test claim content",
            "--confidence", "0.8",
            "--user", "test_user"
        ])

        # Should succeed
        assert result.exit_code == 0
        mock_backend.create_claim.assert_called_once_with(
            "Test claim content", 0.8, "test_user", False
        )

    @patch('src.cli.enhanced_modular_cli.get_backend')
    def test_create_command_with_analysis(self, mock_get_backend, mock_runner, mock_backend):
        """Test claim creation with analysis"""
        mock_get_backend.return_value = mock_backend

        result = mock_runner.invoke(app, [
            "create",
            "Test claim content",
            "--confidence", "0.8",
            "--user", "test_user",
            "--analyze"
        ])

        assert result.exit_code == 0
        mock_backend.create_claim.assert_called_once_with(
            "Test claim content", 0.8, "test_user", True
        )

    @patch('src.cli.enhanced_modular_cli.get_backend')
    def test_create_command_with_tags(self, mock_get_backend, mock_runner, mock_backend):
        """Test claim creation with tags"""
        mock_get_backend.return_value = mock_backend

        result = mock_runner.invoke(app, [
            "create",
            "Test claim content",
            "--confidence", "0.8",
            "--user", "test_user",
            "--tag", "test",
            "--tag", "example"
        ])

        assert result.exit_code == 0
        mock_backend.create_claim.assert_called_once()

    @patch('src.cli.enhanced_modular_cli.get_backend')
    def test_create_command_validation_error(self, mock_get_backend, mock_runner, mock_backend):
        """Test claim creation with validation error"""
        mock_get_backend.return_value = mock_backend

        # Test with too short content
        result = mock_runner.invoke(app, [
            "create",
            "Short",  # Less than 5 characters
            "--confidence", "0.8",
            "--user", "test_user"
        ])

        assert result.exit_code == 1

        # Test with invalid confidence
        result = mock_runner.invoke(app, [
            "create",
            "Valid claim content",
            "--confidence", "1.5",  # Invalid confidence
            "--user", "test_user"
        ])

        assert result.exit_code == 1

    @patch('src.cli.enhanced_modular_cli.get_backend')
    def test_get_command_success(self, mock_get_backend, mock_runner, mock_backend):
        """Test successful claim retrieval"""
        mock_get_backend.return_value = mock_backend

        result = mock_runner.invoke(app, [
            "get",
            "c0000001"
        ])

        assert result.exit_code == 0
        mock_backend.get_claim.assert_called_once_with("c0000001")

    @patch('src.cli.enhanced_modular_cli.get_backend')
    def test_get_command_not_found(self, mock_get_backend, mock_runner, mock_backend):
        """Test claim retrieval when claim not found"""
        mock_backend.get_claim.return_value = None
        mock_get_backend.return_value = mock_backend

        result = mock_runner.invoke(app, [
            "get",
            "c9999999"
        ])

        assert result.exit_code == 1
        mock_backend.get_claim.assert_called_once_with("c9999999")

    @patch('src.cli.enhanced_modular_cli.get_backend')
    def test_search_command_success(self, mock_get_backend, mock_runner, mock_backend):
        """Test successful claim search"""
        mock_get_backend.return_value = mock_backend

        result = mock_runner.invoke(app, [
            "search",
            "test query",
            "--limit", "5"
        ])

        assert result.exit_code == 0
        mock_backend.search_claims.assert_called_once_with("test query", 5)

    @patch('src.cli.enhanced_modular_cli.get_backend')
    def test_search_command_no_results(self, mock_get_backend, mock_runner, mock_backend):
        """Test claim search with no results"""
        mock_backend.search_claims.return_value = []
        mock_get_backend.return_value = mock_backend

        result = mock_runner.invoke(app, [
            "search",
            "no results query"
        ])

        assert result.exit_code == 0  # Still successful, just no results
        mock_backend.search_claims.assert_called_once()

    @patch('src.cli.enhanced_modular_cli.get_backend')
    def test_analyze_command_success(self, mock_get_backend, mock_runner, mock_backend):
        """Test successful claim analysis"""
        mock_get_backend.return_value = mock_backend

        result = mock_runner.invoke(app, [
            "analyze",
            "c0000001"
        ])

        assert result.exit_code == 0
        mock_backend.get_claim.assert_called_once_with("c0000001")
        mock_backend.analyze_claim.assert_called_once_with("c0000001")

    @patch('src.cli.enhanced_modular_cli.get_backend')
    def test_analyze_command_claim_not_found(self, mock_get_backend, mock_runner, mock_backend):
        """Test claim analysis when claim not found"""
        mock_backend.get_claim.return_value = None
        mock_get_backend.return_value = mock_backend

        result = mock_runner.invoke(app, [
            "analyze",
            "c9999999"
        ])

        assert result.exit_code == 1

    @patch('src.cli.enhanced_modular_cli.get_backend')
    def test_prompt_command_success(self, mock_get_backend, mock_runner, mock_backend):
        """Test successful prompt processing"""
        mock_get_backend.return_value = mock_backend

        result = mock_runner.invoke(app, [
            "prompt",
            "Test prompt text",
            "--confidence", "0.8",
            "--verbose", "1"
        ])

        assert result.exit_code == 0
        mock_backend.process_prompt.assert_called_once_with("Test prompt text", 0.8, 1)

    @patch('src.cli.enhanced_modular_cli.validate_config')
    @patch('src.cli.enhanced_modular_cli.get_backend')
    def test_config_command_success(self, mock_get_backend, mock_validate, mock_runner, mock_backend):
        """Test configuration check"""
        mock_get_backend.return_value = mock_backend
        mock_validate.return_value = True

        result = mock_runner.invoke(app, [
            "config"
        ])

        assert result.exit_code == 0

    @patch('src.cli.enhanced_modular_cli.get_backend')
    def test_stats_command_success(self, mock_get_backend, mock_runner, mock_backend):
        """Test statistics command"""
        mock_backend._get_database_stats.return_value = {
            "total_claims": 100,
            "avg_confidence": 0.75,
            "unique_users": 5
        }
        mock_backend.db_path = "/test/path"
        mock_get_backend.return_value = mock_backend

        result = mock_runner.invoke(app, [
            "stats"
        ])

        assert result.exit_code == 0
        mock_backend._get_database_stats.assert_called_once()

    @patch('src.cli.enhanced_modular_cli.get_backend')
    def test_health_command_success(self, mock_get_backend, mock_runner, mock_backend):
        """Test health check command"""
        mock_get_backend.return_value = mock_backend

        result = mock_runner.invoke(app, [
            "health"
        ])

        assert result.exit_code == 0

    @patch('src.cli.enhanced_modular_cli.get_backend')
    def test_health_command_no_providers(self, mock_get_backend, mock_runner):
        """Test health check with no providers"""
        mock_backend = Mock()
        mock_backend.provider_manager.get_providers.return_value = []
        mock_get_backend.return_value = mock_backend

        result = mock_runner.invoke(app, [
            "health"
        ])

        assert result.exit_code == 0  # Still successful, just warning

    def test_quickstart_command(self, mock_runner):
        """Test quickstart command"""
        result = mock_runner.invoke(app, [
            "quickstart"
        ])

        assert result.exit_code == 0


class TestEnhancedCLIOutputModes:
    """Test enhanced CLI with different output modes"""

    @pytest.fixture
    def mock_backend(self):
        """Create a mock backend"""
        backend = Mock()
        backend.create_claim.return_value = "c0000001"
        backend.get_claim.return_value = {
            "id": "c0000001",
            "content": "Test claim",
            "confidence": 0.8,
            "tags": ["test"],
            "state": "validated"
        }
        backend.provider_manager = Mock()
        backend.provider_manager.get_providers.return_value = []
        return backend

    @patch('src.cli.enhanced_modular_cli.get_backend')
    def test_json_output_mode(self, mock_get_backend, mock_runner, mock_backend):
        """Test JSON output mode"""
        mock_get_backend.return_value = mock_backend

        result = mock_runner.invoke(app, [
            "--json",
            "config"
        ])

        assert result.exit_code == 0
        # Output should be valid JSON
        try:
            json.loads(result.stdout)
        except json.JSONDecodeError:
            pytest.fail("Output is not valid JSON")

    @patch('src.cli.enhanced_modular_cli.get_backend')
    def test_quiet_mode(self, mock_get_backend, mock_runner, mock_backend):
        """Test quiet output mode"""
        mock_get_backend.return_value = mock_backend

        result = mock_runner.invoke(app, [
            "--quiet",
            "health"
        ])

        assert result.exit_code == 0
        # Output should be minimal in quiet mode

    @patch('src.cli.enhanced_modular_cli.get_backend')
    def test_verbose_mode(self, mock_get_backend, mock_runner, mock_backend):
        """Test verbose output mode"""
        mock_get_backend.return_value = mock_backend

        result = mock_runner.invoke(app, [
            "--verbose",
            "create",
            "Test claim content",
            "--confidence", "0.8"
        ])

        assert result.exit_code == 0
        # Should have verbose output

    @patch('src.cli.enhanced_modular_cli.get_backend')
    def test_no_progress_mode(self, mock_get_backend, mock_runner, mock_backend):
        """Test no progress indicators mode"""
        mock_get_backend.return_value = mock_backend

        result = mock_runner.invoke(app, [
            "--no-progress",
            "create",
            "Test claim content",
            "--confidence", "0.8"
        ])

        assert result.exit_code == 0

    @patch('src.cli.enhanced_modular_cli.get_backend')
    def test_no_color_mode(self, mock_get_backend, mock_runner, mock_backend):
        """Test no color mode"""
        mock_get_backend.return_value = mock_backend

        result = mock_runner.invoke(app, [
            "--no-color",
            "create",
            "Test claim content",
            "--confidence", "0.8"
        ])

        assert result.exit_code == 0

    @patch('src.cli.enhanced_modular_cli.get_backend')
    def test_accessible_mode(self, mock_get_backend, mock_runner, mock_backend):
        """Test accessible mode"""
        mock_get_backend.return_value = mock_backend

        result = mock_runner.invoke(app, [
            "--accessible",
            "create",
            "Test claim content",
            "--confidence", "0.8"
        ])

        assert result.exit_code == 0


class TestEnhancedCLIErrorHandling:
    """Test enhanced CLI error handling"""

    @pytest.fixture
    def mock_runner(self):
        """Create a mock Typer runner"""
        return CliRunner()

    @patch('src.cli.enhanced_modular_cli.get_backend')
    def test_backend_initialization_failure(self, mock_get_backend, mock_runner):
        """Test CLI handles backend initialization failure"""
        mock_get_backend.side_effect = Exception("Backend initialization failed")

        result = mock_runner.invoke(app, [
            "create",
            "Test claim content",
            "--confidence", "0.8"
        ])

        assert result.exit_code == 1

    @patch('src.cli.enhanced_modular_cli.get_backend')
    def test_create_command_backend_error(self, mock_get_backend, mock_runner):
        """Test create command handles backend errors"""
        mock_backend = Mock()
        mock_backend.create_claim.side_effect = Exception("Database error")
        mock_get_backend.return_value = mock_backend

        result = mock_runner.invoke(app, [
            "create",
            "Test claim content",
            "--confidence", "0.8"
        ])

        assert result.exit_code == 1

    @patch('src.cli.enhanced_modular_cli.get_backend')
    def test_search_command_backend_error(self, mock_get_backend, mock_runner):
        """Test search command handles backend errors"""
        mock_backend = Mock()
        mock_backend.search_claims.side_effect = Exception("Search error")
        mock_get_backend.return_value = mock_backend

        result = mock_runner.invoke(app, [
            "search",
            "test query"
        ])

        assert result.exit_code == 1


class TestEnhancedCLIIntegration:
    """Integration tests for enhanced CLI"""

    @pytest.fixture
    def mock_backend(self):
        """Create a comprehensive mock backend"""
        backend = Mock()
        backend.create_claim.return_value = "c0000001"
        backend.get_claim.return_value = {
            "id": "c0000001",
            "content": "Test claim content",
            "confidence": 0.8,
            "tags": ["test", "integration"],
            "state": "validated",
            "created_at": "2023-01-01T00:00:00Z",
            "created_by": "test_user"
        }
        backend.search_claims.return_value = [
            {
                "id": "c0000001",
                "content": "Integration test claim",
                "confidence": 0.85,
                "tags": ["test", "integration"],
                "state": "validated"
            }
        ]
        backend.analyze_claim.return_value = {
            "claim_id": "c0000001",
            "backend": "test",
            "analysis_type": "comprehensive",
            "confidence_score": 0.9,
            "sentiment": "positive",
            "topics": ["test", "integration"],
            "verification_status": "verified",
            "reasoning": "This is a test reasoning for the analysis"
        }
        backend.process_prompt.return_value = "Integration test prompt processed"
        backend.get_backend_info.return_value = {
            "name": "test_backend",
            "configured": True,
            "provider": "test_provider",
            "type": "test"
        }
        backend._get_backend_type.return_value = "test"
        backend._get_database_stats.return_value = {
            "total_claims": 50,
            "avg_confidence": 0.78,
            "unique_users": 3
        }
        backend.db_path = "/test/integration.db"
        backend.provider_manager = Mock()
        backend.provider_manager.get_providers.return_value = [
            {
                "name": "test_provider",
                "url": "http://test.example.com",
                "status": "active"
            }
        ]
        return backend

    @patch('src.cli.enhanced_modular_cli.validate_config')
    @patch('src.cli.enhanced_modular_cli.get_backend')
    def test_complete_workflow(self, mock_get_backend, mock_validate, mock_runner, mock_backend):
        """Test a complete workflow with the enhanced CLI"""
        mock_get_backend.return_value = mock_backend
        mock_validate.return_value = True

        # Step 1: Check configuration
        result = mock_runner.invoke(app, ["config"])
        assert result.exit_code == 0

        # Step 2: Check health
        result = mock_runner.invoke(app, ["health"])
        assert result.exit_code == 0

        # Step 3: Create a claim
        result = mock_runner.invoke(app, [
            "create",
            "Integration test claim with enhanced features",
            "--confidence", "0.85",
            "--user", "integration_test_user",
            "--tag", "integration",
            "--tag", "test",
            "--analyze"
        ])
        assert result.exit_code == 0

        # Step 4: Get the claim
        result = mock_runner.invoke(app, ["get", "c0000001"])
        assert result.exit_code == 0

        # Step 5: Search claims
        result = mock_runner.invoke(app, [
            "search",
            "integration",
            "--limit", "5"
        ])
        assert result.exit_code == 0

        # Step 6: Analyze the claim
        result = mock_runner.invoke(app, ["analyze", "c0000001"])
        assert result.exit_code == 0

        # Step 7: Process a prompt
        result = mock_runner.invoke(app, [
            "prompt",
            "Integration test prompt for enhanced CLI",
            "--confidence", "0.9"
        ])
        assert result.exit_code == 0

        # Step 8: Get statistics
        result = mock_runner.invoke(app, ["stats"])
        assert result.exit_code == 0

        # Step 9: Quick start guide
        result = mock_runner.invoke(app, ["quickstart"])
        assert result.exit_code == 0

        # All commands should have succeeded
        assert True

    @patch('src.cli.enhanced_modular_cli.get_backend')
    def test_workflow_with_different_output_modes(self, mock_get_backend, mock_runner, mock_backend):
        """Test workflow with different output modes"""
        mock_get_backend.return_value = mock_backend

        output_modes = [
            ["--output", "normal"],
            ["--quiet"],
            ["--verbose"],
            ["--json"],
            ["--accessible"],
            ["--no-progress"],
            ["--no-color"]
        ]

        for mode_args in output_modes:
            # Test each mode with a simple command
            result = mock_runner.invoke(app, mode_args + ["health"])
            assert result.exit_code == 0, f"Failed with mode: {mode_args}"

    @patch('src.cli.enhanced_modular_cli.get_backend')
    def test_error_recovery_workflow(self, mock_get_backend, mock_runner):
        """Test workflow error recovery"""
        # Mock a backend that initially fails, then succeeds
        call_count = 0
        def mock_get_backend_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Temporary backend failure")
            else:
                backend = Mock()
                backend.provider_manager = Mock()
                backend.provider_manager.get_providers.return_value = []
                return backend

        mock_get_backend.side_effect = mock_get_backend_side_effect

        # First call should fail
        result = mock_runner.invoke(app, ["health"])
        assert result.exit_code == 1

        # Second call should succeed (error recovery)
        result = mock_runner.invoke(app, ["health"])
        assert result.exit_code == 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])