#!/usr/bin/env python3
"""
Comprehensive Tests for Modular Conjecture CLI
Tests all backends, commands, and functionality
"""

import unittest
import tempfile
import os
import sys
import json
import sqlite3
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cli.base_cli import BaseCLI, ClaimValidationError, DatabaseError, BackendNotAvailableError
from cli.backends.local_backend import LocalBackend
from cli.backends.cloud_backend import CloudBackend
from cli.backends.hybrid_backend import HybridBackend
from cli.backends.auto_backend import AutoBackend
from cli.backends import BACKEND_REGISTRY


class TestBaseCLI(unittest.TestCase):
    """Test base CLI functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        
        # Mock base CLI class for testing
        class MockBaseCLI(BaseCLI):
            def __init__(self):
                super().__init__("mock", "Mock CLI")
                self.db_path = self.db_path
            
            def create_claim(self, content, confidence, user, analyze=False, **kwargs):
                return self._save_claim(content, confidence, user, {"test": True})
            
            def get_claim(self, claim_id):
                return self._get_claim(claim_id)
            
            def search_claims(self, query, limit=10, **kwargs):
                return self._search_claims(query, limit)
            
            def analyze_claim(self, claim_id, **kwargs):
                return {"claim_id": claim_id, "mock": True}
            
            def is_available(self):
                return True
        
        self.cli = MockBaseCLI()
    
    def tearDown(self):
        """Clean up after tests."""
        # Clean up temporary files
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_init_database(self):
        """Test database initialization."""
        self.cli._init_database()
        
        # Check that database file was created
        self.assertTrue(os.path.exists(self.db_path))
        
        # Check table structure
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='claims'")
        table_exists = cursor.fetchone() is not None
        conn.close()
        
        self.assertTrue(table_exists)
    
    def test_save_claim(self):
        """Test saving a claim."""
        self.cli._init_database()
        
        # Mock embedding model
        self.cli.embedding_model = Mock()
        self.cli.embedding_model.encode.return_value = [0.1, 0.2, 0.3]
        
        claim_id = self.cli._save_claim("Test claim", 0.8, "test_user", {"test": True})
        
        # Check that claim was saved
        claim = self.cli._get_claim(claim_id)
        self.assertIsNotNone(claim)
        self.assertEqual(claim['content'], "Test claim")
        self.assertEqual(claim['confidence'], 0.8)
        self.assertEqual(claim['user_id'], "test_user")
    
    def test_get_claim(self):
        """Test getting a claim."""
        self.cli._init_database()
        self.cli.embedding_model = Mock()
        self.cli.embedding_model.encode.return_value = [0.1, 0.2, 0.3]
        
        claim_id = self.cli._save_claim("Test claim", 0.8, "test_user")
        claim = self.cli._get_claim(claim_id)
        
        self.assertEqual(claim['content'], "Test claim")
        self.assertEqual(claim['confidence'], 0.8)
    
    def test_search_claims(self):
        """Test searching claims."""
        self.cli._init_database()
        self.cli.embedding_model = Mock()
        self.cli.embedding_model.encode.return_value = [0.1, 0.2, 0.3]
        
        # Save test claims
        self.cli._save_claim("Python programming", 0.9, "user1")
        self.cli._save_claim("Java development", 0.7, "user2")
        self.cli._save_claim("Python tutorials", 0.8, "user1")
        
        results = self.cli._search_claims("Python")
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIn("Python", result['content'])
    
    def test_get_backend_info(self):
        """Test getting backend information."""
        info = self.cli.get_backend_info()
        self.assertIn("name", info)
        self.assertIn("configured", info)
        self.assertEqual(info["name"], "mock")


class TestLocalBackend(unittest.TestCase):
    """Test local backend functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.backend = LocalBackend()
        self.temp_dir = tempfile.mkdtemp()
        self.backend.db_path = os.path.join(self.temp_dir, "test.db")
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.backend.db_path):
            os.remove(self.backend.db_path)
        os.rmdir(self.temp_dir)
    
    @patch('cli.backends.local_backend.validate_config')
    @patch('cli.backends.local_backend.get_primary_provider')
    def test_is_available_success(self, mock_get_provider, mock_validate):
        """Test backend availability when properly configured."""
        mock_validate.return_value = Mock(success=True)
        mock_provider = Mock()
        mock_provider.is_local = True
        mock_get_provider.return_value = mock_provider
        
        self.assertTrue(self.backend.is_available())
    
    @patch('cli.backends.local_backend.validate_config')
    def test_is_available_failure(self, mock_validate):
        """Test backend availability when not configured."""
        mock_validate.return_value = Mock(success=False)
        
        self.assertFalse(self.backend.is_available())
    
    @patch('cli.backends.local_backend.validate_config')
    @patch('cli.backends.local_backend.get_primary_provider')
    @patch.object(LocalBackend, '_init_services')
    @patch.object(LocalBackend, '_init_database')
    def test_create_claim_success(self, mock_init_db, mock_init_services, mock_get_provider, mock_validate):
        """Test successful claim creation."""
        mock_validate.return_value = Mock(success=True)
        mock_provider = Mock()
        mock_provider.is_local = True
        mock_provider.name = "ollama"
        mock_get_provider.return_value = mock_provider
        mock_init_services.return_value = None
        mock_init_db.return_value = None
        
        # Mock embedding generation
        self.backend.embedding_model = Mock()
        self.backend.embedding_model.encode.return_value = [0.1, 0.2, 0.3]
        
        claim_id = self.backend.create_claim("Test claim", 0.8, "test_user")
        
        self.assertIsNotNone(claim_id)
        self.assertTrue(claim_id.startswith('c'))
    
    def test_get_local_services_status(self):
        """Test getting local services status."""
        status = self.backend.get_local_services_status()
        
        self.assertIn("backend_type", status)
        self.assertIn("available", status)
        self.assertIn("supported_providers", status)
        self.assertIn("offline_capable", status)
        self.assertEqual(status["backend_type"], "local")
        self.assertTrue(status["offline_capable"])


class TestCloudBackend(unittest.TestCase):
    """Test cloud backend functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.backend = CloudBackend()
        self.temp_dir = tempfile.mkdtemp()
        self.backend.db_path = os.path.join(self.temp_dir, "test.db")
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.backend.db_path):
            os.remove(self.backend.db_path)
        os.rmdir(self.temp_dir)
    
    @patch('cli.backends.cloud_backend.validate_config')
    @patch('cli.backends.cloud_backend.get_primary_provider')
    def test_is_available_success(self, mock_get_provider, mock_validate):
        """Test backend availability when properly configured."""
        mock_validate.return_value = Mock(success=True)
        mock_provider = Mock()
        mock_provider.is_local = False
        mock_get_provider.return_value = mock_provider
        
        self.assertTrue(self.backend.is_available())
    
    def test_get_cloud_services_status(self):
        """Test getting cloud services status."""
        status = self.backend.get_cloud_services_status()
        
        self.assertIn("backend_type", status)
        self.assertIn("available", status)
        self.assertIn("supported_providers", status)
        self.assertIn("requires_internet", status)
        self.assertEqual(status["backend_type"], "cloud")
        self.assertTrue(status["requires_internet"])
    
    def test_list_cloud_models(self):
        """Test listing cloud models."""
        # Mock current provider config
        self.backend.current_provider_config = {"name": "OpenAI"}
        models = self.backend.list_cloud_models()
        
        self.assertIsInstance(models, list)
        self.assertIn("gpt-4", models)


class TestHybridBackend(unittest.TestCase):
    """Test hybrid backend functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.backend = HybridBackend()
        self.temp_dir = tempfile.mkdtemp()
        self.backend.db_path = os.path.join(self.temp_dir, "test.db")
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.backend.db_path):
            os.remove(self.backend.db_path)
        os.rmdir(self.temp_dir)
    
    @patch.object(LocalBackend, 'is_available')
    @patch.object(CloudBackend, 'is_available')
    def test_detect_available_backends(self, mock_cloud_available, mock_local_available):
        """Test backend detection."""
        mock_local_available.return_value = True
        mock_cloud_available.return_value = False
        
        available = self.backend._detect_available_backends()
        
        self.assertTrue(available["local"])
        self.assertFalse(available["cloud"])
    
    @patch.object(LocalBackend, 'is_available')
    @patch.object(CloudBackend, 'is_available')
    def test_is_available(self, mock_cloud_available, mock_local_available):
        """Test overall backend availability."""
        mock_local_available.return_value = False
        mock_cloud_available.return_value = True
        
        self.assertTrue(self.backend.is_available())
        
        mock_local_available.return_value = False
        mock_cloud_available.return_value = False
        
        self.assertFalse(self.backend.is_available())
    
    def test_set_preferred_mode(self):
        """Test setting preferred mode."""
        self.backend.set_preferred_mode("local")
        self.assertEqual(self.backend.preferred_mode, "local")
        
        self.backend.set_preferred_mode("cloud")
        self.assertEqual(self.backend.preferred_mode, "cloud")
        
        self.backend.set_preferred_mode("auto")
        self.assertEqual(self.backend.preferred_mode, "auto")
        
        # Test invalid mode
        with self.assertRaises(ValueError):
            self.backend.set_preferred_mode("invalid")


class TestAutoBackend(unittest.TestCase):
    """Test auto backend functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.backend = AutoBackend()
        self.temp_dir = tempfile.mkdtemp()
        self.backend.db_path = os.path.join(self.temp_dir, "test.db")
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.backend.db_path):
            os.remove(self.backend.db_path)
        os.rmdir(self.temp_dir)
    
    @patch.object(LocalBackend, 'is_available')
    @patch.object(CloudBackend, 'is_available')
    def test_run_detection(self, mock_cloud_available, mock_local_available):
        """Test backend detection."""
        mock_local_available.return_value = True
        mock_cloud_available.return_value = False
        
        detection = self.backend._run_detection()
        
        self.assertIn("available_backends", detection)
        self.assertIn("recommendations", detection)
        self.assertIn("performance_metrics", detection)
        self.assertTrue(detection["available_backends"]["local"])
        self.assertFalse(detection["available_backends"]["cloud"])
    
    def test_get_detection_report(self):
        """Test getting detection report."""
        with patch.object(self.backend, '_run_detection') as mock_detection:
            mock_detection.return_value = {
                "available_backends": {"local": True, "cloud": False},
                "recommendations": {"primary": "local", "reason": "Local available"},
                "timestamp": "2025-11-11T12:00:00Z"
            }
            
            report = self.backend.get_detection_report()
            
            self.assertIn("current_selection", report)
            self.assertIn("auto_detection", report)
    
    @patch.object(AutoBackend, '_select_and_initialize_backend')
    def test_create_claim(self, mock_select):
        """Test claim creation with auto backend."""
        mock_backend = Mock()
        mock_backend.create_claim.return_value = "c1234567"
        mock_select.return_value = mock_backend
        
        claim_id = self.backend.create_claim("Test claim", 0.8, "test_user")
        
        self.assertEqual(claim_id, "c1234567")
        mock_select.assert_called_once_with("create")
    
    def test_get_optimization_tips(self):
        """Test getting optimization tips."""
        with patch.object(self.backend, '_run_detection') as mock_detection:
            mock_detection.return_value = {
                "available_backends": {"local": True, "cloud": False},
                "recommendations": {"primary": "local", "reason": "Local available"}
            }
            
            tips = self.backend.get_optimization_tips()
            
            self.assertIsInstance(tips, list)
            self.assertGreater(len(tips), 0)


class TestBackendRegistry(unittest.TestCase):
    """Test backend registry functionality."""
    
    def test_backend_registry(self):
        """Test that all backends are properly registered."""
        # Test that registry contains expected backends
        expected_backends = {"local", "cloud", "hybrid", "auto"}
        self.assertEqual(set(BACKEND_REGISTRY.keys()), expected_backends)
        
        # Test that all backends can be instantiated
        for name, backend_class in BACKEND_REGISTRY.items():
            backend = backend_class()
            self.assertIsNotNone(backend)
            self.assertTrue(hasattr(backend, 'is_available'))
            self.assertTrue(hasattr(backend, 'create_claim'))
            self.assertTrue(hasattr(backend, 'get_claim'))
            self.assertTrue(hasattr(backend, 'search_claims'))
            self.assertTrue(hasattr(backend, 'analyze_claim'))


class TestModularCLIIntegration(unittest.TestCase):
    """Test integration between modular CLI components."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('cli.modular_cli.get_backend')
    def test_backend_selection(self, mock_get_backend):
        """Test backend selection in modular CLI."""
        from cli.modular_cli import get_backend
        
        # Test valid backend
        mock_backend = Mock()
        mock_backend.is_available.return_value = True
        mock_get_backend.return_value = mock_backend
        
        with patch.dict(BACKEND_REGISTRY, {"auto": Mock(return_value=mock_backend)}):
            result = get_backend("auto")
            self.assertIsNotNone(result)
        
        # Test invalid backend
        with self.assertRaises(SystemExit):
            get_backend("invalid_backend")
    
    def test_command_consistency(self):
        """Test that all backends have consistent command interfaces."""
        required_methods = [
            'create_claim', 'get_claim', 'search_claims', 
            'analyze_claim', 'is_available', 'get_backend_info'
        ]
        
        for name, backend_class in BACKEND_REGISTRY.items():
            backend = backend_class()
            for method in required_methods:
                self.assertTrue(hasattr(backend, method), 
                              f"Backend {name} missing method {method}")


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestBaseCLI,
        TestLocalBackend,
        TestCloudBackend,
        TestHybridBackend,
        TestAutoBackend,
        TestBackendRegistry,
        TestModularCLIIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)