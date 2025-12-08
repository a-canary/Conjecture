"""
Tests for EndPoint App Provider Management Endpoints
Validates provider status, metrics, management, testing, and health endpoints
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

from src.endpoint_app import app
from src.processing.enhanced_llm_router import RoutingStrategy


class TestProviderStatusEndpoint:
    """Test provider status endpoint"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @patch('src.endpoint_app.get_enhanced_llm_router')
    def test_get_providers_status_success(self, mock_get_router, client):
        """Test successful provider status retrieval"""
        # Mock router
        mock_router = Mock()
        mock_router.get_provider_status.return_value = {
            "total_providers": 3,
            "enabled_providers": 2,
            "healthy_providers": 2,
            "failed_providers": ["provider3"],
            "routing_strategy": "priority",
            "providers": {
                "provider1": {
                    "enabled": True,
                    "priority": 1,
                    "status": "healthy",
                    "current_load": 1,
                    "success_rate": 0.95
                },
                "provider2": {
                    "enabled": True,
                    "priority": 2,
                    "status": "healthy",
                    "current_load": 0,
                    "success_rate": 0.88
                },
                "provider3": {
                    "enabled": False,
                    "priority": 3,
                    "status": "unhealthy",
                    "current_load": 0,
                    "success_rate": 0.0
                }
            }
        }
        mock_get_router.return_value = mock_router
        
        # Make request
        response = client.get("/providers/status")
        
        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "data" in data
        assert data["data"]["total_providers"] == 3
        assert data["data"]["healthy_providers"] == 2
        assert "provider3" in data["data"]["failed_providers"]
        assert data["message"] == "Provider status retrieved successfully"
    
    @patch('src.endpoint_app.get_enhanced_llm_router')
    def test_get_providers_status_error(self, mock_get_router, client):
        """Test provider status endpoint with error"""
        # Mock router to raise exception
        mock_router = Mock()
        mock_router.get_provider_status.side_effect = Exception("Router error")
        mock_get_router.return_value = mock_router
        
        # Make request
        response = client.get("/providers/status")
        
        # Check error response
        assert response.status_code == 500


class TestProviderMetricsEndpoint:
    """Test provider metrics endpoint"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @patch('src.endpoint_app.get_enhanced_llm_router')
    def test_get_providers_metrics_success(self, mock_get_router, client):
        """Test successful provider metrics retrieval"""
        # Mock router
        mock_router = Mock()
        mock_router.get_provider_metrics.return_value = {
            "provider1": {
                "provider_name": "provider1",
                "total_requests": 100,
                "successful_requests": 95,
                "failed_requests": 5,
                "success_rate": 0.95,
                "average_response_time": 1.2,
                "current_load": 2,
                "enabled": True,
                "priority": 1
            },
            "provider2": {
                "provider_name": "provider2",
                "total_requests": 50,
                "successful_requests": 44,
                "failed_requests": 6,
                "success_rate": 0.88,
                "average_response_time": 2.1,
                "current_load": 1,
                "enabled": True,
                "priority": 2
            }
        }
        mock_get_router.return_value = mock_router
        
        # Make request
        response = client.get("/providers/metrics")
        
        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "data" in data
        assert "provider1" in data["data"]
        assert "provider2" in data["data"]
        assert data["data"]["provider1"]["total_requests"] == 100
        assert data["data"]["provider1"]["success_rate"] == 0.95
        assert data["message"] == "Provider metrics retrieved successfully"


class TestProviderManagementEndpoint:
    """Test provider management endpoint"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @patch('src.endpoint_app.get_enhanced_llm_router')
    def test_enable_provider_success(self, mock_get_router, client):
        """Test enabling a provider"""
        # Mock router
        mock_router = Mock()
        mock_router.enable_provider.return_value = True
        mock_get_router.return_value = mock_router
        
        # Make request
        request_data = {
            "action": "enable",
            "provider_name": "test_provider"
        }
        response = client.post("/providers/manage", json=request_data)
        
        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["data"]["action"] == "enable"
        assert data["data"]["provider"] == "test_provider"
        assert "enabled" in data["message"]
        
        # Check router call
        mock_router.enable_provider.assert_called_once_with("test_provider")
    
    @patch('src.endpoint_app.get_enhanced_llm_router')
    def test_disable_provider_success(self, mock_get_router, client):
        """Test disabling a provider"""
        # Mock router
        mock_router = Mock()
        mock_router.disable_provider.return_value = True
        mock_get_router.return_value = mock_router
        
        # Make request
        request_data = {
            "action": "disable",
            "provider_name": "test_provider"
        }
        response = client.post("/providers/manage", json=request_data)
        
        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["data"]["action"] == "disable"
        assert data["data"]["provider"] == "test_provider"
        assert "disabled" in data["message"]
        
        # Check router call
        mock_router.disable_provider.assert_called_once_with("test_provider")
    
    @patch('src.endpoint_app.get_enhanced_llm_router')
    def test_reset_metrics_all_providers(self, mock_get_router, client):
        """Test resetting metrics for all providers"""
        # Mock router
        mock_router = Mock()
        mock_get_router.return_value = mock_router
        
        # Make request
        request_data = {
            "action": "reset_metrics"
        }
        response = client.post("/providers/manage", json=request_data)
        
        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "all providers" in data["message"]
        
        # Check router call
        mock_router.reset_provider_metrics.assert_called_once_with(None)
    
    @patch('src.endpoint_app.get_enhanced_llm_router')
    def test_reset_metrics_specific_provider(self, mock_get_router, client):
        """Test resetting metrics for specific provider"""
        # Mock router
        mock_router = Mock()
        mock_get_router.return_value = mock_router
        
        # Make request
        request_data = {
            "action": "reset_metrics",
            "provider_name": "test_provider"
        }
        response = client.post("/providers/manage", json=request_data)
        
        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "test_provider" in data["message"]
        
        # Check router call
        mock_router.reset_provider_metrics.assert_called_once_with("test_provider")
    
    @patch('src.endpoint_app.get_enhanced_llm_router')
    def test_set_routing_strategy_success(self, mock_get_router, client):
        """Test setting routing strategy"""
        # Mock router
        mock_router = Mock()
        mock_get_router.return_value = mock_router
        
        # Make request
        request_data = {
            "action": "set_strategy",
            "routing_strategy": "round_robin"
        }
        response = client.post("/providers/manage", json=request_data)
        
        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["data"]["action"] == "set_strategy"
        assert "round_robin" in data["message"]
        
        # Check router call
        mock_router.set_routing_strategy.assert_called_once_with(RoutingStrategy.ROUND_ROBIN)
    
    def test_enable_provider_missing_name(self, client):
        """Test enable provider without name"""
        # Make request without provider_name
        request_data = {
            "action": "enable"
        }
        response = client.post("/providers/manage", json=request_data)
        
        # Check error response
        assert response.status_code == 400
    
    def test_disable_provider_missing_name(self, client):
        """Test disable provider without name"""
        # Make request without provider_name
        request_data = {
            "action": "disable"
        }
        response = client.post("/providers/manage", json=request_data)
        
        # Check error response
        assert response.status_code == 400
    
    def test_set_strategy_missing_strategy(self, client):
        """Test set strategy without routing_strategy"""
        # Make request without routing_strategy
        request_data = {
            "action": "set_strategy"
        }
        response = client.post("/providers/manage", json=request_data)
        
        # Check error response
        assert response.status_code == 400
    
    def test_invalid_action(self, client):
        """Test invalid management action"""
        # Make request with invalid action
        request_data = {
            "action": "invalid_action"
        }
        response = client.post("/providers/manage", json=request_data)
        
        # Check error response
        assert response.status_code == 400
    
    def test_invalid_routing_strategy(self, client):
        """Test invalid routing strategy"""
        # Mock router to raise ValueError for invalid strategy
        with patch('src.endpoint_app.get_enhanced_llm_router') as mock_get_router:
            mock_router = Mock()
            mock_router.set_routing_strategy.side_effect = ValueError("Invalid strategy")
            mock_get_router.return_value = mock_router
            
            # Make request with invalid strategy
            request_data = {
                "action": "set_strategy",
                "routing_strategy": "invalid_strategy"
            }
            response = client.post("/providers/manage", json=request_data)
            
            # Check error response
            assert response.status_code == 400


class TestProviderTestEndpoint:
    """Test provider testing endpoint"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @patch('src.endpoint_app.get_enhanced_llm_router')
    def test_test_all_providers_success(self, mock_get_router, client):
        """Test all providers successfully"""
        # Mock router
        mock_router = Mock()
        mock_router.get_provider_status.return_value = {
            "total_providers": 2,
            "healthy_providers": 2
        }
        
        # Mock successful test results
        async def mock_generate_response(prompt, preferred_provider=None):
            return Mock(
                success=True,
                processing_time=1.0,
                tokens_used=50,
                model_used="test-model",
                errors=[],
                content=f"Response from {preferred_provider or 'default'}"
            )
        
        mock_router.generate_response.side_effect = mock_generate_response
        mock_get_router.return_value = mock_router
        
        # Make request
        request_data = {
            "prompt": "Hello, please respond with a brief greeting."
        }
        response = client.post("/providers/test", json=request_data)
        
        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "test_results" in data["data"]
        assert "initial_status" in data["data"]
        assert "final_status" in data["data"]
        assert data["data"]["test_prompt"] == "Hello, please respond with a brief greeting."
        
        # Check that generate_response was called for each provider
        assert mock_router.generate_response.call_count >= 2
    
    @patch('src.endpoint_app.get_enhanced_llm_router')
    def test_test_specific_provider_success(self, mock_get_router, client):
        """Test specific provider successfully"""
        # Mock router
        mock_router = Mock()
        
        # Mock successful test result
        mock_result = Mock(
            success=True,
            processing_time=0.8,
            tokens_used=30,
            model_used="specific-model",
            errors=[],
            content="Hello! This is a test response."
        )
        
        async def mock_generate_response(prompt, preferred_provider=None):
            if preferred_provider == "test_provider":
                return mock_result
            raise Exception("Not this provider")
        
        mock_router.generate_response.side_effect = mock_generate_response
        mock_get_router.return_value = mock_router
        
        # Make request
        request_data = {
            "prompt": "Hello, please respond with a brief greeting.",
            "provider_name": "test_provider"
        }
        response = client.post("/providers/test", json=request_data)
        
        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "test_provider" in data["data"]["test_results"]
        
        test_result = data["data"]["test_results"]["test_provider"]
        assert test_result["success"] == True
        assert test_result["response_time"] == 0.8
        assert test_result["tokens_used"] == 30
        assert test_result["model_used"] == "specific-model"
        assert "Hello!" in test_result["content_preview"]
    
    @patch('src.endpoint_app.get_enhanced_llm_router')
    def test_test_provider_failure(self, mock_get_router, client):
        """Test provider with failure"""
        # Mock router
        mock_router = Mock()
        
        # Mock failed test result
        async def mock_generate_response(prompt, preferred_provider=None):
            raise Exception("Provider failed")
        
        mock_router.generate_response.side_effect = mock_generate_response
        mock_get_router.return_value = mock_router
        
        # Make request
        request_data = {
            "prompt": "Hello, please respond with a brief greeting.",
            "provider_name": "failing_provider"
        }
        response = client.post("/providers/test", json=request_data)
        
        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "failing_provider" in data["data"]["test_results"]
        
        test_result = data["data"]["test_results"]["failing_provider"]
        assert test_result["success"] == False
        assert "error" in test_result


class TestProviderHealthEndpoint:
    """Test provider health check endpoint"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @patch('src.endpoint_app.get_enhanced_llm_router')
    def test_health_check_all_healthy(self, mock_get_router, client):
        """Test health check with all providers healthy"""
        # Mock router
        mock_router = Mock()
        mock_router._perform_health_checks = AsyncMock()
        mock_router.get_provider_status.return_value = {
            "total_providers": 3,
            "healthy_providers": 3
        }
        mock_get_router.return_value = mock_router
        
        # Make request
        response = client.get("/providers/health")
        
        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["data"]["overall_health"] == "healthy"
        assert data["data"]["health_score"] == 1.0
        assert data["data"]["healthy_providers"] == 3
        assert data["data"]["total_providers"] == 3
        assert data["data"]["health_percentage"] == 100.0
        assert "healthy" in data["message"]
        
        # Check that health check was performed
        mock_router._perform_health_checks.assert_called_once()
    
    @patch('src.endpoint_app.get_enhanced_llm_router')
    def test_health_check_degraded(self, mock_get_router, client):
        """Test health check with degraded providers"""
        # Mock router
        mock_router = Mock()
        mock_router._perform_health_checks = AsyncMock()
        mock_router.get_provider_status.return_value = {
            "total_providers": 4,
            "healthy_providers": 2
        }
        mock_get_router.return_value = mock_router
        
        # Make request
        response = client.get("/providers/health")
        
        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["data"]["overall_health"] == "degraded"
        assert data["data"]["health_score"] == 0.5
        assert data["data"]["healthy_providers"] == 2
        assert data["data"]["total_providers"] == 4
        assert data["data"]["health_percentage"] == 50.0
        assert "degraded" in data["message"]
    
    @patch('src.endpoint_app.get_enhanced_llm_router')
    def test_health_check_unhealthy(self, mock_get_router, client):
        """Test health check with no healthy providers"""
        # Mock router
        mock_router = Mock()
        mock_router._perform_health_checks = AsyncMock()
        mock_router.get_provider_status.return_value = {
            "total_providers": 2,
            "healthy_providers": 0
        }
        mock_get_router.return_value = mock_router
        
        # Make request
        response = client.get("/providers/health")
        
        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["data"]["overall_health"] == "unhealthy"
        assert data["data"]["health_score"] == 0.0
        assert data["data"]["healthy_providers"] == 0
        assert data["data"]["total_providers"] == 2
        assert data["data"]["health_percentage"] == 0.0
        assert "unhealthy" in data["message"]
    
    @patch('src.endpoint_app.get_enhanced_llm_router')
    def test_health_check_error(self, mock_get_router, client):
        """Test health check with error"""
        # Mock router to raise exception
        mock_router = Mock()
        mock_router._perform_health_checks.side_effect = Exception("Health check failed")
        mock_get_router.return_value = mock_router
        
        # Make request
        response = client.get("/providers/health")
        
        # Check error response
        assert response.status_code == 500


class TestIntegrationWithExistingEndpoints:
    """Test integration with existing EndPoint App endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_root_endpoint_includes_providers(self, client):
        """Test that root endpoint includes provider management endpoints"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "providers" in data["data"]["endpoints"]
        
        providers_endpoints = data["data"]["endpoints"]["providers"]
        assert "status" in providers_endpoints
        assert "metrics" in providers_endpoints
        assert "manage" in providers_endpoints
        assert "test" in providers_endpoints
        assert "health" in providers_endpoints
        
        assert providers_endpoints["status"] == "/providers/status"
        assert providers_endpoints["metrics"] == "/providers/metrics"
        assert providers_endpoints["manage"] == "/providers/manage"
        assert providers_endpoints["test"] == "/providers/test"
        assert providers_endpoints["health"] == "/providers/health"


class TestAsyncOperations:
    """Test async operations and error handling"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @patch('src.endpoint_app.get_enhanced_llm_router')
    @pytest.mark.asyncio
    async def test_async_provider_operations(self, mock_get_router, client):
        """Test async provider operations"""
        # Mock router with async methods
        mock_router = Mock()
        mock_router._perform_health_checks = AsyncMock()
        mock_router.generate_response = AsyncMock(
            return_value=Mock(
                success=True,
                processing_time=1.0,
                tokens_used=50,
                content="Async response"
            )
        )
        mock_get_router.return_value = mock_router
        
        # Test async health check
        response = client.get("/providers/health")
        assert response.status_code == 200
        
        # Test async provider test
        request_data = {"prompt": "Test async"}
        response = client.post("/providers/test", json=request_data)
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])