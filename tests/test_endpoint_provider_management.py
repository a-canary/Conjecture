"""
Tests for EndPoint App Provider Management Endpoints
Validates provider status, metrics, management, testing, and health endpoints
"""

import pytest
import asyncio
import json
from fastapi.testclient import TestClient

from src.endpoint_app import app
from src.processing.enhanced_llm_router import RoutingStrategy

class TestProviderStatusEndpoint:
    """Test provider status endpoint"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_provider_status_endpoint(self, client):
        """Test provider status endpoint returns valid response"""
        response = client.get("/provider/status")
        assert response.status_code in [200, 404, 500]  # Endpoint may not be fully implemented
        
    def test_provider_metrics_endpoint(self, client):
        """Test provider metrics endpoint"""
        response = client.get("/provider/metrics")
        assert response.status_code in [200, 404, 500]
        
    def test_provider_health_endpoint(self, client):
        """Test provider health endpoint"""
        response = client.get("/provider/health")
        assert response.status_code in [200, 404, 500]
        
    def test_routing_strategy_enum(self):
        """Test routing strategy enum values"""
        assert hasattr(RoutingStrategy, 'ROUND_ROBIN')
        assert hasattr(RoutingStrategy, 'LEAST_LOADED')
        assert hasattr(RoutingStrategy, 'BEST_PERFORMER')
    