"""
Comprehensive tests for Enhanced LLM Router
Tests provider routing, health monitoring, failover, and management endpoints
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from src.processing.enhanced_llm_router import (
    EnhancedLLMRouter,
    ProviderConfig,
    ProviderMetrics,
    ProviderStatus,
    RoutingStrategy
)
from src.processing.llm.common import GenerationConfig, LLMProcessingResult
from src.core.models import Claim, ClaimType, ClaimState


class TestProviderConfig:
    """Test provider configuration validation and management"""
    
    def test_provider_config_creation(self):
        """Test creating valid provider configuration"""
        config = ProviderConfig(
            name="test_provider",
            url="https://api.test.com/v1",
            api_key="test_key",
            model="test-model",
            priority=1,
            max_retries=3,
            timeout=60
        )
        
        assert config.name == "test_provider"
        assert config.url == "https://api.test.com/v1"
        assert config.api_key == "test_key"
        assert config.model == "test-model"
        assert config.priority == 1
        assert config.max_retries == 3
        assert config.timeout == 60
        assert config.enabled == True
        assert config.routing_strategy == RoutingStrategy.PRIORITY
    
    def test_provider_config_to_dict(self):
        """Test converting provider config to dictionary"""
        config = ProviderConfig(
            name="test_provider",
            url="https://api.test.com/v1",
            api_key="test_key",
            model="test-model",
            priority=1
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["name"] == "test_provider"
        assert config_dict["url"] == "https://api.test.com/v1"
        assert config_dict["api_key"] == "test_key"
        assert config_dict["model"] == "test-model"
        assert config_dict["priority"] == 1
        assert config_dict["routing_strategy"] == "priority"
    
    def test_local_provider_detection(self):
        """Test local provider detection"""
        local_config = ProviderConfig(
            name="local_provider",
            url="http://localhost:1234",
            model="local-model"
        )
        
        # Test with router initialization to set is_local flag
        router = EnhancedLLMRouter([local_config.to_dict()])
        config = router.provider_configs["local_provider"]
        
        assert config.is_local == True
        
        # Test remote provider
        remote_config = ProviderConfig(
            name="remote_provider",
            url="https://api.openai.com/v1",
            model="gpt-3.5-turbo"
        )
        
        router = EnhancedLLMRouter([remote_config.to_dict()])
        config = router.provider_configs["remote_provider"]
        
        assert config.is_local == False


class TestProviderMetrics:
    """Test provider metrics tracking and calculations"""
    
    def test_metrics_initialization(self):
        """Test metrics initialization"""
        metrics = ProviderMetrics(provider_name="test_provider")
        
        assert metrics.provider_name == "test_provider"
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.total_response_time == 0.0
        assert metrics.consecutive_failures == 0
        assert metrics.success_rate == 0.0
        assert metrics.average_response_time == 0.0
    
    def test_metrics_success_update(self):
        """Test updating metrics after successful request"""
        metrics = ProviderMetrics(provider_name="test_provider")
        
        # Update with success
        metrics.update_success(1.5)
        
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 0
        assert metrics.total_response_time == 1.5
        assert metrics.consecutive_failures == 0
        assert metrics.success_rate == 1.0
        assert metrics.average_response_time == 1.5
        assert metrics.last_success_time is not None
    
    def test_metrics_failure_update(self):
        """Test updating metrics after failed request"""
        metrics = ProviderMetrics(provider_name="test_provider")
        
        # Update with failure
        metrics.update_failure()
        
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 1
        assert metrics.consecutive_failures == 1
        assert metrics.success_rate == 0.0
        assert metrics.last_failure_time is not None
    
    def test_metrics_mixed_updates(self):
        """Test metrics with mixed success and failure updates"""
        metrics = ProviderMetrics(provider_name="test_provider")
        
        # Add some successes and failures
        metrics.update_success(1.0)
        metrics.update_success(2.0)
        metrics.update_failure()
        metrics.update_success(1.5)
        
        assert metrics.total_requests == 4
        assert metrics.successful_requests == 3
        assert metrics.failed_requests == 1
        assert metrics.total_response_time == 4.5
        assert metrics.consecutive_failures == 0  # Reset after success
        assert metrics.success_rate == 0.75  # 3/4
        assert metrics.average_response_time == 1.125  # 4.5/4


class TestRoutingStrategies:
    """Test different routing strategies"""
    
    @pytest.fixture
    async def mock_router(self):
        """Create a mock router with multiple providers"""
        providers = [
            {
                "name": "high_priority",
                "url": "https://high.example.com/v1",
                "model": "model1",
                "priority": 1
            },
            {
                "name": "medium_priority",
                "url": "https://medium.example.com/v1",
                "model": "model2",
                "priority": 5
            },
            {
                "name": "low_priority",
                "url": "https://low.example.com/v1",
                "model": "model3",
                "priority": 10
            }
        ]
        
        with patch('src.processing.enhanced_llm_router.create_openai_compatible_processor'):
            router = EnhancedLLMRouter(providers)
            await router.start_health_monitoring_async()
            yield router
            if router.health_check_task:
                router.health_check_task.cancel()
    
    @pytest.mark.asyncio
    async def test_priority_routing(self, mock_router):
        """Test priority-based routing"""
        mock_router.set_routing_strategy(RoutingStrategy.PRIORITY)
        
        # Should select highest priority (lowest number)
        selected = mock_router._select_provider()
        assert selected == "high_priority"
    
    @pytest.mark.asyncio
    async def test_round_robin_routing(self, mock_router):
        """Test round-robin routing"""
        mock_router.set_routing_strategy(RoutingStrategy.ROUND_ROBIN)
        
        # Should cycle through providers
        first = mock_router._select_provider()
        second = mock_router._select_provider()
        third = mock_router._select_provider()
        fourth = mock_router._select_provider()  # Should wrap around
        
        assert first == "high_priority"
        assert second == "medium_priority"
        assert third == "low_priority"
        assert fourth == "high_priority"  # Wrapped around
    
    @pytest.mark.asyncio
    async def test_load_balanced_routing(self, mock_router):
        """Test load-balanced routing"""
        mock_router.set_routing_strategy(RoutingStrategy.LOAD_BALANCED)
        
        # Simulate different loads
        mock_router.provider_metrics["high_priority"].current_load = 5
        mock_router.provider_metrics["medium_priority"].current_load = 2
        mock_router.provider_metrics["low_priority"].current_load = 8
        
        # Should select least loaded
        selected = mock_router._select_provider()
        assert selected == "medium_priority"
    
    @pytest.mark.asyncio
    async def test_fastest_response_routing(self, mock_router):
        """Test fastest-response routing"""
        mock_router.set_routing_strategy(RoutingStrategy.FASTEST_RESPONSE)
        
        # Set different response times
        mock_router.provider_metrics["high_priority"].update_success(2.0)
        mock_router.provider_metrics["medium_priority"].update_success(1.0)
        mock_router.provider_metrics["low_priority"].update_success(3.0)
        
        # Should select fastest
        selected = mock_router._select_provider()
        assert selected == "medium_priority"


class TestHealthMonitoring:
    """Test health monitoring and provider status tracking"""
    
    @pytest.fixture
    def mock_router_with_health(self):
        """Create router with mocked health checks"""
        providers = [
            {
                "name": "healthy_provider",
                "url": "https://healthy.example.com/v1",
                "model": "model1",
                "priority": 1
            },
            {
                "name": "unhealthy_provider",
                "url": "https://unhealthy.example.com/v1",
                "model": "model2",
                "priority": 2
            }
        ]
        
        with patch('src.processing.enhanced_llm_router.create_openai_compatible_processor'):
            router = EnhancedLLMRouter(providers)
            return router
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_router_with_health):
        """Test successful health check"""
        # Mock successful health check
        mock_processor = Mock()
        mock_processor.health_check.return_value = {
            "status": "healthy",
            "model": "test-model",
            "provider": "healthy_provider"
        }
        
        mock_router_with_health.providers["healthy_provider"] = mock_processor
        
        # Perform health check
        result = await mock_router_with_health._async_health_check(
            mock_processor, 10
        )
        
        assert result["status"] == "healthy"
        assert result["model"] == "test-model"
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, mock_router_with_health):
        """Test failed health check"""
        # Mock failed health check
        mock_processor = Mock()
        mock_processor.health_check.side_effect = Exception("Connection failed")
        
        mock_router_with_health.providers["unhealthy_provider"] = mock_processor
        
        # Perform health check
        result = await mock_router_with_health._async_health_check(
            mock_processor, 10
        )
        
        assert result["status"] == "unhealthy"
        assert "Connection failed" in result["error"]
    
    @pytest.mark.asyncio
    async def test_health_monitoring_loop(self, mock_router_with_health):
        """Test health monitoring loop"""
        # Mock health checks
        for provider_name, processor in mock_router_with_health.providers.items():
            processor.health_check = Mock(return_value={"status": "healthy"})
        
        # Run one iteration of health monitoring
        await mock_router_with_health._perform_health_checks()
        
        # Check that health checks were called
        for processor in mock_router_with_health.providers.values():
            processor.health_check.assert_called_once()


class TestFailoverLogic:
    """Test failover logic and provider recovery"""
    
    @pytest.fixture
    def mock_router_failover(self):
        """Create router for failover testing"""
        providers = [
            {
                "name": "primary",
                "url": "https://primary.example.com/v1",
                "model": "model1",
                "priority": 1
            },
            {
                "name": "secondary",
                "url": "https://secondary.example.com/v1",
                "model": "model2",
                "priority": 2
            },
            {
                "name": "tertiary",
                "url": "https://tertiary.example.com/v1",
                "model": "model3",
                "priority": 3
            }
        ]
        
        with patch('src.processing.enhanced_llm_router.create_openai_compatible_processor'):
            router = EnhancedLLMRouter(providers)
            return router
    
    @pytest.mark.asyncio
    async def test_provider_failover(self, mock_router_failover):
        """Test automatic failover when provider fails"""
        # Mock processors
        primary_processor = Mock()
        primary_processor.generate_response.side_effect = Exception("Primary failed")
        
        secondary_processor = Mock()
        secondary_processor.generate_response.return_value = LLMProcessingResult(
            success=True,
            content="Response from secondary",
            model_used="model2",
            tokens_used=50,
            processing_time=1.0
        )
        
        mock_router_failover.providers["primary"] = primary_processor
        mock_router_failover.providers["secondary"] = secondary_processor
        
        # Generate response - should failover to secondary
        result = await mock_router_failover.generate_response("test prompt")
        
        # Should have result from secondary provider
        assert result.success == True
        assert result.content == "Response from secondary"
        assert result.model_used == "model2"
        
        # Primary should be in failed providers
        assert "primary" in mock_router_failover.failed_providers
    
    @pytest.mark.asyncio
    async def test_preferred_provider_fallback(self, mock_router_failover):
        """Test fallback when preferred provider fails"""
        # Mock processors
        preferred_processor = Mock()
        preferred_processor.generate_response.side_effect = Exception("Preferred failed")
        
        fallback_processor = Mock()
        fallback_processor.generate_response.return_value = LLMProcessingResult(
            success=True,
            content="Response from fallback",
            model_used="model2",
            tokens_used=50,
            processing_time=1.0
        )
        
        mock_router_failover.providers["primary"] = preferred_processor
        mock_router_failover.providers["secondary"] = fallback_processor
        
        # Request specific provider that fails
        result = await mock_router_failover.generate_response(
            "test prompt",
            preferred_provider="primary"
        )
        
        # Should fallback to secondary
        assert result.success == True
        assert result.content == "Response from fallback"
    
    @pytest.mark.asyncio
    async def test_all_providers_failed(self, mock_router_failover):
        """Test behavior when all providers fail"""
        # Mock all processors to fail
        for processor in mock_router_failover.providers.values():
            processor.generate_response.side_effect = Exception("All failed")
        
        # Should raise exception when all fail
        with pytest.raises(RuntimeError, match="All available providers failed"):
            await mock_router_failover.generate_response("test prompt")


class TestProviderManagement:
    """Test provider management operations"""
    
    @pytest.fixture
    def mock_router_management(self):
        """Create router for management testing"""
        providers = [
            {
                "name": "test_provider1",
                "url": "https://test1.example.com/v1",
                "model": "model1",
                "priority": 1
            },
            {
                "name": "test_provider2",
                "url": "https://test2.example.com/v1",
                "model": "model2",
                "priority": 2
            }
        ]
        
        with patch('src.processing.enhanced_llm_router.create_openai_compatible_processor'):
            router = EnhancedLLMRouter(providers)
            return router
    
    def test_enable_provider(self, mock_router_management):
        """Test enabling a provider"""
        # Disable provider first
        mock_router_management.disable_provider("test_provider1")
        assert "test_provider1" in mock_router_management.failed_providers
        assert mock_router_management.provider_configs["test_provider1"].enabled == False
        
        # Enable provider
        success = mock_router_management.enable_provider("test_provider1")
        
        assert success == True
        assert "test_provider1" not in mock_router_management.failed_providers
        assert mock_router_management.provider_configs["test_provider1"].enabled == True
    
    def test_disable_provider(self, mock_router_management):
        """Test disabling a provider"""
        success = mock_router_management.disable_provider("test_provider1")
        
        assert success == True
        assert "test_provider1" in mock_router_management.failed_providers
        assert mock_router_management.provider_configs["test_provider1"].enabled == False
    
    def test_enable_nonexistent_provider(self, mock_router_management):
        """Test enabling non-existent provider"""
        success = mock_router_management.enable_provider("nonexistent")
        assert success == False
    
    def test_reset_provider_metrics(self, mock_router_management):
        """Test resetting provider metrics"""
        # Add some metrics
        mock_router_management.provider_metrics["test_provider1"].update_success(1.0)
        mock_router_management.provider_metrics["test_provider1"].update_failure()
        
        assert mock_router_management.provider_metrics["test_provider1"].total_requests == 2
        
        # Reset metrics
        mock_router_management.reset_provider_metrics("test_provider1")
        
        metrics = mock_router_management.provider_metrics["test_provider1"]
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.consecutive_failures == 0
    
    def test_reset_all_metrics(self, mock_router_management):
        """Test resetting all provider metrics"""
        # Add metrics to all providers
        for metrics in mock_router_management.provider_metrics.values():
            metrics.update_success(1.0)
        
        # Reset all
        mock_router_management.reset_provider_metrics()
        
        # Check all are reset
        for metrics in mock_router_management.provider_metrics.values():
            assert metrics.total_requests == 0
            assert metrics.successful_requests == 0
            assert metrics.failed_requests == 0
    
    def test_set_routing_strategy(self, mock_router_management):
        """Test setting routing strategy"""
        # Test different strategies
        for strategy in RoutingStrategy:
            mock_router_management.set_routing_strategy(strategy)
            assert mock_router_management.routing_strategy == strategy


class TestIntegrationWithClaims:
    """Test integration with claim processing"""
    
    @pytest.fixture
    def mock_router_claims(self):
        """Create router for claim processing testing"""
        providers = [
            {
                "name": "claim_provider",
                "url": "https://claims.example.com/v1",
                "model": "claim-model",
                "priority": 1
            }
        ]
        
        with patch('src.processing.enhanced_llm_router.create_openai_compatible_processor'):
            router = EnhancedLLMRouter(providers)
            return router
    
    @pytest.mark.asyncio
    async def test_process_claims_success(self, mock_router_claims):
        """Test successful claim processing"""
        # Mock processor
        mock_processor = Mock()
        mock_processor.process_claims.return_value = LLMProcessingResult(
            success=True,
            processed_claims=[
                Claim(
                    id="c1234567",
                    content="Test claim",
                    claim_type=ClaimType.FACT,
                    state=ClaimState.VALIDATED,
                    confidence=0.9
                )
            ],
            model_used="claim-model",
            tokens_used=100,
            processing_time=2.0
        )
        
        mock_router_claims.providers["claim_provider"] = mock_processor
        
        # Create test claims
        claims = [
            Claim(
                id="c1234567",
                content="Test claim",
                claim_type=ClaimType.FACT,
                state=ClaimState.EXPLORE,
                confidence=0.5
            )
        ]
        
        # Process claims
        result = await mock_router_claims.process_claims(claims, "analyze")
        
        assert result.success == True
        assert len(result.processed_claims) == 1
        assert result.processed_claims[0].state == ClaimState.VALIDATED
        assert result.processed_claims[0].confidence == 0.9
        assert result.model_used == "claim-model"
    
    @pytest.mark.asyncio
    async def test_process_claims_with_config(self, mock_router_claims):
        """Test claim processing with custom config"""
        # Mock processor
        mock_processor = Mock()
        mock_processor.process_claims.return_value = LLMProcessingResult(
            success=True,
            processed_claims=[],
            model_used="claim-model",
            tokens_used=50,
            processing_time=1.0
        )
        
        mock_router_claims.providers["claim_provider"] = mock_processor
        
        # Create custom config
        custom_config = GenerationConfig(
            temperature=0.5,
            max_tokens=1000,
            top_p=0.9
        )
        
        # Process with custom config
        claims = [Claim(id="test", content="test", claim_type=ClaimType.FACT, state=ClaimState.EXPLORE, confidence=0.5)]
        await mock_router_claims.process_claims(claims, "analyze", config=custom_config)
        
        # Check that processor was called with merged config
        mock_processor.process_claims.assert_called_once()
        call_args = mock_processor.process_claims.call_args
        assert call_args[0][0] == claims  # claims
        assert call_args[0][1] == "analyze"  # task
        
        # Check config merging
        config_arg = call_args[0][2]  # config
        assert config_arg.temperature == 0.5
        assert config_arg.max_tokens == 1000
        assert config_arg.top_p == 0.9


class TestStatusAndMetrics:
    """Test status reporting and metrics collection"""
    
    @pytest.fixture
    def mock_router_status(self):
        """Create router for status testing"""
        providers = [
            {
                "name": "status_provider1",
                "url": "https://status1.example.com/v1",
                "model": "model1",
                "priority": 1,
                "max_concurrent_requests": 5
            },
            {
                "name": "status_provider2",
                "url": "https://status2.example.com/v1",
                "model": "model2",
                "priority": 2,
                "max_concurrent_requests": 10
            }
        ]
        
        with patch('src.processing.enhanced_llm_router.create_openai_compatible_processor'):
            router = EnhancedLLMRouter(providers)
            return router
    
    def test_get_provider_status(self, mock_router_status):
        """Test getting comprehensive provider status"""
        # Add some metrics
        mock_router_status.provider_metrics["status_provider1"].update_success(1.0)
        mock_router_status.provider_metrics["status_provider1"].current_load = 3
        mock_router_status.provider_metrics["status_provider2"].update_failure()
        
        # Get status
        status = mock_router_status.get_provider_status()
        
        assert status["total_providers"] == 2
        assert status["enabled_providers"] == 2
        assert status["routing_strategy"] == "priority"
        assert "providers" in status
        
        # Check individual provider status
        provider1_status = status["providers"]["status_provider1"]
        assert provider1_status["enabled"] == True
        assert provider1_status["priority"] == 1
        assert provider1_status["current_load"] == 3
        assert provider1_status["max_concurrent_requests"] == 5
        assert provider1_status["total_requests"] == 1
        assert provider1_status["success_rate"] == 1.0
        
        provider2_status = status["providers"]["status_provider2"]
        assert provider2_status["total_requests"] == 1
        assert provider2_status["success_rate"] == 0.0
    
    def test_get_provider_metrics(self, mock_router_status):
        """Test getting detailed provider metrics"""
        # Add some metrics
        mock_router_status.provider_metrics["status_provider1"].update_success(1.5)
        mock_router_status.provider_metrics["status_provider1"].update_success(2.5)
        mock_router_status.provider_metrics["status_provider2"].update_failure()
        
        # Get metrics
        metrics = mock_router_status.get_provider_metrics()
        
        assert "status_provider1" in metrics
        assert "status_provider2" in metrics
        
        # Check detailed metrics
        provider1_metrics = metrics["status_provider1"]
        assert provider1_metrics["total_requests"] == 2
        assert provider1_metrics["successful_requests"] == 2
        assert provider1_metrics["failed_requests"] == 0
        assert provider1_metrics["success_rate"] == 1.0
        assert provider1_metrics["average_response_time"] == 2.0  # (1.5 + 2.5) / 2
        assert provider1_metrics["enabled"] == True
        assert provider1_metrics["priority"] == 1
        assert provider1_metrics["model"] == "model1"
        
        provider2_metrics = metrics["status_provider2"]
        assert provider2_metrics["total_requests"] == 1
        assert provider2_metrics["successful_requests"] == 0
        assert provider2_metrics["failed_requests"] == 1
        assert provider2_metrics["success_rate"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])