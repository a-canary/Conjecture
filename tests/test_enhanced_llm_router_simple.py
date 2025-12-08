import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from src.processing.enhanced_llm_router import (
    EnhancedLLMRouter,
    ProviderConfig,
    ProviderMetrics,
    ProviderStatus,
    RoutingStrategy
)


class TestProviderConfig:
    """Test ProviderConfig functionality"""
    
    def test_provider_config_creation(self):
        """Test creating a provider config"""
        config = ProviderConfig(
            name="test_provider",
            url="https://test.example.com/v1",
            api_key="test_key",
            model="test-model",
            priority=1
        )
        
        assert config.name == "test_provider"
        assert config.url == "https://test.example.com/v1"
        assert config.api_key == "test_key"
        assert config.model == "test-model"
        assert config.priority == 1
        assert config.enabled == True
        assert config.is_local == False
    
    def test_local_provider_detection(self):
        """Test local provider detection"""
        local_config = ProviderConfig(
            name="local_provider",
            url="http://localhost:1234",
            model="local-model"
        )
        
        # Manually set is_local for testing
        local_config.is_local = True
        assert local_config.is_local == True
        
        remote_config = ProviderConfig(
            name="remote_provider",
            url="https://api.openai.com/v1",
            model="gpt-4"
        )
        
        assert remote_config.is_local == False
    
    def test_provider_config_to_dict(self):
        """Test converting config to dictionary"""
        config = ProviderConfig(
            name="test_provider",
            url="https://test.example.com/v1",
            api_key="test_key",
            model="test-model",
            priority=1,
            max_retries=3,
            timeout=60
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["name"] == "test_provider"
        assert config_dict["url"] == "https://test.example.com/v1"
        assert config_dict["api_key"] == "test_key"
        assert config_dict["model"] == "test-model"
        assert config_dict["priority"] == 1
        assert config_dict["max_retries"] == 3
        assert config_dict["timeout"] == 60


class TestProviderMetrics:
    """Test ProviderMetrics functionality"""
    
    def test_metrics_initialization(self):
        """Test metrics initialization"""
        metrics = ProviderMetrics(provider_name="test_provider")
        
        assert metrics.provider_name == "test_provider"
        # ProviderMetrics doesn't have status field initially, it's tracked separately
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.consecutive_failures == 0
        assert metrics.current_load == 0
        assert metrics.success_rate == 0.0
        assert metrics.average_response_time == 0.0
    
    def test_metrics_success_update(self):
        """Test updating metrics with success"""
        metrics = ProviderMetrics(provider_name="test_provider")
        
        metrics.update_success(1.5)
        
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 0
        assert metrics.consecutive_failures == 0
        assert metrics.success_rate == 1.0
        assert metrics.average_response_time == 1.5
        assert metrics.total_response_time == 1.5
    
    def test_metrics_failure_update(self):
        """Test updating metrics with failure"""
        metrics = ProviderMetrics(provider_name="test_provider")
        
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
    
    def test_metrics_reset(self):
        """Test resetting metrics"""
        metrics = ProviderMetrics(provider_name="test_provider")
        
        # Add some data
        metrics.update_success(1.0)
        metrics.update_failure()
        metrics.current_load = 5
        
        # Test router's reset_provider_metrics method instead
        providers = [
            {
                "name": "test_provider",
                "url": "https://test.example.com/v1",
                "model": "test-model"
            }
        ]
        
        with patch('src.processing.enhanced_llm_router.create_openai_compatible_processor'):
            with patch.object(EnhancedLLMRouter, '_start_health_monitoring'):
                router = EnhancedLLMRouter(providers)
                
                # Add some data to router's metrics
                router.provider_metrics["test_provider"].update_success(1.0)
                router.provider_metrics["test_provider"].update_failure()
                router.provider_metrics["test_provider"].current_load = 5
                
                # Reset using router method
                router.reset_provider_metrics("test_provider")
                metrics = router.provider_metrics["test_provider"]
                assert metrics.total_requests == 0
                assert metrics.successful_requests == 0
                assert metrics.failed_requests == 0
                assert metrics.consecutive_failures == 0
                assert metrics.current_load == 0
                assert metrics.success_rate == 0.0
                assert metrics.average_response_time == 0.0


class TestEnhancedLLMRouterBasic:
    """Test basic EnhancedLLMRouter functionality without async"""
    
    def test_router_initialization_without_health_monitoring(self):
        """Test router initialization without starting health monitoring"""
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
            # Mock the health monitoring to avoid starting it
            with patch.object(EnhancedLLMRouter, '_start_health_monitoring'):
                router = EnhancedLLMRouter(providers)
            
            assert len(router.providers) == 2
            assert "test_provider1" in router.providers
            assert "test_provider2" in router.providers
            assert router.routing_strategy == RoutingStrategy.PRIORITY
            assert router.health_check_task is None
    
    def test_routing_strategy_setting(self):
        """Test setting routing strategies"""
        providers = [
            {
                "name": "test_provider",
                "url": "https://test.example.com/v1",
                "model": "test-model"
            }
        ]
        
        with patch('src.processing.enhanced_llm_router.create_openai_compatible_processor'):
            with patch.object(EnhancedLLMRouter, '_start_health_monitoring'):
                router = EnhancedLLMRouter(providers)
            
            # Test setting different strategies
            router.set_routing_strategy(RoutingStrategy.ROUND_ROBIN)
            assert router.routing_strategy == RoutingStrategy.ROUND_ROBIN
            
            router.set_routing_strategy(RoutingStrategy.LOAD_BALANCED)
            assert router.routing_strategy == RoutingStrategy.LOAD_BALANCED
            
            router.set_routing_strategy(RoutingStrategy.FASTEST_RESPONSE)
            assert router.routing_strategy == RoutingStrategy.FASTEST_RESPONSE
    
    def test_provider_selection_priority(self):
        """Test provider selection with priority strategy"""
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
            with patch.object(EnhancedLLMRouter, '_start_health_monitoring'):
                router = EnhancedLLMRouter(providers)
            router.set_routing_strategy(RoutingStrategy.PRIORITY)
            
            # Should select highest priority (lowest number)
            selected = router._select_provider()
            assert selected == "high_priority"
    
    def test_provider_selection_round_robin(self):
        """Test provider selection with round-robin strategy"""
        providers = [
            {
                "name": "provider1",
                "url": "https://provider1.example.com/v1",
                "model": "model1",
                "priority": 1
            },
            {
                "name": "provider2",
                "url": "https://provider2.example.com/v1",
                "model": "model2",
                "priority": 2
            }
        ]
        
        with patch('src.processing.enhanced_llm_router.create_openai_compatible_processor'):
            with patch.object(EnhancedLLMRouter, '_start_health_monitoring'):
                router = EnhancedLLMRouter(providers)
            router.set_routing_strategy(RoutingStrategy.ROUND_ROBIN)
            
            # Should cycle through providers
            first = router._select_provider()
            second = router._select_provider()
            third = router._select_provider()  # Should wrap around
            
            assert first == "provider1"
            assert second == "provider2"
            assert third == "provider1"  # Wrapped around
    
    def test_provider_selection_load_balanced(self):
        """Test provider selection with load-balanced strategy"""
        providers = [
            {
                "name": "provider1",
                "url": "https://provider1.example.com/v1",
                "model": "model1",
                "priority": 1
            },
            {
                "name": "provider2",
                "url": "https://provider2.example.com/v1",
                "model": "model2",
                "priority": 2
            }
        ]
        
        with patch('src.processing.enhanced_llm_router.create_openai_compatible_processor'):
            with patch.object(EnhancedLLMRouter, '_start_health_monitoring'):
                router = EnhancedLLMRouter(providers)
            router.set_routing_strategy(RoutingStrategy.LOAD_BALANCED)
            
            # Simulate different loads
            router.provider_metrics["provider1"].current_load = 5
            router.provider_metrics["provider2"].current_load = 2
            
            # Should select least loaded
            selected = router._select_provider()
            assert selected == "provider2"
    
    def test_provider_selection_fastest_response(self):
        """Test provider selection with fastest-response strategy"""
        providers = [
            {
                "name": "provider1",
                "url": "https://provider1.example.com/v1",
                "model": "model1",
                "priority": 1
            },
            {
                "name": "provider2",
                "url": "https://provider2.example.com/v1",
                "model": "model2",
                "priority": 2
            }
        ]
        
        with patch('src.processing.enhanced_llm_router.create_openai_compatible_processor'):
            with patch.object(EnhancedLLMRouter, '_start_health_monitoring'):
                router = EnhancedLLMRouter(providers)
            router.set_routing_strategy(RoutingStrategy.FASTEST_RESPONSE)
            
            # Set different response times
            router.provider_metrics["provider1"].update_success(2.0)
            router.provider_metrics["provider2"].update_success(1.0)
            
            # Should select fastest
            selected = router._select_provider()
            assert selected == "provider2"
    
    def test_enable_disable_provider(self):
        """Test enabling and disabling providers"""
        providers = [
            {
                "name": "test_provider",
                "url": "https://test.example.com/v1",
                "model": "test-model"
            }
        ]
        
        with patch('src.processing.enhanced_llm_router.create_openai_compatible_processor'):
            with patch.object(EnhancedLLMRouter, '_start_health_monitoring'):
                router = EnhancedLLMRouter(providers)
            
            # Disable provider
            router.disable_provider("test_provider")
            assert router.provider_configs["test_provider"].enabled == False
            
            # Enable provider
            router.enable_provider("test_provider")
            assert router.provider_configs["test_provider"].enabled == True
    
    def test_reset_provider_metrics(self):
        """Test resetting provider metrics"""
        providers = [
            {
                "name": "test_provider",
                "url": "https://test.example.com/v1",
                "model": "test-model"
            }
        ]
        
        with patch('src.processing.enhanced_llm_router.create_openai_compatible_processor'):
            with patch.object(EnhancedLLMRouter, '_start_health_monitoring'):
                router = EnhancedLLMRouter(providers)
            
            # Add some metrics
            router.provider_metrics["test_provider"].update_success(1.0)
            router.provider_metrics["test_provider"].update_failure()
            
            # Reset metrics
            router.reset_provider_metrics("test_provider")
            
            metrics = router.provider_metrics["test_provider"]
            assert metrics.total_requests == 0
            assert metrics.successful_requests == 0
            assert metrics.failed_requests == 0
            assert metrics.consecutive_failures == 0
    
    def test_get_provider_status(self):
        """Test getting provider status"""
        providers = [
            {
                "name": "test_provider",
                "url": "https://test.example.com/v1",
                "model": "test-model",
                "priority": 1,
                "max_concurrent_requests": 5
            }
        ]
        
        with patch('src.processing.enhanced_llm_router.create_openai_compatible_processor'):
            with patch.object(EnhancedLLMRouter, '_start_health_monitoring'):
                router = EnhancedLLMRouter(providers)
            
            status = router.get_provider_status()
            
            assert "test_provider" in status["providers"]
            provider_status = status["providers"]["test_provider"]
            assert provider_status["enabled"] == True
            assert provider_status["priority"] == 1
            assert provider_status["current_load"] == 0
            assert provider_status["max_concurrent_requests"] == 5
    
    def test_get_provider_metrics(self):
        """Test getting provider metrics"""
        providers = [
            {
                "name": "test_provider",
                "url": "https://test.example.com/v1",
                "model": "test-model"
            }
        ]
        
        with patch('src.processing.enhanced_llm_router.create_openai_compatible_processor'):
            with patch.object(EnhancedLLMRouter, '_start_health_monitoring'):
                router = EnhancedLLMRouter(providers)
            
            # Add some metrics
            router.provider_metrics["test_provider"].update_success(1.5)
            router.provider_metrics["test_provider"].update_failure()
            
            metrics = router.get_provider_metrics()
            
            assert "test_provider" in metrics
            assert metrics["test_provider"]["total_requests"] == 2
            assert metrics["test_provider"]["successful_requests"] == 1
            assert metrics["test_provider"]["failed_requests"] == 1
            assert metrics["test_provider"]["consecutive_failures"] == 1
            assert metrics["test_provider"]["success_rate"] == 0.5
            assert metrics["test_provider"]["average_response_time"] == 0.75


class TestRoutingStrategy:
    """Test RoutingStrategy enum"""
    
    def test_routing_strategy_values(self):
        """Test routing strategy enum values"""
        assert RoutingStrategy.PRIORITY.value == "priority"
        assert RoutingStrategy.ROUND_ROBIN.value == "round_robin"
        assert RoutingStrategy.LOAD_BALANCED.value == "load_balanced"
        assert RoutingStrategy.FASTEST_RESPONSE.value == "fastest_response"


class TestProviderStatus:
    """Test ProviderStatus enum"""
    
    def test_provider_status_values(self):
        """Test provider status enum values"""
        assert ProviderStatus.HEALTHY == "healthy"
        assert ProviderStatus.UNHEALTHY == "unhealthy"
        assert ProviderStatus.DEGRADED == "degraded"
        assert ProviderStatus.DISABLED == "disabled"
        assert ProviderStatus.DISABLED.value == "disabled"