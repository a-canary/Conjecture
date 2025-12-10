#!/usr/bin/env python3
"""
Provider Integration Testing Script
Tests all configured LLM providers with real or test responses
"""

import os
import sys
import json
import asyncio
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.models import BasicClaim, ClaimState, ClaimType
from src.processing.llm.llm_manager import LLMManager
from src.config.unified_provider_validator import UnifiedProviderValidator

@dataclass
class ProviderTestResult:
    """Results from testing a specific provider"""
    provider_name: str
    initialization_success: bool
    initialization_time: float
    initialization_error: Optional[str]
    health_check_success: bool
    health_check_time: float
    health_check_error: Optional[str]
    processing_success: bool
    processing_time: float
    processing_error: Optional[str]
    response_quality_score: Optional[float]
    total_tokens_used: int

class ProviderIntegrationTester:
    """Comprehensive provider integration tester"""
    
    def __init__(self):
        self.results: List[ProviderTestResult] = []
        self.test_claims = [
            BasicClaim(
                claim_id="test_1",
                content="The Earth orbits around the Sun",
                claim_type=ClaimType.ASSERTION.value,
                confidence=0.9
            ),
            BasicClaim(
                claim_id="test_2",
                content="Water boils at 100 degrees Celsius at sea level",
                claim_type=ClaimType.ASSERTION.value,
                confidence=0.95
            ),
            BasicClaim(
                claim_id="test_3",
                content="All cats are mammals",
                claim_type=ClaimType.ASSERTION.value,
                confidence=0.85
            )
        ]
    
    def test_configuration(self) -> Dict[str, Any]:
        """Test current configuration setup"""
        print("🔧 Testing Configuration Setup")
        print("=" * 50)
        
        validator = UnifiedProviderValidator()
        config_info = validator.get_provider_info()
        
        print(f"Provider configured: {config_info.get('configured', False)}")
        
        if config_info.get('configured'):
            print(f"Provider name: {config_info.get('name')}")
            print(f"Model: {config_info.get('model')}")
            print(f"Local: {config_info.get('is_local')}")
            print(f"API URL: {config_info.get('api_url')}")
            print(f"API Key configured: {config_info.get('api_key_configured')}")
            print(f"Ready: {config_info.get('ready')}")
        
        # Validate configuration
        is_valid, errors = validator.validate_configuration()
        
        print(f"\nConfiguration valid: {is_valid}")
        if errors:
            print("Errors:")
            for error in errors:
                print(f"  • {error}")
        
        return config_info
    
    def test_llm_manager_initialization(self) -> Dict[str, Any]:
        """Test LLM Manager initialization"""
        print("\n🚀 Testing LLM Manager Initialization")
        print("=" * 50)
        
        start_time = time.time()
        
        try:
            manager = LLMManager()
            init_time = time.time() - start_time
            
            available_providers = manager.get_available_providers()
            all_providers = manager.get_all_configured_providers()
            failed_providers = manager.failed_providers
            primary_provider = manager.primary_provider
            
            print(f"Initialization time: {init_time:.2f}s")
            print(f"Available providers: {available_providers}")
            print(f"All configured providers: {all_providers}")
            print(f"Failed providers: {list(failed_providers)}")
            print(f"Primary provider: {primary_provider}")
            
            return {
                "success": True,
                "init_time": init_time,
                "available_providers": available_providers,
                "all_providers": all_providers,
                "failed_providers": list(failed_providers),
                "primary_provider": primary_provider,
                "manager": manager
            }
            
        except Exception as e:
            init_time = time.time() - start_time
            print(f"Initialization failed: {e}")
            return {
                "success": False,
                "init_time": init_time,
                "error": str(e)
            }
    
    def test_provider_health(self, manager: LLMManager) -> Dict[str, ProviderTestResult]:
        """Test health of all providers"""
        print("\n🏥 Testing Provider Health")
        print("=" * 50)
        
        health_results = {}
        
        for provider_name in manager.processors.keys():
            print(f"\nTesting {provider_name}...")
            
            result = ProviderTestResult(
                provider_name=provider_name,
                initialization_success=True,
                initialization_time=0,
                initialization_error=None,
                health_check_success=False,
                health_check_time=0,
                health_check_error=None,
                processing_success=False,
                processing_time=0,
                processing_error=None,
                response_quality_score=None,
                total_tokens_used=0
            )
            
            # Health check
            try:
                start_time = time.time()
                health = manager.processors[provider_name].health_check()
                health_time = time.time() - start_time
                
                result.health_check_success = health.get("status") == "healthy"
                result.health_check_time = health_time
                
                if not result.health_check_success:
                    result.health_check_error = health.get("error", "Unknown error")
                
                print(f"  Health: {'✅' if result.health_check_success else '❌'} ({health_time:.2f}s)")
                
            except Exception as e:
                result.health_check_error = str(e)
                print(f"  Health: ❌ Exception: {e}")
            
            health_results[provider_name] = result
        
        return health_results
    
    def test_provider_processing(self, manager: LLMManager) -> Dict[str, ProviderTestResult]:
        """Test claim processing for all providers"""
        print("\n🧠 Testing Claim Processing")
        print("=" * 50)
        
        processing_results = {}
        
        for provider_name in manager.get_available_providers():
            print(f"\nTesting {provider_name} processing...")
            
            result = ProviderTestResult(
                provider_name=provider_name,
                initialization_success=True,
                initialization_time=0,
                initialization_error=None,
                health_check_success=False,
                health_check_time=0,
                health_check_error=None,
                processing_success=False,
                processing_time=0,
                processing_error=None,
                response_quality_score=None,
                total_tokens_used=0
            )
            
            try:
                start_time = time.time()
                processor_result = manager.process_claims(
                    self.test_claims, 
                    task="analyze",
                    provider=provider_name
                )
                processing_time = time.time() - start_time
                
                result.processing_success = processor_result.success
                result.processing_time = processing_time
                result.total_tokens_used = processor_result.tokens_used
                
                if result.processing_success:
                    # Evaluate response quality
                    quality_score = self.evaluate_response_quality(processor_result)
                    result.response_quality_score = quality_score
                    
                    print(f"  Processing: ✅ ({processing_time:.2f}s, {processor_result.tokens_used} tokens)")
                    print(f"  Quality score: {quality_score:.2f}")
                else:
                    result.processing_error = ", ".join(processor_result.errors)
                    print(f"  Processing: ❌ {result.processing_error}")
                
            except Exception as e:
                result.processing_error = str(e)
                print(f"  Processing: ❌ Exception: {e}")
            
            processing_results[provider_name] = result
        
        return processing_results
    
    def evaluate_response_quality(self, result) -> float:
        """Evaluate the quality of a processing result"""
        if not result.success:
            return 0.0
        
        score = 0.0
        max_score = 10.0
        
        processed_claims = result.processed_claims
        
        # Check if all claims were processed
        if len(processed_claims) == len(self.test_claims):
            score += 2.0
        
        # Check claim states
        valid_states = [ClaimState.VERIFIED, ClaimState.UNVERIFIED, ClaimState.DEBUNKED]
        for claim in processed_claims:
            if claim.state in valid_states:
                score += 1.0
        
        # Check confidence scores
        confidence_scores = [claim.confidence for claim in processed_claims if claim.confidence is not None]
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            score += avg_confidence * 2.0
        
        # Check analysis presence
        for claim in processed_claims:
            if hasattr(claim, 'analysis') and claim.analysis:
                score += 1.0
        
        # Bonus for verification info
        for claim in processed_claims:
            if hasattr(claim, 'verification') and claim.verification:
                score += 1.0
        
        return min(score, max_score)
    
    def test_fallback_mechanism(self, manager: LLMManager) -> bool:
        """Test provider fallback mechanism"""
        print("\n🔄 Testing Fallback Mechanism")
        print("=" * 50)
        
        if len(manager.get_available_providers()) < 2:
            print("❌ Need at least 2 available providers for fallback testing")
            return False
        
        # Get primary provider
        primary = manager.primary_provider
        
        # Temporarily mark primary as failed
        original_failed = manager.failed_providers.copy()
        manager.failed_providers.add(primary)
        
        try:
            # Try processing - should use fallback
            result = manager.process_claims(self.test_claims, task="analyze")
            
            if result.success:
                print(f"✅ Fallback successful (original: {primary}, used fallback)")
                success = True
            else:
                print(f"❌ Fallback failed")
                success = False
                
        except Exception as e:
            print(f"❌ Fallback exception: {e}")
            success = False
        
        finally:
            # Restore original state
            manager.failed_providers = original_failed
        
        return success
    
    def test_error_scenarios(self, manager: LLMManager) -> Dict[str, bool]:
        """Test various error scenarios"""
        print("\n⚠️ Testing Error Scenarios")
        print("=" * 50)
        
        results = {}
        
        # Test invalid provider
        try:
            result = manager.process_claims(self.test_claims, task="analyze", provider="invalid_provider")
            results["invalid_provider"] = False  # Should fail
            print("❌ Invalid provider test - should have failed")
        except Exception:
            results["invalid_provider"] = True
            print("✅ Invalid provider properly rejected")
        
        # Test empty claims
        try:
            result = manager.process_claims([], task="analyze")
            results["empty_claims"] = result.success
            print(f"{'✅' if result.success else '❌'} Empty claims handling")
        except Exception as e:
            results["empty_claims"] = False
            print(f"❌ Empty claims exception: {e}")
        
        # Test very long claims (potential token limit issue)
        long_claims = [
            BasicClaim(
                claim_id="long_test",
                content="This is a very long claim " * 1000,  # Very long claim
                claim_type=ClaimType.ASSERTION.value,
                confidence=0.5
            )
        ]
        
        try:
            start_time = time.time()
            result = manager.process_claims(long_claims, task="analyze")
            elapsed = time.time() - start_time
            results["long_claims"] = result.success
            print(f"{'✅' if result.success else '❌'} Long claims handling ({elapsed:.2f}s)")
        except Exception as e:
            results["long_claims"] = False
            print(f"❌ Long claims exception: {e}")
        
        return results
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive provider integration tests"""
        print("🚀 Starting Comprehensive LLM Provider Integration Tests")
        print("=" * 60)
        
        # Test configuration
        config_info = self.test_configuration()
        
        # Test manager initialization
        init_result = self.test_llm_manager_initialization()
        
        if not init_result.get("success"):
            return {
                "overall_success": False,
                "error": "LLM Manager initialization failed",
                "config": config_info,
                "init": init_result
            }
        
        manager = init_result["manager"]
        
        # Test provider health
        health_results = self.test_provider_health(manager)
        
        # Test claim processing
        processing_results = self.test_provider_processing(manager)
        
        # Test fallback mechanism
        fallback_success = self.test_fallback_mechanism(manager)
        
        # Test error scenarios
        error_results = self.test_error_scenarios(manager)
        
        # Calculate overall statistics
        total_providers = len(manager.processors)
        healthy_providers = sum(1 for r in health_results.values() if r.health_check_success)
        working_providers = sum(1 for r in processing_results.values() if r.processing_success)
        avg_quality = sum(p.response_quality_score or 0 for p in processing_results.values()) / max(working_providers, 1)
        
        overall_success = (
            len(manager.get_available_providers()) > 0 and
            working_providers > 0 and
            fallback_success
        )
        
        # Generate summary
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE TEST SUMMARY")
        print("=" * 60)
        print(f"Overall Success: {'✅' if overall_success else '❌'}")
        print(f"Total Providers Configured: {total_providers}")
        print(f"Providers Available: {len(manager.get_available_providers())}")
        print(f"Providers Healthy: {healthy_providers}")
        print(f"Providers Working: {working_providers}")
        print(f"Average Response Quality: {avg_quality:.2f}/10")
        print(f"Fallback Mechanism: {'✅' if fallback_success else '❌'}")
        
        print(f"\nProvider Details:")
        for name, result in processing_results.items():
            health_ok = health_results.get(name, ProviderTestResult("", False, 0, None, False, 0, None, False, 0, None, None, 0)).health_check_success
            status = "✅" if result.processing_success and health_ok else "❌"
            quality = result.response_quality_score or 0
            tokens = result.total_tokens_used
            time_taken = result.processing_time
            
            print(f"  {name}: {status} (Quality: {quality:.1f}, Tokens: {tokens}, Time: {time_taken:.2f}s)")
        
        error_test_success_rate = sum(error_results.values()) / len(error_results) if error_results else 0
        print(f"\nError Scenario Tests: {error_test_success_rate:.1%} passed")
        
        return {
            "overall_success": overall_success,
            "config": config_info,
            "init": init_result,
            "health": {name: {
                "success": result.health_check_success,
                "time": result.health_check_time,
                "error": result.health_check_error
            } for name, result in health_results.items()},
            "processing": {name: {
                "success": result.processing_success,
                "time": result.processing_time,
                "tokens": result.total_tokens_used,
                "quality": result.response_quality_score,
                "error": result.processing_error
            } for name, result in processing_results.items()},
            "fallback": fallback_success,
            "error_tests": error_results,
            "statistics": {
                "total_providers": total_providers,
                "available_providers": len(manager.get_available_providers()),
                "healthy_providers": healthy_providers,
                "working_providers": working_providers,
                "average_quality": avg_quality,
                "error_test_success_rate": error_test_success_rate
            }
        }

def main():
    """Main testing function"""
    print("🧪 LLM Provider Integration Tester")
    print("This script tests all configured LLM providers")
    print()
    
    tester = ProviderIntegrationTester()
    
    try:
        results = tester.run_comprehensive_test()
        
        # Save results to file
        output_file = "provider_test_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: {output_file}")
        
        if results["overall_success"]:
            print("\n🎉 All tests passed! Your LLM provider integration is working correctly.")
            return 0
        else:
            print("\n⚠️ Some tests failed. Please check the results above.")
            return 1
            
    except Exception as e:
        print(f"\n💥 Testing failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())