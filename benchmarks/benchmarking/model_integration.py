"""
Real LLM Model Integration for Benchmarks - FIXED VERSION
Uses Conjecture's actual infrastructure with specified models
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

class ModelIntegration:
    """Integrates real LLM models with Conjecture infrastructure"""

    def __init__(self):
        self.providers = {
            "granite-tiny": "ibm/granite-4-h-tiny",
            "gpt-oss-20b": "openrouter/gpt-oss-20b",
            "glm-4.6": "zai/GLM-4.6"
        }

    async def get_model_response(self, model_name: str, prompt: str) -> str:
        """Get response from specified model using Conjecture infrastructure"""
        import time
        start_time = time.time()
        
        try:
            # Import Conjecture components
            from processing.simplified_llm_manager import get_simplified_llm_manager
            from processing.unified_bridge import UnifiedLLMBridge, LLMRequest

            # Initialize Conjecture infrastructure
            llm_manager = get_simplified_llm_manager()
            bridge = UnifiedLLMBridge(llm_manager=llm_manager)

            # Create enhanced prompt for benchmark tasks
            enhanced_prompt = self._enhance_prompt_for_benchmark(prompt)

            # Make LLM request
            request = LLMRequest(
                prompt=enhanced_prompt,
                task_type="benchmark",
                max_tokens=2000,
                temperature=0.1,  # Low temperature for benchmark consistency
                top_p=0.9
            )

            response = bridge.process(request)

            if response.success:
                elapsed_time = time.time() - start_time
                print(f"[TIMING] {model_name} response time: {elapsed_time:.3f}s")
                return response.content
            else:
                # Fallback to provider-specific request
                elapsed_time = time.time() - start_time
                print(f"[TIMING] {model_name} fallback response time: {elapsed_time:.3f}s")
                return await self._fallback_provider_request(model_name, prompt)

        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"[ERROR] {model_name} failed after {elapsed_time:.3f}s: {e}")
            # Real error handling - no test responses
            raise Exception(f"LLM call failed for {model_name}: {str(e)}")

    def _enhance_prompt_for_benchmark(self, prompt: str) -> str:
        """Enhance prompt for better benchmark performance"""
        return f"""You are an expert AI assistant responding to a benchmark evaluation question.
Provide a precise, accurate, and well-reasoned answer to the following:

{prompt}

Please ensure your answer is complete and directly addresses the question asked."""

    async def _fallback_provider_request(self, model_name: str, prompt: str) -> str:
        """Fallback request to specific provider"""
        import time
        start_time = time.time()
        
        try:
            # Try to use specific provider if available
            provider = self.providers.get(model_name.lower())
            if provider:
                # Import Conjecture's actual provider system
                from cli.backends.unified_backend import UnifiedBackend
                from config.unified_config import get_config
                
                config = get_config()
                backend = UnifiedBackend(config)
                
                # Make real provider call
                response = await backend.call_provider(provider, prompt)
                
                elapsed_time = time.time() - start_time
                print(f"[TIMING] {model_name} fallback via {provider}: {elapsed_time:.3f}s")
                return response
            else:
                raise Exception(f"No provider configuration found for {model_name}")
                
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"[ERROR] {model_name} fallback failed after {elapsed_time:.3f}s: {e}")
            raise Exception(f"Fallback provider request failed for {model_name}: {str(e)}")

    async def get_conjecture_enhanced_response(self, model_name: str, prompt: str) -> str:
        """Get response using Conjecture enhancement techniques"""
        import time
        start_time = time.time()
        
        try:
            # Apply Conjecture's multi-step enhancement
            enhanced_prompt = f"""Solve this step-by-step with maximum accuracy:

1. Analyze problem thoroughly
2. Consider multiple approaches
3. Select best method
4. Provide complete solution

{prompt}

Please show your work and ensure final answer is clearly stated."""

            response = await self.get_model_response(model_name, enhanced_prompt)
            
            elapsed_time = time.time() - start_time
            print(f"[TIMING] {model_name} conjecture-enhanced response time: {elapsed_time:.3f}s")
            return response
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"[ERROR] {model_name} conjecture enhancement failed after {elapsed_time:.3f}s: {e}")
            raise Exception(f"Conjecture enhancement failed for {model_name}: {str(e)}")

# Model factory functions for benchmark runner
async def granite_tiny_model(prompt: str) -> str:
    """Granite Tiny model with real Conjecture integration"""
    integration = ModelIntegration()
    return await integration.get_model_response("granite-tiny", prompt)

async def gpt_oss_20b_model(prompt: str) -> str:
    """GPT-OSS-20B model with real Conjecture integration"""
    integration = ModelIntegration()
    return await integration.get_model_response("gpt-oss-20b", prompt)

async def glm_46_model(prompt: str) -> str:
    """GLM-4.6 model with real Conjecture integration"""
    integration = ModelIntegration()
    return await integration.get_model_response("glm-4.6", prompt)

# Conjecture-enhanced versions
async def granite_tiny_conjecture(prompt: str) -> str:
    """Granite Tiny with Conjecture enhancement"""
    integration = ModelIntegration()
    return await integration.get_conjecture_enhanced_response("granite-tiny", prompt)

async def gpt_oss_20b_conjecture(prompt: str) -> str:
    """GPT-OSS-20B with Conjecture enhancement"""
    integration = ModelIntegration()
    return await integration.get_conjecture_enhanced_response("gpt-oss-20b", prompt)

async def glm_46_conjecture(prompt: str) -> str:
    """GLM-4.6 with Conjecture enhancement"""
    integration = ModelIntegration()
    return await integration.get_conjecture_enhanced_response("glm-4.6", prompt)