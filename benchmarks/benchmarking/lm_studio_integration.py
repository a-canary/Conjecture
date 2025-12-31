"""
Direct LM Studio Integration for Benchmarks
Connects directly to LM Studio API for benchmark evaluation
"""

import aiohttp
import asyncio
import json
from typing import Dict, Any, Optional

class LMStudioIntegration:
    """Direct integration with LM Studio API"""

    def __init__(self, base_url: str = "http://localhost:1234/v1", model: str = "ibm/granite-4-h-tiny"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_response(self, prompt: str, **kwargs) -> str:
        """Get response from LM Studio model"""
        if not self.session:
            self.session = aiohttp.ClientSession()

        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert AI assistant responding to a mathematics benchmark evaluation. Provide clear, accurate, step-by-step solutions to math problems. Always provide the final numerical answer clearly stated."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": kwargs.get('max_tokens', 4000),
            "temperature": kwargs.get('temperature', 0.1),
            "top_p": kwargs.get('top_p', 0.9)
        }

        try:
            async with self.session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    raise Exception(f"LM Studio API error {response.status}: {error_text}")

        except asyncio.TimeoutError:
            raise Exception("LM Studio request timed out")
        except Exception as e:
            raise Exception(f"Failed to get response from LM Studio: {str(e)}")

    async def test_connection(self) -> bool:
        """Test if LM Studio is accessible"""
        try:
            response = await self.get_response("Hello, can you respond with 'Connection successful'?", max_tokens=50)
            return "connection successful" in response.lower()
        except:
            return False

# Direct model functions for benchmark runner
async def granite_tiny_direct(prompt: str) -> str:
    """Direct Granite Tiny model via LM Studio"""
    async with LMStudioIntegration() as lm_studio:
        return await lm_studio.get_response(prompt)

async def granite_tiny_direct_conjecture(prompt: str) -> str:
    """Granite Tiny with Conjecture-style enhancement via LM Studio"""
    enhanced_prompt = f"""Solve this step-by-step with maximum accuracy:

1. Analyze the problem thoroughly
2. Consider multiple approaches
3. Select the best method
4. Provide a complete solution with clear reasoning
5. State the final answer clearly

{prompt}

Please show your work and ensure your final answer is clearly stated."""

    async with LMStudioIntegration() as lm_studio:
        return await lm_studio.get_response(enhanced_prompt)

# Test function
async def test_lm_studio_connection():
    """Test connection to LM Studio"""
    try:
        async with LMStudioIntegration() as lm_studio:
            if await lm_studio.test_connection():
                print("✓ LM Studio connection successful")
                return True
            else:
                print("✗ LM Studio connection failed")
                return False
    except Exception as e:
        print(f"✗ LM Studio connection error: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_lm_studio_connection())