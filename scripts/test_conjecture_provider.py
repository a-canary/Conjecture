#!/usr/bin/env python3
"""
Conjecture Provider Test Script
Tests all endpoints of the Conjecture local LLM provider
"""

import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, Optional

import aiohttp

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_root, "src"))

# Provider configuration
PROVIDER_URL = "http://127.0.0.1:5678"
TIMEOUT = 30  # seconds


async def test_endpoint(
    session: aiohttp.ClientSession,
    method: str,
    endpoint: str,
    data: Optional[Dict[str, Any]] = None,
):
    """Test a specific endpoint"""
    url = f"{PROVIDER_URL}{endpoint}"

    try:
        if method.upper() == "GET":
            async with session.get(url, timeout=TIMEOUT) as response:
                result = {
                    "status": response.status,
                    "success": response.status == 200,
                    "data": await response.json()
                    if response.content_type == "application/json"
                    else await response.text(),
                }
        elif method.upper() == "POST":
            headers = {"Content-Type": "application/json"}
            async with session.post(
                url, json=data, headers=headers, timeout=TIMEOUT
            ) as response:
                result = {
                    "status": response.status,
                    "success": response.status == 200,
                    "data": await response.json()
                    if response.content_type == "application/json"
                    else await response.text(),
                }
        else:
            result = {
                "status": -1,
                "success": False,
                "data": f"Unsupported method: {method}",
            }

        return result
    except asyncio.TimeoutError:
        return {
            "status": -2,
            "success": False,
            "data": f"Timeout after {TIMEOUT} seconds",
        }
    except Exception as e:
        return {"status": -3, "success": False, "data": f"Exception: {str(e)}"}


async def test_health_endpoint():
    """Test the health check endpoint"""
    print("\n🔍 Testing Health Check Endpoint...")

    async with aiohttp.ClientSession() as session:
        result = await test_endpoint(session, "GET", "/health")

        if result["success"]:
            data = result["data"]
            print(f"✅ Health check successful")
            print(f"   Status: {data.get('status', 'Unknown')}")
            print(f"   Timestamp: {data.get('timestamp', 'Unknown')}")
            print(
                f"   Conjecture Initialized: {data.get('conjecture_initialized', 'Unknown')}"
            )
            return True
        else:
            print(f"❌ Health check failed: {result['data']}")
            return False


async def test_models_endpoint():
    """Test the models listing endpoint"""
    print("\n🔍 Testing Models Endpoint...")

    async with aiohttp.ClientSession() as session:
        result = await test_endpoint(session, "GET", "/models")

        if result["success"]:
            data = result["data"]
            print(f"✅ Models endpoint successful")
            print(f"   Object: {data.get('object', 'Unknown')}")
            if "data" in data and len(data["data"]) > 0:
                model = data["data"][0]
                print(f"   Model ID: {model.get('id', 'Unknown')}")
                print(f"   Model Object: {model.get('object', 'Unknown')}")
                print(f"   Owned By: {model.get('owned_by', 'Unknown')}")
            return True
        else:
            print(f"❌ Models endpoint failed: {result['data']}")
            return False


async def test_chat_completion_endpoint():
    """Test the chat completions endpoint"""
    print("\n🔍 Testing Chat Completions Endpoint...")

    test_request = {
        "model": "conjecture-local",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        "max_tokens": 100,
        "temperature": 0.7,
    }

    async with aiohttp.ClientSession() as session:
        result = await test_endpoint(
            session, "POST", "/v1/chat/completions", test_request
        )

        if result["success"]:
            data = result["data"]
            print(f"✅ Chat completion successful")
            print(f"   ID: {data.get('id', 'Unknown')}")
            print(f"   Model: {data.get('model', 'Unknown')}")
            if "choices" in data and len(data["choices"]) > 0:
                message = data["choices"][0].get("message", {})
                print(f"   Response: {message.get('content', 'No content')[:100]}...")
            if "usage" in data:
                usage = data["usage"]
                print(
                    f"   Tokens: {usage.get('prompt_tokens', 0)} prompt, {usage.get('completion_tokens', 0)} completion"
                )
            return True
        else:
            print(f"❌ Chat completion failed: {result['data']}")
            return False


async def test_tell_user_tool():
    """Test the TellUser tool endpoint"""
    print("\n🔍 Testing TellUser Tool...")

    test_request = {
        "message": "This is a test message from the TellUser tool",
        "context": {"test": True, "source": "test_script"},
    }

    async with aiohttp.ClientSession() as session:
        result = await test_endpoint(session, "POST", "/tools/tell_user", test_request)

        if result["success"]:
            data = result["data"]
            print(f"✅ TellUser tool successful")
            print(f"   Success: {data.get('success', 'Unknown')}")
            print(f"   Message: {data.get('message', 'Unknown')[:50]}...")
            print(f"   Timestamp: {data.get('timestamp', 'Unknown')}")
            return True
        else:
            print(f"❌ TellUser tool failed: {result['data']}")
            return False


async def test_ask_user_tool():
    """Test the AskUser tool endpoint"""
    print("\n🔍 Testing AskUser Tool...")

    test_request = {
        "question": "What is your favorite programming language?",
        "context": {"test": True, "source": "test_script"},
        "options": ["Python", "JavaScript", "Java", "C++", "Other"],
    }

    async with aiohttp.ClientSession() as session:
        result = await test_endpoint(session, "POST", "/tools/ask_user", test_request)

        if result["success"]:
            data = result["data"]
            print(f"✅ AskUser tool successful")
            print(f"   Success: {data.get('success', 'Unknown')}")
            print(f"   Question: {data.get('question', 'Unknown')}")
            print(f"   Options: {data.get('options', 'Unknown')}")
            print(f"   Timestamp: {data.get('timestamp', 'Unknown')}")
            return True
        else:
            print(f"❌ AskUser tool failed: {result['data']}")
            return False


async def test_root_endpoint():
    """Test the root endpoint"""
    print("\n🔍 Testing Root Endpoint...")

    async with aiohttp.ClientSession() as session:
        result = await test_endpoint(session, "GET", "/")

        if result["success"]:
            data = result["data"]
            print(f"✅ Root endpoint successful")
            print(f"   Message: {data.get('message', 'Unknown')}")
            print(f"   Version: {data.get('version', 'Unknown')}")
            return True
        else:
            print(f"❌ Root endpoint failed: {result['data']}")
            return False


async def main():
    """Main test runner"""
    print("🚀 Starting Conjecture Provider Tests")
    print(f"📍 Provider URL: {PROVIDER_URL}")
    print(f"⏱️ Timeout: {TIMEOUT} seconds")

    # Check if aiohttp is installed
    try:
        import aiohttp
    except ImportError:
        print("❌ aiohttp not installed. Please install it with: pip install aiohttp")
        return False

    # Run all tests
    tests = [
        ("Health Check", test_health_endpoint),
        ("Models", test_models_endpoint),
        ("Chat Completions", test_chat_completion_endpoint),
        ("TellUser Tool", test_tell_user_tool),
        ("AskUser Tool", test_ask_user_tool),
        ("Root Endpoint", test_root_endpoint),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
            time.sleep(1)  # Small delay between tests
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {str(e)}")
            results.append((test_name, False))

    # Print summary
    print("\n📊 Test Results Summary:")
    print("-" * 40)
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1

    print("-" * 40)
    print(
        f"Total: {len(results)} tests, {passed} passed, {len(results) - passed} failed"
    )

    if passed == len(results):
        print("🎉 All tests passed! The Conjecture provider is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the provider configuration.")
        return False


if __name__ == "__main__":
    # Check if provider is running
    import socket

    def is_port_open(host, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.settimeout(1)
                result = s.connect_ex((host, port))
                return result == 0
            except:
                return False

    if not is_port_open("127.0.0.1", 5678):
        print("❌ Conjecture provider is not running on port 5678")
        print(
            "Please start the provider with: python scripts/start_conjecture_provider.py"
        )
        sys.exit(1)

    # Run the tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
