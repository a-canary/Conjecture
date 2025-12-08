#!/usr/bin/env python3
"""
Test script for Conjecture EndPoint App
Validates all API endpoints and ProcessingInterface integration
"""

import asyncio
import json
import time
from typing import Dict, Any

import requests
import websockets
from datetime import datetime

# Import the new endpoint manager
from src.utils.endpoint_manager import EndpointManager, TemporaryEndpoint, with_endpoint


class EndPointAppTester:
    """Test suite for Conjecture EndPoint App"""
    
    def __init__(self, base_url: str = "http://localhost:8001", endpoint_manager: EndpointManager = None):
        self.base_url = base_url.rstrip('/')
        self.ws_url = base_url.replace('http://', 'ws://').replace('https://', 'wss://')
        self.test_results = []
        self.session_id = None
        self.created_claim_ids = []
        self.endpoint_manager = endpoint_manager
    
    def log_test(self, test_name: str, success: bool, details: str = "", response_data: Any = None):
        """Log test result"""
        result = {
            "test": test_name,
            "success": success,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details,
            "response_data": response_data
        }
        self.test_results.append(result)
        
        status = "PASS" if success else "FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"    {details}")
        if not success and response_data:
            print(f"    Response: {json.dumps(response_data, indent=2)[:200]}...")
        print()
    
    def make_request(self, method: str, endpoint: str, data: Dict[str, Any] = None, params: Dict[str, Any] = None) -> requests.Response:
        """Make HTTP request to endpoint"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == "GET":
                return requests.get(url, params=params, timeout=10)
            elif method.upper() == "POST":
                return requests.post(url, json=data, params=params, timeout=30)
            elif method.upper() == "PUT":
                return requests.put(url, json=data, params=params, timeout=10)
            elif method.upper() == "DELETE":
                return requests.delete(url, params=params, timeout=10)
            else:
                raise ValueError(f"Unsupported method: {method}")
        except requests.exceptions.RequestException as e:
            return type('MockResponse', (), {
                'status_code': 500,
                'text': str(e),
                'json': lambda: {"error": str(e)}
            })()
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = self.make_request("GET", "/health")
        
        success = (
            response.status_code == 200 and
            response.json().get("success", False) and
            "data" in response.json()
        )
        
        self.log_test(
            "Health Check",
            success,
            f"Status: {response.status_code}",
            response.json() if success else None
        )
    
    def test_config_info(self):
        """Test configuration info endpoint"""
        response = self.make_request("GET", "/config")
        
        success = (
            response.status_code == 200 and
            response.json().get("success", False) and
            "providers" in response.json().get("data", {})
        )
        
        self.log_test(
            "Configuration Info",
            success,
            f"Status: {response.status_code}",
            response.json().get("data") if success else None
        )
    
    def test_session_creation(self):
        """Test session creation"""
        test_data = {
            "user_data": {
                "test_user": True,
                "client": "endpoint_app_tester"
            }
        }
        
        response = self.make_request("POST", "/sessions", test_data)
        
        success = (
            response.status_code == 200 and
            response.json().get("success", False)
        )
        
        if success:
            session_data = response.json().get("data", {})
            self.session_id = session_data.get("session_id")
            success = success and self.session_id is not None
        
        self.log_test(
            "Session Creation",
            success,
            f"Status: {response.status_code}, Session ID: {self.session_id}",
            response.json().get("data") if success else None
        )
    
    def test_claim_creation(self):
        """Test claim creation"""
        if not self.session_id:
            self.test_session_creation()
        
        test_data = {
            "content": "Test claim for endpoint app validation - This is a comprehensive test of the FastAPI integration with ProcessingInterface",
            "confidence": 0.85,
            "tags": ["test", "validation", "endpoint_app"],
            "session_id": self.session_id
        }
        
        response = self.make_request("POST", "/claims", test_data)
        
        success = (
            response.status_code == 200 and
            response.json().get("success", False)
        )
        
        if success:
            claim_data = response.json().get("data", {})
            claim_id = claim_data.get("id")
            success = success and claim_id is not None
            if success:
                self.created_claim_ids.append(claim_id)
        
        self.log_test(
            "Claim Creation",
            success,
            f"Status: {response.status_code}, Claim ID: {claim_id if 'claim_id' in locals() else 'None'}",
            response.json().get("data") if success else None
        )
    
    def test_claim_retrieval(self):
        """Test claim retrieval"""
        if not self.created_claim_ids:
            self.test_claim_creation()
        
        claim_id = self.created_claim_ids[0]
        response = self.make_request("GET", f"/claims/{claim_id}")
        
        success = (
            response.status_code == 200 and
            response.json().get("success", False) and
            response.json().get("data", {}).get("id") == claim_id
        )
        
        self.log_test(
            "Claim Retrieval",
            success,
            f"Status: {response.status_code}, Claim ID: {claim_id}",
            response.json().get("data") if success else None
        )
    
    def test_claim_search(self):
        """Test claim search"""
        params = {
            "query": "test claim validation",
            "limit": 5,
            "confidence_min": 0.5
        }
        
        response = self.make_request("GET", "/claims/search", params=params)
        
        success = (
            response.status_code == 200 and
            response.json().get("success", False) and
            isinstance(response.json().get("data", []), list)
        )
        
        self.log_test(
            "Claim Search",
            success,
            f"Status: {response.status_code}, Results: {len(response.json().get('data', []))}",
            response.json().get("data") if success else None
        )
    
    def test_claim_update(self):
        """Test claim update"""
        if not self.created_claim_ids:
            self.test_claim_creation()
        
        claim_id = self.created_claim_ids[0]
        update_data = {
            "updates": {
                "confidence": 0.9,
                "tags": ["test", "validation", "endpoint_app", "updated"]
            },
            "session_id": self.session_id
        }
        
        response = self.make_request("PUT", f"/claims/{claim_id}", update_data)
        
        success = (
            response.status_code == 200 and
            response.json().get("success", False)
        )
        
        self.log_test(
            "Claim Update",
            success,
            f"Status: {response.status_code}, Claim ID: {claim_id}",
            response.json().get("data") if success else None
        )
    
    def test_claim_evaluation(self):
        """Test claim evaluation"""
        if not self.created_claim_ids:
            self.test_claim_creation()
        
        claim_id = self.created_claim_ids[0]
        eval_data = {
            "session_id": self.session_id
        }
        
        response = self.make_request("POST", f"/evaluate?claim_id={claim_id}", eval_data)
        
        success = (
            response.status_code == 200 and
            response.json().get("success", False)
        )
        
        self.log_test(
            "Claim Evaluation",
            success,
            f"Status: {response.status_code}, Claim ID: {claim_id}",
            response.json().get("data") if success else None
        )
    
    def test_context_building(self):
        """Test context building"""
        if not self.created_claim_ids:
            self.test_claim_creation()
        
        context_data = {
            "claim_ids": self.created_claim_ids[:2],  # Use first 2 claims
            "max_skills": 3,
            "max_samples": 5,
            "session_id": self.session_id
        }
        
        response = self.make_request("GET", "/context", params=context_data)
        
        success = (
            response.status_code == 200 and
            response.json().get("success", False) and
            "claim_ids" in response.json().get("data", {})
        )
        
        self.log_test(
            "Context Building",
            success,
            f"Status: {response.status_code}",
            response.json().get("data") if success else None
        )
    
    def test_tools_list(self):
        """Test tools listing"""
        response = self.make_request("GET", "/tools")
        
        success = (
            response.status_code == 200 and
            response.json().get("success", False) and
            isinstance(response.json().get("data", []), list)
        )
        
        self.log_test(
            "Tools List",
            success,
            f"Status: {response.status_code}, Tools: {len(response.json().get('data', []))}",
            response.json().get("data") if success else None
        )
    
    def test_statistics(self):
        """Test statistics endpoint"""
        response = self.make_request("GET", "/stats")
        
        success = (
            response.status_code == 200 and
            response.json().get("success", False) and
            "data" in response.json()
        )
        
        self.log_test(
            "Statistics",
            success,
            f"Status: {response.status_code}",
            response.json().get("data") if success else None
        )
    
    async def test_event_streaming(self):
        """Test Server-Sent Events streaming"""
        try:
            url = f"{self.base_url}/events/stream"
            
            async with requests.aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        # Read some events
                        event_count = 0
                        async for line in response.content:
                            if line.startswith(b'data: '):
                                event_count += 1
                                if event_count >= 3:  # Read a few events
                                    break
                        
                        success = event_count > 0
                        self.log_test(
                            "Event Streaming (SSE)",
                            success,
                            f"Status: {response.status}, Events received: {event_count}"
                        )
                    else:
                        self.log_test(
                            "Event Streaming (SSE)",
                            False,
                            f"Status: {response.status}"
                        )
        except Exception as e:
            self.log_test(
                "Event Streaming (SSE)",
                False,
                f"Exception: {str(e)}"
            )
    
    async def test_websocket_events(self):
        """Test WebSocket event streaming"""
        try:
            uri = f"{self.ws_url}/events/ws"
            
            async with websockets.connect(uri) as websocket:
                # Send a test message to keep connection alive
                await websocket.send("ping")
                
                # Wait for events
                event_count = 0
                timeout = 10  # seconds
                start_time = time.time()
                
                while time.time() - start_time < timeout:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        if message and message != "pong":
                            event_count += 1
                            if event_count >= 2:  # Received enough events
                                break
                    except asyncio.TimeoutError:
                        continue
                
                success = event_count > 0
                self.log_test(
                    "WebSocket Events",
                    success,
                    f"Events received: {event_count}"
                )
                
        except Exception as e:
            self.log_test(
                "WebSocket Events",
                False,
                f"Exception: {str(e)}"
            )
    
    def test_batch_operations(self):
        """Test batch claim creation"""
        if not self.session_id:
            self.test_session_creation()
        
        batch_data = {
            "claims_data": [
                {
                    "content": "Batch test claim 1 - Testing batch creation functionality",
                    "confidence": 0.8,
                    "tags": ["batch", "test"]
                },
                {
                    "content": "Batch test claim 2 - Testing batch creation functionality with multiple claims",
                    "confidence": 0.75,
                    "tags": ["batch", "test", "multiple"]
                }
            ],
            "session_id": self.session_id
        }
        
        response = self.make_request("POST", "/claims/batch", batch_data)
        
        success = (
            response.status_code == 200 and
            response.json().get("success", False) and
            len(response.json().get("data", [])) == 2
        )
        
        self.log_test(
            "Batch Operations",
            success,
            f"Status: {response.status_code}, Claims created: {len(response.json().get('data', []))}",
            response.json().get("data") if success else None
        )
    
    def test_error_handling(self):
        """Test error handling for invalid requests"""
        # Test invalid claim ID
        response = self.make_request("GET", "/claims/invalid_id")
        success = response.status_code == 404
        
        self.log_test(
            "Error Handling (404)",
            success,
            f"Status: {response.status_code} (expected 404)"
        )
        
        # Test invalid claim creation
        invalid_data = {
            "content": "Too short",  # Violates min_length
            "confidence": 1.5  # Violates max confidence
        }
        
        response = self.make_request("POST", "/claims", invalid_data)
        success = response.status_code == 400
        
        self.log_test(
            "Error Handling (400)",
            success,
            f"Status: {response.status_code} (expected 400)"
        )
    
    async def run_all_tests(self):
        """Run all tests"""
        print("Starting Conjecture EndPoint App Tests")
        print(f"Testing against: {self.base_url}")
        print("=" * 60)
        
        # Basic functionality tests
        self.test_health_check()
        self.test_config_info()
        self.test_session_creation()
        self.test_claim_creation()
        self.test_claim_retrieval()
        self.test_claim_search()
        self.test_claim_update()
        self.test_claim_evaluation()
        self.test_context_building()
        self.test_tools_list()
        self.test_statistics()
        self.test_batch_operations()
        self.test_error_handling()
        
        # Real-time features tests
        await self.test_event_streaming()
        await self.test_websocket_events()
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print("=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\nFAILED TESTS:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"  • {result['test']}: {result['details']}")
        
        print("\nCreated Resources:")
        print(f"  • Session ID: {self.session_id}")
        print(f"  • Claim IDs: {self.created_claim_ids}")
        
        # Save results to file
        with open("endpoint_test_results.json", "w") as f:
            json.dump(self.test_results, f, indent=2, default=str)
        print(f"\nDetailed results saved to: endpoint_test_results.json")


async def main():
    """Main test execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Conjecture EndPoint App")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the endpoint app")
    parser.add_argument("--tests", nargs="+", help="Specific tests to run (optional)")
    
    args = parser.parse_args()
    
    tester = EndPointAppTester(args.url)
    
    if args.tests:
        # Run specific tests
        for test_name in args.tests:
            test_method = getattr(tester, f"test_{test_name.lower()}", None)
            if test_method:
                if asyncio.iscoroutinefunction(test_method):
                    await test_method()
                else:
                    test_method()
            else:
                print(f"Unknown test: {test_name}")
    else:
        # Run all tests
        await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())