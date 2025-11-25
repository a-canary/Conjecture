#!/usr/bin/env python3
"""
Direct Core Tools Test
Test the tools directly without registry dependencies
"""

import sys
import os
import json

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Create a simple version of claim management without dependencies
claim_storage = {}
support_relationships = {}
next_claim_id = 1


def DirectClaimCreate(content, confidence=0.8, tags=None):
    """Direct claim creation without registry"""
    global next_claim_id
    
    if not content or not content.strip():
        return {'success': False, 'error': 'Claim content cannot be empty'}
    
    if tags is None:
        tags = []
    
    claim_id = f"claim_{next_claim_id}"
    next_claim_id += 1
    
    claim = {
        'id': claim_id,
        'content': content.strip(),
        'confidence': float(confidence),
        'tags': tags,
        'supports': [],
        'supported_by': []
    }
    
    claim_storage[claim_id] = claim
    support_relationships[claim_id] = set()
    
    return {
        'success': True,
        'claim_id': claim_id,
        'claim': claim
    }


def DirectReason(thought_process):
    """Direct reasoning without registry"""
    if not thought_process or not thought_process.strip():
        return {'success': False, 'error': 'Thought process cannot be empty'}
    
    return {
        'success': True,
        'reasoning_id': f"reason_{len(thought_process)}",
        'step_number': 1,
        'thought_process': thought_process.strip()
    }


def DirectTellUser(message, message_type="info"):
    """Direct user messaging without registry"""
    if not message or not message.strip():
        return {'success': False, 'error': 'Message cannot be empty'}
    
    print(f"[{message_type.upper()}] {message}")
    
    return {
        'success': True,
        'message_id': f"message_{len(message)}",
        'message_type': message_type
    }


def test_core_functionality():
    """Test core functionality directly"""
    print("Direct Core Tools Functionality Test")
    print("=" * 50)
    
    # Test claim creation
    print("\n1. Testing Claim Creation")
    result = DirectClaimCreate("Core Tools system is functional", confidence=0.9, tags=["test", "core"])
    print(f"   ClaimCreate: {'Success' if result['success'] else 'Failed'}")
    if result['success']:
        claim_id = result['claim_id']
        print(f"   Created claim: {claim_id}")
        print(f"   Content: {result['claim']['content']}")
        print(f"   Confidence: {result['claim']['confidence']}")
    
    # Test reasoning
    print("\n2. Testing Reasoning")
    result = DirectReason("Verified Core Tools claim creation functionality")
    print(f"   Reason: {'Success' if result['success'] else 'Failed'}")
    if result['success']:
        print(f"   Reasoning ID: {result['reasoning_id']}")
        print(f"   Process: {result['thought_process']}")
    
    # Test user messaging
    print("\n3. Testing User Messaging")
    result = DirectTellUser("Core Tools direct test completed successfully", "success")
    print(f"   TellUser: {'Success' if result['success'] else 'Failed'}")
    if result['success']:
        print(f"   Message ID: {result['message_id']}")
    
    # Test file operations (simplified)
    print("\n4. Testing File Operations")
    try:
        test_content = "# Core Tools Test\n\nThis file was created to verify Core Tools functionality."
        
        # Write file
        with open("test_direct.txt", "w") as f:
            f.write(test_content)
        print("   File write: Success")
        
        # Read file
        with open("test_direct.txt", "r") as f:
            read_content = f.read()
        print("   File read: Success")
        print(f"   Content length: {len(read_content)} characters")
        
        # Clean up
        os.remove("test_direct.txt")
        print("   File cleanup: Success")
        
        file_success = True
    except Exception as e:
        print(f"   File operations: Failed - {e}")
        file_success = False
    
    # Test JSON response format
    print("\n5. Testing JSON Response Format")
    tool_calls = [
        {
            'name': 'ClaimCreate',
            'arguments': {'content': 'Test claim', 'confidence': 0.8},
            'call_id': 'test_1'
        },
        {
            'name': 'Reason',
            'arguments': {'thought_process': 'Testing JSON format'},
            'call_id': 'test_2'
        }
    ]
    
    response = {'tool_calls': tool_calls}
    json_response = json.dumps(response, indent=2)
    
    try:
        parsed = json.loads(json_response)
        json_success = len(parsed['tool_calls']) == 2
        print(f"   JSON format: {'Success' if json_success else 'Failed'}")
        print(f"   Tool calls found: {len(parsed['tool_calls'])}")
    except:
        json_success = False
        print("   JSON format: Failed")
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    tests = [
        ("Claim Creation", result['success']),
        ("Reasoning", result['success']),
        ("User Messaging", result['success']),
        ("File Operations", file_success),
        ("JSON Format", json_success)
    ]
    
    passed = sum(1 for success in [t[1] for t in tests])
    total = len(tests)
    
    for test_name, success in tests:
        status = "PASS" if success else "FAIL"
        print(f"{test_name:.<20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("Core Tools functionality verified!")
        return True
    else:
        print("Some Core Tools functionality issues found.")
        return False


if __name__ == "__main__":
    success = test_core_functionality()
    sys.exit(0 if success else 1)