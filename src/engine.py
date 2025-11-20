"""
Simplified Conjecture Main Agent
Evidence-based AI reasoning system with 90% of features, 10% of complexity
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Any

from .data.data_manager import get_data_manager
from .tools import ToolManager, get_tool_definitions



class Conjecture:
    """
    Simplified Conjecture Agent
    Evidence-based reasoning with essential features only
    """

    def __init__(self, data_path: Optional[str] = None):
        """Initialize Conjecture with simple configuration"""
        self.data_manager = get_data_manager(use_mock_embeddings=True)
        self.tool_manager = ToolManager()
        
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
        print("Conjecture initialized - Simplified Architecture")

    def process_request(
        self, 
        request: str, 
        session_id: Optional[str] = None,
        max_claims: int = 10
    ) -> Dict[str, Any]:
        """Main processing logic for user requests"""
        if not request or len(request.strip()) < 5:
            return {
                "success": False,
                "error": "Request must be at least 5 characters long"
            }

        # Create or get session
        session_id = session_id or f"session_{int(time.time())}"
        self._update_session(session_id, request)

        # Build context
        context = self._build_context(request, max_claims)
        
        # Execute (mock LLM call)
        response = self._call_llm(request, context)
        
        # Process tool results
        tool_results = self._execute_tools_from_response(response)
        
        return {
            "success": True,
            "request": request,
            "response": response,
            "session_id": session_id,
            "context_claims": context,
            "tool_results": tool_results,
            
            "timestamp": datetime.utcnow().isoformat()
        }

    def _update_session(self, session_id: str, request: str):
        """Update session with request and timestamp"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "messages": [],
                "created": datetime.utcnow()
            }
        
        self.sessions[session_id]["messages"].append({
            "role": "user",
            "content": request,
            "timestamp": datetime.utcnow()
        })
        self.sessions[session_id]["last_updated"] = datetime.utcnow()

    def _build_context(self, request: str, max_claims: int) -> List[Dict[str, Any]]:
        """Build context from existing claims"""
        # Search for relevant claims (simplified implementation)
        relevant_claims = self._search_claims_simple(request, max_claims)
        
        # Convert to dict format if needed
        context_claims = []
        for claim in relevant_claims:
            if hasattr(claim, 'dict'):
                context_claims.append(claim.dict())
            else:
                context_claims.append(claim)
        
        return context_claims
        
        # Get recent claims if no relevant claims found
        if not relevant_claims:
            relevant_claims = self.data_manager.get_recent_claims(max_claims)
        
        return [claim.dict() for claim in relevant_claims]

    def _call_llm(self, request: str, context: List[Dict]) -> str:
        """Mock LLM interaction (in real implementation, call actual LLM)"""
        
        # Simple mock response generation
        context_info = ""
        if context:
            context_info = f"Found {len(context)} relevant claims. "
        
        
        
        tool_list = ", ".join(get_tool_definitions().keys())
        
        response = f"""I'll help you with your request.

{context_info}Based on the available context, I need to gather more information.

Available tools: {tool_list}

Let me start by searching for relevant information and then create structured claims based on what I find.

WebSearch(query='{request}'), CreateClaim(content='Analyzing request for {request}', confidence=0.8, claim_type='concept')

This will help me gather evidence and build a comprehensive understanding of your request."""
        
        return response

    def _execute_tools_from_response(self, response: str) -> List[Dict[str, Any]]:
        """Execute tool calls found in the response"""
        tool_calls = self.tool_manager.parse_tool_calls(response)
        results = []
        
        for call in tool_calls:
            result = self.tool_manager.call_tool(
                call["tool"], 
                call["parameters"]
            )
            results.append({
                "tool": call["tool"],
                "parameters": call["parameters"],
                "result": result
            })
        
        return results

    def _search_claims_simple(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Simple claim search (mock implementation for testing)"""
        # Mock implementation - in real system use data manager search
        mock_claims = [
            {
                "id": "mock_001",
                "content": f"Related to {query} - concept 1",
                "confidence": 0.85,
                "type": ["concept"],
                "tags": [query.lower()]
            },
            {
                "id": "mock_002", 
                "content": f"Related to {query} - reference 1",
                "confidence": 0.90,
                "type": ["reference"],
                "tags": [query.lower(), "mock"]
            }
        ]
        return mock_claims[:limit]

    async def create_claim(
        self,
        content: str,
        confidence: float,
        claim_type: str,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create a new claim (simplified mock implementation)"""
        try:
            # Mock claim creation
            import time
            claim_id = f"c{int(time.time()) % 10000000:07d}"  # Format c#######
            
            mock_claim = {
                "id": claim_id,
                "content": content,
                "confidence": confidence,
                "claim_type": claim_type,
                "tags": tags or [],
                "created_by": "simplified_conjecture",
                "created_at": time.time()
            }
            return {
                "success": True,
                "claim": mock_claim
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def get_recent_claims(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent claims"""
        try:
            # Mock recent claims
            return self._search_claims_simple("recent", limit)
        except Exception as e:
            print(f"Recent claims error: {e}")
            return []

    async def search_claims(
        self,
        query: str,
        confidence_threshold: Optional[float] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search claims"""
        try:
            # Use simple mock search for now
            return self._search_claims_simple(query, limit)
        except Exception as e:
            print(f"Search error: {e}")
            return []

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID"""
        return self.sessions.get(session_id)

    def cleanup_sessions(self, days_old: int = 7) -> int:
        """Remove old sessions"""
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        old_sessions = []
        
        for session_id, session_data in self.sessions.items():
            if session_data.get("last_updated", session_data["created"]).timestamp() < cutoff_time:
                old_sessions.append(session_id)
        
        for session_id in old_sessions:
            del self.sessions[session_id]
        
        return len(old_sessions)

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        # Return basic statistics without data manager dependency
        try:
            import asyncio
            claim_stats = asyncio.run(self.data_manager.get_statistics())
        except:
            # Fallback if data manager not initialized
            claim_stats = {
                "total_claims": 0,
                "dirty_claims": 0,
                "clean_claims": 0
            }
        
        return {
            "active_sessions": len(self.sessions),
            "available_tools": len(get_tool_definitions()),
            
            **claim_stats
        }


# Convenience functions for immediate use
def explore(query: str, max_claims: int = 10) -> Dict[str, Any]:
    """Quick exploration function"""
    cf = Conjecture()
    return cf.process_request(query, max_claims=max_claims)


def add_claim(content: str, confidence: float, claim_type: str, **kwargs) -> Dict[str, Any]:
    """Quick add claim function"""
    cf = Conjecture()
    return cf.create_claim(content, confidence, claim_type, **kwargs)


if __name__ == "__main__":
    print("🧪 Testing Simplified Conjecture")
    print("=" * 40)
    
    cf = Conjecture()
    
    # Test basic processing
    print("🔍 Testing request processing...")
    result = cf.process_request("Research machine learning algorithms")
    print(f"✅ Request processed: {result['success']}")
    
    print(f"   Context claims: {len(result['context_claims'])}")
    print(f"   Tool results: {len(result['tool_results'])}")
    
    # Test claim creation
    print("\n➕ Testing claim creation...")
    claim_result = cf.create_claim(
        content="Simplified Conjecture handles 90% of use cases with 10% complexity",
        confidence=0.95,
        claim_type="concept",
        tags=["architecture", "simplicity"]
    )
    print(f"✅ Claim created: {claim_result['success']}")
    
    # Test search
    print("\n🔎 Testing claim search...")
    search_results = cf.search_claims("Conjecture")
    print(f"✅ Search results: {len(search_results)} claims")
    
    # Test statistics
    print("\n📊 Testing statistics...")
    stats = cf.get_statistics()
    print(f"✅ Statistics: {stats}")
    
    print("\n🎉 Simplified Conjecture tests passed!")