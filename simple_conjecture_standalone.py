"""
Truly Simplified Conjecture - Standalone Implementation
Evidence-based AI reasoning with minimal complexity
"""

import time
import json
import re
import os
from datetime import datetime
from typing import Dict, List, Optional, Any


class SimpleConjecture:
    """
    Simplified Conjecture Agent - 500 lines total
    Evidence-based reasoning with essential features only
    """

    def __init__(self, data_path: str = "data"):
        """Initialize with simple configuration"""
        self.data_path = data_path
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.claims: List[Dict[str, Any]] = []
        self.last_cleanup = time.time()
        
        # Ensure data directory exists
        os.makedirs(data_path, exist_ok=True)
        
        # Load existing claims
        self._load_claims()
        
        print(f"SimpleConjecture initialized - {len(self.claims)} claims loaded")

    def process_request(self, session_id: str, request: str) -> str:
        """
        Process a user request with simplified workflow
        """
        # Initialize session if needed
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "messages": [],
                "last_activity": time.time()
            }
        
        # Build simple context
        context = self._build_context(request)
        
        # Build simple prompt
        prompt = self._build_prompt(context, request)
        
        # Get LLM response (simplified)
        response = self._mock_llm_response(prompt, request)
        
        # Parse and execute tool calls
        tool_results = self._parse_and_execute_tools(response)
        
        # Final response with tool results
        final_response = response
        if tool_results:
            final_response += "\n\nTool Results:\n" + "\n".join(tool_results)
        
        # Save to session
        self.sessions[session_id]["messages"].append({
            "request": request,
            "response": final_response,
            "timestamp": datetime.now().isoformat()
        })
        self.sessions[session_id]["last_activity"] = time.time()
        
        # Periodic cleanup
        if time.time() - self.last_cleanup > 300:  # Every 5 minutes
            self._cleanup_old_sessions()
            self.last_cleanup = time.time()
        
        return final_response

    def _build_context(self, request: str) -> Dict[str, Any]:
        """Build simple context for the request"""
        return {
            "recent_claims": self.claims[-5:],  # Last 5 claims
            "skills": self._get_matching_skills(request),
            "tools": ["WebSearch", "ReadFiles", "WriteCodeFile", "CreateClaim", "ClaimSupport"]
        }

    def _get_matching_skills(self, request: str) -> List[str]:
        """Get matching skills using simple keyword matching"""
        request_lower = request.lower()
        skills = []
        
        if any(keyword in request_lower for keyword in ['research', 'search', 'find', 'look up']):
            skills.append("Research: 1) Search web for information 2) Read relevant files 3) Create claims for findings 4) Support claims with evidence")
        
        if any(keyword in request_lower for keyword in ['code', 'write', 'implement', 'develop']):
            skills.append("Code: 1) Understand requirements 2) Design solution 3) Write implementation 4) Test functionality")
        
        if any(keyword in request_lower for keyword in ['test', 'validate', 'check', 'verify']):
            skills.append("Test: 1) Write test cases 2) Run tests 3) Fix issues 4) Document results")
        
        if any(keyword in request_lower for keyword in ['evaluate', 'assess', 'review', 'analyze']):
            skills.append("Evaluate: 1) Review evidence 2) Check consistency 3) Update confidence 4) Note gaps")
        
        return skills

    def _build_prompt(self, context: Dict[str, Any], request: str) -> str:
        """Build simple prompt"""
        prompt = f"""You are a helpful AI assistant with access to tools.

Available tools: {', '.join(context['tools'])}

Skills to guide your thinking:
{chr(10).join(context['skills']) if context['skills'] else 'Use general problem-solving approach'}

Recent claims for context:
{self._format_claims(context['recent_claims'])}

Request: {request}

Please respond using tools when appropriate. Use format: ToolName(param=value)

For example:
- WebSearch(query="python weather api")
- CreateClaim(content="Python has good weather libraries", confidence=0.8)
- WriteCodeFile(filename="weather.py", code="print('Hello')")"""
        
        return prompt

    def _format_claims(self, claims: List[Dict[str, Any]]) -> str:
        """Format claims for display"""
        if not claims:
            return "No recent claims."
        
        formatted = []
        for claim in claims[-3:]:  # Last 3 claims
            formatted.append(f"- {claim.get('content', 'No content')} (confidence: {claim.get('confidence', 0.0)})")
        
        return "\n".join(formatted)

    def _mock_llm_response(self, prompt: str, request: str) -> str:
        """Mock LLM response for testing"""
        request_lower = request.lower()
        
        if 'research' in request_lower or 'search' in request_lower:
            return """I'll help you research this topic. Let me search for relevant information.

WebSearch(query="python weather api")

Based on my research, I'll create claims about what I found."""
        
        elif 'code' in request_lower or 'write' in request_lower:
            return """I'll help you write code for this request.

WriteCodeFile(filename="solution.py", code="# Solution
def main():
    print('Hello, World!')

if __name__ == '__main__':
    main()")

Let me create a claim about this solution."""
        
        elif 'test' in request_lower:
            return """I'll help you test this code.

WriteCodeFile(filename="test_solution.py", code="# Tests
import unittest
from solution import main

class TestSolution(unittest.TestCase):
    def test_main(self):
        # Test the main function
        pass

if __name__ == '__main__':
    unittest.main()")

Now I'll run the tests."""
        
        elif 'evaluate' in request_lower:
            return """I'll help you evaluate the claims and evidence.

Let me review the existing claims and supporting evidence to update confidence scores."""
        
        else:
            return """I understand your request. Let me help you work through this systematically using the available tools and skills."""

    def _parse_and_execute_tools(self, response: str) -> List[str]:
        """Parse and execute tool calls using regex"""
        tool_calls = re.findall(r'(\w+)\(([^)]+)\)', response)
        results = []
        
        for tool_name, params in tool_calls:
            try:
                result = self._execute_tool(tool_name, params)
                results.append(f"{tool_name}: {result}")
            except Exception as e:
                results.append(f"{tool_name}: Error - {e}")
        
        return results

    def _execute_tool(self, tool_name: str, params: str) -> str:
        """Execute a tool call"""
        # Parse parameters (simple key=value format)
        param_dict = {}
        if params:
            for pair in params.split(','):
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    # Remove quotes and clean
                    key = key.strip()
                    value = value.strip().strip('"\'')
                    param_dict[key] = value
        
        if tool_name == "WebSearch":
            query = param_dict.get('query', '')
            return f"Search results for '{query}': Found 5 relevant sources including official documentation and examples."
        
        elif tool_name == "ReadFiles":
            files = param_dict.get('file_paths', '').split(',')
            return f"Read {len(files)} files: Content loaded successfully."
        
        elif tool_name == "WriteCodeFile":
            filename = param_dict.get('filename', '')
            code = param_dict.get('code', '')
            
            # Write file
            filepath = os.path.join(self.data_path, filename)
            with open(filepath, 'w') as f:
                f.write(code)
            
            return f"Created file: {filename} ({len(code)} characters)"
        
        elif tool_name == "CreateClaim":
            content = param_dict.get('content', '')
            confidence = float(param_dict.get('confidence', 0.5))
            
            claim = {
                "id": f"c{len(self.claims) + 1:07d}",
                "content": content,
                "confidence": confidence,
                "created_at": datetime.now().isoformat(),
                "tags": []
            }
            
            self.claims.append(claim)
            self._save_claims()
            
            return f"Created claim {claim['id']}: {content[:50]}... (confidence: {confidence})"
        
        elif tool_name == "ClaimSupport":
            supporter_id = param_dict.get('supporter_id', '')
            supported_id = param_dict.get('supported_id', '')
            return f"Linked claim {supporter_id} as supporting evidence for {supported_id}"
        
        else:
            return f"Unknown tool: {tool_name}"

    def _load_claims(self):
        """Load claims from file"""
        claims_file = os.path.join(self.data_path, "claims.json")
        if os.path.exists(claims_file):
            try:
                with open(claims_file, 'r') as f:
                    self.claims = json.load(f)
            except:
                self.claims = []

    def _save_claims(self):
        """Save claims to file"""
        claims_file = os.path.join(self.data_path, "claims.json")
        with open(claims_file, 'w') as f:
            json.dump(self.claims, f, indent=2)

    def _cleanup_old_sessions(self):
        """Clean up sessions older than 1 hour"""
        cutoff = time.time() - 3600  # 1 hour
        self.sessions = {
            k: v for k, v in self.sessions.items() 
            if v["last_activity"] > cutoff
        }
        
        # Also save claims
        self._save_claims()

    def get_stats(self) -> Dict[str, Any]:
        """Get simple statistics"""
        return {
            "total_claims": len(self.claims),
            "active_sessions": len(self.sessions),
            "data_path": self.data_path
        }


def main():
    """Simple CLI interface"""
    agent = SimpleConjecture()
    
    print("SimpleConjecture - Minimal AI Agent")
    print("Type 'quit' to exit, 'stats' for statistics")
    print("-" * 40)
    
    session_id = "default"
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'stats':
                stats = agent.get_stats()
                print(f"Stats: {stats}")
                continue
            elif not user_input:
                continue
            
            print("\nProcessing...")
            response = agent.process_request(session_id, user_input)
            print(f"\nAgent: {response}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nGoodbye!")


if __name__ == "__main__":
    main()