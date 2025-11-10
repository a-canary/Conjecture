"""
Test Suite for Simplified Conjecture Architecture
Covers basic workflows and functionality
"""

import os
import tempfile
import unittest
from src.conjecture import Conjecture
from src.data import DataManager
from src.tools import ToolManager
from src.skills import SkillManager


class TestSimplifiedConjecture(unittest.TestCase):
    """Test the simplified Conjecture architecture"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_path = os.path.join(self.temp_dir, "test_claims.json")
        self.conjecture = Conjecture(data_path=self.test_data_path)

    def test_conjecture_initialization(self):
        """Test Conjecture initialization"""
        self.assertIsNotNone(self.conjecture.data_manager)
        self.assertIsNotNone(self.conjecture.tool_manager)
        self.assertIsNotNone(self.conjecture.skill_manager)
        self.assertEqual(len(self.conjecture.sessions), 0)

    def test_claim_creation(self):
        """Test basic claim creation"""
        result = self.conjecture.create_claim(
            content="Test claim for unit testing",
            confidence=0.85,
            claim_type="concept",
            tags=["test", "unit"]
        )
        
        self.assertTrue(result["success"])
        self.assertIn("claim", result)
        self.assertEqual(result["claim"]["content"], "Test claim for unit testing")

    def test_request_processing(self):
        """Test basic request processing"""
        result = self.conjecture.process_request("Research machine learning basics")
        
        self.assertTrue(result["success"])
        self.assertIn("response", result)
        self.assertIn("session_id", result)
        self.assertIn("skill_used", result)
        self.assertIn("tool_results", result)

    def test_session_management(self):
        """Test session creation and retrieval"""
        # Process first request
        result1 = self.conjecture.process_request("Test request 1")
        session_id = result1["session_id"]
        
        # Check session exists
        session = self.conjecture.get_session(session_id)
        self.assertIsNotNone(session)
        self.assertEqual(len(session["messages"]), 1)
        
        # Process second request in same session
        result2 = self.conjecture.process_request("Test request 2", session_id=session_id)
        self.assertEqual(result2["session_id"], session_id)
        
        # Check session updated
        session = self.conjecture.get_session(session_id)
        self.assertEqual(len(session["messages"]), 2)

    def test_claim_search(self):
        """Test claim search functionality"""
        # Create test claims
        self.conjecture.create_claim(
            content="Python is a programming language",
            confidence=0.95,
            claim_type="concept",
            tags=["python", "programming"]
        )
        
        self.conjecture.create_claim(
            content="JavaScript runs in web browsers",
            confidence=0.90,
            claim_type="concept",
            tags=["javascript", "web"]
        )
        
        # Test search
        results = self.conjecture.search_claims("python")
        self.assertGreater(len(results), 0)
        self.assertIn("python", results[0]["tags"])

    def test_recent_claims(self):
        """Test getting recent claims"""
        # Create test claims
        for i in range(5):
            self.conjecture.create_claim(
                content=f"Test claim {i+1}",
                confidence=0.8 + i * 0.02,
                claim_type="concept",
                tags=[f"tag{i+1}"]
            )
        
        # Get recent claims
        recent = self.conjecture.get_recent_claims(limit=3)
        self.assertEqual(len(recent), 3)
        
        # Check ordering (newest first)
        for i in range(len(recent) - 1):
            self.assertTrue(recent[i]["created"] >= recent[i+1]["created"])

    def test_statistics(self):
        """Test statistics generation"""
        # Create test claims
        self.conjecture.create_claim("Test claim 1", 0.8, "concept")
        self.conjecture.create_claim("Test claim 2", 0.9, "reference")
        
        stats = self.conjecture.get_statistics()
        
        self.assertIn("active_sessions", stats)
        self.assertIn("available_tools", stats)
        self.assertIn("available_skills", stats)
        self.assertIn("total_claims", stats)
        self.assertGreaterEqual(stats["total_claims"], 2)

    def test_session_cleanup(self):
        """Test session cleanup functionality"""
        # Create sessions by processing requests
        for i in range(3):
            self.conjecture.process_request(f"Test request {i+1}")
        
        self.assertEqual(len(self.conjecture.sessions), 3)
        
        # Clean up old sessions (using 0 days to clean all)
        cleaned = self.conjecture.cleanup_sessions(days_old=0)
        self.assertEqual(cleaned, 3)
        self.assertEqual(len(self.conjecture.sessions), 0)


class TestToolManager(unittest.TestCase):
    """Test the simplified Tool Manager"""

    def setUp(self):
        """Set up test environment"""
        self.tool_manager = ToolManager()

    def test_tool_definition_retrieval(self):
        """Test getting tool definitions"""
        tools = self.tool_manager.get_tool_definitions()
        
        expected_tools = ["WebSearch", "ReadFiles", "WriteCodeFile", "CreateClaim", "ClaimSupport"]
        for tool in expected_tools:
            self.assertIn(tool, tools)
            self.assertIn("description", tools[tool])
            self.assertIn("parameters", tools[tool])

    def test_web_search_tool(self):
        """Test WebSearch tool"""
        result = self.tool_manager.call_tool("WebSearch", {
            "query": "machine learning",
            "max_results": 3
        })
        
        self.assertTrue(result["success"])
        self.assertIn("results", result)
        self.assertLessEqual(len(result["results"]), 3)

    def test_write_code_file_tool(self):
        """Test WriteCodeFile tool"""
        test_content = "print('Hello, World!')"
        test_file = os.path.join(tempfile.gettempdir(), "test_output.py")
        
        result = self.tool_manager.call_tool("WriteCodeFile", {
            "file_path": test_file,
            "content": test_content
        })
        
        self.assertTrue(result["success"])
        
        # Verify file was created
        self.assertTrue(os.path.exists(test_file))
        
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)

    def test_create_claim_tool(self):
        """Test CreateClaim tool"""
        result = self.tool_manager.call_tool("CreateClaim", {
            "content": "Test claim from tool",
            "confidence": 0.85,
            "claim_type": "concept",
            "tags": ["test", "tool"]
        })
        
        self.assertTrue(result["success"])
        self.assertIn("claim_id", result)
        self.assertIn("claim", result)

    def test_tool_call_parsing(self):
        """Test tool call parsing from text"""
        text = "WebSearch(query='AI research'), CreateClaim(content='Test claim', confidence=0.8, claim_type='concept')"
        
        calls = self.tool_manager.parse_tool_calls(text)
        
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["tool"], "WebSearch")
        self.assertEqual(calls[0]["parameters"]["query"], "AI research")
        self.assertEqual(calls[1]["tool"], "CreateClaim")
        self.assertEqual(calls[1]["parameters"]["content"], "Test claim")


class TestSkillManager(unittest.TestCase):
    """Test the simplified Skill Manager"""

    def setUp(self):
        """Set up test environment"""
        self.skill_manager = SkillManager()

    def test_skill_matching(self):
        """Test skill matching for queries"""
        queries = [
            ("Research machine learning", ["research"]),
            ("Write Python code", ["code"]),
            ("Test the application", ["test"]),
            ("Evaluate performance", ["evaluate"])
        ]
        
        for query, expected_skills in queries:
            matching_skills = self.skill_manager.get_matching_skills(query)
            self.assertTrue(any(skill in matching_skills for skill in expected_skills))

    def test_skill_template_formatting(self):
        """Test skill template formatting"""
        skill_prompt = self.skill_manager.format_skill_prompt("research")
        
        self.assertIn("Skill: Research", skill_prompt)
        self.assertIn("Description:", skill_prompt)
        self.assertIn("Steps to follow:", skill_prompt)
        self.assertIn("1.", skill_prompt)  # Should have numbered steps

    def test_all_skills_available(self):
        """Test that all expected skills are available"""
        skills = self.skill_manager.get_all_skills()
        
        expected_skills = ["research", "code", "test", "evaluate"]
        for skill in expected_skills:
            self.assertIn(skill, skills)
            self.assertIn("name", skills[skill])
            self.assertIn("description", skills[skill])
            self.assertIn("steps", skills[skill])


class TestDataManager(unittest.TestCase):
    """Test the simplified Data Manager"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_path = os.path.join(self.temp_dir, "test_claims.json")
        self.data_manager = DataManager(self.test_data_path)

    def test_claim_creation_and_retrieval(self):
        """Test claim creation and retrieval"""
        claim = self.data_manager.create_claim(
            content="Test claim for data manager",
            confidence=0.85,
            claim_type="concept",
            tags=["test", "data"]
        )
        
        # Test retrieval
        retrieved = self.data_manager.get_claim(claim.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.content, claim.content)
        self.assertEqual(retrieved.confidence, claim.confidence)

    def test_search_functionality(self):
        """Test claim search"""
        # Create test claims
        self.data_manager.create_claim(
            "Python programming language",
            0.9, "concept", ["python"]
        )
        self.data_manager.create_claim(
            "JavaScript web development",
            0.85, "concept", ["javascript", "web"]
        )
        
        # Test search
        results = self.data_manager.search_claims("python")
        self.assertEqual(len(results), 1)
        self.assertIn("python", results[0].content.lower())

    def test_statistics_calculation(self):
        """Test statistics calculation"""
        # Create test claims
        self.data_manager.create_claim("Claim 1", 0.8, "concept")
        self.data_manager.create_claim("Claim 2", 0.9, "reference")
        
        stats = self.data_manager.get_statistics()
        
        self.assertEqual(stats["total_claims"], 2)
        self.assertGreater(stats["avg_confidence"], 0)
        self.assertIn("claim_types", stats)

    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.test_data_path):
            os.remove(self.test_data_path)
        os.rmdir(self.temp_dir)


def run_integration_tests():
    """Run integration tests for basic workflows"""
    print("🧪 Running Integration Tests for Simplified Conjecture")
    print("=" * 60)
    
    # Test all workflows
    test_workflow_research()
    test_workflow_code() 
    test_workflow_test()
    test_workflow_evaluate()
    
    print("🎉 All integration tests completed!")


def test_workflow_research():
    """Test research workflow"""
    print("\n🔍 Testing Research Workflow")
    print("-" * 30)
    
    cf = Conjecture()
    
    # Research request
    result = cf.process_request("Research artificial intelligence basics")
    
    assert result["success"], "Research request should succeed"
    assert result["skill_used"] == "research", "Should use research skill"
    assert len(result["tool_results"]) > 0, "Should execute tools"
    
    print("✅ Research workflow: PASSED")


def test_workflow_code():
    """Test code development workflow"""
    print("\n💻 Testing Code Development Workflow")
    print("-" * 30)
    
    cf = Conjecture()
    
    # Code development request
    result = cf.process_request("Write a simple Python function")
    
    assert result["success"], "Code request should succeed"
    assert result["skill_used"] == "code", "Should use code skill"
    assert len(result["tool_results"]) > 0, "Should execute tools"
    
    print("✅ Code development workflow: PASSED")


def test_workflow_test():
    """Test testing workflow"""
    print("\n🧪 Testing Validation Workflow") 
    print("-" * 30)
    
    cf = Conjecture()
    
    # Testing request
    result = cf.process_request("Test the application functionality")
    
    assert result["success"], "Testing request should succeed"
    assert result["skill_used"] == "test", "Should use test skill"
    assert len(result["tool_results"]) > 0, "Should execute tools"
    
    print("✅ Testing workflow: PASSED")


def test_workflow_evaluate():
    """Test evaluation workflow"""
    print("\n📊 Testing Evaluation Workflow")
    print("-" * 30)
    
    cf = Conjecture()
    
    # Evaluation request
    result = cf.process_request("Evaluate the system performance")
    
    assert result["success"], "Evaluation request should succeed"
    assert result["skill_used"] == "evaluate", "Should use evaluate skill"
    assert len(result["tool_results"]) > 0, "Should execute tools"
    
    print("✅ Evaluation workflow: PASSED")


if __name__ == "__main__":
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run integration tests
    run_integration_tests()