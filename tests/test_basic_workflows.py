"""
Basic workflow tests for Simplified Conjecture
Focus on testing the core functionality
"""

import unittest
from src.tools import ToolManager
from src.skills import SkillManager


class TestBasicWorkflows(unittest.TestCase):
    """Test basic workflows without async complexity"""

    def test_research_workflow(self):
        """Test research workflow skills and tools"""
        print("\n🔍 Testing Research Workflow")
        print("-" * 30)
        
        skill_manager = SkillManager()
        tool_manager = ToolManager()
        
        # Test skill matching
        skills = skill_manager.get_matching_skills("Research machine learning basics")
        self.assertIn("research", skills)
        
        # Test tool availability
        tools = tool_manager.get_tool_definitions()
        self.assertIn("WebSearch", tools)
        self.assertIn("CreateClaim", tools)
        
        # Test tool execution
        web_search_result = tool_manager.call_tool("WebSearch", {
            "query": "machine learning",
            "max_results": 3
        })
        self.assertTrue(web_search_result["success"])
        
        claim_result = tool_manager.call_tool("CreateClaim", {
            "content": "Machine learning requires data for training",
            "confidence": 0.85,
            "claim_type": "concept",
            "tags": ["ml", "data"]
        })
        self.assertTrue(claim_result["success"])
        
        print("✅ Research workflow: PASSED")

    def test_code_workflow(self):
        """Test code development workflow"""
        print("\n💻 Testing Code Development Workflow")
        print("-" * 30)
        
        skill_manager = SkillManager()
        tool_manager = ToolManager()
        
        # Test skill matching
        skills = skill_manager.get_matching_skills("Write Python code")
        self.assertIn("code", skills)
        
        # Test tool execution for coding
        write_result = tool_manager.call_tool("WriteCodeFile", {
            "file_path": "test_output.py",
            "content": "print('Hello from Conjecture!')\ndef test():\n    return 'success'"
        })
        self.assertTrue(write_result["success"])
        
        # Test file reading
        read_result = tool_manager.call_tool("ReadFiles", {
            "files": ["test_output.py"]
        })
        self.assertTrue(read_result["success"])
        
        print("✅ Code development workflow: PASSED")

    def test_test_workflow(self):
        """Test testing workflow"""
        print("\n🧪 Testing Validation Workflow")
        print("-" * 30)
        
        skill_manager = SkillManager()
        tool_manager = ToolManager()
        
        # Test skill matching
        skills = skill_manager.get_matching_skills("Test the application")
        self.assertIn("test", skills)
        
        # Test tool execution for testing
        test_code = """
def test_sample():
    assert True == True
    return "pass"
"""
        
        write_test_result = tool_manager.call_tool("WriteCodeFile", {
            "file_path": "test_sample.py", 
            "content": test_code
        })
        self.assertTrue(write_test_result["success"])
        
        # Create claims about testing
        claim_result = tool_manager.call_tool("CreateClaim", {
            "content": "Tests verify system functionality",
            "confidence": 0.90,
            "claim_type": "reference",
            "tags": ["testing", "validation"]
        })
        self.assertTrue(claim_result["success"])
        
        print("✅ Testing workflow: PASSED")

    def test_evaluation_workflow(self):
        """Test evaluation workflow"""
        print("\n📊 Testing Evaluation Workflow")
        print("-" * 30)
        
        skill_manager = SkillManager()
        tool_manager = ToolManager()
        
        # Test skill matching
        skills = skill_manager.get_matching_skills("Evaluate system performance")
        self.assertIn("evaluate", skills)
        
        # Test claim support tool
        support_result = tool_manager.call_tool("ClaimSupport", {
            "claim_id": "test_claim_001",
            "max_results": 3
        })
        self.assertTrue(support_result["success"])
        
        # Create evaluation claims
        eval_result = tool_manager.call_tool("CreateClaim", {
            "content": "System evaluation shows 85% efficiency",
            "confidence": 0.85,
            "claim_type": "reference",
            "tags": ["evaluation", "performance"]
        })
        self.assertTrue(eval_result["success"])
        
        print("✅ Evaluation workflow: PASSED")

    def test_tool_call_parsing(self):
        """Test tool call parsing from text responses"""
        print("\n🔧 Testing Tool Call Parsing")
        print("-" * 30)
        
        tool_manager = ToolManager()
        
        # Test various tool call formats
        test_texts = [
            "WebSearch(query='AI research'), CreateClaim(content='Test claim', confidence=0.8, claim_type='concept')",
            "ReadFiles(files=['file1.txt', 'file2.txt'])",
            "WriteCodeFile(file_path='test.py', content='print(\"hello\")')"
        ]
        
        for text in test_texts:
            calls = tool_manager.parse_tool_calls(text)
            self.assertGreater(len(calls), 0)
            
        print("✅ Tool call parsing: PASSED")

    def test_skill_templates(self):
        """Test skill template generation"""
        print("\n📋 Testing Skill Templates")
        print("-" * 30)
        
        skill_manager = SkillManager()
        
        # Test all skill templates
        all_skills = skill_manager.get_all_skills()
        expected_skills = ["research", "code", "test", "evaluate"]
        
        for skill_name in expected_skills:
            self.assertIn(skill_name, all_skills)
            skill = all_skills[skill_name]
            
            # Check required fields
            self.assertIn("name", skill)
            self.assertIn("description", skill)
            self.assertIn("steps", skill)
            self.assertEqual(len(skill["steps"]), 4)  # 4-step process
            
            # Test prompt formatting
            prompt = skill_manager.format_skill_prompt(skill_name)
            self.assertIn("Skill:", prompt)
            self.assertIn("Description:", prompt)
            self.assertIn("Steps", prompt)
        
        print("✅ Skill templates: PASSED")


def run_basic_workflow_tests():
    """Run all basic workflow tests"""
    print("🧪 Running Basic Workflow Tests")
    print("=" * 50)
    
    suite = unittest.TestSuite()
    
    # Add test methods
    suite.addTest(TestBasicWorkflows('test_research_workflow'))
    suite.addTest(TestBasicWorkflows('test_code_workflow'))
    suite.addTest(TestBasicWorkflows('test_test_workflow'))
    suite.addTest(TestBasicWorkflows('test_evaluation_workflow'))
    suite.addTest(TestBasicWorkflows('test_tool_call_parsing'))
    suite.addTest(TestBasicWorkflows('test_skill_templates'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_basic_workflow_tests()
    if success:
        print("\n🎉 All basic workflow tests passed!")
    else:
        print("\n❌ Some tests failed!")