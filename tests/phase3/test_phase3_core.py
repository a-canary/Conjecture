"""
Phase 3 Tests - Basic Skills Templates Implementation
Tests the Agent Harness architecture and core functionality.
"""
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from agent.agent_harness import AgentHarness, Session, SessionState, SessionStatus
from agent.support_systems import ContextBuilder, Context
from agent.prompt_system import PromptBuilder, ResponseParser
from data.data_manager import DataManager, DataConfig


class TestPhase3Core:
    """Test core Phase 3 functionality."""
    
    @pytest.fixture
    async def setup_components(self):
        """Set up test components."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize data manager
        config = DataConfig(
            sqlite_path=str(Path(self.temp_dir) / "test.db"),
            chroma_path=str(Path(self.temp_dir) / "chroma")
        )
        
        data_manager = DataManager(config, use_mock_embeddings=True)
        await data_manager.initialize()
        
        # Initialize agent harness
        agent_harness = AgentHarness(data_manager)
        await agent_harness.initialize()
        
        yield {
            'agent_harness': agent_harness,
            'data_manager': data_manager,
            'temp_dir': self.temp_dir
        }
        
        # Cleanup
        await agent_harness.close()
        await data_manager.close()
        shutil.rmtree(self.temp_dir)
    
    @pytest.mark.asyncio
    async def test_session_management(self, setup_components):
        """Test session creation, retrieval, and cleanup."""
        agent_harness = setup_components['agent_harness']
        
        # Test session creation
        session_id = await agent_harness.create_session()
        assert session_id is not None
        assert len(session_id) > 10  # UUID length
        
        # Test session retrieval
        session = await agent_harness.get_session(session_id)
        assert session is not None
        assert session.session_id == session_id
        assert session.state.status == SessionStatus.ACTIVE
        
        # Test session listing
        sessions = agent_harness.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]['session_id'] == session_id
        
        # Test session cleanup
        cleanup_result = await agent_harness.cleanup_session(session_id)
        assert cleanup_result is True
        
        # Session should be gone
        session = await agent_harness.get_session(session_id)
        assert session is None
    
    @pytest.mark.asyncio
    async def test_context_building(self, setup_components):
        """Test context building functionality."""
        data_manager = setup_components['data_manager']
        
        # Initialize context builder
        context_builder = ContextBuilder(data_manager)
        await context_builder.initialize()
        
        # Create mock session
        session = Session(
            session_id="test-session",
            state=SessionState(session_id="test-session")
        )
        
        # Test context building
        context = await context_builder.build_context(session, "research Python weather APIs")
        
        assert context is not None
        assert context.current_focus == "research Python weather APIs"
        assert len(context.available_tools) > 0
        assert len(context.skill_templates) > 0
        
        # Should include research skill for "research" keyword
        skill_names = [skill['name'] for skill in context.skill_templates]
        assert 'Research' in skill_names
        
        # Should include all available tools
        tool_names = [tool['name'] for tool in context.available_tools]
        expected_tools = ['WebSearch', 'ReadFiles', 'WriteCodeFile', 'CreateClaim', 'ClaimSupport']
        for tool in expected_tools:
            assert tool in tool_names
    
    @pytest.mark.asyncio
    async def test_prompt_building(self, setup_components):
        """Test prompt assembly functionality."""
        # Create context
        context = Context()
        context.current_focus = "test request"
        
        # Add some tools and skills
        context.add_tool({
            'name': 'TestTool',
            'description': 'A test tool',
            'example': 'TestTool(param="value")'
        })
        
        context.add_skill_template({
            'name': 'TestSkill',
            'description': 'A test skill',
            'steps': ['Step 1', 'Step 2', 'Step 3', 'Step 4'],
            'example_usage': 'Use this skill for testing'
        })
        
        # Build prompt
        prompt_builder = PromptBuilder()
        prompt = prompt_builder.assemble_prompt(context, "test user request")
        
        assert prompt is not None
        assert len(prompt) > 0
        assert 'test user request' in prompt
        assert 'TestTool' in prompt
        assert 'TestSkill' in prompt
        assert 'AVAILABLE TOOLS' in prompt
        assert 'RELEVANT SKILLS' in prompt
    
    @pytest.mark.asyncio
    async def test_response_parsing(self, setup_components):
        """Test response parsing functionality."""
        response_parser = ResponseParser()
        
        # Test response with tool calls
        response_with_tools = """I'll help you research this topic.

<tool_calls>
  <invoke name="WebSearch">
    <parameter name="query">Python weather API</parameter>
  </invoke>
</tool_calls>

Based on my research, I claim that Python has several weather libraries available."""
        
        parsed = response_parser.parse_response(response_with_tools)
        
        assert parsed['tool_calls'] is not None
        assert len(parsed['tool_calls']) == 1
        assert parsed['tool_calls'][0]['name'] == 'WebSearch'
        assert parsed['tool_calls'][0]['parameters']['query'] == 'Python weather API'
        
        assert parsed['claims'] is not None
        assert len(parsed['claims']) > 0
        assert any('Python' in claim['content'] for claim in parsed['claims'])
        
        assert parsed['text_content'] is not None
        assert 'I\'ll help you research this topic' in parsed['text_content']
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, setup_components):
        """Test complete end-to-end workflow."""
        agent_harness = setup_components['agent_harness']
        
        # Create session
        session_id = await agent_harness.create_session()
        
        # Process research request
        response = await agent_harness.process_request(
            session_id, 
            "I need to research Python weather APIs for a project"
        )
        
        assert response['success'] is True
        assert response['session_id'] == session_id
        assert response['response'] is not None
        assert response['execution_time_ms'] > 0
        assert response['session_status'] == SessionStatus.IDLE.value
        
        # Check session state
        session = await agent_harness.get_session(session_id)
        assert session is not None
        assert len(session.interactions) == 1
        assert session.interactions[0].user_request == "I need to research Python weather APIs for a project"
        
        # Process code request
        code_response = await agent_harness.process_request(
            session_id,
            "Now write a simple Python script to get weather data"
        )
        
        assert code_response['success'] is True
        assert len(session.interactions) == 2
        
        # Process test request
        test_response = await agent_harness.process_request(
            session_id,
            "Test the weather script you created"
        )
        
        assert test_response['success'] is True
        assert len(session.interactions) == 3
        
        # Process evaluation request
        eval_response = await agent_harness.process_request(
            session_id,
            "Evaluate the claims we've made about weather APIs"
        )
        
        assert eval_response['success'] is True
        assert len(session.interactions) == 4
    
    @pytest.mark.asyncio
    async def test_skill_template_selection(self, setup_components):
        """Test that appropriate skill templates are selected."""
        data_manager = setup_components['data_manager']
        context_builder = ContextBuilder(data_manager)
        await context_builder.initialize()
        
        # Create mock session
        session = Session(
            session_id="test-session",
            state=SessionState(session_id="test-session")
        )
        
        # Test research skill selection
        context = await context_builder.build_context(session, "I need to research machine learning algorithms")
        skill_names = [skill['name'] for skill in context.skill_templates]
        assert 'Research' in skill_names
        
        # Test code skill selection
        context = await context_builder.build_context(session, "Write a Python function to calculate factorial")
        skill_names = [skill['name'] for skill in context.skill_templates]
        assert 'WriteCode' in skill_names
        
        # Test code skill selection (alternative)
        context = await context_builder.build_context(session, "Can you implement a sorting algorithm?")
        skill_names = [skill['name'] for skill in context.skill_templates]
        assert 'WriteCode' in skill_names
        
        # Test test skill selection
        context = await context_builder.build_context(session, "Test the function you just wrote")
        skill_names = [skill['name'] for skill in context.skill_templates]
        assert 'TestCode' in skill_names
        
        # Test evaluation skill selection
        context = await context_builder.build_context(session, "Evaluate the evidence for this claim")
        skill_names = [skill['name'] for skill in context.skill_templates]
        assert 'EndClaimEval' in skill_names
    
    @pytest.mark.asyncio
    async def test_error_handling(self, setup_components):
        """Test error handling in various scenarios."""
        agent_harness = setup_components['agent_harness']
        
        # Test invalid session ID
        response = await agent_harness.process_request("invalid-session", "test request")
        assert response['success'] is False
        assert 'not found or expired' in response['error']
        
        # Test session expiration (simulate by creating and waiting)
        session_id = await agent_harness.create_session()
        
        # Manually expire session
        session = await agent_harness.get_session(session_id)
        session.state.last_activity = session.state.last_activity.replace(year=2020)  # Make it old
        
        # Try to use expired session
        response = await agent_harness.process_request(session_id, "test request")
        assert response['success'] is False
        assert 'not found or expired' in response['error']
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, setup_components):
        """Test that performance benchmarks are met."""
        agent_harness = setup_components['agent_harness']
        
        # Test session creation time
        import time
        start_time = time.time()
        session_id = await agent_harness.create_session()
        creation_time = (time.time() - start_time) * 1000
        assert creation_time < 100  # Should be under 100ms
        
        # Test request processing time
        start_time = time.time()
        response = await agent_harness.process_request(session_id, "test request")
        processing_time = response['execution_time_ms']
        assert processing_time < 500  # Should be under 500ms
        
        # Test multiple concurrent sessions
        session_ids = []
        start_time = time.time()
        
        for i in range(5):
            session_id = await agent_harness.create_session()
            session_ids.append(session_id)
        
        concurrent_creation_time = (time.time() - start_time) * 1000
        assert concurrent_creation_time < 500  # Should be under 500ms for 5 sessions
        
        # Cleanup
        for session_id in session_ids:
            await agent_harness.cleanup_session(session_id)
    
    def test_skill_template_quality(self):
        """Test that skill templates meet quality standards."""
        context_builder = ContextBuilder(None)  # No data manager needed for this test
        skill_templates = context_builder._load_skill_templates()
        
        # Check that all required skills exist
        required_skills = ['research', 'write_code', 'test_code', 'end_claim_eval']
        for skill in required_skills:
            assert skill in skill_templates
        
        # Check skill template structure
        for skill_name, skill in skill_templates.items():
            assert 'name' in skill
            assert 'description' in skill
            assert 'steps' in skill
            assert len(skill['steps']) == 4  # Should have exactly 4 steps
            assert 'suggested_tools' in skill
            assert 'example_usage' in skill
            
            # Check that steps are actionable
            for step in skill['steps']:
                assert len(step) > 10  # Steps should be descriptive
                assert any(action in step.lower() for action in ['search', 'read', 'create', 'write', 'test', 'check', 'review', 'update'])
    
    def test_tool_definitions_quality(self):
        """Test that tool definitions meet quality standards."""
        context_builder = ContextBuilder(None)
        tools = context_builder._load_available_tools()
        
        # Check that all required tools exist
        required_tools = ['WebSearch', 'ReadFiles', 'WriteCodeFile', 'CreateClaim', 'ClaimSupport']
        tool_names = [tool['name'] for tool in tools]
        for tool in required_tools:
            assert tool in tool_names
        
        # Check tool structure
        for tool in tools:
            assert 'name' in tool
            assert 'description' in tool
            assert 'parameters' in tool
            assert 'example' in tool
            
            # Check that parameters are well-defined
            for param_name, param_desc in tool['parameters'].items():
                assert isinstance(param_desc, str)
                assert len(param_desc) > 5  # Should be descriptive


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])