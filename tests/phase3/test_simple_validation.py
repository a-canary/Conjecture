"""
Simple Phase 3 Validation Test
Tests the core concepts without complex imports.
"""
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def test_skill_templates():
    """Test that skill templates are properly defined."""
    print("Testing Skill Templates...")
    
    # Import context builder
    from agent.support_systems import ContextBuilder
    
    # Create context builder (no data manager needed for templates)
    context_builder = ContextBuilder(None)
    skill_templates = context_builder._load_skill_templates()
    
    # Check required skills
    required_skills = ['research', 'write_code', 'test_code', 'end_claim_eval']
    for skill in required_skills:
        assert skill in skill_templates, f"Missing skill: {skill}"
        print(f"✅ Found skill: {skill}")
    
    # Check skill structure
    for skill_name, skill in skill_templates.items():
        assert 'name' in skill, f"Skill {skill_name} missing name"
        assert 'description' in skill, f"Skill {skill_name} missing description"
        assert 'steps' in skill, f"Skill {skill_name} missing steps"
        assert len(skill['steps']) == 4, f"Skill {skill_name} should have 4 steps, has {len(skill['steps'])}"
        assert 'suggested_tools' in skill, f"Skill {skill_name} missing suggested_tools"
        assert 'example_usage' in skill, f"Skill {skill_name} missing example_usage"
        
        # Check step quality
        for i, step in enumerate(skill['steps'], 1):
            assert len(step) > 10, f"Skill {skill_name} step {i} too short: {step}"
            print(f"✅ {skill_name} step {i}: {step[:50]}...")
    
    print("✅ All skill templates passed validation")


def test_tool_definitions():
    """Test that tool definitions are properly defined."""
    print("\nTesting Tool Definitions...")
    
    from agent.support_systems import ContextBuilder
    
    context_builder = ContextBuilder(None)
    tools = context_builder._load_available_tools()
    
    # Check required tools
    required_tools = ['WebSearch', 'ReadFiles', 'WriteCodeFile', 'CreateClaim', 'ClaimSupport']
    tool_names = [tool['name'] for tool in tools]
    for tool in required_tools:
        assert tool in tool_names, f"Missing tool: {tool}"
        print(f"✅ Found tool: {tool}")
    
    # Check tool structure
    for tool in tools:
        assert 'name' in tool, f"Tool {tool.get('name', 'unknown')} missing name"
        assert 'description' in tool, f"Tool {tool.get('name', 'unknown')} missing description"
        assert 'parameters' in tool, f"Tool {tool.get('name', 'unknown')} missing parameters"
        assert 'example' in tool, f"Tool {tool.get('name', 'unknown')} missing example"
        
        # Check parameter definitions
        for param_name, param_desc in tool['parameters'].items():
            assert isinstance(param_desc, str), f"Tool {tool['name']} parameter {param_name} should be string"
            assert len(param_desc) > 5, f"Tool {tool['name']} parameter {param_name} description too short"
        
        print(f"✅ Tool {tool['name']}: {tool['description']}")
    
    print("✅ All tool definitions passed validation")


def test_prompt_building():
    """Test prompt building functionality."""
    print("\nTesting Prompt Building...")
    
    from agent.support_systems import Context
    from agent.prompt_system import PromptBuilder
    
    # Create context
    context = Context()
    context.current_focus = "test request"
    
    # Add tools and skills
    context.add_tool({
        'name': 'TestTool',
        'description': 'A test tool for validation',
        'parameters': {'param': 'string - test parameter'},
        'example': 'TestTool(param="value")'
    })
    
    context.add_skill_template({
        'name': 'TestSkill',
        'description': 'A test skill for validation',
        'steps': ['Step 1: Do this', 'Step 2: Do that', 'Step 3: Do another', 'Step 4: Finish'],
        'suggested_tools': ['TestTool'],
        'example_usage': 'Use TestSkill for testing purposes'
    })
    
    # Build prompt
    prompt_builder = PromptBuilder()
    prompt = prompt_builder.assemble_prompt(context, "test user request")
    
    assert prompt is not None, "Prompt should not be None"
    assert len(prompt) > 0, "Prompt should not be empty"
    assert 'test user request' in prompt, "Prompt should contain user request"
    assert 'TestTool' in prompt, "Prompt should contain tool information"
    assert 'TestSkill' in prompt, "Prompt should contain skill information"
    assert 'AVAILABLE TOOLS' in prompt, "Prompt should have tools section"
    assert 'RELEVANT SKILLS' in prompt, "Prompt should have skills section"
    
    print(f"✅ Prompt built successfully ({len(prompt)} characters)")
    print(f"✅ Contains tools section: {'AVAILABLE TOOLS' in prompt}")
    print(f"✅ Contains skills section: {'RELEVANT SKILLS' in prompt}")
    print(f"✅ Contains user request: {'test user request' in prompt}")


def test_response_parsing():
    """Test response parsing functionality."""
    print("\nTesting Response Parsing...")
    
    from agent.prompt_system import ResponseParser
    
    response_parser = ResponseParser()
    
    # Test response with tool calls
    response_with_tools = """I'll help you research this topic.

<tool_calls>
  <invoke name="WebSearch">
    <parameter name="query">Python weather API</parameter>
  </invoke>
</tool_calls>

Based on my research, I found several weather libraries available."""
    
    parsed = response_parser.parse_response(response_with_tools)
    
    assert parsed is not None, "Parsed response should not be None"
    assert 'tool_calls' in parsed, "Should have tool_calls key"
    assert 'claims' in parsed, "Should have claims key"
    assert 'text_content' in parsed, "Should have text_content key"
    assert 'raw_response' in parsed, "Should have raw_response key"
    
    # Check tool calls
    assert len(parsed['tool_calls']) == 1, "Should have one tool call"
    assert parsed['tool_calls'][0]['name'] == 'WebSearch', "Tool call should be WebSearch"
    assert parsed['tool_calls'][0]['parameters']['query'] == 'Python weather API', "Parameter should match"
    
    # Check claims
    assert len(parsed['claims']) > 0, "Should extract at least one claim"
    
    # Check text content
    assert 'I\'ll help you research this topic' in parsed['text_content'], "Should contain text content"
    assert '<tool_calls>' not in parsed['text_content'], "Should not contain tool calls in text content"
    
    print(f"✅ Parsed {len(parsed['tool_calls'])} tool calls")
    print(f"✅ Extracted {len(parsed['claims'])} claims")
    print(f"✅ Text content length: {len(parsed['text_content'])} characters")


def test_agent_harness_concepts():
    """Test agent harness core concepts."""
    print("\nTesting Agent Harness Concepts...")
    
    from agent.agent_harness import SessionState, SessionStatus, Interaction
    from datetime import datetime
    
    # Test session state
    session_state = SessionState(session_id="test-session")
    assert session_state.session_id == "test-session"
    assert session_state.status == SessionStatus.ACTIVE
    assert session_state.error_count == 0
    assert session_state.created_at is not None
    
    print("✅ SessionState created successfully")
    
    # Test interaction
    interaction = Interaction(
        timestamp=datetime.utcnow(),
        user_request="test request",
        llm_response="test response"
    )
    
    assert interaction.user_request == "test request"
    assert interaction.llm_response == "test response"
    assert interaction.execution_time_ms == 0  # Default value
    assert interaction.error is None
    
    print("✅ Interaction created successfully")
    
    # Test session status enum
    assert SessionStatus.ACTIVE.value == "active"
    assert SessionStatus.IDLE.value == "idle"
    assert SessionStatus.ERROR.value == "error"
    assert SessionStatus.TERMINATED.value == "terminated"
    
    print("✅ SessionStatus enum working correctly")


def test_context_functionality():
    """Test context building and management."""
    print("\nTesting Context Functionality...")
    
    from agent.support_systems import Context
    from core.refined_skill_models import Claim
    
    # Create context
    context = Context()
    assert context.relevant_claims == []
    assert context.skill_templates == []
    assert context.available_tools == []
    assert context.session_history == []
    assert context.current_focus is None
    
    print("✅ Empty context created successfully")
    
    # Add claim
    claim = Claim(
        content="Test claim for validation",
        confidence=0.8,
        tags=["test"],
        created_by="system"
    )
    context.add_claim(claim)
    assert len(context.relevant_claims) == 1
    assert context.relevant_claims[0].content == "Test claim for validation"
    
    print("✅ Claim added to context successfully")
    
    # Add skill template
    skill = {
        'name': 'TestSkill',
        'description': 'Test description',
        'steps': ['Step 1', 'Step 2', 'Step 3', 'Step 4'],
        'suggested_tools': ['TestTool'],
        'example_usage': 'Test usage example'
    }
    context.add_skill_template(skill)
    assert len(context.skill_templates) == 1
    assert context.skill_templates[0]['name'] == 'TestSkill'
    
    print("✅ Skill template added to context successfully")
    
    # Add tool
    tool = {
        'name': 'TestTool',
        'description': 'Test tool description',
        'parameters': {'param': 'string - test parameter'},
        'example': 'TestTool(param="value")'
    }
    context.add_tool(tool)
    assert len(context.available_tools) == 1
    assert context.available_tools[0]['name'] == 'TestTool'
    
    print("✅ Tool added to context successfully")
    
    # Test size estimation
    size = context.estimate_size()
    assert size > 0, "Context size should be greater than 0"
    print(f"✅ Context size estimated: {size} tokens")


def main():
    """Run all Phase 3 validation tests."""
    print("Phase 3 Validation Tests")
    print("=" * 50)
    
    try:
        test_skill_templates()
        test_tool_definitions()
        test_prompt_building()
        test_response_parsing()
        test_agent_harness_concepts()
        test_context_functionality()
        
        print("\n" + "=" * 50)
        print("🎉 ALL PHASE 3 TESTS PASSED!")
        print("\n✅ Agent Harness Architecture: VALID")
        print("✅ Support Systems: VALID")
        print("✅ Prompt System: VALID")
        print("✅ Core Components: VALID")
        print("✅ Separation of Concerns: VALID")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)