"""
Weather Example Test for Refined Architecture
Tests the complete workflow: LLM needs weather → tool creation → skill/sample → context retrieval
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

from data.data_manager import DataManager, DataConfig
from processing.tool_manager import ToolManager
from processing.tool_creator import ToolCreator
from processing.context_collector import ContextCollector
from core.refined_skill_models import SkillClaim, SampleClaim, ToolCreationClaim


class TestWeatherExample:
    """Test the complete weather example workflow."""
    
    @pytest.fixture
    async def setup_components(self):
        """Set up all components for testing."""
        # Create temporary directory for tools
        self.temp_dir = tempfile.mkdtemp()
        tools_dir = Path(self.temp_dir) / "tools"
        tools_dir.mkdir()
        
        # Create temporary database
        self.db_path = Path(self.temp_dir) / "test.db"
        
        # Initialize components
        config = DataConfig(
            sqlite_path=str(self.db_path),
            chroma_path=str(Path(self.temp_dir) / "chroma")
        )
        
        data_manager = DataManager(config, use_mock_embeddings=True)
        await data_manager.initialize()
        
        tool_manager = ToolManager(str(tools_dir))
        await tool_manager.load_all_tools()
        
        tool_creator = ToolCreator(data_manager, tool_manager)
        context_collector = ContextCollector(data_manager)
        
        yield {
            'data_manager': data_manager,
            'tool_manager': tool_manager,
            'tool_creator': tool_creator,
            'context_collector': context_collector,
            'temp_dir': self.temp_dir
        }
        
        # Cleanup
        await data_manager.close()
        shutil.rmtree(self.temp_dir)
    
    @pytest.mark.asyncio
    async def test_weather_tool_creation_workflow(self, setup_components):
        """Test the complete weather tool creation workflow."""
        components = setup_components
        data_manager = components['data_manager']
        tool_creator = components['tool_creator']
        context_collector = components['context_collector']
        
        # Step 1: LLM encounters claim needing weather information
        weather_claim = "I need to get the weather forecast for zipcode 90210 to plan my outdoor activities."
        
        # Step 2: Tool creator detects need and creates tool
        creation_claim = await tool_creator.create_tool_for_claim(
            weather_claim, 
            {'user_intent': 'weather_planning'}
        )
        
        # Verify tool creation claim was created
        assert creation_claim is not None
        assert creation_claim.tool_name == 'get_weather_by_zipcode'
        assert 'weather' in creation_claim.creation_reason.lower()
        
        # Step 3: Verify tool was loaded
        tool = components['tool_manager'].get_tool('get_weather_by_zipcode')
        assert tool is not None
        assert tool.name == 'get_weather_by_zipcode'
        
        # Step 4: Verify skill claim was created
        skills = await context_collector.collect_relevant_skills(weather_claim, {}, 5)
        assert len(skills) > 0
        
        weather_skill = None
        for skill_data in skills:
            if skill_data['skill'].tool_name == 'get_weather_by_zipcode':
                weather_skill = skill_data['skill']
                break
        
        assert weather_skill is not None
        assert weather_skill.tool_name == 'get_weather_by_zipcode'
        assert len(weather_skill.procedure_steps) > 0
        
        # Step 5: Verify sample claims were created
        samples = await context_collector.collect_relevant_samples(weather_claim, {}, 10)
        assert len(samples) > 0
        
        weather_sample = None
        for sample_data in samples:
            if sample_data['sample'].tool_name == 'get_weather_by_zipcode':
                weather_sample = sample_data['sample']
                break
        
        assert weather_sample is not None
        assert weather_sample.tool_name == 'get_weather_by_zipcode'
        assert '<invoke name="get_weather_by_zipcode">' in weather_sample.llm_call_xml
        
        # Step 6: Test context collection for future weather claims
        context_result = await context_collector.collect_context_for_claim(
            "What's the weather like in 10001?",
            {'user_intent': 'weather_inquiry'}
        )
        
        assert context_result['total_skills'] > 0
        assert context_result['total_samples'] > 0
        
        # Verify weather skill is included in context
        context_skills = [s['skill'] for s in context_result['skills']]
        weather_context_skill = next(
            (s for s in context_skills if s.tool_name == 'get_weather_by_zipcode'), 
            None
        )
        assert weather_context_skill is not None
        
        # Step 7: Test LLM context string generation
        context_string = context_collector.build_llm_context_string(context_result)
        assert 'RELEVANT SKILLS:' in context_string
        assert 'RELEVANT SAMPLES:' in context_string
        assert 'get_weather_by_zipcode' in context_string
        assert '<invoke name="get_weather_by_zipcode">' in context_string
        
        print("✅ Weather tool creation workflow test passed!")
    
    @pytest.mark.asyncio
    async def test_weather_tool_execution(self, setup_components):
        """Test actual execution of the created weather tool."""
        components = setup_components
        tool_manager = components['tool_manager']
        
        # First, create the weather tool
        tool_creator = components['tool_creator']
        creation_claim = await tool_creator.create_tool_for_claim(
            "I need weather data for zipcode 90210",
            {}
        )
        
        # Test tool execution
        tool = tool_manager.get_tool('get_weather_by_zipcode')
        assert tool is not None
        
        # Execute with valid zipcode
        result = await tool.execute({'zipcode': '90210'})
        assert result is not None
        assert 'zipcode' in result
        assert result['zipcode'] == '90210'
        assert 'temperature' in result
        assert 'conditions' in result
        
        # Execute with invalid zipcode
        with pytest.raises(ValueError):
            await tool.execute({'zipcode': 'invalid'})
        
        print("✅ Weather tool execution test passed!")
    
    @pytest.mark.asyncio
    async def test_sample_claim_xml_format(self, setup_components):
        """Test that sample claims have proper XML format."""
        components = setup_components
        tool_creator = components['tool_creator']
        context_collector = components['context_collector']
        
        # Create weather tool
        await tool_creator.create_tool_for_claim(
            "Need weather for 90210",
            {}
        )
        
        # Get samples
        samples = await context_collector.collect_relevant_samples(
            "weather inquiry",
            {},
            5
        )
        
        assert len(samples) > 0
        
        # Check XML format
        weather_sample = samples[0]['sample']
        assert weather_sample.llm_call_xml.startswith('<tool_calls>')
        assert weather_sample.llm_call_xml.endswith('</tool_calls>')
        assert '<invoke name="get_weather_by_zipcode">' in weather_sample.llm_call_xml
        assert '<parameter name="zipcode">' in weather_sample.llm_call_xml
        assert '</parameter>' in weather_sample.llm_call_xml
        assert '</invoke>' in weather_sample.llm_call_xml
        
        # Test parameter extraction
        extracted_params = weather_sample.extract_parameters_from_xml()
        assert 'zipcode' in extracted_params
        assert extracted_params['zipcode'] == '90210'
        
        print("✅ Sample claim XML format test passed!")
    
    @pytest.mark.asyncio
    async def test_skill_claim_procedure_format(self, setup_components):
        """Test that skill claims have proper procedure format."""
        components = setup_components
        tool_creator = components['tool_creator']
        
        # Create weather tool
        await tool_creator.create_tool_for_claim(
            "Need weather functionality",
            {}
        )
        
        # Get skill claims
        skill_claims = await components['data_manager'].filter_claims(
            filters=None  # Will filter by type.skill
        )
        
        weather_skills = [
            SkillClaim(**claim) for claim in skill_claims
            if 'type.skill' in claim.get('tags', []) and 'weather' in claim.get('content', '').lower()
        ]
        
        assert len(weather_skills) > 0
        
        weather_skill = weather_skills[0]
        
        # Check procedure format
        assert weather_skill.tool_name == 'get_weather_by_zipcode'
        assert len(weather_skill.procedure_steps) > 0
        
        # Check procedure steps
        for step in weather_skill.procedure_steps:
            assert step.instruction != ""
            assert step.step_number > 0
            assert step.tool_name == 'get_weather_by_zipcode'
            assert step.parameters is not None
            assert 'zipcode' in step.parameters
        
        # Check LLM context format
        context_format = weather_skill.to_llm_context()
        assert 'Procedure:' in context_format
        assert 'Step 1:' in context_format
        assert 'get_weather_by_zipcode' in context_format
        
        print("✅ Skill claim procedure format test passed!")
    
    @pytest.mark.asyncio
    async def test_context_relevance_scoring(self, setup_components):
        """Test that context relevance scoring works correctly."""
        components = setup_components
        tool_creator = components['tool_creator']
        context_collector = components['context_collector']
        
        # Create weather tool
        await tool_creator.create_tool_for_claim(
            "Weather data needed",
            {}
        )
        
        # Test relevance scoring for weather-related claims
        weather_claims = [
            "What's the weather forecast for 10001?",
            "I need to check temperature in Beverly Hills",
            "Planning outdoor activities, need weather info",
            "What's the climate like in Florida?",
            "I need to know if it will rain tomorrow"
        ]
        
        for claim in weather_claims:
            context_result = await context_collector.collect_context_for_claim(
                claim, {}, 3, 5
            )
            
            # Should find relevant skills and samples
            assert context_result['total_skills'] > 0, f"No skills found for: {claim}"
            assert context_result['total_samples'] > 0, f"No samples found for: {claim}"
            
            # Check that weather-related items are included
            skill_names = [s['skill'].tool_name for s in context_result['skills']]
            sample_tools = [s['sample'].tool_name for s in context_result['samples']]
            
            assert 'get_weather_by_zipcode' in skill_names or 'get_weather_by_zipcode' in sample_tools, \
                f"Weather tool not found for: {claim}"
        
        # Test non-weather claim (should have lower relevance)
        non_weather_claim = "I need to calculate 2+2"
        non_weather_context = await context_collector.collect_context_for_claim(
            non_weather_claim, {}, 3, 5
        )
        
        # Weather tool might still appear but with lower relevance
        weather_skills = [
            s for s in non_weather_context['skills']
            if s['skill'].tool_name == 'get_weather_by_zipcode'
        ]
        
        if weather_skills:
            # Should have lower relevance score for non-weather claim
            assert weather_skills[0]['relevance_score'] < 0.5
        
        print("✅ Context relevance scoring test passed!")
    
    @pytest.mark.asyncio
    async def test_complete_weather_workflow_end_to_end(self, setup_components):
        """Test the complete end-to-end workflow as described in requirements."""
        components = setup_components
        
        # Initial state: No weather tools
        initial_tools = components['tool_manager'].list_tools()
        weather_tools = [t for t in initial_tools if 'weather' in t.name.lower()]
        assert len(weather_tools) == 0
        
        # Step 1: LLM decides it needs weather functionality
        claim_needing_weather = "I need to get the weather forecast for zipcode 90210 to plan my picnic"
        
        # Step 2: System detects need and creates tool
        tool_creator = components['tool_creator']
        creation_claim = await tool_creator.create_tool_for_claim(
            claim_needing_weather,
            {'user_context': 'planning_picnic'}
        )
        
        assert creation_claim is not None
        assert creation_claim.tool_name == 'get_weather_by_zipcode'
        
        # Step 3: Verify tool file was created
        tool_path = Path(components['temp_dir']) / "tools" / "get_weather_by_zipcode.py"
        assert tool_path.exists()
        
        # Step 4: Verify skill claim describes procedure
        context_collector = components['context_collector']
        skills = await context_collector.collect_relevant_skills(
            claim_needing_weather, {}, 5
        )
        
        weather_skill = next(
            (s['skill'] for s in skills if s['skill'].tool_name == 'get_weather_by_zipcode'),
            None
        )
        assert weather_skill is not None
        assert len(weather_skill.procedure_steps) > 0
        
        # Step 5: Verify sample claim shows exact XML format
        samples = await context_collector.collect_relevant_samples(
            claim_needing_weather, {}, 5
        )
        
        weather_sample = next(
            (s['sample'] for s in samples if s['sample'].tool_name == 'get_weather_by_zipcode'),
            None
        )
        assert weather_sample is not None
        assert '<invoke name="get_weather_by_zipcode">' in weather_sample.llm_call_xml
        
        # Step 6: Future claim about weather retrieves context
        future_weather_claim = "Should I bring an umbrella for 10001 today?"
        future_context = await context_collector.collect_context_for_claim(
            future_weather_claim, {}, 3, 5
        )
        
        # Should retrieve weather skills and samples
        assert future_context['total_skills'] > 0
        assert future_context['total_samples'] > 0
        
        # Step 7: LLM context string contains weather information
        context_string = context_collector.build_llm_context_string(future_context)
        assert 'get_weather_by_zipcode' in context_string
        assert 'zipcode' in context_string
        
        # Step 8: LLM now knows exactly how to use the tool
        assert 'Step 1:' in context_string  # Procedure steps
        assert '<parameter name="zipcode">' in context_string  # Sample XML
        
        print("✅ Complete weather workflow end-to-end test passed!")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])