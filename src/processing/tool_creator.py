"""
Dynamic Tool Creation System for the Conjecture skill-based agency system.
Handles LLM-driven tool discovery, creation, and validation.
"""
import asyncio
import re
import json
import tempfile
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
import hashlib

from .tool_manager import ToolManager
from .response_parser import ResponseParser
from .tool_executor import ToolExecutor, ExecutionLimits
from ..core.unified_models import Claim
from ..data.data_manager import DataManager


logger = logging.getLogger(__name__)


class ToolDiscoveryEngine:
    """Discovers tool creation methods through web search and analysis."""
    
    def __init__(self):
        self.search_patterns = {
            'weather': [
                'how to get weather forecast by zipcode python',
                'python weather api zipcode',
                'weather data python library zipcode'
            ],
            'calculator': [
                'python calculator function basic operations',
                'how to create calculator in python',
                'python math operations function'
            ],
            'search': [
                'python search function text',
                'how to implement search in python',
                'python text search algorithm'
            ]
        }
    
    async def discover_tool_need(self, claim_content: str, context: Dict[str, Any]) -> Optional[str]:
        """
        Analyze claim content to determine if a new tool is needed.
        
        Args:
            claim_content: Content of the claim being evaluated
            context: Additional context for analysis
            
        Returns:
            Tool need description or None if no tool needed
        """
        # Look for patterns that indicate tool needs
        tool_need_patterns = [
            r'need to.*(?:get|fetch|calculate|compute|search|find)',
            r'how to.*(?:weather|calculate|search|lookup)',
            r'want to.*(?:get|fetch|calculate|compute)',
            r'require.*(?:function|method|tool|utility)',
            r'looking for.*way to.*(?:get|fetch|calculate)'
        ]
        
        for pattern in tool_need_patterns:
            if re.search(pattern, claim_content, re.IGNORECASE):
                # Extract the specific need
                match = re.search(pattern, claim_content, re.IGNORECASE)
                if match:
                    return match.group(0)
        
        return None
    
    async def suggest_search_queries(self, tool_need: str) -> List[str]:
        """
        Suggest search queries for discovering tool creation methods.
        
        Args:
            tool_need: Description of the tool need
            
        Returns:
            List of search query suggestions
        """
        # Extract key terms from the need
        key_terms = []
        
        # Common tool categories
        categories = {
            'weather': ['weather', 'forecast', 'temperature', 'zipcode', 'climate'],
            'calculator': ['calculate', 'compute', 'math', 'arithmetic', 'operation'],
            'search': ['search', 'find', 'lookup', 'query', 'filter'],
            'data': ['data', 'parse', 'process', 'transform', 'convert'],
            'web': ['web', 'http', 'api', 'request', 'fetch'],
            'file': ['file', 'read', 'write', 'save', 'load']
        }
        
        need_lower = tool_need.lower()
        
        for category, terms in categories.items():
            if any(term in need_lower for term in terms):
                key_terms.append(category)
        
        # Generate queries based on identified categories
        queries = []
        
        for category in key_terms:
            if category in self.search_patterns:
                queries.extend(self.search_patterns[category])
        
        # Generate generic queries if no specific category found
        if not queries:
            generic_queries = [
                f"python {tool_need} function",
                f"how to implement {tool_need} in python",
                f"python library for {tool_need}",
                f"{tool_need} python code example"
            ]
            queries.extend(generic_queries)
        
        return list(set(queries))  # Remove duplicates
    
    async def analyze_search_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze search results to find suitable tool creation methods.
        
        Args:
            search_results: List of search result dictionaries
            
        Returns:
            List of analyzed methods with relevance scores
        """
        analyzed_methods = []
        
        for result in search_results:
            method = {
                'title': result.get('title', ''),
                'url': result.get('url', ''),
                'snippet': result.get('snippet', ''),
                'relevance_score': 0.0,
                'code_indicators': [],
                'complexity': 'unknown'
            }
            
            # Look for code indicators
            code_patterns = [
                r'```python',
                r'def ',
                r'import ',
                r'from ',
                r'function',
                r'example',
                r'tutorial',
                r'code'
            ]
            
            for pattern in code_patterns:
                if re.search(pattern, method['snippet'], re.IGNORECASE):
                    method['code_indicators'].append(pattern)
                    method['relevance_score'] += 0.2
            
            # Look for simplicity indicators
            simple_patterns = [
                r'simple',
                r'basic',
                r'easy',
                r'quick',
                r'beginner'
            ]
            
            for pattern in simple_patterns:
                if re.search(pattern, method['snippet'], re.IGNORECASE):
                    method['relevance_score'] += 0.1
                    method['complexity'] = 'simple'
            
            # Look for complexity indicators
            complex_patterns = [
                r'advanced',
                r'complex',
                r'difficult',
                r'expert'
            ]
            
            for pattern in complex_patterns:
                if re.search(pattern, method['snippet'], re.IGNORECASE):
                    method['relevance_score'] -= 0.1
                    method['complexity'] = 'complex'
            
            # Cap relevance score
            method['relevance_score'] = min(1.0, max(0.0, method['relevance_score']))
            
            if method['relevance_score'] > 0.3:  # Only include relevant results
                analyzed_methods.append(method)
        
        # Sort by relevance score
        analyzed_methods.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return analyzed_methods


class ToolCodeGenerator:
    """Generates Python code for tools based on discovered methods."""
    
    def __init__(self):
        self.code_templates = {
            'weather': '''
def get_weather_by_zipcode(zipcode: str) -> dict:
    """
    Get weather information for a given zipcode.
    
    Args:
        zipcode: 5-digit zipcode string
        
    Returns:
        Dictionary with weather information including temperature, conditions, etc.
    """
    import json
    import urllib.request
    import urllib.parse
    
    # Validate zipcode format
    if not zipcode.isdigit() or len(zipcode) != 5:
        return {{"error": "Invalid zipcode format. Use 5-digit zipcode."}}
    
    try:
        # Use a free weather API (you may need to sign up for API key)
        # For demo purposes, returning mock data
        mock_weather_data = {{
            "zipcode": zipcode,
            "temperature": 72,
            "conditions": "Sunny",
            "humidity": 45,
            "wind_speed": 10,
            "location": "Unknown"
        }}
        
        return mock_weather_data
        
    except Exception as e:
        return {{"error": f"Weather data unavailable: {{str(e)}}"}}
''',
            'calculator': '''
def calculate(expression: str) -> float:
    """
    Safely evaluate mathematical expressions.
    
    Args:
        expression: Mathematical expression as string
        
    Returns:
        Result of calculation
    """
    import ast
    import operator
    
    # Define safe operators
    operators = {{
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg
    }}
    
    def eval_node(node):
        if isinstance(node, ast.Num):  # For Python < 3.8
            return node.n
        elif isinstance(node, ast.Constant):  # For Python >= 3.8
            return node.value
        elif isinstance(node, ast.BinOp):
            return operators[type(node.op)](eval_node(node.left), eval_node(node.right))
        elif isinstance(node, ast.UnaryOp):
            return operators[type(node.op)](eval_node(node.operand))
        else:
            raise ValueError(f"Unsupported operation: {{type(node).__name__}}")
    
    try:
        # Parse and evaluate the expression
        parsed = ast.parse(expression, mode='eval')
        result = eval_node(parsed.body)
        return result
    except Exception as e:
        raise ValueError(f"Invalid expression: {{str(e)}}")
''',
            'search': '''
def search_text(text: str, query: str, case_sensitive: bool = False) -> list:
    """
    Search for query string in text.
    
    Args:
        text: Text to search in
        query: Query string to find
        case_sensitive: Whether search should be case sensitive
        
    Returns:
        List of match positions
    """
    if not case_sensitive:
        text = text.lower()
        query = query.lower()
    
    matches = []
    start = 0
    
    while True:
        pos = text.find(query, start)
        if pos == -1:
            break
        matches.append(pos)
        start = pos + 1
    
    return matches
'''
        }
    
    async def generate_tool_code(self, tool_name: str, method_info: Dict[str, Any]) -> str:
        """
        Generate Python code for a tool based on discovered method.
        
        Args:
            tool_name: Name of the tool to create
            method_info: Information about the discovered method
            
        Returns:
            Generated Python code
        """
        # Try to use template if available
        tool_category = self._categorize_tool(tool_name, method_info)
        
        if tool_category in self.code_templates:
            template = self.code_templates[tool_category]
            return template.format(tool_name=tool_name)
        
        # Generate generic code
        return await self._generate_generic_code(tool_name, method_info)
    
    def _categorize_tool(self, tool_name: str, method_info: Dict[str, Any]) -> str:
        """Categorize tool based on name and method info."""
        name_lower = tool_name.lower()
        snippet_lower = method_info.get('snippet', '').lower()
        
        categories = {
            'weather': ['weather', 'forecast', 'temperature', 'zipcode'],
            'calculator': ['calculate', 'math', 'compute', 'arithmetic'],
            'search': ['search', 'find', 'lookup', 'query']
        }
        
        for category, keywords in categories.items():
            if any(keyword in name_lower for keyword in keywords):
                return category
            if any(keyword in snippet_lower for keyword in keywords):
                return category
        
        return 'generic'
    
    async def _generate_generic_code(self, tool_name: str, method_info: Dict[str, Any]) -> str:
        """Generate generic tool code when no template is available."""
        return f'''
def {tool_name}(*args, **kwargs):
    """
    Auto-generated tool: {tool_name}
    
    This tool was automatically created based on discovered methods.
    You should customize this implementation based on your specific needs.
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Result of the tool operation
    """
    # TODO: Implement the actual tool logic
    # This is a placeholder implementation
    
    # For now, just return the arguments for testing
    if args:
        return {{"args": args, "kwargs": kwargs}}
    elif kwargs:
        return {{"kwargs": kwargs}}
    else:
        return {{"message": "Tool {tool_name} called with no arguments"}}
'''


class ToolCreator:
    """
    Coordinates the complete tool creation workflow:
    1. Discover tool need
    2. Search for methods
    3. Generate code
    4. Create tool file
    5. Create skill and sample claims
    """
    
    def __init__(self, data_manager: DataManager, tool_manager: ToolManager):
        self.data_manager = data_manager
        self.tool_manager = tool_manager
        self.discovery_engine = ToolDiscoveryEngine()
        self.code_generator = ToolCodeGenerator()
        self.response_parser = ResponseParser()
    
    async def create_tool_for_claim(self, claim_content: str, 
                                  context: Dict[str, Any]) -> Optional[Claim]:
        """
        Complete workflow to create a tool for a specific claim need.
        
        Args:
            claim_content: Content that indicates tool need
            context: Additional context for tool creation
            
        Returns:
            Claim if successful, None otherwise
        """
        try:
            # Step 1: Discover tool need
            tool_need = await self.discovery_engine.discover_tool_need(claim_content, context)
            if not tool_need:
                logger.info("No tool need detected in claim")
                return None
            
            logger.info(f"Tool need detected: {tool_need}")
            
            # Step 2: Suggest search queries
            search_queries = await self.discovery_engine.suggest_search_queries(tool_need)
            logger.info(f"Suggested search queries: {search_queries}")
            
            # Step 3: Mock search results (in real implementation, would use actual search)
            search_results = await self._mock_web_search(search_queries)
            
            # Step 4: Analyze search results
            analyzed_methods = await self.discovery_engine.analyze_search_results(search_results)
            if not analyzed_methods:
                logger.warning("No suitable methods found in search results")
                return None
            
            best_method = analyzed_methods[0]
            logger.info(f"Best method found: {best_method['title']}")
            
            # Step 5: Generate tool code
            tool_name = self._generate_tool_name(tool_need)
            tool_code = await self.code_generator.generate_tool_code(tool_name, best_method)
            
            # Step 6: Create tool file
            tool_file_path = await self.tool_manager.create_tool_file(
                tool_name, tool_code, f"Auto-generated tool for: {tool_need}"
            )
            
            if not tool_file_path:
                logger.error("Failed to create tool file")
                return None
            
            # Step 7: Load the tool
            created_tool = await self.tool_manager.load_tool_from_file(tool_file_path)
            if not created_tool:
                logger.error("Failed to load created tool")
                return None
            
            # Step 8: Create tool creation claim
            creation_claim = Claim(
                content=f"Created tool '{tool_name}' to handle: {tool_need}",
                tool_name=tool_name,
                creation_method="llm_discovered",
                websearch_query=search_queries[0] if search_queries else None,
                discovery_source=best_method.get('url', ''),
                tool_code=tool_code,
                tool_file_path=tool_file_path,
                creation_reason=tool_need,
                validation_status="created",
                confidence=0.8,
                tags=['type.tool_creation', 'auto_generated'],
                created_by='tool_creator'
            )
            
            # Save creation claim
            await self.data_manager.create_claim(
                content=creation_claim.content,
                confidence=creation_claim.confidence,
                tags=creation_claim.tags,
                created_by=creation_claim.created_by
            )
            
            # Step 9: Create skill claim
            await self._create_skill_claim_for_tool(created_tool, tool_need)
            
            # Step 10: Create sample claims
            await self._create_sample_claims_for_tool(created_tool)
            
            logger.info(f"Successfully created tool: {tool_name}")
            return creation_claim
            
        except Exception as e:
            logger.error(f"Tool creation failed: {e}")
            return None
    
    async def _mock_web_search(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Mock web search for demonstration. In real implementation, use actual search API."""
        mock_results = []
        
        for query in queries:
            # Generate mock results based on query
            if 'weather' in query.lower():
                mock_results.extend([
                    {
                        'title': 'Python Weather API Tutorial',
                        'url': 'https://example.com/weather-tutorial',
                        'snippet': 'Learn how to create a Python function to get weather data by zipcode using simple API calls.'
                    },
                    {
                        'title': 'Building a Weather App in Python',
                        'url': 'https://example.com/weather-app',
                        'snippet': 'Complete guide to building weather applications with Python, including zipcode validation and error handling.'
                    }
                ])
            elif 'calculator' in query.lower():
                mock_results.extend([
                    {
                        'title': 'Safe Math Expression Evaluation in Python',
                        'url': 'https://example.com/safe-math',
                        'snippet': 'How to safely evaluate mathematical expressions in Python using ast module for security.'
                    }
                ])
            else:
                mock_results.append({
                    'title': f'Python {query} Example',
                    'url': 'https://example.com/generic',
                    'snippet': f'Example implementation of {query} in Python with proper error handling.'
                })
        
        return mock_results
    
    def _generate_tool_name(self, tool_need: str) -> str:
        """Generate a suitable tool name from the need description."""
        # Extract key terms and create name
        need_lower = tool_need.lower()
        
        if 'weather' in need_lower:
            return 'get_weather_by_zipcode'
        elif 'calculate' in need_lower or 'math' in need_lower:
            return 'calculator'
        elif 'search' in need_lower:
            return 'search_text'
        else:
            # Generate generic name
            words = re.findall(r'\b\w+\b', tool_need)
            if words:
                return f"{'_'.join(words[:3])}_tool"
            return 'generic_tool'
    
    async def _create_skill_claim_for_tool(self, tool, tool_need: str) -> None:
        """Create a skill claim that describes how to use the tool."""
        skill_claim = Claim(
            content=f"To handle {tool_need}, use the {tool.name} tool with proper parameters and error handling.",
            tool_name=tool.name,
            confidence=0.8,
            tags=['type.concept', 'auto_generated'],
            created_by='tool_creator'
        )
        
        # Add procedure steps based on tool parameters
        if tool.parameters:
            for param_name, param_info in tool.parameters.items():
                instruction = f"Provide {param_name} parameter"
                if param_info.get('type_hint') == 'str':
                    instruction += " as a string"
                elif param_info.get('type_hint') == 'int':
                    instruction += " as an integer"
                
                skill_claim.add_procedure_step(
                    instruction=instruction,
                    tool_name=tool.name,
                    parameters={param_name: f"<{param_name}>"}
                )
        
        # Save skill claim
        await self.data_manager.create_claim(
            content=skill_claim.content,
            confidence=skill_claim.confidence,
            tags=skill_claim.tags,
            created_by=skill_claim.created_by
        )
    
    async def _create_sample_claims_for_tool(self, tool) -> None:
        """Create sample claims showing how to call the tool."""
        # Create successful sample
        if tool.parameters:
            # Generate sample parameters
            sample_params = {}
            for param_name, param_info in tool.parameters.items():
                if param_info.get('type_hint') == 'str':
                    if 'zip' in param_name.lower():
                        sample_params[param_name] = "90210"
                    else:
                        sample_params[param_name] = "example"
                elif param_info.get('type_hint') == 'int':
                    sample_params[param_name] = 42
                else:
                    sample_params[param_name] = "example_value"
            
            # Create XML call
            xml_params = []
            for param_name, param_value in sample_params.items():
                xml_params.append(f'<parameter name="{param_name}">{param_value}</parameter>')
            
            llm_call_xml = f'''<tool_calls>
  <invoke name="{tool.name}">
    {"".join(xml_params)}
  </invoke>
</tool_calls>'''
            
            # Try to execute the tool for real response
            try:
                tool_response = await tool.execute(sample_params)
                is_success = True
                error_message = None
            except Exception as e:
                tool_response = None
                is_success = False
                error_message = str(e)
            
            # Create sample claim
            sample_claim = Claim(
                content=f"Sample call to {tool.name} tool with parameters: {sample_params}",
                tool_name=tool.name,
                llm_call_xml=llm_call_xml,
                tool_response=tool_response,
                is_success=is_success,
                error_message=error_message,
                sample_quality=0.8 if is_success else 0.3,
                confidence=0.8,
                tags=['type.sample', 'auto_generated'],
                created_by='tool_creator'
            )
            
            # Save sample claim
            await self.data_manager.create_claim(
                content=sample_claim.content,
                confidence=sample_claim.confidence,
                tags=sample_claim.tags,
                created_by=sample_claim.created_by
            )
    
    async def get_tool_creation_stats(self) -> Dict[str, Any]:
        """Get statistics about tool creation activities."""
        try:
            # Get tool creation claims
            creation_claims = await self.data_manager.filter_claims(
                filters=None  # Will be implemented to filter by tags
            )
            
            creation_count = 0
            successful_creations = 0
            
            for claim_dict in creation_claims:
                if 'type.tool_creation' in claim_dict.get('tags', []):
                    creation_count += 1
                    if claim_dict.get('validation_status') == 'created':
                        successful_creations += 1
            
            return {
                'total_tools_created': creation_count,
                'successful_creations': successful_creations,
                'success_rate': successful_creations / creation_count if creation_count > 0 else 0.0,
                'loaded_tools': len(self.tool_manager.loaded_tools)
            }
            
        except Exception as e:
            logger.error(f"Error getting tool creation stats: {e}")
            return {
                'total_tools_created': 0,
                'successful_creations': 0,
                'success_rate': 0.0,
                'loaded_tools': len(self.tool_manager.loaded_tools)
            }