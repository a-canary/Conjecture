"""
Skill Claim: Tool Creation Rubric and Requirements
This file contains skill claims for creating new tools following specific standards
"""

from datetime import datetime
from typing import List
from src.core.unified_models import Claim, ClaimType


def create_tool_creation_skills() -> List[Claim]:
    """
    Create skill claims for tool creation rubric and requirements
    """
    
    skill_1 = Claim(
        id="skill_tool_creation_rubric",
        content="""
Follow this comprehensive rubric when creating new tools:

1. **Tool Structure Requirements**:
   - File must be in tools/ directory with descriptive name (snake_case)
   - Must have a clear, descriptive docstring at the top
   - All functions must have type hints and docstrings
   - Must include an examples() function that returns List[str]

2. **Function Design Standards**:
   - Functions should be pure and stateless when possible
   - Use descriptive parameter names with type hints
   - Return structured data (dictionaries) with metadata
   - Include error handling with meaningful messages

3. **Examples Function Requirements**:
   - Must return List[str] with example usage claims
   - Examples should be in format: 'functionName(params) returns outcome description'
   - Include diverse examples covering different use cases
   - Examples should help LLM understand when to use the tool

4. **Error Handling Standards**:
   - Use try-catch blocks for external dependencies
   - Return error information in structured format
   - Never let exceptions propagate to caller
   - Include error context and debugging information

5. **Security and Safety**:
   - Validate all inputs and sanitize when necessary
   - Avoid dangerous operations (file system, network) without safeguards
   - Use timeouts for external operations
   - Limit resource usage (memory, CPU, file size)

6. **Testing and Documentation**:
   - Include __main__ section with basic functionality tests
   - Test examples should demonstrate typical usage
   - Include performance considerations in docstrings
   - Document dependencies and requirements

Use writeFiles to create new tool files following this rubric exactly.
""",
        confidence=0.96,
        type=[ClaimType.SKILL],
        tags=["tool-creation", "rubric", "standards", "requirements"],
        created=datetime.utcnow()
    )
    
    skill_2 = Claim(
        id="skill_tool_examples_best_practices",
        content="""
Write effective examples() functions following these best practices:

1. **Example Format Standards**:
   - Each example should be a complete, descriptive claim
   - Format: 'functionName(param1, param2) returns description of outcome'
   - Use concrete, realistic parameter values
   - Describe both the result and its usefulness

2. **Coverage Requirements**:
   - Include examples for all major functions in the tool
   - Show different parameter combinations and options
   - Demonstrate error handling scenarios
   - Include both simple and advanced use cases

3. **Contextual Clarity**:
   - Examples should help LLM understand WHEN to use the tool
   - Show how the tool fits into larger workflows
   - Include examples that demonstrate integration with other tools
   - Explain the value proposition of each example

4. **Example Quality Standards**:
   - Examples must be syntactically correct
   - Parameter values should be realistic and valid
   - Outcomes should be accurately described
   - Avoid ambiguous or vague descriptions

5. **Integration Examples**:
   - Show how the tool works with readFiles, writeFiles, webSearch
   - Demonstrate common workflow patterns
   - Include examples that solve real problems
   - Show chaining of multiple tool calls

6. **Validation Testing**:
   - Test all examples in __main__ section
   - Verify examples actually work as described
   - Check that examples cover the tool's main capabilities
   - Ensure examples are up-to-date with current implementation

The examples() function is critical for LLM tool discovery and usage.
""",
        confidence=0.93,
        type=[ClaimType.SKILL],
        tags=["examples", "documentation", "llm-integration", "best-practices"],
        created=datetime.utcnow()
    )
    
    skill_3 = Claim(
        id="skill_tool_naming_conventions",
        content="""
Follow these naming conventions for tools and functions:

1. **File Naming Standards**:
   - Use snake_case for tool file names (e.g., web_search.py)
   - Names should be descriptive and concise
   - Avoid abbreviations unless widely understood
   - Use domain-specific terminology when appropriate

2. **Function Naming Standards**:
   - Use camelCase for function names (e.g., webSearch, readFiles)
   - Names should clearly indicate the function's purpose
   - Use action verbs for functions that perform operations
   - Be consistent with naming patterns across tools

3. **Parameter Naming Standards**:
   - Use snake_case for parameter names
   - Choose descriptive names that indicate purpose and type
   - Use standard names for common parameters (path, content, encoding)
   - Include units or constraints in names when helpful (max_files, timeout_ms)

4. **Return Value Standards**:
   - Return structured dictionaries with consistent keys
   - Include success/failure indicators (success: bool, error: str)
   - Provide metadata (size_bytes, line_count, execution_time)
   - Use consistent field names across similar tools

5. **Class and Type Naming**:
   - Use PascalCase for classes and type names
   - Use descriptive names that indicate purpose
   - Include context in type names when needed
   - Follow language-specific conventions

6. **Consistency Guidelines**:
   - Use the same naming patterns across all tools
   - Follow established conventions from existing tools
   - Consider how names will be used by the LLM
   - Test names for clarity and discoverability

Good naming makes tools more discoverable and easier to use correctly.
""",
        confidence=0.91,
        type=[ClaimType.SKILL],
        tags=["naming", "conventions", "consistency", "usability"],
        created=datetime.utcnow()
    )
    
    skill_4 = Claim(
        id="skill_tool_error_handling_patterns",
        content="""
Implement robust error handling using these patterns:

1. **Structured Error Returns**:
   - Always return dictionaries with success indicators
   - Include error messages in 'error' field
   - Provide context about what failed and why
   - Use consistent error field names across tools

2. **Error Classification**:
   - Distinguish between user errors and system errors
   - Use specific error types and messages
   - Include suggestions for fixing the error when possible
   - Log errors for debugging while providing user-friendly messages

3. **Graceful Degradation**:
   - Provide fallback behavior when possible
   - Return partial results when complete results fail
   - Offer alternative approaches in error messages
   - Maintain tool availability even with reduced functionality

4. **Input Validation**:
   - Validate all parameters before processing
   - Check for required vs optional parameters
   - Validate parameter types and ranges
   - Sanitize inputs to prevent injection attacks

5. **External Dependency Handling**:
   - Handle network timeouts and connection failures
   - Manage file system permissions and errors
   - Deal with API rate limits and service unavailability
   - Implement retry logic with exponential backoff

6. **Error Recovery Patterns**:
   - Provide clear paths for error resolution
   - Include error codes for programmatic handling
   - Suggest alternative tools or approaches
   - Document common error scenarios and solutions

Robust error handling makes tools reliable and user-friendly.
""",
        confidence=0.94,
        type=[ClaimType.SKILL],
        tags=["error-handling", "reliability", "robustness", "patterns"],
        created=datetime.utcnow()
    )
    
    skill_5 = Claim(
        id="skill_tool_integration_guidelines",
        content="""
Design tools for seamless integration with the Conjecture ecosystem:

1. **Tool Interoperability**:
   - Design tools to work well with readFiles, writeFiles, webSearch
   - Use compatible data formats and structures
   - Consider how tools will be chained together
   - Design for common workflow patterns

2. **Data Flow Standards**:
   - Use consistent data structures across tools
   - Include metadata that helps with tool chaining
   - Design output formats that can be used as input to other tools
   - Consider data transformation needs

3. **Workflow Integration**:
   - Think about how tools fit into larger processes
   - Design for common research and development workflows
   - Consider both human and LLM usage patterns
   - Plan for incremental processing

4. **Performance Considerations**:
   - Design for efficiency with large datasets
   - Implement streaming for large files when appropriate
   - Consider memory usage and CPU impact
   - Provide options for limiting resource usage

5. **Discovery and Usability**:
   - Write examples that show integration patterns
   - Use consistent parameter names across related tools
   - Design for discoverability through examples
   - Consider how LLM will find and select tools

6. **Ecosystem Evolution**:
   - Design tools to be extensible and maintainable
   - Consider future integration needs
   - Plan for backward compatibility
   - Document integration patterns and best practices

Well-integrated tools create a powerful, cohesive ecosystem.
""",
        confidence=0.92,
        type=[ClaimType.SKILL],
        tags=["integration", "ecosystem", "workflow", "interoperability"],
        created=datetime.utcnow()
    )
    
    return [skill_1, skill_2, skill_3, skill_4, skill_5]


if __name__ == "__main__":
    # Create and display the tool creation skills
    skills = create_tool_creation_skills()
    
    print("Tool Creation Rubric and Requirements - Skill Claims")
    print("=" * 55)
    
    for i, skill in enumerate(skills, 1):
        print(f"\n{i}. {skill.id}")
        print(f"   Confidence: {skill.confidence}")
        print(f"   Tags: {', '.join(skill.tags)}")
        print(f"   Content: {skill.content[:200]}...")
        print("-" * 35)