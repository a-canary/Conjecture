"""
Skill Claim: Research Coding Projects Methodology
This file contains skill claims for systematic research of coding projects
"""

from datetime import datetime
from typing import List
from src.core.unified_models import Claim, ClaimType


def create_research_coding_skills() -> List[Claim]:
    """
    Create skill claims for research coding projects methodology
    """
    
    skill_1 = Claim(
        id="skill_research_coding_projects_breakdown",
        content="""
When researching a coding project, break it down into these systematic steps:

1. **Project Requirements Analysis**: 
   - Identify the core functionality and features needed
   - Determine the technology stack and constraints
   - Define success criteria and acceptance tests

2. **Technology Research**:
   - Research best practices for the chosen language/framework
   - Find popular libraries and tools for the domain
   - Look for similar open-source projects as reference

3. **Architecture Planning**:
   - Design the overall structure and module organization
   - Plan data structures and algorithms needed
   - Consider performance, security, and scalability factors

4. **Implementation Strategy**:
   - Break down into smaller, manageable components
   - Plan the development order (core features first)
   - Set up testing and validation approach

5. **Documentation and Examples**:
   - Research existing documentation patterns
   - Find code examples and tutorials
   - Plan for code comments and README files

Use webSearch for external research and readFiles for examining existing code examples.
""",
        confidence=0.95,
        type=[ClaimType.SKILL],
        tags=["research", "methodology", "coding", "planning"],
        created=datetime.utcnow()
    )
    
    skill_2 = Claim(
        id="skill_component_research_strategy",
        content="""
For each component of a coding project, follow this research strategy:

1. **Identify Component Purpose**: Clearly define what this component should accomplish
2. **Search for Existing Solutions**: Look for libraries, frameworks, or similar implementations
3. **Study Best Practices**: Research idiomatic patterns and conventions for the technology
4. **Examine Code Examples**: Find and analyze working code samples
5. **Understand Dependencies**: Identify what other components or libraries this needs
6. **Plan Integration**: Consider how this component connects with the overall system

Always prioritize understanding over copying - focus on learning the principles behind the code.
""",
        confidence=0.90,
        type=[ClaimType.SKILL],
        tags=["research", "components", "integration", "best-practices"],
        created=datetime.utcnow()
    )
    
    skill_3 = Claim(
        id="skill_technology_stack_research",
        content="""
When researching a new technology stack for a project:

1. **Language Fundamentals**: 
   - Search for "getting started" guides and official documentation
   - Learn basic syntax, data types, and control structures
   - Understand the language's philosophy and design principles

2. **Ecosystem and Tools**:
   - Research package managers and build tools
   - Find popular libraries for common tasks (HTTP, JSON, testing, etc.)
   - Learn about IDE support and debugging tools

3. **Project Structure**:
   - Study typical project layouts and conventions
   - Understand configuration files and their purposes
   - Learn about testing frameworks and patterns

4. **Community and Resources**:
   - Find official documentation, tutorials, and examples
   - Identify active communities (forums, Discord, Reddit)
   - Look for well-maintained open-source projects to study

Use webSearch to find current information and readFiles to examine example projects.
""",
        confidence=0.92,
        type=[ClaimType.SKILL],
        tags=["technology", "research", "ecosystem", "learning"],
        created=datetime.utcnow()
    )
    
    skill_4 = Claim(
        id="skill_problem_decomposition_technique",
        content="""
Break down complex coding problems using this technique:

1. **Problem Statement Clarification**:
   - Restate the problem in your own words
   - Identify inputs, outputs, and constraints
   - Define edge cases and error conditions

2. **High-Level Decomposition**:
   - Identify the main functional areas
   - Group related functionality together
   - Define interfaces between components

3. **Detailed Breakdown**:
   - For each component, list specific functions/methods needed
   - Identify data structures required
   - Plan the algorithmic approach

4. **Implementation Order**:
   - Start with core data structures and utilities
   - Implement basic functionality first
   - Add features incrementally with testing

5. **Validation Strategy**:
   - Plan how to test each component
   - Define integration test scenarios
   - Consider performance and error handling

This systematic approach prevents overwhelm and ensures comprehensive coverage.
""",
        confidence=0.88,
        type=[ClaimType.SKILL],
        tags=["problem-solving", "decomposition", "planning", "methodology"],
        created=datetime.utcnow()
    )
    
    skill_5 = Claim(
        id="skill_research_validation_process",
        content="""
Validate research findings through this systematic process:

1. **Source Credibility Assessment**:
   - Prioritize official documentation and reputable sources
   - Check the recency of information (technology changes fast)
   - Look for consensus across multiple sources

2. **Code Example Verification**:
   - Test code examples in a sandbox environment
   - Verify that examples actually work as described
   - Check for compatibility with your target version

3. **Best Practice Confirmation**:
   - Look for patterns repeated across multiple projects
   - Check if practices are recommended by official sources
   - Consider the context and scale of your project

4. **Alternative Comparison**:
   - Research multiple approaches to the same problem
   - Compare trade-offs (performance, complexity, maintainability)
   - Consider community adoption and support

5. **Documentation Planning**:
   - Note sources and rationale for decisions
   - Plan to document why specific approaches were chosen
   - Keep track of alternatives considered and rejected

This validation process ensures reliable, well-researched implementation decisions.
""",
        confidence=0.91,
        type=[ClaimType.SKILL],
        tags=["validation", "research", "quality", "documentation"],
        created=datetime.utcnow()
    )
    
    return [skill_1, skill_2, skill_3, skill_4, skill_5]


if __name__ == "__main__":
    # Create and display the research coding skills
    skills = create_research_coding_skills()
    
    print("Research Coding Projects - Skill Claims")
    print("=" * 50)
    
    for i, skill in enumerate(skills, 1):
        print(f"\n{i}. {skill.id}")
        print(f"   Confidence: {skill.confidence}")
        print(f"   Tags: {', '.join(skill.tags)}")
        print(f"   Content: {skill.content[:200]}...")
        print("-" * 30)