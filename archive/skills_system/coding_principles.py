"""
Skill Claim: Coding Principles and Design Documentation
This file contains skill claims for writing clean code and creating design documents
"""

from datetime import datetime
from typing import List
from src.core.unified_models import Claim, ClaimType


def create_coding_principles_skills() -> List[Claim]:
    """
    Create skill claims for coding principles and design documentation
    """
    
    skill_1 = Claim(
        id="skill_clean_code_principles",
        content="""
Follow these clean code principles when writing any code:

1. **Meaningful Names**:
   - Use descriptive variable and function names
   - Avoid abbreviations unless universally understood
   - Name should reveal intent and purpose

2. **Small Functions**:
   - Functions should do one thing well
   - Keep functions under 20-30 lines when possible
   - Use descriptive function names instead of comments

3. **Comments and Documentation**:
   - Comment WHY, not WHAT
   - Write self-documenting code that needs minimal comments
   - Use docstrings for functions and classes

4. **Error Handling**:
   - Handle errors explicitly and gracefully
   - Use appropriate error types and messages
   - Fail fast and provide clear error information

5. **Code Organization**:
   - Group related functionality together
   - Use consistent formatting and style
   - Follow language-specific conventions

6. **Testing Considerations**:
   - Write testable code from the start
   - Separate concerns for easier testing
   - Consider edge cases and error conditions

Use writeFiles to create well-structured code files following these principles.
""",
        confidence=0.94,
        type=[ClaimType.SKILL],
        tags=["clean-code", "principles", "quality", "best-practices"],
        created=datetime.utcnow()
    )
    
    skill_2 = Claim(
        id="skill_design_documentation_structure",
        content="""
Create comprehensive design documents using this structure:

1. **Project Overview**:
   - Problem statement and goals
   - Target users and use cases
   - Success criteria and metrics

2. **Technical Architecture**:
   - High-level system design
   - Component relationships and data flow
   - Technology choices and rationale

3. **Detailed Specifications**:
   - API designs and interfaces
   - Data models and structures
   - Algorithm descriptions

4. **Implementation Plan**:
   - Development phases and milestones
   - Task breakdown and dependencies
   - Risk assessment and mitigation

5. **Testing Strategy**:
   - Unit testing approach
   - Integration testing plan
   - Performance and security considerations

6. **Maintenance and Evolution**:
   - Deployment procedures
   - Monitoring and logging
   - Future enhancement possibilities

Use writeFiles to create markdown documents following this structure for each project.
""",
        confidence=0.91,
        type=[ClaimType.SKILL],
        tags=["design", "documentation", "architecture", "planning"],
        created=datetime.utcnow()
    )
    
    skill_3 = Claim(
        id="skill_code_review_guidelines",
        content="""
Conduct effective code reviews using these guidelines:

1. **Review Preparation**:
   - Understand the context and purpose of the changes
   - Review requirements and acceptance criteria
   - Set aside dedicated time for thorough review

2. **Code Quality Assessment**:
   - Check for adherence to coding standards
   - Verify error handling and edge cases
   - Assess performance and security implications

3. **Design and Architecture**:
   - Evaluate if the solution fits the overall architecture
   - Check for proper separation of concerns
   - Assess scalability and maintainability

4. **Testing and Validation**:
   - Verify adequate test coverage
   - Check that tests actually test the right things
   - Look for missing edge case tests

5. **Documentation and Comments**:
   - Ensure code is self-documenting where possible
   - Check that complex logic is explained
   - Verify API documentation is accurate

6. **Constructive Feedback**:
   - Provide specific, actionable suggestions
   - Explain the reasoning behind recommendations
   - Acknowledge good practices and improvements

Use writeFiles to create code review checklists and templates.
""",
        confidence=0.89,
        type=[ClaimType.SKILL],
        tags=["code-review", "quality", "collaboration", "feedback"],
        created=datetime.utcnow()
    )
    
    skill_4 = Claim(
        id="skill_api_design_principles",
        content="""
Design clean, intuitive APIs following these principles:

1. **Consistency and Predictability**:
   - Use consistent naming conventions
   - Follow RESTful principles for web APIs
   - Maintain backward compatibility when possible

2. **Clear Contracts**:
   - Define explicit input/output specifications
   - Document error responses and status codes
   - Provide examples for common use cases

3. **Resource-Oriented Design**:
   - Use nouns for resource names
   - Implement standard CRUD operations
   - Design resource relationships clearly

4. **Error Handling**:
   - Use appropriate HTTP status codes
   - Provide meaningful error messages
   - Include error codes for programmatic handling

5. **Security Considerations**:
   - Implement proper authentication and authorization
   - Validate all inputs and sanitize outputs
   - Use HTTPS and secure communication protocols

6. **Performance and Scalability**:
   - Design for pagination and filtering
   - Implement caching strategies
   - Consider rate limiting and throttling

Use writeFiles to create API specification documents and implementation templates.
""",
        confidence=0.92,
        type=[ClaimType.SKILL],
        tags=["api-design", "rest", "security", "performance"],
        created=datetime.utcnow()
    )
    
    skill_5 = Claim(
        id="skill_refactoring_techniques",
        content="""
Apply systematic refactoring techniques to improve code quality:

1. **Identify Code Smells**:
   - Long methods and classes
   - Duplicate code and logic
   - Complex conditional statements
   - Poor naming and unclear intent

2. **Refactoring Strategies**:
   - Extract Method: Break down long functions
   - Extract Class: Separate responsibilities
   - Rename: Improve clarity and understanding
   - Move Method: Place behavior where it belongs

3. **Safe Refactoring Process**:
   - Write tests before refactoring
   - Make small, incremental changes
   - Run tests after each change
   - Commit frequently with descriptive messages

4. **Design Pattern Application**:
   - Strategy Pattern for interchangeable algorithms
   - Observer Pattern for event handling
   - Factory Pattern for object creation
   - Singleton Pattern for shared resources

5. **Performance Optimization**:
   - Profile before optimizing
   - Focus on bottlenecks and hot paths
   - Consider algorithmic improvements first
   - Measure impact of optimizations

6. **Documentation Updates**:
   - Update comments and documentation
   - Reflect changes in API docs
   - Update examples and tutorials

Use writeFiles to create refactoring checklists and before/after code examples.
""",
        confidence=0.90,
        type=[ClaimType.SKILL],
        tags=["refactoring", "code-quality", "design-patterns", "optimization"],
        created=datetime.utcnow()
    )
    
    return [skill_1, skill_2, skill_3, skill_4, skill_5]


if __name__ == "__main__":
    # Create and display the coding principles skills
    skills = create_coding_principles_skills()
    
    print("Coding Principles and Design Documentation - Skill Claims")
    print("=" * 60)
    
    for i, skill in enumerate(skills, 1):
        print(f"\n{i}. {skill.id}")
        print(f"   Confidence: {skill.confidence}")
        print(f"   Tags: {', '.join(skill.tags)}")
        print(f"   Content: {skill.content[:200]}...")
        print("-" * 40)