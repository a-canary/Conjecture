"""
WriteCode Skill Template for Phase 3
Guides code development, design, and implementation best practices
"""

from datetime import datetime
from typing import Dict, List, Optional, Any

from .models import SkillTemplate, SkillType, GuidanceStep


class WriteCodeSkillTemplate:
    """WriteCode skill template implementation"""
    
    @staticmethod
    def get_template() -> SkillTemplate:
        """Get the writecode skill template"""
        return SkillTemplate(
            id="writecode_skill_v1",
            skill_type=SkillType.WRITECODE,
            name="Code Development & Implementation",
            description="Comprehensive guidance for developing high-quality code, from requirements to implementation",
            guidance_steps=[
                GuidanceStep(
                    step_number=1,
                    title="Analyze Requirements and Constraints",
                    description="Thoroughly understand what needs to be implemented and any constraints",
                    instructions=[
                        "Carefully read and understand all functional requirements",
                        "Identify non-functional requirements (performance, security, maintainability)",
                        "Clarify ambiguous requirements with stakeholders when needed",
                        "Identify technical constraints and limitations",
                        "Document assumptions and decisions made during analysis"
                    ],
                    tips=[
                        "Break complex requirements into smaller, manageable pieces",
                        "Look for hidden requirements or implicit expectations",
                        "Consider edge cases and boundary conditions",
                        "Think about how requirements might change in the future",
                        "Document requirement prioritization if applicable"
                    ],
                    expected_output="Clear understanding of requirements with documented constraints and assumptions",
                    common_pitfalls=[
                        "Making assumptions about unclear requirements",
                        "Missing important constraints or edge cases",
                        "Not considering performance or security requirements",
                        "Failing to document requirement interpretations"
                    ]
                ),
                
                GuidanceStep(
                    step_number=2,
                    title="Design Solution Architecture",
                    description="Plan the overall structure and approach to your solution",
                    instructions=[
                        "Choose appropriate data structures and algorithms for the problem",
                        "Design the overall system architecture and component structure",
                        "Plan interfaces between different components",
                        "Consider trade-offs between different design approaches",
                        "Document key design decisions and their rationale"
                    ],
                    tips=[
                        "Start with a high-level design before diving into details",
                        "Consider scalability and maintainability in your design",
                        "Use established design patterns when appropriate",
                        "Think about error handling and edge cases in your design",
                        "Plan for testing and debugging capabilities"
                    ],
                    expected_output="Architectural design document with key decisions and rationale",
                    common_pitfalls=[
                        "Skipping design and jumping directly to coding",
                        "Over-engineering simple problems",
                        "Not considering scalability or maintenance",
                        "Choosing inappropriate data structures or algorithms"
                    ]
                ),
                
                GuidanceStep(
                    step_number=3,
                    title="Plan Implementation Strategy",
                    description="Create a step-by-step plan for implementing your solution",
                    instructions=[
                        "Break down the implementation into logical, testable phases",
                        "Identify dependencies between different implementation parts",
                        "Plan the order of implementation to minimize integration issues",
                        "Set up development environment and necessary tools",
                        "Create a clear directory structure and organization"
                    ],
                    tips=[
                        "Implement and test one feature at a time",
                        "Start with core functionality before adding features",
                        "Plan regular integration testing points",
                        "Set up automated testing early in the process",
                        "Use version control from the beginning"
                    ],
                    expected_output="Detailed implementation plan with clear phases and milestones",
                    common_pitfalls=[
                        "Trying to implement everything at once",
                        "Poor planning leading to frequent refactoring",
                        "Not setting up proper development environment",
                        "Ignoring version control until problems occur"
                    ]
                ),
                
                GuidanceStep(
                    step_number=4,
                    title="Create Core Implementation",
                    description="Implement the main functionality following your design",
                    instructions=[
                        "Write clean, readable code following established conventions",
                        "Implement core business logic and algorithms",
                        "Follow your planned architecture and component structure",
                        "Include meaningful comments and documentation",
                        "Write code that is easy to understand and maintain"
                    ],
                    tips=[
                        "Write code for clarity over cleverness",
                        "Use meaningful variable and function names",
                        "Keep functions and methods focused on single responsibilities",
                        "Write code as if someone else will need to understand it",
                        "Follow language-specific best practices and conventions"
                    ],
                    expected_output="Working implementation of core functionality with clean, readable code",
                    common_pitfalls=[
                        "Writing overly complex or clever code",
                        "Poor naming conventions making code hard to understand",
                        "Not following language conventions",
                        "Ignoring code readability and maintainability"
                    ]
                ),
                
                GuidanceStep(
                    step_number=5,
                    title="Implement Error Handling and Validation",
                    description="Add comprehensive error handling and input validation",
                    instructions=[
                        "Identify all possible error conditions and edge cases",
                        "Implement input validation to prevent invalid data",
                        "Create meaningful error messages and error codes",
                        "Design graceful degradation and recovery strategies",
                        "Log errors appropriately for debugging and monitoring"
                    ],
                    tips=[
                        "Validate inputs at system boundaries",
                        "Use exceptions for exceptional conditions, not program flow",
                        "Provide users with clear, actionable error messages",
                        "Don't ignore or silently swallow errors",
                        "Consider security implications of error messages"
                    ],
                    expected_output="Robust error handling with proper validation and meaningful error messages",
                    common_pitfalls=[
                        "Ignoring potential error conditions",
                        "Poor error messages that don't help users",
                        "Exposing sensitive information in error messages",
                        "Not validating inputs properly"
                    ]
                ),
                
                GuidanceStep(
                    step_number=6,
                    title="Add Logging, Monitoring, and Debugging Support",
                    description="Implement support for monitoring, logging, and debugging",
                    instructions=[
                        "Add appropriate logging to track application behavior",
                        "Include logging at different severity levels",
                        "Add performance monitoring and metrics",
                        "Implement debugging features and diagnostic tools",
                        "Create monitoring alerts for critical issues"
                    ],
                    tips=[
                        "Log business-critical operations and decisions",
                        "Include enough context in log messages",
                        "Avoid logging sensitive personal information",
                        "Use structured logging formats for easier analysis",
                        "Monitor both technical and business metrics"
                    ],
                    expected_output="Comprehensive logging and monitoring with debugging support",
                    common_pitfalls=[
                        "Not enough logging making debugging difficult",
                        "Too much logging creating noise",
                        "Logging sensitive information",
                        "Not monitoring key business or technical metrics"
                    ]
                ),
                
                GuidanceStep(
                    step_number=7,
                    title="Optimize and Refactor",
                    description="Optimize performance and refactor for better maintainability",
                    instructions=[
                        "Profile the code to identify performance bottlenecks",
                        "Optimize critical paths for better performance",
                        "Refactor complex or hard-to-understand code",
                        "Remove duplicate code and improve reusability",
                        "Ensure code follows consistent style and conventions"
                    ],
                    tips=[
                        "Optimize based on actual measurements, not assumptions",
                        "Focus optimization on the most critical bottlenecks",
                        "Balance optimization with code readability",
                        "Regular refactor to prevent code entropy",
                        "Use automated tools for code quality checking"
                    ],
                    expected_output="Optimized, refactored code with improved performance and maintainability",
                    common_pitfalls=[
                        "Premature optimization without measurements",
                        "Over-optimizing at the expense of readability",
                        "Not refactoring regularly leading to code rot",
                        "Focusing on micro-optimizations instead of real bottlenecks"
                    ]
                ),
                
                GuidanceStep(
                    step_number=8,
                    title="Write Documentation and Examples",
                    description="Create comprehensive documentation for your code",
                    instructions=[
                        "Write clear docstrings and API documentation",
                        "Create user guides and usage examples",
                        "Document architecture and design decisions",
                        "Provide setup and installation instructions",
                        "Include troubleshooting and FAQ sections"
                    ],
                    tips=[
                        "Write documentation from the user's perspective",
                        "Include code examples for common use cases",
                        "Document limitations and known issues",
                        "Keep documentation in sync with code changes",
                        "Use multiple formats for different documentation needs"
                    ],
                    expected_output="Complete documentation with examples and user guides",
                    common_pitfalls=[
                        "Writing unclear or incomplete documentation",
                        "Not updating documentation when code changes",
                        "Missing examples or practical usage information",
                        "Documentation that only describes what, not why"
                    ]
                )
            ],
            prerequisites=[
                "Understanding of programming fundamentals",
                "Knowledge of relevant programming language(s)",
                "Basic software development skills"
            ],
            success_criteria=[
                "Code implements all requirements correctly",
                "Code is well-structured, readable, and maintainable",
                "Appropriate error handling and validation are implemented",
                "Code is properly tested and documented",
                "Solution is optimized and follows best practices"
            ],
            version="1.0",
            metadata={
                "category": "development",
                "complexity": "intermediate to advanced",
                "estimated_time_minutes": 90,
                "skill_dependencies": [],
                "common_use_cases": [
                    "Software feature development",
                    "Algorithm implementation", 
                    "API development",
                    "Data processing pipelines",
                    "Tool and utility creation"
                ]
            }
        )
    
    @staticmethod
    def get_customization_options() -> Dict[str, Any]:
        """Get customization options for the writecode template"""
        return {
            "programming_languages": [
                "python",
                "javascript",
                "typescript",
                "java",
                "c_sharp",
                "go",
                "rust",
                "generic"
            ],
            "project_types": [
                "web_application",
                "api_service",
                "data_processing",
                "cli_tool",
                "library_package",
                "algorithm_implementation",
                "ui_component",
                "system_integration"
            ],
            "complexity_levels": [
                "simple_script",
                "feature_implementation",
                "system_design",
                "complex_algorithm",
                "enterprise_application"
            ],
            "quality_standards": [
                "basic_functionality",
                "production_ready",
                "enterprise_grade",
                "high_performance",
                "security_focused"
            ]
        }
    
    @staticmethod
    def adapt_for_context(template: SkillTemplate, context: Dict[str, Any]) -> SkillTemplate:
        """Adapt the writecode template based on specific context"""
        programming_language = context.get("programming_language", "generic")
        project_type = context.get("project_type", "generic")
        complexity_level = context.get("complexity_level", "feature_implementation")
        
        # Copy the template
        adapted = template.copy(deep=True)
        
        # Language-specific adaptations
        if programming_language == "python":
            adapted.guidance_steps[4].tips.extend([
                "Follow PEP 8 style guidelines",
                "Use type hints for better code documentation",
                "Consider using dataclasses for data structures",
                "Leverage Python's standard library extensively"
            ])
            adapted.metadata["specific_considerations"] = ["PEP 8", "Type hints", "Virtual environments"]
            
        elif programming_language == "javascript":
            adapted.guidance_steps[4].tips.extend([
                "Use modern ES6+ features appropriately",
                "Handle asynchronous operations carefully",
                "Be mindful of browser compatibility if applicable",
                "Consider using TypeScript for large projects"
            ])
            adapted.metadata["specific_considerations"] = ["ES6+ features", "Async/await", "Browser compatibility"]
            
        elif programming_language == "java":
            adapted.guidance_steps[4].tips.extend([
                "Follow Java naming conventions consistently",
                "Use appropriate access modifiers",
                "Consider design patterns and OOP principles",
                "Be mindful of memory management and garbage collection"
            ])
            adapted.metadata["specific_considerations"] = ["OOP principles", "Design patterns", "JVM optimization"]
        
        # Project type adaptations
        if project_type == "web_application":
            adapted.guidance_steps[1].instructions.extend([
                "Consider user experience and interface requirements",
                "Plan for responsive design and accessibility",
                "Think about security and data protection",
                "Consider performance on various devices and connections"
            ])
            adapted.metadata["domain_focus"] = "Web development"
            
        elif project_type == "api_service":
            adapted.guidance_steps[1].instructions.extend([
                "Design clear, consistent API endpoints",
                "Consider authentication and authorization",
                "Plan for rate limiting and load balancing",
                "Design for versioning and backward compatibility"
            ])
            adapted.metadata["domain_focus"] = "API development"
            
        elif project_type == "data_processing":
            adapted.guidance_steps[2].instructions.extend([
                "Consider data volume and processing requirements",
                "Plan for efficient data flow and transformations",
                "Design for scalability and parallel processing",
                "Consider memory usage and optimization strategies"
            ])
            adapted.metadata["domain_focus"] = "Data processing"
        
        # Complexity level adaptations
        if complexity_level == "simple_script":
            # Simplify for basic implementations
            for step in adapted.guidance_steps[:6]:  # Keep first 6 steps simplified
                step.instructions = step.instructions[:3]
            adapted.metadata["estimated_time_minutes"] = 30
            
        elif complexity_level == "system_design":
            # Add more comprehensive guidance
            additional_steps = [
                GuidanceStep(
                    step_number=9,
                    title="System Integration and Deployment",
                    description="Plan and implement system integration and deployment strategies",
                    instructions=[
                        "Design integration points with external systems",
                        "Plan deployment processes and environment configuration",
                        "Consider monitoring and alerting in production",
                        "Plan for scaling and redundancy",
                        "Design for maintainability and future enhancements"
                    ],
                    tips=[
                        "Use containerization for consistent deployments",
                        "Implement CI/CD pipelines for reliable releases",
                        "Plan for rollback strategies",
                        "Monitor system health and performance metrics",
                        "Document operational procedures"
                    ],
                    expected_output="System deployment and integration plan with operational procedures",
                    common_pitfalls=[
                        "Not planning for deployment and operations",
                        "Ignoring monitoring and observability",
                        "Poor integration leading to reliability issues",
                        "Not planning for scaling and maintenance"
                    ]
                )
            ]
            adapted.guidance_steps.extend(additional_steps)
            adapted.metadata["estimated_time_minutes"] = 180
            
        elif complexity_level == "enterprise_application":
            # Add enterprise-specific considerations
            adapted.guidance_steps[2].tips.extend([
                "Consider enterprise architecture patterns",
                "Plan for distributed systems and microservices",
                "Design for horizontal scalability and high availability",
                "Consider data consistency and transaction management"
            ])
            adapted.metadata["domain_focus"] = "Enterprise systems"
            adapted.metadata["estimated_time_minutes"] = 240
        
        adapted.updated_at = datetime.utcnow()
        return adapted