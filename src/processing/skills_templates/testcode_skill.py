"""
TestCode Skill Template for Phase 3
Guides testing strategy, implementation, and quality assurance
"""

from datetime import datetime
from typing import Dict, List, Optional, Any

from .models import SkillTemplate, SkillType, GuidanceStep


class TestCodeSkillTemplate:
    """TestCode skill template implementation"""
    
    @staticmethod
    def get_template() -> SkillTemplate:
        """Get the testcode skill template"""
        return SkillTemplate(
            id="testcode_skill_v1",
            skill_type=SkillType.TESTCODE,
            name="Testing & Quality Assurance",
            description="Comprehensive guidance for designing and implementing effective tests to ensure code quality and reliability",
            guidance_steps=[
                GuidanceStep(
                    step_number=1,
                    title="Understand Testing Requirements",
                    description="Analyze the code to determine what needs to be tested and why",
                    instructions=[
                        "Review the code structure, functionality, and business logic",
                        "Identify critical paths and important edge cases",
                        "Understand the intended behavior and expected outcomes",
                        "Consider the impact of failures and security implications",
                        "Identify external dependencies and integration points"
                    ],
                    tips=[
                        "Focus on testing business-critical functionality first",
                        "Consider both functional and non-functional requirements",
                        "Think about error conditions as well as success cases",
                        "Identify areas that are likely to change or be error-prone",
                        "Consider the user perspective in testing scenarios"
                    ],
                    expected_output="Clear understanding of testing scope, priorities, and critical areas",
                    common_pitfalls=[
                        "Not understanding the business context of the code",
                        "Skipping important edge cases and error conditions",
                        "Focus only on happy path scenarios",
                        "Not considering integration and system-level testing"
                    ]
                ),
                
                GuidanceStep(
                    step_number=2,
                    title="Design Test Strategy",
                    description="Create a comprehensive plan for testing approach and coverage",
                    instructions=[
                        "Choose appropriate testing types and levels (unit, integration, system)",
                        "Define test coverage goals and acceptance criteria",
                        "Plan test data scenarios and edge cases",
                        "Design test automation strategy and framework",
                        "Plan testing environment and infrastructure needs"
                    ],
                    tips=[
                        "Start with unit tests, then build up to integration and system tests",
                        "Aim for high coverage of critical business logic",
                        "Plan for both positive (success) and negative (failure) test cases",
                        "Consider test maintenance as part of the strategy",
                        "Plan for performance and security testing if applicable"
                    ],
                    expected_output="Comprehensive test strategy with defined scope, coverage goals, and approach",
                    common_pitfalls=[
                        "Unclear testing strategy with vague goals",
                        "Not planning for different types of testing",
                        "Over or under-estimating coverage requirements",
                        "Not considering test maintenance and automation"
                    ]
                ),
                
                GuidanceStep(
                    step_number=3,
                    title="Plan Test Cases and Scenarios",
                    description="Design specific test cases covering various scenarios and edge cases",
                    instructions=[
                        "Create test cases for normal expected behavior",
                        "Design tests for edge cases and boundary conditions",
                        "Plan tests for error conditions and invalid inputs",
                        "Consider integration scenarios with external systems",
                        "Plan tests for performance and load requirements if relevant"
                    ],
                    tips=[
                        "Use boundary value analysis for numeric inputs",
                        "Consider equivalence partitioning to reduce test redundancy",
                        "Design tests that are independent and can run in any order",
                        "Include tests for security vulnerabilities and data validation",
                        "Plan test data setup and teardown procedures"
                    ],
                    expected_output="Detailed test cases covering all identified scenarios and edge cases",
                    common_pitfalls=[
                        "Missing important edge cases and boundary conditions",
                        "Tests that are dependent on execution order",
                        "Inadequate test data planning",
                        "Not testing error conditions and failure scenarios"
                    ]
                ),
                
                GuidanceStep(
                    step_number=4,
                    title="Set Up Testing Framework and Environment",
                    description="Configure testing infrastructure and tools",
                    instructions=[
                        "Choose and set up appropriate testing frameworks",
                        "Configure test databases and mock environments",
                        "Set up continuous integration and automated testing",
                        "Configure test reporting and coverage tools",
                        "Establish test data management procedures"
                    ],
                    tips=[
                        "Use established testing frameworks for your language/platform",
                        "Set up isolated test environments to ensure test independence",
                        "Configure automated test execution as part of development workflow",
                        "Use tools to measure test coverage and identify gaps",
                        "Plan for test data versioning and cleanup"
                    ],
                    expected_output="Configured testing environment with automated execution and reporting",
                    common_pitfalls=[
                        "Not setting up proper test isolation",
                        "Missing test automation and CI/CD integration",
                        "Poor test data management leading to unreliable tests",
                        "Not using coverage tools to identify test gaps"
                    ]
                ),
                
                GuidanceStep(
                    step_number=5,
                    title="Implement Unit Tests",
                    description="Write focused unit tests for individual components and functions",
                    instructions=[
                        "Write unit tests for core business logic functions",
                        "Test individual components in isolation from dependencies",
                        "Use mocks and stubs to isolate units under test",
                        "Ensure tests are fast, focused, and independent",
                        "Test both success and failure conditions for each unit"
                    ],
                    tips=[
                        "Follow the Arrange-Act-Assert pattern in test structure",
                        "Write descriptive test names that indicate what is being tested",
                        "Use parameterized tests for multiple test scenarios",
                        "Keep unit tests simple and focused on single behaviors",
                        "Mock external dependencies to ensure test isolation"
                    ],
                    expected_output="Comprehensive unit tests with high code coverage",
                    common_pitfalls=[
                        "Tests that depend on external systems or databases",
                        "Unit tests that are too complex or test multiple things",
                        "Poor test naming making it hard to understand what's tested",
                        "Not mocking dependencies leading to test unreliability"
                    ]
                ),
                
                GuidanceStep(
                    step_number=6,
                    title="Implement Integration Tests",
                    description="Write tests that verify interactions between components",
                    instructions=[
                        "Test interactions between different components and modules",
                        "Verify external API integrations and data flows",
                        "Test database operations and data persistence",
                        "Validate configuration and environment dependencies",
                        "Test error handling across component boundaries"
                    ],
                    tips=[
                        "Use realistic but controlled environments for integration testing",
                        "Focus on critical integration points and failure scenarios",
                        "Design integration tests to be more comprehensive than unit tests",
                        "Consider using test databases or service virtualization",
                        "Plan for test environment setup and cleanup"
                    ],
                    expected_output="Integration tests covering critical component interactions",
                    common_pitfalls=[
                        "Integration tests that are too slow or flaky",
                        "Not testing critical integration paths",
                        "Tests that depend on external availability",
                        "Poor test environment isolation leading to test conflicts"
                    ]
                ),
                
                GuidanceStep(
                    step_number=7,
                    title="Implement System and End-to-End Tests",
                    description="Write tests that verify the complete system functionality",
                    instructions=[
                        "Write end-to-end tests for critical user workflows",
                        "Test complete system behavior from user perspective",
                        "Verify system performance under realistic conditions",
                        "Test system configuration and deployment processes",
                        "Validate security and compliance requirements"
                    ],
                    tips=[
                        "Focus on high-value user journeys and business-critical scenarios",
                        "Keep end-to-end tests reliable and maintainable",
                        "Use realistic test data and environments",
                        "Monitor test execution time and performance",
                        "Balance coverage vs. test execution time for E2E tests"
                    ],
                    expected_output="System and end-to-end tests for critical workflows",
                    common_pitfalls=[
                        "End-to-end tests that are too numerous or slow",
                        "Tests that are unreliable due to external dependencies",
                        "Not focusing on the most important user scenarios",
                        "Poor test environment management"
                    ]
                ),
                
                GuidanceStep(
                    step_number=8,
                    title="Add Performance and Load Testing",
                    description="Implement tests to verify system performance characteristics",
                    instructions=[
                        "Define performance requirements and benchmarks",
                        "Create load tests for expected usage patterns",
                        "Test system behavior under stress and peak loads",
                        "Monitor resource usage during performance tests",
                        "Identify and address performance bottlenecks"
                    ],
                    tips=[
                        "Use realistic data volumes and user behavior patterns",
                        "Test both average and peak load scenarios",
                        "Monitor key performance indicators during tests",
                        "Plan for gradual load increase in testing",
                        "Consider both frontend and backend performance"
                    ],
                    expected_output="Performance test results with identified bottlenecks and optimizations",
                    common_pitfalls=[
                        "Not defining clear performance requirements",
                        "Unrealistic test scenarios that don't reflect actual usage",
                        "Not monitoring the right metrics during performance tests",
                        "Ignoring frontend and client-side performance"
                    ]
                ),
                
                GuidanceStep(
                    step_number=9,
                    title="Review and Optimize Test Suite",
                    description="Continuously improve and maintain the test suite",
                    instructions=[
                        "Analyze test coverage and identify gaps",
                        "Review test execution time and optimize slow tests",
                        "Remove redundant or obsolete tests",
                        "Improve test reliability and reduce flakiness",
                        "Continuously improve test automation and tooling"
                    ],
                    tips=[
                        "Regularly review and update tests as code changes",
                        "Use code coverage tools to identify untested code paths",
                        "Monitor test execution metrics and identify trends",
                        "Investigate and fix flaky tests systematically",
                        "Maintain good test quality standards"
                    ],
                    expected_output="Optimized, reliable test suite with good coverage and performance",
                    common_pitfalls=[
                        "Ignoring test maintenance allowing test suite quality to decay",
                        "Not addressing test failures promptly",
                        "Keeping obsolete or redundant tests",
                        "Allowing test suite to become too slow"
                    ]
                )
            ],
            prerequisites=[
                "Understanding of software testing concepts",
                "Knowledge of testing frameworks for the relevant language",
                "Ability to write clean, maintainable test code"
            ],
            success_criteria=[
                "Test suite provides good coverage of critical functionality",
                "Tests are reliable, fast, and easy to maintain",
                "Both positive and negative scenarios are tested",
                "Automated testing is integrated into development workflow",
                "System quality and reliability are verified through testing"
            ],
            version="1.0",
            metadata={
                "category": "testing",
                "complexity": "intermediate",
                "estimated_time_minutes": 100,
                "skill_dependencies": ["writecode"],
                "common_use_cases": [
                    "Software quality assurance",
                    "Continuous integration pipelines",
                    "Requirement validation",
                    "Performance optimization",
                    "Security verification"
                ]
            }
        )
    
    @staticmethod
    def get_customization_options() -> Dict[str, Any]:
        """Get customization options for the testcode template"""
        return {
            "testing_frameworks": [
                "pytest",
                "jest", 
                "junit",
                "mocha",
                "rspec",
                "googletest",
                "unittest"
            ],
            "testing_types": [
                "unit_testing",
                "integration_testing",
                "end_to_end_testing",
                "api_testing",
                "ui_testing",
                "performance_testing",
                "security_testing"
            ],
            "application_types": [
                "web_application",
                "api_service",
                "cli_tool",
                "mobile_app",
                "desktop_application",
                "embedded_system",
                "data_pipeline"
            ],
            "coverage_goals": [
                "basic_coverage",
                "high_coverage",
                "critical_path_coverage",
                "regression_testing",
                "comprehensive_testing"
            ]
        }
    
    @staticmethod
    def adapt_for_context(template: SkillTemplate, context: Dict[str, Any]) -> SkillTemplate:
        """Adapt the testcode template based on specific context"""
        testing_framework = context.get("testing_framework", "generic")
        application_type = context.get("application_type", "generic")
        testing_type = context.get("testing_type", "unit_testing")
        
        # Copy the template
        adapted = template.copy(deep=True)
        
        # Framework-specific adaptations
        if testing_framework == "pytest":
            adapted.guidance_steps[5].tips.extend([
                "Use pytest fixtures for test setup and teardown",
                "Leverage parameterized tests with @pytest.mark.parametrize",
                "Use pytest markers for categorizing tests",
                "Take advantage of pytest's rich assertion messages",
                "Use pytest-bdd for behavior-driven development if needed"
            ])
            adapted.metadata["framework_specific"] = ["Fixtures", "Parametrization", "Markers"]
            
        elif testing_framework == "jest":
            adapted.guidance_steps[5].tips.extend([
                "Use Jest's mocking capabilities for external dependencies",
                "Leverage snapshot testing for UI components",
                "Use test.each for parameterized tests",
                "Take advantage of Jest's built-in code coverage",
                "Use Jest's watch mode for efficient development testing"
            ])
            adapted.metadata["framework_specific"] = ["Mocking", "Snapshots", "Coverage"]
            
        elif testing_framework == "junit":
            adapted.guidance_steps[5].tips.extend([
                "Use JUnit's @Before and @After annotations for setup",
                "Leverage parameterized tests with @ParameterizedTest",
                "Use @Test annotations clearly indicating what's being tested",
                "Take advantage of JUnit's assertion library",
                "Use JUnit categories for organizing tests"
            ])
            adapted.metadata["framework_specific"] = ["Annotations", "Parameterized tests", "Assertions"]
        
        # Application type adaptations
        if application_type == "web_application":
            adapted.guidance_steps[7].instructions.extend([
                "Test user workflows across different browsers and devices",
                "Test responsive design and mobile compatibility",
                "Consider accessibility testing with screen readers",
                "Test form validation and user input handling",
                "Include cross-browser compatibility testing"
            ])
            adapted.metadata["domain_focus"] = "Web application testing"
            
        elif application_type == "api_service":
            adapted.guidance_steps[6].instructions.extend([
                "Test API endpoints with various request types and data",
                "Test API authentication and authorization mechanisms",
                "Consider rate limiting and API throttling in tests",
                "Test API error responses and error codes",
                "Include contract testing for API specifications"
            ])
            adapted.metadata["domain_focus"] = "API testing"
            
        elif application_type == "mobile_app":
            adapted.guidance_steps[7].instructions.extend([
                "Test app behavior on different device sizes and OS versions",
                "Consider network connectivity scenarios and offline behavior",
                "Test app lifecycle events (background, foreground, etc.)",
                "Include device-specific testing (cameras, GPS, sensors)",
                "Test app performance and memory usage"
            ])
            adapted.metadata["domain_focus"] = "Mobile application testing"
        
        # Testing type adaptations
        if testing_type == "unit_testing":
            # Focus primarily on unit testing
            adapted.guidance_steps = adapted.guidance_steps[:6]  # Keep first 6 steps
            adapted.metadata["estimated_time_minutes"] = 60
            adapted.metadata["focus_area"] = "Unit testing"
            
        elif testing_type == "performance_testing":
            # Emphasize performance testing aspects
            adapted.guidance_steps[8].instructions.extend([
                "Define specific performance metrics and SLAs",
                "Load test with realistic user behavior patterns",
                "Test scalability under increasing load",
                "Monitor server response times and throughput",
                "Consider testing under various network conditions"
            ])
            adapted.metadata["focus_area"] = "Performance testing"
            adapted.metadata["estimated_time_minutes"] = 120
            
        elif testing_type == "security_testing":
            # Add security testing focus
            security_step = GuidanceStep(
                step_number=10,
                title="Security Testing and Vulnerability Assessment",
                description="Implement security testing to identify vulnerabilities",
                instructions=[
                    "Test for common web vulnerabilities (XSS, SQL injection, etc.)",
                    "Test authentication and authorization mechanisms",
                    "Validate input sanitization and data protection",
                    "Test for information disclosure and data leakage",
                    "Consider security headers and HTTPS implementation"
                ],
                tips=[
                    "Use OWASP testing guidelines as a reference",
                    "Perform both automated and manual security testing",
                    "Test with various user roles and privilege levels",
                    "Consider security implications in all test scenarios",
                    "Document security findings with severity levels"
                ],
                expected_output="Security test results with identified vulnerabilities and remediation",
                common_pitfalls=[
                    "Skipping security testing due to complexity",
                    "Not testing authentication edge cases",
                    "Ignoring security implications in normal test scenarios",
                    "Not documenting security findings properly"
                ]
            )
            adapted.guidance_steps.append(security_step)
            adapted.metadata["focus_area"] = "Security testing"
        
        # Coverage goal adaptations
        coverage_goal = context.get("coverage_goal", "basic_coverage")
        if coverage_goal == "basic_coverage":
            adapted.guidance_steps[9].instructions.insert(0, 
                "Ensure all critical business paths are tested")
            adapted.metadata["coverage_target"] = "Critical paths only"
            
        elif coverage_goal == "high_coverage":
            adapted.guidance_steps[9].instructions.insert(0,
                "Aim for >80% code coverage across all modules")
            adapted.metadata["coverage_target"] = "80%+ coverage"
            
        elif coverage_goal == "regression_testing":
            adapted.guidance_steps[9].instructions.insert(0,
                "Focus on preventing regressions in existing functionality")
            adapted.metadata["coverage_target"] = "Regression prevention"
        
        adapted.updated_at = datetime.utcnow()
        return adapted