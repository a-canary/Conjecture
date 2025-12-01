"""
Skill Claim: Skill Creation Best Practices
This file contains skill claims for creating effective skill claims
"""

from datetime import datetime
from typing import List
from src.core.unified_models import Claim, ClaimType


def create_skill_creation_skills() -> List[Claim]:
    """
    Create skill claims for skill creation best practices
    """
    
    skill_1 = Claim(
        id="skill_skill_creation_methodology",
        content="""
Follow this methodology when creating new skill claims:

1. **Research and Analysis**:
   - Research best practices and established methodologies
   - Study existing examples and patterns in the domain
   - Identify common pitfalls and success factors
   - Use webSearch to find current information and examples

2. **Content Structuring**:
   - Break down complex skills into step-by-step processes
   - Use numbered lists for sequential procedures
   - Include clear headings and subheadings
   - Provide context and rationale for each step

3. **Actionable Guidance**:
   - Focus on practical, implementable advice
   - Include specific techniques and methods
   - Provide concrete examples and scenarios
   - Explain when and how to apply the skill

4. **Integration Points**:
   - Reference relevant tools that support the skill
   - Connect to related skills and methodologies
   - Show how the skill fits into larger workflows
   - Consider dependencies and prerequisites

5. **Validation and Refinement**:
   - Test the skill against real-world scenarios
   - Review for clarity and completeness
   - Ensure confidence scores reflect reliability
   - Update based on feedback and new insights

6. **Documentation Standards**:
   - Use clear, concise language
   - Avoid jargon unless necessary and explained
   - Include relevant tags for discoverability
   - Set appropriate confidence levels

Skills should be practical, well-researched guidance that can be applied immediately.
""",
        confidence=0.95,
        type=[ClaimType.SKILL],
        tags=["skill-creation", "methodology", "research", "best-practices"],
        created=datetime.utcnow()
    )
    
    skill_2 = Claim(
        id="skill_effective_skill_content",
        content="""
Create effective skill content following these guidelines:

1. **Clarity and Precision**:
   - Use clear, unambiguous language
   - Define technical terms and concepts
   - Be specific about actions and outcomes
   - Avoid vague or generic advice

2. **Action-Oriented Writing**:
   - Start each step with action verbs
   - Focus on what the user should DO
   - Provide clear instructions and procedures
   - Include decision points and criteria

3. **Context and Rationale**:
   - Explain WHY each step is important
   - Provide background information when helpful
   - Connect steps to overall objectives
   - Include success criteria and metrics

4. **Comprehensive Coverage**:
   - Cover the complete process from start to finish
   - Include common variations and alternatives
   - Address edge cases and special situations
   - Provide troubleshooting guidance

5. **Practical Examples**:
   - Include concrete examples for each major step
   - Use realistic scenarios and contexts
   - Show before/after comparisons when applicable
   - Provide templates and checklists

6. **Continuous Improvement**:
   - Review and update content regularly
   - Incorporate feedback and lessons learned
   - Stay current with industry best practices
   - Adapt to new tools and technologies

Effective skills are both comprehensive and immediately applicable.
""",
        confidence=0.92,
        type=[ClaimType.SKILL],
        tags=["content-creation", "clarity", "practicality", "examples"],
        created=datetime.utcnow()
    )
    
    skill_3 = Claim(
        id="skill_skill_taxonomy_organization",
        content="""
Organize skills using this taxonomy and classification system:

1. **Domain Classification**:
   - Group skills by functional domains (research, coding, design, etc.)
   - Use consistent naming conventions for skill categories
   - Create hierarchical relationships between skills
   - Consider cross-domain connections and dependencies

2. **Skill Level Tagging**:
   - Use tags to indicate skill complexity and prerequisites
   - Include domain-specific tags for discoverability
   - Tag skills with relevant tools and technologies
   - Use consistent tag vocabulary across skills

3. **Relationship Mapping**:
   - Identify prerequisite skills and dependencies
   - Map skill progression paths and learning sequences
   - Connect complementary skills that work well together
   - Document skill conflicts and incompatibilities

4. **Metadata Standards**:
   - Use descriptive IDs that reflect skill content
   - Set appropriate confidence scores based on validation
   - Include creation and update timestamps
   - Document sources and references

5. **Discoverability Optimization**:
   - Use keywords that match common search queries
   - Include synonyms and related terms in tags
   - Structure content for easy scanning and comprehension
   - Consider how LLM will discover and select skills

6. **Maintenance Strategy**:
   - Regularly review and update skill classifications
   - Retire or merge redundant skills
   - Split complex skills into focused components
   - Maintain consistency across the skill taxonomy

Well-organized skills are easier to discover, understand, and apply.
""",
        confidence=0.90,
        type=[ClaimType.SKILL],
        tags=["taxonomy", "organization", "classification", "metadata"],
        created=datetime.utcnow()
    )
    
    skill_4 = Claim(
        id="skill_skill_validation_process",
        content="""
Validate skills through this systematic process:

1. **Content Accuracy Review**:
   - Verify all factual claims and technical details
   - Check that procedures actually work as described
   - Validate examples and code snippets
   - Ensure instructions are complete and correct

2. **Practical Testing**:
   - Apply the skill to real-world scenarios
   - Test with different contexts and variations
   - Verify that expected outcomes are achieved
   - Identify gaps or missing steps

3. **Peer Review Process**:
   - Have subject matter experts review the content
   - Get feedback from target users
   - Test clarity and comprehensibility
   - Collect suggestions for improvement

4. **Integration Testing**:
   - Verify the skill works with referenced tools
   - Test compatibility with related skills
   - Check workflow integration points
   - Validate dependencies and prerequisites

5. **Confidence Assessment**:
   - Set confidence scores based on validation results
   - Consider the strength of evidence and testing
   - Account for context-specific reliability
   - Document limitations and caveats

6. **Continuous Monitoring**:
   - Track skill usage and effectiveness
   - Monitor for changes in underlying technologies
   - Collect user feedback and success stories
   - Update content based on new information

Validated skills are more reliable and trustworthy for users.
""",
        confidence=0.93,
        type=[ClaimType.SKILL],
        tags=["validation", "testing", "quality-assurance", "reliability"],
        created=datetime.utcnow()
    )
    
    skill_5 = Claim(
        id="skill_skill_evolution_maintenance",
        content="""
Maintain and evolve skills using this systematic approach:

1. **Regular Review Schedule**:
   - Schedule periodic reviews of all skills
   - Prioritize frequently used or critical skills
   - Review skills when dependencies change
   - Update based on new research or best practices

2. **Version Management**:
   - Track changes and updates to skill content
   - Maintain version history and change logs
   - Document reasons for modifications
   - Consider backward compatibility

3. **Feedback Integration**:
   - Collect user feedback and success metrics
   - Analyze usage patterns and common issues
   - Incorporate lessons learned from application
   - Address reported problems or gaps

4. **Technology Evolution**:
   - Monitor changes in relevant technologies
   - Update skills to reflect new tools or methods
   - Retire skills that become obsolete
   - Create new skills for emerging practices

5. **Quality Improvement**:
   - Continuously improve clarity and usability
   - Add new examples and use cases
   - Enhance integration with other skills
   - Optimize for different experience levels

6. **Community Contribution**:
   - Encourage community contributions and improvements
   - Share successful applications and variations
   - Collaborate on skill development and refinement
   - Build a knowledge base of best practices

Evolving skills stay relevant and valuable over time.
""",
        confidence=0.91,
        type=[ClaimType.SKILL],
        tags=["maintenance", "evolution", "updates", "community"],
        created=datetime.utcnow()
    )
    
    return [skill_1, skill_2, skill_3, skill_4, skill_5]


if __name__ == "__main__":
    # Create and display the skill creation skills
    skills = create_skill_creation_skills()
    
    print("Skill Creation Best Practices - Skill Claims")
    print("=" * 45)
    
    for i, skill in enumerate(skills, 1):
        print(f"\n{i}. {skill.id}")
        print(f"   Confidence: {skill.confidence}")
        print(f"   Tags: {', '.join(skill.tags)}")
        print(f"   Content: {skill.content[:200]}...")
        print("-" * 30)