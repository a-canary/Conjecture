"""
Research Skill Template for Phase 3
Guides information gathering, claim creation, and knowledge assessment
"""

from datetime import datetime
from typing import Dict, List, Optional, Any

from .models import SkillTemplate, SkillType, GuidanceStep


class ResearchSkillTemplate:
    """Research skill template implementation"""
    
    @staticmethod
    def get_template() -> SkillTemplate:
        """Get the research skill template"""
        return SkillTemplate(
            id="research_skill_v1",
            skill_type=SkillType.RESEARCH,
            name="Research & Information Gathering",
            description="Comprehensive guidance for conducting research, gathering information, and creating evidence-based claims",
            guidance_steps=[
                GuidanceStep(
                    step_number=1,
                    title="Define Research Objectives",
                    description="Clearly define what you need to research and what specific information you're seeking",
                    instructions=[
                        "Break down your research question into smaller, specific components",
                        "Identify key terms, concepts, and topics to investigate",
                        "Determine the scope and boundaries of your research",
                        "Define what constitutes sufficient information for your needs",
                        "Set clear success criteria for your research outcomes"
                    ],
                    tips=[
                        "Start with broad questions, then narrow down to specifics",
                        "Write down your research objectives before you begin",
                        "Consider what information you already have versus what you need",
                        "Be realistic about the depth and breadth of your research"
                    ],
                    expected_output="Clear research objectives and defined information requirements",
                    common_pitfalls=[
                        "Starting research without clear objectives",
                        "Research scope is too broad or too narrow",
                        "Not knowing when you have enough information",
                        "Failing to define success criteria"
                    ]
                ),
                
                GuidanceStep(
                    step_number=2,
                    title="Plan Information Gathering Strategy",
                    description="Develop a systematic approach to gather relevant information",
                    instructions=[
                        "Identify the best sources for your information needs",
                        "Plan your search strategy with relevant keywords and queries",
                        "Consider multiple sources to ensure comprehensive coverage",
                        "Plan how you'll organize and document your findings",
                        "Set a timeline for different phases of information gathering"
                    ],
                    tips=[
                        "Use multiple search terms and phrases to capture different aspects",
                        "Consider both primary and secondary sources",
                        "Plan for source verification and credibility assessment",
                        "Include reputable academic, professional, and official sources"
                    ],
                    expected_output="Structured research plan with source identification and search strategy",
                    common_pitfalls=[
                        "Relying on a single source or type of source",
                        "Not planning for source verification",
                        "Inadequate search terms leading to incomplete results",
                        "Not organizing findings systematically"
                    ]
                ),
                
                GuidanceStep(
                    step_number=3,
                    title="Execute Information Collection",
                    description="Systematically gather information from your planned sources",
                    instructions=[
                        "Conduct searches using your predefined strategy",
                        "Collect relevant information from multiple sources",
                        "Document sources and important context for each finding",
                        "Organize information as you collect it",
                        "Track which research objectives each piece of information addresses"
                    ],
                    tips=[
                        "Take notes systematically as you gather information",
                        "Capture direct quotes when accuracy is important",
                        "Record source details immediately for proper citation",
                        "Look for patterns, contradictions, and gaps as you collect",
                        "Be prepared to adjust your search strategy based on findings"
                    ],
                    expected_output="Collected information organized by research objectives with complete source documentation",
                    common_pitfalls=[
                        "Not documenting sources properly",
                        "Getting sidetracked by irrelevant information",
                        "Inadequate note-taking leading to confusion",
                        "Not organizing information effectively"
                    ]
                ),
                
                GuidanceStep(
                    step_number=4,
                    title="Evaluate Source Credibility",
                    description="Assess the reliability and credibility of your sources and information",
                    instructions=[
                        "Evaluate the expertise and qualifications of source authors",
                        "Check for publication dates to ensure information is current",
                        "Assess potential biases or conflicts of interest",
                        "Verify information across multiple independent sources",
                        "Distinguish between facts, opinions, and speculative content"
                    ],
                    tips=[
                        "Look for peer-reviewed or professionally edited sources",
                        "Be cautious of sources with strong commercial or political biases",
                        "Consider the methodology used in research studies",
                        "Assess whether sources are primary (original) or secondary (interpretation)",
                        "Track confidence levels for each piece of information"
                    ],
                    expected_output="Source credibility assessments with confidence ratings for key information",
                    common_pitfalls=[
                        "Accepting information without source verification",
                        "Confusing opinion with fact",
                        "Not checking for source biases",
                        "Overlooking publication date relevance"
                    ]
                ),
                
                GuidanceStep(
                    step_number=5,
                    title="Synthesize and Organize Information",
                    description="Integrate findings from multiple sources into a coherent structure",
                    instructions=[
                        "Group related information from different sources",
                        "Identify patterns, themes, and connections in your findings",
                        "Organize information according to your research objectives",
                        "Create a structured outline of your findings",
                        "Highlight areas where information is complete versus incomplete"
                    ],
                    tips=[
                        "Use visual tools like mind maps or tables to see connections",
                        "Look for consensus versus controversy in your findings",
                        "Organize information logically rather than by source order",
                        "Identify the most important versus less critical findings",
                        "Note areas where different sources disagree"
                    ],
                    expected_output="Organized synthesis of research findings aligned with your objectives",
                    common_pitfalls=[
                        "Simply listing findings without synthesis",
                        "Not identifying patterns or themes",
                        "Poor organization making information hard to use",
                        "Ignoring contradictions or disagreements between sources"
                    ]
                ),
                
                GuidanceStep(
                    step_number=6,
                    title="Formulate Evidence-Based Claims",
                    description="Create clear, specific claims based on your research findings",
                    instructions=[
                        "Review your research objectives and collected evidence",
                        "Formulate claims that directly address your research questions",
                        "Ensure each claim is supported by sufficient evidence",
                        "Distinguish between well-supported and tentative claims",
                        "Express claims clearly and unambiguously"
                    ],
                    tips=[
                        "Make claims specific rather than overly broad",
                        "Qualify claims appropriately based on evidence strength",
                        "Separate facts from interpretations and opinions",
                        "Use precise language in your claim formulations",
                        "Consider alternative explanations for your findings"
                    ],
                    expected_output="Set of clear, evidence-based claims with supporting evidence documentation",
                    common_pitfalls=[
                        "Making claims not supported by evidence",
                        "Overgeneralizing from limited evidence",
                        "Confusing correlations with causation",
                        "Being too tentative or too absolute in claim language"
                    ]
                ),
                
                GuidanceStep(
                    step_number=7,
                    title="Assign Confidence and Evidence",
                    description="Evaluate and document confidence levels for each claim",
                    instructions=[
                        "Assess the strength and quality of supporting evidence for each claim",
                        "Consider consistency across multiple independent sources",
                        "Evaluate potential alternative explanations or counter-evidence",
                        "Assign appropriate confidence scores to each claim",
                        "Document the reasoning behind confidence assessments"
                    ],
                    tips=[
                        "Use a systematic approach to confidence assessment",
                        "Consider both evidence quality and quantity",
                        "Be more confident with claims supported by independent sources",
                        "Lower confidence for claims based on single or biased sources",
                        "Document evidence gaps and uncertainties"
                    ],
                    expected_output="Confidence scores with detailed evidence assessments for each claim",
                    common_pitfalls=[
                        "Being overconfident based on limited evidence",
                        "Not considering counter-evidence or alternatives",
                        "Using inconsistent confidence assessment criteria",
                        "Failing to document reasoning for confidence levels"
                    ]
                ),
                
                GuidanceStep(
                    step_number=8,
                    title="Identify Knowledge Gaps",
                    description="Recognize and document areas where information is insufficient or contradictory",
                    instructions=[
                        "Review your original research objectives versus what you actually found",
                        "Identify questions that remain unanswered or partially answered",
                        "Note areas where sources disagree or provide conflicting information",
                        "Assess the reliability of areas with limited supporting evidence",
                        "Prioritize knowledge gaps by importance to your research objectives"
                    ],
                    tips=[
                        "Be honest about limitations in your research",
                        "Distinguish between unknowns and unknowables",
                        "Consider practical constraints on finding missing information",
                        "Identify which gaps are critical versus nice-to-have",
                        "Document specific next steps for addressing important gaps"
                    ],
                    expected_output="Comprehensive assessment of knowledge gaps with prioritization and next steps",
                    common_pitfalls=[
                        "Not acknowledging limitations in the research",
                        "Assuming information gaps are not important",
                        "Failing to distinguish between different types of gaps",
                        "Not providing specific guidance for addressing gaps"
                    ]
                )
            ],
            prerequisites=[
                "Basic ability to formulate research questions",
                "Access to relevant information sources",
                "Note-taking and organization skills"
            ],
            success_criteria=[
                "Research objectives are clearly defined and addressed",
                "Information gathered from multiple credible sources",
                "Claims are well-supported by evidence with appropriate confidence levels",
                "Knowledge gaps are identified and documented",
                "Research is well-organized and reproducible"
            ],
            version="1.0",
            metadata={
                "category": "research",
                "complexity": "intermediate",
                "estimated_time_minutes": 60,
                "skill_dependencies": [],
                "common_use_cases": [
                    "Academic research",
                    "Business intelligence gathering", 
                    "Problem investigation",
                    "Technology assessment",
                    "Market research"
                ]
            }
        )
    
    @staticmethod
    def get_customization_options() -> Dict[str, Any]:
        """Get customization options for the research template"""
        return {
            "research_types": [
                "academic_research",
                "business_intelligence", 
                "technical_research",
                "market_analysis",
                "fact_checking",
                "literature_review"
            ],
            "source_preferences": [
                "academic_sources",
                "industry_reports",
                "news_media",
                "government_data",
                "expert_opinions",
                "primary_research"
            ],
            "depth_levels": [
                "quick_overview",
                "standard_research",
                "comprehensive_analysis",
                "expert_level_analysis"
            ],
            "output_formats": [
                "structured_report",
                "presentation_slides",
                "executive_summary",
                "detailed_findings",
                "claim_database"
            ]
        }
    
    @staticmethod
    def adapt_for_context(template: SkillTemplate, context: Dict[str, Any]) -> SkillTemplate:
        """Adapt the research template based on specific context"""
        research_type = context.get("research_type", "standard_research")
        depth_level = context.get("depth_level", "standard_research")
        domain_area = context.get("domain_area", "general")
        
        # Copy the template
        adapted = template.copy(deep=True)
        
        # Adjust based on research type
        if research_type == "academic_research":
            adapted.guidance_steps[1].tips.extend([
                "Focus on peer-reviewed journals and scholarly sources",
                "Pay attention to methodology sections in research papers",
                "Consider citation metrics and journal reputation"
            ])
            adapted.metadata["common_use_cases"] = ["Scholarly papers", "Literature reviews"]
            adapted.metadata["estimated_time_minutes"] = 120
            
        elif research_type == "business_intelligence":
            adapted.guidance_steps[1].tips.extend([
                "Focus on market reports and industry analysis",
                "Consider competitor information and market trends",
                "Look for financial data and business metrics"
            ])
            adapted.metadata["common_use_cases"] = ["Market analysis", "Competitive intelligence"]
            adapted.metadata["estimated_time_minutes"] = 90
            
        elif research_type == "fact_checking":
            adapted.guidance_steps[4].instructions.extend([
                "Prioritize primary sources when verifying facts",
                "Look for official documentation and records",
                "Check for recent updates or corrections"
            ])
            adapted.metadata["common_use_cases"] = ["Verification", "Myth-busting"]
            adapted.metadata["estimated_time_minutes"] = 45
        
        # Adjust based on depth level
        if depth_level == "quick_overview":
            # Simplify steps for quick research
            for step in adapted.guidance_steps:
                step.instructions = step.instructions[:3]  # Keep only key instructions
            adapted.metadata["estimated_time_minutes"] = 30
            
        elif depth_level == "comprehensive_analysis":
            # Add more detail for comprehensive research
            for step in adapted.guidance_steps:
                step.tips.extend([
                    "Consider interdisciplinary perspectives where relevant",
                    "Document your research methodology rigorously",
                    "Plan for stakeholder review of findings"
                ])
            adapted.metadata["estimated_time_minutes"] = 180
        
        # Domain-specific adaptations
        if domain_area:
            adapted.metadata["domain"] = domain_area
            adapted.description += f" (Adapted for {domain_area})"
        
        adapted.updated_at = datetime.utcnow()
        return adapted