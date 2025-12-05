"""
XML-Optimized Prompt Templates for Enhanced LLM Performance

This module provides XML-structured prompt templates that should improve
LLM understanding of task requirements and increase claim creation success rates.
"""

import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from .models import PromptTemplate, PromptTemplateType, OptimizedPrompt


class XMLOptimizedTemplateManager:
    """
    Manages XML-optimized prompt templates for better LLM performance
    """
    
    def __init__(self):
        """Initialize XML template manager"""
        self.templates = {
            "research_xml": self._create_research_template_xml(),
            "research_enhanced_xml": self._create_enhanced_research_template_xml(),
            "research_enhanced_xml": self._create_enhanced_research_template_xml(),
            "analysis_xml": self._create_analysis_template_xml(),
            "validation_xml": self._create_validation_template_xml(),
            "synthesis_xml": self._create_synthesis_template_xml(),
            "task_decomposition_xml": self._create_task_decomposition_template_xml(),
        }
    
    def _create_research_template_xml(self) -> PromptTemplate:
        """Create XML-optimized research template"""
        return PromptTemplate(
            id="research_xml",
            name="XML Research Template",
            description="XML-structured template for research tasks with enhanced context handling",
            template_type=PromptTemplateType.RESEARCH,
            template_content=''''''You are Conjecture, an AI system that uses evidence-based reasoning to explore new claims based on existing context.

<research_task>
{{user_query}}
</research_task>

<available_context>
{{relevant_context}}
</available_context>

RESEARCH GUIDANCE:
1. Information Gathering
   - Identify key concepts and definitions related to {{user_query}}
   - Determine what information needs to be collected
   - Plan search strategies for comprehensive coverage
   - Use structured approach to organize findings

2. Source Evaluation
   - Evaluate credibility and reliability of sources
   - Check for bias and conflicting information
   - Verify information accuracy through multiple sources
   - Distinguish between facts, opinions, and speculation

3. Information Synthesis
   - Organize findings in logical hierarchy
   - Identify patterns and connections
   - Highlight gaps in current knowledge
   - Create clear, actionable insights

4. Claim Creation
   - Formulate clear, specific claims based on evidence
   - Assign appropriate confidence scores (0.0-1.0)
   - Use structured claim types: fact, concept, example, goal, reference, assertion
   - Include uncertainty estimates for speculative claims
   - Note limitations and assumptions

Available Context:
{{relevant_context}}

Please provide step-by-step guidance for this research task.
Generate up to 10 high-quality claims using this XML structure:

<claims>
  <claim id="c1" type="fact" confidence="0.9">
    <content>Factual claim about {{user_query}}</content>
    <evidence>Supporting evidence with sources</evidence>
    <reasoning>Step-by-step reasoning process</reasoning>
    <limitations>Any known limitations or uncertainties</limitations>
  </claim>
  
  <claim id="c2" type="concept" confidence="0.8">
    <content>Key concept related to {{user_query}}</content>
    <evidence>Explanatory evidence and examples</evidence>
    <reasoning>Conceptual analysis and relationships</reasoning>
    <limitations>Scope boundaries or assumptions</limitations>
  </claim>
  
  <!-- Add more claims as needed up to 10 -->
</claims>

<research_summary>
Summarize your overall research approach, key findings, and methodology.
Include any sources used in the sources field.
Describe your methodology in the methodology field.
</research_summary>'''''',
            variables=[
                {
                    'name': 'user_query', 
                    'type': 'string', 
                    'required': True, 
                    'description': 'User research query'
                },
                {
                    'name': 'relevant_context', 
                    'type': 'string', 
                    'required': False, 
                    'description': 'Relevant context information'
                }
            ]
        )
    
    def _create_enhanced_research_template_xml(self) -> PromptTemplate:
        """Create enhanced XML-optimized research template with chain-of-thought examples and confidence calibration"""
        return PromptTemplate(
            id="research_enhanced_xml",
            name="Enhanced XML Research Template with Chain-of-Thought",
            description="XML-optimized template with chain-of-thought examples, confidence calibration guidelines, and enhanced reasoning",
            template_type=PromptTemplateType.RESEARCH,
            template_content='''You are Conjecture, an advanced AI reasoning system that creates structured claims using XML format with chain-of-thought reasoning and calibrated confidence scores.

<research_task>
{{user_query}}
</research_task>

<available_context>
{{relevant_context}}
</available_context>

CHAIN-OF-THOUGHT RESEARCH PROCESS:
Follow this 6-step reasoning process for each claim you generate:

Step 1: Query Analysis
- Identify key concepts, entities, and relationships in the query
- Determine the scope and boundaries of what needs to be researched
- Clarify any ambiguities or assumptions in the query

Step 2: Evidence Evaluation
- Assess source credibility and reliability
- Look for peer-reviewed, expert consensus sources first
- Distinguish between empirical data, theoretical reasoning, and speculation
- Note any conflicts or contradictions between sources

Step 3: Claim Formulation
- Draft clear, specific, verifiable claims
- Ensure each claim addresses a distinct aspect of the query
- Use precise language and avoid vague statements
- Make claims falsifiable and testable where possible

Step 4: Confidence Assessment (CALIBRATION GUIDELINES)
Map evidence strength to confidence scores using these guidelines:

0.9-1.0 (Very High Confidence):
- Multiple peer-reviewed sources with expert consensus
- Reproducible empirical data with low error margins
- Established scientific facts with extensive verification
- Example: "Water boils at 100°C at standard atmospheric pressure"

0.7-0.8 (High Confidence):
- Several reliable sources with strong logical support
- Consistent findings across multiple studies
- Expert consensus with minor disagreements
- Example: "Regular exercise reduces risk of cardiovascular disease by approximately 30%"

0.5-0.6 (Moderate Confidence):
- Some supporting evidence but limited scope
- Logical reasoning with partial empirical support
- Emerging research with preliminary findings
- Example: "Machine learning interpretability techniques can improve model trustworthiness in specific applications"

0.3-0.4 (Low Confidence):
- Few sources or single-source claims
- Theoretical reasoning without empirical validation
- Speculative claims requiring further research
- Example: "Quantum computing may solve certain optimization problems 1000x faster by 2030"

0.1-0.2 (Very Low Confidence):
- Highly speculative or theoretical claims
- No direct empirical evidence
- Contradictory findings in existing research
- Example: "Artificial general intelligence will be achieved by 2025"

Step 5: Evidence Integration
- Select the strongest supporting evidence for each claim
- Cite specific sources, data points, or logical chains
- Address counter-evidence or alternative explanations
- Note any limitations or boundary conditions

Step 6: Claim Refinement
- Review claim for clarity, precision, and accuracy
- Ensure confidence score matches evidence strength
- Add uncertainty notes for speculative elements
- Verify claim directly addresses the research query

CHAIN-OF-THOUGHT EXAMPLES:

Example 1: Fact Claim with High Confidence
Query: "What are the health effects of regular exercise?"
Step 1: Key concepts = exercise, health effects, regular physical activity
Step 2: Evidence = WHO guidelines, multiple peer-reviewed studies, meta-analyses
Step 3: Claim = "Regular moderate exercise reduces cardiovascular disease risk by 30%"
Step 4: Confidence = 0.85 (multiple reliable sources, strong consensus)
Step 5: Evidence = WHO 2020 guidelines, 15+ longitudinal studies
Step 6: Refinement = Added "moderate" qualifier, specified "cardiovascular disease"

<claim type="fact" confidence="0.85">
<content>Regular moderate exercise reduces the risk of cardiovascular disease by approximately 30%</content>
<support>World Health Organization 2020 physical activity guidelines; meta-analysis of 15 longitudinal studies published in Lancet 2019; consistent findings across diverse populations</support>
<uncertainty>Risk reduction varies by exercise type, intensity, and individual factors; based on observational studies with some potential confounding variables</uncertainty>
</claim>

Example 2: Concept Claim with Moderate Confidence
Query: "How does machine learning interpretability work?"
Step 1: Key concepts = ML interpretability, black-box models, explanation methods
Step 2: Evidence = Research papers, technical documentation, expert surveys
Step 3: Claim = "SHAP and LIME are the most widely used local interpretation methods"
Step 4: Confidence = 0.65 (some empirical support, but rapidly evolving field)
Step 5: Evidence = Survey of ML practitioners 2023, citation analysis, tool adoption data
Step 6: Refinement = Specified "local interpretation", acknowledged rapidly evolving field

<claim type="concept" confidence="0.65">
<content>SHAP and LIME are currently the most widely used local interpretation methods for explaining individual predictions from black-box machine learning models</content>
<support>2023 ML practitioner survey showing 68% adoption rate; high citation counts in academic literature; integration in major ML frameworks</support>
<uncertainty>Rapidly evolving field with new methods emerging; adoption rates vary by industry and application domain; global adoption data limited</uncertainty>
</claim>

Example 3: Hypothesis Claim with Low Confidence
Query: "When will quantum computing break current encryption?"
Step 3: Claim = "Quantum computers may break RSA-2048 encryption by 2035"
Step 4: Confidence = 0.25 (highly speculative, limited empirical evidence)
Step 5: Evidence = Current quantum computing roadmaps, theoretical complexity analysis
Step 6: Refinement = Added "may", specified RSA-2048, gave conservative timeline

<claim type="hypothesis" confidence="0.25">
<content>Quantum computers may be capable of breaking RSA-2048 encryption by approximately 2035, assuming continued progress in qubit stability and error correction</content>
<support>Current quantum computing roadmap from major tech companies; theoretical analysis of quantum algorithms; exponential progress in qubit counts over past decade</support>
<uncertainty>Highly dependent on breakthroughs in error correction and qubit stability; may face unforeseen technical barriers; alternative post-quantum cryptography may render this irrelevant</uncertainty>
</claim>

ENHANCED CLAIM CREATION REQUIREMENTS:

1. **Chain-of-Thought Reasoning**: For each claim, mentally follow the 6-step process above
2. **Confidence Calibration**: Use the evidence strength mapping to assign realistic confidence scores
3. **XML Format**: Use this EXACT structure:
<claim type="[fact|concept|example|goal|reference|hypothesis]" confidence="[0.0-1.0]">
<content>Your clear, specific claim content here</content>
<support>Supporting evidence or reasoning</support>
<uncertainty>Any limitations or confidence notes</uncertainty>
</claim>

4. **Claim Types and Expected Confidence Ranges**:
   * fact: Verifiable statements (0.8-1.0)
   * concept: Explanatory claims (0.6-0.9)
   * example: Illustrative cases (0.4-0.7)
   * goal: Objectives or recommendations (0.7-0.9)
   * reference: Citations of external sources (0.8-1.0)
   * hypothesis: Speculative claims (0.3-0.6)

5. **Quality Standards**:
   - Generate 5-10 high-quality claims covering different aspects
   - Include detailed <support> tags with specific evidence
   - Use <uncertainty> to acknowledge limitations
   - Ensure confidence scores match evidence strength
   - Follow chain-of-thought reasoning for each claim

Available Context:
{{relevant_context}}

<research_summary>
Provide a comprehensive summary of your research approach and key findings in this XML structure:
<research_methodology>
Describe your step-by-step research process, including how you applied the 6-step chain-of-thought reasoning to each claim type
</research_methodology>
<key_findings>
Main discoveries and insights organized by confidence level
</key_findings>
<sources>
Important sources consulted and their credibility assessment
</sources>
<confidence_assessment>
Overall confidence in findings and any calibration adjustments made
</confidence_assessment>
</research_summary>''',
            variables=[
                {
                    'name': 'user_query',
                    'type': 'string',
                    'required': True,
                    'description': 'User research query'
                },
                {
                    'name': 'relevant_context',
                    'type': 'string',
                    'required': False,
                    'description': 'Relevant context information'
                }
            ]
        )
    
    def _create_enhanced_research_template_xml(self) -> PromptTemplate:
        """Create enhanced XML-optimized research template with better claim structure"""
        return PromptTemplate(
            id="research_enhanced_xml",
            name="Enhanced XML Research Template",
            description="XML-optimized template with enhanced claim structure for better LLM performance",
            template_type=PromptTemplateType.RESEARCH,
            template_content='''You are Conjecture, an advanced AI reasoning system that creates structured claims using XML format. Your task is to conduct thorough research analysis.

<research_task>
{user_query}
</research_task>

<available_context>
{relevant_context}
</available_context>

ANALYSIS APPROACH:
1. Multi-Perspective Research
   - Examine topic from multiple angles
   - Consider different methodologies and viewpoints
   - Identify areas of consensus and debate

2. Evidence-Based Investigation
   - Prioritize verifiable facts over opinions
   - Look for empirical data and expert sources
   - Distinguish between established knowledge and speculation

3. Structured Synthesis
   - Organize findings in logical hierarchy
   - Identify patterns, relationships, and connections
   - Highlight knowledge gaps and uncertainties

4. Quality Control
   - Verify information accuracy through cross-checking
   - Assess source credibility and potential biases
   - Note limitations and confidence levels

CLAIM CREATION REQUIREMENTS:
- Use this EXACT XML format for each claim:
<claim type="[fact|concept|example|goal|reference|hypothesis]" confidence="[0.0-1.0]">
<content>Your clear, specific claim content here</content>
<support>Supporting evidence or reasoning</support>
<uncertainty>Any limitations or confidence notes</uncertainty>
</claim>

- Claim Types:
  * fact: Verifiable statements with high confidence (0.8-1.0)
  * concept: Explanatory claims with moderate confidence (0.6-0.9)
  * example: Illustrative cases with lower confidence (0.4-0.7)
  * goal: Objectives or recommendations (0.7-0.9)
  * reference: Citations of external sources (0.8-1.0)
  * hypothesis: Speculative claims requiring validation (0.3-0.6)

- Generate 5-10 high-quality claims covering different aspects
- Include <support> tags with evidence or reasoning
- Use <uncertainty> for speculative claims
- Assign realistic confidence scores

Available Context:
{relevant_context}

<research_summary>
Provide a comprehensive summary of your research approach and key findings in this XML structure:
<research_methodology>
Your step-by-step research process
</research_methodology>
<key_findings>
Main discoveries and insights
</key_findings>
<sources>
Important sources consulted
</sources>
<confidence_assessment>
Overall confidence in findings and limitations
</confidence_assessment>
</research_summary>''',
            variables=[
                {
                    'name': 'user_query', 
                    'type': 'string', 
                    'required': True, 
                    'description': 'User research query'
                },
                {
                    'name': 'relevant_context', 
                    'type': 'string', 
                    'required': False, 
                    'description': 'Relevant context information'
                }
            ]
        )
    
    def _create_analysis_template_xml(self) -> PromptTemplate:
        return PromptTemplate(
            id="analysis_xml",
            name="XML Analysis Template",
            description="XML-structured template for analyzing claims with enhanced reasoning",
            template_type=PromptTemplateType.ANALYSIS,
            template_content=''''''You are Conjecture, an AI system that analyzes claims using evidence-based reasoning.

<analysis_task>
Analyze the following claims for accuracy, consistency, and logical coherence:

<claims_to_analyze>
{{claims_for_analysis}}
</claims_to_analyze>

ANALYSIS REQUIREMENTS:
1. Factual Accuracy Check
   - Verify each claim against known facts
   - Identify any factual errors or misconceptions
   - Check for logical consistency between related claims

2. Logical Coherence Analysis
   - Evaluate reasoning chains within each claim
   - Identify any logical fallacies or inconsistencies
   - Check claim relationships for proper support structure

3. Confidence Assessment
   - Evaluate if confidence scores are justified by evidence
   - Identify overconfident or underconfident claims
   - Suggest confidence adjustments if needed

4. Structured Evaluation
   - Provide analysis in clear, organized format
   - Use evidence-based reasoning for all assessments
   - Include specific recommendations for improvement

Available Context:
{{relevant_context}}

Please analyze the provided claims and return structured findings in XML format:

<analysis_result>
  <overall_assessment>
    <factual_accuracy>Assessment of factual accuracy</factual_accuracy>
    <logical_coherence>Assessment of logical reasoning</logical_coherence>
    <confidence_appropriateness>Assessment of confidence scoring</confidence_appropriateness>
  </overall_assessment>
  
  <claim_evaluations>
    <!-- Individual claim evaluations -->
  </claim_evaluations>
  
  <recommendations>
    <!-- Specific recommendations for claim improvements -->
  </recommendations>
</analysis_result>'''''',
            variables=[
                {
                    'name': 'claims_for_analysis', 
                    'type': 'string', 
                    'required': True, 
                    'description': 'Claims to be analyzed'
                },
                {
                    'name': 'relevant_context', 
                    'type': 'string', 
                    'required': False, 
                    'description': 'Relevant context information'
                }
            ]
        )
    
    def _create_validation_template_xml(self) -> PromptTemplate:
        """Create XML-optimized validation template"""
        return PromptTemplate(
            id="validation_xml",
            name="XML Validation Template", 
            description="XML-structured template for validating claim accuracy and consistency",
            template_type=PromptTemplateType.VALIDATION,
            template_content=''''''You are Conjecture, an AI system that validates claims using evidence-based reasoning.

<validation_task>
Validate the following claim for accuracy, consistency, and logical soundness:

<claim_to_validate>
{{claim_to_validate}}
</claim_to_validate>

VALIDATION REQUIREMENTS:
1. Factual Verification
   - Cross-reference claim content with reliable sources
   - Check for factual errors or misconceptions
   - Verify statistical claims with proper methodology

2. Logical Consistency
   - Evaluate internal reasoning coherence
   - Check for contradictions with established knowledge
   - Assess claim relationships and dependencies

3. Confidence Scoring
   - Verify confidence scores are evidence-based
   - Ensure confidence levels are appropriate for claim type
   - Identify any confidence inflation or deflation

Available Context:
{{relevant_context}}

Please provide detailed validation analysis in XML format:

<validation_result>
  <validation_status>VALID or INVALID</validation_status>
  <factual_accuracy>Assessment of factual correctness</factual_accuracy>
  <logical_consistency>Assessment of logical reasoning</logical_consistency>
  <confidence_assessment>Assessment of confidence scoring</confidence_assessment>
  <specific_issues>List of specific issues found</specific_issues>
  <recommendations>Validation recommendations</recommendations>
</validation_result>'''''',
            variables=[
                {
                    'name': 'claim_to_validate', 
                    'type': 'string', 
                    'required': True, 
                    'description': 'Claim to be validated'
                },
                {
                    'name': 'relevant_context', 
                    'type': 'string', 
                    'required': False, 
                    'description': 'Relevant context information'
                }
            ]
        )
    
    def _create_synthesis_template_xml(self) -> PromptTemplate:
        """Create enhanced XML-optimized synthesis template with tree-of-thought and confidence aggregation"""
        return PromptTemplate(
            id="synthesis_xml",
            name="Enhanced XML Synthesis Template with Tree-of-Thought",
            description="XML-structured template with tree-of-thought synthesis and confidence aggregation for comprehensive answer generation",
            template_type=PromptTemplateType.SYNTHESIS,
            template_content='''You are Conjecture, an advanced AI reasoning system that synthesizes comprehensive answers using evidence-based reasoning with systematic tree-of-thought processes and calibrated confidence aggregation.

<synthesis_task>
Synthesize a comprehensive answer to the following task using the provided analysis and context:

<original_task>{{original_task}}</original_task>

<analysis_results>{{analysis_results}}</analysis_results>

<claims_evaluated>{{claims_evaluated}}</claims_evaluated>

TREE-OF-THOUGHT SYNTHESIS PROCESS:
Follow this 7-step systematic synthesis process with branching reasoning paths:

Step 1: Claim Tree Construction
- Map all evaluated claims into a hierarchical tree structure
- Identify primary claims (confidence ≥0.7) as main branches
- Group supporting claims (confidence 0.5-0.69) as sub-branches
- Note speculative claims (confidence <0.5) as tentative branches
- Identify conflicting or contradictory branches

Step 2: Evidence Forest Analysis
- Treat each claim branch as requiring its own evidence forest
- Assess evidence quality for each branch independently
- Identify converging evidence that strengthens multiple branches
- Note diverging evidence that creates alternative reasoning paths
- Map evidence connections between related branches

Step 3: Confidence Tree Aggregation
Apply systematic confidence aggregation across the claim tree:

TREE AGGREGATION RULES:
- Main branches (≥0.7 confidence): Full weight in final synthesis
- Supporting branches (0.5-0.69): Moderate weight, clearly labeled
- Tentative branches (<0.5): Include with uncertainty qualifiers
- Conflicting branches: Present both with evidence comparison
- Converging branches: Boost confidence by 0.1 when they support each other

HIERARCHICAL CONFIDENCE CALCULATION:
Branch_Confidence = Weighted_Average(Claims_in_Branch × Evidence_Weight)
Overall_Confidence = Weighted_Average(Branch_Confidences × Branch_Importance)

Step 4: Reasoning Path Exploration
- Trace multiple reasoning paths through the claim tree
- Identify the strongest logical chain (highest confidence path)
- Explore alternative paths and their implications
- Note where different paths converge or diverge
- Select the most coherent and well-supported path as primary

Step 5: Structural Tree Organization
- Organize answer following the claim tree structure
- Start with highest-confidence main branches
- Develop supporting branches with clear hierarchy
- Include tentative branches with appropriate uncertainty
- Address conflicting branches with balanced presentation
- Ensure logical flow from trunk to leaves

Step 6: Completeness Verification
- Verify all aspects of original task are addressed in the tree
- Check for missing branches or underdeveloped reasoning paths
- Identify areas where evidence is insufficient for strong claims
- Note limitations and boundary conditions of the tree structure
- Suggest areas where the tree could grow with more evidence

Step 7: Quality Tree Pruning
- Remove weak branches that don't contribute to understanding
- Strengthen connections between related branches
- Ensure confidence scores reflect evidence quality
- Verify logical coherence throughout the tree structure
- Review for clarity, accuracy, and appropriate uncertainty expression

TREE-OF-THOUGHT SYNTHESIS EXAMPLES:

Example 1: Strong Tree Structure
Task: "What are the health benefits of regular exercise?"
Claim Tree:
- Main Branch: Cardiovascular benefits (confidence 0.85)
  - Sub-branch: Disease risk reduction (confidence 0.80)
  - Sub-branch: Blood pressure improvement (confidence 0.75)
- Main Branch: Mental health benefits (confidence 0.80)
  - Sub-branch: Depression reduction (confidence 0.70)
  - Sub-branch: Cognitive function (confidence 0.65)

Step 1: Tree = Clear hierarchical structure with strong main branches
Step 2: Evidence = Strong empirical support for main branches
Step 3: Confidence = Overall confidence 0.82 (strong tree structure)
Step 4: Paths = Cardiovascular → Mental health with supporting sub-branches
Step 5: Structure = Organized by benefit type with evidence hierarchy
Step 6: Completeness = Covers major health domains
Step 7: Quality = Well-supported tree with appropriate confidence

Synthesis Result: Overall confidence 0.82, comprehensive tree structure

Example 2: Tree with Conflicting Branches
Task: "Will quantum computing break current encryption?"
Claim Tree:
- Main Branch: Quantum threat to RSA (confidence 0.75)
  - Sub-branch: Shor's algorithm effectiveness (confidence 0.80)
- Main Branch: Post-quantum cryptography (confidence 0.85)
  - Sub-branch: Lattice-based solutions (confidence 0.80)
- Tentative Branch: Timeline uncertainty (confidence 0.40)

Step 1: Tree = Two strong main branches, one tentative branch
Step 2: Evidence = Theoretical threat vs practical countermeasures
Step 3: Confidence = Overall confidence 0.68 (conflicting branches)
Step 4: Paths = Threat → Countermeasures → Timeline uncertainty
Step 5: Structure = Balanced presentation of conflicting branches
Step 6: Completeness = Covers technical and practical aspects
Step 7: Quality = Appropriate uncertainty for conflicting claims

Synthesis Result: Overall confidence 0.65, acknowledges tree conflicts

ENHANCED SYNTHESIS REQUIREMENTS:

1. **Tree-Based Reasoning**: Apply all 7 synthesis steps using tree-of-thought approach
2. **Confidence Calibration**: Use hierarchical aggregation and conflict resolution
3. **Evidence-Based**: Base all conclusions on validated claim trees
4. **Logical Coherence**: Ensure proper tree structure and reasoning paths
5. **Comprehensive Coverage**: Address all aspects of original task in tree structure
6. **Transparency**: Clearly indicate confidence levels and tree uncertainties

Available Context:
{{relevant_context}}

Please provide a comprehensive tree-based synthesis in XML format:

<synthesis_result>
  <claim_tree_structure>Description of how claims were organized into hierarchical tree structure</claim_tree_structure>
  <evidence_forest_analysis>Analysis of evidence quality and connections across tree branches</evidence_forest_analysis>
  <confidence_tree_aggregation>
    <aggregated_confidence>Overall confidence score with tree-based calculation</aggregated_confidence>
    <tree_confidence_breakdown>How confidence was aggregated across tree levels</tree_confidence_breakdown>
    <branch_conflict_resolution>How conflicting tree branches were resolved</branch_conflict_resolution>
  </confidence_tree_aggregation>
  <reasoning_path_exploration>Analysis of multiple reasoning paths through the claim tree</reasoning_path_exploration>
  <structural_tree_organization>Description of answer organization following tree structure</structural_tree_organization>
  <completeness_verification>Assessment that all task aspects are covered in the tree</completeness_verification>
  <answer>Direct, comprehensive answer following tree-of-thought structure</answer>
  <reasoning>Detailed step-by-step tree-based synthesis reasoning</reasoning>
  <key_findings>Most important insights and conclusions from tree analysis</key_findings>
  <evidence_used>Summary of evidence incorporated across tree branches</evidence_used>
  <uncertainties_and_limitations>Areas of uncertainty and tree structure limitations</uncertainties_and_limitations>
  <recommendations>Actionable recommendations and further tree growth suggestions</recommendations>
  <synthesis_quality>Self-assessment of tree completeness and accuracy</synthesis_quality>
</synthesis_result>''',
            variables=[
                {
                    'name': 'original_task',
                    'type': 'string',
                    'required': True,
                    'description': 'Original task to be addressed'
                },
                {
                    'name': 'analysis_results',
                    'type': 'string',
                    'required': True,
                    'description': 'Results from claim analysis'
                },
                {
                    'name': 'claims_evaluated',
                    'type': 'string',
                    'required': True,
                    'description': 'Evaluated claims with evidence'
                },
                {
                    'name': 'relevant_context',
                    'type': 'string',
                    'required': False,
                    'description': 'Relevant context information'
                }
            ]
        )
    
    def _create_task_decomposition_template_xml(self) -> PromptTemplate:
        """Create enhanced XML-optimized task decomposition template with tree-of-thought reasoning"""
        return PromptTemplate(
            id="task_decomposition_xml",
            name="Enhanced XML Task Decomposition Template with Tree-of-Thought",
            description="XML-structured template with tree-of-thought reasoning for breaking complex tasks into subtasks",
            template_type=PromptTemplateType.TASK_DECOMPOSITION,
            template_content='''You are Conjecture, an advanced AI reasoning system that decomposes complex tasks into manageable subtasks using systematic tree-of-thought reasoning processes.

<decomposition_task>
{{complex_task}}</decomposition_task>

TREE-OF-THOUGHT TASK DECOMPOSITION PROCESS:
Follow this 6-step systematic decomposition process with hierarchical reasoning:

Step 1: Task Analysis and Root Identification
- Parse the complex task into its core components and objectives
- Identify the main goal or root objective that drives the entire task
- Extract key constraints, requirements, and success criteria
- Recognize implicit assumptions and dependencies
- Determine the scope and boundaries of what needs to be accomplished

Step 2: Hierarchical Task Tree Construction
- Break down the root objective into primary branches (main subtasks)
- Further decompose primary branches into secondary branches (detailed subtasks)
- Continue decomposition until reaching atomic, actionable tasks
- Ensure each branch represents a distinct, logical component
- Maintain clear parent-child relationships in the task tree

Step 3: Dependency Mapping and Sequencing
- Identify dependencies between different branches and subtasks
- Determine which branches can be executed in parallel vs sequentially
- Map critical path that determines overall project timeline
- Identify potential bottlenecks or blocking dependencies
- Establish logical flow and execution order

Step 4: Resource and Complexity Assessment
- Evaluate each subtask for required resources, skills, and tools
- Assess complexity level and estimated effort for each branch
- Identify subtasks that require specialized expertise or external resources
- Note potential risks or challenges for each task branch
- Assign confidence levels to decomposition accuracy (0.7-0.9 typical)

Step 5: Completeness and Validation Verification
- Verify that all task tree branches collectively cover the original task
- Check for gaps, overlaps, or redundancies in the decomposition
- Ensure each leaf node represents an actionable, testable subtask
- Validate that the task tree maintains logical coherence
- Confirm that success criteria are preserved in the decomposition

Step 6: Optimization and Refinement
- Optimize the task tree for efficiency and clarity
- Combine related subtasks where beneficial
- Split complex subtasks where they become more manageable
- Ensure each subtask has clear success criteria and deliverables
- Review and refine the overall tree structure for optimal execution

TREE-OF-THOUGHT DECOMPOSITION EXAMPLES:

Example 1: Software Development Task
Complex Task: "Build a machine learning recommendation system"

Step 1: Root = ML recommendation system with user personalization
Step 2: Task Tree:
  - Data Collection Branch
    - User data gathering (confidence 0.85)
    - Item data collection (confidence 0.90)
    - Data cleaning pipeline (confidence 0.80)
  - Model Development Branch
    - Algorithm selection (confidence 0.75)
    - Model training (confidence 0.85)
    - Hyperparameter tuning (confidence 0.70)
  - System Integration Branch
    - API development (confidence 0.80)
    - Frontend integration (confidence 0.75)
    - Performance testing (confidence 0.85)

Step 3: Dependencies = Data → Model → Integration (sequential)
Step 4: Resources = Data scientists, ML engineers, web developers
Step 5: Completeness = All aspects covered, no gaps
Step 6: Optimization = Clear deliverables for each subtask

Decomposition Result: 3 main branches, 9 atomic subtasks, overall confidence 0.81

Example 2: Research Task
Complex Task: "Analyze the impact of remote work on employee productivity"

Step 1: Root = Remote work productivity analysis with multiple factors
Step 2: Task Tree:
  - Literature Review Branch
    - Academic studies search (confidence 0.90)
    - Industry reports analysis (confidence 0.85)
    - Meta-analysis compilation (confidence 0.75)
  - Data Collection Branch
    - Survey design and distribution (confidence 0.80)
    - Productivity metrics collection (confidence 0.85)
    - Qualitative interviews (confidence 0.70)
  - Analysis Branch
    - Statistical correlation analysis (confidence 0.80)
    - Factor identification (confidence 0.75)
    - Recommendation formulation (confidence 0.70)

Step 3: Dependencies = Literature → Data → Analysis (sequential)
Step 4: Resources = Research team, survey tools, statistical software
Step 5: Completeness = Comprehensive coverage of research methodology
Step 6: Optimization = Clear research methodology with defined outputs

Decomposition Result: 3 main branches, 8 atomic subtasks, overall confidence 0.79

ENHANCED TASK DECOMPOSITION REQUIREMENTS:

1. **Tree-Based Reasoning**: Apply all 6 decomposition steps systematically
2. **Hierarchical Structure**: Create clear parent-child relationships in task tree
3. **Dependency Clarity**: Explicitly map dependencies and execution order
4. **Completeness Assurance**: Ensure task tree covers all aspects of original task
5. **Actionable Subtasks**: Each leaf node must be testable and executable
6. **Confidence Assessment**: Provide confidence levels for decomposition accuracy

Available Context:
{{relevant_context}}

Please break down the task into XML-structured hierarchical subtasks:

<task_decomposition_result>
  <task_analysis>Detailed breakdown of root objective and key components</task_analysis>
  <hierarchical_tree_structure>
    <main_branches>Primary task branches with confidence levels</main_branches>
    <subtask_hierarchy>Detailed breakdown into atomic, actionable subtasks</subtask_hierarchy>
    <tree_depth>Number of levels in the task tree</tree_depth>
    <total_subtasks>Total count of atomic subtasks identified</total_subtasks>
  </hierarchical_tree_structure>
  <dependency_mapping>
    <sequential_dependencies>Subtasks that must be executed in order</sequential_dependencies>
    <parallel_opportunities>Subtasks that can be executed simultaneously</parallel_opportunities>
    <critical_path>Critical path that determines overall timeline</critical_path>
  </dependency_mapping>
  <resource_assessment>
    <skill_requirements>Skills and expertise needed for each branch</skill_requirements>
    <tool_requirements>Tools, software, or resources required</tool_requirements>
    <complexity_levels>Complexity assessment for each subtask</complexity_levels>
    <risk_factors>Potential challenges or bottlenecks</risk_factors>
  </resource_assessment>
  <completeness_verification>
    <coverage_analysis>How task tree covers original requirements</coverage_analysis>
    <gap_identification>Any missing components or overlooked aspects</gap_identification>
    <redundancy_check>Overlapping or redundant subtasks identified</redundancy_check>
    <validation_status>Confirmation that decomposition is complete and logical</validation_status>
  </completeness_verification>
  <optimization_summary>
    <efficiency_improvements>Optimizations made to task tree structure</efficiency_improvements>
    <clarity_enhancements>Improvements to subtask definitions and deliverables</clarity_enhancements>
    <overall_confidence>Confidence level in decomposition accuracy (0.7-0.9)</overall_confidence>
  </optimization_summary>
  <subtasks>
    <!-- XML structure for each atomic subtask -->
    <subtask id="[subtask_id]">
      <branch>Main task branch this subtask belongs to</branch>
      <objective>Clear, specific objective for this subtask</objective>
      <deliverables>Specific outputs or results expected</deliverables>
      <success_criteria>Measurable criteria for subtask completion</success_criteria>
      <dependencies>Other subtasks that must be completed first</dependencies>
      <estimated_complexity>Complexity level (low/medium/high) and effort</estimated_complexity>
      <required_resources>Skills, tools, and resources needed</required_resources>
      <confidence_level>Confidence in subtask definition and feasibility (0.7-0.9)</confidence_level>
    </subtask>
  </subtasks>
  <decomposition_summary>
    <approach_description>Summary of tree-of-thought decomposition approach</approach_description>
    <key_decisions>Important decisions made during decomposition process</key_decisions>
    <execution_recommendations>Recommended execution order and strategy</execution_recommendations>
    <quality_assessment>Self-assessment of decomposition quality and completeness</quality_assessment>
  </decomposition_summary>
</task_decomposition_result>''',
            variables=[
                {
                    'name': 'complex_task',
                    'type': 'string',
                    'required': True,
                    'description': 'Complex task to decompose'
                },
                {
                    'name': 'relevant_context',
                    'type': 'string',
                    'required': False,
                    'description': 'Relevant context information'
                }
            ]
        )
    
    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """Get XML-optimized template by ID"""
        return self.templates.get(template_id)
    
    def list_templates(self) -> List[PromptTemplate]:
        """List all XML-optimized templates"""
        return list(self.templates.values())
    
    def create_optimized_prompt(
        self, 
        original_prompt: str, 
        template_id: str, 
        variables: Dict[str, Any]
    ) -> OptimizedPrompt:
        """
        Create an optimized prompt using XML templates
        """
        try:
            template = self.get_template(template_id)
            if not template:
                raise ValueError(f"Template {template_id} not found")
            
            # Render template with variables
            optimized_prompt = template.template_content
            
            # Apply template-specific optimizations
            if template_id == "research_xml":
                optimized_prompt = self._optimize_for_research_xml(optimized_prompt, variables)
            elif template_id == "analysis_xml":
                optimized_prompt = self._optimize_for_analysis_xml(optimized_prompt, variables)
            elif template_id == "validation_xml":
                optimized_prompt = self._optimize_for_validation_xml(optimized_prompt, variables)
            elif template_id == "synthesis_xml":
                optimized_prompt = self._optimize_for_synthesis_xml(optimized_prompt, variables)
            elif template_id == "task_decomposition_xml":
                optimized_prompt = self._optimize_for_decomposition_xml(optimized_prompt, variables)
            
            return OptimizedPrompt(
                original_prompt=original_prompt,
                optimized_prompt=optimized_prompt,
                optimization_strategy="xml_structure",
                changes_made=["XML formatting", "template_optimization"]
            )
            
        except Exception as e:
            # Return original prompt if optimization fails
            return OptimizedPrompt(
                original_prompt=original_prompt,
                optimized_prompt=original_prompt,
                optimization_strategy="failed",
                changes_made=[]
            )
    
    def _optimize_for_research_xml(
        self, prompt: str, variables: Dict[str, Any]
    ) -> str:
        """Optimize research prompt for XML structure"""
        # Add XML declaration and encoding
        xml_header = '<?xml version="1.0" encoding="UTF-8"?>'
        
        # Replace template variables with XML-safe content
        optimized_prompt = prompt
        for var_name, var_value in variables.items():
            if var_name == "user_query":
                # Escape XML special characters in user input
                safe_value = str(var_value).replace('&', '&').replace('<', '<').replace('>', '>')
                optimized_prompt = optimized_prompt.replace(f'{{{{{var_name}}}}', safe_value)
        
        # Ensure proper XML structure
        optimized_prompt = f"{xml_header}\n{optimized_prompt}"
        return optimized_prompt
    
    def _optimize_for_analysis_xml(
        self, prompt: str, variables: Dict[str, Any]
    ) -> str:
        """Optimize analysis prompt for XML structure"""
        # Add XML declaration
        xml_header = '<?xml version="1.0" encoding="UTF-8"?>'
        
        # Structure claims in XML format
        claims_var = variables.get('claims_for_analysis', '')
        if claims_var:
            # Convert claims to XML structure
            claims_xml = self._convert_claims_to_xml(claims_var)
            optimized_prompt = prompt.replace(f'{{{{{variables["claims_for_analysis"]}}}', claims_xml)
        
        # Add XML declaration and structure
        optimized_prompt = f"{xml_header}\n{optimized_prompt}"
        return optimized_prompt
    
    def _optimize_for_validation_xml(
        self, prompt: str, variables: Dict[str, Any]
    ) -> str:
        """Optimize validation prompt for XML structure"""
        xml_header = '<?xml version="1.0" encoding="UTF-8"?>'
        
        claim_var = variables.get('claim_to_validate', '')
        if claim_var:
            # Escape XML special characters in claim content
            safe_claim = str(claim_var).replace('&', '&').replace('<', '<').replace('>', '>').replace('"', '"')
            optimized_prompt = prompt.replace(f'{{{{{variables["claim_to_validate"]}}}', safe_claim)
        
        optimized_prompt = f"{xml_header}\n{optimized_prompt}"
        return optimized_prompt
    
    def _optimize_for_synthesis_xml(
        self, prompt: str, variables: Dict[str, Any]
    ) -> str:
        """Optimize synthesis prompt for XML structure"""
        xml_header = '<?xml version="1.0" encoding="UTF-8"?>'
        
        # Replace variables with XML-safe content
        for var_name, var_value in variables.items():
            if var_name in ['original_task', 'analysis_results', 'claims_evaluated', 'relevant_context']:
                # Escape XML special characters
                safe_value = str(var_value).replace('&', '&').replace('<', '<').replace('>', '>').replace('"', '"')
                optimized_prompt = optimized_prompt.replace(f'{{{{{var_name}}}}}', safe_value)
        
        optimized_prompt = f"{xml_header}\n{optimized_prompt}"
        return optimized_prompt
    
    def _optimize_for_decomposition_xml(
        self, prompt: str, variables: Dict[str, Any]
    ) -> str:
        """Optimize decomposition prompt for XML structure"""
        xml_header = '<?xml version="1.0" encoding="UTF-8"?>'
        
        task_var = variables.get('complex_task', '')
        if task_var:
            # Escape XML special characters
            safe_task = str(task_var).replace('&', '&').replace('<', '<').replace('>', '>').replace('"', '"')
            optimized_prompt = prompt.replace(f'{{{{{variables["complex_task"]}}}', safe_task)
        
        optimized_prompt = f"{xml_header}\n{optimized_prompt}"
        return optimized_prompt
    
    def _convert_claims_to_xml(self, claims_text: str) -> str:
        """Convert claims text to XML structure"""
        # Simple claim to XML conversion (would need parser for production)
        lines = claims_text.split('\n')
        xml_claims = []
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if line and line.startswith('[c') and line.endswith(']'):
                # Extract claim ID, type, confidence
                match = re.match(r'\[c(\d+)\)\s*\[([^\]]+)\]\s*\[([^\]]+)\]\s*\[([0-9\.]+)\](.*)', line)
                if match:
                    claim_id, claim_type, confidence, content = match.groups()
                    xml_claim = f'    <claim id="{claim_id}" type="{claim_type}" confidence="{confidence}">{content}</claim>'
                    xml_claims.append(xml_claim)
        
        return '\n'.join(xml_claims)