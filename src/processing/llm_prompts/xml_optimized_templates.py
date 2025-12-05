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
    
    def _create_analysis_template_xml(self) -> PromptTemplate:
        """Create XML-optimized analysis template"""
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
        """Create XML-optimized synthesis template"""
        return PromptTemplate(
            id="synthesis_xml",
            name="XML Synthesis Template",
            description="XML-structured template for synthesizing comprehensive answers",
            template_type=PromptTemplateType.SYNTHESIS,
            template_content=''''''You are Conjecture, an AI system that synthesizes comprehensive answers using evidence-based reasoning.

<synthesis_task>
Synthesize a comprehensive answer to the following task using the provided analysis and context:

<original_task>{{original_task}}</original_task>

<analysis_results>{{analysis_results}}</analysis_results>

<claims_evaluated>{{claims_evaluated}}</claims_evaluated>

SYNTHESIS REQUIREMENTS:
1. Evidence Integration
   - Incorporate all validated claims and their evidence
   - Use highest-confidence claims as foundation
   - Address contradictions between claims systematically

2. Logical Structure
   - Build answer in clear, logical progression
   - Start with direct answer, then provide supporting reasoning
   - Use transition words for smooth flow

3. Confidence Scoring
   - Assign overall confidence based on evidence quality
   - Be transparent about uncertainty levels
   - Provide confidence intervals when appropriate

4. Completeness
   - Address all aspects of original task
   - Provide actionable recommendations when relevant
   - Note limitations and knowledge gaps

Available Context:
{{relevant_context}}

Please provide a comprehensive synthesis in XML format:

<synthesis_result>
  <answer>Direct answer to the task</answer>
  <confidence>Overall confidence level (0.0-1.0)</confidence>
  <reasoning>Step-by-step synthesis reasoning</reasoning>
  <key_findings>Key insights and conclusions</key_findings>
  <evidence_used>Summary of evidence incorporated</evidence_used>
  <recommendations>Actionable recommendations</recommendations>
</synthesis_result>'''''',
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
        """Create XML-optimized task decomposition template"""
        return PromptTemplate(
            id="task_decomposition_xml",
            name="XML Task Decomposition Template",
            description="XML-structured template for breaking complex tasks into subtasks",
            template_type=PromptTemplateType.TASK_DECOMPOSITION,
            template_content=''''''You are Conjecture, an AI system that decomposes complex tasks into manageable subtasks.

<decomposition_task>
{{complex_task}}</decomposition_task>

DECOMPOSITION REQUIREMENTS:
1. Logical Breakdown
   - Analyze task structure and identify main components
   - Break task into 3-5 logical subtasks
   - Ensure subtasks are actionable and independent
   - Maintain clear dependencies between subtasks

2. Subtask Specification
   - Each subtask should have clear objective
   - Include required resources and context
   - Specify expected deliverables
   - Maintain logical sequencing

3. Completeness
   - Subtasks should collectively address original task
   - No gaps or overlaps between subtasks
   - Each subtask should be testable

Available Context:
{{relevant_context}}

Please break down the task into XML-structured subtasks:

<subtasks>
  <subtask id="1">
    <objective>Clear objective for first subtask</objective>
    <required_resources>Resources needed for this subtask</required_resources>
    <expected_deliverables>What this subtask should produce</expected_deliverables>
    <dependencies>Dependencies on other subtasks</dependencies>
  </subtask>
  
  <!-- Add more subtasks as needed -->
</subtasks>

<decomposition_summary>
Summary of the decomposition approach and key decisions.
</decomposition_summary>'''''',
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