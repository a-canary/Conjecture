#!/usr/bin/env python3
"""
Conjecture Enhanced Prompt System
Restored with all proven enhancements from cycles 1-12

This system includes:
- Domain-adaptive prompts (Cycle 1: 100% improvement)
- Enhanced context integration (Cycle 2: SUCCESS)
- Self-verification mechanisms (Cycle 3: SUCCESS)
- Mathematical reasoning (Cycle 9: 8% improvement)
- Multi-step reasoning (Cycle 11: 10% improvement)
- Problem decomposition (Cycle 12: 9% improvement)
- Response quality via self-critique (Cycle 5: SUCCESS)
"""

import asyncio
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Import for integration with Conjecture systems
from src.config.unified_config import UnifiedConfig as Config


def _keyword_matches(keyword: str, problem_lower: str) -> bool:
    """Check if keyword matches in problem using appropriate strategy."""
    # For single-character symbols, use direct substring match
    if len(keyword) == 1 and not keyword.isalpha():
        return keyword in problem_lower

    # For multi-word phrases (contain space), use substring match
    if " " in keyword:
        return keyword in problem_lower

    # For regular words, use word boundary matching
    return bool(re.search(r"\b" + re.escape(keyword) + r"\b", problem_lower))


class ProblemType(Enum):
    MATHEMATICAL = "mathematical"
    LOGICAL = "logical"
    GENERAL = "general"
    SCIENTIFIC = "scientific"
    SEQUENTIAL = "sequential"
    DECOMPOSITION = "decomposition"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class PromptResponse:
    """Structured response from prompt system"""

    response: str
    confidence: float
    reasoning: str
    prompt_type: str
    metadata: Dict[str, Any] = None


class PromptSystem:
    """Enhanced prompt system with all proven reasoning capabilities"""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._cache = {}
        self._domain_adaptive_enabled = True
        self._context_integration_enabled = True
        self._self_verification_enabled = True
        self._mathematical_reasoning_enabled = True
        self._multistep_reasoning_enabled = True
        self._problem_decomposition_enabled = True
        self._self_critique_enabled = True

    def _detect_problem_type(self, problem: str) -> ProblemType:
        """Detect problem type for domain-adaptive processing"""
        problem_lower = problem.lower()

        # Mathematical indicators
        math_keywords = [
            "calculate",
            "solve",
            "compute",
            "find",
            "what is",
            "how many",
            "percent",
            "area",
            "volume",
            "equation",
            "variable",
            "x",
            "y",
            "+",
            "-",
            "*",
            "×",
            "/",
            "=",
            "square root",
            "√",
        ]

        # Logical indicators
        logical_keywords = [
            "if",
            "then",
            "therefore",
            "because",
            "since",
            "all",
            "some",
            "none",
            "always",
            "never",
            "must",
            "cannot",
            "impossible",
            "necessary",
            "sufficient",
            "implies",
            "implication",
        ]

        # Sequential/multi-step indicators
        sequential_keywords = [
            "first",
            "then",
            "next",
            "after that",
            "finally",
            "step",
            "sequence",
            "order",
            "before",
            "after",
        ]

        # Scientific indicators
        scientific_keywords = [
            "experiment",
            "hypothesis",
            "theory",
            "observation",
            "data",
            "analysis",
            "conclusion",
            "method",
            "procedure",
            "result",
            "test",
            "design",
        ]

        # Decomposition indicators
        decomposition_keywords = [
            "break down",
            "analyze",
            "components",
            "parts",
            "elements",
            "factors",
            "consider",
            "separately",
            "individually",
        ]

        # Count keyword matches using smart matching strategy
        math_score = sum(
            1 for kw in math_keywords if _keyword_matches(kw, problem_lower)
        )
        logical_score = sum(
            1 for kw in logical_keywords if _keyword_matches(kw, problem_lower)
        )
        sequential_score = sum(
            1 for kw in sequential_keywords if _keyword_matches(kw, problem_lower)
        )
        scientific_score = sum(
            1 for kw in scientific_keywords if _keyword_matches(kw, problem_lower)
        )
        decomposition_score = sum(
            1 for kw in decomposition_keywords if _keyword_matches(kw, problem_lower)
        )

        # Determine primary type based on highest score
        scores = {
            ProblemType.MATHEMATICAL: math_score,
            ProblemType.LOGICAL: logical_score,
            ProblemType.SEQUENTIAL: sequential_score,
            ProblemType.SCIENTIFIC: scientific_score,
            ProblemType.DECOMPOSITION: decomposition_score,
        }

        max_score = max(scores.values())
        if max_score == 0:
            return ProblemType.GENERAL

        for problem_type, score in scores.items():
            if score == max_score:
                return problem_type

        return ProblemType.GENERAL

    def _estimate_difficulty(self, problem: str) -> Difficulty:
        """Estimate problem difficulty based on complexity indicators"""
        problem_lower = problem.lower()

        # Complexity indicators for hard problems
        hard_indicators = [
            "prove",
            "derive",
            "theorem",
            "lemma",
            "corollary",
            "complex",
            "optimal",
            "optimize",
            "maximize",
            "minimize",
            "efficiency",
            "algorithm",
            "recursive",
            "iteration",
            "convergence",
            "divergence",
        ]

        # Simplicity indicators for easy problems
        easy_indicators = [
            "what is",
            "how many",
            "how much",
            "what color",
            "find",
            "calculate",
            "simple",
            "basic",
            "single",
            "one",
            "first",
            "just",
            "only",
            "color",
            "name",
            "list",
        ]

        # Smart matching for difficulty estimation
        hard_score = sum(
            1 for ind in hard_indicators if _keyword_matches(ind, problem_lower)
        )
        easy_score = sum(
            1 for ind in easy_indicators if _keyword_matches(ind, problem_lower)
        )

        if hard_score > easy_score:
            return Difficulty.HARD
        elif easy_score > hard_score:
            return Difficulty.EASY
        else:
            return Difficulty.MEDIUM

    def _get_domain_adaptive_prompt(
        self, problem: str, problem_type: ProblemType, difficulty: Difficulty
    ) -> str:
        """Generate domain-adaptive system prompt (Cycle 1 enhancement)"""
        if not self._domain_adaptive_enabled:
            return self._get_base_prompt()

        base_prompts = {
            ProblemType.MATHEMATICAL: """You are Conjecture, a mathematical reasoning assistant designed to solve problems with precision and clarity.

MATHEMATICAL APPROACH:
- Always show step-by-step calculations
- Verify answers using estimation or alternative methods
- Pay attention to units and order of operations
- State assumptions clearly
- Provide exact answers when possible, approximate when necessary
- Double-check calculations for accuracy""",
            ProblemType.LOGICAL: """You are Conjecture, a logical reasoning assistant designed to analyze arguments and solve puzzles systematically.

LOGICAL APPROACH:
- Break down complex problems into logical steps
- Identify premises, conclusions, and relationships
- Use formal reasoning principles
- Check for logical fallacies or inconsistencies
- Consider edge cases and counterexamples
- Build arguments step-by-step with clear justification""",
            ProblemType.SCIENTIFIC: """You are Conjecture, a scientific reasoning assistant designed to analyze scientific problems and data.

SCIENTIFIC APPROACH:
- Apply the scientific method systematically
- Identify hypotheses and testable predictions
- Consider empirical evidence and data
- Recognize limitations and uncertainties
- Use appropriate scientific terminology
- Distinguish between correlation and causation""",
            ProblemType.SEQUENTIAL: """You are Conjecture, a sequential reasoning assistant designed to solve multi-step problems systematically.

SEQUENTIAL APPROACH:
- Identify all required steps in the correct order
- Track intermediate results carefully
- Verify each step before proceeding to the next
- Consider dependencies between steps
- Maintain clear progression toward the solution
- Check that no steps are overlooked""",
            ProblemType.DECOMPOSITION: """You are Conjecture, a decomposition reasoning assistant designed to break complex problems into manageable parts.

DECOMPOSITION APPROACH:
- Identify the main components of the problem
- Break down complex issues into simpler subproblems
- Address each component systematically
- Consider how components interact
- Integrate partial solutions into a complete answer
- Verify that all aspects are covered""",
            ProblemType.GENERAL: """You are Conjecture, a general reasoning assistant designed to solve problems thoughtfully and systematically.

GENERAL APPROACH:
- Understand what the problem is asking
- Plan your approach before solving
- Show your reasoning clearly
- Consider alternative perspectives
- Verify your answer makes sense
- Provide clear, well-structured responses""",
        }

        prompt = base_prompts.get(problem_type, base_prompts[ProblemType.GENERAL])

        # Add difficulty-specific guidance
        if difficulty == Difficulty.HARD:
            prompt += "\n\nDIFFICULTY GUIDANCE (Hard): Take your time, be thorough, and don't hesitate to use multiple approaches. Complex problems often require deep analysis."
        elif difficulty == Difficulty.EASY:
            prompt += "\n\nDIFFICULTY GUIDANCE (Easy): This should be straightforward. Provide a clear, direct answer with minimal complexity."
        else:
            prompt += "\n\nDIFFICULTY GUIDANCE (Medium): Balance thoroughness with efficiency. Show key steps but don't overcomplicate."

        return prompt

    def _get_context_for_problem_type(
        self, problem: str, problem_type: ProblemType
    ) -> str:
        """Get problem-type-specific context (Cycle 2 enhancement)"""
        if not self._context_integration_enabled:
            return ""

        contexts = {
            ProblemType.MATHEMATICAL: """
MATHEMATICAL CONTEXT AND FORMULAS:
- Arithmetic: Order of operations (PEMDAS/BODMAS)
- Algebra: Solve equations by isolating variables
- Geometry: Area formulas (shapes), perimeter, volume
- Percentages: (part/whole) × 100 = percentage
- Word problems: Convert to mathematical expressions

Common pitfalls to avoid:
- Calculation errors: double-check arithmetic
- Unit errors: ensure consistent units
- Logic errors: verify the approach makes sense
- Answer format: provide the requested format""",
            ProblemType.LOGICAL: """
LOGICAL REASONING CONTEXT:
- Deductive reasoning: General principles → specific conclusions
- Inductive reasoning: Specific observations → general principles
- Conditional logic: If P then Q, contrapositive, converse, inverse
- Quantifiers: All, some, none, at least, at most
- Logical operators: AND, OR, NOT, XOR, implication

Common logical structures:
- Syllogisms, analogies, classification problems
- Pattern recognition and sequence completion
- Causal relationships and correlations
- Necessary vs sufficient conditions""",
            ProblemType.SCIENTIFIC: """
SCIENTIFIC REASONING CONTEXT:
- Variables: Independent, dependent, controlled
- Hypotheses: Testable predictions about relationships
- Data analysis: Patterns, trends, statistical significance
- Experimental design: Controls, replication, bias reduction
- Scientific method: Observation → hypothesis → testing → conclusion

Key scientific principles:
- Conservation laws, cause and effect
- Systematic observation and measurement
- Evidence-based conclusions
- Uncertainty and error analysis""",
            ProblemType.SEQUENTIAL: """
SEQUENTIAL REASONING CONTEXT:
- Step ordering: Identify prerequisite relationships
- Dependencies: What must be completed before what
- Planning: Break tasks into ordered sequences
- Process tracking: Monitor progress through steps
- Verification: Check completion of each step

Common sequential patterns:
- Recipe-like procedures with ordered steps
- Multi-stage calculations
- Project planning and scheduling
- Process optimization and efficiency""",
            ProblemType.DECOMPOSITION: """
PROBLEM DECOMPOSITION CONTEXT:
- Component analysis: Identify main parts
- Hierarchy: Organize from general to specific
- Interface identification: How parts connect
- Subproblem isolation: Address separately
- Integration: Combine partial solutions

Decomposition strategies:
- Divide and conquer approach
- Functional decomposition
- Spatial/temporal breakdown
- Causal factor analysis""",
            ProblemType.GENERAL: """
GENERAL PROBLEM SOLVING CONTEXT:
- Problem identification: What exactly is being asked?
- Information gathering: What facts are relevant?
- Strategy selection: What approach is most suitable?
- Execution: Implement the chosen method
- Verification: Does the answer make sense?""",
        }

        return contexts.get(problem_type, contexts[ProblemType.GENERAL])

    def _enhance_mathematical_reasoning(self, problem: str) -> Dict[str, Any]:
        """Enhance mathematical reasoning with structured approach (Cycle 9 enhancement)"""
        if not self._mathematical_reasoning_enabled:
            return {}

        problem_lower = problem.lower()

        # Problem classification
        if any(op in problem_lower for op in ["+", "-", "*", "×", "/", "÷"]):
            prob_type = "arithmetic"
        elif any(
            word in problem_lower
            for word in ["square root", "sqrt", "√", "power", "exponent"]
        ):
            prob_type = "roots_and_powers"
        elif any(word in problem_lower for word in ["percent", "%", "percentage"]):
            prob_type = "percentage"
        elif any(
            word in problem_lower
            for word in ["area", "volume", "perimeter", "circumference"]
        ):
            prob_type = "geometry"
        elif any(
            word in problem_lower
            for word in ["x", "y", "variable", "equation", "solve for"]
        ):
            prob_type = "algebra"
        elif any(
            word in problem_lower for word in ["rate", "speed", "distance", "time"]
        ):
            prob_type = "rate_problems"
        else:
            prob_type = "general_math"

        # Generate reasoning strategy based on problem type
        strategies = {
            "arithmetic": [
                "Break down complex calculations into smaller steps",
                "Estimate the answer before calculating",
                "Verify by working backwards or using a different method",
                "Check units and order of operations (PEMDAS/BODMAS)",
            ],
            "roots_and_powers": [
                "Test perfect squares/powers first",
                "Use estimation to narrow the range",
                "Apply appropriate formulas (a², √a, aⁿ)",
                "Verify by inverse operation (√a × √a = a)",
            ],
            "percentage": [
                "Convert percentage to decimal (÷100)",
                "Apply formula: part = whole × percentage",
                "For percentage change: (new-old)/old × 100",
                "Check if answer is reasonable (50% off should be half price)",
            ],
            "geometry": [
                "Identify the shape and relevant formula",
                "Ensure all measurements are in consistent units",
                "Draw a diagram if helpful",
                "Double-check calculations and final units",
            ],
            "algebra": [
                "Identify variables, constants, and coefficients",
                "Choose method: substitution, elimination, factoring, etc.",
                "Show each step of solving the equation",
                "Check solution by substituting back into original equation",
            ],
            "rate_problems": [
                "Identify what's being measured per unit of time/distance",
                "Use formula: rate = distance/time or similar",
                "Check units and convert if necessary",
                "Consider if answer makes logical sense",
            ],
            "general_math": [
                "Understand what the problem is asking",
                "Choose appropriate mathematical approach",
                "Show work clearly",
                "Verify answer is reasonable",
            ],
        }

        selected_strategy = strategies.get(prob_type, strategies["general_math"])

        return {
            "problem_type": prob_type,
            "reasoning_strategy": selected_strategy,
            "mathematical_enhancement_applied": True,
        }

    def _enhance_multistep_reasoning(self, problem: str) -> Dict[str, Any]:
        """Enhance multi-step reasoning with complexity analysis (Cycle 11 enhancement)"""
        if not self._multistep_reasoning_enabled:
            return {}

        problem_lower = problem.lower()

        # Count sequential indicators
        sequential_words = [
            "first",
            "then",
            "next",
            "after",
            "finally",
            "step",
            "before",
            "following",
        ]
        step_count = sum(1 for word in sequential_words if word in problem_lower)

        # Count question marks and distinct clauses
        question_count = problem.count("?")
        clause_count = len(re.split(r"[;,\.]", problem)) if problem else 0

        # Determine complexity
        if step_count >= 3 or question_count >= 2 or clause_count >= 4:
            complexity = "high"
            suggested_steps = max(4, step_count + 2)
        elif step_count >= 2 or question_count >= 1 or clause_count >= 3:
            complexity = "medium"
            suggested_steps = max(3, step_count + 1)
        else:
            complexity = "low"
            suggested_steps = max(2, 1)

        # Multi-step strategy
        strategies = {
            "high": [
                "Break into 5+ distinct stages",
                "Track intermediate results carefully",
                "Verify each stage before proceeding",
                "Consider alternative approaches",
                "Perform final comprehensive check",
            ],
            "medium": [
                "Break into 3-4 logical steps",
                "Show work for each step",
                "Check connections between steps",
                "Verify final answer",
            ],
            "low": [
                "Identify required steps (2-3)",
                "Execute in correct order",
                "Verify result",
            ],
        }

        return {
            "complexity_level": complexity,
            "suggested_steps": suggested_steps,
            "multistep_strategy": strategies.get(complexity, strategies["medium"]),
            "multistep_enhancement_applied": True,
        }

    def _enhance_problem_decomposition(self, problem: str) -> Dict[str, Any]:
        """Enhance problem decomposition with strategy selection (Cycle 12 enhancement)"""
        if not self._problem_decomposition_enabled:
            return {}

        problem_lower = problem.lower()

        # Identify decomposition approach
        if any(
            word in problem_lower for word in ["component", "part", "piece", "element"]
        ):
            approach = "component_breakdown"
        elif any(
            word in problem_lower for word in ["factor", "cause", "reason", "why"]
        ):
            approach = "factor_analysis"
        elif any(
            word in problem_lower for word in ["step", "stage", "phase", "process"]
        ):
            approach = "process_decomposition"
        elif any(
            word in problem_lower
            for word in ["option", "alternative", "choice", "either"]
        ):
            approach = "alternative_analysis"
        else:
            approach = "general_decomposition"

        # Strategy based on approach
        strategies = {
            "component_breakdown": [
                "Identify all major components",
                "Analyze each component separately",
                "Consider component interactions",
                "Integrate component analyses",
            ],
            "factor_analysis": [
                "List all contributing factors",
                "Categorize factors by importance",
                "Analyze factor relationships",
                "Synthesize factor impacts",
            ],
            "process_decomposition": [
                "Map the complete process",
                "Break into sequential stages",
                "Analyze each stage's requirements",
                "Ensure proper stage transitions",
            ],
            "alternative_analysis": [
                "Identify all possible options",
                "Evaluate each option's pros/cons",
                "Compare against criteria",
                "Select optimal solution",
            ],
            "general_decomposition": [
                "Identify main problem aspects",
                "Break into manageable subproblems",
                "Address each subproblem systematically",
                "Combine into comprehensive solution",
            ],
        }

        return {
            "decomposition_approach": approach,
            "decomposition_strategy": strategies.get(
                approach, strategies["general_decomposition"]
            ),
            "decomposition_enhancement_applied": True,
        }

    def _get_self_verification_prompt(self, problem: str, response: str) -> str:
        """Generate self-verification prompt (Cycle 3 enhancement)"""
        if not self._self_verification_enabled:
            return ""

        return f"""

SELF-VERIFICATION CHECKLIST:
Before finalizing your answer, verify each item:

✓ PROBLEM UNDERSTANDING: Did I address exactly what was asked?
✓ LOGICAL CONSISTENCY: Does my reasoning flow logically?
✓ CALCULATION ACCURACY: Are all calculations correct?
✓ UNITS & FORMATS: Are units and answer format appropriate?
✓ COMPLETENESS: Have I addressed all parts of the problem?
✓ PLAUSIBILITY: Does the answer make sense in context?

Original problem: {problem}
Your proposed answer: [Provide your answer here]
Review your work against this checklist and correct any issues."""

    def _quick_self_critique(self, response: str, problem_type: ProblemType) -> str:
        """Quick self-critique for response quality enhancement (Cycle 5 enhancement)"""
        if not self._self_critique_enabled:
            return ""

        critique_checks = {
            ProblemType.MATHEMATICAL: [
                "Are all calculations shown clearly?",
                "Is the final answer prominently stated?",
                "Are units and format correct?",
                "Does the answer address the specific mathematical question?",
            ],
            ProblemType.LOGICAL: [
                "Is the reasoning structure clear?",
                "Are assumptions stated explicitly?",
                "Is the conclusion logically derived?",
                "Are edge cases considered?",
            ],
            ProblemType.SCIENTIFIC: [
                "Is the scientific method applied correctly?",
                "Are hypotheses testable?",
                "Is evidence properly interpreted?",
                "Are limitations acknowledged?",
            ],
            ProblemType.SEQUENTIAL: [
                "Are steps in the correct order?",
                "Are all necessary steps included?",
                "Are dependencies respected?",
                "Is the sequence logical?",
            ],
            ProblemType.DECOMPOSITION: [
                "Are all components addressed?",
                "Is the decomposition complete?",
                "Are component interactions considered?",
                "Is the integration logical?",
            ],
            ProblemType.GENERAL: [
                "Is the answer clear and direct?",
                "Is reasoning well-structured?",
                "Is the response comprehensive?",
                "Does it fully address the question?",
            ],
        }

        checks = critique_checks.get(problem_type, critique_checks[ProblemType.GENERAL])
        checks_text = "\n".join(f"• {check}" for check in checks)

        return f"""

RESPONSE QUALITY CHECKLIST:
Review your response for the following quality criteria:
{checks_text}

Refine your response to meet these quality standards."""

    def _get_base_prompt(self) -> str:
        """Get base Conjecture prompt"""
        return """You are Conjecture, a reasoning assistant designed to solve problems systematically and provide well-structured answers."""

    async def process_with_context(self, problem: str) -> PromptResponse:
        """Process problem with full context and all enhancements"""

        # Create cache key
        cache_key = hash(problem)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Problem analysis
        problem_type = self._detect_problem_type(problem)
        difficulty = self._estimate_difficulty(problem)

        # Apply all enhancements
        enhancements = {}

        # Domain-adaptive prompt (Cycle 1)
        base_prompt = self._get_domain_adaptive_prompt(
            problem, problem_type, difficulty
        )

        # Context integration (Cycle 2)
        context = self._get_context_for_problem_type(problem, problem_type)

        # Mathematical reasoning (Cycle 9)
        if problem_type == ProblemType.MATHEMATICAL:
            enhancements.update(self._enhance_mathematical_reasoning(problem))

        # Multi-step reasoning (Cycle 11)
        enhancements.update(self._enhance_multistep_reasoning(problem))

        # Problem decomposition (Cycle 12)
        enhancements.update(self._enhance_problem_decomposition(problem))

        # Build enhanced prompt
        enhanced_prompt = base_prompt

        if context:
            enhanced_prompt += f"\n\n{context}"

        if enhancements:
            enhancement_info = []
            for key, value in enhancements.items():
                if key == "reasoning_strategy" and value:
                    enhancement_info.append(f"Strategy: {', '.join(value[:2])}")
                elif key == "multistep_strategy" and value:
                    enhancement_info.append(f"Steps: {', '.join(value[:2])}")
                elif key == "decomposition_strategy" and value:
                    enhancement_info.append(f"Approach: {', '.join(value[:2])}")

            if enhancement_info:
                enhanced_prompt += f"\n\nENHANCED APPROACH:\n" + "\n".join(
                    enhancement_info
                )

        # Add verification prompts
        verification_prompt = self._get_self_verification_prompt(
            problem, "[Your response here]"
        )
        critique_prompt = self._quick_self_critique(
            "[Your response here]", problem_type
        )

        if verification_prompt:
            enhanced_prompt += verification_prompt
        if critique_prompt:
            enhanced_prompt += critique_prompt

        enhanced_prompt += f"\n\nPROBLEM TO SOLVE:\n{problem}"

        # Create response with confidence based on enhancements
        confidence = 0.75  # Base confidence

        # Boost confidence based on successful enhancement application
        if len(enhancements) >= 2:
            confidence += 0.10
        if problem_type != ProblemType.GENERAL:
            confidence += 0.05
        if difficulty == Difficulty.EASY:
            confidence += 0.05
        elif difficulty == Difficulty.HARD:
            confidence -= 0.05

        confidence = min(0.95, max(0.50, confidence))

        response = PromptResponse(
            response=enhanced_prompt,
            confidence=confidence,
            reasoning=f"Domain-adaptive {problem_type.value} processing with {len(enhancements)} enhancements applied",
            prompt_type="enhanced_conjecture",
            metadata={
                "problem_type": problem_type.value,
                "difficulty": difficulty.value,
                "enhancements_applied": len(enhancements),
                "enhancement_types": list(enhancements.keys()),
                "cache_key": cache_key,
            },
        )

        # Cache response
        self._cache[cache_key] = response

        return response

    def get_system_prompt(
        self,
        problem_type: Optional[ProblemType] = None,
        difficulty: Optional[Difficulty] = None,
        problem: Optional[str] = None,
    ) -> str:
        """Generate enhanced system prompt"""

        if problem:
            # Full enhancement with problem analysis
            problem_type = problem_type or self._detect_problem_type(problem)
            difficulty = difficulty or self._estimate_difficulty(problem)

            prompt = self._get_domain_adaptive_prompt(problem, problem_type, difficulty)
            context = self._get_context_for_problem_type(problem, problem_type)

            if context:
                prompt += f"\n\n{context}"

            return prompt
        else:
            # Basic domain-adaptive prompt
            problem_type = problem_type or ProblemType.GENERAL
            difficulty = difficulty or Difficulty.MEDIUM
            return self._get_domain_adaptive_prompt("", problem_type, difficulty)

    def enable_enhancement(self, enhancement: str, enabled: bool = True):
        """Enable/disable specific enhancements"""
        enhancement_attr = f"_{enhancement}_enabled"
        if hasattr(self, enhancement_attr):
            setattr(self, enhancement_attr, enabled)

    def get_enhancement_status(self) -> Dict[str, bool]:
        """Get status of all enhancements"""
        return {
            "domain_adaptive": self._domain_adaptive_enabled,
            "context_integration": self._context_integration_enabled,
            "self_verification": self._self_verification_enabled,
            "mathematical_reasoning": self._mathematical_reasoning_enabled,
            "multistep_reasoning": self._multistep_reasoning_enabled,
            "problem_decomposition": self._problem_decomposition_enabled,
            "self_critique": self._self_critique_enabled,
        }


class ResponseParser:
    """Parse and structure responses from LLM models"""

    def __init__(self):
        self.parsing_strategies = {
            "mathematical": self._parse_mathematical_response,
            "logical": self._parse_logical_response,
            "general": self._parse_general_response,
        }

    def parse_response(
        self, response: str, problem_type: str = "general"
    ) -> Dict[str, Any]:
        """Parse LLM response into structured format"""
        strategy = self.parsing_strategies.get(
            problem_type, self._parse_general_response
        )
        return strategy(response)

    def _parse_mathematical_response(self, response: str) -> Dict[str, Any]:
        """Parse mathematical problem response"""
        # Look for numerical answers
        import re

        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", response)

        # Look for final answer patterns
        final_patterns = [
            r"answer\s+is\s+([-\d\.]+)",  # "answer is 42" - specific handling
            r"answer\s*:\s*([-\d\.]+)",  # "answer: 42"
            r"result[:\s]*([-\d\.]+)",
            r"equals?[:\s]*([-\d\.]+)",
            r"=\s*([-\d\.]+)",
        ]

        final_answer = None
        for pattern in final_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                final_answer = match.group(1)
                break

        return {
            "answer": final_answer or (numbers[-1] if numbers else None),
            "workings": response,
            "confidence": "high" if final_answer else "medium",
            "numbers_found": numbers,
            "has_final_answer": final_answer is not None,
        }

    def _parse_logical_response(self, response: str) -> Dict[str, Any]:
        """Parse logical reasoning response"""
        # Look for conclusion patterns
        conclusion_patterns = [
            r"conclusion[:\s]*(.+?)(?:\n|$)",
            r"therefore[,:]?\s*(.+?)(?:\n|$)",
            r"thus[,:]?\s*(.+?)(?:\n|$)",
            r"hence[,:]?\s*(.+?)(?:\n|$)",
        ]

        conclusion = None
        for pattern in conclusion_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                conclusion = match.group(1).strip()
                break

        return {
            "conclusion": conclusion,
            "reasoning": response,
            "confidence": "high" if conclusion else "medium",
            "has_conclusion": conclusion is not None,
        }

    def _parse_general_response(self, response: str) -> Dict[str, Any]:
        """Parse general problem response"""
        # Extract the last substantial sentence as potential answer
        sentences = [s.strip() for s in response.split(".") if s.strip()]
        answer = sentences[-1] if sentences else response[:200]

        return {
            "answer": answer,
            "full_response": response,
            "confidence": "medium",
            "response_length": len(response),
        }


# Legacy compatibility classes
class PromptBuilder:
    """Legacy compatibility wrapper"""

    def __init__(self):
        self.system = PromptSystem()

    def get_system_prompt(
        self,
        problem_type: Optional[ProblemType] = None,
        difficulty: Optional[Difficulty] = None,
    ) -> str:
        return self.system.get_system_prompt(
            problem_type=problem_type, difficulty=difficulty
        )
