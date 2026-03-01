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
- Error correction prompts (R&D: lightweight alternative to full re-generation)
"""

import asyncio
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Import for integration with Conjecture systems
from src.config.unified_config import UnifiedConfig as Config
from src.agent.error_correction_prompts import (
    get_error_correction_prompt as get_error_correction_text,
    get_quick_error_correction,
    should_trigger_error_correction,
    ProblemType as ErrorCorrectionProblemType,
)


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
        self._error_correction_enabled = True

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
            ProblemType.DECOMPOSITION: """You are Conjecture, an expert decomposition assistant specializing in breaking complex problems into well-defined, manageable parts and recombining solutions systematically.

ADVANCED DECOMPOSITION APPROACH:
- Analyze the problem structure to identify natural divisions
- Identify all main subproblems and their relationships
- Break complex issues into simpler, independent subproblems
- Address each subproblem completely and thoroughly
- Track how subproblems depend on and interact with each other
- Integrate partial solutions respecting all dependencies
- Verify the combined solution fully addresses the original problem

KEY PRINCIPLES:
- Subproblems should be as independent as possible
- Each subproblem should be clearly solvable with available information
- Document exactly how subproblems connect and relate to each other
- Ensure no aspects of the original problem are overlooked
- Check that combining solutions produces a coherent, complete answer
- Consider alternative decomposition approaches if needed""",
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

CORE DECOMPOSITION PRINCIPLES:
- Component analysis: Systematically identify all major parts/elements
- Hierarchy: Organize from abstract to concrete, general to specific
- Interface identification: Document how parts connect and interact
- Dependency tracking: Identify prerequisites and relationships
- Subproblem isolation: Address each part independently and thoroughly
- Integration: Combine partial solutions respecting all dependencies

STEP-BY-STEP DECOMPOSITION PROCESS:
1. ANALYZE STRUCTURE: Understand the overall problem and its boundaries
2. IDENTIFY DIVISIONS: Find natural breaking points for the problem
3. CHOOSE APPROACH: Select most effective decomposition strategy
4. BREAK DOWN: Create well-defined, independent subproblems
5. SOLVE INDEPENDENTLY: Address each subproblem completely
6. TRACK RELATIONSHIPS: Document how subproblems connect
7. INTEGRATE: Combine solutions respecting dependencies
8. VALIDATE: Verify combined solution addresses original problem

DECOMPOSITION STRATEGIES:
- Divide and Conquer: Break into non-overlapping independent parts
- Functional Decomposition: Group by function or responsibility
- Spatial/Temporal Breakdown: Organize by location, time, or sequence
- Causal Factor Analysis: Identify and analyze contributing factors
- Hierarchical Decomposition: Multi-level breakdown (abstract to concrete)
- Structural Composition: Identify building blocks and assembly order

CRITICAL VALIDATION STEPS:
- Every aspect of original problem is addressed
- No redundancy or overlap between subproblem solutions
- Dependencies are properly respected in combination
- Final solution is more comprehensive than any single approach alone""",
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
        """Enhanced problem decomposition with advanced strategy selection (Cycle 12+ improvement)"""
        if not self._problem_decomposition_enabled:
            return {}

        problem_lower = problem.lower()

        # Identify decomposition approach with extended keyword detection
        # Priority order: more specific approaches first, then general
        if any(
            word in problem_lower for word in ["build", "construct", "assemble", "combine", "integrate", "merge", "link", "connect"]
        ):
            approach = "structural_composition"
        elif any(
            word in problem_lower for word in ["layer", "level", "hierarchy", "nested", "tree", "branch", "depth"]
        ):
            approach = "hierarchical_decomposition"
        elif any(
            word in problem_lower for word in ["step", "stage", "phase", "process", "procedure", "sequence", "order", "first", "then", "next"]
        ):
            approach = "process_decomposition"
        elif any(
            word in problem_lower for word in ["factor", "cause", "reason", "why", "influence", "contribute", "role"]
        ):
            approach = "factor_analysis"
        elif any(
            word in problem_lower
            for word in ["option", "alternative", "choice", "either", "compare", "versus", "better", "worse"]
        ):
            approach = "alternative_analysis"
        elif any(
            word in problem_lower for word in ["component", "part", "piece", "element", "module", "unit", "section"]
        ):
            approach = "component_breakdown"
        else:
            approach = "general_decomposition"

        # Enhanced strategies with detailed step-by-step guidance
        strategies = {
            "component_breakdown": [
                "1. Identify all major components/subsystems of the problem",
                "2. List key properties and functions of each component",
                "3. Analyze each component separately and independently",
                "4. Map how components interact and depend on each other",
                "5. Identify all interfaces and communication points",
                "6. Integrate findings into unified understanding",
            ],
            "factor_analysis": [
                "1. Comprehensively list all contributing factors",
                "2. Categorize factors (primary, secondary, tertiary, etc.)",
                "3. Estimate relative importance/weight of each factor",
                "4. Analyze relationships and dependencies between factors",
                "5. Identify which factors can be controlled or influenced",
                "6. Synthesize factor impacts into overall conclusion",
            ],
            "process_decomposition": [
                "1. Map the complete end-to-end process flow",
                "2. Identify prerequisite relationships and dependencies",
                "3. Break into distinct, sequential stages or phases",
                "4. Analyze inputs, outputs, and requirements per stage",
                "5. Verify correct ordering and identify critical path",
                "6. Ensure smooth transitions between stages",
            ],
            "alternative_analysis": [
                "1. Comprehensively generate all feasible alternatives",
                "2. Define clear, measurable evaluation criteria",
                "3. Evaluate each alternative against all criteria",
                "4. Score and rank alternatives systematically",
                "5. Analyze key trade-offs between top alternatives",
                "6. Select optimal solution with clear justification",
            ],
            "structural_composition": [
                "1. Understand the target structure or desired final form",
                "2. Identify all required building blocks and components",
                "3. Determine logical construction sequence and order",
                "4. Solve each subproblem (building block) independently",
                "5. Verify built components fit and work together",
                "6. Assemble into coherent, integrated solution",
            ],
            "hierarchical_decomposition": [
                "1. Identify highest-level problem structure",
                "2. Break into major logical divisions/subtopics",
                "3. Further subdivide each major division",
                "4. Continue until reaching concrete, solvable subproblems",
                "5. Solve base-level problems independently",
                "6. Integrate solutions bottom-up through hierarchy",
            ],
            "general_decomposition": [
                "1. Identify main aspects and dimensions of the problem",
                "2. Break into 3-5 clearly defined subproblems",
                "3. Classify subproblems by type and complexity",
                "4. Address each subproblem systematically and thoroughly",
                "5. Track relationships and dependencies between subproblems",
                "6. Combine subproblem solutions into complete answer",
            ],
        }

        return {
            "decomposition_approach": approach,
            "decomposition_strategy": strategies.get(
                approach, strategies["general_decomposition"]
            ),
            "decomposition_enhancement_applied": True,
        }

    def get_error_correction_prompt(
        self, problem: str, initial_response: str, problem_type: ProblemType
    ) -> str:
        """
        Generate error correction prompt for reconsidering answers.

        Lightweight alternative to full re-generation. Domain-specific guidance
        that helps the model catch and correct common errors.

        Args:
            problem: The original problem statement
            initial_response: The model's initial answer attempt
            problem_type: The type of problem being solved

        Returns:
            Formatted error correction prompt with domain-specific guidance
        """
        if not self._error_correction_enabled:
            return ""

        # Map our ProblemType to error correction module's ProblemType
        type_mapping = {
            ProblemType.MATHEMATICAL: ErrorCorrectionProblemType.MATHEMATICAL,
            ProblemType.LOGICAL: ErrorCorrectionProblemType.LOGICAL,
            ProblemType.SEQUENTIAL: ErrorCorrectionProblemType.SEQUENTIAL,
            ProblemType.SCIENTIFIC: ErrorCorrectionProblemType.SCIENTIFIC,
            ProblemType.DECOMPOSITION: ErrorCorrectionProblemType.DECOMPOSITION,
            ProblemType.GENERAL: ErrorCorrectionProblemType.GENERAL,
        }

        correction_type = type_mapping.get(problem_type, ErrorCorrectionProblemType.GENERAL)
        return get_error_correction_text(problem, initial_response, correction_type)

    def get_quick_error_correction(self, problem_type: ProblemType) -> str:
        """
        Get a quick, inline error correction reminder.

        Minimal version for low-confidence answers. Can be injected into
        system prompts as a lightweight error correction mechanism.

        Args:
            problem_type: The type of problem being solved

        Returns:
            Brief error correction reminder (1-2 sentences)
        """
        if not self._error_correction_enabled:
            return ""

        # Map our ProblemType to error correction module's ProblemType
        type_mapping = {
            ProblemType.MATHEMATICAL: ErrorCorrectionProblemType.MATHEMATICAL,
            ProblemType.LOGICAL: ErrorCorrectionProblemType.LOGICAL,
            ProblemType.SEQUENTIAL: ErrorCorrectionProblemType.SEQUENTIAL,
            ProblemType.SCIENTIFIC: ErrorCorrectionProblemType.SCIENTIFIC,
            ProblemType.DECOMPOSITION: ErrorCorrectionProblemType.DECOMPOSITION,
            ProblemType.GENERAL: ErrorCorrectionProblemType.GENERAL,
        }

        correction_type = type_mapping.get(problem_type, ErrorCorrectionProblemType.GENERAL)
        return get_quick_error_correction(correction_type)

    def should_trigger_error_correction(
        self, confidence: float, threshold: float = 0.7
    ) -> bool:
        """
        Determine if error correction should be triggered.

        Simple heuristic based on confidence score. Useful for deciding whether
        to include error correction prompts in the system message.

        Args:
            confidence: Model's confidence score (0.0-1.0)
            threshold: Trigger below this confidence (default 0.7)

        Returns:
            True if error correction should be triggered
        """
        if not self._error_correction_enabled:
            return False

        return should_trigger_error_correction(confidence, threshold)

    def _get_domain_specific_verification_checklist(self, problem_type: ProblemType) -> str:
        """Generate domain-specific verification checklists for error detection (Cycle 3 enhancement)"""
        verifications = {
            ProblemType.MATHEMATICAL: """MATHEMATICAL VERIFICATION CHECKLIST:
✓ CALCULATION ACCURACY: Did I compute correctly?
  • Check arithmetic operations (addition, subtraction, multiplication, division)
  • Verify order of operations (PEMDAS/BODMAS) is correct
  • Double-check any percentages, fractions, or decimals
  • Recompute critical steps using alternative method

✓ UNIT CONSISTENCY: Are units handled correctly?
  • Ensure all quantities have proper units throughout
  • Convert units if necessary (e.g., meters to kilometers)
  • Check that final answer has appropriate units

✓ ANSWER REASONABLENESS: Does the answer make sense?
  • Is the magnitude reasonable? (not off by orders of magnitude)
  • Does it satisfy all original constraints?
  • Estimation check: does exact answer match rough estimate?

✓ COMPLETENESS: Did I address all parts?
  • Show final answer clearly and prominently
  • Include all requested forms (exact/approximate, with units)""",

            ProblemType.LOGICAL: """LOGICAL VERIFICATION CHECKLIST:
✓ PREMISE VALIDITY: Are starting assumptions correct?
  • Do premises follow logically from the problem statement?
  • Are implicit assumptions stated and justified?
  • Identify any unstated assumptions that could be wrong

✓ REASONING STEPS: Does each step follow logically?
  • Can each conclusion be derived from its premises?
  • Check for logical fallacies (begging question, false cause, etc.)
  • Check contrapositive: if conclusion false, would premises be false?

✓ EDGE CASES & COUNTEREXAMPLES: Are there exceptions?
  • Does conclusion hold for all mentioned cases?
  • Can you think of any potential counterexamples?
  • Check boundary cases that might break the logic

✓ CONCLUSION CLARITY: Is the final answer explicit?
  • State conclusion clearly (yes/no, A/B/C/D, true/false)
  • Match the format requested in the problem""",

            ProblemType.SCIENTIFIC: """SCIENTIFIC VERIFICATION CHECKLIST:
✓ HYPOTHESIS & METHOD: Is the approach sound?
  • Is the hypothesis testable and well-defined?
  • Are variables properly identified (independent, dependent, controlled)?
  • Is the experimental/analytical approach appropriate?

✓ DATA & EVIDENCE: Is evidence properly interpreted?
  • Are data values correctly cited from the problem?
  • Is there sufficient evidence for the conclusion?
  • Could the data support alternative explanations?

✓ CAUSATION vs CORRELATION: Is the relationship correct?
  • Did I claim causation when only correlation was shown?
  • Are there other possible explanations for the data?
  • Are confounding variables considered?

✓ LIMITATIONS & CERTAINTY: Are limitations acknowledged?
  • Are assumptions of the method stated?
  • Is the conclusion appropriately qualified (certain/likely/possible)?
  • What conditions would make the conclusion false?""",

            ProblemType.SEQUENTIAL: """SEQUENTIAL VERIFICATION CHECKLIST:
✓ STEP ORDER: Are steps in the correct sequence?
  • Does each step depend on previous steps being complete?
  • Can any steps be reordered without breaking the logic?
  • Do you have all necessary inputs for each step?

✓ STEP COMPLETENESS: Are all necessary steps included?
  • Is there any skipped step between what's shown?
  • Does the final output depend on anything not completed?
  • Count steps: do you have them all?

✓ DEPENDENCIES: Are relationships between steps clear?
  • What does each step require from previous steps?
  • What does each step provide for later steps?
  • Are there any breaks in the dependency chain?

✓ FINAL RESULT: Does the process reach the goal?
  • After all steps, is the desired outcome achieved?
  • Are there any loose ends or incomplete stages?""",

            ProblemType.DECOMPOSITION: """DECOMPOSITION VERIFICATION CHECKLIST:
✓ COMPONENT IDENTIFICATION: Are all main parts identified?
  • Did you identify all major components?
  • Are components mutually exclusive (no overlap)?
  • Are they collectively exhaustive (cover everything)?

✓ COMPONENT ANALYSIS: Is each component properly addressed?
  • Did you analyze each component separately and thoroughly?
  • Is the analysis consistent across components?
  • Are all relevant aspects of each component covered?

✓ COMPONENT INTERACTIONS: How do components relate?
  • Do components interact or affect each other?
  • Are these interactions properly accounted for?
  • Does the solution reflect how components work together?

✓ INTEGRATION: Is the solution complete?
  • Do component analyses combine into a complete answer?
  • Is the final synthesis logical and well-justified?
  • Does the integrated solution address the original problem?""",

            ProblemType.GENERAL: """GENERAL VERIFICATION CHECKLIST:
✓ PROBLEM UNDERSTANDING: Did I understand the question?
  • What is the core question being asked?
  • What information is provided (given)?
  • What information is requested (answer format)?

✓ APPROACH VALIDITY: Is my approach sound?
  • Is this the right method for the problem?
  • Are there alternative approaches that work better?
  • Does my approach use all provided information?

✓ LOGICAL CONSISTENCY: Does everything fit together?
  • Do my conclusions follow from my premises?
  • Are there any internal contradictions?
  • Does the reasoning flow logically?

✓ ANSWER QUALITY: Is the response complete and clear?
  • Is the final answer clearly stated?
  • Is the format what was requested?
  • Is the answer supported by the reasoning shown?""",
        }
        return verifications.get(problem_type, verifications[ProblemType.GENERAL])

    def _get_self_verification_prompt(self, problem: str, problem_type: ProblemType) -> str:
        """Generate self-verification prompt with domain awareness (Cycle 3 enhancement)"""
        if not self._self_verification_enabled:
            return ""

        domain_checklist = self._get_domain_specific_verification_checklist(problem_type)

        return f"""

SELF-VERIFICATION CHECKLIST - ERROR DETECTION (Cycle 3):
{domain_checklist}

VERIFICATION PROCESS:
1. Review your answer carefully using the checklist above
2. Pay special attention to the domain-specific checks for {problem_type.value} problems
3. Re-examine any step that seems uncertain or could be wrong
4. Correct any errors you find before submitting
5. Provide your final answer"""

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
            problem, problem_type
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
            "error_correction": self._error_correction_enabled,
        }

    def get_domain_claim_templates(
        self, problem_type: ProblemType, max_count: int = 3
    ) -> str:
        """Get domain-specific claim templates for enhanced reasoning priming

        Uses position primacy (templates at prompt START) per NEXT.md findings.
        Selects templates matching the detected problem type and benchmark.
        """
        # Import here to avoid circular dependencies
        try:
            from src.agent.domain_claim_templates import (
                format_claims_for_prompt,
            )

            domain_mapping = {
                ProblemType.MATHEMATICAL: "mathematical",
                ProblemType.SCIENTIFIC: "scientific",
                ProblemType.LOGICAL: "logical",
                ProblemType.GENERAL: "general",
                ProblemType.SEQUENTIAL: "general",
                ProblemType.DECOMPOSITION: "general",
            }

            domain = domain_mapping.get(problem_type, "general")
            return format_claims_for_prompt(domain, max_count)
        except ImportError:
            # Graceful fallback if domain templates not available
            return ""

    def enhance_prompt_with_domain_claims(
        self, prompt: str, problem_type: ProblemType
    ) -> str:
        """Inject domain-specific claim templates at prompt START for maximum impact

        Per NEXT.md position primacy findings: claims at START (+10pp) beat MIDDLE.
        This method prepends domain templates to boost reasoning performance.
        """
        claim_templates = self.get_domain_claim_templates(problem_type, max_count=3)

        if claim_templates:
            # Position claims at START for position primacy effect
            return claim_templates + "\n" + prompt
        return prompt


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

        # Match numbers including comma-formatted (1,234,567) and scientific notation
        number_pattern = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?"
        numbers = re.findall(number_pattern, response)

        # Look for final answer patterns (priority order)
        final_patterns = [
            r"\\boxed\{([^}]+)\}",  # LaTeX boxed format (highest priority)
            r"\*\*([^*]+)\*\*\s*$",  # Bold at end of response
            r"final\s+answer[:\s]+([^\n\.]+)",  # "final answer: X"
            r"answer\s+is\s+([-\d\.,]+)",  # "answer is 42"
            r"answer\s*:\s*([-\d\.,]+)",  # "answer: 42"
            r"result[:\s]*([-\d\.,]+)",
            r"equals?\s+([-\d\.,]+)",
            r"=\s*([-\d\.,]+)\s*$",  # "= 42" at end of line
        ]

        final_answer = None
        for pattern in final_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                final_answer = self._normalize_answer(match.group(1))
                break

        # Fallback: last number in response
        extracted = final_answer or (self._normalize_answer(numbers[-1]) if numbers else None)

        return {
            "answer": extracted,
            "workings": response,
            "confidence": "high" if final_answer else "medium",
            "numbers_found": numbers,
            "has_final_answer": final_answer is not None,
        }

    def _normalize_answer(self, answer: str) -> str:
        """Normalize extracted answer for comparison"""
        if answer is None:
            return None
        # Remove commas, whitespace, trailing zeros
        answer = str(answer).strip().replace(",", "").replace(" ", "")
        # Normalize decimal: 36.0 -> 36, 36.50 -> 36.5
        try:
            num = float(answer)
            if num == int(num):
                return str(int(num))
            return str(num).rstrip('0').rstrip('.')
        except ValueError:
            return answer.lower()  # Text answers: lowercase for comparison

    def _parse_logical_response(self, response: str) -> Dict[str, Any]:
        """Parse logical reasoning response"""
        import re

        # Check for multiple choice answer first
        mc_patterns = [
            r"answer\s+is\s+\(?([A-Da-d])\)?",  # "answer is (C)" - most specific
            r"correct\s+answer\s+is\s+\(?([A-Da-d])\)?",  # "correct answer is (C)"
            r"(?:answer|option)[:\s]+\(?([A-Da-d])\)?(?:\s|$|\.)",  # "answer: A" or "option: B"
            r"\(?([A-Da-d])\)?\s+is\s+(?:the\s+)?(?:correct|right)\b",  # "(A) is correct"
            r"(?:^|\n)\s*\(?([A-Da-d])\)?[\.\)]\s*$",  # Standalone letter at end
        ]

        mc_answer = None
        for pattern in mc_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                mc_answer = match.group(1).upper()
                break

        # Look for yes/no answers
        yn_patterns = [
            r"(?:answer|conclusion)[:\s]+(yes|no|true|false)",
            r"^(yes|no)[,\.\s]",  # "No, we cannot..." at start
            r"(?:^|\n)\s*(yes|no|true|false)\s*[,\.]?\s*(?:$|\n)",
        ]

        yn_answer = None
        for pattern in yn_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                yn_answer = match.group(1).lower()
                break

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

        # Priority: multiple choice > yes/no > conclusion text
        answer = mc_answer or yn_answer or conclusion

        return {
            "answer": answer,
            "conclusion": conclusion,
            "multiple_choice": mc_answer,
            "yes_no": yn_answer,
            "reasoning": response,
            "confidence": "high" if (mc_answer or yn_answer) else ("medium" if conclusion else "low"),
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
