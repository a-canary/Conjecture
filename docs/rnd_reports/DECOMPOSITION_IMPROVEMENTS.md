# Multi-Step Problem Decomposition Improvements (Cycle 12+)

## Overview
Enhanced the problem decomposition capabilities in `src/agent/prompt_system.py` to provide more sophisticated strategies for breaking complex problems into manageable subproblems and recombining solutions.

## Key Improvements

### 1. Extended Decomposition Type Detection
Added detection for two new decomposition approaches plus extended keyword coverage:

#### New Decomposition Types:
- **structural_composition**: For problems involving building/constructing/assembling components
  - Keywords: build, construct, assemble, combine, integrate, merge, link, connect

- **hierarchical_decomposition**: For problems involving organizational or conceptual hierarchies
  - Keywords: layer, level, hierarchy, nested, tree, branch, depth

#### Enhanced Existing Types:
- **component_breakdown**: Added keywords (module, unit, section)
- **factor_analysis**: Added keywords (influence, contribute, role)
- **process_decomposition**: Extended with (procedure, sequence, order, first, then, next)
- **alternative_analysis**: Added comparison keywords (compare, versus, better, worse)

### 2. Detailed Step-by-Step Strategies
Replaced generic 4-step strategies with numbered 6-step strategies for each approach:

**Component Breakdown:**
1. Identify all major components/subsystems of the problem
2. List key properties and functions of each component
3. Analyze each component separately and independently
4. Map how components interact and depend on each other
5. Identify all interfaces and communication points
6. Integrate findings into unified understanding

**Factor Analysis:**
1. Comprehensively list all contributing factors
2. Categorize factors (primary, secondary, tertiary, etc.)
3. Estimate relative importance/weight of each factor
4. Analyze relationships and dependencies between factors
5. Identify which factors can be controlled or influenced
6. Synthesize factor impacts into overall conclusion

**Process Decomposition:**
1. Map the complete end-to-end process flow
2. Identify prerequisite relationships and dependencies
3. Break into distinct, sequential stages or phases
4. Analyze inputs, outputs, and requirements per stage
5. Verify correct ordering and identify critical path
6. Ensure smooth transitions between stages

**Alternative Analysis:**
1. Comprehensively generate all feasible alternatives
2. Define clear, measurable evaluation criteria
3. Evaluate each alternative against all criteria
4. Score and rank alternatives systematically
5. Analyze key trade-offs between top alternatives
6. Select optimal solution with clear justification

**Structural Composition (NEW):**
1. Understand the target structure or desired final form
2. Identify all required building blocks and components
3. Determine logical construction sequence and order
4. Solve each subproblem (building block) independently
5. Verify built components fit and work together
6. Assemble into coherent, integrated solution

**Hierarchical Decomposition (NEW):**
1. Identify highest-level problem structure
2. Break into major logical divisions/subtopics
3. Further subdivide each major division
4. Continue until reaching concrete, solvable subproblems
5. Solve base-level problems independently
6. Integrate solutions bottom-up through hierarchy

**General Decomposition:**
1. Identify main aspects and dimensions of the problem
2. Break into 3-5 clearly defined subproblems
3. Classify subproblems by type and complexity
4. Address each subproblem systematically and thoroughly
5. Track relationships and dependencies between subproblems
6. Combine subproblem solutions into complete answer

### 3. Enhanced Decomposition Context

Updated the decomposition context in `_get_context_for_problem_type()` to include:

**Core Decomposition Principles:**
- Component analysis: Systematically identify all major parts/elements
- Hierarchy: Organize from abstract to concrete, general to specific
- Interface identification: Document how parts connect and interact
- Dependency tracking: Identify prerequisites and relationships
- Subproblem isolation: Address each part independently and thoroughly
- Integration: Combine partial solutions respecting all dependencies

**8-Step Decomposition Process:**
1. ANALYZE STRUCTURE: Understand the overall problem and its boundaries
2. IDENTIFY DIVISIONS: Find natural breaking points for the problem
3. CHOOSE APPROACH: Select most effective decomposition strategy
4. BREAK DOWN: Create well-defined, independent subproblems
5. SOLVE INDEPENDENTLY: Address each subproblem completely
6. TRACK RELATIONSHIPS: Document how subproblems connect
7. INTEGRATE: Combine solutions respecting dependencies
8. VALIDATE: Verify combined solution addresses original problem

**Extended Strategy List:**
- Divide and Conquer: Break into non-overlapping independent parts
- Functional Decomposition: Group by function or responsibility
- Spatial/Temporal Breakdown: Organize by location, time, or sequence
- Causal Factor Analysis: Identify and analyze contributing factors
- Hierarchical Decomposition: Multi-level breakdown (abstract to concrete)
- Structural Composition: Identify building blocks and assembly order

**Critical Validation Steps:**
- Every aspect of original problem is addressed
- No redundancy or overlap between subproblem solutions
- Dependencies are properly respected in combination
- Final solution is more comprehensive than any single approach alone

### 4. Improved Domain-Adaptive Prompt

Enhanced the DECOMPOSITION problem type prompt to emphasize:
- Analysis of problem structure to find natural divisions
- Making subproblems as independent as possible
- Thorough documentation of subproblem relationships
- Proper tracking of dependencies
- Verification that combined solution addresses original problem
- Consideration of alternative decomposition approaches

### 5. Better Enhancement Presentation

Updated the `process_with_context()` method to present decomposition enhancements more clearly:
- Shows decomposition approach used (e.g., "COMPONENT BREAKDOWN", "FACTOR ANALYSIS")
- Displays first step of strategy for context
- Includes decomposition approach name in enhancement info
- Makes it clear which decomposition strategy was selected

## File Changes

### `/workspace/src/agent/prompt_system.py`

1. **`_enhance_problem_decomposition()` method (lines 610-720)**
   - Extended keyword detection for 6 decomposition types
   - Enhanced strategies with numbered 6-step guidance
   - Added structural_composition and hierarchical_decomposition approaches

2. **`_get_context_for_problem_type()` method - DECOMPOSITION section (lines 414-443)**
   - Added core decomposition principles section
   - Added 8-step decomposition process
   - Extended strategy list with details
   - Critical validation steps documented

3. **`_get_domain_adaptive_prompt()` method - DECOMPOSITION section (lines 313-328)**
   - Enhanced problem type prompt for decomposition
   - Emphasized independence of subproblems
   - Added key principles for decomposition
   - Improved guidance on tracking dependencies

4. **`process_with_context()` method - enhancement presentation (lines 782-795)**
   - Improved decomposition enhancement display
   - Shows decomposition approach in output
   - Better formatting with bullet points
   - More informative strategy presentation

## Benefits

1. **Better Problem Analysis**: Users can choose the most appropriate decomposition strategy
2. **Clearer Guidance**: Numbered steps make it explicit what needs to be done
3. **Improved Integration**: Emphasis on dependencies and relationships between subproblems
4. **Comprehensive Coverage**: 6 distinct decomposition approaches handle different problem types
5. **Enhanced Learning**: Users understand both the strategy and the reasoning
6. **Better Results**: More structured approach leads to more complete solutions

## Integration with Existing System

- Fully backward compatible with existing prompt system
- Works with all problem type detection
- Integrates with difficulty estimation
- Compatible with other enhancement mechanisms (mathematical reasoning, multi-step reasoning)
- Uses existing caching and response mechanisms

## Testing

To verify the improvements:
```bash
python3 test_decomposition_direct.py
```

This script tests:
- Decomposition type detection accuracy
- Strategy improvement verification
- Context content availability
- Numbered step generation
