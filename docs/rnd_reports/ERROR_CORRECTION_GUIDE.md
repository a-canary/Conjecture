# Error Correction Prompts - Implementation Guide

## Overview

Error correction prompts provide a **lightweight alternative to full re-generation** for catching and correcting model errors. Instead of asking the model to solve the problem from scratch again, we prompt it to reconsider its answer and check for common domain-specific errors.

**Key advantage**: Minimal overhead (typically 1-2 additional tokens/pass) compared to full regeneration (doubling inference cost)

## Strategy

1. **Model generates initial answer** with confidence score
2. **Evaluate confidence**: If below threshold (e.g., 0.7), trigger error correction
3. **Send error correction prompt** with domain-specific guidance
4. **Model reconsiders** and either confirms or corrects the answer
5. **Compare results**: Return the best answer

## Architecture

### Files Added

- **`src/agent/error_correction_prompts.py`** - Core error correction module
  - `get_error_correction_prompt()` - Full domain-specific correction prompt
  - `get_quick_error_correction()` - Lightweight inline reminder
  - `should_trigger_error_correction()` - Confidence-based triggering logic
  - `ErrorCorrectionConfig` - Configuration management

- **`src/agent/prompt_system.py`** - Integration with PromptSystem
  - `error_correction_enabled` flag
  - `get_error_correction_prompt()` method
  - `get_quick_error_correction()` method
  - `should_trigger_error_correction()` method
  - Added to enhancement status reporting

### Problem Types Covered

1. **Mathematical** - Calculation checks, order of operations, unit verification
2. **Logical** - Premise verification, reasoning chain validation, conclusion validity
3. **Sequential** - Step order, step completeness, dependency verification
4. **Scientific** - Methodology, evidence interpretation, conclusion support
5. **Decomposition** - Component identification, component analysis, integration
6. **General** - Understanding, content verification, sense checking

## Usage Patterns

### Pattern 1: Inline Error Correction (Lightweight)

Include a quick reminder in the system prompt for answers with lower confidence:

```python
from src.agent.prompt_system import PromptSystem, ProblemType

prompt_system = PromptSystem()
problem_type = ProblemType.MATHEMATICAL
confidence = 0.65

# Get quick reminder
if prompt_system.should_trigger_error_correction(confidence):
    quick_reminder = prompt_system.get_quick_error_correction(problem_type)
    system_prompt = f"""
{base_prompt}

SAFETY CHECK:
{quick_reminder}
"""
```

**Result for Mathematical**: "If your first answer seems wrong, reconsider: did you verify the calculation step-by-step and check for arithmetic errors?"

### Pattern 2: Full Error Correction (When Confidence Low)

Generate detailed correction prompt when model confidence is below threshold:

```python
from src.agent.prompt_system import PromptSystem, ProblemType

prompt_system = PromptSystem()
problem_type = ProblemType.LOGICAL
confidence = 0.6

if confidence < 0.7:
    correction_prompt = prompt_system.get_error_correction_prompt(
        problem="All humans are mortal. Socrates is human. Is Socrates mortal?",
        initial_response="Not necessarily.",
        problem_type=problem_type
    )
    # Send correction_prompt to model for reconsideration
```

**Result**: Detailed prompt with:
1. Premise verification checks
2. Reasoning chain validation
3. Conclusion validity assessment

### Pattern 3: Automatic Correction Workflow

Integrated workflow with automatic decision-making:

```python
from src.agent.prompt_system import PromptSystem, ProblemType

prompt_system = PromptSystem()

# Step 1: Get initial answer with confidence
initial_answer = model.generate(problem, confidence_score=0.65)

# Step 2: Check if correction needed
if prompt_system.should_trigger_error_correction(initial_answer.confidence):
    # Step 3: Generate correction prompt
    correction_prompt = prompt_system.get_error_correction_prompt(
        problem=problem,
        initial_response=initial_answer.text,
        problem_type=problem_type
    )

    # Step 4: Get corrected answer
    corrected_answer = model.generate(correction_prompt)

    # Step 5: Return best result
    final_answer = corrected_answer if corrected_answer.confidence > initial_answer.confidence else initial_answer
else:
    final_answer = initial_answer
```

## Configuration

### Default Settings

```python
from src.agent.error_correction_prompts import ErrorCorrectionConfig

config = ErrorCorrectionConfig(
    enabled=True,                    # Enable error correction
    confidence_threshold=0.7,        # Trigger below this confidence
    apply_to_all_domains=False,      # Apply selectively
    target_domains=[                 # Target these problem types
        ProblemType.MATHEMATICAL,
        ProblemType.LOGICAL,
        ProblemType.SEQUENTIAL,
    ],
    max_correction_attempts=1,       # Max correction loops
)
```

### Enable/Disable

```python
from src.agent.prompt_system import PromptSystem

prompt_system = PromptSystem()

# Check status
status = prompt_system.get_enhancement_status()
print(f"Error correction: {status['error_correction']}")

# Disable if needed
prompt_system.enable_enhancement('error_correction', False)
```

## Domain-Specific Error Checks

### Mathematical Problems

Checks for:
- Calculation errors (arithmetic, PEMDAS/BODMAS)
- Interpretation errors (units, format)
- Verification methods (working backwards, estimation)

Example:
```
ERROR CORRECTION STEP - Mathematical Check:

1. CALCULATION REVIEW: Did you double-check arithmetic?
   - Verify order of operations (PEMDAS/BODMAS)
   - Recalculate key steps
   - Check for sign errors (-/+) or decimal placement

2. LOGIC VERIFICATION: Did you interpret the problem correctly?
   - Does your answer match what was asked?
   - Did you use correct units?
   - Are assumptions valid?

3. ALTERNATIVE CHECK: Can you verify using a different method?
   - Try working backwards from your answer
   - Use estimation to check if answer is reasonable
   - Check for off-by-one errors
```

### Logical Problems

Checks for:
- Premise validity (facts, assumptions)
- Reasoning chain (logical steps, fallacies)
- Conclusion validity (counterexamples, alternatives)

### Sequential Problems

Checks for:
- Step order (correct sequence, prerequisites)
- Step completeness (no missing steps)
- Dependency verification (step relationships)

### Scientific Problems

Checks for:
- Methodology (scientific method application)
- Evidence interpretation (correlation vs causation)
- Conclusion support (evidence justification)

### Decomposition Problems

Checks for:
- Component identification (all major parts)
- Component analysis (thorough evaluation)
- Integration (proper combination)

## Performance Characteristics

### Token Cost

- **Quick correction**: ~50-100 tokens overhead
- **Full correction**: ~300-500 tokens overhead
- **Full regeneration**: 2x inference cost

### Effectiveness Metrics

Expected improvements based on domain:

| Domain | Expected Catch Rate | Precision |
|--------|-------------------|-----------|
| Mathematical | 30-40% | 70-80% |
| Logical | 25-35% | 65-75% |
| Sequential | 20-30% | 60-70% |
| Scientific | 15-25% | 55-65% |
| General | 10-20% | 50-60% |

### When to Apply

**Trigger error correction when:**
- Model confidence < 0.7
- Response contains uncertainty markers ("maybe", "might", "possibly")
- Response is unusually short (< 50 characters)
- Problem type is mathematical or logical
- Previous answer was incorrect (for retry scenarios)

**Skip error correction when:**
- Model confidence ≥ 0.8
- Task requires novel generation (not reconsideration)
- Error correction attempts already exhausted
- Problem type is factual recall (not reasoning)

## Examples

### Example 1: Catching Calculation Error

**Problem**: "If a rectangle has length 8 and width 5, what is its area?"

**Initial Answer** (confidence: 0.45): "The area is 8 + 5 = 13 square units."

**Error Correction Prompt**:
```
ERROR CORRECTION STEP - Mathematical Check:
Before finalizing your answer, consider if your solution may contain errors:

1. CALCULATION REVIEW: Did you double-check arithmetic?
   - Verify order of operations (PEMDAS/BODMAS)
   - Recalculate key steps
   - Check for sign errors (-/+) or decimal placement

[Full prompt...]

Your initial answer: The area is 8 + 5 = 13 square units.
```

**Corrected Answer** (confidence: 0.95): "The area is 8 × 5 = 40 square units."

### Example 2: Catching Logic Error

**Problem**: "All humans are mortal. Socrates is human. Is Socrates mortal?"

**Initial Answer** (confidence: 0.55): "Not necessarily, because the statement might not apply universally."

**Error Correction Prompt**:
```
ERROR CORRECTION STEP - Logic Check:
Before finalizing your conclusion, reconsider if your reasoning may contain errors:

1. PREMISE VERIFICATION: Are your starting assumptions sound?
   - Did you misread any facts?
   - Are there hidden assumptions?
   - Did you consider all given information?

[Full prompt...]
```

**Corrected Answer** (confidence: 0.98): "Yes, Socrates is definitely mortal. This is a valid deductive argument."

## Integration with Existing Systems

### With PromptSystem

```python
from src.agent.prompt_system import PromptSystem, ProblemType

prompt_system = PromptSystem()

# Get full system prompt with error correction
problem_type = prompt_system._detect_problem_type(problem)
system_prompt = prompt_system.get_system_prompt(problem=problem)

# Add error correction if needed
if prompt_system.should_trigger_error_correction(confidence):
    error_correction = prompt_system.get_quick_error_correction(problem_type)
    system_prompt += f"\n\nSAFETY CHECK:\n{error_correction}"
```

### With Self-Verification

Error correction complements self-verification:
- **Self-verification**: "Is your answer correct?" (broad check)
- **Error correction**: "Did you check for these specific errors?" (targeted check)

```python
from src.process.self_verification import SelfVerifier

verifier = SelfVerifier()

# Get initial answer
initial = model.generate(problem)

# Self-verify
verification_prompt = verifier.create_verification_prompt(problem, initial)
verification_response = model.generate(verification_prompt)

# If verification flags error, try error correction
if "error" in verification_response.lower():
    error_prompt = prompt_system.get_error_correction_prompt(
        problem, initial, problem_type
    )
    corrected = model.generate(error_prompt)
```

## Testing

Run the error correction tests:

```bash
python3 src/benchmarking/test_error_correction.py
```

Output:
```
Running Error Correction Tests
============================================================

✓ Mathematical error correction test passed
✓ Logical error correction test passed
✓ Sequential error correction test passed
✓ Quick correction for mathematical: 122 chars
...
✓ Error correction available for logical

============================================================
All tests passed! Error correction system is working correctly.
```

## Troubleshooting

### Issue: Error correction not triggering

**Check**:
1. Is `error_correction_enabled` flag True?
   ```python
   print(prompt_system._error_correction_enabled)
   ```

2. Is confidence below threshold?
   ```python
   if prompt_system.should_trigger_error_correction(0.65):
       print("Would trigger")
   ```

3. Is problem type correct?
   ```python
   ptype = prompt_system._detect_problem_type(problem)
   print(ptype)
   ```

### Issue: Error correction not improving answers

**Possible causes**:
1. Threshold too low (catching correct answers)
2. Problem type detection wrong (wrong domain guidance)
3. Model not responding to correction prompt
4. Too many correction attempts (model gets confused)

**Solutions**:
1. Raise threshold: `if confidence < 0.5:` instead of `< 0.7`
2. Verify problem type detection
3. Check model temperature/sampling settings
4. Limit to 1 correction attempt per problem

## Future Enhancements

1. **Adaptive thresholds** - Learn optimal threshold per problem type
2. **Error pattern tracking** - Identify common error types per domain
3. **Multi-round correction** - Chain multiple corrections for complex problems
4. **Confidence calibration** - Adjust confidence based on error correction results
5. **Domain-specific metrics** - Track effectiveness per problem type
6. **Ensemble approach** - Compare initial vs corrected with third verification pass

## References

- Cycle 3 (Self-Verification): R&D showing 20%+ error detection rates
- Cycle 5 (Self-Critique): Response quality enhancement patterns
- Position Primacy: Error checks work best at prompt start
- NEXT.md: Planned enhancements and follow-up research

## Contact & Questions

For questions about error correction implementation:
1. Check test file: `src/benchmarking/test_error_correction.py`
2. Review source: `src/agent/error_correction_prompts.py`
3. Check PromptSystem integration: `src/agent/prompt_system.py`
