# Domain-Specific Claim Templates

## Overview

Domain-specific claim templates prime the model for optimal performance on mathematical (DROP), scientific (ARC), and logical (BBH) reasoning benchmarks. These templates encode domain-specific reasoning patterns and are injected at the prompt START for maximum attention impact (+10pp improvement per NEXT.md findings).

## Key Features

### Position Primacy Implementation
- **START placement**: Templates injected at prompt beginning (+10pp effect)
- **Confidence threshold**: Templates maintain 0.85 base confidence
- **Selective inclusion**: Top N templates selected (default: 3)

### Supported Benchmarks

| Benchmark | Domain | Focus Area | Templates |
|-----------|--------|-----------|-----------|
| DROP | Mathematical | Word problems, calculations, rates | 10 |
| ARC | Scientific | Experimental design, data interpretation | 10 |
| BBH | Logical | Deduction, reasoning, puzzles | 10 |
| General | Cross-Domain | Decomposition, verification | 8 |

## Architecture

### Core Modules

#### 1. `src/agent/domain_claim_templates.py`
- **`DOMAIN_CLAIM_TEMPLATES`**: Central template dictionary
- **`ClaimTemplate`**: Dataclass for individual templates
- **`DomainClaimSelector`**: Manages template selection and formatting
- **Convenience functions**: Global helpers for template access

#### 2. `src/agent/domain_reasoning_enhancement.py`
- **`DomainReasoningEnhancer`**: Applies templates to prompts
- **`BenchmarkType`**: Enum for supported benchmarks
- **Position primacy**: Ensures optimal template placement
- **Benchmark-specific methods**: Tailored enhancement for each benchmark

### Template Structure

Each template consists of:
```python
{
    "domain_name": "Mathematical Reasoning (DROP benchmark)",
    "description": "Templates for mathematical reasoning...",
    "benchmark": "DROP",
    "templates": [
        "Arithmetic rule: Order of operations (PEMDAS/BODMAS)...",
        "Percentage calculation: To find X% of Y...",
        # ... more templates
    ]
}
```

## Mathematical Reasoning Templates (DROP)

Templates for word problems, calculations, percentages, rates:

1. **Arithmetic rule**: Order of operations (PEMDAS/BODMAS)
2. **Percentage calculation**: X% of Y, what % is X of Y
3. **Rate problems**: Speed = Distance/Time relationships
4. **Word problem strategy**: Identify values, setup equation, solve
5. **Estimation technique**: Round before calculating for reasonableness
6. **Unit conversion**: Track units through calculations
7. **Algebraic principle**: Perform same operations on both sides
8. **Geometry basics**: Area/volume formulas
9. **Money and time problems**: Track values through steps
10. **Ratio problems**: Proportional scaling

## Scientific Reasoning Templates (ARC)

Templates for experimental design, data interpretation, scientific method:

1. **Scientific method**: Observation → Hypothesis → Prediction → Experiment → Analysis
2. **Variables**: Independent, dependent, control variables
3. **Experimental design**: Controls essential for isolating cause
4. **Data interpretation**: Patterns, trends, correlation vs causation
5. **Physical principles**: Energy conservation laws
6. **Chemical principle**: Reactants → Products, atom balancing
7. **Biological concept**: Evolution via natural selection
8. **Measurement**: Accuracy vs precision
9. **Uncertainty**: Sample size and outlier handling
10. **Evidence evaluation**: Primary sources > secondary sources

## Logical Reasoning Templates (BBH)

Templates for deduction, logical operators, formal reasoning:

1. **Deductive reasoning**: If premises true and logic valid, conclusion must be true
2. **Premise-conclusion structure**: Facts/rules → necessary conclusion
3. **Logical operators**: AND, OR, NOT, XOR definitions
4. **Quantifiers**: All, Some, None definitions
5. **Conditional logic**: If P then Q, contrapositive, converse, inverse
6. **Logical fallacy**: Common mistakes (affirming consequent)
7. **Set theory**: Subset, intersection, union
8. **Proof by contradiction**: Assume false → contradiction → true
9. **Necessary vs sufficient**: Conditions and requirements
10. **Analogical reasoning**: Pattern matching A:B :: C:?

## Usage Examples

### Basic Template Retrieval

```python
from src.agent.domain_claim_templates import (
    get_templates_for_domain,
    format_claims_for_prompt
)

# Get templates for a domain
templates = get_templates_for_domain("mathematical", max_count=3)

# Format for prompt injection
formatted = format_claims_for_prompt("mathematical", max_count=3)
```

### Benchmark-Specific Enhancement

```python
from src.agent.domain_reasoning_enhancement import (
    enhance_for_benchmark,
    BenchmarkType
)

base_prompt = "Solve this word problem..."

# Enhance for DROP benchmark
enhanced = enhance_for_benchmark(base_prompt, "DROP", max_claims=3)
```

### Custom Enhancer Usage

```python
from src.agent.domain_reasoning_enhancement import DomainReasoningEnhancer

enhancer = DomainReasoningEnhancer()

# Prepare mathematical reasoning prompt
math_prompt = enhancer.prepare_mathematical_reasoning_prompt(
    base_prompt,
    include_templates=True
)

# Get benchmark context
context = enhancer.get_benchmark_context(BenchmarkType.ARC)
print(f"Domain: {context['domain']}")
print(f"Templates: {context['template_count']}")
```

### PromptSystem Integration

```python
from src.agent.prompt_system import PromptSystem, ProblemType

prompt_system = PromptSystem()

# Get domain-specific templates
templates = prompt_system.get_domain_claim_templates(
    ProblemType.MATHEMATICAL,
    max_count=3
)

# Enhance prompt with domain claims at START
enhanced = prompt_system.enhance_prompt_with_domain_claims(
    base_prompt,
    ProblemType.MATHEMATICAL
)
```

## Performance Impact

### Position Primacy Effect
- **START placement**: +10pp improvement
- **MIDDLE placement**: +5pp improvement
- **END placement**: ~0pp improvement

Reference: NEXT.md R&D findings (2026-03-01)

### Confidence Threshold
- **Optimal range**: 0.5-0.8 confidence
- **Template base confidence**: 0.85
- **Over-strict (0.9)**: Rejects useful claims

### Selective Inclusion
- **Top 3 templates**: Best balance of guidance + clarity
- **No semantic filtering**: Simple inclusion beats complex filtering (84% vs 86%)
- **All correct claims**: Include all valid domain templates

## Integration with Benchmarks

### DROP (Mathematical)
- Problem type detection: MATHEMATICAL
- Domain key: "mathematical"
- Top templates: Order of operations, percentage, rates
- Optimal template count: 3-4

### ARC (Scientific)
- Problem type detection: SCIENTIFIC
- Domain key: "scientific"
- Top templates: Scientific method, variables, experimental design
- Optimal template count: 3-4

### BBH (Logical)
- Problem type detection: LOGICAL
- Domain key: "logical"
- Top templates: Deductive reasoning, logical operators, conditionals
- Optimal template count: 3-4

## Testing

Comprehensive test suite in `tests/test_domain_claim_templates.py`:
- Template loading and structure validation
- Domain-specific template retrieval
- Benchmark selection and formatting
- Position primacy implementation
- PromptSystem integration

Run tests:
```bash
python -m pytest tests/test_domain_claim_templates.py -v
```

## Future Enhancements

### Potential Improvements
1. **Adaptive template selection**: Choose templates based on problem complexity
2. **Dynamic confidence scoring**: Adjust confidence based on historical performance
3. **Template chaining**: Combine related templates for complex problems
4. **Learning-based selection**: Track which templates most improve performance

### Research Directions
- Test template count optimization (3 vs 5 vs 10)
- Explore position variants (START vs START-AFTER-SYSTEM)
- Investigate template redundancy (overlap reduction)
- Benchmark against direct prompting (GSM8K: Direct 96% vs Conjecture 65%)

## References

- **NEXT.md**: Position primacy findings (+10pp)
- **MEMORY.md**: Skill patterns extracted from project history
- **CLAUDE.md**: R&D findings and systematic improvement framework
- **ARC-AGI-2 Benchmark**: Primary benchmark per CHOICES.md
- **DROP Dataset**: Mathematical reasoning benchmark
- **BBH (Big Bench Hard)**: Logical reasoning benchmark

## File Structure

```
src/agent/
├── domain_claim_templates.py        # Core templates and selector
├── domain_reasoning_enhancement.py  # Benchmark-specific enhancement
└── prompt_system.py                 # Integration point

tests/
└── test_domain_claim_templates.py   # Comprehensive test suite

docs/
└── DOMAIN_CLAIM_TEMPLATES.md        # This documentation
```
