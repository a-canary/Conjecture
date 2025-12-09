# The Philosophy of Claims in Conjecture

## Critical Understanding: Claims Are NOT Facts

The most fundamental principle of the Conjecture system is that **claims are not facts**. Claims are impressions, assumptions, observations, and conjectures that have a variable or unknown amount of truth. Every claim in the system is provisional and subject to revision based on new evidence.

## What Claims Are

### 1. Impressions
- Initial intuitions or gut feelings about something
- First impressions that may or may not be accurate
- Subjective interpretations of information
- Example: "This algorithm seems inefficient"

### 2. Assumptions
- Things taken as true without proof for the purpose of reasoning
- Working hypotheses that enable further progress
- Temporary acceptance of something as true
- Example: "Assuming the user has Python installed"

### 3. Observations
- Things noticed or perceived through senses or data
- Empirical findings that may be incomplete
- Direct experiences or measurements
- Example: "The response time increased after the update"

### 4. Conjectures
- Conclusions formed on incomplete evidence
- Inferences that go beyond available data
- Speculative connections between observations
- Example: "The performance issue might be related to memory allocation"

## What Claims Are NOT

### Facts
- Claims are never facts, regardless of confidence level
- Even 0.99 confidence claims are still provisional
- High confidence means more evidence, NOT truth
- Facts exist independently of claims

### Truth
- No claim represents absolute truth
- Truth is always partial and contextual
- Claims approximate truth but never capture it completely
- New evidence can overturn any claim

### Certainty
- All claims carry uncertainty
- Confidence scores reflect evidence strength, not certainty
- Even well-supported claims can be wrong
- Uncertainty is fundamental to knowledge

## Confidence Scores

Confidence scores (0.0-1.0) represent:
- **0.0-0.3**: Weak evidence, high uncertainty
- **0.3-0.6**: Moderate evidence, significant uncertainty
- **0.6-0.8**: Strong evidence, but still provisional
- **0.8-0.95**: Very strong evidence, but still not a fact
- **0.95-1.0**: Exceptional evidence, but still subject to revision

**Key Point**: 0.95 confidence does NOT mean 95% true. It means the claim has very strong supporting evidence relative to current knowledge.

## Claim Types and Their Nature

### Primary Types
- **Impression**: Initial, subjective response (0.3-0.7 confidence)
- **Assumption**: Taken as true for reasoning (0.2-0.6 confidence)
- **Observation**: Perceived through senses/data (0.4-0.8 confidence)
- **Conjecture**: Conclusion on incomplete evidence (0.3-0.7 confidence)

### Secondary Types
- **Concept**: Abstract explanatory claim (0.5-0.8 confidence)
- **Example**: Specific instance or case (0.4-0.9 confidence)
- **Goal**: Desired outcome or objective (0.3-0.8 confidence)
- **Reference**: Pointer to external information (0.2-0.7 confidence)
- **Assertion**: Strong statement made with confidence (0.4-0.8 confidence)
- **Thesis**: Proposition for consideration (0.3-0.7 confidence)
- **Hypothesis**: Proposed explanation for investigation (0.2-0.6 confidence)
- **Question**: Query seeking information (0.1-0.5 confidence)
- **Task**: Action item or work to be done (0.2-0.6 confidence)

## The Provisional Nature of Knowledge

Conjecture embraces the philosophical position that all knowledge is provisional:
- Today's "well-established" claims may be tomorrow's errors
- New evidence can overturn even the most confident claims
- Progress happens through the refinement and replacement of claims
- No claim is ever final or absolute

## Practical Implications

### For Users
- Never treat a claim as absolute truth
- Always consider the confidence score and uncertainty
- Look for supporting evidence when evaluating claims
- Be prepared to revise your understanding

### For the System
- Every claim must have a confidence score
- Claims should include uncertainty estimates
- The system must track evidence supporting claims
- Claims should be updated when new evidence arrives

### For AI Processing
- Generate claims as impressions, assumptions, observations, or conjectures
- Never present anything as a fact
- Always include uncertainty and limitations
- Use appropriate confidence calibration

## Example: Proper Claim Formation

### Incorrect (Fact-Based)
```
"Water boils at 100°C at sea level" (confidence: 1.0)
```

### Correct (Claim-Based)
```
"Water typically boils at approximately 100°C at standard sea-level pressure" (confidence: 0.85)
Support: Multiple experimental observations under standard conditions
Uncertainty: May vary with exact pressure, purity, and measurement precision
```

## Conclusion

The Conjecture system's power comes from its honest embrace of uncertainty. By treating all knowledge as provisional claims rather than facts, the system remains open to revision, learning, and improvement. This philosophical foundation enables more robust reasoning and better decision-making in the face of incomplete information.

Remember: **Claims are not facts. Claims are impressions, assumptions, observations, and conjectures with variable truth.**