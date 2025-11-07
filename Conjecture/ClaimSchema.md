---
title: Conjecture Claim Schema
version: 2.0
created: 2024-10-30
---

# Conjecture Claim Schema

## Overview

The Conjecture claim schema provides a unified structure for all knowledge representation in the system. Claims are factual statements with confidence scores that form the foundation of evidence-based reasoning, enabling transparent traceability and cumulative intelligence growth.

## Claim Structure

### Core Schema

```yaml
claim:
  id: unique_identifier          # Required: Globally unique identifier
  content: "The factual statement"  # Required: Human-readable claim content
  confidence: 0.0-1.0            # Required: Confidence score (0.0=unverified, 1.0=certain)
  supported_by: [claim_ids]      # Optional: Claims that support this claim
  supports: [claim_ids]          # Optional: Claims this claim supports
  type: [concept, reference, thesis, skill, example]  # Optional: Claim type for retrieval
  tags: [topic-tags]             # Optional: Topic-based categorization
  created: timestamp             # Required: ISO 8601 creation timestamp
```

### Field Definitions

#### `id` (Required)
- **Format**: `c[timestamp]_[sequence]` (e.g., `c20241030_001`)
- **Purpose**: Globally unique identifier across the entire system
- **Constraints**: Must be unique, immutable once created

#### `content` (Required)
- **Format**: Plain text statement
- **Purpose**: The actual factual claim being made
- **Guidelines**: 
  - Clear, unambiguous language
  - Single factual statement per claim
  - avoid conjunctions that combine multiple facts

#### `confidence` (Required)
- **Format**: Float between 0.0 and 1.0
- **Purpose**: Quantifies certainty in the claim's accuracy
- **Interpretation**:
  - `0.95-1.00`: Direct observation/peer-reviewed research
  - `0.85-0.94`: Multiple independent credible sources
  - `0.70-0.84`: Single credible source with methodology
  - `0.40-0.69`: Unverified but plausible information
  - `0.10-0.39`: Assumptions or preliminary hypotheses
  - `0.00-0.09`: Speculative or unverified claims

#### `supported_by` (Optional)
- **Format**: Array of claim IDs
- **Purpose**: Links to claims that provide evidence for this claim
- **Usage**: Creates bidirectional support relationships
- **Example**: `["c20241030_001", "c20241030_002"]`

#### `supports` (Optional)
- **Format**: Array of claim IDs
- **Purpose**: Links to claims that this claim provides evidence for
- **Usage**: Automatically maintained when other claims reference this claim
- **Note**: System should automatically update this field when `supported_by` relationships are created

#### `type` (Optional)
- **Format**: Array of claim types
- **Purpose**: Enables type-specific retrieval and processing
- **Valid Values**:
  - `concept`: Building block understanding (≤50 words)
  - `reference`: Source citations and external references
  - `thesis`: Comprehensive explanations and theories
  - `skill`: How-to instructions for performing actions
  - `example`: Action-result demonstrations
- **Default**: `["concept"]` if not specified

#### `tags` (Optional)
- **Format**: Array of topic tags
- **Purpose**: Topic-based categorization and semantic discovery
- **Guidelines**:
  - Use specific, meaningful topics
  - Avoid redundancy with content
  - Good examples: `["Shakespeare", "Quantum-Physics", "AI-Research"]`
  - Avoid: `["fact", "information", "data"]`

#### `created` (Required)
- **Format**: ISO 8601 timestamp (e.g., `2024-10-30T14:30:00Z`)
- **Purpose**: Tracks when the claim was created
- **Usage**: Temporal sorting, confidence decay calculations

## Claim Types

### Concept Claims
- **Purpose**: Building blocks of understanding
- **Length**: ≤50 words
- **Example**: 
```yaml
claim:
  id: c20241030_001
  content: "Photosynthesis converts light energy into chemical energy in plants."
  confidence: 0.95
  type: [concept]
  tags: [Biology, Botany, Plant-Physiology]
  created: 2024-10-30T14:30:00Z
```

### Reference Claims
- **Purpose**: Source citations and external references
- **Usage**: Replace direct source links with reference claims
- **Example**:
```yaml
claim:
  id: c20241030_002
  content: "According to Nature journal 2023, climate change accelerates species extinction rates."
  confidence: 0.90
  type: [reference]
  tags: [Climate-Change, Biology, Research]
  created: 2024-10-30T14:31:00Z
```

### Thesis Claims
- **Purpose**: Comprehensive explanations and theories
- **Length**: ≤500 words
- **Example**:
```yaml
claim:
  id: c20241030_003
  content: "Quantum computing leverages superposition and entanglement to process information in ways that classical computers cannot, potentially solving certain problems exponentially faster."
  confidence: 0.85
  type: [thesis]
  tags: [Quantum-Computing, Computer-Science, Physics]
  created: 2024-10-30T14:32:00Z
```

### Skill Claims
- **Purpose**: How-to instructions for performing actions
- **Length**: ≤500 words
- **Example**:
```yaml
claim:
  id: c20241030_004
  content: "Implement recursive binary search by checking middle element, recursively searching left or right half based on comparison, and returning index when found or -1 when not found."
  confidence: 0.95
  type: [skill]
  tags: [Algorithms, Programming, Search-Algorithms]
  created: 2024-10-30T14:33:00Z
```

### Example Claims
- **Purpose**: Action-result demonstrations
- **Length**: ≤500 words
- **Example**:
```yaml
claim:
  id: c20241030_005
  content: "Example: Applied binary search to sorted array [1,3,5,7,9] looking for 7, found at index 3 after 2 iterations."
  confidence: 0.95
  type: [example]
  tags: [Algorithms, Programming, Binary-Search]
  created: 2024-10-30T14:34:00Z
```

## Support Relationships

### Bidirectional Support Structure

```yaml
# Main claim
claim:
  id: c20241030_010
  content: "Regular exercise improves cardiovascular health."
  confidence: 0.85
  supported_by: ["c20241030_011", "c20241030_012"]
  created: 2024-10-30T14:40:00Z

# Supporting claim 1
claim:
  id: c20241030_011
  content: "According to American Heart Association 2023, exercise reduces blood pressure."
  confidence: 0.90
  supports: ["c20241030_010"]  # Automatically maintained
  created: 2024-10-30T14:41:00Z

# Supporting claim 2
claim:
  id: c20241030_012
  content: "Studies in Journal of Cardiology show exercise improves cholesterol levels."
  confidence: 0.85
  supports: ["c20241030_010"]  # Automatically maintained
  created: 2024-10-30T14:42:00Z
```

### Relationship Guidelines

#### Creating Support Relationships
1. **Direct Support**: Claim B directly supports Claim A
2. **Transitive Support**: Claim C supports B which supports A
3. **Cross-Validation**: Multiple independent claims support the same conclusion
4. **Contradiction Handling**: Track contradictory claims with resolution notes

#### Confidence Evolution
- **Support Increases Confidence**: Multiple supporting claims can increase confidence
- **Strong Sources**: High-confidence supporting claims have more weight
- **Contradictory Evidence**: Lower confidence when conflicting claims exist
- **Temporal Decay**: Confidence may decrease over time without new support

## Validation Rules

### Structural Validation
- **Required Fields**: All claims must have `id`, `content`, `confidence`, `created`
- **Unique IDs**: No duplicate claim IDs allowed
- **Valid Confidence**: Must be between 0.0 and 1.0
- **Valid Timestamp**: Must be parseable ISO 8601 format
- **Valid References**: All IDs in `supported_by` and `supports` must exist

### Content Validation
- **Non-empty Content**: Claim content cannot be empty
- **Single Statement**: Each claim should contain one factual statement
- **Clarity**: Content should be unambiguous and clear
- **Verifiability**: Claims should be potentially verifiable

### Relationship Validation
- **Bidirectional Consistency**: If A supports B, then B must be supported_by A
- **No Self-Reference**: Claims cannot support themselves
- **Circular Reference Prevention**: Detect and handle circular support chains

## API Integration

### Creating Claims
```python
def create_claim(content: str, confidence: float, claim_type: str = "concept", 
                 tags: List[str] = None, supported_by: List[str] = None):
    """
    Create a new claim with automatic ID generation and timestamp.
    
    Args:
        content: The factual statement
        confidence: Confidence score (0.0-1.0)
        claim_type: Type of claim (concept, reference, thesis, skill, example)
        tags: Topic-based tags for categorization
        supported_by: List of claim IDs that support this claim
    
    Returns:
        Claim object with generated ID and timestamp
    """
```

### Querying Claims
```python
def get_claims(confidence_min: float = 0.0, claim_types: List[str] = None,
               tags: List[str] = None, supports: str = None, 
               supported_by: str = None, limit: int = 50):
    """
    Retrieve claims based on various criteria.
    
    Args:
        confidence_min: Minimum confidence score
        claim_types: Filter by claim types
        tags: Filter by topic tags
        supports: Get claims that support the specified claim ID
        supported_by: Get claims supported by the specified claim ID
        limit: Maximum number of claims to return
    
    Returns:
        List of claims matching the criteria
    """
```

### Managing Support Relationships
```python
def add_support_relationship(supporting_claim_id: str, supported_claim_id: str):
    """Add a bidirectional support relationship between two claims"""

def remove_support_relationship(supporting_claim_id: str, supported_claim_id: str):
    """Remove a support relationship between two claims"""

def get_support_tree(claim_id: str, depth: int = 3):
    """Get the complete support tree for a claim up to specified depth"""
```

## Performance Considerations

### Indexing Strategy
- **Primary Index**: Claim ID for direct access
- **Content Index**: Full-text search on content
- **Confidence Index**: Range queries on confidence scores
- **Tag Index**: Multi-value index for tag-based filtering
- **Type Index**: Categorical index for claim types
- **Relationship Index**: Support relationship lookups

### Caching Strategy
- **Popular Claims**: Cache frequently accessed claims
- **Support Trees**: Cache computed support relationships
- **Tag Aggregations**: Cache tag-based query results
- **Type Statistics**: Cache claim type distributions

### Storage Optimization
- **Compression**: Compress claim content for storage efficiency
- **Delta Encoding**: Store only changes for claim updates
- **Relationship Compression**: Efficient storage of support relationships

## Migration from Previous Schema

### Removal of `sources` Field
**Old**: `sources: [{url: "source_url", methodology: "peer_reviewed", confidence: 0.85}]`
**New**: Create reference claims with links instead
```yaml
# Instead of sources field, create a reference claim
claim:
  id: c20241030_200
  content: "Source: Nature Journal 2023, 'Quantum Computing Advances', peer-reviewed methodology."
  confidence: 0.90
  type: [reference]
  tags: [Quantum-Computing, Research]
  created: 2024-10-30T15:00:00Z

# Main claim references the source claim
claim:
  id: c20241030_201
  content: "Quantum computing achieved 99.9% fidelity in qubit operations."
  confidence: 0.85
  supported_by: ["c20241030_200"]
  created: 2024-10-30T15:01:00Z
```

### Removal of Evidence Quality Tags
**Old**: `tags: [primary, validated, credible, unverified, assumption]`
**New**: Use confidence score instead and topic tags for categorization
```yaml
# Instead of evidence quality tags
claim:
  id: c20241030_202
  content: "Climate change affects biodiversity patterns."
  confidence: 0.85  # Single confidence score instead of evidence tier tags
  tags: [Climate-Change, Biodiversity, Ecology]  # Topic-based tags instead
  created: 2024-10-30T15:02:00Z
```

### Improved Relationship Naming
**Old**: `parents: [claim_ids]`, `children: [claim_ids]`
**New**: `supported_by: [claim_ids]`, `supports: [claim_ids]`
```yaml
# Clearer relationship naming
claim:
  id: c20241030_203
  content: "Regular exercise improves mental health."
  confidence: 0.80
  supported_by: ["c20241030_204", "c20241030_205"]  # Claims supporting this claim
  supports: ["c20241030_206"]  # Claims this claim supports
  created: 2024-10-30T15:03:00Z
```

## Best Practices

### Claim Creation
1. **Single Facts**: Each claim should contain one factual statement
2. **Clear Language**: Use unambiguous, precise language
3. **Verifiable Content**: Claims should be potentially verifiable
4. **Appropriate Confidence**: Set confidence based on source quality
5. **Meaningful Tags**: Use specific, relevant topic tags
6. **Consistent Types**: Choose appropriate claim types for retrieval

### Relationship Management
1. **Explicit Support**: Clearly document why claims support each other
2. **Independent Sources**: Use multiple independent sources for strong claims
3. **Avoid Circular Logic**: Prevent circular support relationships
4. **Track Contradictions**: Note and resolve contradictory claims
5. **Update Relationships**: Maintain bidirectional consistency

### Confidence Management
1. **Source Quality**: Base confidence on source methodology and credibility
2. **Evidence Strength**: Higher confidence for multiple independent sources
3. **Temporal Factors**: Consider recency and relevance
4. **Domain Expertise**: Account for specialized knowledge requirements
5. **Uncertainty Transparency**: Be explicit about uncertainty levels

This schema provides a clean, intuitive foundation for evidence-based reasoning while maintaining the flexibility needed for complex knowledge representation and retrieval.
