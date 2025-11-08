# Claim Schema - Unified Structure and Rules

## 🎯 **Purpose**

This document serves as the single source of truth for all claim structure definitions, confidence scoring rules, and relationship patterns in Conjecture. It eliminates duplication across 8+ claim type documents while providing comprehensive guidelines for claim creation and validation.

## 📋 **Universal Claim Structure**

### **Core YAML Schema**

```yaml
claim:
  id: unique_identifier                    # Required: Unique across system
  content: "The actual claim text"         # Required: Human-readable content
  confidence: 0.0-1.0                      # Required: Confidence score
  parents: [claim_ids]                     # Optional: Source/parent claims
  children: [claim_ids]                    # Optional: Dependent/child claims  
  tags: [primary, secondary, domain]       # Required: Classification tags
  created: timestamp                       # Required: ISO 8601 timestamp
```

### **Field Specifications**

#### **id** (Required)
- **Format**: `cYYYYMMDD_[type]_[sequence]`
- **Example**: `c20241028_concept_001`
- **Validation**: Must be unique across entire system
- **Purpose**: Precise reference and cross-linking

#### **content** (Required)
- **Length**: Varies by claim type (see specific guidelines)
- **Format**: Plain text, single sentence preferred
- **Validation**: Must be grammatically correct and meaningful
- **Purpose**: Human-readable claim statement

#### **confidence** (Required)
- **Range**: 0.0 (no confidence) to 1.0 (complete confidence)
- **Precision**: Two decimal places maximum
- **Validation**: Must align with confidence scoring guidelines
- **Purpose**: Uncertainty quantification and decision weighting

#### **parents** (Optional)
- **Type**: Array of claim IDs
- **Purpose**: Source references, supporting evidence, prerequisite knowledge
- **Validation**: All referenced claims must exist
- **Usage**: Replaces `source_ref` from earlier versions

#### **children** (Optional)
- **Type**: Array of claim IDs
- **Purpose**: Dependent claims, examples, applications
- **Validation**: Automatically managed by system
- **Usage**: Enables relationship traversal and impact analysis

#### **tags** (Required)
- **Type**: Array of strings
- **Purpose**: Classification, routing, discovery
- **Validation**: Must include at least one primary tag
- **Usage**: Replaces rigid `type` field with flexible categorization

#### **created** (Required)
- **Format**: ISO 8601 timestamp
- **Example**: `2024-10-28T21:50:00Z`
- **Purpose**: Temporal ordering and freshness tracking
- **Validation**: Must be valid timestamp

## 🏷️ **Tag Strategy Matrix**

### **Primary Classification Tags**

| Tag | Purpose | Typical Confidence | Example Usage |
|-----|---------|-------------------|---------------|
| `concept` | Fundamental ideas, facts, questions | 0.10-0.95 | "Ocean temperature causes coral bleaching" |
| `thesis` | Comprehensive explanations, theories | 0.20-0.95 | "Four-day work weeks increase productivity" |
| `goal` | Progress tracking, outcomes | 0.10-0.95 | "Reduce customer response time to 2 hours" |
| `reference` | Source provenance, citations | 0.10-0.95 | "Harvard Business Review, 2024, Dr. Martinez" |
| `skill` | How-to instructions, methodologies | 0.40-0.95 | "Execute data analysis using pandas" |
| `example` | Action-result demonstrations | 0.60-0.95 | "Used grep to find function definitions" |

### **Processing Intent Tags**

| Tag | Processing Approach | Use Case |
|-----|-------------------|----------|
| `research` | Discovery Processing | Exploratory investigation |
| `query` | Discovery Processing | Specific questions |
| `task` | Task Execution | Action-oriented items |
| `todo` | Task Execution | Completion tracking |
| `hypothesis` | Hypothesis Validation | Proposition testing |
| `plan` | Goal Achievement | Progress tracking |

### **Quality Assessment Tags**

| Tag | Confidence Range | Meaning |
|-----|------------------|---------|
| `validated` | 0.85-0.95 | Peer-reviewed, scientifically validated |
| `credible` | 0.70-0.84 | Reputable sources, supporting evidence |
| `unverified` | 0.30-0.69 | Claims without supporting evidence |
| `assumption` | 0.10-0.20 | Common beliefs for investigation |

### **Domain-Specific Tags**

| Domain | Example Tags | Usage |
|--------|-------------|-------|
| Technical | `python`, `api`, `database`, `security` | Technical concepts and skills |
| Business | `strategy`, `marketing`, `finance`, `operations` | Business contexts |
| Research | `scientific`, `academic`, `analysis`, `methodology` | Research activities |
| Communication | `writing`, `presentation`, `documentation` | Communication skills |

## 📊 **Confidence Scoring Guidelines**

### **Universal Confidence Tiers**

| Tier | Range | Description | Validation Requirements |
|------|-------|-------------|------------------------|
| **Primary** | 0.95 | Direct validation using scientific method | Peer-reviewed research, empirical evidence |
| **Validated** | 0.85 | External references with citations | Multiple credible sources, no fallacies |
| **Credible** | 0.70 | Reputable sources with evidence | Established publications, expert consensus |
| **Developing** | 0.55 | Emerging practices with some validation | Limited testing, expert opinion |
| **Basic** | 0.40 | Fundamental guidance with limitations | Common practices, some coverage gaps |
| **Unverified** | 0.30 | Claims without supporting evidence | Preliminary findings, anecdotal |
| **Assumption** | 0.10-0.20 | Hypotheses for investigation | Common beliefs, research questions |

### **Claim-Type Specific Confidence Patterns**

#### **Concept Claims**
- **Questions/Exploration**: 0.10-0.30 (low confidence, high uncertainty)
- **Preliminary Findings**: 0.40-0.60 (initial evidence, requires validation)
- **Established Facts**: 0.70-0.95 (validated knowledge, high confidence)

#### **Thesis Claims**
- **Hypotheses**: 0.20-0.40 (propositional, requires testing)
- **Developing Theories**: 0.50-0.70 (partial evidence, evolving)
- **Validated Theories**: 0.80-0.95 (strong evidence, established)

#### **Goal Claims**
- **Planning Phase**: 0.10-0.30 (initial planning, high uncertainty)
- **Active Progress**: 0.40-0.70 (implementation underway)
- **Near Completion**: 0.80-0.95 (achieving objectives, high confidence)

#### **Reference Claims**
- **Unverified Sources**: 0.30 (blogs, social media, anecdotal)
- **Credible Sources**: 0.70 (reputable publications, expert content)
- **Validated Sources**: 0.85 (peer-reviewed, scientific method)
- **Primary Sources**: 0.95 (direct research, empirical validation)

#### **Skill Claims**
- **Basic Instructions**: 0.40-0.55 (fundamental guidance, limited detail)
- **Reliable Methods**: 0.60-0.75 (well-tested approaches, good coverage)
- **Expert Practices**: 0.80-0.95 (industry standards, comprehensive coverage)

#### **Example Claims**
- **Illustrative**: 0.60-0.65 (educational focus, simplified scenarios)
- **Partial Demonstration**: 0.70-0.75 (useful but limited)
- **Complete Demonstration**: 0.90-0.95 (verifiable, comprehensive results)

## 🔗 **Relationship Patterns**

### **Parent-Child Relationships**

#### **Valid Relationship Types**

| Parent Type | Child Type | Relationship Purpose | Example |
|-------------|------------|---------------------|---------|
| `concept` | `thesis` | Concept enables comprehensive explanation | Basic climate science → Climate change theory |
| `thesis` | `example` | Theory demonstrated through application | Economic theory → Market analysis example |
| `skill` | `example` | Skill demonstrated in practice | Python programming → Data analysis example |
| `reference` | `concept` | Source supports factual claim | Research paper → Scientific finding |
| `goal` | `task` | Goal broken into executable tasks | System deployment → Database setup task |

#### **Relationship Rules**

1. **Circular References**: Not allowed (A → B → A)
2. **Depth Limits**: Maximum 5 levels deep to prevent complexity
3. **Cross-Type Validation**: Child must be logically dependent on parent
4. **Confidence Inheritance**: Children cannot exceed parent confidence without additional evidence

### **Relationship Management**

#### **Automatic Relationship Detection**
```yaml
auto_detection:
  semantic_similarity: 0.85 threshold for potential relationships
  content_reference: Explicit mentions of other claims
  tag_patterns: Complementary tag combinations
  temporal_sequence: Creation time patterns suggesting dependencies
```

#### **Manual Relationship Management**
```yaml
manual_guidelines:
  explicit_parenting: Clear parent-child relationships for learning progression
  cross_reference: Related but not dependent claims
  conflicting_relationships: Contradictory claims marked appropriately
  evidence_chains: Source → evidence → conclusion chains
```

## ✅ **Validation Rules**

### **Structural Validation**
- **Required Fields**: All required fields must be present and non-empty
- **Field Formats**: IDs, timestamps, and confidence values must match specifications
- **Tag Validation**: At least one primary tag required, valid tag combinations
- **Reference Integrity**: All parent/child references must exist

### **Content Validation**
- **Length Constraints**: Content length appropriate for claim type
- **Grammar Check**: Content must be grammatically correct
- **Meaningfulness**: Content must convey meaningful information
- **Consistency**: Content must align with assigned tags and confidence

### **Semantic Validation**
- **Confidence Alignment**: Confidence score must match content certainty
- **Tag Appropriateness**: Tags must accurately reflect content nature
- **Relationship Logic**: Parent-child relationships must be logical
- **Domain Consistency**: Content must fit assigned domain tags

## 🔧 **Implementation Guidelines**

### **Claim Creation Process**
1. **Determine Claim Type**: Select appropriate primary tag
2. **Draft Content**: Create clear, concise claim statement
3. **Assign Confidence**: Use scoring guidelines for accurate assessment
4. **Add Relationships**: Link to relevant parent/child claims
5. **Apply Tags**: Use comprehensive tag strategy
6. **Validate**: Run through all validation rules

### **Claim Maintenance**
- **Regular Reviews**: Periodic confidence reassessment
- **Relationship Updates**: Add new relationships as discovered
- **Content Refinement**: Improve clarity and accuracy over time
- **Tag Optimization**: Refine tag usage for better discovery

### **Quality Assurance**
- **Automated Validation**: Structural and format validation
- **Peer Review**: Content accuracy and confidence assessment
- **Semantic Analysis**: Tag and relationship validation
- **Consistency Checks**: Cross-document consistency verification

## 📈 **Usage Examples**

### **Complete Claim Examples**

#### **Concept Claim Example**
```yaml
claim:
  id: c20241028_concept_001
  content: "Ocean temperature increases of 1.5°C cause irreversible coral bleaching"
  confidence: 0.85
  parents: [c20241028_ref_001, c20241028_ref_002]
  children: [c20241028_thesis_001]
  tags: [concept, validated, environmental, climate, marine-biology]
  created: 2024-10-28T21:50:00Z
```

#### **Skill Claim Example**
```yaml
claim:
  id: c20241028_skill_001
  content: "Execute data analysis using pandas with proper error handling, data validation, and statistical methods"
  confidence: 0.85
  parents: [c20241028_concept_001]
  children: [c20241028_example_001]
  tags: [skill, data-analysis, python, pandas, statistics]
  created: 2024-10-28T21:50:00Z
```

#### **Example Claim Example**
```yaml
claim:
  id: c20241028_example_001
  content: "Used pandas read_csv() to load 10,000 records, applied data cleaning, removed duplicates, final dataset: 9,600 clean rows"
  confidence: 0.95
  parents: [c20241028_skill_001]
  children: []
  tags: [example, data-analysis, python, pandas, result]
  created: 2024-10-28T21:50:00Z
```

## 🎯 **Success Metrics**

### **Schema Compliance**
- **100%** of claims follow unified structure
- **0%** structural validation errors
- **95%** confidence scoring accuracy
- **100%** tag compliance with guidelines

### **Relationship Quality**
- **90%** of claims have appropriate relationships
- **0%** circular references
- **95%** logical relationship validation
- **100%** reference integrity

### **Content Quality**
- **95%** grammatically correct content
- **90%** appropriate confidence assignments
- **95%** accurate tag usage
- **100%** meaningful content validation

This ClaimSchema document provides the comprehensive foundation for consistent, high-quality claim creation and management across the entire Conjecture system, eliminating duplication while ensuring robust validation and quality standards.
