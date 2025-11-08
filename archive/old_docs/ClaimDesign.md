# Claim Design - Foundation Layer Architecture

## Executive Summary

The Claim foundation layer provides unified data structures and processing mechanisms that enable Conjecture's radical simplicity through flattened YAML architecture, flexible tagging systems, and relationship-driven processing.

## Simplified Claim Schema

### Core Claim Structure

```yaml
claim:
  id: string              # Unique identifier (format: c[timestamp]_[hash])
  content: string         # Claim text (50 words for concepts, 500 for thesis)
  confidence: float       # 0.0-1.0 quality score based on source validation
  parents: array          # Parent claim IDs for relationship tracking
  children: array         # Child claim IDs for forward traversal
  tags: array            # Flexible categorization replacing rigid types
  created: timestamp      # ISO 8601 creation timestamp
```

### Tag Taxonomy

**Primary Claim Types**: concept, thesis, goal, reference
**Processing Patterns**: research, query, task, hypothesis, plan, todo
**Cross-cutting Concerns**: validated, pending, archived

## System Architecture

### Unified Claim Engine

The foundation layer consists of:
- **Claim Storage**: Persistent claim database
- **Claim Engine**: Core processing logic
- **Relationship Manager**: Parent-child relationship tracking
- **Confidence Manager**: Quality scoring and inheritance
- **Semantic Search**: Content-based discovery
- **Tag Processor**: Flexible categorization

### Claim Lifecycle

Claims progress through states: Created → Validating → Active → Processing → Updated → Resolved → Archived. Invalid structures are rejected early in the lifecycle.

## Relationship Management

### Relationship Types

- **supports**: Parent provides evidence for child
- **contradicts**: Parent opposes child claim
- **clarifies**: Parent explains or elaborates child
- **enables**: Parent makes child possible
- **requires**: Child depends on parent
- **questions**: Parent queries child claim
- **refines**: Parent improves or corrects child

## Confidence Scoring System

### Source-Based Tiers

- **primary**: 0.95 (Peer-reviewed, scientific method)
- **validated**: 0.85 (External references, citations)
- **credible**: 0.70 (Reputable sources, evidence)
- **unverified**: 0.30 (Claims without evidence)
- **assumption**: 0.15 (Common beliefs for investigation)

### Confidence Inheritance

Child claims inherit reduced confidence from parents (typically 0.05-0.10 decrease per generation). Multiple supporting sources can boost confidence through aggregation.

## Processing Integration

### Unified Pipeline

Input → Parse → Validate → Tag → Store → Search → Match → Process → Execute → Generate → Update

### Tag-Based Routing

- **concept**: semantic validation, similarity matching, relationship inference
- **thesis**: concept aggregation, evidence validation, logical consistency check
- **goal**: progress tracking, dependency analysis, completion assessment
- **reference**: source validation, reliability assessment, provenance tracking

## Performance Optimization

### Indexing Strategy

**Primary indexes**: claim_id, created_timestamp, confidence_score
**Composite indexes**: tags_confidence, relationships, semantic_content
**Specialized indexes**: tag_lookup, confidence_range, relationship_graph

### Caching Architecture

Multi-layer caching with hot claims, recent relationships, tag indexes, and semantic embeddings for optimal performance.

## Quality Assurance

### Validation Rules

**Structural validation**: Required fields, ID format, content length limits, confidence range
**Semantic validation**: Content coherence, tag relevance, relationship consistency, confidence appropriateness

### Conflict Resolution

When conflicts occur:
1. Compare confidence scores - higher wins
2. If equal confidence, compare source quality
3. If sources equal, most recent wins
4. Consider merging compatible claims

## Integration Interfaces

### API Endpoints

- **POST /api/claims**: Create new claim
- **GET /api/claims/{id}**: Retrieve claim with relationships
- **POST /api/claims/search**: Search claims with filters
- **PUT /api/claims/{id}/relationships**: Update claim relationships

The Claim foundation layer provides robust, simplified architecture enabling Conjecture's unified approach to evidence-based intelligence processing through flattened structures, flexible tagging, and relationship-driven processing.
