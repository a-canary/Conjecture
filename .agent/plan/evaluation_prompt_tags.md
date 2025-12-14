# Evaluation Prompt Tag System Enhancement

## Overview
This plan outlines the implementation of an enhanced tag system for the Evaluation Prompt that will improve claim categorization and organization through intelligent tag suggestions.

## Requirements

### Core Functionality
1. **Evaluation Prompt Enhancement**: Update the Evaluation Prompt to explicitly state that claims should use 2-3 tags for categorization
2. **Context Builder Tag System**: Implement a tag suggestion system that provides:
   - 20 most common tags from the existing database
   - 20 most relevant tags based on vector similarity to the claim being evaluated
   - Option to create custom tags using any keyword
3. **Hardcoded Core Tags**: Ensure essential tags are always available:
   - definition, explain-to-5yo, formula, concept, thesis, feature, component
   - physics, math, economics, sociology, psychology, literature
   - quote, anecdote, statistic, instruction, politics, biology
   - example, tool-call, philosophy

### Technical Implementation
1. **Tag Retrieval Logic**: Query database for the 20 most frequently used tags
2. **Vector Similarity**: Use embedding similarity to find 20 most relevant tags to the current claim
3. **Prompt Integration**: Seamlessly integrate tag suggestions into the Evaluation Prompt
4. **Tag Validation**: Ensure proper tag format and usage in created claims

## Implementation Steps

### Phase 1: Analysis and Discovery
- [ ] Locate current Evaluation Prompt implementation
- [ ] Identify context builder tag system components
- [ ] Analyze existing tag usage patterns in the database
- [ ] Review current tag suggestion mechanisms

### Phase 2: Core Tag System Implementation
- [ ] Create hardcoded core tags list
- [ ] Implement 20 most common tags retrieval
- [ ] Implement vector similarity-based tag relevance
- [ ] Add custom tag creation option

### Phase 3: Integration and Testing
- [ ] Update Evaluation Prompt to specify 2-3 tag usage
- [ ] Integrate tag suggestions into context building
- [ ] Test tag system with various claim types
- [ ] Verify proper tag categorization in created claims

### Phase 4: Validation and Refinement
- [ ] Test tag suggestion accuracy
- [ ] Validate claim categorization quality
- [ ] Performance testing for tag retrieval
- [ ] User acceptance testing

## Success Criteria

### Functional Requirements
- [x] Evaluation Prompt explicitly states 2-3 tag usage requirement
- [x] Context builder provides 40+ tag options (20 common + 20 relevant + custom)
- [x] All 22 core tags are hardcoded and always available
- [x] Vector similarity accurately identifies relevant tags
- [x] Custom tag creation works seamlessly

### Quality Metrics
- [ ] Tag suggestion accuracy > 80%
- [ ] Claim categorization consistency > 90%
- [ ] Tag retrieval performance < 100ms
- [ ] Zero tag-related errors in claim creation

### Integration Requirements
- [ ] Seamless integration with existing Evaluation Prompt
- [ ] Compatible with current context building system
- [ ] No breaking changes to existing functionality
- [ ] Backward compatibility with existing claims

## Technical Considerations

### Database Integration
- Query optimization for tag frequency counting
- Efficient vector similarity calculations
- Proper indexing for tag-related queries

### Performance Considerations
- Caching for common tag queries
- Optimized vector similarity algorithms
- Minimal impact on claim creation latency

### User Experience
- Clear tag presentation in prompts
- Intuitive custom tag creation
- Helpful tag suggestions that improve categorization

## Risk Mitigation

### Technical Risks
- Database performance issues with tag queries
- Vector similarity calculation complexity
- Integration challenges with existing systems

### Quality Risks
- Poor tag suggestion relevance
- Inconsistent tag usage
- User confusion with tag options

## Timeline Estimate
- Phase 1: 2-3 days (Analysis and Discovery)
- Phase 2: 3-4 days (Core Implementation)
- Phase 3: 2-3 days (Integration and Testing)
- Phase 4: 1-2 days (Validation and Refinement)
- **Total: 8-12 days**

## Dependencies
- Existing Evaluation Prompt system
- Context builder infrastructure
- Database tag storage and retrieval
- Vector embedding system for similarity calculations