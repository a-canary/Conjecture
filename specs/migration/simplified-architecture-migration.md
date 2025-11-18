# Simplified Architecture Migration Strategy

**Last Updated:** November 11, 2025  
**Version:** 1.0  
**Author:** Design Documentation Writer

## Overview

This migration strategy outlines the process of transitioning from the existing complex claim architecture to the Simple Universal Claim Architecture with LLM-Driven Instruction Support. The migration is designed to be incremental, reversible, and minimize disruption to existing functionality while achieving the benefits of simplified data models and LLM-driven intelligence.

## Current State Analysis

### Existing Complex Architecture

The current system contains multiple specialized data structures and processes:

**Data Models:**
- Multiple claim types (EnhancedClaim, InstructionClaim, EvidenceClaim, StepClaim)
- Complex hierarchical relationships
- Metadata-heavy claim structures
- Specialized instruction and evidence models

**Processing Components:**
- Multiple context builders for different use cases
- Complex relationship validators and converters
- Instruction-specific processors
- Multi-stage claim transformation pipelines

**Challenges Identified:**
- High maintenance overhead due to model complexity
- Difficult debugging across multiple model types
- Performance bottlenecks in data transformation
- Limited flexibility for new claim types
- Complex testing matrix across model variations

### Migration Benefits

**Simplification Gains:**
- 90% reduction in data model complexity
- Elimination of 15+ specialized data structures
- Single unified claim model
- Simplified database schema
- Reduced testing surface area

**Performance Improvements:**
- Faster context building (simple traversal)
- Reduced memory footprint
- Simplified database queries
- Better caching opportunities

**Maintenance Benefits:**
- Single code path for claim operations
- Easier debugging and troubleshooting
- Clear separation of concerns
- Reduced technical debt

## Migration Approach

### Migration Principles

1. **Incremental Migration**: Migrate functionality in phases to minimize risk
2. **Backward Compatibility**: Maintain API compatibility during transition
3. **Data Preservation**: No loss of existing claim data or relationships
4. **Parallel Operations**: Run old and new systems side-by-side during transition
5. **Rollback Capability**: Ability to revert to old system if issues arise
6. **Validation at Each Step**: Verify data integrity and functionality after each phase

### Migration Phases

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Phase 1:      │    │   Phase 2:      │    │   Phase 3:      │
│   Data Model    │───▶│   Context       │───▶│   LLM          │
│   Unification   │    │   Builder       │    │   Integration   │
│                 │    │                 │    │                 │
│ • Schema Update │    │ • Algorithm     │    │ • Protocol      │
│ • Data Mapping  │    │   Migration     │    │   Implementation│
│ • Validation    │    │ • Performance   │    │ • Testing       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                       ┌─────────────────┐
                       │   Phase 4:      │
                       │   Cleanup       │
                       │                 │
                       │ • Remove Old    │
                       │   Code          │
                       │ • Documentation│
                       │ • Monitoring    │
                       └─────────────────┘
```

## Phase 1: Data Model Unification

### 1.1 Schema Analysis and Mapping

**Current Schema Analysis:**
```python
# Map existing models to unified structure
MODEL_MAPPING = {
    "EnhancedClaim": {
        "target": "Claim",
        "field_mappings": {
            "enhanced_metadata": "merge_into tags",
            "specialized_properties": "merge_into tags",
            "extra_confidence": "max(confidence, extra_confidence)",
            "hierarchy_level": "ignore"  # Handled by relationships
        },
        "conversion_required": True
    },
    "InstructionClaim": {
        "target": "Claim", 
        "field_mappings": {
            "instruction_type": "merge_into tags as 'instruction:{type}'",
            "required_actions": "merge_into content",
            "dependencies": "convert_to supported_by relationships"
        },
        "conversion_required": True
    },
    "EvidenceClaim": {
        "target": "Claim",
        "field_mappings": {
            "evidence_source": "merge_into tags as 'source:{value}'",
            "relevance_score": "use as confidence",
            "supporting_data": "merge_into content"
        },
        "conversion_required": True
    }
}
```

**Data Conversion Process:**
```python
class ClaimDataMigrator:
    """Handles conversion from old claim models to unified Claim model"""
    
    async def analyze_existing_data(self) -> Dict[str, Any]:
        """Analyze existing claim data and identify conversion requirements"""
        
        # Query all existing claim types
        enhanced_claims = await self.db.query("SELECT * FROM enhanced_claims")
        instruction_claims = await self.db.query("SELECT * FROM instruction_claims")
        evidence_claims = await self.db.query("SELECT * FROM evidence_claims")
        
        analysis = {
            "total_claims": len(enhanced_claims) + len(instruction_claims) + len(evidence_claims),
            "by_type": {
                "enhanced": len(enhanced_claims),
                "instruction": len(instruction_claims), 
                "evidence": len(evidence_claims)
            },
            "conversion_complexity": self.assess_conversion_complexity(
                enhanced_claims, instruction_claims, evidence_claims
            ),
            "potential_conflicts": self.identify_conversion_conflicts()
        }
        
        return analysis
    
    def assess_conversion_complexity(
        self, 
        enhanced_claims, 
        instruction_claims, 
        evidence_claims
    ) -> str:
        """Assess how complex the migration will be"""
        
        total = len(enhanced_claims) + len(instruction_claims) + len(evidence_claims)
        
        if total > 10000:
            return "high"
        elif total > 1000:
            return "medium"
        else:
            return "low"
    
    async def convert_claim_batch(self, old_claims: List[Dict]) -> List[Claim]:
        """Convert a batch of old claims to unified Claim models"""
        
        converted_claims = []
        
        for old_claim in old_claims:
            try:
                # Map old claim to unified structure
                unified_claim = self.map_to_unified_claim(old_claim)
                
                # Validate conversion
                validation_errors = self.validate_conversion(old_claim, unified_claim)
                if validation_errors:
                    logger.warning(f"Conversion validation errors: {validation_errors}")
                    # Continue but log issues
                
                converted_claims.append(unified_claim)
                
            except Exception as e:
                logger.error(f"Failed to convert claim {old_claim.get('id', 'unknown')}: {e}")
                # Continue with other claims
        
        return converted_claims
    
    def map_to_unified_claim(self, old_claim: Dict) -> Claim:
        """Map an old claim structure to unified Claim model"""
        
        claim_type = old_claim.get("_type", "EnhancedClaim")
        mapping = MODEL_MAPPING.get(claim_type, MODEL_MAPPING["EnhancedClaim"])
        
        # Extract basic fields
        unified_data = {
            "id": old_claim["id"],
            "content": self.extract_content(old_claim, claim_type),
            "confidence": self.extract_confidence(old_claim, claim_type),
            "state": self.extract_state(old_claim),
            "supported_by": self.extract_supporting_relationships(old_claim),
            "supports": self.extract_supported_relationships(old_claim),
            "type": self.extract_claim_types(old_claim, claim_type),
            "tags": self.extract_tags(old_claim, claim_type),
            "created_by": old_claim.get("created_by", "migration"),
            "created": old_claim.get("created", datetime.utcnow()),
            "updated": datetime.utcnow()
        }
        
        return Claim(**unified_data)
```

### 1.2 Database Schema Migration

**Migration Script Structure:**
```sql
-- Step 1: Create new unified claims table
CREATE TABLE claims_unified (
    id VARCHAR(50) PRIMARY KEY,
    content TEXT NOT NULL,
    confidence FLOAT CHECK (confidence >= 0.0 AND confidence <= 1.0),
    state VARCHAR(20) DEFAULT 'Explore' CHECK (state IN ('Explore', 'Validated', 'Orphaned', 'Queued')),
    supported_by TEXT[] DEFAULT '{}',
    supports TEXT[] DEFAULT '{}',
    type VARCHAR(50)[] NOT NULL CHECK (array_length(type, 1) > 0),
    tags TEXT[] DEFAULT '{}',
    created_by VARCHAR(100) NOT NULL,
    created TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    embedding VECTOR(384),  -- For ChromaDB integration
    is_dirty BOOLEAN DEFAULT FALSE,
    dirty_reason VARCHAR(50),
    dirty_timestamp TIMESTAMP WITH TIME ZONE,
    dirty_priority INTEGER DEFAULT 0
);

-- Step 2: Create indexes for performance
CREATE INDEX idx_claims_unified_state ON claims_unified(state);
CREATE INDEX idx_claims_unified_confidence ON claims_unified(confidence);
CREATE INDEX idx_claims_unified_type ON claims_unified USING GIN(type);
CREATE INDEX idx_claims_unified_tags ON claims_unified USING GIN(tags);
CREATE INDEX idx_claims_unified_dirty ON claims_unified(is_dirty) WHERE is_dirty = TRUE;

-- Step 3: Create migration tracking table
CREATE TABLE migration_log (
    id SERIAL PRIMARY KEY,
    phase VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    records_migrated INTEGER DEFAULT 0,
    errors TEXT,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);
```

**Data Migration Procedure:**
```python
class DatabaseMigrator:
    """Handles database schema and data migration"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.migration_log = MigrationLogger(db_connection)
    
    async def execute_phase1_migration(self) -> MigrationResult:
        """Execute Phase 1: Data Model Unification"""
        
        result = MigrationResult(phase="Phase1_Data_Unification")
        
        try:
            # Step 1: Create new schema
            await self.create_unified_schema()
            result.add_step("schema_created", True)
            
            # Step 2: Migrate existing data in batches
            total_migrated = await self.migrate_claim_data()
            result.add_step("data_migrated", True, records=total_migrated)
            
            # Step 3: Validate migrated data
            validation_result = await self.validate_migrated_data()
            result.add_step("validation", validation_result.success, details=validation_result)
            
            # Step 4: Create backup of old data
            await self.create_backup_tables()
            result.add_step("backup_created", True)
            
            result.status = "completed"
            
        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            logger.error(f"Phase 1 migration failed: {e}")
            
            # Attempt rollback if possible
            await self.attempt_rollback_phase1()
        
        await self.migration_log.log_migration(result)
        return result
    
    async def migrate_claim_data(self) -> int:
        """Migrate claim data from old tables to unified table"""
        
        total_migrated = 0
        batch_size = 1000
        
        # Get all old claim IDs
        old_claim_ids = await self.get_all_old_claim_ids()
        
        logger.info(f"Starting migration of {len(old_claim_ids)} claims")
        
        for i in range(0, len(old_claim_ids), batch_size):
            batch_ids = old_claim_ids[i:i + batch_size]
            
            # Get batch of old claims
            old_claims = await self.get_old_claims_batch(batch_ids)
            
            # Convert to unified format
            migrator = ClaimDataMigrator(self.db)
            unified_claims = await migrator.convert_claim_batch(old_claims)
            
            # Insert into unified table
            await self.insert_unified_claims(unified_claims)
            
            total_migrated += len(unified_claims)
            
            logger.info(f"Migrated batch {i//batch_size + 1}: {len(unified_claims)} claims")
            
            # Update progress
            if i % (batch_size * 5) == 0:  # Every 5 batches
                progress = (i + batch_size) / len(old_claim_ids) * 100
                logger.info(f"Migration progress: {progress:.1f}%")
        
        return total_migrated
    
    async def validate_migrated_data(self) -> ValidationResult:
        """Validate that data was migrated correctly"""
        
        validation = ValidationResult()
        
        # Check record counts
        old_count = await self.count_old_claims()
        new_count = await self.count_unified_claims()
        
        if old_count != new_count:
            validation.add_error(
                f"Record count mismatch: old={old_count}, new={new_count}"
            )
        
        # Check relationship integrity
        relationship_issues = await self.validate_relationship_integrity()
        if relationship_issues:
            validation.add_error(f"Relationship issues: {relationship_issues}")
        
        # Validate data quality
        quality_issues = await self.validate_data_quality()
        if quality_issues:
            validation.add_error(f"Data quality issues: {quality_issues}")
        
        # Sample verification
        sample_issues = await self.verify_sample_data(sample_size=100)
        if sample_issues:
            validation.add_error(f"Sample verification issues: {sample_issues}")
        
        validation.success = len(validation.errors) == 0
        return validation
```

### 1.3 API Compatibility Layer

During migration, maintain backward compatibility with existing APIs:

```python
class BackwardCompatibilityAPI:
    """Provides backward compatibility while using unified models"""
    
    def __init__(self, unified_system: UnifiedClaimSystem):
        self.unified = unified_system
    
    async def get_enhanced_claim(self, claim_id: str) -> Dict[str, Any]:
        """Get claim in old EnhancedClaim format"""
        
        # Get unified claim
        unified_claim = await self.unified.data_manager.get_claim(claim_id)
        if not unified_claim:
            return None
        
        # Convert to old format
        return {
            "id": unified_claim.id,
            "content": unified_claim.content,
            "confidence": unified_claim.confidence,
            "enhanced_metadata": {"tags": unified_claim.tags},
            "specialized_properties": {},
            "type": unified_claim.type,
            "state": unified_claim.state.value,
            "created": unified_claim.created,
            "updated": unified_claim.updated
        }
    
    async def get_instruction_claim(self, claim_id: str) -> Dict[str, Any]:
        """Get claim in old InstructionClaim format if applicable"""
        
        unified_claim = await self.unified.data_manager.get_claim(claim_id)
        if not unified_claim or "instruction" not in [t.value for t in unified_claim.type]:
            return None
        
        # Extract instruction-specific data from tags
        instruction_tags = [tag for tag in unified_claim.tags if tag.startswith("instruction:")]
        instruction_type = instruction_tags[0].split(":", 1)[1] if instruction_tags else "guidance"
        
        return {
            "id": unified_claim.id,
            "content": unified_claim.content,
            "instruction_type": instruction_type,
            "required_actions": [],  # Extract from content if needed
            "dependencies": unified_claim.supported_by,
            "confidence": unified_claim.confidence,
            "state": unified_claim.state.value
        }
```

## Phase 2: Context Builder Migration

### 2.1 Algorithm Migration

**Old Context Builder Analysis:**
```python
class OldContextAnalyzer:
    """Analyzes existing context building to understand migration requirements"""
    
    def analyze_existing_builders(self) -> Dict[str, Any]:
        """Analyze different context builders in the existing system"""
        
        builders = {
            "simple_context": self.analyze_simple_builder(),
            "instruction_context": self.analyze_instruction_builder(),
            "evidence_context": self.analyze_evidence_builder(),
            "hierarchical_context": self.analyze_hierarchical_builder()
        }
        
        # Identify common patterns
        common_patterns = self.identify_common_patterns(builders)
        
        # Calculate migration complexity
        complexity_score = self.calculate_migration_complexity(builders)
        
        return {
            "builders": builders,
            "common_patterns": common_patterns,
            "complexity_score": complexity_score,
            "migration_strategy": self.recommend_migration_strategy(complexity_score)
        }
    
    def analyze_simple_builder(self) -> Dict[str, Any]:
        """Analyze simple context builder implementation"""
        return {
            "algorithm": "basic_similarity_search",
            "data_sources": ["claims"],
            "complexity": "low",
            "migrate_as_is": True
        }
    
    def analyze_instruction_builder(self) -> Dict[str, Any]:
        """Analyze instruction-specific context builder"""
        return {
            "algorithm": "instruction_focused_search",
            "data_sources": ["instruction_claims", "supporting_evidence"],
            "complexity": "medium",
            "migrate_to": "semantic_similarity + relationship_traversal"
        }
```

**New Context Builder Implementation:**
```python
class MigratingContextBuilder:
    """Context builder that can operate with old and new systems during migration"""
    
    def __init__(self, migration_config: MigrationConfig):
        self.config = migration_config
        self.old_builders = self.initialize_old_builders()
        self.new_builder = CompleteRelationshipContextBuilder()
        self.mode = migration_config.mode  # "old", "new", or "hybrid"
    
    async def build_context(
        self, 
        target_claim_id: str, 
        max_tokens: int = 8000
    ) -> Dict[str, Any]:
        """Build context using appropriate builder based on migration mode"""
        
        if self.mode == "old":
            return await self.build_with_old_system(target_claim_id, max_tokens)
        elif self.mode == "new":
            return await self.build_with_new_system(target_claim_id, max_tokens)
        elif self.mode == "hybrid":
            return await self.build_hybrid_context(target_claim_id, max_tokens)
        else:
            raise ValueError(f"Unknown migration mode: {self.mode}")
    
    async def build_hybrid_context(
        self, 
        target_claim_id: str, 
        max_tokens: int
    ) -> Dict[str, Any]:
        """Build context using both old and new systems for comparison"""
        
        # Get results from both systems
        old_result = await self.build_with_old_system(target_claim_id, max_tokens)
        new_result = await self.build_with_new_system(target_claim_id, max_tokens)
        
        # Compare results
        comparison = self.compare_context_results(old_result, new_result)
        
        # Log differences for analysis
        if comparison["significant_differences"]:
            logger.warning(f"Context building differences detected: {comparison}")
        
        # Return new result with comparison metadata
        new_result["migration_metadata"] = comparison
        return new_result
    
    def compare_context_results(
        self, 
        old_result: Dict, 
        new_result: Dict
    ) -> Dict[str, Any]:
        """Compare results from old and new context builders"""
        
        comparison = {
            "token_difference": new_result["tokens_used"] - old_result["tokens_used"],
            "claim_count_difference": (
                new_result["total_claims"] - old_result["total_claims"]
            ),
            "coverage_difference": self.calculate_coverage_difference(old_result, new_result),
            "significant_differences": False
        }
        
        # Determine if differences are significant
        if abs(comparison["token_difference"]) > max_tokens * 0.1:  # 10% difference
            comparison["significant_differences"] = True
        
        return comparison
```

### 2.2 Performance Validation

**Performance Comparison Framework:**
```python
class ContextBuilderPerformanceValidator:
    """Validates that new context builder meets or exceeds performance"""
    
    async def run_performance_comparison(
        self, 
        test_claims: List[str],
        iterations: int = 100
    ) -> PerformanceComparisonResult:
        
        results = {
            "old_builder": await self.benchmark_builder(
                self.old_builder, test_claims, iterations
            ),
            "new_builder": await self.benchmark_builder(
                self.new_builder, test_claims, iterations
            )
        }
        
        # Calculate improvement metrics
        comparison = self.calculate_improvement_metrics(results)
        
        # Validate performance requirements
        validation = self.validate_performance_requirements(comparison)
        
        return PerformanceComparisonResult(
            raw_results=results,
            comparison=comparison,
            validation=validation
        )
    
    async def benchmark_builder(
        self, 
        builder, 
        test_claims: List[str], 
        iterations: int
    ) -> Dict[str, float]:
        
        metrics = {
            "avg_response_time": 0.0,
            "avg_tokens_used": 0.0,
            "avg_claims_included": 0.0,
            "success_rate": 0.0,
            "memory_usage": 0.0
        }
        
        total_response_time = 0.0
        total_tokens = 0.0
        total_claims = 0
        successful_requests = 0
        
        for claim_id in test_claims:
            for _ in range(iterations // len(test_claims)):
                start_time = time.time()
                
                try:
                    result = await builder.build_context(claim_id, 8000)
                    
                    end_time = time.time()
                    response_time = end_time - start_time
                    
                    total_response_time += response_time
                    total_tokens += result.get("tokens_used", 0)
                    total_claims += result.get("total_claims", 0)
                    successful_requests += 1
                    
                except Exception as e:
                    logger.error(f"Builder benchmark error for {claim_id}: {e}")
        
        total_requests = len(test_claims) * (iterations // len(test_claims))
        
        if successful_requests > 0:
            metrics["avg_response_time"] = total_response_time / successful_requests
            metrics["avg_tokens_used"] = total_tokens / successful_requests
            metrics["avg_claims_included"] = total_claims / successful_requests
            metrics["success_rate"] = successful_requests / total_requests
        
        return metrics
```

## Phase 3: LLM Integration

### 3.1 LLM Provider Migration

**LLM Integration Analysis:**
```python
class LLMIntegrationMigrator:
    """Manages migration to new LLM integration protocol"""
    
    def analyze_existing_llm_usage(self) -> Dict[str, Any]:
        """Analyze how LLMs are currently used in the system"""
        
        current_usage = {
            "prompt_templates": self.scan_prompt_templates(),
            "llm_calls": self.analyze_llm_call_patterns(),
            "response_processing": self.analyze_response_processing(),
            "error_handling": self.analyze_error_handling()
        }
        
        migration_complexity = self.assess_llm_migration_complexity(current_usage)
        
        return {
            "current_usage": current_usage,
            "migration_complexity": migration_complexity,
            "recommended_approach": self.recommend_llm_migration_approach(migration_complexity)
        }
    
    async def migrate_prompt_templates(self) -> TemplateMigrationResult:
        """Migrate existing prompt templates to new protocol"""
        
        # Find all existing template files
        template_files = self.find_template_files()
        
        migrated_templates = []
        migration_issues = []
        
        for template_file in template_files:
            try:
                # Analyze template purpose and structure
                template_analysis = self.analyze_template(template_file)
                
                # Convert to new template format
                new_template = self.convert_template(template_analysis)
                
                # Validate new template
                validation_result = await self.validate_new_template(new_template)
                
                if validation_result.valid:
                    migrated_templates.append({
                        "old_file": template_file,
                        "new_template": new_template,
                        "validation": validation_result
                    })
                else:
                    migration_issues.append({
                        "template_file": template_file,
                        "issues": validation_result.errors
                    })
                
            except Exception as e:
                migration_issues.append({
                    "template_file": template_file,
                    "error": str(e)
                })
        
        return TemplateMigrationResult(
            total_templates=len(template_files),
            migrated=len(migrated_templates),
            issues=migration_issues,
            templates=migrated_templates
        )
```

### 3.2 Gradual Rollout Strategy

**Progressive LLM Integration:**
```python
class ProgressiveLLMIntegration:
    """Manages gradual rollout of new LLM integration"""
    
    def __init__(self, rollout_config: RolloutConfig):
        self.config = rollout_config
        self.old_llm_client = OldLLMClient()
        self.new_llm_client = NewLLMClient()
        self.traffic_splitter = TrafficSplitter(rollout_config)
    
    async def process_request(
        self, 
        request: LLMRequest, 
        context: str
    ) -> LLMResponse:
        """Process request using appropriate LLM client based on rollout strategy"""
        
        # Determine which client to use
        use_new_client = self.traffic_splitter.should_use_new_client(
            user_id=request.user_id,
            request_type=request.type,
            rollout_percentage=self.config.current_percentage
        )
        
        if use_new_client:
            try:
                # Use new LLM integration
                logger.info(f"Using new LLM client for request {request.id}")
                return await self.new_llm_client.process_request(request, context)
            except Exception as e:
                logger.error(f"New LLM client failed: {e}, falling back to old client")
                # Fall back to old client for reliability
                return await self.old_llm_client.process_request(request, context)
        else:
            # Use old LLM integration
            logger.debug(f"Using old LLM client for request {request.id}")
            return await self.old_llm_client.process_request(request, context)
    
    async def compare_responses(self, request: LLMRequest, context: str) -> ComparisonResult:
        """Compare responses from old and new LLM clients for validation"""
        
        # Get responses from both clients
        old_response = await self.old_llm_client.process_request(request, context)
        new_response = await self.new_llm_client.process_request(request, context)
        
        # Compare results
        comparison = self.llm_response_comparator.compare(old_response, new_response)
        
        # Log significant differences
        if comparison.significance_score > 0.8:
            logger.warning(f"Significant LLM response difference: {comparison}")
        
        return comparison

class TrafficSplitter:
    """Manages traffic splitting for gradual rollout"""
    
    def __init__(self, config: RolloutConfig):
        self.config = config
        self.user_buckets = {}
    
    def should_use_new_client(
        self, 
        user_id: str, 
        request_type: str, 
        rollout_percentage: float
    ) -> bool:
        """Determine if request should use new client based on rollout strategy"""
        
        # Check for explicit feature flags
        if user_id in self.config.force_new_users:
            return True
        if user_id in self.config.force_old_users:
            return False
        
        # Check request type specific rollout
        type_rollout = self.config.type_rollout.get(request_type, rollout_percentage)
        
        # Consistent hashing for consistent experience
        user_bucket = self.get_user_bucket(user_id)
        threshold = int(user_bucket * type_rollout)
        
        return random.randint(0, user_bucket) < threshold
    
    def get_user_bucket(self, user_id: str) -> int:
        """Get consistent bucket for user"""
        if user_id not in self.user_buckets:
            # Hash user ID and map to bucket
            hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            self.user_buckets[user_id] = hash_value % self.config.bucket_count
        
        return self.user_buckets[user_id]
```

## Phase 4: Cleanup and Optimization

### 4.1 Legacy Code Removal

**Safe Code Removal Process:**
```python
class LegacyCodeRemover:
    """Safely removes old code after successful migration"""
    
    def __init__(self, removal_config: RemovalConfig):
        self.config = removal_config
        self.dependency_analyzer = CodeDependencyAnalyzer()
    
    async def plan_legacy_removal(self) -> RemovalPlan:
        """Plan safe removal of legacy code"""
        
        # Analyze code dependencies
        dependency_map = await self.dependency_analyzer.analyze_dependencies()
        
        # Identify removable code
        removable_modules = self.identify_removable_modules(dependency_map)
        
        # Create removal schedule
        removal_schedule = self.create_removal_schedule(removable_modules)
        
        return RemovalPlan(
            total_modules=len(removable_modules),
            removal_schedule=removal_schedule,
            risk_assessment=self.assess_removal_risk(removable_modules)
        )
    
    def identify_removable_modules(self, dependency_map: Dict) -> List[str]:
        """Identify modules that can be safely removed"""
        
        removable = []
        
        for module_path, dependencies in dependency_map.items():
            if self.is_legacy_module(module_path):
                # Check if still has active dependencies
                active_dependencies = [
                    dep for dep in dependencies 
                    if not self.is_legacy_module(dep)
                ]
                
                if len(active_dependencies) == 0:
                    removable.append(module_path)
                elif self.config.force_remove_orphans and len(active_dependencies) < 2:
                    # Allow removal if only few minor dependencies
                    removable.append(module_path)
        
        return removable
    
    async def execute_removal_plan(self, plan: RemovalPlan) -> RemovalResult:
        """Execute legacy code removal plan"""
        
        result = RemovalResult(total_modules=plan.total_modules)
        
        for phase, modules in plan.removal_schedule.items():
            logger.info(f"Starting removal phase {phase}: {len(modules)} modules")
            
            for module_path in modules:
                try:
                    # Create backup before removal
                    await self.create_module_backup(module_path)
                    
                    # Remove module
                    await self.remove_module(module_path)
                    
                    # Update imports
                    await self.update_module_imports(module_path)
                    
                    result.add_success(module_path)
                    logger.info(f"Successfully removed legacy module: {module_path}")
                    
                except Exception as e:
                    result.add_failure(module_path, str(e))
                    logger.error(f"Failed to remove legacy module {module_path}: {e}")
                    
                    # Attempt rollback
                    await self.restore_module_backup(module_path)
        
        return result
```

### 4.2 Performance Optimization

**System Optimization Post-Migration:**
```python
class PostMigrationOptimizer:
    """Optimizes system performance after migration"""
    
    async def optimize_system(self) -> OptimizationResult:
        """Run comprehensive post-migration optimization"""
        
        result = OptimizationResult()
        
        # Database optimization
        db_optimization = await self.optimize_database()
        result.add_phase("database", db_optimization)
        
        # Memory optimization
        memory_optimization = await self.optimize_memory_usage()
        result.add_phase("memory", memory_optimization)
        
        # Caching optimization
        cache_optimization = await self.optimize_caching()
        result.add_phase("caching", cache_optimization)
        
        # API optimization
        api_optimization = await self.optimize_api_performance()
        result.add_phase("api", api_optimization)
        
        return result
    
    async def optimize_database(self) -> DatabaseOptimizationResult:
        """Optimize database performance"""
        
        optimizations = []
        
        # Analyze query performance
        slow_queries = await self.identify_slow_queries()
        for query in slow_queries:
            optimization = await self.optimize_query(query)
            optimizations.append(optimization)
        
        # Update database statistics
        await self.update_database_statistics()
        
        # Rebuild indexes if needed
        index_optimizations = await self.optimize_indexes()
        optimizations.extend(index_optimizations)
        
        return DatabaseOptimizationResult(
            optimized_queries=len(optimizations),
            performance_improvement=self.calculate_performance_improvement(optimizations)
        )
    
    async def optimize_memory_usage(self) -> MemoryOptimizationResult:
        """Optimize memory usage"""
        
        # Analyze memory patterns
        memory_analysis = await self.analyze_memory_patterns()
        
        optimizations = []
        
        # Implement object pooling for frequently created objects
        if memory_analysis.high_allocation_rate:
            pool_optimization = await self.implement_object_pooling()
            optimizations.append(pool_optimization)
        
        # Optimize claim caching
        cache_optimization = await self.optimize_claim_cache()
        optimizations.append(cache_optimization)
        
        # Reduce memory footprint in context building
        context_optimization = await self.optimize_context_builder_memory()
        optimizations.append(context_optimization)
        
        return MemoryOptimizationResult(
            optimizations=optimizations,
            memory_reduction=self.calculate_memory_reduction(optimizations)
        )
```

## Risk Mitigation Strategies

### Migration Risk Assessment

```python
class MigrationRiskAssessor:
    """Assesses and mitigates migration risks"""
    
    def assess_migration_risks(self) -> RiskAssessment:
        """Comprehensive migration risk assessment"""
        
        risks = {
            "data_loss": self.assess_data_loss_risk(),
            "performance_degradation": self.assess_performance_risk(),
            "compatibility_issues": self.assess_compatibility_risk(),
            "user_impact": self.assess_user_impact_risk(),
            "rollback_complexity": self.assess_rollback_risk()
        }
        
        # Calculate overall risk score
        overall_score = self.calculate_overall_risk_score(risks)
        
        # Determine mitigation strategies
        mitigation_strategies = self.recommend_mitigation_strategies(risks)
        
        return RiskAssessment(
            individual_risks=risks,
            overall_score=overall_score,
            mitigation_strategies=mitigation_strategies,
            go_no_go_recommendation=self.make_go_no_go_recommendation(overall_score)
        )
    
    def assess_data_loss_risk(self) -> RiskLevel:
        """Assess risk of data loss during migration"""
        
        risk_factors = []
        
        # Check data complexity
        data_complexity = self.analyze_data_complexity()
        if data_complexity > 0.8:
            risk_factors.append("high_data_complexity")
        
        # Check conversion complexity
        conversion_complexity = self.analyze_conversion_complexity()
        if conversion_complexity > 0.7:
            risk_factors.append("complex_conversion_logic")
        
        # Check backup availability
        backup_availability = self.check_backup_availability()
        if backup_availability < 0.9:
            risk_factors.append("inadequate_backups")
        
        # Calculate risk level
        risk_score = len(risk_factors) / 3.0  # Normalized by max factors
        
        return RiskLevel(
            score=risk_score,
            factors=risk_factors,
            severity="high" if risk_score > 0.6 else "medium" if risk_score > 0.3 else "low"
        )
```

### Rollback Procedures

```python
class MigrationRollbackManager:
    """Handles rollback procedures if migration fails"""
    
    def __init__(self, rollback_config: RollbackConfig):
        self.config = rollback_config
        self.backup_manager = BackupManager()
    
    async def rollback_phase(self, phase: int) -> RollbackResult:
        """Rollback a specific migration phase"""
        
        logger.warning(f"Initiating rollback for Phase {phase}")
        
        try:
            if phase == 1:
                result = await self.rollback_phase1()
            elif phase == 2:
                result = await self.rollback_phase2()
            elif phase == 3:
                result = await self.rollback_phase3()
            else:
                raise ValueError(f"Unknown phase for rollback: {phase}")
            
            # Validate rollback success
            validation = await self.validate_rollback(phase)
            result.validation = validation
            
            return result
            
        except Exception as e:
            logger.error(f"Rollback failed for Phase {phase}: {e}")
            return RollbackResult(
                success=False,
                error=str(e),
                phase=phase
            )
    
    async def rollback_phase1(self) -> RollbackResult:
        """Rollback Phase 1: Data Model Unification"""
        
        steps_completed = []
        errors = []
        
        try:
            # Step 1: Restore old tables from backup
            await self.backup_manager.restore_tables(["enhanced_claims", "instruction_claims", "evidence_claims"])
            steps_completed.append("old_tables_restored")
            
            # Step 2: Drop unified tables
            await self.drop_unified_tables()
            steps_completed.append("unified_tables_dropped")
            
            # Step 3: Restore old indexes
            await self.restore_old_indexes()
            steps_completed.append("old_indexes_restored")
            
            # Step 4: Update API to use old models
            await self.switch_api_to_old_models()
            steps_completed.append("api_switched_to_old")
            
            return RollbackResult(
                success=True,
                phase=1,
                steps_completed=steps_completed
            )
            
        except Exception as e:
            errors.append(str(e))
            return RollbackResult(
                success=False,
                phase=1,
                steps_completed=steps_completed,
                errors=errors
            )
```

## Timeline and Milestones

### Migration Timeline

```
Phase 1: Data Model Unification (2-3 weeks)
├── Week 1: Schema analysis and conversion scripts
├── Week 2: Data migration and validation
└── Week 3: API compatibility and testing

Phase 2: Context Builder Migration (2 weeks)
├── Week 4: Algorithm migration and performance验证
└── Week 5: Complete rollout and monitoring

Phase 3: LLM Integration (2-3 weeks)
├── Week 6: LLM protocol implementation
├── Week 7: Gradual rollout and comparison
└── Week 8: Complete migration and optimization

Phase 4: Cleanup and Optimization (1-2 weeks)
├── Week 9: Legacy code removal
└── Week 10: System optimization and documentation

Total Estimated Time: 8-10 weeks
```

### Success Metrics

**Phase 1 Success Criteria:**
- 100% data migration without loss
- Performance within 10% of baseline
- Zero critical bugs in compatibility layer
- All automated tests passing

**Phase 2 Success Criteria:**
- Context building time < 200ms average
- Memory usage < 100MB for 10,000 claims
- Coverage equivalent or better than old system
- 95%+ API compatibility maintained

**Phase 3 Success Criteria:**
- LLM response quality maintained or improved
- Response time within 10% of baseline
- Zero compatibility issues
- Complete feature parity achieved

**Phase 4 Success Criteria:**
- 50%+ reduction in codebase complexity
- 20%+ performance improvement
- All legacy dependencies removed
- Documentation updated and complete

This migration strategy provides a comprehensive, safe approach to transitioning to the Simple Universal Claim Architecture while minimizing risk and ensuring business continuity.