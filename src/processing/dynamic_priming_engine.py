"""
Dynamic Priming Engine for Conjecture Database Enhancement

Generates foundational knowledge claims through LLM-powered dynamic priming
across 4 domains: fact-checking, programming, scientific method, critical thinking.
Integrates with existing SQLite and ChromaDB infrastructure for seamless knowledge enhancement.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import json
import uuid

from src.core.models import Claim, ClaimType, ClaimState, DirtyReason
from src.data.data_manager import DataManager
from src.processing.unified_bridge import UnifiedLLMBridge as LLMBridge, LLMRequest
from src.processing.llm_prompts.xml_optimized_templates import XMLOptimizedTemplateManager


class PrimingDomain(str, Enum):
    """Priming domains for foundational knowledge enhancement"""
    
    FACT_CHECKING = "fact_checking"
    PROGRAMMING = "programming"
    SCIENTIFIC_METHOD = "scientific_method"
    CRITICAL_THINKING = "critical_thinking"


class PrimingImpact:
    """Tracks priming impact metrics"""
    
    def __init__(self):
        self.claims_generated = 0
        self.claims_stored = 0
        self.embedding_generation_time = 0.0
        self.llm_generation_time = 0.0
        self.total_time = 0.0
        self.errors = []
        self.domain_coverage = {}
        self.quality_scores = []


class DynamicPrimingEngine:
    """
    Dynamic LLM-generated foundational claims engine for database priming.
    
    Enhances reasoning quality by generating domain-specific foundational knowledge
    claims that improve context quality and evidence utilization.
    """
    
    def __init__(
        self, 
        data_manager: DataManager,
        llm_bridge: LLMBridge,
        template_manager: Optional[XMLOptimizedTemplateManager] = None
    ):
        """Initialize Dynamic Priming Engine"""
        self.data_manager = data_manager
        self.llm_bridge = llm_bridge
        self.template_manager = template_manager or XMLOptimizedTemplateManager()
        
        # Priming configuration
        self.priming_config = {
            "max_claims_per_domain": 25,
            "confidence_threshold": 0.8,
            "min_quality_score": 0.7,
            "regeneration_interval_hours": 24,
            "similarity_threshold": 0.85,  # Avoid duplicate claims
        }
        
        # Domain-specific priming prompts
        self.domain_prompts = {
            PrimingDomain.FACT_CHECKING: self._create_fact_checking_prompt(),
            PrimingDomain.PROGRAMMING: self._create_programming_prompt(),
            PrimingDomain.SCIENTIFIC_METHOD: self._create_scientific_method_prompt(),
            PrimingDomain.CRITICAL_THINKING: self._create_critical_thinking_prompt(),
        }
        
        # Impact tracking
        self.impact_history = []
        self.current_session_impact = PrimingImpact()
        
        # State tracking
        self._priming_active = False
        self._last_priming_time = None
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the priming engine"""
        try:
            # Ensure data manager is initialized
            if not self.data_manager._initialized:
                await self.data_manager.initialize()
            
            # Verify LLM bridge connectivity
            test_request = LLMRequest(
                prompt="Test connectivity",
                max_tokens=10,
                temperature=0.1
            )
            await self.llm_bridge.generate_response(test_request)
            
            self.logger.info("Dynamic Priming Engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Dynamic Priming Engine: {e}")
            raise
    
    async def prime_database(
        self, 
        domains: Optional[List[PrimingDomain]] = None,
        force_regeneration: bool = False
    ) -> PrimingImpact:
        """
        Prime the database with foundational claims across specified domains.
        
        Args:
            domains: List of domains to prime (default: all domains)
            force_regeneration: Force regeneration even if recently primed
            
        Returns:
            PrimingImpact: Metrics and results from priming operation
        """
        if self._priming_active:
            raise RuntimeError("Priming already in progress")
        
        start_time = datetime.utcnow()
        self._priming_active = True
        self.current_session_impact = PrimingImpact()
        
        try:
            # Check if priming is needed
            if not force_regeneration and not self._should_prime():
                self.logger.info("Database priming not needed at this time")
                return self.current_session_impact
            
            # Use all domains if none specified
            if not domains:
                domains = list(PrimingDomain)
            
            self.logger.info(f"Starting database priming for domains: {domains}")
            
            # Process each domain
            for domain in domains:
                await self._prime_domain(domain)
            
            # Calculate total time
            self.current_session_impact.total_time = (
                datetime.utcnow() - start_time
            ).total_seconds()
            
            # Store impact metrics
            self.impact_history.append(self.current_session_impact)
            self._last_priming_time = start_time
            
            self.logger.info(
                f"Database priming completed: "
                f"{self.current_session_impact.claims_stored} claims generated, "
                f"{self.current_session_impact.total_time:.2f}s total"
            )
            
            return self.current_session_impact
            
        except Exception as e:
            self.current_session_impact.errors.append(str(e))
            self.logger.error(f"Database priming failed: {e}")
            raise
        finally:
            self._priming_active = False
    
    async def _prime_domain(self, domain: PrimingDomain) -> None:
        """Prime a specific domain with foundational claims"""
        try:
            self.logger.info(f"Priming domain: {domain.value}")
            
            # Generate domain-specific claims
            domain_claims = await self._generate_domain_claims(domain)
            self.current_session_impact.claims_generated += len(domain_claims)
            
            # Filter and store high-quality claims
            stored_claims = await self._filter_and_store_claims(domain_claims, domain)
            self.current_session_impact.claims_stored += len(stored_claims)
            
            # Track domain coverage
            self.current_session_impact.domain_coverage[domain.value] = {
                "generated": len(domain_claims),
                "stored": len(stored_claims),
                "quality_score": self._calculate_domain_quality(stored_claims)
            }
            
        except Exception as e:
            self.current_session_impact.errors.append(f"Domain {domain.value}: {e}")
            self.logger.error(f"Failed to prime domain {domain.value}: {e}")
    
    async def _generate_domain_claims(self, domain: PrimingDomain) -> List[Claim]:
        """Generate claims for a specific domain using LLM"""
        start_time = datetime.utcnow()
        
        try:
            # Get domain-specific prompt
            prompt = self.domain_prompts[domain]
            
            # Create LLM request
            llm_request = LLMRequest(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.3,  # Lower temperature for consistent foundational knowledge
                response_format="xml"
            )
            
            # Generate response
            response = await self.llm_bridge.generate_response(llm_request)
            
            # Parse XML response to extract claims
            claims = self._parse_xml_claims(response.content, domain)
            
            # Track LLM generation time
            llm_time = (datetime.utcnow() - start_time).total_seconds()
            self.current_session_impact.llm_generation_time += llm_time
            
            self.logger.info(f"Generated {len(claims)} claims for domain {domain.value}")
            return claims
            
        except Exception as e:
            self.logger.error(f"Failed to generate claims for domain {domain.value}: {e}")
            return []
    
    def _parse_xml_claims(self, xml_content: str, domain: PrimingDomain) -> List[Claim]:
        """Parse XML content to extract Claim objects"""
        claims = []
        
        try:
            import xml.etree.ElementTree as ET
            
            # Parse XML
            root = ET.fromstring(xml_content)
            
            # Extract claim elements
            for claim_elem in root.findall(".//claim"):
                try:
                    # Extract claim data
                    claim_id = f"prime_{domain.value[:3]}_{uuid.uuid4().hex[:8]}"
                    content = claim_elem.findtext("content", "").strip()
                    claim_type = claim_elem.get("type", "concept")
                    confidence = float(claim_elem.get("confidence", 0.8))
                    evidence = claim_elem.findtext("evidence", "").strip()
                    reasoning = claim_elem.findtext("reasoning", "").strip()
                    
                    # Validate required fields
                    if not content or len(content) < 10:
                        continue
                    
                    # Create claim object
                    claim = Claim(
                        id=claim_id,
                        content=content,
                        confidence=confidence,
                        state=ClaimState.VALIDATED,
                        tags=[domain.value, "primed", "foundational"],
                        scope="public"  # Foundational knowledge is public
                    )
                    
                    # Add domain-specific metadata
                    if evidence:
                        claim.tags.append("has_evidence")
                    if reasoning:
                        claim.tags.append("has_reasoning")
                    
                    claims.append(claim)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to parse claim element: {e}")
                    continue
            
        except Exception as e:
            self.logger.error(f"Failed to parse XML claims: {e}")
        
        return claims
    
    async def _filter_and_store_claims(
        self, 
        claims: List[Claim], 
        domain: PrimingDomain
    ) -> List[Claim]:
        """Filter claims by quality and similarity, then store in database"""
        start_time = datetime.utcnow()
        stored_claims = []
        
        try:
            # Filter by confidence threshold
            high_confidence_claims = [
                claim for claim in claims 
                if claim.confidence >= self.priming_config["confidence_threshold"]
            ]
            
            # Check for duplicates using similarity search
            for claim in high_confidence_claims:
                try:
                    # Search for similar existing claims
                    similar_claims = await self.data_manager.search_claims(
                        query=claim.content,
                        limit=5,
                        confidence_threshold=0.5
                    )
                    
                    # Check similarity threshold
                    is_duplicate = False
                    for similar in similar_claims:
                        if similar.get("similarity", 0) > self.priming_config["similarity_threshold"]:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        # Store the claim
                        await self.data_manager.create_claim(
                            content=claim.content,
                            confidence=claim.confidence,
                            tags=claim.tags,
                            claim_id=claim.id,
                            state=claim.state,
                            scope=claim.scope
                        )
                        stored_claims.append(claim)
                        
                        # Limit claims per domain
                        if len(stored_claims) >= self.priming_config["max_claims_per_domain"]:
                            break
                
                except Exception as e:
                    self.logger.warning(f"Failed to store claim {claim.id}: {e}")
                    continue
            
            # Track embedding generation time
            embed_time = (datetime.utcnow() - start_time).total_seconds()
            self.current_session_impact.embedding_generation_time += embed_time
            
            self.logger.info(f"Stored {len(stored_claims)} claims for domain {domain.value}")
            return stored_claims
            
        except Exception as e:
            self.logger.error(f"Failed to filter and store claims for domain {domain.value}: {e}")
            return []
    
    def _calculate_domain_quality(self, claims: List[Claim]) -> float:
        """Calculate average quality score for domain claims"""
        if not claims:
            return 0.0
        
        total_confidence = sum(claim.confidence for claim in claims)
        return total_confidence / len(claims)
    
    def _should_prime(self) -> bool:
        """Check if database priming should be performed"""
        if not self._last_priming_time:
            return True
        
        interval = timedelta(hours=self.priming_config["regeneration_interval_hours"])
        return datetime.utcnow() - self._last_priming_time > interval
    
    def _create_fact_checking_prompt(self) -> str:
        """Create fact-checking domain priming prompt"""
        return '''Generate foundational fact-checking claims using XML format.

<fact_checking_priming>
Create 15-20 high-quality foundational claims about fact-checking methodologies, 
verification techniques, and critical evaluation of information sources.

Focus areas:
- Source credibility assessment
- Verification methodologies
- Bias detection techniques
- Information quality evaluation
- Cross-referencing strategies
- Logical fallacy identification
- Evidence strength assessment
- Fact-checking workflow processes
</fact_checking_priming>

Generate claims using this XML structure:
<claims>
  <claim id="fc1" type="fact" confidence="0.95">
    <content>Clear factual claim about fact-checking methodology</content>
    <evidence>Supporting evidence and best practices</evidence>
    <reasoning>Logical reasoning and methodology explanation</reasoning>
  </claim>
  
  <claim id="fc2" type="concept" confidence="0.90">
    <content>Key concept in information verification</content>
    <evidence>Explanatory evidence and examples</evidence>
    <reasoning>Conceptual framework and application</reasoning>
  </claim>
  
  <!-- Add more claims up to 20 -->
</claims>

Requirements:
- Each claim must be actionable and specific
- Confidence scores should reflect certainty level (0.8-1.0)
- Include practical verification techniques
- Cover diverse fact-checking scenarios
- Ensure claims are foundational and widely applicable'''
    
    def _create_programming_prompt(self) -> str:
        """Create programming domain priming prompt"""
        return '''Generate foundational programming claims using XML format.

<programming_priming>
Create 15-20 high-quality foundational claims about programming best practices,
code quality, software engineering principles, and development methodologies.

Focus areas:
- Code readability and maintainability
- Testing strategies and methodologies
- Design patterns and architecture
- Performance optimization techniques
- Security best practices
- Debugging and troubleshooting
- Version control and collaboration
- Documentation standards
</programming_priming>

Generate claims using this XML structure:
<claims>
  <claim id="pg1" type="fact" confidence="0.95">
    <content>Factual claim about programming best practice</content>
    <evidence>Supporting evidence from software engineering</evidence>
    <reasoning>Technical reasoning and benefits</reasoning>
  </claim>
  
  <claim id="pg2" type="concept" confidence="0.90">
    <content>Key programming concept or principle</content>
    <evidence>Explanatory evidence and examples</evidence>
    <reasoning>Conceptual framework and application</reasoning>
  </claim>
  
  <!-- Add more claims up to 20 -->
</claims>

Requirements:
- Each claim must be language-agnostic where possible
- Include modern software engineering practices
- Confidence scores should reflect certainty level (0.8-1.0)
- Cover diverse programming scenarios
- Ensure claims are foundational and time-tested'''
    
    def _create_scientific_method_prompt(self) -> str:
        """Create scientific method domain priming prompt"""
        return '''Generate foundational scientific method claims using XML format.

<scientific_method_priming>
Create 15-20 high-quality foundational claims about scientific methodology,
research practices, experimental design, and evidence evaluation.

Focus areas:
- Hypothesis formulation and testing
- Experimental design principles
- Data collection and analysis
- Statistical reasoning and interpretation
- Peer review and validation
- Reproducibility and replicability
- Scientific ethics and integrity
- Theory development and validation
</scientific_method_priming>

Generate claims using this XML structure:
<claims>
  <claim id="sm1" type="fact" confidence="0.95">
    <content>Factual claim about scientific methodology</content>
    <evidence>Supporting evidence from scientific literature</evidence>
    <reasoning>Scientific reasoning and justification</reasoning>
  </claim>
  
  <claim id="sm2" type="concept" confidence="0.90">
    <content>Key scientific method concept</content>
    <evidence>Explanatory evidence and examples</evidence>
    <reasoning>Conceptual framework and application</reasoning>
  </claim>
  
  <!-- Add more claims up to 20 -->
</claims>

Requirements:
- Each claim must be scientifically accurate and verifiable
- Include established scientific principles
- Confidence scores should reflect certainty level (0.8-1.0)
- Cover diverse scientific disciplines
- Ensure claims are foundational to scientific practice'''
    
    def _create_critical_thinking_prompt(self) -> str:
        """Create critical thinking domain priming prompt"""
        return '''Generate foundational critical thinking claims using XML format.

<critical_thinking_priming>
Create 15-20 high-quality foundational claims about critical thinking,
logical reasoning, argument analysis, and cognitive biases.

Focus areas:
- Logical reasoning principles
- Argument structure and evaluation
- Cognitive bias identification
- Evidence assessment techniques
- Logical fallacy recognition
- Analytical thinking methods
- Decision-making frameworks
- Problem-solving strategies
</critical_thinking_priming>

Generate claims using this XML structure:
<claims>
  <claim id="ct1" type="fact" confidence="0.95">
    <content>Factual claim about critical thinking principle</content>
    <evidence>Supporting evidence from cognitive science</evidence>
    <reasoning>Logical reasoning and justification</reasoning>
  </claim>
  
  <claim id="ct2" type="concept" confidence="0.90">
    <content>Key critical thinking concept</content>
    <evidence>Explanatory evidence and examples</evidence>
    <reasoning>Conceptual framework and application</reasoning>
  </claim>
  
  <!-- Add more claims up to 20 -->
</claims>

Requirements:
- Each claim must be logically sound and well-established
- Include cognitive science and psychology insights
- Confidence scores should reflect certainty level (0.8-1.0)
- Cover diverse thinking scenarios
- Ensure claims are foundational to rational thought'''
    
    async def get_priming_statistics(self) -> Dict[str, Any]:
        """Get comprehensive priming statistics and impact metrics"""
        try:
            # Get database statistics
            db_stats = await self.data_manager.get_statistics()
            
            # Count primed claims by domain
            primed_counts = {}
            for domain in PrimingDomain:
                domain_claims = await self.data_manager.search_claims(
                    query=domain.value,
                    limit=100,
                    claim_types=["primed"]
                )
                primed_counts[domain.value] = len(domain_claims)
            
            # Calculate impact metrics
            total_claims_generated = sum(
                impact.claims_generated for impact in self.impact_history
            )
            total_claims_stored = sum(
                impact.claims_stored for impact in self.impact_history
            )
            avg_quality = sum(
                impact.quality_scores for impact in self.impact_history
            ) / len(self.impact_history) if self.impact_history else 0.0
            
            return {
                "priming_engine": {
                    "active": self._priming_active,
                    "last_priming": self._last_priming_time.isoformat() if self._last_priming_time else None,
                    "total_sessions": len(self.impact_history),
                    "configuration": self.priming_config
                },
                "impact_metrics": {
                    "total_claims_generated": total_claims_generated,
                    "total_claims_stored": total_claims_stored,
                    "storage_efficiency": total_claims_stored / total_claims_generated if total_claims_generated > 0 else 0.0,
                    "average_quality_score": avg_quality
                },
                "domain_coverage": primed_counts,
                "database_stats": db_stats,
                "current_session": {
                    "claims_generated": self.current_session_impact.claims_generated,
                    "claims_stored": self.current_session_impact.claims_stored,
                    "errors": len(self.current_session_impact.errors),
                    "domain_coverage": self.current_session_impact.domain_coverage
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get priming statistics: {e}")
            return {"error": str(e)}
    
    async def reset_priming_data(self) -> bool:
        """Reset all priming data (for testing/recovery)"""
        try:
            # Delete all primed claims
            for domain in PrimingDomain:
                domain_claims = await self.data_manager.search_claims(
                    query=domain.value,
                    limit=1000,
                    claim_types=["primed"]
                )
                
                for claim in domain_claims:
                    await self.data_manager.delete_claim(claim["id"])
            
            # Reset impact history
            self.impact_history.clear()
            self.current_session_impact = PrimingImpact()
            self._last_priming_time = None
            
            self.logger.info("Priming data reset completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to reset priming data: {e}")
            return False