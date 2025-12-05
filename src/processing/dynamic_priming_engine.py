"""
Dynamic Priming Engine for Conjecture - REAL IMPLEMENTATION

Generates foundational claims using actual LLM provider calls for database priming.
This implementation uses REAL API calls, not simulation.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from ..core.models import Claim, ClaimState
from .unified_bridge import UnifiedLLMBridge, LLMRequest
from .simplified_llm_manager import get_simplified_llm_manager
from ..data.repositories import get_data_manager, RepositoryFactory


class DynamicPrimingEngine:
    """
    Dynamic Priming Engine that generates foundational claims using real LLM calls
    for database priming to improve reasoning quality.
    """
    
    def __init__(self):
        """Initialize the dynamic priming engine with real LLM integration"""
        self.logger = logging.getLogger(__name__)
        
        # Initialize real LLM components
        self.llm_manager = get_simplified_llm_manager()
        self.llm_bridge = UnifiedLLMBridge(self.llm_manager)
        self.data_manager = get_data_manager(use_mock_embeddings=False)
        self.claim_repository = RepositoryFactory.create_claim_repository(self.data_manager)
        
        # Priming domains with specific queries
        self.priming_domains = {
            "fact_checking": {
                "query": "What are the best practices for fact checking and verifying information accuracy?",
                "description": "Methodologies and principles for verifying factual information"
            },
            "programming": {
                "query": "What are the fundamental principles and best practices for software development and programming?",
                "description": "Core programming concepts, methodologies, and best practices"
            },
            "scientific_method": {
                "query": "What is the scientific method and how does it ensure reliable knowledge generation?",
                "description": "Principles of scientific inquiry and knowledge validation"
            },
            "critical_thinking": {
                "query": "What are the core principles and techniques of critical thinking for logical analysis?",
                "description": "Methods for logical reasoning and critical analysis"
            }
        }
        
        # Performance tracking
        self.priming_stats = {
            "total_claims_generated": 0,
            "total_claims_stored": 0,
            "domains_processed": 0,
            "processing_time": 0.0,
            "llm_calls_made": 0,
            "errors": []
        }
        
        self.logger.info("DynamicPrimingEngine initialized with real LLM integration")
    
    async def initialize(self) -> bool:
        """Initialize the priming engine and verify LLM availability"""
        try:
            # Initialize data manager
            await self.data_manager.initialize()
            
            # Verify LLM availability
            if not self.llm_bridge.is_available():
                raise RuntimeError("No LLM providers available for priming")
            
            providers = self.llm_bridge.get_available_providers()
            self.logger.info(f"Priming engine initialized with providers: {providers}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize priming engine: {e}")
            self.priming_stats["errors"].append(str(e))
            return False
    
    async def prime_database(self, domains: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Prime database with foundational claims using real LLM calls
        
        Args:
            domains: List of domains to prime. If None, primes all domains.
            
        Returns:
            Dictionary with priming results and statistics
        """
        start_time = time.time()
        
        try:
            if domains is None:
                domains = list(self.priming_domains.keys())
            
            self.logger.info(f"Starting database priming for domains: {domains}")
            
            priming_results = {}
            
            for domain in domains:
                if domain not in self.priming_domains:
                    self.logger.warning(f"Unknown domain: {domain}")
                    continue
                
                try:
                    # Generate foundational claims for this domain
                    domain_result = await self._prime_domain(domain)
                    priming_results[domain] = domain_result
                    
                    if domain_result.get("success", False):
                        self.priming_stats["domains_processed"] += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to prime domain {domain}: {e}")
                    priming_results[domain] = {
                        "success": False,
                        "error": str(e),
                        "claims_generated": 0,
                        "claims_stored": 0
                    }
                    self.priming_stats["errors"].append(f"{domain}: {e}")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.priming_stats["processing_time"] += processing_time
            
            # Create comprehensive result
            result = {
                "success": True,
                "domains_processed": self.priming_stats["domains_processed"],
                "total_claims_generated": self.priming_stats["total_claims_generated"],
                "total_claims_stored": self.priming_stats["total_claims_stored"],
                "processing_time": processing_time,
                "llm_calls_made": self.priming_stats["llm_calls_made"],
                "domain_results": priming_results,
                "priming_stats": self.priming_stats.copy(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Database priming completed: {self.priming_stats['total_claims_stored']} claims stored")
            return result
            
        except Exception as e:
            self.logger.error(f"Database priming failed: {e}")
            self.priming_stats["errors"].append(str(e))
            return {
                "success": False,
                "error": str(e),
                "priming_stats": self.priming_stats.copy()
            }
    
    async def _prime_domain(self, domain: str) -> Dict[str, Any]:
        """
        Prime a specific domain with foundational claims
        
        Args:
            domain: Domain name to prime
            
        Returns:
            Dictionary with domain priming results
        """
        try:
            domain_config = self.priming_domains[domain]
            query = domain_config["query"]
            description = domain_config["description"]
            
            self.logger.info(f"Priming domain: {domain}")
            
            # Generate foundational claims using real LLM call
            claims = await self._generate_foundational_claims(domain, query, description)
            
            if not claims:
                return {
                    "success": False,
                    "error": "No claims generated",
                    "domain": domain,
                    "claims_generated": 0,
                    "claims_stored": 0
                }
            
            # Store claims in database
            stored_claims = await self._store_priming_claims(claims, domain)
            
            # Update statistics
            self.priming_stats["total_claims_generated"] += len(claims)
            self.priming_stats["total_claims_stored"] += len(stored_claims)
            self.priming_stats["llm_calls_made"] += 1
            
            result = {
                "success": True,
                "domain": domain,
                "query": query,
                "description": description,
                "claims_generated": len(claims),
                "claims_stored": len(stored_claims),
                "storage_success_rate": len(stored_claims) / len(claims) if claims else 0,
                "claim_ids": [claim.id for claim in stored_claims]
            }
            
            self.logger.info(f"Domain {domain} primed: {len(stored_claims)}/{len(claims)} claims stored")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to prime domain {domain}: {e}")
            return {
                "success": False,
                "error": str(e),
                "domain": domain,
                "claims_generated": 0,
                "claims_stored": 0
            }
    
    async def _generate_foundational_claims(self, domain: str, query: str, description: str) -> List[Claim]:
        """
        Generate foundational claims for a domain using real LLM calls
        
        Args:
            domain: Domain name
            query: Priming query for the domain
            description: Domain description
            
        Returns:
            List of generated foundational claims
        """
        try:
            # Build comprehensive priming prompt
            prompt = f"""You are an expert in {domain} tasked with generating foundational knowledge claims.

DOMAIN: {domain}
DESCRIPTION: {description}
PRIMING QUERY: {query}

Generate 5-7 foundational claims that establish core principles and best practices for {domain}.
These claims will serve as knowledge priming for future AI reasoning tasks.

Requirements:
1. Focus on foundational, widely accepted principles and practices
2. Use XML format: <claim type="[principle|fact|methodology|guideline]" confidence="[0.8-1.0]">content</claim>
3. Include clear, actionable guidance that can be applied broadly
4. Provide high confidence scores (0.8-1.0) for well-established principles
5. Cover different aspects: core principles, practical methods, quality guidelines
6. Ensure claims are educational and foundational

Generate claims using this XML structure:
<claims>
  <claim type="principle" confidence="0.95">Core foundational principle for {domain}</claim>
  <claim type="fact" confidence="0.90">Established fact in {domain}</claim>
  <claim type="methodology" confidence="0.85">Practical methodology for {domain}</claim>
  <claim type="guideline" confidence="0.88">Best practice guideline for {domain}</claim>
  <!-- Add more claims as needed -->
</claims>"""

            # Make REAL LLM call
            llm_request = LLMRequest(
                prompt=prompt,
                max_tokens=2500,
                temperature=0.5,  # Lower temperature for foundational knowledge
                task_type="priming"
            )
            
            self.logger.info(f"Making real LLM call for {domain} priming")
            response = self.llm_bridge.process(llm_request)
            
            if not response.success:
                raise Exception(f"LLM priming call failed: {response.errors}")
            
            # Parse claims from response
            claims = self._parse_xml_claims(response.content, domain)
            
            self.logger.info(f"Generated {len(claims)} foundational claims for {domain}")
            return claims
            
        except Exception as e:
            self.logger.error(f"Failed to generate foundational claims for {domain}: {e}")
            return []
    
    def _parse_xml_claims(self, response: str, domain: str) -> List[Claim]:
        """
        Parse claims from XML response
        
        Args:
            response: LLM response containing XML claims
            domain: Domain name for metadata
            
        Returns:
            List of parsed Claim objects
        """
        claims = []
        
        try:
            import re
            
            # Find all claim tags
            claim_pattern = r'<claim\s+type="([^"]*)"\s+confidence="([^"]*)"[^>]*>(.*?)</claim>'
            matches = re.findall(claim_pattern, response, re.DOTALL)
            
            for i, (claim_type, confidence, content) in enumerate(matches):
                try:
                    # Validate and clean claim data
                    claim_type = claim_type.strip()
                    confidence = float(confidence.strip())
                    content = content.strip()
                    
                    # Skip empty or invalid claims
                    if not content or len(content) < 10:
                        continue
                    
                    claim = Claim(
                        id=f"priming_{domain}_{int(time.time())}_{i}",
                        content=content,
                        claim_type=claim_type,
                        confidence=confidence,
                        state=ClaimState.VALIDATED,  # Priming claims are pre-validated
                        tags=["priming", "foundational", domain]
                    )
                    claims.append(claim)
                    
                except (ValueError, AttributeError) as e:
                    self.logger.warning(f"Failed to parse claim for {domain}: {e}")
                    continue
            
            if not claims:
                # Fallback: try to extract meaningful content
                self.logger.warning(f"No XML claims found for {domain}, trying fallback parsing")
                lines = response.split('\n')
                for i, line in enumerate(lines):
                    if line.strip() and len(line.strip()) > 20:
                        try:
                            claim = Claim(
                                id=f"fallback_{domain}_{int(time.time())}_{i}",
                                content=line.strip(),
                                claim_type="principle",
                                confidence=0.8,
                                state=ClaimState.VALIDATED,
                                tags=["priming", "foundational", domain, "fallback"]
                            )
                            claims.append(claim)
                        except Exception:
                            continue
            
            self.logger.info(f"Parsed {len(claims)} claims from XML response for {domain}")
            return claims[:10]  # Limit to 10 claims per domain
            
        except Exception as e:
            self.logger.error(f"Failed to parse XML claims for {domain}: {e}")
            return []
    
    async def _store_priming_claims(self, claims: List[Claim], domain: str) -> List[Claim]:
        """
        Store priming claims in the database
        
        Args:
            claims: List of claims to store
            domain: Domain name for metadata
            
        Returns:
            List of successfully stored claims
        """
        stored_claims = []
        
        for claim in claims:
            try:
                # Prepare claim data with metadata
                claim_data = {
                    "content": claim.content,
                    "confidence": claim.confidence,
                    "claim_type": claim.claim_type,
                    "tags": claim.tags + ["database_priming"],
                    "state": ClaimState.VALIDATED,
                    "metadata": {
                        "domain": domain,
                        "priming_timestamp": datetime.utcnow().isoformat(),
                        "priming_engine": "DynamicPrimingEngine",
                        "claim_purpose": "foundational_knowledge"
                    }
                }
                
                # Store in database
                stored_claim = await self.claim_repository.create(claim_data)
                stored_claims.append(stored_claim)
                
            except Exception as e:
                self.logger.warning(f"Failed to store priming claim: {e}")
                continue
        
        self.logger.info(f"Stored {len(stored_claims)}/{len(claims)} priming claims for {domain}")
        return stored_claims
    
    async def get_priming_status(self) -> Dict[str, Any]:
        """
        Get current priming status and statistics
        
        Returns:
            Dictionary with priming status information
        """
        try:
            # Count priming claims in database
            priming_claims = await self.claim_repository.search_by_tags(
                ["priming"], 
                limit=1000
            )
            
            # Group by domain
            domain_counts = {}
            for claim in priming_claims:
                for tag in claim.tags:
                    if tag in self.priming_domains:
                        domain_counts[tag] = domain_counts.get(tag, 0) + 1
            
            return {
                "total_priming_claims": len(priming_claims),
                "claims_by_domain": domain_counts,
                "priming_stats": self.priming_stats.copy(),
                "llm_providers_available": self.llm_bridge.get_available_providers(),
                "last_primed": datetime.utcnow().isoformat(),
                "engine_ready": self.llm_bridge.is_available()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get priming status: {e}")
            return {
                "error": str(e),
                "priming_stats": self.priming_stats.copy()
            }
    
    async def clear_priming_data(self) -> Dict[str, Any]:
        """
        Clear all priming data from database (for testing/reset purposes)
        
        Returns:
            Dictionary with clearing results
        """
        try:
            # Find all priming claims
            priming_claims = await self.claim_repository.search_by_tags(
                ["priming"], 
                limit=10000
            )
            
            # Delete priming claims
            deleted_count = 0
            for claim in priming_claims:
                try:
                    await self.claim_repository.delete(claim.id)
                    deleted_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to delete priming claim {claim.id}: {e}")
            
            # Reset statistics
            self.priming_stats = {
                "total_claims_generated": 0,
                "total_claims_stored": 0,
                "domains_processed": 0,
                "processing_time": 0.0,
                "llm_calls_made": 0,
                "errors": []
            }
            
            self.logger.info(f"Cleared {deleted_count} priming claims from database")
            
            return {
                "success": True,
                "claims_deleted": deleted_count,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to clear priming data: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_priming_domains(self) -> List[str]:
        """
        Get list of available priming domains
        
        Returns:
            List of domain names
        """
        return list(self.priming_domains.keys())
    
    def add_priming_domain(self, domain: str, query: str, description: str) -> bool:
        """
        Add a new priming domain
        
        Args:
            domain: Domain name
            query: Priming query for the domain
            description: Domain description
            
        Returns:
            True if domain added successfully
        """
        try:
            self.priming_domains[domain] = {
                "query": query,
                "description": description
            }
            self.logger.info(f"Added new priming domain: {domain}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add priming domain {domain}: {e}")
            return False


# Global instance for easy access
_dynamic_priming_engine = None


def get_dynamic_priming_engine() -> DynamicPrimingEngine:
    """Get global dynamic priming engine instance"""
    global _dynamic_priming_engine
    if _dynamic_priming_engine is None:
        _dynamic_priming_engine = DynamicPrimingEngine()
    return _dynamic_priming_engine