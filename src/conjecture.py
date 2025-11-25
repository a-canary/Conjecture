"""
Conjecture: Async Evidence-Based AI Reasoning System
Provides elegant, unified access to all functionality with async support
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging

from src.core.models import Claim, ClaimState, ClaimType
from src.config.config import Config
from src.processing.bridge import LLMBridge, LLMRequest
from src.processing.llm.provider import create_provider
from src.processing.async_eval import AsyncClaimEvaluationService
from src.processing.context_collector import ContextCollector
from src.processing.tool_manager import DynamicToolCreator
from src.data.repositories import get_data_manager, RepositoryFactory


class Conjecture:
    """
    Enhanced Conjecture with Async Claim Evaluation and Dynamic Tool Creation
    Implements the full architecture described in the specifications
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize Enhanced Conjecture with all components"""
        self.config = config or Config()

        # Initialize data layer with repository pattern
        self.data_manager = get_data_manager(use_mock_embeddings=False)
        self.claim_repository = RepositoryFactory.create_claim_repository(
            self.data_manager
        )

        # Initialize LLM bridge
        self._initialize_llm_bridge()

        # Initialize processing components
        self.context_collector = ContextCollector(self.data_manager)
        self.async_evaluation = AsyncClaimEvaluationService(
            llm_bridge=self.llm_bridge, context_collector=self.context_collector
        )
        self.tool_creator = DynamicToolCreator(
            llm_bridge=self.llm_bridge, tools_dir="tools"
        )

        # Service state
        self._services_started = False

        # Statistics
        self._stats = {
            "claims_processed": 0,
            "tools_created": 0,
            "evaluation_time_total": 0.0,
            "session_count": 0,
        }

        self.logger = logging.getLogger(__name__)

        print(f"Enhanced Conjecture initialized with config: {self.config}")

    async def start_services(self):
        """Start background services"""
        if self._services_started:
            return

        await self.async_evaluation.start()
        self._services_started = True

        self.logger.info("Enhanced Conjecture services started")

    async def stop_services(self):
        """Stop background services"""
        if not self._services_started:
            return

        await self.async_evaluation.stop()
        self._services_started = False

        self.logger.info("Enhanced Conjecture services stopped")

    def _initialize_llm_bridge(self):
        """Initialize LLM bridge with simple unified provider"""
        try:
            provider = create_provider(self.config)
            self.llm_bridge = LLMBridge(provider=provider)

            if self.llm_bridge.is_available():
                print(f"LLM Bridge: {self.config.llm_provider or 'chutes'} connected")
            else:
                print("LLM Bridge: No providers available, using mock mode")

        except Exception as e:
            print(f"LLM Bridge initialization failed: {e}")
            self.llm_bridge = LLMBridge()

    async def explore(
        self,
        query: str,
        max_claims: int = 10,
        claim_types: Optional[List[str]] = None,
        confidence_threshold: Optional[float] = None,
        auto_evaluate: bool = True,
    ) -> "ExplorationResult":
        """
        Enhanced exploration with automatic claim evaluation

        Args:
            query: Research question or topic
            max_claims: Maximum claims to return
            claim_types: Specific claim types to include
            confidence_threshold: Minimum confidence level
            auto_evaluate: Whether to automatically evaluate created claims

        Returns:
            ExplorationResult with claims and evaluation status
        """
        start_time = time.time()

        if not query or len(query.strip()) < 5:
            raise ValueError("Query must be at least 5 characters long")

        confidence_threshold = confidence_threshold or self.config.confidence_threshold

        print(f"Enhanced exploration: '{query}'")

        try:
            # Start services if not already running
            if not self._services_started:
                await self.start_services()

            # Generate initial claims using LLM
            initial_claims = await self._generate_initial_claims(query, max_claims)

            # Filter by confidence threshold
            filtered_claims = [
                claim
                for claim in initial_claims
                if claim.confidence >= confidence_threshold
            ]

            # Store claims using repository pattern
            stored_claims = []
            for claim in filtered_claims:
                claim_data = {
                    "content": claim.content,
                    "confidence": claim.confidence,
                    "claim_type": claim.type[0].value,
                    "tags": claim.tags,
                    "state": ClaimState.EXPLORE,
                }
                stored_claim = await self.claim_repository.create(claim_data)
                stored_claims.append(stored_claim)

                # Submit for evaluation if enabled
                if auto_evaluate:
                    await self.async_evaluation.submit_claim(stored_claim)

            # Check for tool creation opportunities
            await self._check_tool_needs(stored_claims)

            processing_time = time.time() - start_time
            self._update_stats(processing_time, len(stored_claims))

            result = ExplorationResult(
                query=query,
                claims=stored_claims,
                total_found=len(filtered_claims),
                search_time=processing_time,
                confidence_threshold=confidence_threshold,
                max_claims=max_claims,
                evaluation_pending=auto_evaluate,
                tools_created=len(self.tool_creator.get_created_tools()),
            )

            print(
                f"Enhanced exploration completed: {len(result.claims)} claims in {result.search_time:.2f}s"
            )
            return result

        except Exception as e:
            self.logger.error(f"Error in enhanced exploration: {e}")
            raise

    async def _generate_initial_claims(
        self, query: str, max_claims: int
    ) -> List[Claim]:
        """Generate initial claims using LLM with context awareness"""
        try:
            # Get relevant context for query
            context_claims = await self.context_collector.collect_context_for_claim(
                query, {"task": "exploration"}, max_skills=3, max_samples=5
            )

            # Build context string
            context_string = ""
            if context_claims.get("skills"):
                context_string += "RELEVANT SKILLS:\n"
                for skill in context_claims["skills"]:
                    context_string += f"- {skill['context_format']}\n"
                context_string += "\n"

            if context_claims.get("samples"):
                context_string += "RELEVANT EXAMPLES:\n"
                for sample in context_claims["samples"]:
                    context_string += f"- {sample['context_format']}\n"
                context_string += "\n"

            # Build exploration prompt
            prompt = f"""Research and analyze the topic: "{query}"

{context_string}

Generate comprehensive claims about this topic. Focus on:
1. Factual accuracy and verifiable information
2. Key concepts and definitions
3. Important relationships and dependencies
4. Practical applications and examples
5. Current state and future directions

For each claim, provide:
- Clear, specific statement
- Confidence score (0.0-1.0) based on certainty
- Appropriate claim type (concept, reference, thesis, example, goal)
- Relevant tags for categorization

Generate up to {max_claims} high-quality claims."""

            llm_request = LLMRequest(
                prompt=prompt, max_tokens=3000, temperature=0.7, task_type="explore"
            )

            response = self.llm_bridge.process(llm_request)

            if response.success:
                return self._parse_claims_from_response(response.content)
            else:
                raise Exception(f"LLM processing failed: {response.errors}")

        except Exception as e:
            self.logger.error(f"Error generating initial claims: {e}")
            return []

    def _parse_claims_from_response(self, response: str) -> List[Claim]:
        """Parse claims from LLM response"""
        claims = []

        try:
            # Simple parsing - look for claim patterns
            import re

            # Pattern for claims with confidence and type
            claim_pattern = (
                r'Claim:\s*"([^"]+)"\s*Confidence:\s*([\d.]+)\s*Type:\s*(\w+)'
            )
            matches = re.findall(claim_pattern, response, re.IGNORECASE)

            for i, (content, confidence, claim_type) in enumerate(matches):
                try:
                    claim = Claim(
                        id=f"exploration_{int(time.time())}_{i}",
                        content=content.strip(),
                        confidence=float(confidence),
                        type=[ClaimType(claim_type.lower())],
                        tags=["exploration", "auto_generated"],
                        state=ClaimState.EXPLORE,
                    )
                    claims.append(claim)
                except (ValueError, KeyError) as e:
                    self.logger.warning(f"Failed to parse claim: {e}")
                    continue

            # If no structured claims found, try simpler parsing
            if not claims:
                lines = response.split("\n")
                for i, line in enumerate(lines):
                    if line.strip() and len(line.strip()) > 20:
                        claim = Claim(
                            id=f"exploration_{int(time.time())}_{i}",
                            content=line.strip(),
                            confidence=0.7,  # Default confidence
                            type=[ClaimType.CONCEPT],
                            tags=["exploration", "auto_generated"],
                            state=ClaimState.EXPLORE,
                        )
                        claims.append(claim)

        except Exception as e:
            self.logger.error(f"Error parsing claims from response: {e}")

        return claims[:10]  # Limit to 10 claims

    async def _check_tool_needs(self, claims: List[Claim]):
        """Check if any claims indicate need for new tools"""
        for claim in claims:
            try:
                tool_need = await self.tool_creator.discover_tool_need(claim)
                if tool_need:
                    print(f"Tool need detected: {tool_need[:100]}...")

                    # Search for implementation methods
                    methods = await self.tool_creator.websearch_tool_methods(tool_need)

                    if methods:
                        # Create tool
                        tool_name = f"tool_{claim.id[:8]}"
                        tool_path = await self.tool_creator.create_tool_file(
                            tool_name, tool_need, methods
                        )

                        if tool_path:
                            # Create skill and sample claims
                            skill_claim = await self.tool_creator.create_skill_claim(
                                tool_name, tool_need, tool_path
                            )
                            sample_claim = await self.tool_creator.create_sample_claim(
                                tool_name, tool_path
                            )

                            # Store skill and sample claims
                            await self.data_manager.create_claim(
                                content=skill_claim.content,
                                confidence=skill_claim.confidence,
                                claim_type=skill_claim.type[0].value,
                                tags=skill_claim.tags,
                                state=skill_claim.state,
                            )

                            await self.data_manager.create_claim(
                                content=sample_claim.content,
                                confidence=sample_claim.confidence,
                                claim_type=sample_claim.type[0].value,
                                tags=sample_claim.tags,
                                state=sample_claim.state,
                            )

                            self._stats["tools_created"] += 1
                            print(f"Created new tool: {tool_name}")

            except Exception as e:
                self.logger.error(
                    f"Error checking tool needs for claim {claim.id}: {e}"
                )

    async def add_claim(
        self,
        content: str,
        confidence: float,
        claim_type: str,
        tags: Optional[List[str]] = None,
        auto_evaluate: bool = True,
        **kwargs,
    ) -> Claim:
        """
        Enhanced claim creation with automatic evaluation

        Args:
            content: Claim content
            confidence: Initial confidence score
            claim_type: Type of claim
            tags: Optional tags
            auto_evaluate: Whether to submit for evaluation
            **kwargs: Additional claim attributes

        Returns:
            Created claim
        """
        # Validate inputs
        if len(content.strip()) < 10:
            raise ValueError("Content must be at least 10 characters long")
        if not (0.0 <= confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")

        try:
            claim_type_enum = ClaimType(claim_type.lower())
        except ValueError:
            valid_types = [t.value for t in ClaimType]
            raise ValueError(
                f"Invalid claim type: {claim_type}. Valid types: {valid_types}"
            )

        # Create claim using repository
        claim_data = {
            "content": content.strip(),
            "confidence": confidence,
            "claim_type": claim_type_enum.value,
            "tags": tags or [],
            "state": ClaimState.EXPLORE,
            **kwargs,
        }
        claim = await self.claim_repository.create(claim_data)

        # Submit for evaluation if enabled
        if auto_evaluate and self._services_started:
            await self.async_evaluation.submit_claim(claim)

        self._stats["claims_processed"] += 1
        print(f"Created claim: {claim.id}")

        return claim

    async def get_evaluation_status(self, claim_id: str) -> Dict[str, Any]:
        """Get evaluation status for a specific claim"""
        try:
            claim = await self.claim_repository.get_by_id(claim_id)
            if not claim:
                return {"error": "Claim not found"}

            return {
                "claim_id": claim_id,
                "state": claim.state.value,
                "confidence": claim.confidence,
                "last_updated": claim.updated.isoformat(),
                "evaluation_active": claim_id
                in self.async_evaluation._active_evaluations,
            }
        except Exception as e:
            return {"error": str(e)}

    async def wait_for_evaluation(
        self, claim_id: str, timeout: int = 60
    ) -> Dict[str, Any]:
        """Wait for claim evaluation to complete"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = await self.get_evaluation_status(claim_id)

            if "error" in status:
                return status

            if status["state"] == ClaimState.VALIDATED.value:
                return {"success": True, "claim": status}

            if not status["evaluation_active"]:
                return {
                    "success": False,
                    "reason": "Evaluation not active",
                    "claim": status,
                }

            await asyncio.sleep(2)

        return {"success": False, "reason": "Timeout", "claim": status}

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        base_stats = {
            "config": self.config.to_dict(),
            "services_running": self._services_started,
            "claims_processed": self._stats["claims_processed"],
            "tools_created": self._stats["tools_created"],
            "total_evaluation_time": self._stats["evaluation_time_total"],
            "session_count": self._stats["session_count"],
        }

        # Add async evaluation stats
        if self._services_started:
            eval_stats = self.async_evaluation.get_statistics()
            base_stats.update({"evaluation_service": eval_stats})

        # Add tool creator stats
        tool_stats = self.tool_creator.get_created_tools()
        base_stats.update({"created_tools": tool_stats})

        return base_stats

    def _update_stats(self, processing_time: float, claims_count: int):
        """Update internal statistics"""
        self._stats["evaluation_time_total"] += processing_time
        self._stats["claims_processed"] += claims_count

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start_services()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop_services()


class ExplorationResult:
    """Enhanced exploration result with evaluation information"""

    def __init__(
        self,
        query: str,
        claims: List[Claim],
        total_found: int,
        search_time: float,
        confidence_threshold: float,
        max_claims: int,
        evaluation_pending: bool = False,
        tools_created: int = 0,
    ):
        self.query = query
        self.claims = claims
        self.total_found = total_found
        self.search_time = search_time
        self.confidence_threshold = confidence_threshold
        self.max_claims = max_claims
        self.evaluation_pending = evaluation_pending
        self.tools_created = tools_created
        self.timestamp = datetime.utcnow()

    def summary(self) -> str:
        """Provide enhanced summary"""
        if not self.claims:
            return f"No claims found for '{self.query}' above confidence threshold {self.confidence_threshold}"

        lines = [
            f"🎯 Enhanced Exploration: '{self.query}'",
            f"📊 Found: {len(self.claims)} claims (of {self.total_found} total)",
            f"⏱️  Time: {self.search_time:.2f}s",
            f"🎚️  Confidence: ≥{self.confidence_threshold}",
            f"🔄 Evaluation: {'Pending' if self.evaluation_pending else 'Disabled'}",
            f"🔧 Tools Created: {self.tools_created}",
            "",
            "📋 Top Claims:",
        ]

        for i, claim in enumerate(self.claims[:5], 1):
            type_str = claim.type[0].value if claim.type else "unknown"
            lines.append(
                f"  {i}. [{claim.confidence:.2f}, {type_str}, {claim.state.value}] {claim.content[:100]}{'...' if len(claim.content) > 100 else ''}"
            )

        if len(self.claims) > 5:
            lines.append(f"  ... and {len(self.claims) - 5} more claims")

        return "\n".join(lines)


# Convenience functions
async def explore(query: str, **kwargs) -> ExplorationResult:
    """Quick exploration function"""
    async with Conjecture() as cf:
        return await cf.explore(query, **kwargs)


async def add_claim(
    content: str, confidence: float, claim_type: str, **kwargs
) -> Claim:
    """Quick claim creation function"""
    async with Conjecture() as cf:
        return await cf.add_claim(content, confidence, claim_type, **kwargs)


if __name__ == "__main__":

    async def test_enhanced_conjecture():
        print("🧪 Testing Enhanced Conjecture")
        print("=" * 40)

        async with Conjecture() as cf:
            # Test enhanced exploration
            print("\n🔍 Testing enhanced exploration...")
            result = await cf.explore("quantum computing applications", max_claims=3)
            print(result.summary())

            # Test claim creation
            print("\n➕ Testing enhanced claim creation...")
            claim = await cf.add_claim(
                content="Enhanced Conjecture provides async evaluation and dynamic tool creation",
                confidence=0.9,
                claim_type="concept",
                tags=["architecture", "enhancement"],
            )
            print(f"✅ Created claim: {claim.id}")

            # Wait for evaluation
            print("\n⏳ Waiting for evaluation...")
            eval_result = await cf.wait_for_evaluation(claim.id, timeout=10)
            print(f"Evaluation result: {eval_result}")

            # Test statistics
            print("\n📊 Testing enhanced statistics...")
            stats = cf.get_statistics()
            print(f"Enhanced stats: {stats}")

        print("\n🎉 Enhanced Conjecture tests completed!")

    asyncio.run(test_enhanced_conjecture())
