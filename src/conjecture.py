"""
Conjecture: Simple API Interface
Provides elegant, unified access to all functionality
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import os
import sys
import importlib.util
import glob
from pathlib import Path

from .config.simple_config import Config
from .core.models import Claim, ClaimState, ClaimType
from .processing.bridge import LLMBridge, LLMRequest
from .processing.chutes_adapter import create_chutes_adapter_from_config
from .processing.llm.lm_studio_adapter import create_lm_studio_adapter_from_config


class Conjecture:
    """
    Simple, elegant API for Conjecture
    Single interface for all evidence-based AI reasoning
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize with optional custom configuration"""
        self.config = config or Config()

        # Initialize backend (will be implemented based on database_type)
        self._initialize_backend()

        # Initialize LLM bridge with Chutes.ai
        self._initialize_llm_bridge()

        print(f"Conjecture initialized: {self.config}")

    def _initialize_backend(self):
        """Initialize appropriate backend based on configuration"""
        if self.config.database_type == "chroma":
            print("Using ChromaDB backend")
            # TODO: Initialize ChromaDB backend
        elif self.config.database_type == "file":
            print("Using file-based backend")
            # TODO: Initialize file-based backend
        else:
            print("Using mock backend")
            # TODO: Initialize mock backend

    def _initialize_llm_bridge(self):
        """Initialize LLM bridge with appropriate provider based on configuration"""
        try:
            # Determine which provider to use based on configuration
            llm_provider = (
                self.config.llm_provider.lower()
                if self.config.llm_provider
                else "chutes"
            )

            primary_adapter = None
            fallback_adapter = None

            if llm_provider == "lm_studio":
                # Use LM Studio as primary provider
                primary_adapter = create_lm_studio_adapter_from_config()
                # Use Chutes as fallback if available
                try:
                    fallback_adapter = create_chutes_adapter_from_config()
                except:
                    fallback_adapter = None
            else:
                # Use Chutes as primary provider (default)
                primary_adapter = create_chutes_adapter_from_config()
                # Use LM Studio as fallback if available
                try:
                    fallback_adapter = create_lm_studio_adapter_from_config()
                except:
                    fallback_adapter = None

            # Initialize bridge
            self.llm_bridge = LLMBridge(provider=primary_adapter)
            if fallback_adapter:
                self.llm_bridge.set_fallback(fallback_adapter)

            if self.llm_bridge.is_available():
                print(f"LLM Bridge: {llm_provider} connected (primary)")
            else:
                print(f"LLM Bridge: {llm_provider} not available, using fallback")

        except Exception as e:
            print(f"LLM Bridge initialization failed: {e}")
            self.llm_bridge = LLMBridge()  # Empty bridge for graceful degradation

    def explore(
        self,
        query: str,
        max_claims: int = 10,
        claim_types: Optional[List[str]] = None,
        confidence_threshold: Optional[float] = None,
    ) -> "ExplorationResult":
        """
        Explore knowledge base for claims related to query

        Args:
            query: Your question or topic to explore
            max_claims: Maximum number of claims to return
            claim_types: Specific claim types to include
            confidence_threshold: Minimum confidence level

        Returns:
            ExplorationResult: Comprehensive results with claims and insights
        """
        if not query or len(query.strip()) < 5:
            raise ValueError("Query must be at least 5 characters long")

        max_claims = max(1, min(max_claims, 50))  # Clamp between 1-50
        confidence_threshold = confidence_threshold or self.config.confidence_threshold

        print(f"Exploring: '{query}'")
        print(f"   Max claims: {max_claims}, Confidence: {confidence_threshold}")

        # Try LLM-powered exploration first
        if self.llm_bridge and self.llm_bridge.is_available():
            try:
                return self._explore_with_llm(
                    query, max_claims, claim_types, confidence_threshold
                )
            except Exception as e:
                print(f"⚠️  LLM exploration failed: {e}, falling back to mock")

        # Fallback to mock results
        return self._explore_mock(query, max_claims, claim_types, confidence_threshold)

    def _explore_with_llm(
        self,
        query: str,
        max_claims: int,
        claim_types: Optional[List[str]],
        confidence_threshold: float,
    ) -> "ExplorationResult":
        """Explore using LLM bridge"""
        import time

        start_time = time.time()

        # Build exploration prompt
        prompt = f"""You are an expert knowledge explorer. Analyze the topic "{query}" and generate relevant claims.

Please provide claims in this format:
<claim type="concept" confidence="0.8">A factual claim about {query}</claim>
<claim type="thesis" confidence="0.7">An analytical insight about {query}</claim>
<claim type="example" confidence="0.6">A specific example related to {query}</claim>

Focus on:
- Accuracy and factual correctness
- Appropriate confidence scores (0.0-1.0)
- Clear, concise claims
- Different claim types for comprehensive coverage

Generate up to {max_claims} claims with confidence ≥ {confidence_threshold}."""

        # Create LLM request
        llm_request = LLMRequest(
            prompt=prompt, max_tokens=2048, temperature=0.7, task_type="explore"
        )

        # Process with LLM
        response = self.llm_bridge.process(llm_request)

        # Filter and format results
        processing_time = time.time() - start_time

        if response.success:
            # Filter by confidence threshold and claim types
            filtered_claims = []
            for claim in response.generated_claims:
                if claim.confidence >= confidence_threshold:
                    if claim_types is None or any(
                        t.value in claim_types for t in claim.type
                    ):
                        filtered_claims.append(claim)

            # Limit to max_claims
            final_claims = filtered_claims[:max_claims]

            result = ExplorationResult(
                query=query,
                claims=final_claims,
                total_found=len(filtered_claims),
                search_time=processing_time,
                confidence_threshold=confidence_threshold,
                max_claims=max_claims,
            )

            print(
                f"Found {len(result.claims)} claims via LLM in {result.search_time:.2f}s"
            )
            return result
        else:
            raise Exception(f"LLM processing failed: {response.errors}")

    def _explore_mock(
        self,
        query: str,
        max_claims: int,
        claim_types: Optional[List[str]],
        confidence_threshold: float,
    ) -> "ExplorationResult":
        """Fallback mock exploration"""
        mock_claims = self._generate_mock_claims(query, max_claims, claim_types)
        filtered_claims = [
            c for c in mock_claims if c.confidence >= confidence_threshold
        ]

        result = ExplorationResult(
            query=query,
            claims=filtered_claims[:max_claims],
            total_found=len(filtered_claims),
            search_time=0.1,  # Mock timing
            confidence_threshold=confidence_threshold,
            max_claims=max_claims,
        )

        print(f"Found {len(result.claims)} claims (mock) in {result.search_time:.2f}s")
        return result

    def add_claim(
        self,
        content: str,
        confidence: float,
        claim_type: str,
        tags: Optional[List[str]] = None,
        validate_with_llm: bool = True,
        **kwargs,
    ) -> Claim:
        """
        Add a new claim to the knowledge base

        Args:
            content: Claim content (minimum 10 characters)
            confidence: Confidence score (0.0-1.0)
            claim_type: Type of claim ("concept", "reference", etc.)
            tags: Optional topic tags
            validate_with_llm: Whether to validate claim with LLM
            **kwargs: Additional claim attributes

        Returns:
            Claim: The created claim with generated ID
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

        # Create initial claim
        claim = Claim(
            id=f"claim_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(content)}",
            content=content.strip(),
            confidence=confidence,
            type=[claim_type_enum],
            tags=tags or [],
            **kwargs,
        )

        # Validate with LLM if requested and available
        if validate_with_llm and self.llm_bridge and self.llm_bridge.is_available():
            try:
                validated_claim = self._validate_claim_with_llm(claim)
                print(f"Created and validated claim: {validated_claim}")
                return validated_claim
            except Exception as e:
                print(f"⚠️  LLM validation failed: {e}, using original claim")

        print(f"Created claim: {claim}")

        # TODO: Store claim in backend
        return claim

    def _validate_claim_with_llm(self, claim: Claim) -> Claim:
        """Validate claim using LLM bridge"""
        prompt = f"""You are an expert fact-checker. Evaluate this claim for accuracy and provide an appropriate confidence score.

Claim: "{claim.content}"
Original confidence: {claim.confidence}
Claim type: {claim.type[0].value if claim.type else "unknown"}

Please analyze:
1. Factual accuracy
2. Completeness of the claim
3. Any missing context or qualifications
4. Appropriate confidence score (0.0-1.0)

Respond with:
- VALIDATED: [True/False] 
- CONFIDENCE: [0.0-1.0]
- REASONING: [Brief explanation]
- SUGGESTED_EDIT: [Improved claim text if needed, otherwise "NO_CHANGE"]"""

        # Create LLM request
        llm_request = LLMRequest(
            prompt=prompt,
            context_claims=[claim],
            max_tokens=500,
            temperature=0.3,
            task_type="validate",
        )

        # Process with LLM
        response = self.llm_bridge.process(llm_request)

        if response.success and response.content:
            # Parse validation response
            validated_claim = self._parse_validation_response(claim, response.content)
            return validated_claim
        else:
            raise Exception(f"LLM validation failed: {response.errors}")

    def _parse_validation_response(
        self, original_claim: Claim, validation_text: str
    ) -> Claim:
        """Parse LLM validation response and update claim"""
        try:
            # Simple parsing of validation response
            lines = validation_text.strip().split("\n")
            updates = {}

            for line in lines:
                if line.startswith("VALIDATED:"):
                    updates["validated"] = "True" in line
                elif line.startswith("CONFIDENCE:"):
                    try:
                        updates["confidence"] = float(line.split(":")[1].strip())
                    except:
                        pass
                elif line.startswith("SUGGESTED_EDIT:"):
                    edit = line.split(":", 1)[1].strip()
                    if edit != "NO_CHANGE":
                        updates["content"] = edit

            # Create updated claim
            updated_claim = Claim(
                id=original_claim.id,
                content=updates.get("content", original_claim.content),
                confidence=updates.get("confidence", original_claim.confidence),
                type=original_claim.type,
                state=ClaimState.VALIDATED
                if updates.get("validated", False)
                else ClaimState.EXPLORE,
                tags=original_claim.tags,
                created=original_claim.created,
                updated=datetime.utcnow(),
            )

            return updated_claim

        except Exception as e:
            print(f"Error parsing validation response: {e}")
            return original_claim

    def _generate_mock_claims(
        self, query: str, max_claims: int, claim_types: Optional[List[str]]
    ) -> List[Claim]:
        """Generate mock claims for demonstration"""
        mock_claims = [
            Claim(
                id="mock_001",
                content=f"{query} requires understanding of fundamental concepts and principles",
                confidence=0.85,
                type=[ClaimType.CONCEPT],
                tags=[
                    "fundamental",
                    query.lower().split()[0] if query.split() else "topic",
                ],
            ),
            Claim(
                id="mock_002",
                content=f"Research on {query} shows significant progress in recent years",
                confidence=0.92,
                type=[ClaimType.REFERENCE],
                tags=["research", "progress"],
            ),
            Claim(
                id="mock_003",
                content=f"Mastering {query} involves developing specific skills and competencies",
                confidence=0.78,
                type=[ClaimType.CONCEPT],
                tags=["mastery", "skills"],
            ),
            Claim(
                id="mock_004",
                content=f"The goal of studying {query} is to achieve comprehensive understanding",
                confidence=0.88,
                type=[ClaimType.GOAL],
                tags=["understanding", "comprehensive"],
            ),
            Claim(
                id="mock_005",
                content=f"An example of {query} can be found in practical applications",
                confidence=0.75,
                type=[ClaimType.EXAMPLE],
                tags=["example", "practical"],
            ),
            Claim(
                id="mock_006",
                content=f"The thesis regarding {query} suggests multiple perspectives exist",
                confidence=0.82,
                type=[ClaimType.THESIS],
                tags=["thesis", "perspectives"],
            ),
        ]

        # Filter by claim types if specified
        if claim_types:
            filtered_types = [t.lower() for t in claim_types]
            mock_claims = [
                c for c in mock_claims if any(t.value in filtered_types for t in c.type)
            ]

        return mock_claims[:max_claims]

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics and health metrics"""
        return {
            "config": self.config.to_dict(),
            "database_type": self.config.database_type,
            "database_path": self.config.database_path,
            "llm_enabled": self.config.llm_enabled,
            "claims_count": 0,  # TODO: Get actual count
            "tools_count": len(self.tools),
            "skills_count": len(self.skills),
            "system_healthy": True,
            "uptime": "N/A",  # TODO: Track actual uptime
        }

    def _load_tools(self):
        """Load all tools from the tools/ directory"""
        tools_dir = Path("tools")
        if not tools_dir.exists():
            print("Tools directory not found")
            return

        # Find all Python files in tools directory
        tool_files = list(tools_dir.glob("*.py"))

        for tool_file in tool_files:
            if tool_file.name == "__init__.py":
                continue

            try:
                # Load the module
                spec = importlib.util.spec_from_file_location(tool_file.stem, tool_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find all functions in the module
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if callable(attr) and not attr_name.startswith("_"):
                        self.tools[attr_name] = attr

                print(f"✅ Loaded tool: {tool_file.name}")

            except Exception as e:
                print(f"❌ Failed to load tool {tool_file.name}: {e}")

    def _load_skills(self):
        """Load all skill claims from the skills/ directory"""
        skills_dir = Path("skills")
        if not skills_dir.exists():
            print("Skills directory not found")
            return

        # Find all Python files in skills directory
        skill_files = list(skills_dir.glob("*.py"))

        for skill_file in skill_files:
            if skill_file.name == "__init__.py":
                continue

            try:
                # Load the module
                spec = importlib.util.spec_from_file_location(
                    skill_file.stem, skill_file
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Look for create_*_skills functions
                for attr_name in dir(module):
                    if attr_name.startswith("create_") and attr_name.endswith(
                        "_skills"
                    ):
                        create_func = getattr(module, attr_name)
                        if callable(create_func):
                            skills = create_func()
                            self.skills.extend(skills)
                            print(
                                f"✅ Loaded {len(skills)} skills from {skill_file.name}"
                            )
                            break

            except Exception as e:
                print(f"❌ Failed to load skills from {skill_file.name}: {e}")

    def get_tool_examples(self) -> List[str]:
        """Get examples from all loaded tools"""
        examples = []
        for tool_name, tool_func in self.tools.items():
            if tool_name == "examples":
                continue  # Skip the examples function itself
            try:
                # Look for examples function in the tool's module
                module_name = tool_func.__module__
                if module_name:
                    module = sys.modules.get(module_name)
                    if module and hasattr(module, "examples"):
                        tool_examples = module.examples()
                        examples.extend(tool_examples)
            except:
                pass
        return examples

    def get_relevant_context(self, query: str) -> List[Claim]:
        """Get relevant skills and tool examples for a query"""
        relevant_claims = []

        # Add skills that match query tags or content
        query_lower = query.lower()
        for skill in self.skills:
            if any(tag.lower() in query_lower for tag in skill.tags) or any(
                keyword in skill.content.lower() for keyword in query_lower.split()
            ):
                relevant_claims.append(skill)

        # Add tool examples as claims
        examples = self.get_tool_examples()
        for example in examples:
            if any(keyword in example.lower() for keyword in query_lower.split()):
                claim = Claim(
                    id=f"tool_example_{len(relevant_claims)}",
                    content=example,
                    confidence=0.8,
                    type=[ClaimType.EXAMPLE],
                    tags=["tool", "example"],
                )
                relevant_claims.append(claim)

        return relevant_claims


class ExplorationResult:
    """
    Result of an exploration query
    Provides comprehensive results with claims and insights
    """

    def __init__(
        self,
        query: str,
        claims: List[Claim],
        total_found: int,
        search_time: float,
        confidence_threshold: float,
        max_claims: int,
    ):
        self.query = query
        self.claims = claims
        self.total_found = total_found
        self.search_time = search_time
        self.confidence_threshold = confidence_threshold
        self.max_claims = max_claims
        self.timestamp = datetime.utcnow()

    def __str__(self) -> str:
        return f"ExplorationResult(query='{self.query}', claims={len(self.claims)}, time={self.search_time:.2f}s)"

    def summary(self) -> str:
        """Provide a human-readable summary of results"""
        if not self.claims:
            return f"No claims found for '{self.query}' above confidence threshold {self.confidence_threshold}"

        lines = [
            f"🎯 Explored: '{self.query}'",
            f"📊 Found: {len(self.claims)} claims (of {self.total_found} total)",
            f"⏱️  Time: {self.search_time:.2f}s",
            f"🎚️  Confidence: ≥{self.confidence_threshold}",
            "",
            "📋 Top Claims:",
        ]

        for i, claim in enumerate(self.claims[:5], 1):  # Show top 5
            type_str = ",".join([t.value for t in claim.type])
            lines.append(
                f"  {i}. [{claim.confidence:.2f}, {type_str}] {claim.content[:100]}{'...' if len(claim.content) > 100 else ''}"
            )

        if len(self.claims) > 5:
            lines.append(f"  ... and {len(self.claims) - 5} more claims")

        return "\n".join(lines)


# Convenience functions for immediate use
def explore(query: str, max_claims: int = 10, **kwargs) -> ExplorationResult:
    """
    Quick exploration function - no setup required
    """
    cf = Conjecture()
    return cf.explore(query, max_claims, **kwargs)


def add_claim(content: str, confidence: float, claim_type: str, **kwargs) -> Claim:
    """
    Quick add claim function - no setup required
    """
    cf = Conjecture()
    return cf.add_claim(content, confidence, claim_type, **kwargs)


if __name__ == "__main__":
    print("🧪 Testing Conjecture API")
    print("=" * 30)

    # Test initialization
    cf = Conjecture()
    print(f"✅ Conjecture initialized: {cf.config}")

    # Test exploration
    print("\n🔍 Testing exploration...")
    result = cf.explore("machine learning", max_claims=3)
    print(result.summary())

    # Test adding claim
    print("\n➕ Testing claim creation...")
    claim = cf.add_claim(
        content="Machine learning algorithms require substantial training data to achieve optimal performance",
        confidence=0.87,
        claim_type="concept",
        tags=["machine learning", "algorithms", "performance"],
    )
    print(f"✅ Created: {claim}")

    # Test statistics
    print("\n📊 Testing statistics...")
    stats = cf.get_statistics()
    print(f"System stats: {stats}")

    print("\n🎉 Conjecture API test completed!")
