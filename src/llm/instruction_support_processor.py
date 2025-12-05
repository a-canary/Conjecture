"""
LLM Instruction Support Processor for Simplified Universal Claim Architecture
Handles LLM-driven instruction identification and support relationship creation
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import re

from core.models import Claim, create_claim_index, create_claim
from core.support_relationship_manager import SupportRelationshipManager
from context.complete_context_builder import CompleteContextBuilder, BuiltContext


@dataclass
class InstructionIdentification:
    """Result of instruction identification from LLM"""

    instruction_claims: List[Dict[str, Any]]
    support_relationships: List[Dict[str, Any]]
    confidence_scores: List[float]
    reasoning: str


@dataclass
class ProcessingResult:
    """Result of LLM instruction support processing"""

    success: bool
    new_instruction_claims: List[Claim]
    created_relationships: List[Tuple[str, str]]  # (supporter_id, supported_id)
    processing_time_ms: float
    llm_response: str
    errors: List[str]


class InstructionSupportProcessor:
    """
    Processes claims with LLM to identify instruction claims and create support relationships.

    Key responsibilities:
    - Parse complete context to identify instruction claims
    - Create support relationships between instructions and target claims
    - Format prompts for instruction identification
    - Process LLM responses and extract relationships
    - Validate and persist new relationships
    """

    def __init__(self, claims: List[Claim], llm_client=None):
        """
        Initialize the instruction support processor

        Args:
            claims: List of claims to process
            llm_client: LLM client for instruction identification (optional for testing)
        """
        self.claims = claims
        self.claim_index = create_claim_index(claims)
        self.relationship_manager = SupportRelationshipManager(claims)
        self.context_builder = CompleteContextBuilder(claims)
        self.llm_client = llm_client

        # Processing configuration
        self.instruction_tags = [
            "instruction",
            "guidance",
            "method",
            "approach",
            "technique",
        ]
        self.min_confidence_threshold = 0.6
        self.max_instruction_length = 500

    def process_with_instruction_support(
        self, target_claim_id: str, user_request: str, max_context_tokens: int = 8000
    ) -> ProcessingResult:
        """
        Process a claim with instruction support using LLM.

        Args:
            target_claim_id: The claim to process
            user_request: The user's request/context
            max_context_tokens: Maximum tokens for context building

        Returns:
            ProcessingResult with instruction identification and relationships
        """
        start_time = datetime.utcnow()

        if target_claim_id not in self.claim_index:
            return ProcessingResult(
                success=False,
                new_instruction_claims=[],
                created_relationships=[],
                processing_time_ms=0,
                llm_response="",
                errors=[f"Target claim {target_claim_id} not found"],
            )

        try:
            # Step 1: Build complete context
            built_context = self.context_builder.build_complete_context(
                target_claim_id, max_context_tokens
            )

            # Step 2: Send to LLM for instruction identification
            llm_response = self._send_to_llm_for_instructions(
                built_context.context_text, user_request
            )

            # Step 3: Parse LLM response
            identification = self._parse_llm_response(llm_response)

            # Step 4: Create instruction claims
            new_claims = self._create_instruction_claims(identification)

            # Step 5: Create support relationships
            relationships = self._create_support_relationships(
                target_claim_id, new_claims, identification
            )

            # Step 6: Validate relationships
            validation_errors = self._validate_relationships(relationships)

            # Step 7: Update claims and relationships
            final_claims, final_relationships = self._persist_changes(
                new_claims, relationships, validation_errors
            )

            # Calculate processing time
            end_time = datetime.utcnow()
            processing_time_ms = (end_time - start_time).total_seconds() * 1000

            return ProcessingResult(
                success=not validation_errors,
                new_instruction_claims=final_claims,
                created_relationships=final_relationships,
                processing_time_ms=processing_time_ms,
                llm_response=llm_response,
                errors=validation_errors,
            )

        except Exception as e:
            end_time = datetime.utcnow()
            processing_time_ms = (end_time - start_time).total_seconds() * 1000

            return ProcessingResult(
                success=False,
                new_instruction_claims=[],
                created_relationships=[],
                processing_time_ms=processing_time_ms,
                llm_response="",
                errors=[f"Processing failed: {str(e)}"],
            )

    def _send_to_llm_for_instructions(self, context: str, user_request: str) -> str:
        """Send context to LLM and get instruction identification"""
        if self.llm_client is None:
            # For testing without real LLM, return mock response
            return self._generate_mock_llm_response(context, user_request)

        # Create the prompt for instruction identification
        prompt = self._create_instruction_prompt(context, user_request)

        # Send to LLM (implementation depends on LLM client)
        try:
            response = self.llm_client.generate(prompt=prompt, max_tokens=2000)
            return response
        except Exception as e:
            raise RuntimeError(f"LLM processing failed: {str(e)}")

    def _create_instruction_prompt(self, context: str, user_request: str) -> str:
        """Create a prompt for LLM to identify instruction claims using JSON frontmatter"""
        from ..processing.json_schemas import ResponseSchemaType, create_prompt_template_for_type
        
        prompt_template = create_prompt_template_for_type(ResponseSchemaType.INSTRUCTION_SUPPORT)
        prompt = f"""You are an expert at identifying instructional content and guidance within claim networks.

TASK: Analyze the provided claim context and identify any claims that serve as instructions, guidance, or methodological support for other claims.

CONTEXT:
{context}

USER REQUEST:
{user_request}

{prompt_template}

## YOUR INSTRUCTION IDENTIFICATION TASK:
1. Carefully read through all claims in the context
2. Identify claims that provide:
   - Step-by-step guidance or procedures
   - Methodological instructions or approaches
   - "How to" or "Should do" type content
   - Prescriptive or normative guidance
   - Best practices or techniques
3. For each identified instruction claim:
   - Extract the complete claim content
   - Assess confidence (0.0-1.0) that it's instructional
   - Identify which target claims it supports
   - Include in the instruction_claims array with type "instruction"
4. For each support relationship:
   - Include in the relationships array with instruction_claim_id and target_claim_id
5. Provide a comprehensive analysis_summary

Format your response using the JSON frontmatter format specified above.

After the JSON frontmatter, you can include additional explanation of your reasoning for identifying instructional content.

Example response format:
{{
    "instruction_claims": [
        {{
            "content": "Complete instruction content",
            "confidence": 0.8,
            "supports_target_claim": true,
            "reasoning": "Why this is instructional"
        }}
    ],
    "support_relationships": [
        {{
            "instruction_claim_index": 0,
            "supported_claim_id": "claim-id",
            "relationship_type": "instructional_support"
        }}
    ],
    "overall_reasoning": "Summary of your analysis"
}}

IMPORTANT:
- Only identify claims that are clearly instructional
- Focus on practical guidance and methods
- Maintain high confidence threshold (>0.6)
- Ensure JSON format is valid

Please provide your analysis:
"""
        return prompt

    def _generate_mock_llm_response(self, context: str, user_request: str) -> str:
        """Generate mock LLM response for testing purposes"""
        # Simple heuristic-based instruction detection for testing
        instruction_keywords = [
            "how to",
            "should",
            "must",
            "step",
            "method",
            "approach",
            "technique",
            "procedure",
            "process",
            "guideline",
            "best practice",
        ]

        # Extract target claim ID from context
        target_claim_match = re.search(r"Target Claim: ([^\s\n]+)", context)
        target_claim_id = (
            target_claim_match.group(1) if target_claim_match else "target"
        )

        # Find instruction-like claims in context
        instruction_claims = []
        support_relationships = []
        claim_lines = re.findall(r"\[([^\]]+)\] (.+)", context)

        claim_index = 0
        for claim_id, content in claim_lines:
            content_lower = content.lower()

            # Check if content contains instruction keywords
            if any(keyword in content_lower for keyword in instruction_keywords):
                if len(content) <= self.max_instruction_length:
                    instruction_claims.append(
                        {
                            "content": content.strip(),
                            "confidence": 0.7,
                            "supports_target_claim": True,
                            "reasoning": f"Contains instructional keyword and provides guidance",
                        }
                    )

                    support_relationships.append(
                        {
                            "instruction_claim_index": claim_index,
                            "supported_claim_id": target_claim_id,
                            "relationship_type": "instructional_support",
                        }
                    )

                    claim_index += 1

        return json.dumps(
            {
                "instruction_claims": instruction_claims,
                "support_relationships": support_relationships,
                "overall_reasoning": "Identified instructional claims based on keyword analysis",
            },
            indent=2,
        )

    def _parse_llm_response(self, llm_response: str) -> InstructionIdentification:
        """Parse LLM response to extract instruction claims and relationships"""
        try:
            # Try to parse as JSON
            response_data = json.loads(llm_response)

            instruction_claims = response_data.get("instruction_claims", [])
            support_relationships = response_data.get("support_relationships", [])
            reasoning = response_data.get("overall_reasoning", "")

            # Extract confidence scores
            confidence_scores = [
                claim.get("confidence", 0.0) for claim in instruction_claims
            ]

            return InstructionIdentification(
                instruction_claims=instruction_claims,
                support_relationships=support_relationships,
                confidence_scores=confidence_scores,
                reasoning=reasoning,
            )

        except json.JSONDecodeError:
            # Fallback parsing for non-JSON responses
            return self._parse_text_response(llm_response)
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response: {str(e)}")

    def _parse_text_response(self, response: str) -> InstructionIdentification:
        """Parse non-JSON LLM response as fallback"""
        # Simple text parsing for fallback
        instruction_claims = []
        support_relationships = []
        confidence_scores = []

        # Look for instruction-like statements
        lines = response.split("\n")
        for line in lines:
            if (
                "instruction" in line.lower()
                and len(line.strip()) < self.max_instruction_length
            ):
                instruction_claims.append(
                    {
                        "content": line.strip(),
                        "confidence": 0.6,
                        "supports_target_claim": True,
                        "reasoning": "Identified from text response",
                    }
                )
                confidence_scores.append(0.6)

        return InstructionIdentification(
            instruction_claims=instruction_claims,
            support_relationships=support_relationships,
            confidence_scores=confidence_scores,
            reasoning="Parsed from text response",
        )

    def _create_instruction_claims(
        self, identification: InstructionIdentification
    ) -> List[Claim]:
        """Create Claim objects from identified instructions"""
        new_claims = []

        for i, instruction_data in enumerate(identification.instruction_claims):
            content = instruction_data.get("content", "").strip()
            confidence = instruction_data.get("confidence", 0.0)

            # Validate and filter
            if (
                confidence >= self.min_confidence_threshold
                and len(content) > 10
                and len(content) <= self.max_instruction_length
            ):
                claim = create_claim(
                    content=content,
                    tag="instruction",
                    confidence=confidence,
                    tags=self.instruction_tags,
                )

                new_claims.append(claim)

        return new_claims

    def _create_support_relationships(
        self,
        target_claim_id: str,
        instruction_claims: List[Claim],
        identification: InstructionIdentification,
    ) -> List[Tuple[str, str]]:
        """Create support relationships between instructions and targets"""
        relationships = []

        for relationship_data in identification.support_relationships:
            instruction_index = relationship_data.get("instruction_claim_index", -1)
            supported_claim_id = relationship_data.get("supported_claim_id", "")

            # Validate indices and IDs
            if (
                0 <= instruction_index < len(instruction_claims)
                and supported_claim_id in self.claim_index
            ):
                instruction_claim = instruction_claims[instruction_index]
                relationships.append((instruction_claim.id, supported_claim_id))

        # Auto-add relationships for claims that support the target
        for claim in instruction_claims:
            if claim.id not in [rel[0] for rel in relationships]:
                # Check if this instruction supports the target claim
                relationships.append((claim.id, target_claim_id))

        return relationships

    def _validate_relationships(
        self, relationships: List[Tuple[str, str]]
    ) -> List[str]:
        """Validate support relationships before persisting"""
        errors = []

        for supporter_id, supported_id in relationships:
            # Check if supporter exists (in instruction claims or existing claims)
            if supporter_id not in self.claim_index:
                # This is a new instruction claim, which is expected
                continue

            # Check if supported claim exists
            if supported_id not in self.claim_index:
                errors.append(f"Supported claim {supported_id} not found")
                continue

            # Check for self-referencing
            if supporter_id == supported_id:
                errors.append(f"Self-referencing relationship: {supporter_id}")
                continue

            # Check for existing relationships
            supporter = self.claim_index[supporter_id]
            if supported_id in supporter.supports:
                errors.append(
                    f"Relationship already exists: {supporter_id} -> {supported_id}"
                )

        return errors

    def _persist_changes(
        self,
        new_claims: List[Claim],
        relationships: List[Tuple[str, str]],
        validation_errors: List[str],
    ) -> Tuple[List[Claim], List[Tuple[str, str]]]:
        """Persist new claims and relationships"""
        if validation_errors:
            return [], []

        final_claims = new_claims.copy()
        final_relationships = relationships.copy()

        # Add new claims to the claim index
        for claim in new_claims:
            self.claim_index[claim.id] = claim
            self.claims.append(claim)

        # Add relationships to claims
        for supporter_id, supported_id in relationships:
            # Update supporter claim
            if supporter_id in self.claim_index:
                supporter = self.claim_index[supporter_id]
                if supported_id not in supporter.supports:
                    supporter.supports.append(supported_id)
                    supporter.updated = datetime.utcnow()

            # Update supported claim
            if supported_id in self.claim_index:
                supported = self.claim_index[supported_id]
                if supporter_id not in supported.supported_by:
                    supported.supported_by.append(supporter_id)
                    supported.updated = datetime.utcnow()

        # Refresh relationship manager
        self.relationship_manager.refresh(self.claims)
        self.context_builder.refresh(self.claims)

        return final_claims, final_relationships

    def batch_process_instructions(
        self,
        target_claim_ids: List[str],
        user_request: str,
        max_context_tokens: int = 8000,
    ) -> List[ProcessingResult]:
        """Process multiple claims for instruction support"""
        results = []

        for claim_id in target_claim_ids:
            result = self.process_with_instruction_support(
                claim_id, user_request, max_context_tokens
            )
            results.append(result)

        return results

    def get_instruction_statistics(self) -> Dict[str, Any]:
        """Get statistics about instruction claims in the system"""
        instruction_claims = [
            claim
            for claim in self.claims
            if any(tag in self.instruction_tags for tag in claim.tags)
        ]

        total_instructions = len(instruction_claims)
        avg_confidence = (
            sum(c.confidence for c in instruction_claims) / total_instructions
            if total_instructions > 0
            else 0.0
        )

        # Count instruction relationships
        instruction_relationships = 0
        for claim in instruction_claims:
            instruction_relationships += len(claim.supports)

        return {
            "total_instruction_claims": total_instructions,
            "average_instruction_confidence": round(avg_confidence, 3),
            "total_instruction_relationships": instruction_relationships,
            "instruction_tags": self.instruction_tags,
            "processing_config": {
                "min_confidence_threshold": self.min_confidence_threshold,
                "max_instruction_length": self.max_instruction_length,
            },
        }

    def refresh(self, new_claims: List[Claim]) -> None:
        """Refresh the processor with new claim data"""
        self.claims = new_claims
        self.claim_index = create_claim_index(new_claims)
        self.relationship_manager.refresh(new_claims)
        self.context_builder.refresh(new_claims)
