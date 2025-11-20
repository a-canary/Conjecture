"""
Example Generator for the Conjecture skill-based agency system.
Automatically generates example claims from successful skill executions.
"""

import asyncio
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import re

from ..core.models import Claim, ExecutionResult
from ..data.data_manager import DataManager
from ..data.models import ClaimNotFoundError


logger = logging.getLogger(__name__)


class ExampleQualityAssessor:
    """Assesses the quality of generated examples."""

    def __init__(self):
        self.quality_factors = {
            "execution_success": 0.4,  # Was the execution successful?
            "execution_time": 0.2,  # Was execution time reasonable?
            "output_complexity": 0.2,  # Is the output interesting/complex?
            "parameter_diversity": 0.1,  # Are parameters diverse?
            "result_uniqueness": 0.1,  # Is this result unique?
        }

    def assess_example_quality(
        self, execution_result: ExecutionResult, existing_examples: List[Claim]
    ) -> float:
        """
        Assess the quality of a potential example.

        Args:
            execution_result: Result from skill execution
            existing_examples: Existing examples for this skill

        Returns:
            Quality score between 0.0 and 1.0
        """
        scores = {}

        # Execution success
        scores["execution_success"] = 1.0 if execution_result.success else 0.0

        # Execution time (prefer faster execution)
        if execution_result.execution_time_ms < 100:
            scores["execution_time"] = 1.0
        elif execution_result.execution_time_ms < 1000:
            scores["execution_time"] = 0.8
        elif execution_result.execution_time_ms < 5000:
            scores["execution_time"] = 0.6
        else:
            scores["execution_time"] = 0.4

        # Output complexity (prefer more complex/interesting outputs)
        scores["output_complexity"] = self._assess_output_complexity(
            execution_result.result
        )

        # Parameter diversity (prefer diverse parameter combinations)
        scores["parameter_diversity"] = self._assess_parameter_diversity(
            execution_result.parameters_used, existing_examples
        )

        # Result uniqueness (prefer unique results)
        scores["result_uniqueness"] = self._assess_result_uniqueness(
            execution_result.result, existing_examples
        )

        # Calculate weighted score
        total_score = sum(
            scores[factor] * weight for factor, weight in self.quality_factors.items()
        )

        return min(1.0, max(0.0, total_score))

    def _assess_output_complexity(self, result: Any) -> float:
        """Assess the complexity of the execution result."""
        if result is None:
            return 0.1

        # Check result type and size
        if isinstance(result, str):
            if len(result) < 10:
                return 0.2
            elif len(result) < 100:
                return 0.5
            elif len(result) < 1000:
                return 0.8
            else:
                return 0.6  # Too long might be less useful

        elif isinstance(result, (int, float)):
            return 0.3

        elif isinstance(result, bool):
            return 0.2

        elif isinstance(result, list):
            if len(result) == 0:
                return 0.1
            elif len(result) < 5:
                return 0.6
            elif len(result) < 20:
                return 0.9
            else:
                return 0.7

        elif isinstance(result, dict):
            if len(result) == 0:
                return 0.1
            elif len(result) < 3:
                return 0.7
            elif len(result) < 10:
                return 0.9
            else:
                return 0.8

        else:
            return 0.5  # Unknown type

    def _assess_parameter_diversity(
        self, parameters: Dict[str, Any], existing_examples: List[Claim]
    ) -> float:
        """Assess how diverse the parameters are compared to existing examples."""
        if not existing_examples:
            return 1.0  # First example is always diverse

        # Create parameter signature
        param_signature = self._create_parameter_signature(parameters)

        # Check against existing examples
        similar_count = 0
        for example in existing_examples:
            existing_signature = self._create_parameter_signature(
                example.input_parameters
            )
            if self._signatures_similar(param_signature, existing_signature):
                similar_count += 1

        # Calculate diversity score
        if similar_count == 0:
            return 1.0
        elif similar_count < len(existing_examples) * 0.3:
            return 0.8
        elif similar_count < len(existing_examples) * 0.6:
            return 0.5
        else:
            return 0.2

    def _assess_result_uniqueness(
        self, result: Any, existing_examples: List[Claim]
    ) -> float:
        """Assess how unique the result is compared to existing examples."""
        if not existing_examples:
            return 1.0

        # Create result signature
        result_signature = self._create_result_signature(result)

        # Check against existing examples
        similar_count = 0
        for example in existing_examples:
            if example.output_result is not None:
                existing_signature = self._create_result_signature(
                    example.output_result
                )
                if result_signature == existing_signature:
                    similar_count += 1

        # Calculate uniqueness score
        if similar_count == 0:
            return 1.0
        elif similar_count < len(existing_examples) * 0.2:
            return 0.8
        elif similar_count < len(existing_examples) * 0.5:
            return 0.5
        else:
            return 0.2

    def _create_parameter_signature(self, parameters: Dict[str, Any]) -> str:
        """Create a signature for parameters to compare similarity."""
        # Sort keys for consistent signature
        sorted_params = {}
        for key in sorted(parameters.keys()):
            value = parameters[key]
            # Simplify value for signature
            if isinstance(value, (list, dict)):
                value = type(value).__name__ + str(len(value))
            elif isinstance(value, str):
                value = "str" + str(len(value))
            else:
                value = type(value).__name__
            sorted_params[key] = value

        return json.dumps(sorted_params, sort_keys=True)

    def _create_result_signature(self, result: Any) -> str:
        """Create a signature for result to compare similarity."""
        if result is None:
            return "null"
        elif isinstance(result, bool):
            return "bool:" + str(result)
        elif isinstance(result, (int, float)):
            return "number:" + str(result)
        elif isinstance(result, str):
            return "str:" + hashlib.md5(result.encode()).hexdigest()[:8]
        elif isinstance(result, list):
            return "list:" + str(len(result))
        elif isinstance(result, dict):
            return (
                "dict:"
                + str(len(result))
                + ":"
                + hashlib.md5(json.dumps(result, sort_keys=True).encode()).hexdigest()[
                    :8
                ]
            )
        else:
            return "unknown:" + type(result).__name__

    def _signatures_similar(self, sig1: str, sig2: str, threshold: float = 0.8) -> bool:
        """Check if two parameter signatures are similar."""
        # Simple similarity check - could be made more sophisticated
        return sig1 == sig2


class ExampleGenerator:
    """
    Generates example claims from successful skill executions.
    """

    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.quality_assessor = ExampleQualityAssessor()
        self.generation_history: List[Dict[str, Any]] = []
        self.max_history_size = 500

        # Generation settings
        self.min_quality_threshold = 0.3
        self.max_examples_per_skill = 50
        self.generation_cooldown_minutes = 5

    async def generate_example_from_execution(
        self, execution_result: ExecutionResult
    ) -> Optional[Claim]:
        """
        Generate an example claim from a successful execution.

        Args:
            execution_result: Result from skill execution

        Returns:
            Generated Claim or None if quality too low
        """
        if not execution_result.success:
            return None

        try:
            # Get existing examples for this skill
            existing_examples = await self.get_examples_for_skill(
                execution_result.skill_id
            )

            # Check if we should generate an example
            if not self._should_generate_example(execution_result, existing_examples):
                return None

            # Assess quality
            quality = self.quality_assessor.assess_example_quality(
                execution_result, existing_examples
            )

            if quality < self.min_quality_threshold:
                logger.debug(f"Example quality too low: {quality}")
                return None

            # Generate example claim
            example_claim = await self._create_example_claim(execution_result, quality)

            if example_claim:
                # Record generation
                self._record_generation(execution_result, example_claim, quality)
                logger.info(
                    f"Generated example for skill {execution_result.skill_id} with quality {quality:.2f}"
                )

            return example_claim

        except Exception as e:
            logger.error(f"Error generating example: {e}")
            return None

    async def get_examples_for_skill(
        self, skill_id: str, limit: int = 100
    ) -> List[Claim]:
        """Get existing examples for a skill."""
        try:
            # Filter for example claims with this skill_id
            example_claims = await self.data_manager.filter_claims(
                filters=None  # Will be implemented to filter by skill_id and tags
            )

            examples = []
            for claim_dict in example_claims:
                if (
                    "type.example" in claim_dict.get("tags", [])
                    and claim_dict.get("skill_id") == skill_id
                ):
                    try:
                        example = Claim(**claim_dict)
                        examples.append(example)
                    except Exception as e:
                        logger.warning(
                            f"Failed to load example claim {claim_dict.get('id', 'unknown')}: {e}"
                        )

            return examples[:limit]

        except Exception as e:
            logger.error(f"Error getting examples for skill {skill_id}: {e}")
            return []

    def _should_generate_example(
        self, execution_result: ExecutionResult, existing_examples: List[Claim]
    ) -> bool:
        """Determine if we should generate an example from this execution."""
        # Check if we have too many examples already
        if len(existing_examples) >= self.max_examples_per_skill:
            return False

        # Check cooldown period
        recent_generations = [
            gen
            for gen in self.generation_history
            if (
                gen["skill_id"] == execution_result.skill_id
                and datetime.fromisoformat(gen["timestamp"])
                > datetime.utcnow()
                - timedelta(minutes=self.generation_cooldown_minutes)
            )
        ]

        if recent_generations:
            return False

        # Check if execution was successful and has meaningful output
        if not execution_result.success or execution_result.result is None:
            return False

        # Check if parameters are meaningful
        if not execution_result.parameters_used:
            return False

        return True

    async def _create_example_claim(
        self, execution_result: ExecutionResult, quality: float
    ) -> Optional[Claim]:
        """Create an example claim from execution result."""
        try:
            # Generate example content
            content = self._generate_example_content(execution_result)

            # Create example claim
            example_claim = Claim(
                skill_id=execution_result.skill_id,
                input_parameters=execution_result.parameters_used,
                output_result=execution_result.result,
                execution_time_ms=execution_result.execution_time_ms,
                example_quality=quality,
                content=content,
                confidence=0.8,  # High confidence for successful executions
                tags=["type.example", "auto_generated"],
                created_by="example_generator",
            )

            # Save to database
            await self.data_manager.create_claim(
                content=example_claim.content,
                confidence=example_claim.confidence,
                tags=example_claim.tags,
                created_by=example_claim.created_by,
            )

            return example_claim

        except Exception as e:
            logger.error(f"Error creating example claim: {e}")
            return None

    def _generate_example_content(self, execution_result: ExecutionResult) -> str:
        """Generate descriptive content for the example."""
        skill_name = execution_result.skill_id

        # Format parameters
        params_str = self._format_parameters(execution_result.parameters_used)

        # Format result
        result_str = self._format_result(execution_result.result)

        # Generate content
        content = f"Example: {skill_name}({params_str}) -> {result_str}"

        # Add execution metadata if interesting
        if execution_result.execution_time_ms > 1000:
            content += f" (took {execution_result.execution_time_ms}ms)"

        return content

    def _format_parameters(self, parameters: Dict[str, Any]) -> str:
        """Format parameters for display."""
        if not parameters:
            return ""

        formatted_params = []
        for key, value in parameters.items():
            if isinstance(value, str):
                if len(value) > 50:
                    value = value[:47] + "..."
                formatted_params.append(f'{key}="{value}"')
            else:
                formatted_params.append(f"{key}={value}")

        return ", ".join(formatted_params)

    def _format_result(self, result: Any) -> str:
        """Format result for display."""
        if result is None:
            return "None"
        elif isinstance(result, str):
            if len(result) > 100:
                return f'"{result[:97]}..."'
            else:
                return f'"{result}"'
        elif isinstance(result, (list, dict)):
            # Truncate large structures
            result_str = json.dumps(result, ensure_ascii=False)
            if len(result_str) > 100:
                return result_str[:97] + "..."
            return result_str
        else:
            return str(result)

    def _record_generation(
        self, execution_result: ExecutionResult, example_claim: Claim, quality: float
    ) -> None:
        """Record example generation for tracking."""
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "skill_id": execution_result.skill_id,
            "execution_id": id(execution_result),
            "example_id": example_claim.id,
            "quality": quality,
            "execution_time_ms": execution_result.execution_time_ms,
            "parameters_count": len(execution_result.parameters_used),
        }

        self.generation_history.append(record)

        # Maintain history size limit
        if len(self.generation_history) > self.max_history_size:
            self.generation_history = self.generation_history[-self.max_history_size :]

    async def batch_generate_examples(
        self, execution_results: List[ExecutionResult]
    ) -> List[Claim]:
        """
        Generate examples from multiple execution results.

        Args:
            execution_results: List of execution results

        Returns:
            List of generated example claims
        """
        generated_examples = []

        for execution_result in execution_results:
            try:
                example = await self.generate_example_from_execution(execution_result)
                if example:
                    generated_examples.append(example)
            except Exception as e:
                logger.error(f"Error in batch generation: {e}")
                continue

        return generated_examples

    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about example generation."""
        if not self.generation_history:
            return {
                "total_generated": 0,
                "average_quality": 0.0,
                "most_generated_skills": [],
                "generation_rate_per_hour": 0.0,
            }

        total_generated = len(self.generation_history)
        average_quality = (
            sum(gen["quality"] for gen in self.generation_history) / total_generated
        )

        # Most generated skills
        skill_counts = {}
        for gen in self.generation_history:
            skill_counts[gen["skill_id"]] = skill_counts.get(gen["skill_id"], 0) + 1

        most_generated_skills = sorted(
            skill_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]

        # Generation rate (last hour)
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_generations = [
            gen
            for gen in self.generation_history
            if datetime.fromisoformat(gen["timestamp"]) > one_hour_ago
        ]
        generation_rate_per_hour = len(recent_generations)

        return {
            "total_generated": total_generated,
            "average_quality": average_quality,
            "most_generated_skills": most_generated_skills,
            "generation_rate_per_hour": generation_rate_per_hour,
        }

    async def cleanup_low_quality_examples(self, min_quality: float = 0.2) -> int:
        """
        Remove low-quality examples from the database.

        Args:
            min_quality: Minimum quality threshold to keep

        Returns:
            Number of examples removed
        """
        try:
            # Get all example claims
            example_claims = await self.data_manager.filter_claims(
                filters=None  # Will be implemented to filter by tags
            )

            removed_count = 0
            for claim_dict in example_claims:
                if "type.example" in claim_dict.get("tags", []):
                    try:
                        example = Claim(**claim_dict)
                        if example.example_quality < min_quality:
                            # Delete low-quality example
                            await self.data_manager.delete_claim(example.id)
                            removed_count += 1
                    except Exception as e:
                        logger.warning(
                            f"Failed to process example claim {claim_dict.get('id', 'unknown')}: {e}"
                        )

            logger.info(f"Cleaned up {removed_count} low-quality examples")
            return removed_count

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0
