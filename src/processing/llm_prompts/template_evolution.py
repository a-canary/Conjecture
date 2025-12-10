"""
Template Evolution System using Genetic Algorithms

Evolves prompt templates over time to optimize performance for tiny LLMs.
Uses genetic algorithms to discover optimal prompt structures and formulations.
"""

import asyncio
import json
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import copy
import re

from .models import PromptTemplate, PromptTemplateType, PromptMetrics
from .tiny_llm_optimizer import OptimizationMetrics, TaskDescription

class MutationType(str, Enum):
    """Types of mutations for template evolution"""
    WORD_SUBSTITUTION = "word_substitution"
    STRUCTURE_CHANGE = "structure_change"
    VARIABLE_MODIFICATION = "variable_modification"
    LENGTH_ADJUSTMENT = "length_adjustment"
    FORMAT_CHANGE = "format_change"
    EXAMPLE_MODIFICATION = "example_modification"
    INSTRUCTION_SIMPLIFICATION = "instruction_simplification"

@dataclass
class EvolutionConfig:
    """Configuration for template evolution"""
    population_size: int = 20
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elitism_rate: float = 0.2  # Keep top N% unchanged
    max_generations: int = 50
    convergence_threshold: float = 0.01  # Stop if improvement < threshold
    tournament_size: int = 3  # For tournament selection

    # Tiny LLM specific settings
    max_template_length: int = 500  # Maximum tokens for tiny models
    min_template_length: int = 50   # Minimum tokens
    preferred_structures: List[str] = field(default_factory=lambda: ["xml", "json", "plain"])

@dataclass
class TemplateGenome:
    """Represents a template's genetic material"""
    template_id: str
    content_segments: List[str]  # Break template into mutable segments
    variables: List[Dict[str, Any]]
    structure_type: str
    metadata: Dict[str, Any]
    fitness_score: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_history: List[str] = field(default_factory=list)

@dataclass
class EvolutionStats:
    """Statistics for evolution process"""
    generation: int
    best_fitness: float
    average_fitness: float
    worst_fitness: float
    diversity_score: float
    convergence_rate: float
    mutations_applied: int
    templates_tested: int

class TemplateMutator:
    """Handles mutation of template genomes"""

    def __init__(self, evolution_config: EvolutionConfig):
        """Initialize template mutator"""
        self.config = evolution_config
        self.word_substitutions = {
            "analyze": ["examine", "study", "review", "investigate"],
            "create": ["generate", "produce", "develop", "make"],
            "provide": ["give", "offer", "supply", "present"],
            "extract": ["pull", "get", "obtain", "retrieve"],
            "classify": ["categorize", "group", "sort", "organize"],
            "evaluate": ["assess", "judge", "measure", "check"]
        }

        self.instruction_simplifications = [
            (r'\bplease\b', ''),
            (r'\bkindly\b', ''),
            (r'\byou are required to\b', 'you must'),
            (r'\bit is important that\b', 'ensure'),
            (r'\bwe need you to\b', 'you should'),
            (r'\bthe goal is to\b', 'aim to'),
        ]

    def mutate(self, genome: TemplateGenome) -> TemplateGenome:
        """Apply mutations to a template genome"""

        mutated_genome = TemplateGenome(
            template_id=f"mutated_{int(time.time())}_{random.randint(1000, 9999)}",
            content_segments=genome.content_segments.copy(),
            variables=copy.deepcopy(genome.variables),
            structure_type=genome.structure_type,
            metadata=copy.deepcopy(genome.metadata),
            generation=genome.generation + 1,
            parent_ids=[genome.template_id]
        )

        mutations_applied = []

        # Apply mutations based on mutation rate
        for i, segment in enumerate(mutated_genome.content_segments):
            if random.random() < self.config.mutation_rate:
                mutation_type = self._select_mutation_type(segment)

                if mutation_type == MutationType.WORD_SUBSTITUTION:
                    mutated_segment = self._word_substitution_mutation(segment)
                    if mutated_segment != segment:
                        mutated_genome.content_segments[i] = mutated_segment
                        mutations_applied.append(f"word_substitution_segment_{i}")

                elif mutation_type == MutationType.INSTRUCTION_SIMPLIFICATION:
                    mutated_segment = self._simplify_instructions(segment)
                    if mutated_segment != segment:
                        mutated_genome.content_segments[i] = mutated_segment
                        mutations_applied.append(f"instruction_simplification_segment_{i}")

                elif mutation_type == MutationType.LENGTH_ADJUSTMENT:
                    mutated_segment = self._adjust_length(segment)
                    if mutated_segment != segment:
                        mutated_genome.content_segments[i] = mutated_segment
                        mutations_applied.append(f"length_adjustment_segment_{i}")

                elif mutation_type == MutationType.STRUCTURE_CHANGE:
                    if random.random() < 0.3:  # Less frequent
                        mutated_genome.structure_type = self._mutate_structure(genome.structure_type)
                        mutations_applied.append("structure_change")

        # Mutate variables occasionally
        if random.random() < self.config.mutation_rate * 0.5:
            mutated_genome.variables = self._mutate_variables(genome.variables)
            mutations_applied.append("variable_modification")

        mutated_genome.mutation_history = mutations_applied

        # Ensure template length constraints
        mutated_genome = self._apply_length_constraints(mutated_genome)

        return mutated_genome

    def _select_mutation_type(self, segment: str) -> MutationType:
        """Select appropriate mutation type for segment"""

        if len(segment) < 20:
            return MutationType.WORD_SUBSTITUTION
        elif "TASK:" in segment or "ROLE:" in segment:
            return MutationType.INSTRUCTION_SIMPLIFICATION
        elif len(segment) > 100:
            return MutationType.LENGTH_ADJUSTMENT
        else:
            return random.choice([
                MutationType.WORD_SUBSTITUTION,
                MutationType.INSTRUCTION_SIMPLIFICATION,
                MutationType.LENGTH_ADJUSTMENT
            ])

    def _word_substitution_mutation(self, segment: str) -> str:
        """Apply word substitution mutation"""

        words = segment.split()
        mutated_words = []

        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())

            if clean_word in self.word_substitutions and random.random() < 0.3:
                substitution = random.choice(self.word_substitutions[clean_word])
                # Preserve original capitalization and punctuation
                if word[0].isupper():
                    substitution = substitution.capitalize()
                if word.endswith('.'):
                    substitution += '.'
                mutated_words.append(substitution)
            else:
                mutated_words.append(word)

        return ' '.join(mutated_words)

    def _simplify_instructions(self, segment: str) -> str:
        """Simplify instructions in segment"""

        simplified = segment
        for pattern, replacement in self.instruction_simplifications:
            simplified = re.sub(pattern, replacement, simplified, flags=re.IGNORECASE)

        # Remove redundant words
        redundant_patterns = [
            r'\bvery\b',
            r'\bquite\b',
            r'\brather\b',
            r'\bsomewhat\b',
            r'\bessentially\b'
        ]

        for pattern in redundant_patterns:
            simplified = re.sub(pattern, '', simplified, flags=re.IGNORECASE)

        # Clean up extra spaces
        simplified = re.sub(r'\s+', ' ', simplified).strip()

        return simplified

    def _adjust_length(self, segment: str) -> str:
        """Adjust segment length"""

        current_length = len(segment.split())

        if current_length > 50 and random.random() < 0.7:
            # Shorten segment
            sentences = segment.split('.')
            if len(sentences) > 1:
                # Remove least important sentence
                sentences = sentences[:-1]
                return '.'.join(sentences) + '.'

        elif current_length < 15 and random.random() < 0.5:
            # Expand segment with clarification
            clarifications = [
                " Be specific.",
                " Provide details.",
                " Include examples.",
                " Be thorough."
            ]
            return segment + random.choice(clarifications)

        return segment

    def _mutate_structure(self, current_structure: str) -> str:
        """Mutate structure type"""
        available_structures = self.config.preferred_structures.copy()
        if current_structure in available_structures:
            available_structures.remove(current_structure)

        return random.choice(available_structures) if available_structures else current_structure

    def _mutate_variables(self, variables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Mutate template variables"""

        mutated_variables = []

        for var in variables:
            mutated_var = var.copy()

            # Occasionally make optional variables required or vice versa
            if random.random() < 0.2:
                mutated_var["required"] = not mutated_var.get("required", True)

            # Occasionally adjust descriptions
            if random.random() < 0.1 and "description" in mutated_var:
                mutated_var["description"] = self._simplify_instructions(mutated_var["description"])

            mutated_variables.append(mutated_var)

        return mutated_variables

    def _apply_length_constraints(self, genome: TemplateGenome) -> TemplateGenome:
        """Apply length constraints to genome"""

        # Reconstruct template content
        content = ' '.join(genome.content_segments)
        content_length = len(content.split())

        if content_length > self.config.max_template_length:
            # Truncate content
            words = content.split()
            truncated_words = words[:self.config.max_template_length]
            genome.content_segments = [' '.join(truncated_words)]

        elif content_length < self.config.min_template_length:
            # Expand with generic content
            expansion = " Please provide a comprehensive response."
            genome.content_segments.append(expansion)

        return genome

class TemplateCrossover:
    """Handles crossover between template genomes"""

    def __init__(self, evolution_config: EvolutionConfig):
        """Initialize template crossover"""
        self.config = evolution_config

    def crossover(self, parent1: TemplateGenome, parent2: TemplateGenome) -> Tuple[TemplateGenome, TemplateGenome]:
        """Perform crossover between two parent genomes"""

        if random.random() > self.config.crossover_rate:
            # No crossover, return copies of parents
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

        # Single-point crossover on content segments
        child1_segments, child2_segments = self._segment_crossover(
            parent1.content_segments,
            parent2.content_segments
        )

        # Create child genomes
        child1 = TemplateGenome(
            template_id=f"cross_{int(time.time())}_1_{random.randint(1000, 9999)}",
            content_segments=child1_segments,
            variables=self._variable_crossover(parent1.variables, parent2.variables),
            structure_type=random.choice([parent1.structure_type, parent2.structure_type]),
            metadata=self._metadata_crossover(parent1.metadata, parent2.metadata),
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.template_id, parent2.template_id]
        )

        child2 = TemplateGenome(
            template_id=f"cross_{int(time.time())}_2_{random.randint(1000, 9999)}",
            content_segments=child2_segments,
            variables=self._variable_crossover(parent1.variables, parent2.variables),
            structure_type=random.choice([parent1.structure_type, parent2.structure_type]),
            metadata=self._metadata_crossover(parent1.metadata, parent2.metadata),
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.template_id, parent2.template_id]
        )

        return child1, child2

    def _segment_crossover(
        self,
        segments1: List[str],
        segments2: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Perform crossover on content segments"""

        # Ensure segments have same length by padding
        max_len = max(len(segments1), len(segments2))
        segments1.extend([''] * (max_len - len(segments1)))
        segments2.extend([''] * (max_len - len(segments2)))

        # Select crossover point
        crossover_point = random.randint(1, max_len - 1)

        # Create children
        child1_segments = segments1[:crossover_point] + segments2[crossover_point:]
        child2_segments = segments2[:crossover_point] + segments1[crossover_point:]

        return child1_segments, child2_segments

    def _variable_crossover(
        self,
        variables1: List[Dict[str, Any]],
        variables2: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Crossover variable definitions"""

        # Combine variables from both parents
        all_variables = variables1 + variables2

        # Remove duplicates (by name)
        seen_names = set()
        unique_variables = []
        for var in all_variables:
            name = var.get("name", "")
            if name and name not in seen_names:
                seen_names.add(name)
                unique_variables.append(var)

        # Randomly select subset
        num_vars = min(len(unique_variables), max(len(variables1), len(variables2)))
        selected_variables = random.sample(unique_variables, num_vars)

        return selected_variables

    def _metadata_crossover(
        self,
        metadata1: Dict[str, Any],
        metadata2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Crossover metadata"""

        # Combine metadata, preferring values from parent1 for conflicts
        combined_metadata = {}
        combined_metadata.update(metadata2)
        combined_metadata.update(metadata1)

        return combined_metadata

class TemplateSelector:
    """Handles selection of templates for next generation"""

    def __init__(self, evolution_config: EvolutionConfig):
        """Initialize template selector"""
        self.config = evolution_config

    def tournament_selection(self, population: List[TemplateGenome]) -> TemplateGenome:
        """Select genome using tournament selection"""

        tournament = random.sample(population, min(self.config.tournament_size, len(population)))
        return max(tournament, key=lambda genome: genome.fitness_score)

    def roulette_wheel_selection(self, population: List[TemplateGenome]) -> TemplateGenome:
        """Select genome using roulette wheel selection"""

        total_fitness = sum(genome.fitness_score for genome in population)
        if total_fitness == 0:
            return random.choice(population)

        selection_point = random.uniform(0, total_fitness)
        current_sum = 0

        for genome in population:
            current_sum += genome.fitness_score
            if current_sum >= selection_point:
                return genome

        return population[-1]  # Fallback

    def select_parents(self, population: List[TemplateGenome]) -> Tuple[TemplateGenome, TemplateGenome]:
        """Select two parents for crossover"""

        parent1 = self.tournament_selection(population)
        parent2 = self.tournament_selection(population)

        # Ensure different parents
        attempts = 0
        while parent2.template_id == parent1.template_id and attempts < 10:
            parent2 = self.tournament_selection(population)
            attempts += 1

        return parent1, parent2

class TemplateEvolution:
    """Main template evolution system using genetic algorithms"""

    def __init__(self, evolution_config: Optional[EvolutionConfig] = None):
        """Initialize template evolution system"""
        self.config = evolution_config or EvolutionConfig()
        self.mutator = TemplateMutator(self.config)
        self.crossover = TemplateCrossover(self.config)
        self.selector = TemplateSelector(self.config)

        # Evolution tracking
        self.generation_history: List[EvolutionStats] = []
        self.best_template: Optional[TemplateGenome] = None
        self.convergence_history: List[float] = []

    async def evolve_templates(
        self,
        initial_templates: List[PromptTemplate],
        fitness_function: Callable[[TemplateGenome], float],
        task_description: Optional[TaskDescription] = None
    ) -> List[PromptTemplate]:
        """Evolve templates using genetic algorithm"""

        # Initialize population
        population = self._initialize_population(initial_templates)

        # Evaluate initial population
        population = await self._evaluate_population(population, fitness_function)

        # Evolution loop
        for generation in range(self.config.max_generations):
            # Selection
            elites = self._select_elites(population)

            # Create new population
            new_population = elites.copy()

            # Generate offspring
            while len(new_population) < self.config.population_size:
                parent1, parent2 = self.selector.select_parents(population)

                # Crossover
                child1, child2 = self.crossover.crossover(parent1, parent2)

                # Mutation
                child1 = self.mutator.mutate(child1)
                child2 = self.mutator.mutate(child2)

                new_population.extend([child1, child2])

            # Trim to exact population size
            new_population = new_population[:self.config.population_size]

            # Evaluate new population
            new_population = await self._evaluate_population(new_population, fitness_function)

            # Update population
            population = new_population

            # Track generation stats
            stats = self._calculate_generation_stats(population, generation)
            self.generation_history.append(stats)

            # Update best template
            current_best = max(population, key=lambda g: g.fitness_score)
            if self.best_template is None or current_best.fitness_score > self.best_template.fitness_score:
                self.best_template = current_best

            # Check convergence
            if self._check_convergence():
                print(f"Convergence reached at generation {generation}")
                break

        # Convert best genomes back to PromptTemplates
        evolved_templates = self._genomes_to_templates(
            sorted(population, key=lambda g: g.fitness_score, reverse=True)[:10]
        )

        return evolved_templates

    def _initialize_population(self, initial_templates: List[PromptTemplate]) -> List[TemplateGenome]:
        """Initialize population from seed templates"""

        population = []

        # Add initial templates
        for template in initial_templates:
            genome = self._template_to_genome(template)
            population.append(genome)

        # Generate random variations to fill population
        while len(population) < self.config.population_size:
            # Copy and mutate a random template
            base_template = random.choice(initial_templates)
            base_genome = self._template_to_genome(base_template)
            mutated_genome = self.mutator.mutate(base_genome)
            population.append(mutated_genome)

        return population

    def _template_to_genome(self, template: PromptTemplate) -> TemplateGenome:
        """Convert PromptTemplate to TemplateGenome"""

        # Break template content into segments
        content_segments = self._segment_content(template.template_content)

        return TemplateGenome(
            template_id=template.id,
            content_segments=content_segments,
            variables=[
                {
                    "name": var.name,
                    "type": var.type,
                    "required": var.required,
                    "description": var.description
                }
                for var in template.variables
            ],
            structure_type=template.metadata.get("structure_type", "plain"),
            metadata=template.metadata,
            generation=0
        )

    def _segment_content(self, content: str) -> List[str]:
        """Segment template content into mutable units"""

        # Split by paragraphs and major sections
        segments = []

        # Split by double newlines (paragraphs)
        paragraphs = content.split('\n\n')

        for paragraph in paragraphs:
            if len(paragraph.strip()) > 0:
                # Further split long paragraphs
                if len(paragraph.split()) > 30:
                    sentences = paragraph.split('.')
                    for sentence in sentences:
                        if len(sentence.strip()) > 0:
                            segments.append(sentence.strip() + '.')
                else:
                    segments.append(paragraph.strip())

        return segments

    def _genomes_to_templates(self, genomes: List[TemplateGenome]) -> List[PromptTemplate]:
        """Convert TemplateGenome back to PromptTemplate"""

        templates = []

        for genome in genomes:
            # Reconstruct content
            content = ' '.join(genome.content_segments)

            # Create PromptTemplate
            template = PromptTemplate(
                id=genome.template_id,
                name=f"Evolved Template {genome.template_id}",
                description=f"Template evolved through genetic algorithm (generation {genome.generation})",
                template_type=PromptTemplateType.GENERAL,
                template_content=content,
                variables=[
                    {
                        "name": var["name"],
                        "type": var["type"],
                        "required": var["required"],
                        "description": var.get("description", ""),
                        "validation_rule": var.get("validation_rule")
                    }
                    for var in genome.variables
                ],
                metadata={
                    **genome.metadata,
                    "structure_type": genome.structure_type,
                    "generation": genome.generation,
                    "parent_ids": genome.parent_ids,
                    "mutation_history": genome.mutation_history,
                    "fitness_score": genome.fitness_score
                },
                version=f"evolved_gen_{genome.generation}"
            )

            templates.append(template)

        return templates

    async def _evaluate_population(
        self,
        population: List[TemplateGenome],
        fitness_function: Callable[[TemplateGenome], float]
    ) -> List[TemplateGenome]:
        """Evaluate fitness of entire population"""

        for genome in population:
            genome.fitness_score = await fitness_function(genome)

        return population

    def _select_elites(self, population: List[TemplateGenome]) -> List[TemplateGenome]:
        """Select elite templates for next generation"""

        num_elites = max(1, int(len(population) * self.config.elitism_rate))
        sorted_population = sorted(population, key=lambda g: g.fitness_score, reverse=True)
        return sorted_population[:num_elites]

    def _calculate_generation_stats(self, population: List[TemplateGenome], generation: int) -> EvolutionStats:
        """Calculate statistics for current generation"""

        fitness_scores = [genome.fitness_score for genome in population]

        # Calculate diversity (average distance between genomes)
        diversity_score = self._calculate_diversity(population)

        # Calculate convergence rate
        convergence_rate = 0.0
        if len(self.generation_history) > 0:
            prev_avg = self.generation_history[-1].average_fitness
            current_avg = sum(fitness_scores) / len(fitness_scores)
            convergence_rate = abs(current_avg - prev_avg)

        stats = EvolutionStats(
            generation=generation,
            best_fitness=max(fitness_scores),
            average_fitness=sum(fitness_scores) / len(fitness_scores),
            worst_fitness=min(fitness_scores),
            diversity_score=diversity_score,
            convergence_rate=convergence_rate,
            mutations_applied=len(set().union(*[genome.mutation_history for genome in population])),
            templates_tested=len(population)
        )

        return stats

    def _calculate_diversity(self, population: List[TemplateGenome]) -> float:
        """Calculate population diversity"""

        if len(population) < 2:
            return 0.0

        # Simple diversity based on content differences
        total_distance = 0
        comparisons = 0

        for i, genome1 in enumerate(population):
            for j, genome2 in enumerate(population[i+1:], i+1):
                # Calculate content difference
                content1 = ' '.join(genome1.content_segments)
                content2 = ' '.join(genome2.content_segments)

                # Simple word-based distance
                words1 = set(content1.lower().split())
                words2 = set(content2.lower().split())

                if words1 or words2:
                    intersection = len(words1.intersection(words2))
                    union = len(words1.union(words2))
                    distance = 1 - (intersection / union) if union > 0 else 0

                    total_distance += distance
                    comparisons += 1

        return total_distance / comparisons if comparisons > 0 else 0.0

    def _check_convergence(self) -> bool:
        """Check if evolution has converged"""

        if len(self.generation_history) < 5:
            return False

        # Check recent improvements
        recent_generations = self.generation_history[-5:]
        improvements = [
            gen.average_fitness for gen in recent_generations
        ]

        # Calculate improvement rate
        if len(improvements) >= 2:
            recent_improvement = abs(improvements[-1] - improvements[0])
            return recent_improvement < self.config.convergence_threshold

        return False

    def get_evolution_report(self) -> Dict[str, Any]:
        """Get comprehensive evolution report"""

        if not self.generation_history:
            return {"status": "No evolution data available"}

        best_fitness = max(gen.best_fitness for gen in self.generation_history)
        final_avg_fitness = self.generation_history[-1].average_fitness
        total_mutations = sum(gen.mutations_applied for gen in self.generation_history)

        return {
            "generations_completed": len(self.generation_history),
            "best_fitness_achieved": best_fitness,
            "final_average_fitness": final_avg_fitness,
            "improvement_from_baseline": final_avg_fitness - self.generation_history[0].average_fitness,
            "total_mutations_applied": total_mutations,
            "convergence_generation": next(
                (i for i, gen in enumerate(self.generation_history)
                 if gen.convergence_rate < self.config.convergence_threshold),
                None
            ),
            "best_template_id": self.best_template.template_id if self.best_template else None,
            "evolution_efficiency": best_fitness / len(self.generation_history) if self.generation_history else 0
        }

    def get_best_template(self) -> Optional[PromptTemplate]:
        """Get the best evolved template as PromptTemplate"""

        if not self.best_template:
            return None

        templates = self._genomes_to_templates([self.best_template])
        return templates[0] if templates else None