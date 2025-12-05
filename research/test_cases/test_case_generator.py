#!/usr/bin/env python3
"""
Test Case Generator for Conjecture Research
Generates diverse test cases for different experiment types
"""

import json
import random
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime


class TestCaseGenerator:
    """Generates test cases for Conjecture experiments"""
    
    def __init__(self, output_dir: str = "research/test_cases"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test case templates and data
        self.logic_puzzles = self._load_logic_puzzles()
        self.math_problems = self._load_math_problems()
        self.reasoning_scenarios = self._load_reasoning_scenarios()
        self.context_passages = self._load_context_passages()
    
    def _load_logic_puzzles(self) -> List[Dict[str, Any]]:
        """Load logic puzzle templates"""
        return [
            {
                "template": "house_color_puzzle",
                "difficulty": "hard",
                "elements": ["houses", "colors", "professions", "items"],
                "constraints": 10
            },
            {
                "template": "seating_arrangement",
                "difficulty": "medium", 
                "elements": ["people", "positions", "attributes"],
                "constraints": 8
            },
            {
                "template": "family_relationships",
                "difficulty": "medium",
                "elements": ["family_members", "relationships", "ages"],
                "constraints": 7
            }
        ]
    
    def _load_math_problems(self) -> List[Dict[str, Any]]:
        """Load math problem templates"""
        return [
            {
                "template": "algebra_word_problem",
                "difficulty": "medium",
                "concepts": ["linear_equations", "quadratic_equations"],
                "steps": 4
            },
            {
                "template": "geometry_problem",
                "difficulty": "hard",
                "concepts": ["area", "perimeter", "volume"],
                "steps": 6
            },
            {
                "template": "rate_problem",
                "difficulty": "medium",
                "concepts": ["speed", "time", "distance"],
                "steps": 5
            }
        ]
    
    def _load_reasoning_scenarios(self) -> List[Dict[str, Any]]:
        """Load reasoning scenario templates"""
        return [
            {
                "template": "ethical_dilemma",
                "difficulty": "hard",
                "factors": ["stakeholders", "consequences", "principles"],
                "complexity": "high"
            },
            {
                "template": "business_decision",
                "difficulty": "medium",
                "factors": ["costs", "benefits", "risks", "alternatives"],
                "complexity": "medium"
            },
            {
                "template": "scientific_hypothesis",
                "difficulty": "hard",
                "factors": ["evidence", "methodology", "conclusions"],
                "complexity": "high"
            }
        ]
    
    def _load_context_passages(self) -> List[Dict[str, Any]]:
        """Load long context passage templates"""
        return [
            {
                "template": "historical_event",
                "topic": "Renaissance",
                "length_words": 600,
                "question_types": ["factual", "causal", "comparative"]
            },
            {
                "template": "scientific_explanation",
                "topic": "Climate Change",
                "length_words": 800,
                "question_types": ["mechanism", "evidence", "implications"]
            },
            {
                "template": "technical_documentation",
                "topic": "Software Architecture",
                "length_words": 700,
                "question_types": ["procedural", "technical", "design"]
            }
        ]
    
    def _generate_seating_puzzle(self) -> Dict[str, Any]:
        """Generate a seating arrangement logic puzzle"""
        people = ["Alice", "Bob", "Charlie", "David", "Eve"]
        attributes = ["engineer", "doctor", "teacher", "artist", "chef"]
        positions = [1, 2, 3, 4, 5]
        
        # Shuffle to create random puzzle
        random.shuffle(attributes)
        
        puzzle = {
            "id": f"seating_arrangement_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "category": "complex_reasoning",
            "difficulty": "medium",
            "description": "Seating arrangement logic puzzle with professions",
            "question": f"""Five people ({', '.join(people)}) are sitting in a row at positions 1-5. Each person has a different profession. Use these clues to determine who sits where and their profession:

1. The engineer sits at one end
2. {people[1]} sits next to the doctor
3. The teacher sits in the middle (position 3)
4. {people[3]} sits immediately to the right of the artist
5. The chef sits next to {people[0]}

Who sits in position 2 and what is their profession?""",
            "ground_truth": f"""Position 2: {people[1]} is the {attributes[1]}

Solution:
- Position 1: {people[0]} is the {attributes[0]}
- Position 2: {people[1]} is the {attributes[1]}
- Position 3: {people[2]} is the {attributes[2]}
- Position 4: {people[3]} is the {attributes[3]}
- Position 5: {people[4]} is the {attributes[4]}""",
            "expected_approach": "break_down_problem",
            "metadata": {
                "type": "logic_puzzle",
                "requires_deduction": True,
                "multiple_constraints": True,
                "spatial_reasoning": True,
                "estimated_time_minutes": 12,
                "claims_based_approach_beneficial": True
            }
        }
        
        return puzzle
    
    def generate_logic_puzzle(self, template_id: int = 0) -> Dict[str, Any]:
        """Generate a logic puzzle test case"""
        template = self.logic_puzzles[template_id]
        
        if template["template"] == "house_color_puzzle":
            return self._generate_house_puzzle()
        elif template["template"] == "seating_arrangement":
            return self._generate_seating_puzzle()
        elif template["template"] == "family_relationships":
            return self._generate_family_puzzle()
    
    def _generate_house_puzzle(self) -> Dict[str, Any]:
        """Generate house color logic puzzle"""
        # Variabilize the puzzle
        colors = ["red", "blue", "green", "yellow", "white"]
        professions = ["doctor", "teacher", "engineer", "artist", "baker"]
        fruits = ["apple", "banana", "cherry", "date", "elderberry"]
        
        random.shuffle(colors)
        random.shuffle(professions)
        random.shuffle(fruits)
        
        # Create a specific solution
        solution = {
            1: {"color": colors[0], "profession": professions[0], "fruit": fruits[0]},
            2: {"color": colors[1], "profession": professions[1], "fruit": fruits[1]},
            3: {"color": colors[2], "profession": professions[2], "fruit": fruits[2]},
            4: {"color": colors[3], "profession": professions[3], "fruit": fruits[3]},
            5: {"color": colors[4], "profession": professions[4], "fruit": fruits[4]}
        }
        
        # Generate clues based on solution
        clues = self._generate_house_clues(solution)
        
        question = f"In a small town, there are five houses in a row, each painted a different color: {', '.join(colors)}. Each house is owned by a person with a different profession: {', '.join(professions)}. Each person has a different favorite fruit: {', '.join(fruits)}. Using the following clues, determine who owns the {colors[1]} house and what is their favorite fruit?\n\nClues:\n" + "\n".join([f"{i+1}. {clue}" for i, clue in enumerate(clues)])
        
        # Find answer
        for house_num, details in solution.items():
            if details["color"] == colors[1]:
                answer = f"The {details['profession']} owns the {colors[1]} house and their favorite fruit is {details['fruit']}."
                break
        
        return {
            "id": f"logic_puzzle_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "category": "complex_reasoning",
            "difficulty": "hard",
            "description": "Multi-step logical reasoning with conditional statements",
            "question": question,
            "ground_truth": answer,
            "expected_approach": "break_down_problem",
            "metadata": {
                "type": "logic_puzzle",
                "requires_deduction": True,
                "multiple_constraints": True,
                "spatial_reasoning": True,
                "estimated_time_minutes": 15,
                "claims_based_approach_beneficial": True
            }
        }
    
    def _generate_house_clues(self, solution: Dict[int, Dict[str, str]]) -> List[str]:
        """Generate clues for house puzzle based on solution"""
        clues = []
        
        # Find middle house
        middle_house = solution[3]
        clues.append(f"The {middle_house['profession']} lives in the middle house.")
        
        # Find first house
        first_house = solution[1]
        clues.append(f"The {first_house['profession']} lives in the first house.")
        
        # Find last house's fruit
        last_house = solution[5]
        clues.append(f"The person who likes {last_house['fruit']}s lives in the last house.")
        
        # Find engineer's color
        for house_num, details in solution.items():
            if details["profession"] == "engineer":
                clues.append(f"The engineer lives in the {details['color']} house.")
                break
        
        # Find teacher's fruit
        for house_num, details in solution.items():
            if details["profession"] == "teacher":
                clues.append(f"The teacher likes {details['fruit']}s.")
                break
        
        # Add more complex clues
        # Find artist position
        for house_num, details in solution.items():
            if details["profession"] == "artist":
                if house_num > 1:
                    neighbor = solution[house_num - 1]
                    clues.append(f"The artist lives next to the person who likes {neighbor['fruit']}s.")
                break
        
        # Find color relationships
        colors_list = [details["color"] for details in solution.values()]
        if "red" in colors_list and "blue" in colors_list:
            red_pos = next(i for i, details in solution.items() if details["color"] == "red")
            blue_pos = next(i for i, details in solution.items() if details["color"] == "blue")
            if red_pos < blue_pos:
                clues.append("The red house is somewhere to the left of the blue house.")
        
        return clues
    
    def _generate_algebra_problem(self) -> Dict[str, Any]:
        """Generate an algebra word problem"""
        problem_types = [
            "quadratic_area",
            "rate_distance",
            "mixture_problem",
            "work_rate"
        ]
        
        problem_type = random.choice(problem_types)
        
        if problem_type == "quadratic_area":
            return self._generate_quadratic_area_problem()
        elif problem_type == "rate_distance":
            return self._generate_rate_distance_problem()
        elif problem_type == "mixture_problem":
            return self._generate_mixture_problem()
        else:
            return self._generate_work_rate_problem()
    
    def _generate_quadratic_area_problem(self) -> Dict[str, Any]:
        """Generate quadratic area problem"""
        # Generate parameters
        length_diff = random.randint(20, 80)  # Length is longer than width
        area = random.randint(8000, 20000)   # Total area
        
        # Calculate actual dimensions (ensure integer solution)
        # w * (w + length_diff) = area
        # w² + length_diff*w - area = 0
        discriminant = length_diff**2 + 4*area
        sqrt_discriminant = int(discriminant**0.5)
        
        # Adjust area to get perfect square discriminant
        while sqrt_discriminant**2 != discriminant:
            area += 100
            discriminant = length_diff**2 + 4*area
            sqrt_discriminant = int(discriminant**0.5)
        
        width = (-length_diff + sqrt_discriminant) // 2
        length = width + length_diff
        
        question = f"""A farmer has a rectangular field that is {length_diff} meters longer than it is wide. The farmer wants to divide the field into two equal rectangular sections by building a fence parallel to the width. If the total area of the field is {area:,} square meters and the fence costs ${random.randint(10, 25)} per meter, what is the total cost of the dividing fence?"""
        
        fence_cost_per_meter = 15  # Default for calculation
        fence_length = width
        total_cost = fence_length * fence_cost_per_meter
        
        return {
            "id": f"algebra_quadratic_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "category": "mathematical_reasoning",
            "difficulty": "medium",
            "description": "Multi-step mathematical problem with algebraic reasoning",
            "question": question,
            "ground_truth": f"The total cost of the dividing fence is ${total_cost:,.2f}.\n\nSolution:\nLet width = w meters, then length = w + {length_diff} meters\nArea = w × (w + {length_diff}) = {area:,}\nw² + {length_diff}w - {area:,} = 0\nUsing quadratic formula: w = [-{length_diff} ± √({length_diff}² + 4×{area:,})]/2\nw = [-{length_diff} + √{discriminant:,}]/2 = [-{length_diff} + {sqrt_discriminant:,}]/2 = {width:,} meters\n\nThe dividing fence is parallel to the width, so it equals the width: {width:,} meters\nCost = {width:,} × ${fence_cost_per_meter} = ${total_cost:,.2f}",
            "expected_approach": "step_by_step",
            "claims_needed": [
                f"Field is rectangular with length = width + {length_diff}",
                f"Total area = {area:,} square meters",
                "Area formula: Area = length × width",
                "Dividing fence is parallel to width",
                "Fence length equals field width",
                f"Cost = length × ${fence_cost_per_meter} per meter"
            ],
            "solution_steps": [
                "Set up variables: let width = w, length = w + {length_diff}",
                "Write area equation: w(w + {length_diff}) = {area:,}",
                "Expand and rearrange: w² + {length_diff}w - {area:,} = 0",
                "Apply quadratic formula",
                "Calculate positive solution for width",
                "Determine fence length equals width",
                "Calculate total cost"
            ],
            "metadata": {
                "type": "algebra",
                "requires_quadratic_formula": True,
                "multiple_steps": True,
                "unit_conversion": False,
                "estimated_time_minutes": 10,
                "claims_based_approach_beneficial": True
            }
        }
    
    def _generate_rate_distance_problem(self) -> Dict[str, Any]:
        """Generate rate-distance-time problem"""
        speed1 = random.randint(40, 80)
        speed2 = random.randint(50, 90)
        time_diff = random.randint(1, 3)
        
        question = f"""Two cars travel from City A to City B, a distance of {random.randint(200, 500)} miles. Car 1 travels at {speed1} mph and Car 2 travels at {speed2} mph. If Car 2 leaves {time_diff} hours after Car 1, which car arrives first and by how much time?"""
        
        return {
            "id": f"rate_distance_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "category": "mathematical_reasoning",
            "difficulty": "medium",
            "description": "Rate-distance-time problem with comparative analysis",
            "question": question,
            "ground_truth": "Solution requires calculating travel times and comparing results",
            "expected_approach": "step_by_step",
            "metadata": {
                "type": "rate_problem",
                "requires_comparative_analysis": True,
                "multiple_steps": True,
                "estimated_time_minutes": 8,
                "claims_based_approach_beneficial": True
            }
        }
    
    def _generate_mixture_problem(self) -> Dict[str, Any]:
        """Generate mixture problem"""
        return {
            "id": f"mixture_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "category": "mathematical_reasoning",
            "difficulty": "medium",
            "description": "Mixture problem with concentration calculations",
            "question": "A chemist needs to create a 30% acid solution by mixing a 50% acid solution with a 10% acid solution. How much of each solution is needed to make 200 liters of the 30% solution?",
            "ground_truth": "100 liters of 50% solution and 100 liters of 10% solution",
            "expected_approach": "step_by_step",
            "metadata": {
                "type": "mixture_problem",
                "requires_system_of_equations": True,
                "estimated_time_minutes": 12,
                "claims_based_approach_beneficial": True
            }
        }
    
    def _generate_work_rate_problem(self) -> Dict[str, Any]:
        """Generate work rate problem"""
        return {
            "id": f"work_rate_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "category": "mathematical_reasoning",
            "difficulty": "medium",
            "description": "Work rate problem with combined efficiency",
            "question": "Worker A can complete a job in 6 hours and Worker B can complete it in 8 hours. Working together, how long will it take them to complete the job?",
            "ground_truth": "Approximately 3.43 hours (3 hours 26 minutes)",
            "expected_approach": "step_by_step",
            "metadata": {
                "type": "work_rate",
                "requires_rate_calculations": True,
                "estimated_time_minutes": 8,
                "claims_based_approach_beneficial": True
            }
        }
    
    def _generate_evidence_scenario(self) -> Dict[str, Any]:
        """Generate evidence evaluation scenario"""
        scenarios = [
            "drug_approval",
            "policy_decision",
            "investment_choice",
            "scientific_controversy"
        ]
        
        scenario = random.choice(scenarios)
        
        if scenario == "drug_approval":
            return self._generate_drug_approval_scenario()
        elif scenario == "policy_decision":
            return self._generate_policy_decision_scenario()
        elif scenario == "investment_choice":
            return self._generate_investment_scenario()
        else:
            return self._generate_scientific_controversy_scenario()
    
    def _generate_drug_approval_scenario(self) -> Dict[str, Any]:
        """Generate drug approval evidence evaluation scenario"""
        return {
            "id": f"evidence_drug_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "category": "evidence_evaluation",
            "difficulty": "medium",
            "description": "Evaluate conflicting evidence and make reasoned judgment",
            "question": """A pharmaceutical company is testing a new drug for treating hypertension. They present the following evidence from their clinical trials. Evaluate the evidence and determine whether the drug should be approved for market use.

Evidence presented:
1. Study A (n=1000): 15% reduction in blood pressure, p=0.03, funded by the company
2. Study B (n=500): 8% reduction in blood pressure, p=0.12, not statistically significant, funded by the company
3. Study C (n=2000): 12% reduction in blood pressure, p=0.01, independent research
4. Study D (n=300): 18% reduction in blood pressure, p=0.08, not statistically significant, independent research
5. Side effects: 5% experienced mild headaches, 2% experienced dizziness
6. Cost: 3x more expensive than existing treatments
7. Long-term effects: Unknown (studies only 6 months duration)
8. Mechanism of action: Well-understood and plausible""",
            "ground_truth": "Based on the evidence, the drug should NOT be approved for market use at this time, though it shows promise and warrants further investigation.",
            "expected_approach": "evidence_weighting",
            "metadata": {
                "type": "evidence_evaluation",
                "requires_critical_thinking": True,
                "conflicting_evidence": True,
                "risk_assessment": True,
                "estimated_time_minutes": 15,
                "claims_based_approach_beneficial": True
            }
        }
    
    def _generate_policy_decision_scenario(self) -> Dict[str, Any]:
        """Generate policy decision scenario"""
        return {
            "id": f"evidence_policy_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "category": "evidence_evaluation",
            "difficulty": "hard",
            "description": "Complex policy decision with multiple stakeholders",
            "question": """A city is considering implementing a congestion pricing policy for downtown traffic. Evaluate the following evidence and make a recommendation:

Evidence:
1. Economic analysis: $10M annual revenue, 20% traffic reduction expected
2. Business impact: 15% of downtown businesses report concerns about customer access
3. Environmental study: 30% reduction in emissions, improved air quality
4. Equity analysis: Disproportionate impact on low-income commuters
5. Public opinion: 45% support, 40% oppose, 15% undecided
6. Comparable cities: London and Singapore show positive long-term results
7. Implementation cost: $5M initial setup, $2M annual maintenance""",
            "ground_truth": "Recommendation should balance economic, environmental, and equity factors",
            "expected_approach": "evidence_weighting",
            "metadata": {
                "type": "policy_analysis",
                "requires_stakeholder_analysis": True,
                "multiple_factors": True,
                "estimated_time_minutes": 20,
                "claims_based_approach_beneficial": True
            }
        }
    
    def _generate_investment_scenario(self) -> Dict[str, Any]:
        """Generate investment decision scenario"""
        return {
            "id": f"evidence_investment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "category": "evidence_evaluation",
            "difficulty": "medium",
            "description": "Investment decision with risk-reward analysis",
            "question": """An investment committee is evaluating three different investment options. Analyze the evidence and provide a recommendation:

Option A - Tech Startup:
- Historical returns: 25% average over 3 years
- Risk level: High (60% failure rate for similar startups)
- Market analysis: Growing market, strong competitive position
- Funding needed: $2M

Option B - Real Estate Fund:
- Historical returns: 8% average over 10 years
- Risk level: Low-Moderate
- Market analysis: Stable market, predictable returns
- Funding needed: $5M

Option C - Green Energy Project:
- Historical returns: 15% average over 5 years
- Risk level: Moderate
- Market analysis: Government incentives, growing demand
- Funding needed: $3M""",
            "ground_truth": "Recommendation should consider risk tolerance, time horizon, and diversification",
            "expected_approach": "evidence_weighting",
            "metadata": {
                "type": "investment_analysis",
                "requires_risk_assessment": True,
                "comparative_analysis": True,
                "estimated_time_minutes": 15,
                "claims_based_approach_beneficial": True
            }
        }
    
    def _generate_scientific_controversy_scenario(self) -> Dict[str, Any]:
        """Generate scientific controversy scenario"""
        return {
            "id": f"evidence_controversy_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "category": "evidence_evaluation",
            "difficulty": "hard",
            "description": "Scientific controversy with conflicting studies",
            "question": """Evaluate the evidence surrounding a controversial scientific hypothesis about the effects of screen time on adolescent mental health:

Supporting Evidence:
1. Longitudinal study (n=5000): 2x increase in depression symptoms with 3+ hours daily screen time
2. Meta-analysis of 50 studies: Consistent correlation between screen time and anxiety
3. Neurological research: Changes in brain development patterns correlated with screen exposure

Contradictory Evidence:
1. Controlled experiment (n=200): No significant causal relationship found
2. Cross-cultural study: Different effects in different cultural contexts
3. Recent review: Publication bias may inflate effect sizes

Moderating Factors:
- Parental involvement
- Type of content (educational vs. entertainment)
- Social interaction during screen use
- Individual personality factors""",
            "ground_truth": "Analysis should acknowledge complexity and avoid oversimplification",
            "expected_approach": "evidence_weighting",
            "metadata": {
                "type": "scientific_controversy",
                "requires_nuanced_analysis": True,
                "conflicting_evidence": True,
                "estimated_time_minutes": 25,
                "claims_based_approach_beneficial": True
            }
        }
    
    def _generate_generic_case(self, category: str) -> Dict[str, Any]:
        """Generate a generic test case for specified category"""
        templates = {
            "task_decomposition": self._generate_task_decomposition_case,
            "context_compression": self._generate_context_compression_case,
            "claims_reasoning": self._generate_claims_reasoning_case,
            "research_synthesis": self._generate_research_synthesis_case,
            "policy_analysis": self._generate_policy_analysis_case,
            "system_analysis": self._generate_system_analysis_case
        }
        
        if category in templates:
            return templates[category]()
        else:
            return self._generate_generic_reasoning_case(category)
    
    def _generate_task_decomposition_case(self) -> Dict[str, Any]:
        """Generate task decomposition test case"""
        tasks = [
            "plan a wedding for 100 guests",
            "design a mobile app for fitness tracking",
            "organize a community fundraiser",
            "write a business plan for a startup",
            "create a study schedule for final exams"
        ]
        
        task = random.choice(tasks)
        
        return {
            "id": f"task_decomp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "category": "task_decomposition",
            "difficulty": "medium",
            "description": "Complex task requiring systematic decomposition",
            "task": f"Break down the following task into manageable steps: {task}",
            "pipeline_stages": ["task_analysis", "subtask_identification", "sequencing", "resource_allocation"],
            "reasoning_requirements": ["planning", "organization", "prioritization"],
            "success_criteria": "Comprehensive task breakdown with logical sequencing",
            "metadata": {
                "type": "planning_task",
                "requires_decomposition": True,
                "estimated_time_minutes": 12,
                "claims_based_approach_beneficial": True
            }
        }
    
    def _generate_context_compression_case(self) -> Dict[str, Any]:
        """Generate context compression test case"""
        topics = [
            "climate change causes and effects",
            "artificial intelligence history and development",
            "world war II economic impacts",
            "human digestive system processes",
            "solar system formation theories"
        ]
        
        topic = random.choice(topics)
        
        return {
            "id": f"context_comp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "category": "context_compression",
            "difficulty": "medium",
            "description": "Large context requiring efficient compression",
            "question": f"Given a detailed article about {topic}, extract and summarize the most critical information while maintaining accuracy and completeness.",
            "context_length_target": "800-1200 words",
            "compression_target": "50-70% reduction while preserving key information",
            "metadata": {
                "type": "context_compression",
                "requires_information_filtering": True,
                "estimated_time_minutes": 15,
                "claims_based_approach_beneficial": True
            }
        }
    
    def _generate_claims_reasoning_case(self) -> Dict[str, Any]:
        """Generate claims-based reasoning test case"""
        topics = [
            "universal basic income feasibility",
            "remote work productivity impact",
            "renewable energy transition timeline",
            "space exploration funding priorities",
            "education system reform approaches"
        ]
        
        topic = random.choice(topics)
        
        return {
            "id": f"claims_reason_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "category": "claims_reasoning",
            "difficulty": "hard",
            "description": "Complex reasoning requiring evidence-based claims",
            "question": f"Analyze the proposition: '{topic}' by developing supporting and opposing claims with evidence-based reasoning.",
            "claims_needed": ["supporting_claims", "opposing_claims", "evidence_evaluation", "conclusion"],
            "reasoning_requirements": ["critical_analysis", "evidence_synthesis", "balanced_perspective"],
            "metadata": {
                "type": "claims_reasoning",
                "requires_evidence_synthesis": True,
                "estimated_time_minutes": 20,
                "claims_based_approach_beneficial": True
            }
        }
    
    def _generate_research_synthesis_case(self) -> Dict[str, Any]:
        """Generate research synthesis test case"""
        topics = [
            "microlearning effectiveness in adult education",
            "mindfulness meditation impact on stress",
            "exercise and cognitive function correlation",
            "social media effects on adolescent development",
            "plant-based diets health outcomes"
        ]
        
        topic = random.choice(topics)
        
        return {
            "id": f"research_syn_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "category": "research_synthesis",
            "difficulty": "hard",
            "description": "Research synthesis across multiple studies",
            "question": f"Synthesize research findings on {topic} from multiple studies to identify patterns, contradictions, and gaps in the literature.",
            "synthesis_requirements": ["pattern_identification", "contradiction_analysis", "gap_identification"],
            "metadata": {
                "type": "research_synthesis",
                "requires_literature_analysis": True,
                "estimated_time_minutes": 25,
                "claims_based_approach_beneficial": True
            }
        }
    
    def _generate_policy_analysis_case(self) -> Dict[str, Any]:
        """Generate policy analysis test case"""
        policies = [
            "carbon tax implementation",
            "healthcare reform proposals",
            "immigration policy changes",
            "education funding allocation",
            "urban development planning"
        ]
        
        policy = random.choice(policies)
        
        return {
            "id": f"policy_anal_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "category": "policy_analysis",
            "difficulty": "hard",
            "description": "Complex policy analysis with multiple stakeholders",
            "question": f"Analyze the potential impacts and trade-offs of implementing {policy}. Consider economic, social, and environmental factors.",
            "analysis_factors": ["economic_impact", "social_consequences", "environmental_effects", "stakeholder_interests"],
            "metadata": {
                "type": "policy_analysis",
                "requires_multidimensional_analysis": True,
                "estimated_time_minutes": 22,
                "claims_based_approach_beneficial": True
            }
        }
    
    def _generate_system_analysis_case(self) -> Dict[str, Any]:
        """Generate system analysis test case"""
        systems = [
            "healthcare delivery system",
            "public transportation network",
            "food supply chain",
            "energy distribution grid",
            "waste management system"
        ]
        
        system = random.choice(systems)
        
        return {
            "id": f"system_anal_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "category": "system_analysis",
            "difficulty": "hard",
            "description": "Complex system analysis with interdependencies",
            "question": f"Analyze the {system} to identify key components, interdependencies, potential failure points, and improvement opportunities.",
            "analysis_aspects": ["component_identification", "dependency_mapping", "vulnerability_assessment", "optimization_opportunities"],
            "metadata": {
                "type": "system_analysis",
                "requires_systems_thinking": True,
                "estimated_time_minutes": 25,
                "claims_based_approach_beneficial": True
            }
        }
    
    def _generate_generic_reasoning_case(self, category: str) -> Dict[str, Any]:
        """Generate a generic reasoning case"""
        return {
            "id": f"generic_{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "category": category,
            "difficulty": "medium",
            "description": f"Generic reasoning task for {category}",
            "question": f"Provide a comprehensive analysis of a typical {category} problem scenario.",
            "metadata": {
                "type": "generic_reasoning",
                "estimated_time_minutes": 15,
                "claims_based_approach_beneficial": True
            }
        }
    
    def generate_test_suite(self, count_per_type: int = 10):
        """Generate a comprehensive test suite"""
        self.logger.info(f"Generating test suite with {count_per_type} cases per type")
        
        categories = [
            "complex_reasoning", "mathematical_reasoning", "evidence_evaluation",
            "task_decomposition", "context_compression", "claims_reasoning",
            "research_synthesis", "policy_analysis", "system_analysis"
        ]
        
        for category in categories:
            for i in range(count_per_type):
                if category == "complex_reasoning":
                    test_case = self.generate_logic_puzzle(i % 3)
                elif category == "mathematical_reasoning":
                    test_case = self._generate_algebra_problem()
                elif category == "evidence_evaluation":
                    test_case = self._generate_evidence_scenario()
                else:
                    test_case = self._generate_generic_case(category)
                
                # Save test case
                filename = f"{category}_{i+1:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                filepath = self.output_dir / filename
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(test_case, f, indent=2)
        
        self.logger.info(f"Generated {len(categories) * count_per_type} test cases")
        
        # Color relationships
        red_pos = None
        blue_pos = None
        for house_num, details in solution.items():
            if details["color"] == "red":
                red_pos = house_num
            elif details["color"] == "blue":
                blue_pos = house_num
        
        if red_pos and blue_pos and red_pos < blue_pos:
            clues.append("The red house is somewhere to the left of the blue house.")
        
        # Adjacent relationships
        for house_num, details in solution.items():
            if details["profession"] == "artist":
                artist_pos = house_num
                break
        
        for house_num, details in solution.items():
            if details["fruit"] == "apple":
                apple_pos = house_num
                break
        
        if abs(artist_pos - apple_pos) == 1:
            clues.append("The artist lives next to the person who likes apples.")
        
        return clues
    
    def generate_math_problem(self, template_id: int = 0) -> Dict[str, Any]:
        """Generate a math word problem"""
        template = self.math_problems[template_id]
        
        if template["template"] == "algebra_word_problem":
            return self._generate_algebra_problem()
        elif template["template"] == "geometry_problem":
            return self._generate_geometry_problem()
        elif template["template"] == "rate_problem":
            return self._generate_rate_problem()
    
    def _generate_algebra_problem(self) -> Dict[str, Any]:
        """Generate an algebra word problem"""
        # Generate parameters
        width = random.randint(50, 150)
        length_diff = random.randint(20, 80)
        area = width * (width + length_diff)
        
        question = f"A farmer has a rectangular field that is {length_diff} meters longer than it is wide. The farmer wants to divide the field into two equal rectangular sections by building a fence parallel to the width. If the total area of the field is {area:,} square meters and the fence costs ${random.randint(10, 25)} per meter, what is the total cost of the dividing fence?"
        
        # Calculate solution
        cost_per_meter = 15  # Default for consistency
        fence_length = width
        total_cost = fence_length * cost_per_meter
        
        return {
            "id": f"math_algebra_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "category": "mathematical_reasoning",
            "difficulty": "medium",
            "description": "Multi-step mathematical problem with algebraic reasoning",
            "question": question,
            "ground_truth": f"The total cost of the dividing fence is ${total_cost:,.2f}.\n\nSolution:\nLet width = w meters, then length = w + {length_diff} meters\nArea = w × (w + {length_diff}) = {area}\nw² + {length_diff}w - {area} = 0\nUsing quadratic formula: w = [{-length_diff} ± √({length_diff}² + 4×{area})]/2\nw = {width} meters\n\nDividing fence length = width = {width} meters\nCost = {width} × ${cost_per_meter} = ${total_cost:,.2f}",
            "expected_approach": "step_by_step",
            "metadata": {
                "type": "algebra",
                "requires_quadratic_formula": True,
                "multiple_steps": True,
                "unit_conversion": False,
                "estimated_time_minutes": 10,
                "claims_based_approach_beneficial": True
            }
        }
    
    def _generate_geometry_problem(self) -> Dict[str, Any]:
        """Generate a geometry problem"""
        # Create a problem about calculating area and perimeter
        length = random.randint(10, 50)
        width = random.randint(5, 30)
        area = length * width
        perimeter = 2 * (length + width)
        
        question = f"A rectangular garden has a length of {length} meters and a width of {width} meters. If you want to build a fence around the garden and cover the entire area with grass sod that costs $3 per square meter, what is the total cost for the sod and how many meters of fencing do you need?"
        
        return {
            "id": f"math_geometry_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "category": "mathematical_reasoning",
            "difficulty": "hard",
            "description": "Multi-step geometry problem with area and perimeter",
            "question": question,
            "ground_truth": f"The total cost for sod is ${area * 3:,.2f} and you need {perimeter} meters of fencing.\n\nSolution:\nArea = length × width = {length} × {width} = {area} square meters\nSod cost = {area} × $3 = ${area * 3:,.2f}\nPerimeter = 2 × (length + width) = 2 × ({length} + {width}) = {perimeter} meters",
            "expected_approach": "step_by_step",
            "metadata": {
                "type": "geometry",
                "requires_area_calculation": True,
                "requires_perimeter_calculation": True,
                "multiple_steps": True,
                "estimated_time_minutes": 12,
                "claims_based_approach_beneficial": True
            }
        }
    
    def _generate_rate_problem(self) -> Dict[str, Any]:
        """Generate a rate/time/distance problem"""
        # Create a problem about speed and time
        speed1 = random.randint(40, 80)
        speed2 = speed1 + random.randint(5, 20)
        distance = random.randint(200, 500)
        
        time1 = distance / speed1
        time2 = distance / speed2
        time_diff = abs(time1 - time2)
        
        question = f"Two cars are traveling to the same destination {distance} miles away. Car A travels at an average speed of {speed1} mph, while Car B travels at {speed2} mph. How much longer does the slower car take to reach the destination? Express your answer in hours and minutes."
        
        hours = int(time_diff)
        minutes = int((time_diff - hours) * 60)
        
        return {
            "id": f"math_rate_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "category": "mathematical_reasoning",
            "difficulty": "medium",
            "description": "Rate, time, and distance problem with unit conversion",
            "question": question,
            "ground_truth": f"The slower car takes {hours} hours and {minutes} minutes longer.\n\nSolution:\nTime for Car A = {distance} ÷ {speed1} = {time1:.2f} hours\nTime for Car B = {distance} ÷ {speed2} = {time2:.2f} hours\nTime difference = {time_diff:.2f} hours = {hours} hours and {minutes} minutes",
            "expected_approach": "step_by_step",
            "metadata": {
                "type": "rate_problem",
                "requires_speed_calculation": True,
                "requires_time_calculation": True,
                "requires_unit_conversion": True,
                "estimated_time_minutes": 10,
                "claims_based_approach_beneficial": True
            }
        }
    
    def _generate_family_puzzle(self) -> Dict[str, Any]:
        """Generate a family relationships logic puzzle"""
        family_members = ["father", "mother", "son", "daughter", "grandmother"]
        ages = [25, 30, 35, 40, 45]
        random.shuffle(ages)
        
        puzzle = {
            "id": f"family_puzzle_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "category": "complex_reasoning",
            "difficulty": "medium",
            "description": "Family relationships logic puzzle with ages",
            "question": f"A family has five members: {', '.join(family_members)}. Each member has a different age. Use these clues to determine each person's age:\n\n1. The father is older than the mother\n2. The grandmother is the oldest\n3. The son is younger than the daughter\n4. The mother is 35 years old\n5. The age difference between father and son is 10 years\n\nWhat is the daughter's age?",
            "ground_truth": f"The daughter is {ages[3]} years old.\n\nSolution:\n- Grandmother: {ages[4]} years old (oldest)\n- Father: {ages[2]} years old\n- Mother: 35 years old\n- Daughter: {ages[3]} years old\n- Son: {ages[0]} years old (youngest)",
            "expected_approach": "break_down_problem",
            "metadata": {
                "type": "logic_puzzle",
                "requires_deduction": True,
                "multiple_constraints": True,
                "estimated_time_minutes": 10,
                "claims_based_approach_beneficial": True
            }
        }
        
        return puzzle
    
    def _generate_science_context(self) -> Dict[str, Any]:
        """Generate scientific context passage"""
        context = "Climate change refers to long-term shifts in global temperatures and weather patterns. While climate change is a natural phenomenon that has occurred throughout Earth's history, the current warming trend is of particular significance because most of it is extremely likely (greater than 95% probability) to be the result of human activity since the mid-20th century and proceeding at a rate that is unprecedented over millennia.\n\nThe primary cause of current climate change is the accumulation of greenhouse gases in the atmosphere, particularly carbon dioxide (CO2), methane (CH4), and nitrous oxide (N2O). These gases trap heat from the sun that would otherwise escape into space, creating a greenhouse effect that warms the planet. Human activities such as burning fossil fuels for energy, deforestation, industrial processes, and agriculture have dramatically increased the concentration of these gases.\n\nThe effects of climate change are widespread and varied. Rising global temperatures cause polar ice sheets and glaciers to melt, leading to sea level rise that threatens coastal communities worldwide. Changes in temperature and precipitation patterns affect agriculture, water supplies, and ecosystems. More frequent and intense extreme weather events, including hurricanes, droughts, heat waves, and heavy rainfall, cause significant damage to infrastructure, economies, and human health.\n\nScientists use various methods to study climate change, including direct measurements of temperature, atmospheric composition, and sea level; analysis of ice cores, tree rings, and sediment layers that provide historical climate data; and computer models that simulate future climate scenarios under different conditions. These multiple lines of evidence consistently show that Earth's climate is warming and that human activities are the primary driver.\n\nInternational efforts to address climate change include the Paris Agreement, adopted in 2015, which aims to limit global warming to well below 2°C above pre-industrial levels. Countries have pledged to reduce their greenhouse gas emissions through various strategies, including transitioning to renewable energy sources, improving energy efficiency, protecting forests, and developing carbon capture technologies.\n\nThe impacts of climate change are not distributed equally. Developing countries, small island nations, and vulnerable populations often face the greatest risks despite contributing least to the problem. Addressing climate change requires both mitigation (reducing emissions) and adaptation (adjusting to changes that are already occurring), as well as international cooperation and equitable solutions."
        
        question = "Based on the text, what are the primary greenhouse gases contributing to climate change, and what are three major effects of climate change mentioned?"
        
        return {
            "id": f"context_science_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "category": "context_compression",
            "difficulty": "hard",
            "description": "Question answering with scientific context",
            "context": context,
            "question": question,
            "ground_truth": "The primary greenhouse gases contributing to climate change are carbon dioxide (CO2), methane (CH4), and nitrous oxide (N2O). Three major effects of climate change mentioned are: 1) Melting of polar ice sheets and glaciers leading to sea level rise that threatens coastal communities; 2) Changes in temperature and precipitation patterns that affect agriculture, water supplies, and ecosystems; and 3) More frequent and intense extreme weather events including hurricanes, droughts, heat waves, and heavy rainfall that cause damage to infrastructure, economies, and human health.",
            "expected_approach": "context_compression",
            "metadata": {
                "type": "science_comprehension",
                "context_length_words": 420,
                "requires_synthesis": True,
                "estimated_time_minutes": 10,
                "claims_based_approach_beneficial": True
            }
        }
    
    def _generate_technical_context(self) -> Dict[str, Any]:
        """Generate technical documentation context"""
        context = "Microservices architecture is an architectural style that structures an application as a collection of small, autonomous services modeled around a business domain. Each microservice is a self-contained unit that can be developed, deployed, and scaled independently. This approach contrasts with traditional monolithic architecture where all components are tightly integrated into a single unit.\n\nKey characteristics of microservices architecture include:\n\n1. Componentization via Services: Applications are composed of small, independently deployable services that communicate over a network using lightweight protocols such as HTTP/REST or messaging queues.\n\n2. Decentralized Governance: Each team can choose the best technology stack for their specific service needs, rather than being constrained to a single standardized technology across the entire application.\n\n3. Decentralized Data Management: Each microservice manages its own database, allowing for polyglot persistence where different services can use different database technologies optimized for their specific data needs.\n\n4. Infrastructure Automation: Continuous Integration and Continuous Deployment (CI/CD) pipelines are essential for managing the complexity of deploying numerous independent services.\n\n5. Evolutionary Design: Microservices architectures are designed to evolve over time, with services being added, removed, or modified as business requirements change.\n\n6. Organizational Alignment: Teams are typically organized around business capabilities rather than technical layers, promoting cross-functional collaboration and ownership.\n\nBenefits of microservices architecture include improved scalability, as individual services can be scaled independently based on demand; technology diversity, allowing teams to use the best tools for each job; fault isolation, where failures in one service don't necessarily bring down the entire application; and faster time-to-market due to independent development and deployment cycles.\n\nHowever, microservices also introduce complexity. Challenges include managing distributed systems, handling network latency and failures, ensuring data consistency across services, implementing effective monitoring and logging, and managing the operational overhead of multiple services.\n\nCommon patterns in microservices architecture include API Gateway for request routing, Service Discovery for locating services, Circuit Breaker for fault tolerance, and Event Sourcing for maintaining data consistency. Containerization technologies like Docker and orchestration platforms like Kubernetes have become essential tools for managing microservices deployments at scale."
        
        question = "According to the text, what are the six key characteristics of microservices architecture, and what are two main benefits and two main challenges mentioned?"
        
        return {
            "id": f"context_technical_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "category": "context_compression",
            "difficulty": "hard",
            "description": "Question answering with technical documentation",
            "context": context,
            "question": question,
            "ground_truth": "The six key characteristics of microservices architecture are: 1) Componentization via Services - small, independently deployable services communicating over networks; 2) Decentralized Governance - teams can choose best technology stack for each service; 3) Decentralized Data Management - each service manages its own database allowing polyglot persistence; 4) Infrastructure Automation - CI/CD pipelines essential for deployment; 5) Evolutionary Design - architectures designed to evolve over time; and 6) Organizational Alignment - teams organized around business capabilities.\n\nTwo main benefits mentioned are: improved scalability (individual services can be scaled independently) and faster time-to-market (due to independent development and deployment cycles).\n\nTwo main challenges mentioned are: managing distributed systems complexity (including network latency and failures) and ensuring data consistency across services.",
            "expected_approach": "context_compression",
            "metadata": {
                "type": "technical_documentation",
                "context_length_words": 380,
                "requires_synthesis": True,
                "estimated_time_minutes": 12,
                "claims_based_approach_beneficial": True
            }
        }
    
    def generate_context_qa(self, template_id: int = 0) -> Dict[str, Any]:
        """Generate a long context QA test case"""
        template = self.context_passages[template_id]
        
        if template["template"] == "historical_event":
            return self._generate_historical_context()
        elif template["template"] == "scientific_explanation":
            return self._generate_science_context()
        elif template["template"] == "technical_documentation":
            return self._generate_technical_context()
    
    def _generate_historical_context(self) -> Dict[str, Any]:
        """Generate historical context passage"""
        # Use the Renaissance example as base
        context = "The Renaissance was a period of cultural, artistic, political and economic rebirth following the Middle Ages in Europe. Generally described as taking place from the 14th to the 17th century, the Renaissance promoted the rediscovery of classical philosophy, literature and art. Some of the greatest thinkers, authors, statesmen, scientists and artists in human history thrived during this era, while global exploration opened up new lands and cultures to European commerce. The Renaissance is credited with bridging the gap between the Middle Ages and modern civilization.\n\nThe Renaissance began in Florence, Italy, in the 14th century. Various theories have been proposed to account for its origins and characteristics, focusing on a variety of factors including the social and civic peculiarities of Florence at the time: its political structure, the patronage of its dominant family, the Medici, and the migration of Greek scholars and texts to Italy following the Fall of Constantinople to the Ottoman Turks.\n\nThe Renaissance has a long and complex historiography, and, in line with general skepticism of discrete periodizations, there has been much debate among historians reacting to the 19th-century glorification of the Renaissance and individual culture heroes as Renaissance men. However, the beginning of the period – the early Renaissance of the 15th century – is comparatively well agreed upon and much less contentious.\n\nOther major centers included northern Italian city-states such as Venice, Genoa, Milan, and Bologna during the late Renaissance. The Italian Renaissance peaked in the mid-16th century as foreign invasions plunged the region into the turmoil of the Italian Wars. However, the ideas and ideals of the Renaissance endured and spread into the rest of Europe, setting off the Northern Renaissance, English Renaissance, and other national and localized movements, each with different characteristics and strengths.\n\nThe Renaissance saw many changes in culture, science, and technology. The printing press was invented, allowing books to be mass-produced for the first time. This led to a dramatic increase in literacy and the spread of new ideas. Artists developed new techniques such as linear perspective and chiaroscuro, creating more realistic and emotionally powerful works. Scientists such as Copernicus and Galileo challenged traditional views of the universe, laying the groundwork for the Scientific Revolution.\n\nIn art, the Renaissance is perhaps best known for its artistic developments and the contributions of Leonardo da Vinci and Michelangelo, who inspired the term Renaissance man. However, many other notable artists made significant contributions during this period, including Raphael, Donatello, Titian, and Dürer. Renaissance art is characterized by realism, expression of human emotion, and the use of classical themes and motifs.\n\nIn science, the Renaissance challenged medieval views of the world. Nicolaus Copernicus formulated a heliocentric model of the universe that placed the Sun rather than Earth at its center. Galileo Galilei improved the telescope and used it to make observations that supported Copernicus's theory. Andreas Vesalius revolutionized the study of anatomy with his detailed drawings of the human body.\n\nIn literature, humanist scholars sought to revive the study of classical Latin and Greek. Petrarch is often called the father of humanism. Other important writers include Boccaccio, Machiavelli, and Erasmus. The invention of the printing press by Johannes Gutenberg around 1440 made books more accessible and helped spread new ideas throughout Europe.\n\nThe Renaissance had a profound impact on modern civilization. It marked the transition from medieval to modern times and laid the foundation for many aspects of contemporary culture, science, and thought. The emphasis on individual achievement, the pursuit of knowledge, and the appreciation of classical beauty continue to influence Western culture today."
        
        question = "Based on the text, what were the four main areas of achievement during the Renaissance, and how did the printing press specifically contribute to the spread of Renaissance ideas?"
        
        return {
            "id": f"context_qa_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "category": "context_compression",
            "difficulty": "hard",
            "description": "Question answering with long context that requires compression",
            "context": context,
            "question": question,
            "ground_truth": "The four main areas of achievement during the Renaissance were: 1) Art, characterized by realism, expression of human emotion, and classical themes, with notable artists like Leonardo da Vinci, Michelangelo, Raphael, and Donatello developing techniques such as linear perspective and chiaroscuro; 2) Science, where thinkers like Copernicus and Galileo challenged traditional views, with Copernicus formulating the heliocentric model and Galileo making telescopic observations, while Vesalius revolutionized anatomy; 3) Literature, where humanist scholars revived classical Latin and Greek studies, with figures like Petrarch, Boccaccio, Machiavelli, and Erasmus; and 4) Technology, most notably the invention of the printing press by Johannes Gutenberg around 1440.\n\nThe printing press specifically contributed to the spread of Renaissance ideas by enabling the mass production of books for the first time, which led to a dramatic increase in literacy rates and allowed new ideas to spread rapidly throughout Europe, making knowledge more accessible to a broader population beyond just the scholarly elite.",
            "expected_approach": "context_compression",
            "metadata": {
                "type": "reading_comprehension",
                "context_length_words": 580,
                "requires_synthesis": True,
                "multiple_categories": True,
                "estimated_time_minutes": 12,
                "claims_based_approach_beneficial": True,
                "compression_ratio_target": 0.3
            }
        }
    
    def generate_evidence_evaluation(self) -> Dict[str, Any]:
        """Generate an evidence evaluation test case"""
        # Create conflicting evidence scenario
        return {
            "id": f"evidence_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "category": "claims_reasoning",
            "difficulty": "medium",
            "description": "Evaluate conflicting evidence and make reasoned judgment",
            "question": "A pharmaceutical company is testing a new drug for treating hypertension. They present the following evidence from their clinical trials. Evaluate the evidence and determine whether the drug should be approved for market use.\n\nEvidence presented:\n1. Study A (n=1000): 15% reduction in blood pressure, p=0.03, funded by the company\n2. Study B (n=500): 8% reduction in blood pressure, p=0.12, not statistically significant, funded by the company\n3. Study C (n=2000): 12% reduction in blood pressure, p=0.01, independent research\n4. Study D (n=300): 18% reduction in blood pressure, p=0.08, not statistically significant, independent research\n5. Side effects: 5% experienced mild headaches, 2% experienced dizziness\n6. Cost: 3x more expensive than existing treatments\n7. Long-term effects: Unknown (studies only 6 months duration)\n8. Mechanism of action: Well-understood and plausible",
            "ground_truth": "Based on the evidence, the drug should NOT be approved for market use at this time, though it shows promise and warrants further investigation.\n\nReasoning:\nPositive evidence:\n- Two studies (A and C) show statistically significant blood pressure reductions\n- Study C is independent and large (n=2000), lending credibility\n- Mechanism of action is well-understood\n- Side effects are relatively mild\n\nConcerns:\n- Mixed results across studies (B and D not statistically significant)\n- Potential funding bias in company-sponsored studies\n- Small effect size (8-18% reduction)\n- Significantly higher cost than existing treatments\n- Unknown long-term effects\n- Short study duration (6 months)\n\nRecommendation: Require larger, longer-term independent studies before approval. The current evidence is insufficient to justify the higher cost and unknown risks.",
            "expected_approach": "evidence_weighting",
            "metadata": {
                "type": "evidence_evaluation",
                "requires_critical_thinking": True,
                "conflicting_evidence": True,
                "risk_assessment": True,
                "estimated_time_minutes": 15,
                "claims_based_approach_beneficial": True
            }
        }
    
    def generate_planning_task(self) -> Dict[str, Any]:
        """Generate a complex planning task"""
        return {
            "id": f"planning_task_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "category": "task_decomposition",
            "difficulty": "medium",
            "description": "Complex planning task that benefits from decomposition",
            "task": "You are a project manager tasked with organizing a company-wide technology conference for 500 attendees. The conference will last 3 days and include keynote speakers, technical workshops, networking events, and a product showcase. Your budget is $150,000 and you have 6 months to plan. Create a comprehensive project plan that breaks this down into manageable phases and key deliverables.",
            "ground_truth": "A comprehensive project plan should include the following phases and key deliverables:\n\nPhase 1: Planning & Foundation (Months 1-2)\n- Define conference objectives and target audience\n- Establish budget breakdown and secure funding\n- Form planning committee with defined roles\n- Research and select venue (capacity 500+, 3-day availability)\n- Create initial timeline and milestone schedule\n- Develop risk management plan\n\nPhase 2: Content & Speakers (Months 2-4)\n- Identify and confirm keynote speakers (3-5 speakers)\n- Develop call for papers for technical workshops\n- Select and coordinate workshop presenters (20-30 sessions)\n- Plan product showcase logistics and exhibitor packages\n- Design conference agenda and schedule\n- Prepare speaker contracts and travel arrangements\n\nPhase 3: Logistics & Operations (Months 3-5)\n- Finalize venue contract and layout design\n- Arrange catering services for all meals and breaks\n- Set up registration system and payment processing\n- Plan audio/visual equipment and technical support\n- Coordinate transportation and accommodation options\n- Design conference materials (badges, programs, signage)\n- Arrange security and medical support\n\nPhase 4: Marketing & Registration (Months 2-6)\n- Create conference website and branding\n- Develop marketing campaign and promotional materials\n- Open early bird registration\n- Manage social media promotion and email campaigns\n- Track registration numbers and adjust capacity as needed\n- Send regular updates to registered attendees\n\nPhase 5: Execution & Follow-up (Month 6)\n- Final venue walkthrough and setup\n- Coordinate speaker and volunteer training\n- Manage on-site registration and check-in\n- Oversee all conference sessions and events\n- Collect feedback and evaluations\n- Process final payments and close out contracts\n- Prepare post-conference report and recommendations",
            "expected_approach": "decompose_and_organize",
            "metadata": {
                "type": "project_planning",
                "complexity_level": "medium",
                "time_constraint": "6 months",
                "budget_constraint": "$150,000",
                "stakeholder_count": "500+",
                "estimated_time_minutes": 20,
                "claims_based_approach_beneficial": True
            }
        }
    
    def generate_test_suite(self, count_per_type: int = 3) -> List[str]:
        """Generate a complete test suite"""
        generated_files = []
        
        # Generate different types of test cases
        for i in range(count_per_type):
            # Logic puzzles
            logic_case = self.generate_logic_puzzle(i % len(self.logic_puzzles))
            filename = f"{logic_case['id']}.json"
            with open(self.output_dir / filename, 'w') as f:
                json.dump(logic_case, f, indent=2)
            generated_files.append(filename)
            
            # Math problems
            math_case = self.generate_math_problem(i % len(self.math_problems))
            filename = f"{math_case['id']}.json"
            with open(self.output_dir / filename, 'w') as f:
                json.dump(math_case, f, indent=2)
            generated_files.append(filename)
            
            # Context QA
            context_case = self.generate_context_qa(i % len(self.context_passages))
            filename = f"{context_case['id']}.json"
            with open(self.output_dir / filename, 'w') as f:
                json.dump(context_case, f, indent=2)
            generated_files.append(filename)
            
            # Evidence evaluation
            evidence_case = self.generate_evidence_evaluation()
            filename = f"{evidence_case['id']}.json"
            with open(self.output_dir / filename, 'w') as f:
                json.dump(evidence_case, f, indent=2)
            generated_files.append(filename)
            
            # Planning tasks
            planning_case = self.generate_planning_task()
            filename = f"{planning_case['id']}.json"
            with open(self.output_dir / filename, 'w') as f:
                json.dump(planning_case, f, indent=2)
            generated_files.append(filename)
        
        return generated_files


def main():
    """Generate test cases"""
    generator = TestCaseGenerator()
    
    print("Generating test cases...")
    generated_files = generator.generate_test_suite(count_per_type=2)
    
    print(f"Generated {len(generated_files)} test case files:")
    for filename in generated_files:
        print(f"  - {filename}")
    
    print(f"\nTest cases saved to: {generator.output_dir}")


if __name__ == "__main__":
    main()