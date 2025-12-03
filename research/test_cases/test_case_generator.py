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