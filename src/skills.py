"""
Simplified Skill Templates for Conjecture
Basic 4-step process guidance and skill matching
"""

from typing import Dict, List


class SkillManager:
    """Simplified skill management with basic templates"""

    def __init__(self):
        self.skills = self._initialize_skills()

    def _initialize_skills(self) -> Dict[str, Dict[str, any]]:
        """Initialize available skills with 4-step templates"""
        return {
            "research": {
                "name": "Research",
                "description": "Find and analyze information about a topic",
                "steps": [
                    "Search for relevant information using WebSearch",
                    "Read and analyze key sources with ReadFiles", 
                    "Identify main concepts and relationships",
                    "Create structured claims from findings"
                ]
            },
            "code": {
                "name": "Code Development",
                "description": "Write and improve code",
                "steps": [
                    "Analyze requirements and existing code with ReadFiles",
                    "Write or modify code using WriteCodeFile", 
                    "Test the implementation and verify functionality",
                    "Document the changes and create examples"
                ]
            },
            "test": {
                "name": "Testing",
                "description": "Create and run tests for validation",
                "steps": [
                    "Identify test requirements and edge cases",
                    "Write comprehensive test cases using WriteCodeFile",
                    "Execute tests and analyze results", 
                    "Create claims about test coverage and findings"
                ]
            },
            "evaluate": {
                "name": "Evaluation",
                "description": "Assess and validate claims or implementations",
                "steps": [
                    "Gather evidence and data for evaluation",
                    "Apply criteria and metrics for assessment",
                    "Analyze results and identify patterns",
                    "Create conclusions and recommendations"
                ]
            }
        }

    def get_matching_skills(self, query: str) -> List[str]:
        """Simple keyword-based skill matching"""
        query_lower = query.lower()
        matching_skills = []

        keyword_to_skill = {
            "research": ["research", "find", "search", "analyze", "study", "investigate"],
            "code": ["code", "write", "implement", "program", "develop", "create file"],
            "test": ["test", "validate", "verify", "check", "run", "execute"],
            "evaluate": ["evaluate", "assess", "compare", "review", "judge", "measure"]
        }

        for skill, keywords in keyword_to_skill.items():
            if any(keyword in query_lower for keyword in keywords):
                matching_skills.append(skill)

        # Default to first skill if no matches
        if not matching_skills:
            matching_skills = list(self.skills.keys())[:1]

        return matching_skills

    def get_skill_template(self, skill_name: str) -> Dict[str, any]:
        """Get skill template by name"""
        return self.skills.get(skill_name, self.skills["research"])

    def get_all_skills(self) -> Dict[str, Dict[str, any]]:
        """Get all available skills"""
        return self.skills

    def format_skill_prompt(self, skill_name: str) -> str:
        """Format skill as prompt template"""
        skill = self.get_skill_template(skill_name)
        
        prompt = f"""Skill: {skill['name']}
Description: {skill['description']}

Steps to follow:
"""
        for i, step in enumerate(skill["steps"], 1):
            prompt += f"{i}. {step}\n"
        
        prompt += "\nUse available tools to complete each step. Create claims for key findings."
        return prompt


def get_matching_skills(query: str) -> List[str]:
    """Convenience function to get matching skills"""
    sm = SkillManager()
    return sm.get_matching_skills(query)


def get_skill_template(skill_name: str) -> Dict[str, any]:
    """Convenience function to get skill template"""
    sm = SkillManager()
    return sm.get_skill_template(skill_name)


if __name__ == "__main__":
    print("🧪 Testing Skill Manager")
    print("=" * 30)
    
    sm = SkillManager()
    
    # Test skill matching
    queries = [
        "Research machine learning algorithms",
        "Write a Python script to process data",
        "Test the new authentication system", 
        "Evaluate the performance improvements"
    ]
    
    for query in queries:
        skills = sm.get_matching_skills(query)
        print(f"✅ Query: '{query}' -> Skills: {skills}")
        
        if skills:
            prompt = sm.format_skill_prompt(skills[0])
            print(f"   Template: {prompt[:100]}...")
    
    # Test all skills
    all_skills = sm.get_all_skills()
    print(f"✅ Total skills available: {len(all_skills)}")
    
    print("🎉 Skill Manager tests passed!")