"""
Skills Management System for Conjecture.

This module provides SkillManager for managing skills in the Conjecture system.
"""

from typing import Dict, Any, List, Optional
import re

class SkillManager:
    """Simple skill manager for backward compatibility with tests"""

    def __init__(self):
        self.skills = {}
        self.skill_registry = {}
        # Initialize with default skills for backward compatibility
        self._initialize_default_skills()

    def _initialize_default_skills(self):
        """Initialize default skills for backward compatibility with tests"""
        default_skills = {
            "research": {
                "name": "research",
                "description": "Research and gather information on various topics",
                "steps": [
                    "Analyze the research topic and identify key areas",
                    "Gather relevant information from multiple sources",
                    "Synthesize findings into coherent insights",
                    "Present research results with proper citations"
                ]
            },
            "code": {
                "name": "code",
                "description": "Write, review, and optimize code",
                "steps": [
                    "Understand requirements and constraints",
                    "Design solution architecture",
                    "Implement clean, efficient code",
                    "Test and validate implementation"
                ]
            },
            "test": {
                "name": "test",
                "description": "Test and validate system functionality",
                "steps": [
                    "Identify test requirements and scenarios",
                    "Design comprehensive test cases",
                    "Execute tests and collect results",
                    "Analyze outcomes and report findings"
                ]
            },
            "evaluate": {
                "name": "evaluate",
                "description": "Evaluate system performance and quality",
                "steps": [
                    "Define evaluation criteria and metrics",
                    "Collect performance and quality data",
                    "Analyze results against benchmarks",
                    "Generate evaluation report with recommendations"
                ]
            }
        }
        
        for skill_name, skill_data in default_skills.items():
            self.skills[skill_name] = skill_data
            self.skill_registry[skill_name] = skill_data

    def register_skill(self, name: str, skill_func):
        """Register a skill function"""
        self.skills[name] = skill_func
        self.skill_registry[name] = {
            "function": skill_func,
            "name": name,
            "description": getattr(skill_func, "__doc__", "No description")
        }

    def get_skill(self, name: str):
        """Get a skill by name"""
        return self.skills.get(name)

    def list_skills(self) -> List[str]:
        """List all available skills"""
        return list(self.skills.keys())

    def get_skill_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a skill"""
        return self.skill_registry.get(name)

    def execute_skill(self, name: str, *args, **kwargs) -> Any:
        """Execute a skill by name"""
        skill = self.get_skill(name)
        if skill is None:
            raise ValueError(f"Skill '{name}' not found")
        return skill(*args, **kwargs)

    def get_matching_skills(self, query: str) -> List[str]:
        """Get skills that match the given query"""
        query_lower = query.lower()
        matching_skills = []
        
        for skill_name in self.skills.keys():
            # Check if skill name matches query
            if skill_name in query_lower:
                matching_skills.append(skill_name)
                continue
            
            # Check if skill description matches query
            skill_info = self.get_skill_info(skill_name)
            if skill_info and "description" in skill_info:
                description = skill_info["description"].lower()
                if any(word in description for word in query_lower.split()):
                    matching_skills.append(skill_name)
        
        return matching_skills

    def get_all_skills(self) -> Dict[str, Dict[str, Any]]:
        """Get all available skills with their details"""
        return self.skill_registry.copy()

    def format_skill_prompt(self, skill_name: str) -> str:
        """Format a skill into a prompt for LLM consumption"""
        skill_info = self.get_skill_info(skill_name)
        if not skill_info:
            return f"Skill '{skill_name}' not found"
        
        prompt_parts = [
            f"Skill: {skill_info['name']}",
            f"Description: {skill_info['description']}"
        ]
        
        if "steps" in skill_info:
            prompt_parts.append("Steps:")
            for i, step in enumerate(skill_info["steps"], 1):
                prompt_parts.append(f"  {i}. {step}")
        
        return "\n".join(prompt_parts)

__all__ = ["SkillManager"]