"""
Skills Management System for Conjecture.

This module provides SkillManager for managing skills in the Conjecture system.
"""

from typing import Dict, Any, List, Optional


class SkillManager:
    """Simple skill manager for backward compatibility with tests"""

    def __init__(self):
        self.skills = {}
        self.skill_registry = {}

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


__all__ = ["SkillManager"]