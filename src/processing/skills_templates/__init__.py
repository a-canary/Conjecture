"""
Basic Skills Templates for Agent Harness
Research, WriteCode, TestCode, and EndClaimEval skill templates
"""

from .skill_template_manager import SkillTemplateManager
from .research_skill import ResearchSkillTemplate
from .writecode_skill import WriteCodeSkillTemplate
from .testcode_skill import TestCodeSkillTemplate
from .endclaimeval_skill import EndClaimEvalSkillTemplate

__all__ = [
    'SkillTemplateManager',
    'ResearchSkillTemplate',
    'WriteCodeSkillTemplate', 
    'TestCodeSkillTemplate',
    'EndClaimEvalSkillTemplate'
]