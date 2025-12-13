"""
Statistics Generation Module for Conjecture Project

Provides real-time statistics and analysis of:
- Project metrics and implementation state
- Test results and coverage
- Benchmark scores and performance
- System configuration status
- Infrastructure health

Main entry point: src.stats.main.generate_statistics()
Output: ./STATS.yaml
"""

from .main import ProjectStatsGenerator

def generate_statistics() -> dict:
    """Generate complete project statistics"""
    generator = ProjectStatsGenerator()
    return generator.generate_all_stats()

__all__ = ['ProjectStatsGenerator', 'generate_statistics']