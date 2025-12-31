"""
Backward Compatibility Layer for JSON Frontmatter Migration

Ensures smooth transition from legacy text-based parsing to
JSON frontmatter format while maintaining all existing functionality.
"""

import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from src.core.models import Claim, ClaimType, ClaimState
from src.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class CompatibilityMetrics:
    """Metrics for monitoring backward compatibility"""
    total_parses: int = 0
    json_frontmatter_success: int = 0
    legacy_format_success: int = 0
    fallback_success: int = 0
    migration_failures: int = 0
    avg_processing_time: float = 0.0

class BackwardCompatibilityManager:
    """
    Manages backward compatibility during JSON frontmatter migration.
    
    Features:
    - Automatic format detection
    - Graceful fallback to legacy parsers
    - Performance monitoring
    - Migration metrics tracking
    """
    
    def __init__(self):
        self.metrics = CompatibilityMetrics()
        self.legacy_parser = None
        self.json_parser = None
        
        # Initialize parsers lazily
        self._initialize_parsers()
    
    def _initialize_parsers(self):
        """Initialize parsers on demand"""
        try:
            from .json_frontmatter_parser import get_json_frontmatter_parser
            self.json_parser = get_json_frontmatter_parser()
        except ImportError as e:
            logger.warning(f"JSON frontmatter parser not available: {e}")
        
        try:
            from .unified_claim_parser import get_unified_parser
            self.legacy_parser = get_unified_parser()
        except ImportError as e:
            logger.warning(f"Legacy parser not available: {e}")
    
    def parse_with_compatibility(
        self, 
        response_text: str,
        prefer_json: bool = True,
        enable_fallback: bool = True
    ) -> List[Claim]:
        """
        Parse response with full backward compatibility support.
        
        Args:
            response_text: Raw LLM response text
            prefer_json: Whether to prefer JSON frontmatter parsing
            enable_fallback: Whether to enable fallback to legacy formats
            
        Returns:
            List of Claim objects
        """
        start_time = time.time()
        self.metrics.total_parses += 1
        
        claims = []
        parse_method = "unknown"
        
        # Try JSON frontmatter first if preferred
        if prefer_json and self.json_parser:
            try:
                result = self.json_parser.parse_response(response_text)
                if result.success:
                    claims = result.claims
                    parse_method = "json_frontmatter"
                    self.metrics.json_frontmatter_success += 1
                    logger.info(f"Successfully parsed {len(claims)} claims using JSON frontmatter")
                else:
                    # JSON parsing failed, try fallback if enabled
                    if enable_fallback:
                        claims, parse_method = self._try_legacy_parsing(response_text)
                    else:
                        self.metrics.migration_failures += 1
                        logger.warning("JSON frontmatter parsing failed and fallback disabled")
                        
            except Exception as e:
                logger.error(f"JSON frontmatter parser error: {e}")
                if enable_fallback:
                    claims, parse_method = self._try_legacy_parsing(response_text)
                else:
                    self.metrics.migration_failures += 1
        
        # Direct legacy parsing if JSON not preferred
        elif self.legacy_parser:
            claims, parse_method = self._try_legacy_parsing(response_text)
        
        # Update processing time
        processing_time = time.time() - start_time
        self._update_processing_time(processing_time)
        
        # Log compatibility metrics
        self._log_compatibility_metrics(parse_method, len(claims))
        
        return claims
    
    def _try_legacy_parsing(self, response_text: str) -> tuple[List[Claim], str]:
        """Try legacy parsing methods"""
        if not self.legacy_parser:
            return [], "no_legacy_parser"
        
        try:
            claims = self.legacy_parser.parse_claims_from_response(response_text)
            if claims:
                self.metrics.legacy_format_success += 1
                return claims, "legacy_unified"
            else:
                self.metrics.fallback_success += 1
                return [], "legacy_failed"
                
        except Exception as e:
            logger.error(f"Legacy parsing failed: {e}")
            self.metrics.fallback_success += 1
            return [], "legacy_error"
    
    def _update_processing_time(self, processing_time: float):
        """Update average processing time"""
        total = self.metrics.total_parses
        current_avg = self.metrics.avg_processing_time
        self.metrics.avg_processing_time = (current_avg * (total - 1) + processing_time) / total
    
    def _log_compatibility_metrics(self, parse_method: str, claim_count: int):
        """Log compatibility metrics"""
        logger.debug(f"Parse method: {parse_method}, claims: {claim_count}")
        
        # Log migration progress
        if self.metrics.total_parses % 10 == 0:  # Log every 10 parses
            json_success_rate = (self.metrics.json_frontmatter_success / self.metrics.total_parses) * 100
            legacy_success_rate = (self.metrics.legacy_format_success / self.metrics.total_parses) * 100
            
            logger.info(f"Migration progress: JSON={json_success_rate:.1f}%, Legacy={legacy_success_rate:.1f}%")
            
            # Warn if migration progress is slow
            if json_success_rate < 50 and self.metrics.total_parses > 50:
                logger.warning("Slow JSON frontmatter adoption detected. Consider updating LLM prompts.")
    
    def get_compatibility_metrics(self) -> Dict[str, Any]:
        """Get comprehensive compatibility metrics"""
        total = self.metrics.total_parses
        
        if total == 0:
            return {
                "total_parses": 0,
                "json_adoption_rate": 0.0,
                "legacy_reliance_rate": 0.0,
                "migration_success_rate": 100.0,
                "avg_processing_time": 0.0
            }
        
        json_adoption_rate = (self.metrics.json_frontmatter_success / total) * 100
        legacy_reliance_rate = (self.metrics.legacy_format_success / total) * 100
        migration_success_rate = ((self.metrics.json_frontmatter_success + self.metrics.legacy_format_success) / total) * 100
        
        return {
            "total_parses": total,
            "json_frontmatter_success": self.metrics.json_frontmatter_success,
            "legacy_format_success": self.metrics.legacy_format_success,
            "fallback_success": self.metrics.fallback_success,
            "migration_failures": self.metrics.migration_failures,
            "json_adoption_rate": json_adoption_rate,
            "legacy_reliance_rate": legacy_reliance_rate,
            "migration_success_rate": migration_success_rate,
            "avg_processing_time": self.metrics.avg_processing_time,
            "migration_health": self._assess_migration_health()
        }
    
    def _assess_migration_health(self) -> str:
        """Assess overall migration health"""
        total = self.metrics.total_parses
        
        if total < 10:
            return "insufficient_data"
        
        json_rate = (self.metrics.json_frontmatter_success / total) * 100
        
        if json_rate >= 90:
            return "excellent"
        elif json_rate >= 75:
            return "good"
        elif json_rate >= 50:
            return "fair"
        elif json_rate >= 25:
            return "poor"
        else:
            return "critical"
    
    def reset_metrics(self):
        """Reset compatibility metrics"""
        self.metrics = CompatibilityMetrics()
        logger.info("Compatibility metrics reset")
    
    def generate_migration_report(self) -> str:
        """Generate detailed migration report"""
        metrics = self.get_compatibility_metrics()
        
        report = [
            "# JSON Frontmatter Migration Report",
            f"Generated: {datetime.utcnow().isoformat()}",
            "",
            "## Summary Metrics",
            f"- Total Parses: {metrics['total_parses']}",
            f"- JSON Frontmatter Success: {metrics['json_frontmatter_success']} ({metrics['json_adoption_rate']:.1f}%)",
            f"- Legacy Format Success: {metrics['legacy_format_success']} ({metrics['legacy_reliance_rate']:.1f}%)",
            f"- Migration Failures: {metrics['migration_failures']}",
            f"- Overall Success Rate: {metrics['migration_success_rate']:.1f}%",
            f"- Average Processing Time: {metrics['avg_processing_time']:.3f}s",
            f"- Migration Health: {metrics['migration_health'].upper()}",
            "",
            "## Recommendations"
        ]
        
        # Add specific recommendations based on metrics
        health = metrics['migration_health']
        if health in ['critical', 'poor']:
            report.extend([
                "🚨 URGENT: JSON frontmatter adoption is very low",
                "- Review and update LLM prompts to use JSON format",
                "- Provide examples of JSON frontmatter in prompts",
                "- Consider temporary forcing of JSON format for testing"
            ])
        elif health == 'fair':
            report.extend([
                "⚠️ MODERATE: JSON frontmatter adoption needs improvement",
                "- Continue refining LLM prompts",
                "- Monitor adoption rates closely",
                "- Provide additional JSON format examples"
            ])
        elif health == 'good':
            report.extend([
                "✅ GOOD: JSON frontmatter adoption is progressing well",
                "- Continue current migration strategy",
                "- Monitor for any regressions"
            ])
        elif health == 'excellent':
            report.extend([
                "🎉 EXCELLENT: JSON frontmatter migration is highly successful",
                "- Consider deprecating legacy parsers",
                "- Plan for full JSON frontmatter deployment"
            ])
        
        # Add processing time recommendations
        if metrics['avg_processing_time'] > 1.0:
            report.extend([
                "",
                "## Performance Concerns",
                f"- Average processing time ({metrics['avg_processing_time']:.3f}s) is high",
                "- Consider optimizing parsing logic",
                "- Monitor for performance bottlenecks"
            ])
        
        return "\n".join(report)
    
    def create_hybrid_parser(self) -> 'HybridClaimParser':
        """Create a hybrid parser that combines both approaches"""
        return HybridClaimParser(self.json_parser, self.legacy_parser)

class HybridClaimParser:
    """
    Hybrid parser that combines JSON frontmatter and legacy parsing.
    Provides intelligent format detection and optimal parsing strategy.
    """
    
    def __init__(self, json_parser=None, legacy_parser=None):
        self.json_parser = json_parser
        self.legacy_parser = legacy_parser
        self.format_detector = FormatDetector()
    
    def parse(self, response_text: str) -> List[Claim]:
        """
        Parse using optimal strategy based on content analysis.
        
        Args:
            response_text: Raw LLM response text
            
        Returns:
            List of Claim objects
        """
        # Detect format
        detected_format = self.format_detector.detect_format(response_text)
        
        # Choose optimal parsing strategy
        if detected_format == 'json_frontmatter' and self.json_parser:
            return self._parse_with_json(response_text)
        elif detected_format in ['bracket', 'xml', 'structured'] and self.legacy_parser:
            return self._parse_with_legacy(response_text)
        else:
            # Try both and take best result
            return self._parse_with_both_strategies(response_text)
    
    def _parse_with_json(self, response_text: str) -> List[Claim]:
        """Parse using JSON frontmatter parser"""
        try:
            result = self.json_parser.parse_response(response_text)
            return result.claims if result.success else []
        except Exception as e:
            logger.error(f"JSON parsing failed in hybrid mode: {e}")
            return []
    
    def _parse_with_legacy(self, response_text: str) -> List[Claim]:
        """Parse using legacy parser"""
        try:
            return self.legacy_parser.parse_claims_from_response(response_text)
        except Exception as e:
            logger.error(f"Legacy parsing failed in hybrid mode: {e}")
            return []
    
    def _parse_with_both_strategies(self, response_text: str) -> List[Claim]:
        """Try both strategies and return best result"""
        json_claims = []
        legacy_claims = []
        
        # Try JSON parsing
        if self.json_parser:
            try:
                result = self.json_parser.parse_response(response_text)
                if result.success:
                    json_claims = result.claims
            except Exception:
                pass
        
        # Try legacy parsing
        if self.legacy_parser:
            try:
                legacy_claims = self.legacy_parser.parse_claims_from_response(response_text)
            except Exception:
                pass
        
        # Choose best result
        if len(json_claims) > len(legacy_claims):
            logger.debug("Hybrid parser selected JSON result")
            return json_claims
        elif len(legacy_claims) > 0:
            logger.debug("Hybrid parser selected legacy result")
            return legacy_claims
        else:
            logger.warning("Hybrid parser: both strategies failed")
            return []

class FormatDetector:
    """Intelligent format detection for LLM responses"""
    
    def detect_format(self, response_text: str) -> str:
        """
        Detect the format of LLM response text.
        
        Returns:
            Format type: 'json_frontmatter', 'bracket', 'xml', 'structured', 'unknown'
        """
        text = response_text.strip()
        
        # Check for JSON frontmatter
        if text.startswith('---') and '\n---' in text:
            try:
                import json
                # Extract content between delimiters
                parts = text.split('---')
                if len(parts) >= 3:
                    json_part = parts[1].strip()
                    json.loads(json_part)  # Test if valid JSON
                    return 'json_frontmatter'
            except (json.JSONDecodeError, IndexError):
                pass
        
        # Check for bracket format
        import re
        bracket_pattern = r'\[c\d+\s*\|\s*[^|]+\s*\|\s*/\s*[0-9.]+\s*\]'
        if re.search(bracket_pattern, text):
            return 'bracket'
        
        # Check for XML format
        xml_pattern = r'<claim\s+[^>]*>.*?</claim>'
        if re.search(xml_pattern, text, re.IGNORECASE | re.DOTALL):
            return 'xml'
        
        # Check for structured format
        structured_pattern = r'Claim:\s*"[^"]+"\s*Confidence:\s*[0-9.]+'
        if re.search(structured_pattern, text, re.IGNORECASE):
            return 'structured'
        
        return 'unknown'

# Global compatibility manager instance
_compatibility_manager = None

def get_compatibility_manager() -> BackwardCompatibilityManager:
    """Get or create the global compatibility manager"""
    global _compatibility_manager
    if _compatibility_manager is None:
        _compatibility_manager = BackwardCompatibilityManager()
    return _compatibility_manager

def parse_with_backward_compatibility(
    response_text: str,
    prefer_json: bool = True,
    enable_fallback: bool = True
) -> List[Claim]:
    """
    Convenience function for backward-compatible parsing.
    
    Args:
        response_text: Raw LLM response text
        prefer_json: Whether to prefer JSON frontmatter parsing
        enable_fallback: Whether to enable fallback to legacy formats
        
    Returns:
        List of Claim objects
    """
    manager = get_compatibility_manager()
    return manager.parse_with_compatibility(response_text, prefer_json, enable_fallback)

def get_migration_metrics() -> Dict[str, Any]:
    """Get migration metrics"""
    manager = get_compatibility_manager()
    return manager.get_compatibility_metrics()

def generate_migration_report() -> str:
    """Generate migration report"""
    manager = get_compatibility_manager()
    return manager.generate_migration_report()