#!/usr/bin/env python3
"""
Auto Backend for Conjecture CLI
Intelligently selects the best backend based on configuration and context
"""

import os
import sys
from typing import List, Optional, Dict, Any
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from .local_backend import LocalBackend
from .cloud_backend import CloudBackend
from .hybrid_backend import HybridBackend
from ..base_cli import BaseCLI, BackendNotAvailableError
from config.unified_validator import validate_config, get_primary_provider


class AutoBackend(BaseCLI):
    """Auto-detecting backend that selects the optimal configuration."""

    def __init__(self):
        super().__init__("conjecture-auto", "Conjecture CLI with Auto-Detection")
        self.backends = {
            "local": LocalBackend(),
            "cloud": CloudBackend(),
            "hybrid": HybridBackend()
        }
        self.selected_backend = None
        self.detection_result = None

    def _run_detection(self) -> Dict[str, Any]:
        """Run comprehensive backend detection."""
        if self.detection_result:
            return self.detection_result

        detection = {
            "timestamp": "2025-11-11T12:00:00Z",
            "available_backends": {},
            "provider_info": None,
            "network_status": "auto",
            "performance_metrics": {},
            "recommendations": {}
        }

        # Check each backend
        for name, backend in self.backends.items():
            is_available = backend.is_available()
            detection["available_backends"][name] = is_available
            
            if is_available:
                backend_info = backend.get_backend_info()
                detection["provider_info"] = backend_info
                detection["performance_metrics"][name] = {
                    "response_time": "fast" if name == "local" else "medium",
                    "cost": "free" if name == "local" else "pay_per_use",
                    "features": self._get_backend_features(name),
                    "privacy": "high" if name == "local" else "medium",
                    "offline_capable": name == "local"
                }

        # Generate recommendations
        detection["recommendations"] = self._generate_recommendations(detection)
        
        self.detection_result = detection
        return detection

    def _get_backend_features(self, backend_name: str) -> List[str]:
        """Get feature list for a backend."""
        features = {
            "local": ["semantic_search", "local_analysis", "offline_capability", "high_privacy"],
            "cloud": ["advanced_analysis", "fact_checking", "web_search", "multiple_models"],
            "hybrid": ["auto_fallback", "cross_platform", "optimized_performance", "flexible_modes"]
        }
        return features.get(backend_name, [])

    def _generate_recommendations(self, detection: Dict[str, Any]) -> Dict[str, Any]:
        """Generate backend recommendations based on detection."""
        available = detection["available_backends"]
        recommendations = {}

        # Primary recommendation
        if available["hybrid"]:
            recommendations["primary"] = "hybrid"
            recommendations["reason"] = "Best of both worlds with automatic optimization"
        elif available["local"]:
            recommendations["primary"] = "local"
            recommendations["reason"] = "Fast, private, and offline capable"
        elif available["cloud"]:
            recommendations["primary"] = "cloud"
            recommendations["reason"] = "Advanced features and powerful models"
        else:
            recommendations["primary"] = None
            recommendations["reason"] = "No backends properly configured"

        # Use case recommendations
        recommendations["use_cases"] = {
            "privacy_focused": available["local"],
            "power_analysis": available["cloud"],
            "balanced_performance": available["hybrid"],
            "offline_first": available["local"]
        }

        return recommendations

    def _select_and_initialize_backend(self, operation: str = "create") -> BaseCLI:
        """Select and initialize the best backend for the operation."""
        detection = self._run_detection()
        primary = detection["recommendations"]["primary"]

        if not primary:
            raise BackendNotAvailableError("No backends are available. Please configure at least one provider.")

        # Use hybrid if available and operation-sensitive
        if primary == "hybrid" and isinstance(self.backends["hybrid"], HybridBackend):
            # Set optimal mode for operation
            if operation == "analyze":
                self.backends["hybrid"].set_preferred_mode("cloud")
            elif operation == "search":
                self.backends["hybrid"].set_preferred_mode("local")
            
            self.selected_backend = self.backends["hybrid"]
        else:
            self.selected_backend = self.backends[primary]

        return self.selected_backend

    def is_available(self) -> bool:
        """Check if any backend is available."""
        detection = self._run_detection()
        return any(detection["available_backends"].values())

    def get_detection_report(self) -> Dict[str, Any]:
        """Get comprehensive backend detection report."""
        detection = self._run_detection()
        
        # Add current selection info
        detection["current_selection"] = {
            "backend": self.selected_backend._get_backend_type() if self.selected_backend else None,
            "reason": detection["recommendations"].get("reason", "Not selected")
        }

        return detection

    def create_claim(self, content: str, confidence: float, user: str, analyze: bool = False, **kwargs) -> str:
        """Create a claim using the auto-selected backend."""
        backend = self._select_and_initialize_backend("create")
        
        self.console.print(f"[blue]🤖 Auto-detected {backend._get_backend_type()} backend for claim creation[/blue]")
        
        return backend.create_claim(content, confidence, user, analyze, **kwargs)

    def get_claim(self, claim_id: str) -> Optional[dict]:
        """Get a claim by ID."""
        backend = self._select_and_initialize_backend("get")
        return backend.get_claim(claim_id)

    def search_claims(self, query: str, limit: int = 10, **kwargs) -> List[dict]:
        """Search claims using the auto-selected backend."""
        backend = self._select_and_initialize_backend("search")
        
        self.console.print(f"[blue]🔍 Auto-detected {backend._get_backend_type()} backend for search[/blue]")
        
        return backend.search_claims(query, limit, **kwargs)

    def analyze_claim(self, claim_id: str, **kwargs) -> Dict[str, Any]:
        """Analyze a claim using the auto-selected backend."""
        backend = self._select_and_initialize_backend("analyze")
        
        self.console.print(f"[blue]🧠 Auto-detected {backend._get_backend_type()} backend for analysis[/blue]")
        
        analysis = backend.analyze_claim(claim_id, **kwargs)
        analysis["auto_detected"] = True
        analysis["backend_selection"] = backend._get_backend_type()
        
        return analysis

    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the auto-detected backend."""
        if not self.selected_backend:
            self._select_and_initialize_backend()
        
        base_info = super().get_backend_info()
        detection = self._run_detection()
        
        base_info.update({
            "auto_detection": {
                "available_backends": detection["available_backends"],
                "recommendation": detection["recommendations"]["primary"],
                "reason": detection["recommendations"].get("reason", ""),
                "detection_timestamp": detection["timestamp"]
            }
        })

        return base_info

    def reconfigure_detection(self, force_network_check: bool = False) -> Dict[str, Any]:
        """Re-run backend detection with optional force check."""
        self.detection_result = None  # Reset detection
        self.selected_backend = None  # Reset selection
        
        if force_network_check:
            self.console.print("[blue]🌐 Running forced network and provider detection...[/blue]")
        
        detection = self._run_detection()
        
        # Display detection results
        self.console.print("[bold]🔍 Backend Detection Results[/bold]")
        for name, available in detection["available_backends"].items():
            status = "✅" if available else "❌"
            self.console.print(f"  {status} {name.title()} Backend")
        
        recommended = detection["recommendations"]["primary"]
        if recommended:
            self.console.print(f"[green]🎯 Recommended: {recommended.upper()}[/green]")
            self.console.print(f"   Reason: {detection['recommendations']['reason']}")
        else:
            self.console.print("[red]❌ No backends available - please configure a provider[/red]")
        
        return detection

    def get_optimization_tips(self) -> List[str]:
        """Get optimization tips based on current setup."""
        detection = self._run_detection()
        tips = []

        available = detection["available_backends"]
        
        if not any(available.values()):
            tips.append("🔧 Configure at least one provider to get started")
            tips.append("📖 Run 'conjecture setup' for configuration help")
            return tips

        if available["local"] and not available["cloud"]:
            tips.append("☁️ Consider adding a cloud provider for advanced analysis features")
            tips.append("🌐 Cloud providers offer more powerful models for fact-checking")
        
        if available["cloud"] and not available["local"]:
            tips.append("🏠 Consider setting up a local provider (Ollama) for privacy and offline use")
            tips.append("📴 Local providers work even without internet connection")
        
        if available["local"] and available["cloud"]:
            tips.append("🔄 Hybrid backend is available and recommended for optimal performance")
            tips.append("⚡ Use 'conjecture --backend hybrid' for automatic optimization")
        
        return tips

    def simulate_scenarios(self) -> Dict[str, Any]:
        """Simulate different usage scenarios with recommendations."""
        detection = self._run_detection()
        scenarios = {
            "scenarios": {
                "privacy_first": {
                    "scenario": "User prioritizes data privacy and offline capability",
                    "recommended_backend": "local" if detection["available_backends"]["local"] else "none",
                    "benefits": ["Complete data privacy", "No internet required", "Fast local processing"]
                },
                "power_analysis": {
                    "scenario": "User needs advanced analysis and fact-checking",
                    "recommended_backend": "cloud" if detection["available_backends"]["cloud"] else "hybrid",
                    "benefits": ["Advanced models", "Web search integration", "Fact-checking capabilities"]
                },
                "balanced": {
                    "scenario": "User wants optimal performance with flexibility",
                    "recommended_backend": "hybrid" if detection["available_backends"]["hybrid"] else "auto",
                    "benefits": ["Automatic optimization", "Fallback support", "Best of both worlds"]
                }
            },
            "current_setup": {
                "available_backends": [name for name, available in detection["available_backends"].items() if available],
                "optimal_for_current": detection["recommendations"]["primary"]
            }
        }

        return scenarios