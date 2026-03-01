#!/usr/bin/env python3
"""
Coverage baseline tracking system for Conjecture project.
Establishes and tracks coverage baselines over time.
"""

import json
import sys
import os
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Handle Unicode encoding for Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

class CoverageBaseline:
    """Manages coverage baselines and tracking."""
    
    def __init__(self, baseline_file: str = "coverage_baseline.json"):
        self.baseline_file = baseline_file
        self.baseline_data = self._load_baseline()
    
    def _load_baseline(self) -> Dict[str, Any]:
        """Load baseline data from file."""
        if os.path.exists(self.baseline_file):
            try:
                with open(self.baseline_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️  Warning: Could not load baseline file: {e}")
                return self._create_empty_baseline()
        else:
            return self._create_empty_baseline()
    
    def _create_empty_baseline(self) -> Dict[str, Any]:
        """Create an empty baseline structure."""
        return {
            "project": "Conjecture",
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "baseline": None,
            "history": [],
            "goals": {
                "line_coverage": 80.0,
                "branch_coverage": 70.0,
                "target_date": "2024-12-31"
            },
            "milestones": [
                {"coverage": 40.0, "description": "Initial baseline", "achieved": False},
                {"coverage": 60.0, "description": "Good progress", "achieved": False},
                {"coverage": 80.0, "description": "Target achieved", "achieved": False}
            ]
        }
    
    def _save_baseline(self):
        """Save baseline data to file."""
        self.baseline_data["last_updated"] = datetime.now().isoformat()
        try:
            with open(self.baseline_file, 'w') as f:
                json.dump(self.baseline_data, f, indent=2)
        except IOError as e:
            print(f"❌ Error saving baseline file: {e}")
            sys.exit(1)
    
    def load_coverage_data(self, coverage_file: str = "coverage.json") -> Dict[str, Any]:
        """Load coverage data from JSON file."""
        try:
            with open(coverage_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Coverage file not found: {coverage_file}")
            print("💡 Run coverage first: python scripts/run_coverage.py")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in coverage file: {e}")
            sys.exit(1)
    
    def extract_metrics(self, coverage_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract key metrics from coverage data."""
        totals = coverage_data.get('totals', {})
        return {
            'line_coverage': totals.get('percent_covered', 0.0),
            'branch_coverage': totals.get('percent_covered_branches', 0.0),
            'lines_covered': totals.get('covered_lines', 0),
            'lines_total': totals.get('num_statements', 0),
            'branches_covered': totals.get('covered_branches', 0),
            'branches_total': totals.get('num_branches', 0),
            'missing_lines': totals.get('missing_lines', 0),
        }
    
    def set_baseline(self, coverage_file: str = "coverage.json", force: bool = False):
        """Set the current coverage as baseline."""
        if self.baseline_data["baseline"] and not force:
            print("⚠️  Baseline already exists. Use --force to overwrite.")
            return
        
        coverage_data = self.load_coverage_data(coverage_file)
        metrics = self.extract_metrics(coverage_data)
        
        baseline_entry = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "coverage_file": coverage_file,
            "description": "Initial coverage baseline"
        }
        
        self.baseline_data["baseline"] = baseline_entry
        self.baseline_data["history"].append(baseline_entry)
        self._save_baseline()
        
        print(f"✅ Baseline established:")
        print(f"  Line Coverage: {metrics['line_coverage']:.1f}%")
        print(f"  Branch Coverage: {metrics['branch_coverage']:.1f}%")
        print(f"  Lines: {metrics['lines_covered']}/{metrics['lines_total']}")
        print(f"  Timestamp: {baseline_entry['timestamp']}")
    
    def check_against_baseline(self, coverage_file: str = "coverage.json"):
        """Check current coverage against baseline."""
        if not self.baseline_data["baseline"]:
            print("❌ No baseline found. Set one first with --set-baseline")
            return
        
        current_data = self.load_coverage_data(coverage_file)
        current_metrics = self.extract_metrics(current_data)
        baseline_metrics = self.baseline_data["baseline"]["metrics"]
        
        # Calculate differences
        line_diff = current_metrics['line_coverage'] - baseline_metrics['line_coverage']
        branch_diff = current_metrics['branch_coverage'] - baseline_metrics['branch_coverage']
        lines_diff = current_metrics['lines_covered'] - baseline_metrics['lines_covered']
        
        print("📊 Coverage vs Baseline Comparison")
        print("=================================")
        print(f"  Baseline: {baseline_metrics['line_coverage']:.1f}% line coverage")
        print(f"  Current:  {current_metrics['line_coverage']:.1f}% line coverage")
        print(f"  Change:   {line_diff:+.1f}% ({'↑' if line_diff > 0 else '↓' if line_diff < 0 else '→'})")
        print()
        print(f"  Baseline: {baseline_metrics['branch_coverage']:.1f}% branch coverage")
        print(f"  Current:  {current_metrics['branch_coverage']:.1f}% branch coverage")
        print(f"  Change:   {branch_diff:+.1f}% ({'↑' if branch_diff > 0 else '↓' if branch_diff < 0 else '→'})")
        print()
        print(f"  Lines covered: {baseline_metrics['lines_covered']} → {current_metrics['lines_covered']} ({lines_diff:+d})")
        
        # Check milestones
        self._check_milestones(current_metrics['line_coverage'])
        
        # Add to history
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "metrics": current_metrics,
            "baseline_comparison": {
                "line_diff": line_diff,
                "branch_diff": branch_diff,
                "lines_diff": lines_diff
            }
        }
        self.baseline_data["history"].append(history_entry)
        self._save_baseline()
    
    def _check_milestones(self, current_coverage: float):
        """Check and update milestone achievements."""
        milestones = self.baseline_data["milestones"]
        updated = False
        
        for milestone in milestones:
            if not milestone["achieved"] and current_coverage >= milestone["coverage"]:
                milestone["achieved"] = True
                milestone["achieved_date"] = datetime.now().isoformat()
                updated = True
                print(f"🎉 MILESTONE ACHIEVED: {milestone['description']} ({milestone['coverage']}%)")
        
        if updated:
            self._save_baseline()
    
    def show_status(self):
        """Show current baseline status and history."""
        print("📋 Coverage Baseline Status")
        print("===========================")
        
        if not self.baseline_data["baseline"]:
            print("❌ No baseline established")
            return
        
        baseline = self.baseline_data["baseline"]
        print(f"📊 Baseline established: {baseline['timestamp']}")
        print(f"   Line Coverage: {baseline['metrics']['line_coverage']:.1f}%")
        print(f"   Branch Coverage: {baseline['metrics']['branch_coverage']:.1f}%")
        print()
        
        # Show goals
        goals = self.baseline_data["goals"]
        print("🎯 Goals:")
        print(f"   Line Coverage: {goals['line_coverage']}% by {goals['target_date']}")
        print(f"   Branch Coverage: {goals['branch_coverage']}%")
        print()
        
        # Show milestones
        print("🏆 Milestones:")
        for milestone in self.baseline_data["milestones"]:
            status = "✅" if milestone["achieved"] else "⏳"
            achieved_date = f" (achieved {milestone['achieved_date'][:10]})" if milestone.get("achieved_date") else ""
            print(f"   {status} {milestone['coverage']}% - {milestone['description']}{achieved_date}")
        print()
        
        # Show recent history
        history = self.baseline_data["history"][-5:]  # Last 5 entries
        if history:
            print("📈 Recent History:")
            for entry in reversed(history):
                if "baseline_comparison" in entry:
                    comp = entry["baseline_comparison"]
                    trend = "↑" if comp["line_diff"] > 0 else "↓" if comp["line_diff"] < 0 else "→"
                    print(f"   {entry['timestamp'][:19]}: {entry['metrics']['line_coverage']:.1f}% {trend}{comp['line_diff']:+.1f}%")
                else:
                    print(f"   {entry['timestamp'][:19]}: {entry['metrics']['line_coverage']:.1f}% (baseline)")
    
    def generate_report(self, output_file: Optional[str] = None):
        """Generate a comprehensive coverage report."""
        if not self.baseline_data["baseline"]:
            print("❌ No baseline established")
            return
        
        report = {
            "report_generated": datetime.now().isoformat(),
            "project": self.baseline_data["project"],
            "baseline": self.baseline_data["baseline"],
            "goals": self.baseline_data["goals"],
            "milestones": self.baseline_data["milestones"],
            "history": self.baseline_data["history"],
            "summary": self._generate_summary()
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"📄 Report saved to: {output_file}")
        else:
            print(json.dumps(report, indent=2))
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of coverage progress."""
        if not self.baseline_data["history"]:
            return {}
        
        baseline = self.baseline_data["baseline"]["metrics"]
        latest = self.baseline_data["history"][-1]["metrics"]
        
        # Calculate overall progress
        line_progress = latest['line_coverage'] - baseline['line_coverage']
        branch_progress = latest['branch_coverage'] - baseline['branch_coverage']
        
        # Calculate trend (last 3 entries if available)
        recent_entries = [h for h in self.baseline_data["history"][-4:] if "baseline_comparison" in h]
        trend = "stable"
        if len(recent_entries) >= 2:
            changes = [h["baseline_comparison"]["line_diff"] for h in recent_entries]
            if all(c > 0 for c in changes):
                trend = "improving"
            elif all(c < 0 for c in changes):
                trend = "declining"
        
        return {
            "baseline_coverage": baseline['line_coverage'],
            "current_coverage": latest['line_coverage'],
            "progress": line_progress,
            "trend": trend,
            "milestones_achieved": sum(1 for m in self.baseline_data["milestones"] if m["achieved"]),
            "total_milestones": len(self.baseline_data["milestones"]),
            "goal_progress": (latest['line_coverage'] / self.baseline_data["goals"]["line_coverage"]) * 100
        }

def main():
    parser = argparse.ArgumentParser(description="Coverage baseline tracking")
    parser.add_argument("--set-baseline", action="store_true", help="Set current coverage as baseline")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing baseline")
    parser.add_argument("--check", action="store_true", help="Check coverage against baseline")
    parser.add_argument("--status", action="store_true", help="Show baseline status")
    parser.add_argument("--report", help="Generate report to file")
    parser.add_argument("--coverage-file", default="coverage.json", help="Coverage file to analyze")
    parser.add_argument("--baseline-file", default="coverage_baseline.json", help="Baseline file to use")
    
    args = parser.parse_args()
    
    baseline = CoverageBaseline(args.baseline_file)
    
    if args.set_baseline:
        baseline.set_baseline(args.coverage_file, args.force)
    elif args.check:
        baseline.check_against_baseline(args.coverage_file)
    elif args.status:
        baseline.show_status()
    elif args.report:
        baseline.generate_report(args.report)
    else:
        # Default action: show status
        baseline.show_status()

if __name__ == "__main__":
    main()