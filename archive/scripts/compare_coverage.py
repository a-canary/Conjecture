#!/usr/bin/env python3
"""
Coverage comparison tool for Conjecture project.
Compares coverage between different runs and shows progress/regression.
"""

import json
import sys
import os
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Handle Unicode encoding for Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

def load_coverage_data(file_path: str) -> Dict[str, Any]:
    """Load coverage data from JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Coverage file not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in coverage file {file_path}: {e}")
        sys.exit(1)

def extract_metrics(coverage_data: Dict[str, Any]) -> Dict[str, float]:
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

def format_percentage(value: float, total: int) -> str:
    """Format percentage with appropriate handling of zero totals."""
    if total == 0:
        return "N/A"
    return f"{value:.1f}%"

def print_metrics(title: str, metrics: Dict[str, float]):
    """Print coverage metrics in a formatted way."""
    print(f"\n📊 {title}")
    print("=" * len(title))
    print(f"  Line Coverage: {format_percentage(metrics['line_coverage'], metrics['lines_total'])} ({metrics['lines_covered']}/{metrics['lines_total']})")
    
    if metrics['branches_total'] > 0:
        print(f"  Branch Coverage: {format_percentage(metrics['branch_coverage'], metrics['branches_total'])} ({metrics['branches_covered']}/{metrics['branches_total']})")
    else:
        print("  Branch Coverage: N/A")
    
    print(f"  Missing Lines: {metrics['missing_lines']}")

def compare_metrics(old_metrics: Dict[str, float], new_metrics: Dict[str, float]) -> Dict[str, float]:
    """Compare two sets of metrics and return the differences."""
    return {
        'line_diff': new_metrics['line_coverage'] - old_metrics['line_coverage'],
        'branch_diff': new_metrics['branch_coverage'] - old_metrics['branch_coverage'],
        'lines_covered_diff': new_metrics['lines_covered'] - old_metrics['lines_covered'],
        'missing_lines_diff': new_metrics['missing_lines'] - old_metrics['missing_lines'],
    }

def print_comparison(old_metrics: Dict[str, float], new_metrics: Dict[str, float], diff: Dict[str, float]):
    """Print comparison between two coverage runs."""
    print("\n📈 Coverage Comparison")
    print("=====================")
    
    # Line coverage
    line_arrow = "↑" if diff['line_diff'] > 0 else "↓" if diff['line_diff'] < 0 else "→"
    line_emoji = "🟢" if diff['line_diff'] > 0 else "🔴" if diff['line_diff'] < 0 else "⚪"
    print(f"  Line Coverage: {format_percentage(old_metrics['line_coverage'], old_metrics['lines_total'])} → {format_percentage(new_metrics['line_coverage'], new_metrics['lines_total'])} {line_arrow} {abs(diff['line_diff']):.1f}% {line_emoji}")
    
    # Branch coverage
    if old_metrics['branches_total'] > 0 and new_metrics['branches_total'] > 0:
        branch_arrow = "↑" if diff['branch_diff'] > 0 else "↓" if diff['branch_diff'] < 0 else "→"
        branch_emoji = "🟢" if diff['branch_diff'] > 0 else "🔴" if diff['branch_diff'] < 0 else "⚪"
        print(f"  Branch Coverage: {format_percentage(old_metrics['branch_coverage'], old_metrics['branches_total'])} → {format_percentage(new_metrics['branch_coverage'], new_metrics['branches_total'])} {branch_arrow} {abs(diff['branch_diff']):.1f}% {branch_emoji}")
    
    # Lines covered
    lines_arrow = "↑" if diff['lines_covered_diff'] > 0 else "↓" if diff['lines_covered_diff'] < 0 else "→"
    lines_emoji = "🟢" if diff['lines_covered_diff'] > 0 else "🔴" if diff['lines_covered_diff'] < 0 else "⚪"
    print(f"  Lines Covered: {old_metrics['lines_covered']} → {new_metrics['lines_covered']} {lines_arrow} {diff['lines_covered_diff']} {lines_emoji}")
    
    # Missing lines
    missing_arrow = "↓" if diff['missing_lines_diff'] < 0 else "↑" if diff['missing_lines_diff'] > 0 else "→"
    missing_emoji = "🟢" if diff['missing_lines_diff'] < 0 else "🔴" if diff['missing_lines_diff'] > 0 else "⚪"
    print(f"  Missing Lines: {old_metrics['missing_lines']} → {new_metrics['missing_lines']} {missing_arrow} {abs(diff['missing_lines_diff'])} {missing_emoji}")

def assess_progress(new_metrics: Dict[str, float], goal: float = 80.0) -> str:
    """Assess progress toward coverage goal."""
    current_coverage = new_metrics['line_coverage']
    
    if current_coverage >= goal:
        return "🎯 GOAL ACHIEVED! You've reached the 80% coverage target!"
    elif current_coverage >= goal * 0.9:
        return "🟡 ALMOST THERE! You're within 10% of the 80% goal!"
    elif current_coverage >= goal * 0.75:
        return "🟠 GOOD PROGRESS! You're at 75% of the goal or higher!"
    elif current_coverage >= goal * 0.5:
        return "🔶 HALF WAY! You've reached 50% of the goal!"
    else:
        return "🔴 JUST STARTING! Keep working toward the 80% goal!"

def find_latest_coverage_files(directory: str = "coverage_reports") -> List[Tuple[str, str]]:
    """Find the latest coverage files in the directory."""
    if not os.path.exists(directory):
        return []
    
    files = []
    for filename in os.listdir(directory):
        if filename.startswith("coverage_") and filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            # Extract timestamp from filename
            timestamp_str = filename[9:-5]  # Remove "coverage_" prefix and ".json" suffix
            try:
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                files.append((filepath, timestamp))
            except ValueError:
                continue
    
    # Sort by timestamp (newest first)
    files.sort(key=lambda x: x[1], reverse=True)
    return files

def main():
    parser = argparse.ArgumentParser(description="Compare coverage between runs")
    parser.add_argument("--old", help="Path to old coverage JSON file")
    parser.add_argument("--new", help="Path to new coverage JSON file")
    parser.add_argument("--latest", action="store_true", help="Compare latest two coverage files")
    parser.add_argument("--goal", type=float, default=80.0, help="Coverage goal percentage (default: 80.0)")
    
    args = parser.parse_args()
    
    # Determine which files to compare
    if args.latest:
        files = find_latest_coverage_files()
        if len(files) < 2:
            print("❌ Need at least 2 coverage files to compare")
            sys.exit(1)
        
        old_file = files[1][0]  # Second newest
        new_file = files[0][0]  # Newest
        print(f"📁 Comparing latest files:")
        print(f"  Old: {os.path.basename(old_file)}")
        print(f"  New: {os.path.basename(new_file)}")
    elif args.old and args.new:
        old_file = args.old
        new_file = args.new
    else:
        # Default: compare current coverage.json with latest in coverage_reports
        if os.path.exists("coverage.json"):
            new_file = "coverage.json"
            files = find_latest_coverage_files()
            if files:
                old_file = files[0][0]
                print(f"📁 Comparing current coverage with latest saved:")
                print(f"  Old: {os.path.basename(old_file)}")
                print(f"  New: {new_file}")
            else:
                print("❌ No previous coverage files found. Run coverage twice to compare.")
                sys.exit(1)
        else:
            print("❌ No coverage.json found. Run coverage first.")
            sys.exit(1)
    
    # Load coverage data
    old_data = load_coverage_data(old_file)
    new_data = load_coverage_data(new_file)
    
    # Extract metrics
    old_metrics = extract_metrics(old_data)
    new_metrics = extract_metrics(new_data)
    
    # Print metrics
    print_metrics("Previous Coverage", old_metrics)
    print_metrics("Current Coverage", new_metrics)
    
    # Compare metrics
    diff = compare_metrics(old_metrics, new_metrics)
    print_comparison(old_metrics, new_metrics, diff)
    
    # Assess progress
    print(f"\n{assess_progress(new_metrics, args.goal)}")
    
    # Summary
    if diff['line_diff'] > 0:
        print(f"✅ Coverage improved by {diff['line_diff']:.1f}%!")
    elif diff['line_diff'] < 0:
        print(f"⚠️  Coverage decreased by {abs(diff['line_diff']):.1f}%!")
    else:
        print("⚪ Coverage unchanged.")

if __name__ == "__main__":
    main()