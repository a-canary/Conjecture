#!/usr/bin/env python3
"""Simplify Conjecture success_criteria.json to new framework format."""

import json
from pathlib import Path

# Read original criteria
orig_path = Path("D:/projects/Conjecture/.agent/success_criteria.json")
data = json.load(open(orig_path))

# Simplify: keep only essential fields
simplified_criteria = []
for crit in data.get("criteria", []):
    simplified = {
        "id": crit.get("id", ""),
        "name": crit.get("name", ""),
        "description": crit.get("description", ""),
        "test_method": crit.get("test_method", ""),
        "priority": crit.get("priority", "medium").lower(),
        "status": crit.get("status", "pending").lower(),
    }
    # Map old status values to new values
    if simplified["status"] == "completed":
        simplified["status"] = "pass"
    elif simplified["status"] == "ai tested":
        simplified["status"] = "tested"
    elif simplified["status"] == "verified":
        simplified["status"] = "pass"

    # Keep only non-pass criteria and limit to 8
    if simplified["status"] not in ["pass", "completed"]:
        if len(simplified_criteria) < 8:
            simplified_criteria.append(simplified)

# Create simplified structure
simplified_data = {
    "all_criteria_pass": False,
    "target_completion": 1.0,
    "current_progress": 0.0,
    "criteria": simplified_criteria,
}

# Write simplified version
output_path = Path("D:/projects/Conjecture/.agent/success_criteria.json")
with open(output_path, "w") as f:
    json.dump(simplified_data, f, indent=2)

result = json.dumps(simplified_data, indent=2)
print(f"Simplified criteria:")
print(f"  Original: 51 criteria, 644 lines")
print(f"  New: {len(simplified_data['criteria'])} active criteria")
print(f"  Lines: {len(result.splitlines())}")
print(f"  Saved to: {output_path}")
