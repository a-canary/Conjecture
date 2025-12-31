#!/usr/bin/env python3
"""Extract learning from backlog.md to learning.yaml format."""

import re
from pathlib import Path

# Read backlog.md
backlog_path = Path("D:/projects/Conjecture/.agent/backlog.md")
content = backlog_path.read_text()

# Extract learning entries
# Pattern: ## ID | ... Learning: <text>
# Skip the format section and only get actual task learning entries
learning_entries = []

lines = content.split("\n")
current_learning = None

for i, line in enumerate(lines):
    if "**Learning**:" in line:
        # Extract learning text
        learning_text = line.split("**Learning**:")[-1].strip()
        # Only add if it's not just explaining the format
        if len(learning_text) > 20 and "where Status" not in learning_text:
            learning_entries.append(learning_text)

print(f"Found {len(learning_entries)} actual learning entries")

# Limit to 10 most recent
learning_entries = learning_entries[:10]

# Convert to learning.yaml format
learning_data = {"bookmarks": [], "corrections": [], "faq": []}

# Process as corrections
for learning_text in learning_entries:
    learning_data["corrections"].append(
        [
            0,  # likes
            learning_text[:80],  # context
            "backlog",  # action
            "completed task",  # cause
            learning_text[:50],  # resolution
        ]
    )

# Write learning.yaml
output_path = Path("D:/projects/Conjecture/.agent/learning.yaml")
with open(output_path, "w") as f:
    f.write("# Learning Memory\n")
    f.write("# Safe format - no code execution, no anchors, max 500 per section\n\n")

    f.write("bookmarks:\n")
    f.write("\n")

    f.write("corrections:\n")
    for item in learning_data["corrections"]:
        f.write(f"  - {item}\n")
    f.write("\n")

    f.write("faq:\n")

lines = len(output_path.read_text().splitlines())
print(f"\nConverted to learning.yaml:")
print(f"  Corrections: {len(learning_data['corrections'])}")
print(f"  Lines: {lines}")
print(f"  Saved to: {output_path}")
