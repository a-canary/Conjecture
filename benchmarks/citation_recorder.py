# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""Citation recorder: every benchmark-shaped figure on conjecture's shipped
surfaces (STATS.yaml, docs index/tutorials, blog, README) must resolve to a
run reference, or sit inside a block explicitly marked superseded/historical.

A figure is "benchmark-shaped" if it matches the percentage-point delta
pattern (+28.6pp, -13.0pp, 20pp) that STATS.yaml, CHOICES.md and the blog
already use for citing benchmark deltas. Bare percentages ("90%+ coverage",
"25% threshold") are targets/thresholds, not claimed results, and are not
in scope — matching PRD intent, this keeps the check from false-positiving
on every stray '%' in prose.

A figure resolves if, within CITATION_WINDOW lines, the surface names:
  - a run reference: a path under experiments/results/ or benchmarks/results/,
    a STATS.yaml section key (run_id-shaped), or CHOICES.md/STATS.yaml itself
  - a supersession/historical marker: 'superseded', 'historical', 'unverified'

In STATS.yaml specifically, a figure also resolves if its enclosing top-level
block carries a timestamp/last_run/source field anywhere in the block (the
block is itself a dated run record — STATS.yaml IS the citation ledger, a
figure inside it doesn't need to cite itself) or a status: historical /
superseded marker anywhere in the block (not just the fixed line window).

The R&D reports tree (docs/rnd_reports/, docs/RND_COMPREHENSIVE_REPORT.md,
research/, RD-DEEP-RESEARCH-*.md) is exempt at the directory/file level —
dated historical investigation, not live claims — and excluded from the
coverage denominator entirely.

STATS.yaml's `benchmark_scores:` block is legacy data collected before the
provenance scheme existed (2025-12-12 timestamp) and is treated the same
way: exempt, not counted.
"""

import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Percentage-point delta shape: +28.6pp, -13.0pp, 20pp, +10pp
PP_FIGURE = re.compile(r"[+-]?\d+\.?\d*\s*pp\b")

CITATION_WINDOW = 3  # lines above/below a figure to look for a resolving reference

RESOLVES = re.compile(
    r"(superseded|historical|unverified"
    r"|experiments/results/|benchmarks/results/"
    r"|STATS\.yaml|CHOICES\.md"
    r"|run_id)",
    re.IGNORECASE,
)

# A STATS.yaml block is self-citing if it's a dated run record: it names its
# own timestamp/last_run/source, or carries an explicit historical marker.
BLOCK_SELF_CITES = re.compile(
    r"(last_run|timestamp|source:|status:\s*(historical|superseded))",
    re.IGNORECASE,
)

# Directory/file-level exemptions: dated historical R&D, not live claims.
EXEMPT_PATHS = (
    "docs/rnd_reports/",
    "docs/RND_COMPREHENSIVE_REPORT.md",
    "research/",
    "RD-DEEP-RESEARCH-2026-05-04.md",
)

# Surfaces a new reader lands on (PRD scope): stats file, docs index +
# tutorials, blog, README. Historical R&D and legacy STATS.yaml blocks are
# excluded by EXEMPT_PATHS / _skip_legacy_stats_block below.
SURFACES = (
    "STATS.yaml",
    "docs/index.md",
    "docs/tutorials/advanced.md",
    "docs/tutorials/basic_usage.md",
    "blog/index.html",
    "README.md",
)


@dataclass
class Uncited:
    file: str
    line: int
    figure: str


def _is_exempt(rel_path: str) -> bool:
    return any(rel_path.startswith(p) or rel_path == p for p in EXEMPT_PATHS)


def _legacy_stats_block_lines(lines: list[str]) -> set[int]:
    """Line numbers (1-indexed) inside STATS.yaml's legacy `benchmark_scores:`
    top-level block, which predates the provenance scheme and is exempt."""
    exempt: set[int] = set()
    in_block = False
    for i, line in enumerate(lines, start=1):
        if re.match(r"^\S", line):  # top-level key: not indented
            in_block = line.startswith("benchmark_scores:")
        if in_block:
            exempt.add(i)
    return exempt


def _top_level_block_bounds(lines: list[str], i: int) -> tuple[int, int]:
    """1-indexed [start, end] of the top-level (unindented) YAML block containing
    line i (i itself 1-indexed)."""
    start = 1
    for j in range(i, 0, -1):
        if re.match(r"^\S", lines[j - 1]):
            start = j
            break
    end = len(lines)
    for j in range(i + 1, len(lines) + 1):
        if re.match(r"^\S", lines[j - 1]):
            end = j - 1
            break
    return start, end


def scan_file(path: Path) -> tuple[list[Uncited], int]:
    """Return (uncited figures, total in-scope figures) for one file."""
    try:
        rel = str(path.relative_to(REPO_ROOT))
    except ValueError:
        rel = str(path)  # out-of-tree (e.g. tmp_path in tests): use as-is
    lines = path.read_text().splitlines()
    is_stats = rel == "STATS.yaml"
    legacy = _legacy_stats_block_lines(lines) if is_stats else set()

    uncited = []
    total = 0
    for i, line in enumerate(lines, start=1):
        if i in legacy:
            continue
        for m in PP_FIGURE.finditer(line):
            total += 1
            window = lines[max(0, i - 1 - CITATION_WINDOW): i + CITATION_WINDOW]
            resolved = bool(RESOLVES.search("\n".join(window)))
            if not resolved and is_stats:
                start, end = _top_level_block_bounds(lines, i)
                block = "\n".join(lines[start - 1: end])
                resolved = bool(BLOCK_SELF_CITES.search(block))
            if not resolved:
                uncited.append(Uncited(rel, i, m.group(0)))
    return uncited, total


def scan_surfaces() -> tuple[list[Uncited], int, int]:
    """Scan all in-scope surfaces. Returns (uncited, cited_count, total_count)."""
    all_uncited: list[Uncited] = []
    total = 0
    for rel in SURFACES:
        path = REPO_ROOT / rel
        if not path.exists() or _is_exempt(rel):
            continue
        uncited, file_total = scan_file(path)
        all_uncited.extend(uncited)
        total += file_total
    cited = total - len(all_uncited)
    return all_uncited, cited, total


def report() -> str:
    uncited, cited, total = scan_surfaces()
    lines = [f"citation coverage: {cited}/{total} benchmark figures cited"]
    for u in uncited:
        lines.append(f"  UNCITED {u.file}:{u.line}: {u.figure!r}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    print(report())
    uncited, _, _ = scan_surfaces()
    sys.exit(1 if uncited else 0)
