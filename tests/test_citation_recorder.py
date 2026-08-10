# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""Citation recorder: every benchmark-shaped (pp-delta) figure on shipped
surfaces resolves to a run reference or a superseded/historical marker.
"""

from pathlib import Path

from benchmarks.citation_recorder import (
    RESOLVES, _is_exempt, scan_file, scan_surfaces,
)


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_uncited_figure_detected_with_file_and_line(tmp_path):
    p = _write(tmp_path, "README.md", "line one\nno support here: +12pp gain\n")
    uncited, total = scan_file(p)
    assert total == 1
    assert len(uncited) == 1
    assert uncited[0].file.endswith("README.md")
    assert uncited[0].line == 2
    assert uncited[0].figure == "+12pp"


def test_figure_with_run_reference_citation_resolves(tmp_path):
    p = _write(tmp_path, "README.md", "see experiments/results/foo.json\n+12pp gain\n")
    uncited, total = scan_file(p)
    assert total == 1
    assert uncited == []


def test_superseded_block_figure_resolves_but_still_counted(tmp_path):
    p = _write(tmp_path, "README.md", "status: superseded, do not cite\n+12pp gain\n")
    uncited, total = scan_file(p)
    assert total == 1  # still in the denominator, not silently exempted
    assert uncited == []


def test_rd_tree_exempt_at_directory_level():
    assert _is_exempt("docs/rnd_reports/foo.md")
    assert _is_exempt("docs/RND_COMPREHENSIVE_REPORT.md")
    assert _is_exempt("research/bar.md")
    assert not _is_exempt("docs/index.md")
    assert not _is_exempt("README.md")


def test_bare_percentage_is_not_benchmark_shaped(tmp_path):
    p = _write(tmp_path, "README.md", "targets 90%+ performance with no citation\n")
    _, total = scan_file(p)
    assert total == 0  # thresholds/targets aren't claimed results


def test_resolves_regex_recognizes_all_markers():
    for token in ("superseded", "historical", "unverified", "experiments/results/",
                  "benchmarks/results/", "STATS.yaml", "CHOICES.md", "run_id"):
        assert RESOLVES.search(token), token


def test_repo_state_has_zero_uncited_figures():
    """Suite-integration: the real repo, post-fix, cites every in-scope figure."""
    uncited, cited, total = scan_surfaces()
    assert uncited == []
    assert total > 0
    assert cited == total
