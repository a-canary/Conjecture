# Plan: UX-0007 Claim Visualization UI

## Objective

Implement visualization of claim support trees and reasoning chains. Users can see how conclusions connect to evidence through an interactive TUI/Web interface.

## Scope

**In Scope:**
- Claim tree visualization with ASCII art (simple TUI)
- Claim relationship graph display (CLI enhancement)
- Web API endpoints for claim graph traversal
- Basic claim visualization endpoints for future WebUI integration

**Out of Scope:**
- Full WebUI (complex, deferred)
- Real-time collaborative editing
- Complex graph layout algorithms

---

## Phase 1: CLI Claim Tree Visualization

### Steps
- [x] 1.1 Add `claim tree <claim_id>` command to CLI showing claim and sub-claims recursively (ASCII tree)
- [x] 1.2 Add `claim trace <claim_id>` command showing chain from root to claim (reverse traversal via supers)
- [x] 1.3 Add `--depth` flag to both commands (default: 3, max: 10)
- [x] 1.4 Add `--confidence` flag to filter claims by confidence threshold
- [x] 1.5 Format output with Rich tree rendering (colored by confidence)

### Gates
- [x] `conjecture tree <id> --depth 2` renders a tree with colored nodes
- [x] `conjecture trace <id>` shows path from root to claim
- [x] Tests: `test_claim_visualization.py` with 22 test cases

---

## Phase 2: Web API for Claim Graph

### Steps
- [x] 2.1 Add `GET /claims/{id}/tree` endpoint returning nested claim structure with sub-claims
- [x] 2.2 Add `GET /claims/{id}/trace` endpoint returning chain from root to claim
- [x] 2.3 Add `GET /claims/{id}/graph` endpoint returning adjacency list (for D3.js/vis.js)
- [x] 2.4 Add `depth` and `min_confidence` query params to all endpoints

### Gates
- [x] HTTP endpoints added to `http_server.py`
- [x] Endpoints use visualization utilities for tree/trace/graph generation
- [x] 963 tests pass (22 new visualization tests)

---

## Phase 3: TUI Interactive Browser (Optional Enhancement)

### Steps
- [ ] 3.1 Create `src/cli/claim_browser.py` with Rich Table/Tree widgets
- [ ] 3.2 Add `conjecture browse` command launching interactive TUI
- [ ] 3.3 Support keyboard navigation: j/k up/down, Enter expand, q quit
- [ ] 3.4 Add search/filter panel with real-time results

### Gates
- `conjecture browse --help` shows keyboard navigation help
- Interactive browser starts without errors on valid workspace
- Tests: `test_claim_browser.py` (if implemented)

---

## Current Phase: 3

## Status: in_progress (Phase 1-2 complete)

---

## Success Criteria

- [x] `conjecture tree <id>` shows claim support tree
- [x] `conjecture trace <id>` shows chain to root
- [x] Web API endpoints return claim graph data
- [x] Tests pass for new visualization features
- [ ] UX-0007 "not started" → "implemented" in NEXT.md
